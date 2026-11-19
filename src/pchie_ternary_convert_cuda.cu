// Bolt PCHIE - ternary converter, CUDA fit (clean-room). Runs fit_group() from pchie_ternary_fit.h, the same code
// the CPU path runs, so the codes and scales match the CPU bit for bit. Built with -fmad=false, IEEE division and
// no flush-to-zero (see CMakeLists.txt).
//
// A block is one warp and fits 32 groups. It copies them (16 KiB, contiguous in the source) into shared memory with
// coalesced loads, each row padded by one float so the threads read different banks. For least squares the warp
// then sorts each group's |w| together with a bitonic network; finally each thread runs the scan and the codes
// (fit_group_sorted) for its own group. Device buffers are kept between calls and only grow; cuda_release() frees them.
#include "pchie/pchie_ternary_convert.h"
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <cstring>
#include <string>

namespace pchie {
namespace ternary {
namespace convert {

namespace {

constexpr int kGroupsPerBlock = 32;

__global__ void fit_kernel(const float* __restrict__ w, unsigned long long n_groups, int fit,
                           int8_t* __restrict__ q, float* __restrict__ gamma) {
    // One warp per block: kGroupsPerBlock == 32, so __syncwarp() orders the shared-memory steps below.
    __shared__ float sw[kGroupsPerBlock][kGroup + 1];   // the weights
    __shared__ float ss[kGroupsPerBlock][kGroup + 1];   // |w|, sorted largest first (least squares only)
    const unsigned long long g0 = static_cast<unsigned long long>(blockIdx.x) * kGroupsPerBlock;
    if (g0 >= n_groups) return;
    const unsigned long long left = n_groups - g0;
    const int ng = left < kGroupsPerBlock ? static_cast<int>(left) : kGroupsPerBlock;
    const int lane = threadIdx.x;
    const float* src = w + g0 * kGroup;
    for (int i = lane; i < ng * kGroup; i += kGroupsPerBlock) {
        const float v = src[i];
        sw[i / kGroup][i % kGroup] = v;
        ss[i / kGroup][i % kGroup] = fabsf(v);
    }
    __syncwarp();
    if (fit == int(FitKind::LeastSquares)) {
        // The warp sorts each group with a bitonic network (descending). Same values as the CPU's insertion sort.
        for (int gi = 0; gi < ng; ++gi) {
            float* a = &ss[gi][0];
            for (int k = 2; k <= kGroup; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    for (int i = lane; i < kGroup; i += kGroupsPerBlock) {
                        const int p = i ^ j;
                        if (p > i) {
                            const float x = a[i], y = a[p];
                            const bool desc = (i & k) == 0;
                            if (desc ? (x < y) : (x > y)) { a[i] = y; a[p] = x; }
                        }
                    }
                    __syncwarp();
                }
            }
        }
    }
    if (lane < ng) {
        const unsigned long long g = g0 + static_cast<unsigned long long>(lane);
        gamma[g] = fit_group_sorted(&sw[lane][0], static_cast<FitKind>(fit), &ss[lane][0], q + g * kGroup);
    }
}

struct DeviceBuffers {
    float* w = nullptr;
    int8_t* q = nullptr;
    float* g = nullptr;
    unsigned long long cap_groups = 0;
};
// Single-threaded use by contract (one converter per process drives the GPU); no lock needed.
DeviceBuffers g_buf;
CudaCounters g_counters;

bool check(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(what) + ": " + cudaGetErrorString(e);
    return false;
}

void free_buffers() noexcept {
    if (g_buf.w) cudaFree(g_buf.w);
    if (g_buf.q) cudaFree(g_buf.q);
    if (g_buf.g) cudaFree(g_buf.g);
    g_buf = DeviceBuffers();
}

bool ensure(unsigned long long n_groups, std::string& err) {
    if (n_groups <= g_buf.cap_groups) return true;
    free_buffers();
    const size_t nw = size_t(n_groups) * kGroup;
    if (!check(cudaMalloc(&g_buf.w, nw * sizeof(float)), "cudaMalloc weights", err) ||
        !check(cudaMalloc(&g_buf.q, nw), "cudaMalloc codes", err) ||
        !check(cudaMalloc(&g_buf.g, size_t(n_groups) * sizeof(float)), "cudaMalloc scales", err)) {
        free_buffers();
        return false;
    }
    g_buf.cap_groups = n_groups;
    const uint64_t own = uint64_t(nw) * (sizeof(float) + 1) + uint64_t(n_groups) * sizeof(float);
    if (own > g_counters.own_bytes_peak) g_counters.own_bytes_peak = own;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess && total_b - free_b > g_counters.device_used_peak)
        g_counters.device_used_peak = total_b - free_b;
    return true;
}

} // namespace

CudaInfo cuda_info() {
    CudaInfo info;
    info.built = true;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return info; }
    info.devices = n;
    if (n > 0) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) { info.device_name = p.name; info.total_bytes = p.totalGlobalMem; }
    }
    return info;
}

const CudaCounters& cuda_counters() noexcept { return g_counters; }
void cuda_release() noexcept { free_buffers(); }

Status fit_groups_cuda(const float* w, uint64_t n_groups, FitKind fit, int8_t* q, float* gamma, std::string& err) {
    if (!w || !q || !gamma || n_groups == 0 || uint8_t(fit) > 1) return Status::BadArgument;
    const uint64_t nw = n_groups * kGroup;
    for (uint64_t i = 0; i < nw; ++i) {
        if (!std::isfinite(w[i])) {   // same contract as the CPU path: zeroed outputs, NonFinite
            std::memset(q, 0, size_t(nw));
            std::memset(gamma, 0, size_t(n_groups) * sizeof(float));
            return Status::NonFinite;
        }
    }
    if (n_groups > (uint64_t(1) << 40)) return Status::TooLarge;
    if (!ensure(n_groups, err)) return Status::BadArgument;
    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    bool ok = true;
    cudaEventRecord(e0);
    ok = ok && check(cudaMemcpy(g_buf.w, w, size_t(nw) * sizeof(float), cudaMemcpyHostToDevice), "copy weights", err);
    cudaEventRecord(e1);
    if (ok) {
        const unsigned long long blocks = (n_groups + kGroupsPerBlock - 1) / kGroupsPerBlock;
        if (blocks > 0x7FFFFFFFull) { ok = false; err = "too many groups for one launch"; }
        else {
            fit_kernel<<<static_cast<unsigned int>(blocks), kGroupsPerBlock>>>(g_buf.w, n_groups, int(fit), g_buf.q, g_buf.g);
            ok = check(cudaGetLastError(), "fit kernel launch", err);
        }
    }
    cudaEventRecord(e2);
    ok = ok && check(cudaMemcpy(q, g_buf.q, size_t(nw), cudaMemcpyDeviceToHost), "copy codes", err);
    ok = ok && check(cudaMemcpy(gamma, g_buf.g, size_t(n_groups) * sizeof(float), cudaMemcpyDeviceToHost), "copy scales", err);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);
    float a = 0, b = 0, c = 0;
    cudaEventElapsedTime(&a, e0, e1); cudaEventElapsedTime(&b, e1, e2); cudaEventElapsedTime(&c, e2, e3);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    g_counters.copy_ms += double(a) + double(c);
    g_counters.kernel_ms += double(b);
    ++g_counters.calls;
    if (!ok) {
        std::memset(q, 0, size_t(nw));
        std::memset(gamma, 0, size_t(n_groups) * sizeof(float));
        return Status::BadArgument;
    }
    return Status::Ok;
}

} // namespace convert
} // namespace ternary
} // namespace pchie
