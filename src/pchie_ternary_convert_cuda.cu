/* Bolt PCHIE - ternary converter, CUDA fit (clean-room, CUDA C). Runs ternary_group_fit_from_sorted() from
 * pchie_ternary_group_fit.h, the same code the CPU path runs, so the codes and scales match the CPU bit for bit.
 * Built with -fmad=false, IEEE division and no flush-to-zero (see CMakeLists.txt).
 *
 * A block is one warp and fits 32 groups. It copies them (16 KiB, contiguous in the source) into shared memory
 * with coalesced loads, each row padded by one float so the threads read different banks. For least squares the
 * warp then sorts each group's |w| together with a bitonic network; finally each thread runs the scan and the
 * codes (ternary_group_fit_from_sorted) for its own group. Device buffers are kept between calls and only grow;
 * ternary_cuda_release() frees them.
 */
#include "pchie/pchie_ternary_convert.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
void ternary_set_errorf(char *error, size_t error_capacity, const char *format, ...);
#ifdef __cplusplus
}
#endif

enum { TERNARY_GROUPS_PER_BLOCK = 32 };
enum { TERNARY_THREADS_PER_BLOCK = 32 };

static __global__ void ternary_group_fit_kernel(const float * __restrict__ weights, unsigned long long group_count,
                                                int fit_kind, int8_t * __restrict__ codes, float * __restrict__ scales) {
    /* One warp per block: TERNARY_GROUPS_PER_BLOCK == 32, so __syncwarp() orders the shared-memory steps below. */
    __shared__ float shared_weights[TERNARY_GROUPS_PER_BLOCK][TERNARY_GROUP_WEIGHT_COUNT + 1];
    __shared__ float shared_sorted[TERNARY_GROUPS_PER_BLOCK][TERNARY_GROUP_WEIGHT_COUNT + 1];
    const unsigned long long first_group = (unsigned long long)blockIdx.x * TERNARY_GROUPS_PER_BLOCK;
    int group_count_in_block;
    const int lane = threadIdx.x;
    int i;
    if (first_group >= group_count) return;
    {
        const unsigned long long remaining = group_count - first_group;
        group_count_in_block = remaining < TERNARY_GROUPS_PER_BLOCK ? (int)remaining : TERNARY_GROUPS_PER_BLOCK;
    }
    {
        const float *src = weights + first_group * TERNARY_GROUP_WEIGHT_COUNT;
        for (i = lane; i < group_count_in_block * TERNARY_GROUP_WEIGHT_COUNT; i += TERNARY_GROUPS_PER_BLOCK) {
            const float value = src[i];
            shared_weights[i / TERNARY_GROUP_WEIGHT_COUNT][i % TERNARY_GROUP_WEIGHT_COUNT] = value;
            shared_sorted[i / TERNARY_GROUP_WEIGHT_COUNT][i % TERNARY_GROUP_WEIGHT_COUNT] = fabsf(value);
        }
    }
    __syncwarp();
    if (fit_kind == (int)TERNARY_GROUP_FIT_LEAST_SQUARES) {
        /* The warp sorts each group with a bitonic network (descending). Same values as the CPU radix sort. */
        int group_in_block, k, j;
        for (group_in_block = 0; group_in_block < group_count_in_block; ++group_in_block) {
            float *absolute = &shared_sorted[group_in_block][0];
            for (k = 2; k <= TERNARY_GROUP_WEIGHT_COUNT; k <<= 1) {
                for (j = k >> 1; j > 0; j >>= 1) {
                    for (i = lane; i < TERNARY_GROUP_WEIGHT_COUNT; i += TERNARY_GROUPS_PER_BLOCK) {
                        const int partner = i ^ j;
                        if (partner > i) {
                            const float x = absolute[i];
                            const float y = absolute[partner];
                            const int descending = (i & k) == 0;
                            if (descending ? (x < y) : (x > y)) {
                                absolute[i] = y;
                                absolute[partner] = x;
                            }
                        }
                    }
                    __syncwarp();
                }
            }
        }
    }
    if (lane < group_count_in_block) {
        const unsigned long long group = first_group + (unsigned long long)lane;
        scales[group] = ternary_group_fit_from_sorted(&shared_weights[lane][0], (ternary_group_fit_kind)fit_kind,
                                                      &shared_sorted[lane][0],
                                                      codes + group * TERNARY_GROUP_WEIGHT_COUNT);
    }
}

typedef struct ternary_device_buffers {
    float *weights;
    int8_t *codes;
    float *scales;
    unsigned long long capacity_groups;
} ternary_device_buffers;

/* Single-threaded use by contract (one converter per process drives the GPU); no lock needed. */
static ternary_device_buffers g_buffers;
static ternary_cuda_counters g_counters;

static int check_cuda(cudaError_t status, const char *what, char *error, size_t error_capacity) {
    if (status == cudaSuccess) return 1;
    ternary_set_errorf(error, error_capacity, "%s: %s", what, cudaGetErrorString(status));
    return 0;
}

static void free_device_buffers(void) {
    if (g_buffers.weights) cudaFree(g_buffers.weights);
    if (g_buffers.codes) cudaFree(g_buffers.codes);
    if (g_buffers.scales) cudaFree(g_buffers.scales);
    g_buffers.weights = NULL;
    g_buffers.codes = NULL;
    g_buffers.scales = NULL;
    g_buffers.capacity_groups = 0;
}

static int ensure_device_buffers(unsigned long long group_count, char *error, size_t error_capacity) {
    size_t weight_count;
    uint64_t own;
    size_t free_bytes = 0, total_bytes = 0;
    if (group_count <= g_buffers.capacity_groups) return 1;
    free_device_buffers();
    weight_count = (size_t)group_count * TERNARY_GROUP_WEIGHT_COUNT;
    if (!check_cuda(cudaMalloc(&g_buffers.weights, weight_count * sizeof(float)), "cudaMalloc weights", error,
                    error_capacity) ||
        !check_cuda(cudaMalloc(&g_buffers.codes, weight_count), "cudaMalloc codes", error, error_capacity) ||
        !check_cuda(cudaMalloc(&g_buffers.scales, (size_t)group_count * sizeof(float)), "cudaMalloc scales", error,
                    error_capacity)) {
        free_device_buffers();
        return 0;
    }
    g_buffers.capacity_groups = group_count;
    own = (uint64_t)weight_count * (sizeof(float) + 1) + (uint64_t)group_count * sizeof(float);
    if (own > g_counters.own_bytes_peak) g_counters.own_bytes_peak = own;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes - free_bytes > g_counters.device_used_peak)
        g_counters.device_used_peak = total_bytes - free_bytes;
    return 1;
}

ternary_cuda_info ternary_query_cuda_info(void) {
    ternary_cuda_info info;
    int count = 0;
    memset(&info, 0, sizeof(info));
    info.built = 1;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return info;
    }
    info.device_count = count;
    if (count > 0) {
        cudaDeviceProp properties;
        if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
            size_t n = strlen(properties.name);
            if (n > sizeof(info.device_name) - 1) n = sizeof(info.device_name) - 1;
            memcpy(info.device_name, properties.name, n);
            info.device_name[n] = 0;
            info.total_bytes = properties.totalGlobalMem;
        }
    }
    return info;
}

const ternary_cuda_counters *ternary_query_cuda_counters(void) { return &g_counters; }
void ternary_cuda_release(void) { free_device_buffers(); }

ternary_convert_status ternary_fit_groups_cuda(const float *weights, uint64_t group_count, ternary_group_fit_kind fit_kind,
                                               int8_t *codes, float *scales, char *error, size_t error_capacity) {
    const uint64_t weight_count = group_count * TERNARY_GROUP_WEIGHT_COUNT;
    uint64_t i;
    cudaEvent_t e0, e1, e2, e3;
    int ok = 1;
    float copy_in = 0, kernel = 0, copy_out = 0;
    if (!weights || !codes || !scales || group_count == 0 || (unsigned)fit_kind > 1) return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < weight_count; ++i) {
        if (!isfinite(weights[i])) {
            memset(codes, 0, (size_t)weight_count);
            memset(scales, 0, (size_t)group_count * sizeof(float));
            return TERNARY_CONVERT_NON_FINITE;
        }
    }
    if (group_count > (UINT64_C(1) << 40)) return TERNARY_CONVERT_TOO_LARGE;
    if (error && error_capacity) error[0] = 0;
    if (!ensure_device_buffers(group_count, error, error_capacity)) return TERNARY_CONVERT_BAD_ARGUMENT;

    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventCreate(&e3);
    cudaEventRecord(e0);
    ok = ok && check_cuda(cudaMemcpy(g_buffers.weights, weights, (size_t)weight_count * sizeof(float),
                                     cudaMemcpyHostToDevice),
                          "copy weights", error, error_capacity);
    cudaEventRecord(e1);
    if (ok) {
        const unsigned long long blocks = (group_count + TERNARY_GROUPS_PER_BLOCK - 1) / TERNARY_GROUPS_PER_BLOCK;
        if (blocks > 0x7FFFFFFFull) {
            ok = 0;
            ternary_set_errorf(error, error_capacity, "too many groups for one launch");
        } else {
            ternary_group_fit_kernel<<<(unsigned)blocks, TERNARY_THREADS_PER_BLOCK>>>(
                g_buffers.weights, group_count, (int)fit_kind, g_buffers.codes, g_buffers.scales);
            ok = check_cuda(cudaGetLastError(), "fit kernel launch", error, error_capacity);
        }
    }
    cudaEventRecord(e2);
    ok = ok && check_cuda(cudaMemcpy(codes, g_buffers.codes, (size_t)weight_count, cudaMemcpyDeviceToHost),
                          "copy codes", error, error_capacity);
    ok = ok && check_cuda(cudaMemcpy(scales, g_buffers.scales, (size_t)group_count * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy scales", error, error_capacity);
    cudaEventRecord(e3);
    cudaEventSynchronize(e3);
    cudaEventElapsedTime(&copy_in, e0, e1);
    cudaEventElapsedTime(&kernel, e1, e2);
    cudaEventElapsedTime(&copy_out, e2, e3);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    g_counters.copy_milliseconds += (double)copy_in + (double)copy_out;
    g_counters.kernel_milliseconds += (double)kernel;
    g_counters.call_count++;
    if (!ok) {
        memset(codes, 0, (size_t)weight_count);
        memset(scales, 0, (size_t)group_count * sizeof(float));
        return TERNARY_CONVERT_BAD_ARGUMENT;
    }
    return TERNARY_CONVERT_OK;
}
