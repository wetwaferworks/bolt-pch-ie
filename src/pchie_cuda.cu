// Bolt PCHIE - CUDA kernels + backend. Built with -fmad=false -prec-div=true -prec-sqrt=true.
// Each kernel names the CPU routine it mirrors.
#include "pchie/pchie_cuda.h"
#include "pchie/pchie_cuda_limits.h"
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_kquants_det.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pchie {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x); } while (0)

bool cuda_query(CudaInfo& o, std::string& err) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { err = "no CUDA device"; return false; }
    cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, 0);
    if (e != cudaSuccess) { err = std::string("cudaGetDeviceProperties failed: ") + cudaGetErrorString(e); return false; }
    o.name = p.name; o.sm = p.major * 10 + p.minor;
    e = cudaMemGetInfo(&o.free_bytes, &o.total_bytes);
    if (e != cudaSuccess) { err = std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(e); return false; }
    return true;
}

// ------------------------------------------------------------------ kernels
// mirrors quantize_row_q8_K_deterministic (one thread per super-block)
__global__ void k_quant_q8k(const float* x, BlockQ8_K* q, int nb) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* s = x + b * QK_K;
    float amax = 0.0f, max_val = 0.0f;
    for (int i = 0; i < QK_K; ++i) { const float ax = fabsf(s[i]); if (ax > amax) { amax = ax; max_val = s[i]; } }
    BlockQ8_K& d = q[b];
    if (amax == 0.0f) { d.d_fp32 = 0.0f; for (int i = 0; i < QK_K; ++i) d.qs[i] = 0; for (int j = 0; j < 16; ++j) d.bsums[j] = 0; return; }
    const float iscale = -127.0f / max_val;
    d.d_fp32 = 1.0f / iscale;
    for (int i = 0; i < QK_K; ++i) {
        int32_t v = det_round_half_even(iscale * s[i]);
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        d.qs[i] = static_cast<int8_t>(v);
    }
    for (int j = 0; j < 16; ++j) { int32_t sum = 0; for (int i = 0; i < 16; ++i) sum += d.qs[j * 16 + i]; d.bsums[j] = static_cast<int16_t>(sum); }
}

// mirrors quantize_row_q8_0_deterministic (one thread per 32-block)
__global__ void k_quant_q8_0(const float* x, BlockQ8_0* q, int nb) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* s = x + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(s[i]));
    const float d = amax / 127.0f;
    const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
    q[b].d_fp32 = d;
    for (int i = 0; i < 32; ++i) { int32_t v = det_round_half_even(s[i] * id); v = v < -127 ? -127 : (v > 127 ? 127 : v); q[b].qs[i] = static_cast<int8_t>(v); }
}

constexpr int MV_WARPS = 4;
constexpr int MV_MAXNB = kCudaMvMaxBlocks;   // init() refuses any weight whose row has more blocks
static_assert(MV_MAXNB == 128, "shared partial-sum array size");

__device__ __forceinline__ int32_t warp8_sum(int32_t v) {   // exact integer sum over 8-lane groups
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    return v;
}

// mirrors dot_q4_K_q8_K: integer parts split over 8 lanes (exact), float combine sequential in lane 0
__global__ void k_mv_q4k(const BlockQ4_K* W, const BlockQ8_K* X0, float* y0, int rows, int nb, int B) {
    __shared__ float sb[MV_WARPS][MV_MAXNB];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * MV_WARPS + warp;
    if (row >= rows) return;
    const BlockQ4_K* w = W + size_t(row) * nb;
    const int s = lane % 8;
  for (int bt = 0; bt < B; ++bt) {    // same weights, one activation row per branch
    const BlockQ8_K* X = X0 + size_t(bt) * nb; float* y = y0 + size_t(bt) * rows;
    __syncwarp();
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + lane / 8;
        int32_t isc = 0, imn = 0;
        if (b < nb) {
            const BlockQ4_K& wb = w[b]; const BlockQ8_K& xb = X[b];
            uint8_t sc, m;
            if (s < 4) { sc = wb.scales[s] & 63u; m = wb.scales[s + 4] & 63u; }
            else { sc = (wb.scales[s + 4] & 0x0Fu) | ((wb.scales[s - 4] >> 6) << 4); m = (wb.scales[s + 4] >> 4) | ((wb.scales[s] >> 6) << 4); }
            const int j = s / 2, hi = s % 2;
            const uint8_t* q4 = wb.qs + 32 * j; const int8_t* q8 = xb.qs + 64 * j + 32 * hi;
            int32_t dot = 0;
            for (int l = 0; l < 32; ++l) dot += static_cast<int32_t>(hi ? (q4[l] >> 4) : (q4[l] & 0x0Fu)) * q8[l];
            isc = static_cast<int32_t>(sc) * dot;
            imn = static_cast<int32_t>(m) * (xb.bsums[2 * s] + xb.bsums[2 * s + 1]);
        }
        isc = warp8_sum(isc); imn = warp8_sum(imn);
        if (b < nb && s == 0) {
            const float d_all    = X[b].d_fp32 * det_fp16_to_f32(w[b].d_fp16);
            const float dmin_all = X[b].d_fp32 * det_fp16_to_f32(w[b].dmin_fp16);
            const float tp = d_all * static_cast<float>(isc);
            const float tn = dmin_all * static_cast<float>(imn);
            sb[warp][b] = tp - tn;
        }
    }
    __syncwarp();
    if (lane == 0) { float total = 0.0f; for (int b = 0; b < nb; ++b) total = total + sb[warp][b]; y[row] = total; }
  }
}

// mirrors dot_q6_K_q8_K
__global__ void k_mv_q6k(const BlockQ6_K* W, const BlockQ8_K* X0, float* y0, int rows, int nb, int B) {
    __shared__ float sb[MV_WARPS][MV_MAXNB];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * MV_WARPS + warp;
    if (row >= rows) return;
    const BlockQ6_K* w = W + size_t(row) * nb;
    const int s = lane % 8, half = s / 4, g = s % 4;
  for (int bt = 0; bt < B; ++bt) {
    const BlockQ8_K* X = X0 + size_t(bt) * nb; float* y = y0 + size_t(bt) * rows;
    __syncwarp();
    for (int b0 = 0; b0 < nb; b0 += 4) {
        const int b = b0 + lane / 8;
        int32_t is = 0;
        if (b < nb) {
            const BlockQ6_K& wb = w[b];
            const uint8_t* ql = wb.ql + 64 * half; const uint8_t* qh = wb.qh + 32 * half; const int8_t* sc = wb.scales + 8 * half;
            const int8_t* q8 = X[b].qs + 128 * half + 32 * g;
            for (int l = 0; l < 32; ++l) {
                int32_t q;
                if (g == 0)      q = static_cast<int32_t>((ql[l]      & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                else if (g == 1) q = static_cast<int32_t>((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                else if (g == 2) q = static_cast<int32_t>((ql[l]      >> 4)    | (((qh[l] >> 4) & 3u) << 4)) - 32;
                else             q = static_cast<int32_t>((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3u) << 4)) - 32;
                is += sc[l / 16 + 2 * g] * q * q8[l];
            }
        }
        is = warp8_sum(is);
        if (b < nb && s == 0) {
            const float d_all = X[b].d_fp32 * det_fp16_to_f32(w[b].d_fp16);
            sb[warp][b] = d_all * static_cast<float>(is);
        }
    }
    __syncwarp();
    if (lane == 0) { float total = 0.0f; for (int b = 0; b < nb; ++b) total = total + sb[warp][b]; y[row] = total; }
  }
}

// mirrors dot_q8_0_q8_0 (one lane per 32-block)
__global__ void k_mv_q8_0(const BlockQ8_0W* W, const BlockQ8_0* X0, float* y0, int rows, int nb, int B) {
    __shared__ float sb[MV_WARPS][MV_MAXNB];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.x * MV_WARPS + warp;
    if (row >= rows) return;
    const BlockQ8_0W* w = W + size_t(row) * nb;
  for (int bt = 0; bt < B; ++bt) {
    const BlockQ8_0* X = X0 + size_t(bt) * nb; float* y = y0 + size_t(bt) * rows;
    __syncwarp();
    for (int b = lane; b < nb; b += 32) {
        int32_t is = 0;
        for (int i = 0; i < 32; ++i) is += static_cast<int32_t>(w[b].qs[i]) * X[b].qs[i];
        const float sc = det_fp16_to_f32(w[b].d_fp16) * X[b].d_fp32;
        sb[warp][b] = sc * static_cast<float>(is);
    }
    __syncwarp();
    if (lane == 0) { float total = 0.0f; for (int b = 0; b < nb; ++b) total = total + sb[warp][b]; y[row] = total; }
  }
}

// mirrors dot_bf16_f32 (one thread per row, 8 fixed lanes, fixed tree)
__global__ void k_mv_bf16(const uint16_t* W, const float* x0, float* y0, int rows, int n, int B) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * B) return;
    const int r = idx % rows, bt = idx / rows;
    const float* x = x0 + size_t(bt) * n; float* y = y0 + size_t(bt) * rows;
    const uint16_t* w = W + size_t(r) * n;
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) { const float p = det_bf16_to_f32(w[i + k]) * x[i + k]; lane[k] = lane[k] + p; }
    for (int k = 0; i + k < n; ++k) { const float p = det_bf16_to_f32(w[i + k]) * x[i + k]; lane[k] = lane[k] + p; }   // never hit: init refuses ne0 % 8
    const float a = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    const float b = (lane[4] + lane[5]) + (lane[6] + lane[7]);
    y[r] = a + b;
}

// mirrors Gemma4Model::rmsnorm, one block per segment (sequential sum in thread 0)
__global__ void k_rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float inv_s;
    const float* xs = x + size_t(blockIdx.x) * n; float* ys = y + size_t(blockIdx.x) * n;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < n; ++i) { const float p = xs[i] * xs[i]; ss = ss + p; }
        const float mean = ss / float(n);
        inv_s = 1.0f / sqrtf(mean + eps);
    }
    __syncthreads();
    const float inv = inv_s;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ys[i] = w ? (xs[i] * inv) * w[i] : xs[i] * inv;
}

// mirrors Gemma4Model::rope with the host-built cos/sin table
__global__ void k_rope(float* v0, const float* c0, int heads, int hd, int half, int B) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * heads * half) return;
    const int bt = idx / (heads * half), h = (idx / half) % heads, i = idx % half;
    const float* c = c0 + size_t(bt) * 2048; const float* s = c + 512;
    float* x = v0 + size_t(bt) * heads * hd + size_t(h) * hd;
    const float x0 = x[i], x1 = x[i + half];
    const float a = x0 * c[i], b = x1 * s[i], d = x0 * s[i], e = x1 * c[i];
    x[i] = a - b; x[i + half] = d + e;
}

__global__ void k_kv_store(float* arena, size_t koff, size_t voff, const float* k, const float* v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { arena[koff + i] = k[i]; arena[voff + i] = v[i]; }
}

// mirrors the attention lambda in Gemma4Model::layer: one block per query head
__global__ void k_attention(const float* q0, const float* arena, const int32_t* pages0, int pt_stride, size_t page_floats,
                            int slot, int hkv, int hd, int group, const uint32_t* j0s, const uint32_t* ts, float* scores, int max_ctx, float* out0) {
    const int h = blockIdx.x, kh = h / group, bt = blockIdx.y, HQ = gridDim.x;
    const uint32_t j0 = j0s[bt], t = ts[bt];
    const int32_t* pages = pages0 + size_t(bt) * pt_stride;
    const float* qh = q0 + (size_t(bt) * HQ + h) * hd;
    float* out = out0 + size_t(bt) * HQ * hd;
    float* sc = scores + (size_t(bt) * HQ + h) * max_ctx;
    const uint32_t n = t - j0 + 1;
    auto base = [&](uint32_t j, int vsel) -> const float* {
        return arena + size_t(pages[j / 16]) * page_floats + ((size_t(slot) * 2 + vsel) * 16 + (j % 16)) * size_t(hkv) * hd + size_t(kh) * hd;
    };
    for (uint32_t jj = threadIdx.x; jj < n; jj += blockDim.x) {
        const float* kj = base(j0 + jj, 0);
        float s = 0.0f;
        for (int i = 0; i < hd; ++i) { const float p = qh[i] * kj[i]; s = s + p; }
        sc[jj] = s;
    }
    __syncthreads();
    __shared__ float mx_s, den_s;
    if (threadIdx.x == 0) { float mx = -INFINITY; for (uint32_t jj = 0; jj < n; ++jj) mx = fmaxf(mx, sc[jj]); mx_s = mx; }
    __syncthreads();
    for (uint32_t jj = threadIdx.x; jj < n; jj += blockDim.x) sc[jj] = exp_deterministic(sc[jj] - mx_s);
    __syncthreads();
    if (threadIdx.x == 0) { float den = 0.0f; for (uint32_t jj = 0; jj < n; ++jj) den = den + sc[jj]; den_s = den; }
    __syncthreads();
    const float den = den_s;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.0f;
        for (uint32_t jj = 0; jj < n; ++jj) { const float p = sc[jj] / den; const float m = p * base(j0 + jj, 1)[i]; o = o + m; }
        out[size_t(h) * hd + i] = o;
    }
}

__global__ void k_add(float* x, const float* d, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = x[i] + d[i]; }
__global__ void k_add_scale(float* x, const float* d, float s, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = (x[i] + d[i]) * s; }
__global__ void k_scale(float* x, float s, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = x[i] * s; }
__global__ void k_geglu(float* g, const float* up, int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) g[i] = gelu_tanh_deterministic(g[i]) * up[i]; }
__global__ void k_gelu_mul(float* g, const float* ple, int l, int NL, int P, int B) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= B * P) return;
    const int bt = i / P; g[i] = gelu_tanh_deterministic(g[i]) * ple[(size_t(bt) * NL + l) * P + (i % P)];
}
__global__ void k_ple_combine(float* p, const float* c, float ple_scale, float comb, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { const float a = p[i] * ple_scale; p[i] = (a + c[i]) * comb; }
}

// ------------------------------------------------------------------ backend
struct DevT { void* p = nullptr; uint32_t type = 0; int ne0 = 0, ne1 = 0; };
struct DevLayer { DevT wq, wk, wv, wo, gate, up, down, inp_gate, proj; float *attn_norm, *q_norm, *k_norm, *post_attn_norm, *ffn_norm, *post_ffw_norm, *post_norm; float out_scale; };

static size_t qbytes(const QTensor& t) {
    const size_t n = size_t(t.ne0) * size_t(t.ne1);
    switch (t.type) { case 0: return n * 4; case 30: return n * 2; case 8: return n / 32 * 34; case 12: return n / 256 * 144; case 14: return n / 256 * 210; }
    throw std::runtime_error("qbytes: type");
}

struct CudaBackend::Impl {
    Gemma4Model* m = nullptr; int G = 0; size_t vram = 0; bool gpu_head = false;
    uint8_t* wbuf = nullptr; size_t woff = 0, wcap = 0;
    std::vector<DevLayer> L;
    DevT ple_proj, tok_embd; float *ple_norm = nullptr, *out_norm = nullptr;
    float *arena_swa = nullptr, *arena_glob = nullptr;
    float *x, *u, *tmp, *q, *k, *v, *att, *g, *up, *pg, *pg2, *ple, *ctx, *scores, *cs, *logits;
    void* xq = nullptr; int32_t *pt_swa = nullptr, *pt_glob = nullptr;
    int max_ctx = 0, maxB = 1;
    uint32_t *d_t = nullptr, *d_j0 = nullptr;
    std::vector<bool> host_kv;   // GPU-owned layer whose KV a CPU layer reads
    std::vector<void*> allocs;
    // Set (and never cleared) when a mirrored copy-on-write page copy fails on the GPU; forward_batch then
    // refuses to run, because the GPU KV arena no longer matches the host page tables.
    std::atomic<bool> copy_failed{false};
    bool hooks_set = false;       // this backend installed the pools' on_copy hooks

    template <class T> T* dalloc(size_t n) { void* p; CK(cudaMalloc(&p, n * sizeof(T))); allocs.push_back(p); vram += n * sizeof(T); return static_cast<T*>(p); }
    DevT up_tensor(const QTensor& t) {
        DevT d; d.type = t.type; d.ne0 = int(t.ne0); d.ne1 = int(t.ne1);
        const size_t n = qbytes(t);
        d.p = wbuf + woff; CK(cudaMemcpy(d.p, t.data, n, cudaMemcpyHostToDevice)); woff += (n + 255) / 256 * 256;
        return d;
    }
    float* up_f32(const float* p, int n) { if (!p) return nullptr; float* d = reinterpret_cast<float*>(wbuf + woff); CK(cudaMemcpy(d, p, size_t(n) * 4, cudaMemcpyHostToDevice)); woff += (size_t(n) * 4 + 255) / 256 * 256; return d; }

    void matvec(const DevT& w, const float* xin, float* y, int B = 1) {
        const int n = w.ne0, rows = w.ne1;
        const int grid = (rows + MV_WARPS - 1) / MV_WARPS;
        if (w.type == 12 || w.type == 14) {
            const int nb = n / QK_K;
            k_quant_q8k<<<(nb * B + 31) / 32, 32>>>(xin, static_cast<BlockQ8_K*>(xq), nb * B);
            if (w.type == 12) k_mv_q4k<<<grid, 32 * MV_WARPS>>>(static_cast<const BlockQ4_K*>(w.p), static_cast<const BlockQ8_K*>(xq), y, rows, nb, B);
            else              k_mv_q6k<<<grid, 32 * MV_WARPS>>>(static_cast<const BlockQ6_K*>(w.p), static_cast<const BlockQ8_K*>(xq), y, rows, nb, B);
        } else if (w.type == 8) {
            const int nb = n / 32;
            k_quant_q8_0<<<(nb * B + 63) / 64, 64>>>(xin, static_cast<BlockQ8_0*>(xq), nb * B);
            k_mv_q8_0<<<grid, 32 * MV_WARPS>>>(static_cast<const BlockQ8_0W*>(w.p), static_cast<const BlockQ8_0*>(xq), y, rows, nb, B);
        } else if (w.type == 30) {
            k_mv_bf16<<<(rows * B + 127) / 128, 128>>>(static_cast<const uint16_t*>(w.p), xin, y, rows, n, B);
        } else throw std::runtime_error("cuda matvec: type");
    }
    void norm(const float* xin, const float* w, float* y, int n, int segs) { k_rmsnorm<<<segs, 256>>>(xin, w, y, n, m->cfg().eps); }
    static int nblk(int n) { return (n + 255) / 256; }
};

CudaBackend::CudaBackend() : p_(new Impl) {}
CudaBackend::~CudaBackend() {
    if (!p_) return;
    // The KV pools belong to the model and outlive this backend: unhook the mirror copies first, so a later
    // copy-on-write on the CPU path never calls cudaMemcpy on freed device memory.
    if (p_->hooks_set && p_->m) { p_->m->pool_swa().on_copy = nullptr; p_->m->pool_glob().on_copy = nullptr; }
    for (void* a : p_->allocs) cudaFree(a);
}
int CudaBackend::gpu_layers() const { return p_->G; }
size_t CudaBackend::vram_bytes() const { return p_->vram; }

bool CudaBackend::init(Gemma4Model& model, int gpu_layers, size_t reserve_mb, bool gpu_lm_head, std::string& err, int max_batch) {
    try {
        Impl& I = *p_; I.m = &model; I.gpu_head = gpu_lm_head; I.maxB = max_batch < 1 ? 1 : max_batch;
        const size_t MB = size_t(I.maxB);
        const Gemma4Config& C = model.cfg();
        CudaInfo ci; if (!cuda_query(ci, err)) return false;
        KVPagePool &ps = model.pool_swa(), &pgl = model.pool_glob();
        I.max_ctx = int(pgl.max_pages() * KV_PAGE_TOKENS);
        // fixed costs: KV arenas, scratch, PLE projection (+ LM head)
        size_t fixed = (ps.max_pages() * ps.page_floats() + pgl.max_pages() * pgl.page_floats()) * 4;
        fixed += MB * size_t(C.n_head) * I.max_ctx * 4 + MB * 8u * 1024 * 1024 + 64u * 1024 * 1024;
        fixed += qbytes(model.ple_proj());
        if (gpu_lm_head) fixed += size_t(C.n_vocab) * C.d_model / 32 * 34 + size_t(C.n_vocab) * 4;
        std::vector<size_t> lb(size_t(C.n_layer));
        for (int l = 0; l < C.n_layer; ++l) {
            const Gemma4Layer& W = model.layer_weights(l);
            size_t b = qbytes(W.wq) + qbytes(W.wo) + qbytes(W.gate) + qbytes(W.up) + qbytes(W.down) + qbytes(W.inp_gate) + qbytes(W.proj) + 64 * 1024;
            if (l < C.n_layer_kv) b += qbytes(W.wk) + qbytes(W.wv);
            lb[size_t(l)] = b;
        }
        if (gpu_layers < 0) {
            const size_t budget = ci.free_bytes > reserve_mb * 1024 * 1024 + fixed ? ci.free_bytes - reserve_mb * 1024 * 1024 - fixed : 0;
            size_t acc = 0; gpu_layers = 0;
            while (gpu_layers < C.n_layer && acc + lb[size_t(gpu_layers)] <= budget) acc += lb[size_t(gpu_layers++)];
        }
        I.G = std::min(gpu_layers, C.n_layer);
        if (I.G == 0) { err = "no layers fit in VRAM"; return false; }
        // Every weight the GPU will run must fit the matvec kernels' fixed shared-memory partial sums.
        // Refuse here (the caller then runs on the CPU) rather than write past the shared array.
        {
            std::string why;
            auto shape = [&](const QTensor& t, const std::string& name) {
                if (!cuda_matvec_shape_ok(t.type, t.ne0, t.ne1, &why)) throw std::runtime_error("CUDA backend refuses " + name + ": " + why);
            };
            shape(model.ple_proj(), "per_layer_model_proj");
            if (gpu_lm_head) shape(model.tok_embd(), "token_embd");
            for (int l = 0; l < I.G; ++l) {
                const Gemma4Layer& W = model.layer_weights(l); const std::string p = "layer " + std::to_string(l) + " ";
                shape(W.wq, p + "attn_q"); shape(W.wo, p + "attn_output"); shape(W.gate, p + "ffn_gate"); shape(W.up, p + "ffn_up");
                shape(W.down, p + "ffn_down"); shape(W.inp_gate, p + "inp_gate"); shape(W.proj, p + "proj");
                if (l < C.n_layer_kv) { shape(W.wk, p + "attn_k"); shape(W.wv, p + "attn_v"); }
            }
        }
        size_t wtot = qbytes(model.ple_proj()) + 4 * 1024 * 1024;
        for (int l = 0; l < I.G; ++l) wtot += lb[size_t(l)];
        if (gpu_lm_head) wtot += size_t(C.n_vocab) * C.d_model / 32 * 34 + 4096;
        I.wbuf = I.dalloc<uint8_t>(wtot); I.wcap = wtot;
        I.ple_proj = I.up_tensor(model.ple_proj());
        I.ple_norm = I.up_f32(model.ple_proj_norm(), C.d_ple);
        I.out_norm = I.up_f32(model.out_norm(), C.d_model);
        I.L.resize(size_t(I.G));
        for (int l = 0; l < I.G; ++l) {
            const Gemma4Layer& W = model.layer_weights(l); DevLayer& D = I.L[size_t(l)];
            const int hd = C.is_swa[size_t(l)] ? C.hd_swa : C.hd_global;
            D.wq = I.up_tensor(W.wq); D.wo = I.up_tensor(W.wo); D.gate = I.up_tensor(W.gate); D.up = I.up_tensor(W.up);
            D.down = I.up_tensor(W.down); D.inp_gate = I.up_tensor(W.inp_gate); D.proj = I.up_tensor(W.proj);
            if (l < C.n_layer_kv) { D.wk = I.up_tensor(W.wk); D.wv = I.up_tensor(W.wv); }
            D.attn_norm = I.up_f32(W.attn_norm, C.d_model); D.q_norm = I.up_f32(W.q_norm, hd); D.k_norm = I.up_f32(W.k_norm, hd);
            D.post_attn_norm = I.up_f32(W.post_attn_norm, C.d_model); D.ffn_norm = I.up_f32(W.ffn_norm, C.d_model);
            D.post_ffw_norm = I.up_f32(W.post_ffw_norm, C.d_model); D.post_norm = I.up_f32(W.post_norm, C.d_model);
            D.out_scale = W.out_scale;
        }
        I.arena_swa = I.dalloc<float>(ps.max_pages() * ps.page_floats());
        I.arena_glob = I.dalloc<float>(pgl.max_pages() * pgl.page_floats());
        float* as = I.arena_swa; float* ag = I.arena_glob; size_t fs = ps.page_floats(), fg = pgl.page_floats();
        std::atomic<bool>* bad = &I.copy_failed;
        ps.on_copy = [as, fs, bad](int32_t s, int32_t d) {
            if (cudaMemcpy(as + size_t(d) * fs, as + size_t(s) * fs, fs * 4, cudaMemcpyDeviceToDevice) != cudaSuccess) bad->store(true);
        };
        pgl.on_copy = [ag, fg, bad](int32_t s, int32_t d) {
            if (cudaMemcpy(ag + size_t(d) * fg, ag + size_t(s) * fg, fg * 4, cudaMemcpyDeviceToDevice) != cudaSuccess) bad->store(true);
        };
        I.hooks_set = true;
        // Scratch is sized from the widest head and the widest matvec input any layer uses (not only the
        // global head and d_ff), so an unusual but accepted config cannot write past a buffer.
        const CudaScratchDims SD = cuda_scratch_dims(C);
        const int D = C.d_model;
        I.x = I.dalloc<float>(MB * D); I.u = I.dalloc<float>(MB * D); I.tmp = I.dalloc<float>(MB * D); I.pg2 = I.dalloc<float>(MB * D);
        I.q = I.dalloc<float>(MB * SD.q_floats); I.att = I.dalloc<float>(MB * SD.q_floats);
        I.k = I.dalloc<float>(MB * SD.kv_floats); I.v = I.dalloc<float>(MB * SD.kv_floats);
        I.g = I.dalloc<float>(MB * C.d_ff); I.up = I.dalloc<float>(MB * C.d_ff); I.pg = I.dalloc<float>(MB * C.d_ple);
        I.ple = I.dalloc<float>(MB * C.n_layer * C.d_ple); I.ctx = I.dalloc<float>(MB * C.n_layer * C.d_ple);
        I.scores = I.dalloc<float>(MB * size_t(C.n_head) * I.max_ctx); I.cs = I.dalloc<float>(MB * 2048);
        I.xq = I.dalloc<uint8_t>(MB * SD.xq_bytes + 1024);
        I.pt_swa = I.dalloc<int32_t>(MB * ps.max_pages()); I.pt_glob = I.dalloc<int32_t>(MB * pgl.max_pages());
        I.d_t = I.dalloc<uint32_t>(MB); I.d_j0 = I.dalloc<uint32_t>(MB * 2);
        if (gpu_lm_head) { I.tok_embd = I.up_tensor(model.tok_embd()); I.logits = I.dalloc<float>(MB * C.n_vocab); }
        I.host_kv.assign(size_t(C.n_layer), false);
        for (int l = I.G; l < C.n_layer; ++l) { const int s = C.kv_source[size_t(l)]; if (s < I.G) I.host_kv[size_t(s)] = true; }
        CK(cudaDeviceSynchronize());
        return true;
    } catch (const std::exception& e) { err = e.what(); return false; }
}

void CudaBackend::forward(Gemma4Model::Branch& br, int32_t token, std::vector<float>& logits, ThreadPool& pool) {
    Gemma4Model::Branch* bs[1] = {&br};
    forward_batch(bs, &token, 1, &logits, pool);
}

void CudaBackend::forward_batch(Gemma4Model::Branch* const* bs, const int32_t* toks, int B, std::vector<float>* logits, ThreadPool& pool) {
    Impl& I = *p_; Gemma4Model& M = *I.m; const Gemma4Config& C = M.cfg();
    if (B > I.maxB) throw std::runtime_error("CudaBackend: batch larger than max_batch");
    if (I.copy_failed.load()) throw std::runtime_error("CudaBackend: a GPU copy-on-write page copy failed; the GPU KV cache is no longer valid");
    const int D = C.d_model, P = C.d_ple, NL = C.n_layer, HQ = C.n_head, HKV = C.n_head_kv, NP = NL * P;
    std::vector<float> x(static_cast<size_t>(D) * B), ple(static_cast<size_t>(NP) * B);
    for (int b = 0; b < B; ++b) { M.begin_token(*bs[b]); M.embed(toks[b], &x[size_t(b) * D], &ple[size_t(b) * NP]); }
    CK(cudaMemcpy(I.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(I.ple, ple.data(), ple.size() * 4, cudaMemcpyHostToDevice));
    // PLE (mirrors Gemma4Model::ple_finish_batch)
    I.matvec(I.ple_proj, I.x, I.ctx, B);
    k_scale<<<Impl::nblk(B * NP), 256>>>(I.ctx, 1.0f / std::sqrt(float(D)), B * NP);
    I.norm(I.ctx, I.ple_norm, I.ctx, P, NL * B);
    k_ple_combine<<<Impl::nblk(B * NP), 256>>>(I.ple, I.ctx, std::sqrt(float(P)), std::sqrt(0.5f), B * NP);
    // KV pages for each branch's position (host bookkeeping, CoW mirrored by on_copy)
    KVPagePool &ps = M.pool_swa(), &pgl = M.pool_glob();
    const int sps = int(ps.max_pages()), spg = int(pgl.max_pages());
    std::vector<int32_t> pts(static_cast<size_t>(sps) * B, -1), ptg(static_cast<size_t>(spg) * B, -1);
    std::vector<uint32_t> ts(static_cast<size_t>(B)), j0s(static_cast<size_t>(2 * B));
    std::vector<float> cs(static_cast<size_t>(2048) * B);
    for (int b = 0; b < B; ++b) {
        Gemma4Model::Branch& br = *bs[b]; const uint32_t t = br.n_past;
        ps.k_write(br.swa, 0, t); pgl.k_write(br.glob, 0, t);
        std::copy(br.swa.pages.begin(), br.swa.pages.end(), pts.begin() + size_t(b) * sps);
        std::copy(br.glob.pages.begin(), br.glob.pages.end(), ptg.begin() + size_t(b) * spg);
        ts[size_t(b)] = t;
        j0s[size_t(b)] = (t + 1 > uint32_t(C.window)) ? t + 1 - uint32_t(C.window) : 0;   // sliding
        j0s[size_t(B + b)] = 0;                                                          // global
        Gemma4Model::rope_table(C.n_rot_swa, C.rope_base_swa, nullptr, t, &cs[size_t(b) * 2048], &cs[size_t(b) * 2048 + 512]);
        Gemma4Model::rope_table(C.n_rot_global, C.rope_base, M.rope_freqs(), t, &cs[size_t(b) * 2048 + 1024], &cs[size_t(b) * 2048 + 1536]);
    }
    if (I.copy_failed.load()) throw std::runtime_error("CudaBackend: a GPU copy-on-write page copy failed; the GPU KV cache is no longer valid");
    CK(cudaMemcpy(I.pt_swa, pts.data(), pts.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(I.pt_glob, ptg.data(), ptg.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(I.d_t, ts.data(), ts.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(I.d_j0, j0s.data(), j0s.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(I.cs, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));

    for (int l = 0; l < I.G; ++l) {
        const DevLayer& W = I.L[size_t(l)];
        const bool swa = C.is_swa[size_t(l)];
        const int hd = swa ? C.hd_swa : C.hd_global, half = (swa ? C.n_rot_swa : C.n_rot_global) / 2;
        const float* c = I.cs + (swa ? 0 : 1024);
        KVPagePool& kvp = swa ? ps : pgl;
        float* arena = swa ? I.arena_swa : I.arena_glob; const int32_t* pt = swa ? I.pt_swa : I.pt_glob;
        const int pstride = swa ? sps : spg;
        const int src = C.kv_source[size_t(l)];
        const uint32_t slot = uint32_t(C.pool_slot[size_t(src)]);
        const size_t KS = size_t(HKV) * hd;

        I.norm(I.x, W.attn_norm, I.u, D, B);
        I.matvec(W.wq, I.u, I.q, B);
        I.norm(I.q, W.q_norm, I.q, hd, HQ * B);
        k_rope<<<Impl::nblk(B * HQ * half), 256>>>(I.q, c, HQ, hd, half, B);
        if (l < C.n_layer_kv) {
            I.matvec(W.wk, I.u, I.k, B); I.matvec(W.wv, I.u, I.v, B);
            I.norm(I.k, W.k_norm, I.k, hd, HKV * B);
            I.norm(I.v, nullptr, I.v, hd, HKV * B);
            k_rope<<<Impl::nblk(B * HKV * half), 256>>>(I.k, c, HKV, hd, half, B);
            for (int b = 0; b < B; ++b) {
                KVPagePool::Table& tab = swa ? bs[b]->swa : bs[b]->glob; const uint32_t t = bs[b]->n_past;
                const size_t pbase = size_t(kvp.page_of(tab, t)) * kvp.page_floats();
                k_kv_store<<<Impl::nblk(int(KS)), 256>>>(arena, pbase + kvp.elem_offset(slot, t, false), pbase + kvp.elem_offset(slot, t, true), I.k + KS * b, I.v + KS * b, int(KS));
                if (I.host_kv[size_t(l)]) {
                    CK(cudaMemcpy(kvp.k_write(tab, slot, t), I.k + KS * b, KS * 4, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(kvp.v_write(tab, slot, t), I.v + KS * b, KS * 4, cudaMemcpyDeviceToHost));
                }
            }
        }
        k_attention<<<dim3(HQ, B), 256>>>(I.q, arena, pt, pstride, kvp.page_floats(), int(slot), HKV, hd, HQ / HKV,
                                          I.d_j0 + (swa ? 0 : B), I.d_t, I.scores, I.max_ctx, I.att);
        I.matvec(W.wo, I.att, I.tmp, B);
        I.norm(I.tmp, W.post_attn_norm, I.tmp, D, B);
        k_add<<<Impl::nblk(B * D), 256>>>(I.x, I.tmp, B * D);
        I.norm(I.x, W.ffn_norm, I.u, D, B);
        I.matvec(W.gate, I.u, I.g, B); I.matvec(W.up, I.u, I.up, B);
        k_geglu<<<Impl::nblk(B * C.d_ff), 256>>>(I.g, I.up, B * C.d_ff);
        I.matvec(W.down, I.g, I.tmp, B);
        I.norm(I.tmp, W.post_ffw_norm, I.tmp, D, B);
        k_add<<<Impl::nblk(B * D), 256>>>(I.x, I.tmp, B * D);
        I.matvec(W.inp_gate, I.x, I.pg, B);
        k_gelu_mul<<<Impl::nblk(B * P), 256>>>(I.pg, I.ple, l, NL, P, B);
        I.matvec(W.proj, I.pg, I.pg2, B);
        I.norm(I.pg2, W.post_norm, I.pg2, D, B);
        k_add_scale<<<Impl::nblk(B * D), 256>>>(I.x, I.pg2, W.out_scale, B * D);
    }
    CK(cudaGetLastError());
    if (I.copy_failed.load()) throw std::runtime_error("CudaBackend: a GPU copy-on-write page copy failed; the GPU KV cache is no longer valid");
    if (I.G < NL) {
        CK(cudaMemcpy(x.data(), I.x, x.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ple.data(), I.ple, ple.size() * 4, cudaMemcpyDeviceToHost));
        for (int l = I.G; l < NL; ++l) M.layer_batch(l, bs, B, x.data(), ple.data(), pool);
        for (int b = 0; b < B; ++b) bs[b]->n_past++;
        M.head_batch(x.data(), B, logits, pool);
        return;
    }
    for (int b = 0; b < B; ++b) bs[b]->n_past++;
    if (I.gpu_head) {   // mirrors Gemma4Model::head_batch
        I.norm(I.x, I.out_norm, I.u, D, B);
        I.matvec(I.tok_embd, I.u, I.logits, B);
        const float cap = C.softcap, icap = 1.0f / C.softcap;
        for (int b = 0; b < B; ++b) {
            logits[b].resize(size_t(C.n_vocab));
            CK(cudaMemcpy(logits[b].data(), I.logits + size_t(b) * C.n_vocab, size_t(C.n_vocab) * 4, cudaMemcpyDeviceToHost));
            for (auto& z : logits[b]) z = cap * tanh_deterministic(z * icap);
        }
        return;
    }
    CK(cudaMemcpy(x.data(), I.x, x.size() * 4, cudaMemcpyDeviceToHost));
    M.head_batch(x.data(), B, logits, pool);
}

} // namespace pchie
