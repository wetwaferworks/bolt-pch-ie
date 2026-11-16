// Bolt PCHIE - shape limits of the CUDA backend, kept host-only so they can be tested without a GPU.
//
// The k-quant and Q8_0 matvec kernels keep one partial sum per block of a row in shared memory
// (sb[warps][kCudaMvMaxBlocks]) and add them in order in lane 0, which is how the GPU stays bit-identical
// to the CPU. A row with more blocks than that would write past the shared array, so such a weight is
// refused at CudaBackend::init (the caller then runs on the CPU) instead of being launched.
// The scratch buffers are sized from the widest input and output any matvec can see, so a model whose
// sliding head is wider than its global head, or whose d_model or d_ple is wider than d_ff, still fits.
#pragma once
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_kquants_det.h"
#include "pchie/pchie_q4_det.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace pchie {

constexpr int kCudaMvMaxBlocks = 128;   // MV_MAXNB in src/pchie_cuda.cu

// Blocks per row the matvec kernel for this ggml type walks; 0 for a type it does not run.
inline uint64_t cuda_matvec_blocks(uint32_t type, uint64_t ne0) noexcept {
    switch (type) {
        case 8:  return ne0 / 32;     // Q8_0
        case 12: return ne0 / 256;    // Q4_K
        case 14: return ne0 / 256;    // Q6_K
        default: return 0;
    }
}

// True when the CUDA matvec can run this weight without leaving its buffers. why gets the reason otherwise.
inline bool cuda_matvec_shape_ok(uint32_t type, uint64_t ne0, uint64_t ne1, std::string* why = nullptr) {
    auto no = [&](const char* m) { if (why) *why = m; return false; };
    if (ne0 == 0 || ne1 == 0 || ne0 > (1u << 30) || ne1 > (1u << 30)) return no("weight shape out of range");
    switch (type) {
        case 8: case 12: case 14: {
            const uint64_t blk = type == 8 ? 32 : 256;
            if (ne0 % blk != 0) return no("row is not a whole number of blocks");
            if (cuda_matvec_blocks(type, ne0) > uint64_t(kCudaMvMaxBlocks)) return no("row has more blocks than the CUDA matvec holds in shared memory");
            return true;
        }
        case 30:
            if (ne0 % 8 != 0) return no("BF16 row is not a multiple of 8 (the matvec reads 8 at a time)");
            return true;
        default:
            return no("weight type the CUDA matvec does not run");
    }
}

// Sizes of the per-branch scratch buffers (in floats, or elements of the activation width).
struct CudaScratchDims {
    size_t head_dim = 0;      // widest head: max(hd_global, hd_swa)
    size_t q_floats = 0;      // n_head * head_dim   (q, att)
    size_t kv_floats = 0;     // n_head_kv * head_dim (k, v)
    size_t act_width = 0;     // widest matvec input: max(d_ff, d_model, d_ple, n_head * head_dim)
    size_t xq_bytes = 0;      // quantised activation scratch for one branch (Q8_0 and Q8_K views share it)
};
inline CudaScratchDims cuda_scratch_dims(const Gemma4Config& C) {
    CudaScratchDims d;
    d.head_dim = size_t(std::max(C.hd_global, C.hd_swa));
    d.q_floats = size_t(C.n_head) * d.head_dim;
    d.kv_floats = size_t(C.n_head_kv) * d.head_dim;
    d.act_width = std::max({size_t(C.d_ff), size_t(C.d_model), size_t(C.d_ple), d.q_floats});
    d.xq_bytes = sizeof(BlockQ8_0) * ((d.act_width + 31) / 32) + sizeof(BlockQ8_K) * ((d.act_width + QK_K - 1) / QK_K);
    return d;
}

} // namespace pchie
