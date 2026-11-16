// CUDA backend shape limits, checked on the host (no GPU needed). The matvec kernels keep one partial sum
// per block of a row in a fixed shared array, so a row with more blocks than that must be refused before
// anything is launched (Helia's review: a Q8_0 weight with ne0 above 4096 wrote past the array).
// The scratch buffers must be sized from the widest head and widest matvec input (Helia's review: they were
// sized from hd_global and d_ff only).
#include "pchie/pchie_cuda_limits.h"

#include <cstdio>
#include <string>

using namespace pchie;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[CUDALIM] %-92s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
}

int main() {
    std::printf("[CUDALIM] CUDA matvec shape limits (host-only)\n");
    std::string why;
    // Q8_0: 32 elements per block, at most 128 blocks per row, so ne0 <= 4096.
    check(cuda_matvec_shape_ok(8, 4096, 32), "Q8_0 ne0 4096 (128 blocks) is accepted");
    check(!cuda_matvec_shape_ok(8, 4128, 32, &why) && !why.empty(), "Q8_0 ne0 4128 (129 blocks) is refused with a reason");
    check(!cuda_matvec_shape_ok(8, 8192, 32), "Q8_0 ne0 8192 is refused");
    check(!cuda_matvec_shape_ok(8, 10240, 2560), "Q8_0 ffn_down of an E4B model (ne0 10240) is refused");
    // Q4_K / Q6_K: 256 elements per block, so ne0 <= 32768.
    check(cuda_matvec_shape_ok(12, 10240, 2560), "Q4_K ne0 10240 (the shipped model's widest row) is accepted");
    check(cuda_matvec_shape_ok(14, 32768, 8), "Q6_K ne0 32768 (128 blocks) is accepted");
    check(!cuda_matvec_shape_ok(12, 33024, 8), "Q4_K ne0 33024 (129 blocks) is refused");
    check(!cuda_matvec_shape_ok(14, 65536, 8), "Q6_K ne0 65536 is refused");
    check(!cuda_matvec_shape_ok(12, 300, 8), "Q4_K row that is not a whole number of blocks is refused");
    // BF16 reads 8 at a time.
    check(cuda_matvec_shape_ok(30, 32, 32), "BF16 ne0 32 is accepted");
    check(!cuda_matvec_shape_ok(30, 12, 32), "BF16 ne0 12 (not a multiple of 8) is refused");
    // Types and sizes the kernels do not handle.
    check(!cuda_matvec_shape_ok(0, 32, 32), "F32 weight is refused (no CUDA matvec for it)");
    check(!cuda_matvec_shape_ok(8, 0, 32), "ne0 0 is refused");
    check(!cuda_matvec_shape_ok(8, 32, 0), "ne1 0 is refused");
    // Scratch sizes come from the widest head and the widest matvec input, not only hd_global and d_ff.
    {
        Gemma4Config c; c.d_model = 32; c.d_ff = 32; c.d_ple = 32; c.n_head = 2; c.n_head_kv = 1; c.hd_global = 32; c.hd_swa = 64;
        const CudaScratchDims d = cuda_scratch_dims(c);
        check(d.head_dim == 64, "scratch head size is max(hd_global, hd_swa) when the sliding head is wider");
        check(d.q_floats == 128 && d.kv_floats == 64, "q/att and k/v scratch hold n_head and n_head_kv sliding heads");
        check(d.act_width == 128, "activation width covers attn_output's input (n_head x head size) when wider than d_ff");
    }
    {
        Gemma4Config c; c.d_model = 2560; c.d_ff = 1024; c.d_ple = 4096; c.n_head = 8; c.n_head_kv = 2; c.hd_global = 512; c.hd_swa = 256;
        const CudaScratchDims d = cuda_scratch_dims(c);
        check(d.act_width == 4096, "activation width covers d_model and d_ple when they are wider than d_ff");
        check(d.xq_bytes >= sizeof(BlockQ8_0) * (4096 / 32) && d.xq_bytes >= sizeof(BlockQ8_K) * (4096 / QK_K), "quantised scratch holds the widest row in both Q8_0 and Q8_K form");
    }
    {   // The shipped E4B shapes: nothing gets smaller than before.
        Gemma4Config c; c.d_model = 2560; c.d_ff = 10240; c.d_ple = 256; c.n_head = 8; c.n_head_kv = 2; c.hd_global = 512; c.hd_swa = 256;
        const CudaScratchDims d = cuda_scratch_dims(c);
        check(d.q_floats == size_t(8 * 512) && d.kv_floats == size_t(2 * 512) && d.act_width == 10240, "E4B shapes give the same scratch sizes as before");
    }
    std::printf("[CUDALIM] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
