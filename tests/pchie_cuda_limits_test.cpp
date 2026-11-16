// CUDA backend shape limits, checked on the host (no GPU needed). The matvec kernels keep one partial sum
// per block of a row in a fixed shared array, so a row with more blocks than that must be refused before
// anything is launched (Helia's review: a Q8_0 weight with ne0 above 4096 wrote past the array).
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
    std::printf("[CUDALIM] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
