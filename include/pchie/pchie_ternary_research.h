// Bolt PCHIE - ternary research path (clean-room): the pieces Archie's Nov 16 brief asked us to measure.
//
// Nothing here is used by the shipping ternary path (pchie_ternary.h) or by any model. It exists so the
// measurements can be rerun and so every alternative is held to the same bar as the shipping code:
// bounded inputs, a safe state on failure (Status plus zeroed outputs), no allocation in the kernels, no locks,
// and every SIMD kernel bit-identical to its scalar twin and to the shipping I2 reference.
//
// Contents
//   1. Converter-side ternarization: absmean (BitNet b1.58), the exact least-squares fit per group (closed form
//      by sorting), and an iterative (Lloyd) fit; three tie / rounding rules for offline conversion only.
//   2. Two more packings for the same ternary values:
//        B3B  five trits per byte, block-aligned: 26 bytes per 128-weight block (1.625 bits per weight).
//             Byte j (0..25) holds weights j, j+26, j+52, j+78, j+104 as trits (q+1) * 3^k, k = 0..4.
//             Slots past weight 127 (k = 4, j = 24, 25) must hold the zero weight. Bytes > 242 are invalid.
//             Decoded with divide-by-3 multiplies; the activation row is laid out in the same plane order once.
//        TL3  sign plus 4-bit index per three weights, tiled by 32 rows for table-lookup kernels.
//             A 128-weight block is 43 triples (weights 3t, 3t+1, 3t+2; the 129th slot must be zero).
//             Triple (a, b, c) -> v = 9(a+1) + 3(b+1) + (c+1) in 0..26; v > 13 is stored as sign 1 with
//             index 26 - v (its mirror), otherwise sign 0 with index v. Index 14, 15 and "sign 1 with index 13"
//             are invalid. Per tile and block, per triple: 16 bytes of indices (low nibble row i, high nibble
//             row 16 + i) then 4 bytes of sign bits (bit r = row r). 860 bytes per 32 x 128 = 1.68 bits per weight.
//   3. Integer-only base-2 softmax (three ways to get 2^f) and integer RMSNorm with a table-seeded Newton
//      reciprocal square root.
#pragma once
#include "pchie/pchie_ternary.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pchie {
namespace ternary {
namespace research {

// ------------------------------------------------------------------------------------------------ converter
enum class Fit : uint8_t { Absmean = 0, LeastSquares = 1, Lloyd = 2 };
enum class Round : uint8_t { HalfEven = 0, HalfAway = 1, Stochastic = 2 };   // ties: to 0 / to +-1 / random
constexpr size_t kMaxGroup = size_t(1) << 26;   // a whole 8,192 x 8,192 tensor (per-tensor scale)
// Ternarizes one group of n weights. gamma is stored as float and the codes are assigned against that float
// (so converter and engine agree). LeastSquares minimizes sum (w - gamma q)^2 exactly over all ternary codes;
// Lloyd alternates threshold gamma/2 and gamma = mean |w| of the non-zeros, starting from absmean.
// Stochastic rounding is seeded (deterministic for a given seed) and is meant for offline conversion only.
// A gamma that would be subnormal in float is flushed to an exact zero group (all codes 0, gamma 0).
// scratch must hold n floats for LeastSquares (no allocation inside). Fails safe: q zeroed, gamma 0.
Status ternarize_group(const float* w, size_t n, Fit fit, Round rnd, uint64_t seed, int lloyd_iters,
                       float* scratch, int8_t* q, float* gamma) noexcept;

// ------------------------------------------------------------------------------------------------ packings
constexpr int kB3BBlockBytes = 26;
constexpr int kTL3Triples = 43;
constexpr int kTL3TileRows = 32;
constexpr int kTL3TripleBytes = 20;   // 16 index bytes + 4 sign bytes
constexpr int kTL3BlockBytes = kTL3Triples * kTL3TripleBytes;   // per 32-row tile

struct PackedTensor {                 // B3B or TL3; scales exactly as in pchie_ternary.h
    enum class Kind : uint8_t { B3B = 1, TL3 = 2 } kind = Kind::B3B;
    ScaleMode scale_mode = ScaleMode::PerTensor;
    uint64_t rows = 0, cols = 0;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
};
// From a shipping I2 tensor (validated by view_of). TL3 needs rows % 32 == 0.
Status pack_b3b(const Tensor& i2, PackedTensor& out);
Status pack_tl3(const Tensor& i2, PackedTensor& out);
// Strict checks (every code, the padding slots, sizes, scales). Kernels assume a tensor that passed this.
Status validate(const PackedTensor& t) noexcept;
// Unpack one row to trits (for round-trip tests and decode benchmarks).
Status unpack_row(const PackedTensor& t, uint64_t row, int8_t* q) noexcept;

// Activation side data, built once per activation row and shared by every weight row.
struct B3BAct { std::vector<int8_t> xp; };                 // 160 slots per block: plane k at k*32 + j, zeros
struct TL3Act { std::vector<uint8_t> lo; std::vector<int8_t> hi; };   // per block, per triple: 16 entries
Status prepare_b3b(const ActRow& x, B3BAct& out);
Status prepare_tl3(const ActRow& x, TL3Act& out);

// Kernels. All must return the shipping I2 reference bits for the same trits, scales and activations.
// y has rows entries. A shape mismatch or a non-finite output zero-fills y and returns an error.
enum class Kernel : uint8_t {
    I2Scalar = 0,      // shipping reference
    I2Avx2Biased,      // the shipping AVX2 row kernel, called directly: sum(c x) - sum(x)
    I2Avx2Signed,      // signed alternative: sign_epi8 then a multiply-add by ones (no bsum)
    I2Avx2ZeroSkip,    // biased kernel plus a per-block "all zero" test
    B3BScalar, B3BAvx2,
    TL3Scalar, TL3Avx2,
};
const char* kernel_name(Kernel k) noexcept;
bool kernel_available(Kernel k) noexcept;   // AVX2 kernels need an AVX2 CPU and OS support
Status matvec_i2(const View& w, const ActRow& x, Kernel k, float* y, uint64_t y_len) noexcept;
Status matvec_b3b(const PackedTensor& w, const ActRow& x, const B3BAct& xa, Kernel k, float* y, uint64_t y_len) noexcept;
Status matvec_tl3(const PackedTensor& w, const ActRow& x, const TL3Act& xa, Kernel k, float* y, uint64_t y_len) noexcept;
// Decode a whole row into int8 trits with SIMD where it exists (decode-throughput measurement only).
Status decode_row_i2_avx2(const View& w, uint64_t row, int8_t* q) noexcept;
Status decode_row_b3b_avx2(const PackedTensor& w, uint64_t row, int8_t* q) noexcept;

// ------------------------------------------------------------------------------------------------ softmax
enum class Exp2 : uint8_t { Table256 = 0, Table64Lerp = 1, Poly3 = 2 };
const char* exp2_name(Exp2 m) noexcept;
// 2^(f / 65536) in Q30 for f in [0, 65536). Result in [2^30, 2^31].
uint32_t exp2_frac_q30(uint32_t f, Exp2 m) noexcept;
constexpr size_t kMaxSoftmax = size_t(1) << 20;
// Logits in Q16 (int32). Max-subtracted inside; differences below -64.0 give exactly 0.
// p with out_bits fractional bits (8..30; 1.0 = 2^out_bits), each p_i rounded half to even, so
// |sum p - 2^out_bits| <= n / 2. softmax_q16 is out_bits = 16.
Status softmax_fixed(const int32_t* z, size_t n, Exp2 m, int out_bits, uint32_t* p) noexcept;
inline Status softmax_q16(const int32_t* z, size_t n, Exp2 m, uint32_t* p) noexcept { return softmax_fixed(z, n, m, 16, p); }

// ------------------------------------------------------------------------------------------------ RMSNorm
constexpr size_t kMaxNorm = size_t(1) << 20;
// y_i = x_i / sqrt(mean(x^2) + eps), in Q16 (int32). Inputs bounded: |x_i| <= 32767 (-32768 refused),
// 1 <= n <= 2^20, 1 <= eps_q8 <= 2^32 (eps in x units squared, Q8). Proof of no overflow in the .cpp.
Status rmsnorm_q16(const int16_t* x, size_t n, uint32_t eps_q8, int32_t* y) noexcept;
// 1/sqrt(u / 2^30) in Q30 for u in [2^30, 2^32), with `steps` Newton steps (0..3) from the 64-entry seed.
uint32_t rsqrt_q30(uint32_t u, int steps) noexcept;

// Round-half-to-even arithmetic right shift (the one rounding helper every fixed-point step uses).
int64_t rshift_rne(int64_t v, int s) noexcept;

}  // namespace research
}  // namespace ternary
}  // namespace pchie
