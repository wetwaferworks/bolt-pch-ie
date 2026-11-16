// Bolt PCHIE - ternary I2 row kernels: SSSE3 (the i3-2100 tier), AVX2 and NEON (clean-room).
//
// All three use the same identity. The I2 code is c = q + 1 in {0, 1, 2}, so
//     sum(q * x) = sum(c * x) - sum(x),
// and sum(x) per block (bsum) is computed once per activation row. c * x is an unsigned-by-signed byte product,
// which is exactly what x86 pmaddubsw does; pairs reach at most 2 * 2 * 127 = 508, far from int16 saturation.
// The integer block sum is therefore exactly the reference value, and the float tail is the shared code in
// pchie_ternary_internal.h, so every kernel returns the reference bits.
//
// These functions are compiled for their instruction set with function attributes (GCC/Clang) or plain
// intrinsics (MSVC), and are only called after run-time detection says the CPU and OS support them.
#include "pchie_ternary_internal.h"

#if PCHIE_TERNARY_X86
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#define PCHIE_TARGET(x)
#else
#define PCHIE_TARGET(x) __attribute__((target(x)))
#endif
#endif
#if PCHIE_TERNARY_ARM64
#include <arm_neon.h>
#endif

namespace pchie {
namespace ternary {
namespace detail {

#if PCHIE_TERNARY_X86
// ---------------------------------------------------------------- SSSE3 (128-bit integer, no AVX2 needed)
PCHIE_TARGET("ssse3")
static inline int32_t hsum_epi32_128(__m128i v) {
    __m128i s = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));   // swap 64-bit halves
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));           // swap 32-bit pairs
    return _mm_cvtsi128_si32(s);
}

PCHIE_TARGET("ssse3")
static int32_t block_cx_ssse3(const uint8_t* blk, const int8_t* x) {
    const __m128i mask = _mm_set1_epi8(3);
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk));
    const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 16));
    __m128i acc = _mm_setzero_si128();
    for (int k = 0; k < 4; ++k) {
        const __m128i sh = _mm_cvtsi32_si128(2 * k);
        const __m128i c0 = _mm_and_si128(_mm_srl_epi16(v0, sh), mask);
        const __m128i c1 = _mm_and_si128(_mm_srl_epi16(v1, sh), mask);
        const __m128i x0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + k * 32));
        const __m128i x1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + k * 32 + 16));
        const __m128i p0 = _mm_madd_epi16(_mm_maddubs_epi16(c0, x0), ones);
        const __m128i p1 = _mm_madd_epi16(_mm_maddubs_epi16(c1, x1), ones);
        acc = _mm_add_epi32(acc, _mm_add_epi32(p0, p1));
    }
    return hsum_epi32_128(acc);
}

PCHIE_TARGET("ssse3")
float row_dot_i2_ssse3(const View& w, uint64_t row, const ActRow& x) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    const int8_t* xq = x.q.data();
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_cx_ssse3(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b];
        return tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, block_cx_ssse3(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b], w.scale_at(row * nb + b));
    return acc * x.scale;
}

// ---------------------------------------------------------------- AVX2 (256-bit integer)
PCHIE_TARGET("avx2")
static int32_t block_cx_avx2(const uint8_t* blk, const int8_t* x) {
    const __m256i mask = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk));
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < 4; ++k) {
        const __m128i sh = _mm_cvtsi32_si128(2 * k);
        const __m256i c = _mm256_and_si256(_mm256_srl_epi16(v, sh), mask);
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + k * 32));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(c, xv), ones));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

PCHIE_TARGET("avx2")
float row_dot_i2_avx2(const View& w, uint64_t row, const ActRow& x) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    const int8_t* xq = x.q.data();
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_cx_avx2(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b];
        return tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, block_cx_avx2(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b], w.scale_at(row * nb + b));
    return acc * x.scale;
}
#else
float row_dot_i2_ssse3(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }   // never dispatched
float row_dot_i2_avx2(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }
#endif

#if PCHIE_TERNARY_ARM64
// ---------------------------------------------------------------- NEON (AArch64; sdot when the CPU has it)
static inline int32_t block_cx_neon(const uint8_t* blk, const int8_t* x) {
    const uint8x16_t mask = vdupq_n_u8(3);
    const uint8x16_t v0 = vld1q_u8(blk), v1 = vld1q_u8(blk + 16);
    int32x4_t acc = vdupq_n_s32(0);
    uint8x16_t c[8];
    c[0] = vandq_u8(v0, mask);              c[1] = vandq_u8(v1, mask);
    c[2] = vandq_u8(vshrq_n_u8(v0, 2), mask); c[3] = vandq_u8(vshrq_n_u8(v1, 2), mask);
    c[4] = vandq_u8(vshrq_n_u8(v0, 4), mask); c[5] = vandq_u8(vshrq_n_u8(v1, 4), mask);
    c[6] = vshrq_n_u8(v0, 6);               c[7] = vshrq_n_u8(v1, 6);
    for (int i = 0; i < 8; ++i) {
        const int8x16_t cs = vreinterpretq_s8_u8(c[i]);   // 0..2 fits a signed byte
        const int8x16_t xv = vld1q_s8(x + i * 16);
#if defined(__ARM_FEATURE_DOTPROD)
        acc = vdotq_s32(acc, cs, xv);
#else
        const int16x8_t lo = vmull_s8(vget_low_s8(cs), vget_low_s8(xv));
        const int16x8_t hi = vmull_high_s8(cs, xv);
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
#endif
    }
    return vaddvq_s32(acc);
}

float row_dot_i2_neon(const View& w, uint64_t row, const ActRow& x) noexcept {
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    const int8_t* xq = x.q.data();
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_cx_neon(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b];
        return tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b)
        acc = tail_block_step(acc, block_cx_neon(rc + b * kBlockBytes, xq + b * kBlock) - x.bsum[b], w.scale_at(row * nb + b));
    return acc * x.scale;
}
#else
float row_dot_i2_neon(const View&, uint64_t, const ActRow&) noexcept { return 0.0f; }   // never dispatched
#endif

bool simd_compiled(Tier t) noexcept {
    switch (t) {
        case Tier::Scalar: return true;
        case Tier::SSSE3: case Tier::AVX2: return PCHIE_TERNARY_X86 != 0;
        case Tier::NEON: return PCHIE_TERNARY_ARM64 != 0;
    }
    return false;
}

}  // namespace detail
}  // namespace ternary
}  // namespace pchie
