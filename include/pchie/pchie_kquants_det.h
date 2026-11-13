// Bolt PCHIE - deterministic K-quant / Q8_0 / BF16 kernels (clean-room).
//
// Weight formats follow the public GGUF block layouts. Every weight x activation
// inner product is exact int32 arithmetic inside a super-block; the only float
// work is one scale multiply per super-block, summed sequentially b = 0..N-1
// with FMA contraction disabled. Because the integer part is exact, the AVX2
// paths below produce the SAME BITS as the scalar reference paths (checked by
// pchie_kernels_test).
//
// tanh_deterministic, gelu_tanh_deterministic and exp_deterministic live in
// pchie_det_math.h (shared with CUDA).
// Base kernels: Archie's gemma4 kernel specification, section 2. Local
// change: activation quantization uses the repo's pinned round_half_even_f32()
// instead of std::nearbyint (same result in the default FP mode, no mode dependence).
#pragma once
#include "pchie_q4_det.h"
#include "pchie_det_math.h"
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace pchie {

constexpr int QK_K = 256;

#pragma pack(push, 1)
struct BlockQ4_K {               // 144 bytes / 256 weights
    uint16_t d_fp16;
    uint16_t dmin_fp16;
    uint8_t  scales[12];
    uint8_t  qs[QK_K / 2];
};
struct BlockQ6_K {               // 210 bytes / 256 weights
    uint8_t  ql[QK_K / 2];
    uint8_t  qh[QK_K / 4];
    int8_t   scales[QK_K / 16];
    uint16_t d_fp16;
};
struct BlockQ8_0W {              // 34 bytes / 32 weights (weight-side Q8_0)
    uint16_t d_fp16;
    int8_t   qs[32];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_K) == 144, "Q4_K");
static_assert(sizeof(BlockQ6_K) == 210, "Q6_K");
static_assert(sizeof(BlockQ8_0W) == 34, "Q8_0");

struct BlockQ8_K {               // activation super-block
    float   d_fp32;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

inline float bf16_to_fp32(uint16_t h) noexcept { return det_bf16_to_f32(h); }

inline void quantize_row_q8_K_deterministic(const float* src, BlockQ8_K* dst, int k) noexcept {
    const int nb = k / QK_K;
    for (int b = 0; b < nb; ++b) {
        const float* x = src + b * QK_K;
        float amax = 0.0f, max_val = 0.0f;
        for (int i = 0; i < QK_K; ++i) {
            const float ax = std::fabs(x[i]);
            if (ax > amax) { amax = ax; max_val = x[i]; }
        }
        if (amax == 0.0f) { std::memset(&dst[b], 0, sizeof(BlockQ8_K)); continue; }
        const float iscale = -127.0f / max_val;
        dst[b].d_fp32 = 1.0f / iscale;
        for (int i = 0; i < QK_K; ++i) {
            int32_t v = round_half_even_f32(iscale * x[i]);
            v = std::max(-127, std::min(127, v));
            dst[b].qs[i] = static_cast<int8_t>(v);
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            int32_t s = 0;
            for (int i = 0; i < 16; ++i) s += dst[b].qs[j * 16 + i];
            dst[b].bsums[j] = static_cast<int16_t>(s);
        }
    }
}

inline void unpack_q4_K_scales(const uint8_t s[12], uint8_t sc[8], uint8_t m[8]) noexcept {
    for (int j = 0; j < 4; ++j) { sc[j] = s[j] & 63u; m[j] = s[j + 4] & 63u; }
    for (int j = 4; j < 8; ++j) {
        sc[j] = static_cast<uint8_t>((s[j + 4] & 0x0Fu) | ((s[j - 4] >> 6) << 4));
        m[j]  = static_cast<uint8_t>((s[j + 4] >> 4)    | ((s[j]     >> 6) << 4));
    }
}

// ---- integer cores: scalar reference -------------------------------------------------
inline void q4k_int_scalar(const BlockQ4_K& wb, const BlockQ8_K& xb, int32_t& isc, int32_t& imn) noexcept {
    uint8_t sc[8], m[8];
    unpack_q4_K_scales(wb.scales, sc, m);
    const uint8_t* q4 = wb.qs; const int8_t* q8 = xb.qs;
    isc = 0; imn = 0;
    for (int j = 0; j < 4; ++j) {
        int32_t d0 = 0, d1 = 0;
        for (int l = 0; l < 32; ++l) {
            d0 += static_cast<int32_t>(q4[l] & 0x0Fu) * q8[l];
            d1 += static_cast<int32_t>(q4[l] >> 4)    * q8[l + 32];
        }
        const int s0 = 2 * j, s1 = 2 * j + 1;
        isc += static_cast<int32_t>(sc[s0]) * d0 + static_cast<int32_t>(sc[s1]) * d1;
        imn += static_cast<int32_t>(m[s0]) * (xb.bsums[2 * s0] + xb.bsums[2 * s0 + 1])
             + static_cast<int32_t>(m[s1]) * (xb.bsums[2 * s1] + xb.bsums[2 * s1 + 1]);
        q4 += 32; q8 += 64;
    }
}
inline int32_t q6k_int_scalar(const BlockQ6_K& wb, const BlockQ8_K& xb) noexcept {
    const uint8_t* ql = wb.ql; const uint8_t* qh = wb.qh; const int8_t* sc = wb.scales; const int8_t* q8 = xb.qs;
    int32_t s = 0;
    for (int half = 0; half < 2; ++half) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int32_t q1 = static_cast<int32_t>((ql[l]      & 0x0Fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
            const int32_t q2 = static_cast<int32_t>((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
            const int32_t q3 = static_cast<int32_t>((ql[l]      >> 4)    | (((qh[l] >> 4) & 3u) << 4)) - 32;
            const int32_t q4 = static_cast<int32_t>((ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3u) << 4)) - 32;
            s += sc[is + 0] * q1 * q8[l + 0];
            s += sc[is + 2] * q2 * q8[l + 32];
            s += sc[is + 4] * q3 * q8[l + 64];
            s += sc[is + 6] * q4 * q8[l + 96];
        }
        ql += 64; qh += 32; sc += 8; q8 += 128;
    }
    return s;
}
inline int32_t q8_0_int_scalar(const BlockQ8_0W& wb, const BlockQ8_0& xb) noexcept {
    int32_t s = 0;
    for (int i = 0; i < 32; ++i) s += static_cast<int32_t>(wb.qs[i]) * xb.qs[i];
    return s;
}

#if defined(__AVX2__)
inline int32_t hsum_i32x8(__m256i v) noexcept {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1));
    return _mm_cvtsi128_si32(a);
}
// u8 (0..63) x s8 -> s32 per 32 lanes, exact (pair sums <= 2*63*127 < 32767).
inline __m256i dot_u8s8_32(__m256i u, __m256i s) noexcept {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
}
inline void q4k_int_avx2(const BlockQ4_K& wb, const BlockQ8_K& xb, int32_t& isc, int32_t& imn) noexcept {
    uint8_t sc[8], m[8];
    unpack_q4_K_scales(wb.scales, sc, m);
    const __m256i lo = _mm256_set1_epi8(0x0F);
    __m256i acc = _mm256_setzero_si256();
    for (int j = 0; j < 4; ++j) {
        const __m256i q4 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb.qs + 32 * j));
        const __m256i a  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs + 64 * j));
        const __m256i b  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs + 64 * j + 32));
        const __m256i p0 = dot_u8s8_32(_mm256_and_si256(q4, lo), a);
        const __m256i p1 = dot_u8s8_32(_mm256_and_si256(_mm256_srli_epi16(q4, 4), lo), b);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p0, _mm256_set1_epi32(sc[2 * j])));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(p1, _mm256_set1_epi32(sc[2 * j + 1])));
    }
    isc = hsum_i32x8(acc);
    imn = 0;
    for (int s = 0; s < 8; ++s) imn += static_cast<int32_t>(m[s]) * (xb.bsums[2 * s] + xb.bsums[2 * s + 1]);
}
inline int32_t q6k_int_avx2(const BlockQ6_K& wb, const BlockQ8_K& xb) noexcept {
    // unsigned 6-bit q (0..63) x q8, then subtract 32 * sum(q8) per 16-group: exact.
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3);
    int32_t total = 0;
    const uint8_t* ql = wb.ql; const uint8_t* qh = wb.qh; const int8_t* sc = wb.scales; const int8_t* q8 = xb.qs;
    const int16_t* bs = xb.bsums;
    for (int half = 0; half < 2; ++half) {
        const __m256i l0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql));
        const __m256i l1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ql + 32));
        const __m256i h  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh));
        const __m256i u1 = _mm256_or_si256(_mm256_and_si256(l0, m4), _mm256_slli_epi16(_mm256_and_si256(h, m2), 4));
        const __m256i u2 = _mm256_or_si256(_mm256_and_si256(l1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 2), m2), 4));
        const __m256i u3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 4), m2), 4));
        const __m256i u4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(h, 6), m2), 4));
        const __m256i us[4] = {u1, u2, u3, u4};
        for (int g = 0; g < 4; ++g) {
            const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8 + 32 * g));
            const __m256i p = dot_u8s8_32(us[g], x); // lanes 0..3 = elems 0..15, lanes 4..7 = 16..31
            alignas(32) int32_t t[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(t), p);
            const int32_t s0 = t[0] + t[1] + t[2] + t[3];
            const int32_t s1 = t[4] + t[5] + t[6] + t[7];
            total += sc[2 * g + 0] * (s0 - 32 * bs[2 * g + 0]);
            total += sc[2 * g + 1] * (s1 - 32 * bs[2 * g + 1]);
        }
        ql += 64; qh += 32; sc += 8; q8 += 128; bs += 8;
    }
    return total;
}
inline int32_t q8_0_int_avx2(const BlockQ8_0W& wb, const BlockQ8_0& xb) noexcept {
    const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wb.qs));
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xb.qs));
    const __m256i aw = _mm256_sign_epi8(w, w);    // |w| read as unsigned (0x80 reads as 128, still exact)
    const __m256i sx = _mm256_sign_epi8(x, w);
    return hsum_i32x8(_mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1)));
}
#endif

// ---- row dots: fixed sequential float combine ---------------------------------------
template <bool Simd>
inline float dot_q4_K_q8_K(const BlockQ4_K* w, const BlockQ8_K* x, int nb) noexcept {
    float total = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t isc, imn;
#if defined(__AVX2__)
        if (Simd) q4k_int_avx2(w[b], x[b], isc, imn); else
#endif
        q4k_int_scalar(w[b], x[b], isc, imn);
        const float d_all    = x[b].d_fp32 * fp16_to_fp32_exact(w[b].d_fp16);
        const float dmin_all = x[b].d_fp32 * fp16_to_fp32_exact(w[b].dmin_fp16);
        const float tp = d_all * static_cast<float>(isc);
        const float tn = dmin_all * static_cast<float>(imn);
        const float sb = tp - tn;
        total = total + sb;
    }
    return total;
}
template <bool Simd>
inline float dot_q6_K_q8_K(const BlockQ6_K* w, const BlockQ8_K* x, int nb) noexcept {
    float total = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t is;
#if defined(__AVX2__)
        if (Simd) is = q6k_int_avx2(w[b], x[b]); else
#endif
        is = q6k_int_scalar(w[b], x[b]);
        const float d_all = x[b].d_fp32 * fp16_to_fp32_exact(w[b].d_fp16);
        const float sb = d_all * static_cast<float>(is);
        total = total + sb;
    }
    return total;
}
template <bool Simd>
inline float dot_q8_0_q8_0(const BlockQ8_0W* w, const BlockQ8_0* x, int nb) noexcept {
    float total = 0.0f;
    for (int b = 0; b < nb; ++b) {
        int32_t is;
#if defined(__AVX2__)
        if (Simd) is = q8_0_int_avx2(w[b], x[b]); else
#endif
        is = q8_0_int_scalar(w[b], x[b]);
        const float sc = fp16_to_fp32_exact(w[b].d_fp16) * x[b].d_fp32;
        const float t = sc * static_cast<float>(is);
        total = total + t;
    }
    return total;
}
// BF16 x fp32: 8 fixed lanes, lanes combined in a fixed tree. Pure function of inputs.
inline float dot_bf16_f32(const uint16_t* w, const float* x, int n) noexcept {
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < n; i += 8)
        for (int k = 0; k < 8; ++k) { const float p = bf16_to_fp32(w[i + k]) * x[i + k]; lane[k] = lane[k] + p; }
    const float a = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    const float b = (lane[4] + lane[5]) + (lane[6] + lane[7]);
    return a + b;
}

// ---- reference dequantizers (tests + embedding row lookups) ---------------------------
inline void dequant_q8_0_row(const BlockQ8_0W* w, float* y, int n) noexcept {
    for (int b = 0; b < n / 32; ++b) {
        const float d = fp16_to_fp32_exact(w[b].d_fp16);
        for (int i = 0; i < 32; ++i) y[b * 32 + i] = d * static_cast<float>(w[b].qs[i]);
    }
}
inline void dequant_q4_K_row(const BlockQ4_K* w, float* y, int n) noexcept {
    for (int b = 0; b < n / QK_K; ++b) {
        uint8_t sc[8], m[8]; unpack_q4_K_scales(w[b].scales, sc, m);
        const float d = fp16_to_fp32_exact(w[b].d_fp16), dm = fp16_to_fp32_exact(w[b].dmin_fp16);
        for (int j = 0; j < 4; ++j)
            for (int l = 0; l < 32; ++l) {
                const uint8_t q = w[b].qs[32 * j + l];
                y[b * QK_K + 64 * j + l]      = d * sc[2 * j]     * (q & 0xF) - dm * m[2 * j];
                y[b * QK_K + 64 * j + 32 + l] = d * sc[2 * j + 1] * (q >> 4)  - dm * m[2 * j + 1];
            }
    }
}
inline void dequant_q6_K_row(const BlockQ6_K* w, float* y, int n) noexcept {
    for (int b = 0; b < n / QK_K; ++b) {
        const float d = fp16_to_fp32_exact(w[b].d_fp16);
        const uint8_t* ql = w[b].ql; const uint8_t* qh = w[b].qh; const int8_t* sc = w[b].scales;
        float* o = y + b * QK_K;
        for (int half = 0; half < 2; ++half) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                o[l]      = d * sc[is + 0] * (static_cast<int>((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                o[l + 32] = d * sc[is + 2] * (static_cast<int>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                o[l + 64] = d * sc[is + 4] * (static_cast<int>((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                o[l + 96] = d * sc[is + 6] * (static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            ql += 64; qh += 32; sc += 8; o += 128;
        }
    }
}

} // namespace pchie
