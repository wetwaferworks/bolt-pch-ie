// Bolt PCHIE - deterministic scalar math shared by the CPU and CUDA paths (clean-room).
// Only IEEE-754 round-to-nearest + - * / and exact operations (compare, floor, bit casts)
// are used, so a build with FMA contraction off (MSVC fp_contract off, nvcc -fmad=false,
// -prec-div=true, -prec-sqrt=true) gives the same bits on both sides.
#pragma once
#include <cstdint>
#include <math.h>
#include <string.h>

#if defined(__CUDACC__)
#define PCHIE_HD __host__ __device__ __forceinline__
#else
#define PCHIE_HD inline
#endif
#if defined(_MSC_VER) && !defined(__clang__) && !defined(__CUDA_ARCH__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

namespace pchie {

PCHIE_HD float det_f32_from_bits(uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float f; memcpy(&f, &u, 4); return f;
#endif
}
PCHIE_HD uint32_t det_bits_from_f32(float f) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(f);
#else
    uint32_t u; memcpy(&u, &f, 4); return u;
#endif
}

// Same algorithm and results as fp16_to_fp32_exact() in pchie_q4_det.h.
PCHIE_HD float det_fp16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    int32_t  e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x03FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else { e = 1; while ((m & 0x0400u) == 0) { m <<= 1; e--; } m &= 0x03FFu; bits = sign | (static_cast<uint32_t>(e + 112) << 23) | (m << 13); }
    } else if (e == 0x1F) bits = sign | 0x7F800000u | (m << 13);
    else bits = sign | (static_cast<uint32_t>(e + 112) << 23) | (m << 13);
    return det_f32_from_bits(bits);
}
PCHIE_HD float det_bf16_to_f32(uint16_t h) { return det_f32_from_bits(static_cast<uint32_t>(h) << 16); }

// Same results as round_half_even_f32() in pchie_q4_det.h. NaN gives 0 and values outside the int32 range
// saturate, so the float-to-int cast is always defined and the CPU and GPU agree (both are undefined or
// differ between x86 and CUDA otherwise). Every value inside the range gives the same result as before.
PCHIE_HD int32_t det_round_half_even(float x) {
    if (!(x == x)) return 0;
    if (x >= 2147483648.0f) return 2147483647;
    if (x < -2147483648.0f) return -2147483647 - 1;
    const float fl = floorf(x);
    const float diff = x - fl;
    int32_t r = static_cast<int32_t>(fl);
    if (diff > 0.5f) r += 1;
    else if (diff == 0.5f && (r & 1)) r += 1;
    return r;
}

// Archie's Pade tanh. Local fix: the rational form overshoots 1 for |x| > ~4.97
// (1.0053 at x = 9), so the output is clamped to [-1, 1]. Max abs error 9.6e-5.
PCHIE_HD float tanh_deterministic(float x) {
    x = (x > 9.0f) ? 9.0f : ((x < -9.0f) ? -9.0f : x);
    const float x2  = x * x;
    const float num = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
    const float den = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
    const float r = num / den;
    return (r > 1.0f) ? 1.0f : ((r < -1.0f) ? -1.0f : r);
}
PCHIE_HD float gelu_tanh_deterministic(float z) {
    const float k0 = 0.7978845608028654f; // sqrt(2/pi)
    const float z3 = z * z * z;
    const float inner = k0 * (z + 0.044715f * z3);
    return 0.5f * z * (1.0f + tanh_deterministic(inner));
}

// exp() from + - * / only: x = k ln2 + r, |r| <= ln2/2, degree-6 Taylor, scale by 2^k
// through the exponent bits. Relative error < 2e-7 on [-87, 88]; 0 below -87.
PCHIE_HD float exp_deterministic(float x) {
    if (!(x == x)) return x;                       // NaN in, NaN out (never cast to int)
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    const int32_t k = det_round_half_even(x * 1.44269504f);
    const float kf = static_cast<float>(k);
    const float r = (x - kf * 0.693145752f) - kf * 1.42860677e-6f;
    float p = 1.0f / 720.0f;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    int32_t e = k + 127;
    if (e < 1) {                                   // keep the scale normal: split it
        const float half = det_f32_from_bits(static_cast<uint32_t>(e + 64) << 23);
        return (p * half) * det_f32_from_bits(static_cast<uint32_t>(127 - 64) << 23);
    }
    if (e > 254) e = 254;
    return p * det_f32_from_bits(static_cast<uint32_t>(e) << 23);
}

} // namespace pchie
