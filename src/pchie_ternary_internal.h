// Bolt PCHIE - ternary path, pieces shared by the reference and the SIMD kernels (clean-room).
// The float tail lives here once, so every kernel does the same float operations in the same order.
#pragma once
#include "pchie/pchie_ternary.h"
#include <cstdint>
#include <cstring>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define PCHIE_TERNARY_X86 1
#else
#define PCHIE_TERNARY_X86 0
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#define PCHIE_TERNARY_ARM64 1
#else
#define PCHIE_TERNARY_ARM64 0
#endif

namespace pchie {
namespace ternary {
namespace detail {

// PerTensor tail: one exact int32 row sum, then two multiplies in a fixed order.
inline float tail_tensor(int32_t isum, float gamma, float sx) noexcept {
    const float a = static_cast<float>(isum) * gamma;
    return a * sx;
}
// PerBlock tail, one step: acc + float(isum_b) * gamma_b (no FMA: contraction is off).
inline float tail_block_step(float acc, int32_t isum_b, float gamma_b) noexcept {
    const float p = static_cast<float>(isum_b) * gamma_b;
    return acc + p;
}
inline float load_f32_le(const uint8_t* p) noexcept {
    uint32_t u = uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
    float f; std::memcpy(&f, &u, 4); return f;
}

// Signed reference dot of one I2 block (128 weights) against 128 int8 activations. Plain obvious math.
inline int32_t block_dot_i2_reference(const uint8_t* blk, const int8_t* x) noexcept {
    int32_t s = 0;
    for (int j = 0; j < kBlockBytes; ++j) {
        const uint8_t byte = blk[j];
        for (int k = 0; k < 4; ++k) {
            const int code = (byte >> (2 * k)) & 3;
            const int w = code - 1;                    // code 3 never reaches here (rejected at load)
            s += w * int32_t(x[k * 32 + j]);
        }
    }
    return s;
}

// Row kernels for I2 (defined in pchie_ternary_simd.cpp). nb = blocks in the row.
float row_dot_i2_ssse3(const View& w, uint64_t row, const ActRow& x) noexcept;
float row_dot_i2_avx2(const View& w, uint64_t row, const ActRow& x) noexcept;
float row_dot_i2_neon(const View& w, uint64_t row, const ActRow& x) noexcept;
bool simd_compiled(Tier t) noexcept;

}  // namespace detail
}  // namespace ternary
}  // namespace pchie
