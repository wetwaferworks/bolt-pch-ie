// Bolt PCHIE - ternary research path, shared internals (clean-room).
#pragma once
#include "pchie/pchie_ternary_research.h"
#include <cstdint>

namespace pchie {
namespace ternary {
namespace research {
constexpr uint64_t kB3BActSlots = 160;   // 5 planes x 32 slots per block (slots 26..31 and past weight 127 are 0)
uint64_t tl3_offset(uint64_t tile, uint64_t nb, uint64_t b, int t) noexcept;
namespace detail {
// AVX2 kernels (pchie_ternary_research_simd.cpp). Only called after the AVX2 check.
float row_i2_signed_avx2(const View& w, uint64_t row, const ActRow& x) noexcept;
float row_i2_zeroskip_avx2(const View& w, uint64_t row, const ActRow& x) noexcept;
float row_b3b_avx2(const PackedTensor& w, uint64_t row, const ActRow& x, const B3BAct& xa) noexcept;
void tile_tl3_avx2(const PackedTensor& w, uint64_t tile, const ActRow& x, const TL3Act& xa, float* y) noexcept;
void decode_i2_avx2(const uint8_t* row_codes, uint64_t nb, int8_t* q) noexcept;
void decode_b3b_avx2(const PackedTensor& w, uint64_t row, int8_t* q) noexcept;
}  // namespace detail
}  // namespace research
}  // namespace ternary
}  // namespace pchie
