// Bolt PCHIE - deterministic Q4_0 x Q8_0 dot product (clean-room).
//
// ROUNDING RULE (pinned): every float->int quantization uses
// round_half_even_f32(), an explicit routine (no dependence on the current
// FP rounding mode, std::nearbyint, or compiler lowering). Ties go to the
// nearest even integer: 0.5->0, 1.5->2, 2.5->2, -0.5->0, -1.5->-2.
// Float sums are plain sequential adds in block order b = 0..N-1; FMA
// contraction is disabled by the build (-ffp-contract=off / MSVC /fp:precise).
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace pchie {

constexpr int Q_BLOCK_SIZE = 32;

#pragma pack(push, 1)
struct BlockQ4_0 {
    uint16_t d_fp16;
    uint8_t  qs[Q_BLOCK_SIZE / 2];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must be 18 bytes");

struct BlockQ8_0 {
    float  d_fp32;
    int8_t qs[Q_BLOCK_SIZE];
};

// NaN gives 0 and values outside the int32 range saturate (the cast would be undefined); same results as
// before for every value inside the range.
inline int32_t round_half_even_f32(float x) noexcept {
    if (!(x == x)) return 0;
    if (x >= 2147483648.0f) return 2147483647;
    if (x < -2147483648.0f) return -2147483647 - 1;
    const float fl = std::floor(x);           // exact
    const float diff = x - fl;                // exact for |x| < 2^23
    int32_t r = static_cast<int32_t>(fl);
    if (diff > 0.5f) r += 1;
    else if (diff == 0.5f && (r & 1)) r += 1;
    return r;
}

inline float fp16_to_fp32_exact(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    int32_t  exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | (static_cast<uint32_t>(exp + 112) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float dot_q4_0_q8_0_deterministic(const BlockQ4_0* w_row, const BlockQ8_0* x_vec, int num_blocks) noexcept;
void  quantize_row_q8_0_deterministic(const float* src, BlockQ8_0* dst, int k_elements) noexcept;

struct InferenceRequest {
    uint32_t fork_id;
    uint32_t node_id;
    uint64_t handshake_tick;
};

inline void canonicalize_batch(std::vector<InferenceRequest>& batch) {
    std::sort(batch.begin(), batch.end(), [](const InferenceRequest& a, const InferenceRequest& b) {
        if (a.fork_id != b.fork_id) return a.fork_id < b.fork_id;
        return a.node_id < b.node_id;
    });
}

} // namespace pchie
