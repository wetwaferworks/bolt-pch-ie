#include "pchie/pchie_q4_det.h"

namespace pchie {

float dot_q4_0_q8_0_deterministic(const BlockQ4_0* w_row, const BlockQ8_0* x_vec, int num_blocks) noexcept {
    float total = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
        const BlockQ4_0& wb = w_row[b];
        const BlockQ8_0& xb = x_vec[b];
        int32_t isum = 0; // exact integer accumulation
        for (int i = 0; i < Q_BLOCK_SIZE / 2; ++i) {
            const int32_t w0 = static_cast<int32_t>(wb.qs[i] & 0x0Fu) - 8;
            const int32_t w1 = static_cast<int32_t>(wb.qs[i] >> 4) - 8;
            isum += w0 * xb.qs[i];
            isum += w1 * xb.qs[i + Q_BLOCK_SIZE / 2];
        }
        const float scale = fp16_to_fp32_exact(wb.d_fp16) * xb.d_fp32;
        const float term  = scale * static_cast<float>(isum);
        total = total + term;
    }
    return total;
}

void quantize_row_q8_0_deterministic(const float* src, BlockQ8_0* dst, int k) noexcept {
    const int nb = k / Q_BLOCK_SIZE;
    for (int b = 0; b < nb; ++b) {
        const float* x = src + b * Q_BLOCK_SIZE;
        float amax = 0.0f;
        for (int i = 0; i < Q_BLOCK_SIZE; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float d  = amax / 127.0f;
        const float id = (amax != 0.0f) ? 127.0f / amax : 0.0f;
        dst[b].d_fp32 = d;
        for (int i = 0; i < Q_BLOCK_SIZE; ++i) {
            int32_t q = round_half_even_f32(x[i] * id);
            q = std::max(-127, std::min(127, q));
            dst[b].qs[i] = static_cast<int8_t>(q);
        }
    }
}

} // namespace pchie
