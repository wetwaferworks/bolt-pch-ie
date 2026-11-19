/* Bolt PCHIE - deterministic dequantizers for the ternary converter (clean-room, C17).
 *
 * Weight formats follow the public GGUF block layouts. The conversions (F16/BF16 to float, and Q8_0 / Q4_K /
 * Q6_K rows to float) match the repo's pchie_kquants_det.h / pchie_det_math.h bit for bit. FMA contraction is off.
 */
#ifndef PCHIE_WEIGHT_DEQUANTIZE_H
#define PCHIE_WEIGHT_DEQUANTIZE_H

#include "pchie/pchie_ternary_group_fit.h"
#include <stdint.h>
#include <string.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { TERNARY_KQUANT_SUPERBLOCK_WEIGHTS = 256 };

#pragma pack(push, 1)
typedef struct ternary_block_q4_k {
    uint16_t d_fp16;
    uint16_t dmin_fp16;
    uint8_t scales[12];
    uint8_t quantized[TERNARY_KQUANT_SUPERBLOCK_WEIGHTS / 2];
} ternary_block_q4_k;

typedef struct ternary_block_q6_k {
    uint8_t quantized_low[TERNARY_KQUANT_SUPERBLOCK_WEIGHTS / 2];
    uint8_t quantized_high[TERNARY_KQUANT_SUPERBLOCK_WEIGHTS / 4];
    int8_t scales[TERNARY_KQUANT_SUPERBLOCK_WEIGHTS / 16];
    uint16_t d_fp16;
} ternary_block_q6_k;

typedef struct ternary_block_q8_0 {
    uint16_t d_fp16;
    int8_t quantized[32];
} ternary_block_q8_0;
#pragma pack(pop)

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ternary_block_q4_k) == 144, "Q4_K block must be 144 bytes");
_Static_assert(sizeof(ternary_block_q6_k) == 210, "Q6_K block must be 210 bytes");
_Static_assert(sizeof(ternary_block_q8_0) == 34, "Q8_0 block must be 34 bytes");
#endif

/* Same algorithm as det_fp16_to_f32 / fp16_to_fp32_exact. */
static inline float ternary_float16_to_float32(uint16_t half) {
    const uint32_t sign = ((uint32_t)(half & 0x8000u)) << 16;
    int32_t exponent = (half >> 10) & 0x1F;
    uint32_t mantissa = half & 0x03FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x03FFu;
            bits = sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((uint32_t)(exponent + 112) << 23) | (mantissa << 13);
    }
    return ternary_float_from_bits(bits);
}

static inline float ternary_bfloat16_to_float32(uint16_t half) {
    return ternary_float_from_bits(((uint32_t)half) << 16);
}

static inline void ternary_unpack_q4_k_scales(const uint8_t packed[12], uint8_t scales[8], uint8_t mins[8]) {
    int j;
    for (j = 0; j < 4; ++j) {
        scales[j] = packed[j] & 63u;
        mins[j] = packed[j + 4] & 63u;
    }
    for (j = 4; j < 8; ++j) {
        scales[j] = (uint8_t)((packed[j + 4] & 0x0Fu) | ((packed[j - 4] >> 6) << 4));
        mins[j] = (uint8_t)((packed[j + 4] >> 4) | ((packed[j] >> 6) << 4));
    }
}

static inline void ternary_dequantize_q8_0_row(const ternary_block_q8_0 *blocks, float *out, int weight_count) {
    const int block_count = weight_count / 32;
    int b, i;
    for (b = 0; b < block_count; ++b) {
        const float scale = ternary_float16_to_float32(blocks[b].d_fp16);
        for (i = 0; i < 32; ++i) out[b * 32 + i] = scale * (float)blocks[b].quantized[i];
    }
}

static inline void ternary_dequantize_q4_k_row(const ternary_block_q4_k *blocks, float *out, int weight_count) {
    const int block_count = weight_count / TERNARY_KQUANT_SUPERBLOCK_WEIGHTS;
    int b, j, l;
    for (b = 0; b < block_count; ++b) {
        uint8_t scales[8], mins[8];
        const float d = ternary_float16_to_float32(blocks[b].d_fp16);
        const float dmin = ternary_float16_to_float32(blocks[b].dmin_fp16);
        ternary_unpack_q4_k_scales(blocks[b].scales, scales, mins);
        for (j = 0; j < 4; ++j) {
            for (l = 0; l < 32; ++l) {
                const uint8_t q = blocks[b].quantized[32 * j + l];
                out[b * TERNARY_KQUANT_SUPERBLOCK_WEIGHTS + 64 * j + l] =
                    d * (float)scales[2 * j] * (float)(q & 0xF) - dmin * (float)mins[2 * j];
                out[b * TERNARY_KQUANT_SUPERBLOCK_WEIGHTS + 64 * j + 32 + l] =
                    d * (float)scales[2 * j + 1] * (float)(q >> 4) - dmin * (float)mins[2 * j + 1];
            }
        }
    }
}

static inline void ternary_dequantize_q6_k_row(const ternary_block_q6_k *blocks, float *out, int weight_count) {
    const int block_count = weight_count / TERNARY_KQUANT_SUPERBLOCK_WEIGHTS;
    int b, half, l;
    for (b = 0; b < block_count; ++b) {
        const float d = ternary_float16_to_float32(blocks[b].d_fp16);
        const uint8_t *low = blocks[b].quantized_low;
        const uint8_t *high = blocks[b].quantized_high;
        const int8_t *scales = blocks[b].scales;
        float *dest = out + b * TERNARY_KQUANT_SUPERBLOCK_WEIGHTS;
        for (half = 0; half < 2; ++half) {
            for (l = 0; l < 32; ++l) {
                const int scale_pair = l / 16;
                dest[l] = d * (float)scales[scale_pair + 0] *
                          (float)((int)((low[l] & 0xF) | (((high[l] >> 0) & 3) << 4)) - 32);
                dest[l + 32] = d * (float)scales[scale_pair + 2] *
                               (float)((int)((low[l + 32] & 0xF) | (((high[l] >> 2) & 3) << 4)) - 32);
                dest[l + 64] = d * (float)scales[scale_pair + 4] *
                               (float)((int)((low[l] >> 4) | (((high[l] >> 4) & 3) << 4)) - 32);
                dest[l + 96] = d * (float)scales[scale_pair + 6] *
                               (float)((int)((low[l + 32] >> 4) | (((high[l] >> 6) & 3) << 4)) - 32);
            }
            low += 64;
            high += 32;
            scales += 8;
            dest += 128;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* PCHIE_WEIGHT_DEQUANTIZE_H */
