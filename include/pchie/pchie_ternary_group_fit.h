/* Bolt PCHIE - the ternary fit of one 128-weight group, shared by the CPU converter and the CUDA kernel
 * (clean-room). Written in C so both sides compile the same source.
 *
 * The same source is compiled by the host C compiler and by nvcc (-fmad=false, IEEE division), and it uses only
 * IEEE round-to-nearest float and double + - * /, fabs, conversions and comparisons. Nothing here can contract
 * into a fused multiply-add, so both sides give the same bits for the same input.
 *
 * The math is the research path's ternarize_group (pchie_ternary_research.h) with half-to-even ties:
 *   Absmean         gamma = mean |w| over the group (double, sequential order), stored as float. This is also
 *                   exactly what quantize_absmean (pchie_ternary.h) does per 128-weight block.
 *   Least squares   sort |w| from largest to smallest; P_k is the sum of the k largest (double, sequential).
 *                   gamma = P_k / k for the first k that maximizes P_k^2 / k. For a fixed set of non-zeros the
 *                   best scale is their mean |w|, and the k largest |w| are the best set of size k, so this is the
 *                   exact least-squares ternary fit of the group.
 * Then a gamma below FLT_MIN (zero or subnormal) becomes an exact zero group, and each code is
 *   q = +1 if w / gamma > 0.5, -1 if w / gamma < -0.5, else 0 (ties go to 0),
 * against the stored float gamma, so the codes always agree with the scale the engine will use.
 *
 * The method is Microsoft's BitNet research (absmean ternarization in the b1.58 papers). This file was written
 * from those papers and from the research path above, not from Microsoft's or anyone else's source code.
 */
#ifndef PCHIE_TERNARY_GROUP_FIT_H
#define PCHIE_TERNARY_GROUP_FIT_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__CUDACC__)
#define TERNARY_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define TERNARY_HOST_DEVICE static inline
#endif

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__CUDA_ARCH__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { TERNARY_GROUP_WEIGHT_COUNT = 128 }; /* weights per group; the same as kBlock in pchie_ternary.h */

typedef enum ternary_group_fit_kind {
    TERNARY_GROUP_FIT_ABSMEAN = 0,
    TERNARY_GROUP_FIT_LEAST_SQUARES = 1
} ternary_group_fit_kind;

TERNARY_HOST_DEVICE float ternary_float_from_bits(uint32_t bits) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value;
    memcpy(&value, &bits, 4);
    return value;
#endif
}

/* |w| sorted from largest to smallest. Any correct sort gives the same sequence of values, so the CUDA kernel
 * may sort another way (it uses a warp-wide bitonic sort) and still match this bit for bit. On the CPU this is
 * an 8-bit radix sort of the IEEE bit patterns of the non-negative finite absolutes, which is the same order
 * as a comparison sort. */
TERNARY_HOST_DEVICE void ternary_sort_absolute_values_descending(const float *weights, float *sorted_absolute_values) {
    uint32_t keys[TERNARY_GROUP_WEIGHT_COUNT];
    uint32_t scratch[TERNARY_GROUP_WEIGHT_COUNT];
    int i, pass;
    for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
        const float absolute = fabsf(weights[i]);
        uint32_t bits;
        memcpy(&bits, &absolute, 4);
        keys[i] = bits;
    }
    /* Four least-significant-digit passes of 8 bits. Histogram prefix is filled from the high bucket so each
     * pass is descending, and the sort is stable, so the final order is descending bit patterns. */
    for (pass = 0; pass < 4; ++pass) {
        int histogram[256];
        int total;
        int bucket;
        const int shift = pass * 8;
        for (bucket = 0; bucket < 256; ++bucket) histogram[bucket] = 0;
        for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i)
            histogram[(keys[i] >> shift) & 255] += 1;
        total = 0;
        for (bucket = 255; bucket >= 0; --bucket) {
            const int count = histogram[bucket];
            histogram[bucket] = total;
            total += count;
        }
        for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
            const uint32_t key = keys[i];
            const int dest = histogram[(key >> shift) & 255];
            scratch[dest] = key;
            histogram[(key >> shift) & 255] = dest + 1;
        }
        for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) keys[i] = scratch[i];
    }
    for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i)
        memcpy(&sorted_absolute_values[i], &keys[i], 4);
}

/* The fit once |w| is sorted (sorted_absolute_values is only read by least squares; absmean may pass NULL).
 * weights: TERNARY_GROUP_WEIGHT_COUNT finite floats (the caller checks). Writes that many codes and returns gamma. */
TERNARY_HOST_DEVICE float ternary_group_fit_from_sorted(const float *weights, ternary_group_fit_kind fit_kind,
                                                        const float *sorted_absolute_values, int8_t *codes) {
    double gamma_double = 0.0;
    float gamma;
    double half_gamma;
    int i;
    if (fit_kind == TERNARY_GROUP_FIT_LEAST_SQUARES) {
        double prefix = 0.0, best_score = 0.0, best_prefix = 0.0;
        int best_k = 0;
        for (i = 1; i <= TERNARY_GROUP_WEIGHT_COUNT; ++i) {
            prefix += (double)sorted_absolute_values[i - 1];
            {
                const double score = prefix * prefix / (double)i;
                if (score > best_score) {
                    best_score = score;
                    best_k = i;
                    best_prefix = prefix;
                }
            }
        }
        gamma_double = best_k ? best_prefix / (double)best_k : 0.0;
    } else {
        double sum = 0.0;
        for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) sum += fabs((double)weights[i]);
        gamma_double = sum / (double)TERNARY_GROUP_WEIGHT_COUNT;
    }
    gamma = (float)gamma_double;
    if (!(gamma >= ternary_float_from_bits(0x00800000u))) gamma = 0.0f; /* below FLT_MIN: an exact zero group */
    /* w / gamma > 0.5 is tested as w > gamma / 2, both in double: halving a float is exact in double, and the two
     * tests agree for every pair of floats (the rounded quotient of two floats can't land on 0.5 from above, since
     * a float w != gamma / 2 differs from it by far more than half a double ulp). This skips 128 double divisions. */
    half_gamma = 0.5 * (double)gamma;
    for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
        const double x = (double)weights[i];
        /* Branchless codes: (x > half) - (x < -half) is +1, -1 or 0. A zero gamma must still force a zero group. */
        const int plus = x > half_gamma;
        const int minus = x < -half_gamma;
        codes[i] = (gamma == 0.0f) ? (int8_t)0 : (int8_t)(plus - minus);
    }
    return gamma;
}

/* The whole fit of one group. sorted_absolute_values_scratch: TERNARY_GROUP_WEIGHT_COUNT floats, used by least
 * squares only. */
TERNARY_HOST_DEVICE float ternary_group_fit(const float *weights, ternary_group_fit_kind fit_kind,
                                            float *sorted_absolute_values_scratch, int8_t *codes) {
    if (fit_kind == TERNARY_GROUP_FIT_LEAST_SQUARES)
        ternary_sort_absolute_values_descending(weights, sorted_absolute_values_scratch);
    return ternary_group_fit_from_sorted(weights, fit_kind, sorted_absolute_values_scratch, codes);
}

#ifdef __cplusplus
}
#endif

#endif /* PCHIE_TERNARY_GROUP_FIT_H */
