// Bolt PCHIE - the ternary fit of one 128-weight group, shared by the CPU converter and the CUDA kernel (clean-room).
//
// The same source is compiled by the host compiler and by nvcc (-fmad=false, IEEE division), and it uses only
// IEEE round-to-nearest float and double + - * /, fabs, conversions and comparisons. Nothing here can contract
// into a fused multiply-add, so both sides give the same bits for the same input.
//
// The math is the research path's ternarize_group (pchie_ternary_research.h) with half-to-even ties:
//   Absmean       gamma = mean |w| over the group (double, sequential order), stored as float. This is also
//                 exactly what quantize_absmean (pchie_ternary.h) does per 128-weight block.
//   LeastSquares  sort |w| from largest to smallest; P_k is the sum of the k largest (double, sequential).
//                 gamma = P_k / k for the first k that maximizes P_k^2 / k. For a fixed set of non-zeros the
//                 best scale is their mean |w|, and the k largest |w| are the best set of size k, so this is the
//                 exact least-squares ternary fit of the group.
// Then a gamma below FLT_MIN (zero or subnormal) becomes an exact zero group, and each code is
//   q = +1 if w / gamma > 0.5, -1 if w / gamma < -0.5, else 0 (ties go to 0),
// against the stored float gamma, so the codes always agree with the scale the engine will use.
#pragma once
#include "pchie/pchie_det_math.h"
#include <cstdint>

namespace pchie {
namespace ternary {
namespace convert {

constexpr int kGroup = 128;   // weights per group; the same as kBlock in pchie_ternary.h (PerBlock scales)
enum class FitKind : uint8_t { Absmean = 0, LeastSquares = 1 };

// |w| sorted from largest to smallest (insertion sort). Any correct sort gives the same sequence of values, so
// the CUDA kernel may sort another way (it uses a warp-wide bitonic sort) and still match this bit for bit.
PCHIE_HD void sort_abs_desc(const float* w, float* sorted) {
    for (int i = 0; i < kGroup; ++i) {
        const float a = fabsf(w[i]);
        int j = i;
        while (j > 0 && sorted[j - 1] < a) { sorted[j] = sorted[j - 1]; --j; }
        sorted[j] = a;
    }
}

// The fit once |w| is sorted (sorted is only read by LeastSquares; Absmean may pass nullptr).
// w: kGroup finite floats (the caller checks). Writes kGroup codes to q and returns gamma.
PCHIE_HD float fit_group_sorted(const float* w, FitKind fit, const float* sorted, int8_t* q) {
    double g = 0.0;
    if (fit == FitKind::LeastSquares) {
        double p = 0.0, best = 0.0, bp = 0.0;
        int bk = 0;
        for (int k = 1; k <= kGroup; ++k) {
            p += double(sorted[k - 1]);
            const double v = p * p / double(k);
            if (v > best) { best = v; bk = k; bp = p; }
        }
        g = bk ? bp / double(bk) : 0.0;
    } else {
        double s = 0.0;
        for (int i = 0; i < kGroup; ++i) s += fabs(double(w[i]));
        g = s / double(kGroup);
    }
    float gf = float(g);
    if (!(gf >= det_f32_from_bits(0x00800000u))) gf = 0.0f;   // below FLT_MIN: an exact zero group
    // w / gamma > 0.5 is tested as w > gamma / 2, both in double: halving a float is exact in double, and the two
    // tests agree for every pair of floats (the rounded quotient of two floats can't land on 0.5 from above, since
    // a float w != gamma / 2 differs from it by far more than half a double ulp). This skips 128 double divisions.
    const double h = 0.5 * double(gf);
    for (int i = 0; i < kGroup; ++i) {
        const double x = double(w[i]);
        q[i] = gf == 0.0f ? int8_t(0) : (x > h ? int8_t(1) : (x < -h ? int8_t(-1) : int8_t(0)));
    }
    return gf;
}

// The whole fit of one group. sorted: kGroup floats of scratch, used by LeastSquares only.
PCHIE_HD float fit_group(const float* w, FitKind fit, float* sorted, int8_t* q) {
    if (fit == FitKind::LeastSquares) sort_abs_desc(w, sorted);
    return fit_group_sorted(w, fit, sorted, q);
}

} // namespace convert
} // namespace ternary
} // namespace pchie
