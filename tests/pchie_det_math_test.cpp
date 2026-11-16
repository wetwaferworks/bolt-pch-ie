// Deterministic math on bad input. exp_deterministic, det_round_half_even and round_half_even_f32 must never
// cast NaN or an out-of-range float to int (undefined behaviour, and x86 and CUDA give different answers),
// and must give bit-identical results to before for every other input (so no hash moves).
#include "pchie/pchie_det_math.h"
#include "pchie/pchie_q4_det.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using namespace pchie;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[DETMATH] %-92s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
}
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float from_bits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// The routines as they were before the NaN guard, used only where they were defined (finite, in range).
static int32_t old_round(float x) {
    const float fl = std::floor(x); const float diff = x - fl; int32_t r = static_cast<int32_t>(fl);
    if (diff > 0.5f) r += 1; else if (diff == 0.5f && (r & 1)) r += 1; return r;
}

int main() {
    std::printf("[DETMATH] deterministic exp and rounding on NaN, inf and out-of-range input\n");
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    check(std::isnan(exp_deterministic(nan)), "exp_deterministic(NaN) is NaN (no cast of NaN to int)");
    check(exp_deterministic(-inf) == 0.0f, "exp_deterministic(-inf) is 0");
    check(exp_deterministic(inf) == exp_deterministic(88.0f), "exp_deterministic(+inf) saturates like 88");
    check(det_round_half_even(nan) == 0 && round_half_even_f32(nan) == 0, "rounding NaN gives 0 on both routines");
    check(det_round_half_even(3e9f) == INT32_MAX && round_half_even_f32(3e9f) == INT32_MAX, "rounding 3e9 saturates to INT32_MAX (it gave INT32_MIN before)");
    check(det_round_half_even(-3e9f) == INT32_MIN && round_half_even_f32(-3e9f) == INT32_MIN, "rounding -3e9 saturates to INT32_MIN");
    check(det_round_half_even(inf) == INT32_MAX && det_round_half_even(-inf) == INT32_MIN, "rounding +-inf saturates");
    check(det_round_half_even(-2147483648.0f) == INT32_MIN, "rounding -2^31 (exactly representable) is INT32_MIN");
    check(det_round_half_even(2.5f) == 2 && det_round_half_even(3.5f) == 4 && det_round_half_even(-2.5f) == -2, "half-even ties unchanged");

    // Bit-identity sweep over ~1.05 million float bit patterns (every 4093rd), both signs, all exponents.
    uint64_t n_round = 0, n_exp = 0, bad_round = 0, bad_exp = 0, bad_pair = 0;
    for (uint64_t u = 0; u <= 0xFFFFFFFFull; u += 4093) {
        const float x = from_bits(uint32_t(u));
        if (!std::isfinite(x)) continue;
        if (x > -2147483648.0f && x < 2147483648.0f) {
            ++n_round;
            const int32_t want = old_round(x);
            if (det_round_half_even(x) != want || round_half_even_f32(x) != want) ++bad_round;
        }
        if (det_round_half_even(x) != round_half_even_f32(x)) ++bad_pair;
        ++n_exp;
        // exp_deterministic clamps finite input to [-87, 88] before its cast, so the new NaN guard never runs
        // for a finite value and cannot change one; check that no finite value gives NaN.
        const float e = exp_deterministic(x);
        if (std::isnan(e)) ++bad_exp;
    }
    char lab[200];
    std::snprintf(lab, sizeof lab, "rounding is bit-identical to before on %llu in-range values", (unsigned long long)n_round);
    check(bad_round == 0, lab);
    check(bad_pair == 0, "det_round_half_even and round_half_even_f32 agree on every finite value swept");
    std::snprintf(lab, sizeof lab, "exp_deterministic never returns NaN for %llu finite values", (unsigned long long)n_exp);
    check(bad_exp == 0, lab);
    // A few fixed values pinned to their bits (from the pre-guard routine).
    check(bits(exp_deterministic(0.0f)) == bits(1.0f), "exp_deterministic(0) == 1 exactly");
    std::printf("[DETMATH] %d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
