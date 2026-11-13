// Bolt PCHIE - deterministic host math built only from + - * / and bit operations (clean-room):
// natural log (atanh series), sigmoid and pow on top of exp_deterministic. No std or CUDA
// transcendental is ever called. Bit outputs are pinned by tests/pchie_search_rules_test.cpp.
#pragma once
#include "pchie/pchie_det_math.h"
#include <cstdint>
#include <cstring>

namespace pchie {

class DetLog {
public:
    static double Ln(double x) noexcept {
        if (!(x > 0.0)) return -1.0e300;
        uint64_t u; std::memcpy(&u, &x, 8);
        int e = int((u >> 52) & 0x7FF);
        if (e == 0) { x = x * 18014398509481984.0; std::memcpy(&u, &x, 8); e = int((u >> 52) & 0x7FF) - 54; }  // subnormal: * 2^54
        e -= 1023;
        u = (u & 0x000FFFFFFFFFFFFFULL) | 0x3FF0000000000000ULL;       // m in [1, 2)
        double m; std::memcpy(&m, &u, 8);
        if (m > 1.41421356237309504880) { m = m * 0.5; e += 1; }
        const double s = (m - 1.0) / (m + 1.0), s2 = s * s;
        double p = 1.0 / 17.0;
        p = 1.0 / 15.0 + s2 * p; p = 1.0 / 13.0 + s2 * p; p = 1.0 / 11.0 + s2 * p;
        p = 1.0 / 9.0 + s2 * p;  p = 1.0 / 7.0 + s2 * p;  p = 1.0 / 5.0 + s2 * p;
        p = 1.0 / 3.0 + s2 * p;  p = 1.0 + s2 * p;
        return double(e) * 0.69314718055994530942 + 2.0 * s * p;
    }
    static double Sigmoid(double x) noexcept { return 1.0 / (1.0 + double(exp_deterministic(float(-x)))); }
    static double Pow(double base, double y) noexcept { return double(exp_deterministic(float(y * Ln(base)))); }  // base > 0
};

} // namespace pchie