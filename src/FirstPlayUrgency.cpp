#include "pchie/FirstPlayUrgency.h"

namespace pchie {

uint64_t FirstPlayUrgency::Isqrt(uint64_t x) noexcept {
    uint64_t r = 0, b = 1ULL << 62;
    while (b > x) b >>= 2;
    while (b) { if (x >= r + b) { x -= r + b; r = (r >> 1) + b; } else r >>= 1; b >>= 2; }
    return r;
}

int64_t FirstPlayUrgency::Value(int64_t v, uint64_t p_q20) const noexcept {
    const uint64_t sq_q20 = Isqrt(p_q20 << 20);                     // sqrt(P) in Q20
    return v - int64_t((uint64_t(num_) * sq_q20) / den_);
}

} // namespace pchie