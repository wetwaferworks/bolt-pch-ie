// Bolt PCHIE - parent-relative first-play urgency (Archie ruling 3), integer Q20, non-virtual:
//   FPU(s) = V(s) - c_fpu * sqrt( sum over visited children b of P(s, b) )
#pragma once
#include <cstdint>

namespace pchie {

class FirstPlayUrgency final {
public:
    FirstPlayUrgency(uint32_t num, uint32_t den) : num_(num), den_(den) {}
    // parent_value_q20: V(s); visited_prior_q20: sum of P(s, b) over visited children.
    int64_t Value(int64_t parent_value_q20, uint64_t visited_prior_q20) const noexcept;
    static uint64_t Isqrt(uint64_t x) noexcept;
private:
    uint32_t num_, den_;
};

} // namespace pchie