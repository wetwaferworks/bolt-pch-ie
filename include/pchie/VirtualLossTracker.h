// Bolt PCHIE - virtual loss (Archie ruling 4), integer Q20, non-virtual (search hot path).
// v_loss = 1.0 in normalized Q space, applied to both the visit count and the value sum:
//   Q_eff(s, a) = ( W(s, a) - v_loss * N_virt(s, a) ) / ( N(s, a) + N_virt(s, a) )
#pragma once
#include <cstdint>

namespace pchie {

class VirtualLossTracker final {
public:
    explicit VirtualLossTracker(int64_t weight_q20) : weight_q20_(weight_q20) {}
    int64_t EffectiveQ(int64_t w_q20, uint32_t n, uint32_t n_virt) const noexcept {
        const int64_t d = int64_t(n) + int64_t(n_virt);
        return d ? (w_q20 - weight_q20_ * int64_t(n_virt)) / d : 0;
    }
    int64_t WeightQ20() const noexcept { return weight_q20_; }
private:
    int64_t weight_q20_;
};

} // namespace pchie