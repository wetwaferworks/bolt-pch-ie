// Bolt PCHIE - log-softmax statistics over one full logit vector (temperature 1).
// Sums run in index order in double, exp from exp_deterministic: bit-identical on every run.
#pragma once
#include <cstdint>
#include <vector>

namespace pchie {

class LogitStats {
public:
    explicit LogitStats(const std::vector<float>& logits);
    double LogProb(int32_t token) const;   // ln P(token)
    double Entropy() const noexcept { return entropy_; }   // nats
private:
    const std::vector<float>& logits_;
    float  max_ = 0.0f;
    double log_z_ = 0.0, entropy_ = 0.0;
};

} // namespace pchie