// Bolt PCHIE - deterministic Dirichlet(1) root exploration noise (Archie ruling 5).
// noise_a = -ln(u_a) / sum_b -ln(u_b),  u_a = Uniform(splitmix64(root_seed ^ token_a)).
// P'(a) = (1 - eps) P(a) + eps noise_a, eps = num / den. eps = 0 (default) leaves priors untouched.
#pragma once
#include "pchie/pchie_mcts.h"
#include <cstdint>
#include <vector>

namespace pchie {

class RootNoise final {
public:
    RootNoise(uint32_t num, uint32_t den) : num_(num), den_(den) {}
    void Apply(std::vector<PuctEdge>& edges, uint64_t root_seed) const;
private:
    uint32_t num_, den_;
};

} // namespace pchie