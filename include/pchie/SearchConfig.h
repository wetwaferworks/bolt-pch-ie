// Bolt PCHIE - PUCT search constants (Archie ruling 3). Rationals keep the selection integer-exact.
#pragma once
#include <cstdint>

namespace pchie {

class SearchConfig final {
public:
    uint32_t c_puct_num = 3, c_puct_den = 2;     // c_puct = 1.5 (ruling: 1.5 to 2.0)
    uint32_t c_fpu_num = 11, c_fpu_den = 50;     // c_fpu = 0.22 (ruling: 0.20 to 0.25)
    float    prior_temp = 1.25f;                 // tau_prior, applied before storing P(s, a)
    int      top_k = 40;                         // deterministic top-K expansion by prior
    uint32_t root_noise_num = 0, root_noise_den = 4;   // Dirichlet(1) root noise eps = num/den; 0 = off (ruling 5)
    double   commit_temp = 0.0;                  // commit from N^(1/T); 0 = most visited (ruling 5)
};

} // namespace pchie