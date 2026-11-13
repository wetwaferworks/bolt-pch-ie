// Bolt PCHIE - what a leaf evaluator sees (Archie ruling 1).
#pragma once
#include <cstddef>
#include <cstdint>

namespace pchie {

class LeafContext {
public:
    uint32_t depth = 0;                   // L: rollout depth from the search root
    double   sum_log_prob = 0.0;          // sum over the path of ln P(y_t | y_<t)
    double   entropy = 0.0;               // H(s_L): next-token entropy at the leaf (0 at a terminal)
    bool     gate_pass = true;            // G(s_L) from the PCH gate
    const float* hidden_state = nullptr;  // final-layer hidden state (for a later value head); may be null
    size_t   hidden_size = 0;
};

} // namespace pchie