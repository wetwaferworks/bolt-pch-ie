// Bolt PCHIE - commit-time action choice (Archie ruling 5).
// T = 0: most-visited child, ties to the lowest token. T > 0: pick from N(s, a)^(1/T) with the
// node's seed, so the committed line is the same on 1 to 16 workers.
#pragma once
#include "pchie/pchie_mcts.h"
#include <cstdint>

namespace pchie {

class CommitSelector final {
public:
    explicit CommitSelector(double temperature) : temp_(temperature) {}
    int Select(const PuctSearch& s, uint32_t node) const;   // edge index, -1 if no visited child
private:
    double temp_;
};

} // namespace pchie