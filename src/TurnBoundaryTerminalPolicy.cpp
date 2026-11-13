#include "pchie/TurnBoundaryTerminalPolicy.h"
#include <algorithm>

namespace pchie {

TurnBoundaryTerminalPolicy::TurnBoundaryTerminalPolicy(std::vector<int32_t> s) {
    for (int32_t t : s) if (t >= 0) stops_.push_back(t);
    std::sort(stops_.begin(), stops_.end());
    stops_.erase(std::unique(stops_.begin(), stops_.end()), stops_.end());
}

bool TurnBoundaryTerminalPolicy::IsTerminal(int32_t token) const {
    return std::binary_search(stops_.begin(), stops_.end(), token);
}

} // namespace pchie