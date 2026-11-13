// Bolt PCHIE - placeholder PCH gate (G = 1 everywhere) until the PCH gate rules are defined.
// Concrete, non-virtual (search hot path).
#pragma once
#include <cstdint>
#include <vector>

namespace pchie {

class AcceptAllGate final {
public:
    bool Pass(const std::vector<int32_t>&) const noexcept { return true; }
};

} // namespace pchie