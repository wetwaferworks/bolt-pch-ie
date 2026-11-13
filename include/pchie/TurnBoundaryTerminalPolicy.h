// Bolt PCHIE - hard stop at end-of-turn, EOS and PCH step-delimiter tokens (Archie ruling 2).
// Concrete, non-virtual (search hot path).
#pragma once
#include <cstdint>
#include <vector>

namespace pchie {

class TurnBoundaryTerminalPolicy final {
public:
    explicit TurnBoundaryTerminalPolicy(std::vector<int32_t> stop_tokens);   // negatives ignored
    bool IsTerminal(int32_t token) const;
    const std::vector<int32_t>& StopTokens() const noexcept { return stops_; }
private:
    std::vector<int32_t> stops_;   // sorted, unique
};

} // namespace pchie