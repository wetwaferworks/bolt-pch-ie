// Bolt PCHIE - HtnState: a deterministic set of true fact ids.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// Facts are opaque uint32 ids chosen by the domain author. The state is built by
// trusted code (the grid / harness), never parsed from world text (B6 rule).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace pchie {

class HtnState final {
 public:
  HtnState() = default;
  explicit HtnState(std::vector<uint32_t> facts) : facts_(std::move(facts)) { normalize(); }

  bool has(uint32_t fact) const {
    return std::binary_search(facts_.begin(), facts_.end(), fact);
  }
  void set(uint32_t fact) {
    auto it = std::lower_bound(facts_.begin(), facts_.end(), fact);
    if (it == facts_.end() || *it != fact) facts_.insert(it, fact);
  }
  void unset(uint32_t fact) {
    auto it = std::lower_bound(facts_.begin(), facts_.end(), fact);
    if (it != facts_.end() && *it == fact) facts_.erase(it);
  }
  const std::vector<uint32_t>& facts() const noexcept { return facts_; }
  size_t size() const noexcept { return facts_.size(); }

  bool operator==(const HtnState& o) const { return facts_ == o.facts_; }
  bool operator!=(const HtnState& o) const { return !(*this == o); }

 private:
  void normalize() {
    std::sort(facts_.begin(), facts_.end());
    facts_.erase(std::unique(facts_.begin(), facts_.end()), facts_.end());
  }
  std::vector<uint32_t> facts_;  // sorted ascending, unique
};

}  // namespace pchie
