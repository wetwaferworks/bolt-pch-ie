// Bolt PCHIE - HtnEffect: what a primitive task makes true / false.
// Concrete final. No mutex. Applied in order: delete, then add (so a fact in both
// lists ends up true). Effects only thread state through decomposition so later
// method preconditions see earlier primitives; they never execute anything.
#pragma once

#include "pchie/HtnState.h"

#include <cstdint>
#include <vector>

namespace pchie {

class HtnEffect final {
 public:
  HtnEffect() = default;
  HtnEffect(std::vector<uint32_t> add, std::vector<uint32_t> del)
      : add_(std::move(add)), del_(std::move(del)) {}

  void apply(HtnState& s) const {
    for (uint32_t f : del_) s.unset(f);
    for (uint32_t f : add_) s.set(f);
  }
  bool empty() const noexcept { return add_.empty() && del_.empty(); }
  const std::vector<uint32_t>& add() const noexcept { return add_; }
  const std::vector<uint32_t>& del() const noexcept { return del_; }

  bool operator==(const HtnEffect& o) const { return add_ == o.add_ && del_ == o.del_; }
  bool operator!=(const HtnEffect& o) const { return !(*this == o); }

 private:
  std::vector<uint32_t> add_;
  std::vector<uint32_t> del_;
};

}  // namespace pchie
