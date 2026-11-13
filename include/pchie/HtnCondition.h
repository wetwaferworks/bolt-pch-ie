// Bolt PCHIE - HtnCondition: a method precondition.
// Concrete final. No mutex. Conjunctive only: every `require` fact must be true
// and every `forbid` fact must be false. An empty condition always holds, so a
// method without a precondition behaves exactly as before the HTN and NCM extensions.
// A fact listed in both lists can never hold; add_method rejects that (fail closed).
#pragma once

#include "pchie/HtnState.h"

#include <cstdint>
#include <vector>

namespace pchie {

class HtnCondition final {
 public:
  HtnCondition() = default;
  HtnCondition(std::vector<uint32_t> require, std::vector<uint32_t> forbid)
      : require_(std::move(require)), forbid_(std::move(forbid)) {}

  bool holds(const HtnState& s) const {
    for (uint32_t f : require_)
      if (!s.has(f)) return false;
    for (uint32_t f : forbid_)
      if (s.has(f)) return false;
    return true;
  }
  bool empty() const noexcept { return require_.empty() && forbid_.empty(); }
  bool contradictory() const {
    for (uint32_t a : require_)
      for (uint32_t b : forbid_)
        if (a == b) return true;
    return false;
  }
  const std::vector<uint32_t>& require() const noexcept { return require_; }
  const std::vector<uint32_t>& forbid() const noexcept { return forbid_; }

  bool operator==(const HtnCondition& o) const { return require_ == o.require_ && forbid_ == o.forbid_; }
  bool operator!=(const HtnCondition& o) const { return !(*this == o); }

 private:
  std::vector<uint32_t> require_;
  std::vector<uint32_t> forbid_;
};

}  // namespace pchie
