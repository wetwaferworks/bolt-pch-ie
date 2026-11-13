// Bolt PCHIE Mirror-Sim gate - default unsafe rule: stepping into a pit is unsafe.
// Concrete final. Deterministic, no mutex. approved default.
#pragma once

#include "pchie/IUnsafeRule.h"

namespace pchie {

class PitTrapRule final : public IUnsafeRule {
 public:
  static constexpr const char* kName = "pit_trap";

  const char* name() const override { return kName; }

  bool is_unsafe(const MazeGrid& pre, MazeAction, const MazeGrid& post) const override {
    return !pre.in_pit() && post.in_pit();
  }
};

}  // namespace pchie
