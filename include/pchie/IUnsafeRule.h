// Bolt PCHIE Mirror-Sim gate - pluggable deterministic unsafe-rule hook.
// New rules are not registered by default; the maintainer judges them at first.
#pragma once

#include "pchie/MazeGrid.h"

namespace pchie {

class IUnsafeRule {
 public:
  virtual ~IUnsafeRule() = default;
  virtual const char* name() const = 0;
  virtual bool is_unsafe(const MazeGrid& pre, MazeAction action, const MazeGrid& post) const = 0;
};

}  // namespace pchie
