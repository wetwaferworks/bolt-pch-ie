// Bolt PCHIE - one L1 curriculum task: given port strengths and a goal output.
#pragma once

#include "pchie/TraceFacing.h"

namespace pchie {

struct TraceL1Task {
  const char* id = "";
  int south = 0;
  int west = 0;
  int east = 0;
  int goal = 0;
  TraceFacing facing = TraceFacing::North;
};

}  // namespace pchie
