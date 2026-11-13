// Bolt PCHIE - one placed headless circuit (a single L1 component).
// Port fields are world-face strengths. L1 tasks face north.
#pragma once

#include "pchie/TraceFacing.h"

namespace pchie {

struct TraceCircuit {
  const char* component = "";
  TraceFacing facing = TraceFacing::North;
  int south = 0;
  int west = 0;
  int east = 0;
  int north = 0;
};

}  // namespace pchie
