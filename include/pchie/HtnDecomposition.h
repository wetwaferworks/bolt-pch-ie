// Bolt PCHIE - result of a state-aware HTN decomposition.
// Plain result record (one per file). No mutex.
#pragma once

#include "pchie/HtnState.h"
#include "pchie/HtnTask.h"

#include <cstdint>
#include <vector>

namespace pchie {

struct HtnDecomposition {
  std::vector<HtnTask> tasks;      // primitives in execution order
  std::vector<uint32_t> methods;   // method id chosen at each compound expansion (DFS order)
  HtnState end_state;              // start state with every primitive effect applied
  size_t expansions = 0;           // method attempts, including ones backtracked out of
  size_t backtracks = 0;           // method attempts that failed and were undone
};

}  // namespace pchie
