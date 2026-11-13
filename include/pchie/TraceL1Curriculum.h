// Bolt PCHIE - L1 curriculum: Multiplexor + Dividor tasks.
// Concrete final. Each task gives port strengths and a goal output. Facing
// is north (locked Dividor / Multiplexor orientation). No mutex.
#pragma once

#include "pchie/TraceL1Task.h"

#include <vector>

namespace pchie {

class TraceL1Curriculum final {
 public:
  static std::vector<TraceL1Task> tasks() {
    // Distinguishing tasks: the other L1 component does not meet the goal.
    return {
        {"L1-D0", /*s*/0, /*w*/64, /*e*/1, /*goal*/0, TraceFacing::North},
        {"L1-D1", 1, 0, 0, 0, TraceFacing::North},
        {"L1-D2", 2, 0, 0, 1, TraceFacing::North},
        {"L1-D3", 3, 0, 0, 1, TraceFacing::North},
        {"L1-D5", 5, 0, 0, 2, TraceFacing::North},
        {"L1-D64", 64, 0, 0, 32, TraceFacing::North},
        {"L1-M-U", 7, 11, 0, 7, TraceFacing::North},
        {"L1-M-P", 7, 11, 1, 11, TraceFacing::North},
        {"L1-M-U0", 2, 64, 0, 2, TraceFacing::North},
        {"L1-M-P64", 0, 64, 5, 64, TraceFacing::North},
    };
  }
};

}  // namespace pchie
