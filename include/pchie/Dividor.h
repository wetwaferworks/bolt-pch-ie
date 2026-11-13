// Bolt PCHIE - Dividor component.
// Concrete final. Deterministic like a compiler keyword. No mutex.
// Rule: halves rounding down, and 0 -> 0.
// Locked pinout: halves BACK, outputs FRONT. Facing north =>
// in=south, out=north. Strengths 0..64; out-of-range is rejected.
#pragma once

#include "pchie/TraceFacing.h"
#include "pchie/TraceStrength.h"

namespace pchie {

class Dividor final {
 public:
  static constexpr const char* kName = "dividor";

  static TraceFacing front_face(TraceFacing facing) { return facing; }
  static TraceFacing back_face(TraceFacing facing) { return trace_opposite(facing); }

  static bool evaluate(int input, int& out) {
    if (!TraceStrength::ok(input)) return false;
    out = input / 2;  // 0 -> 0; positive integers round toward zero == down
    return true;
  }

  static int face_strength(TraceFacing face, int south, int west, int east, int north) {
    switch (face) {
      case TraceFacing::North: return north;
      case TraceFacing::South: return south;
      case TraceFacing::East: return east;
      case TraceFacing::West: return west;
    }
    return 0;
  }

  static bool evaluate_facing(TraceFacing facing, int south, int west, int east, int north,
                              int& front_out) {
    const int back = face_strength(back_face(facing), south, west, east, north);
    return evaluate(back, front_out);
  }
};

}  // namespace pchie
