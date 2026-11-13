// Bolt PCHIE - Multiplexor component.
// Concrete final. Deterministic like a compiler keyword. No mutex.
// Rule: south=in1, west=in2, east=control, north=out.
// Q42: "powered" means any source powering the east side (soft, hard,
// repeater, Thresholdor, etc.). East strength > 0 selects west; 0 selects
// south. Strengths 0..64; out-of-range is rejected.
#pragma once

#include "pchie/TraceStrength.h"

namespace pchie {

class Multiplexor final {
 public:
  static constexpr const char* kName = "multiplexor";

  static bool control_powered(int east) { return east > 0; }

  static bool evaluate(int south_in1, int west_in2, int east_control, int& out) {
    if (!TraceStrength::ok(south_in1) || !TraceStrength::ok(west_in2) ||
        !TraceStrength::ok(east_control))
      return false;
    out = control_powered(east_control) ? west_in2 : south_in1;
    return true;
  }
};

}  // namespace pchie
