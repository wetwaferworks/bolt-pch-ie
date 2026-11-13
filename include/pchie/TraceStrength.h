// Bolt PCHIE - signal strength domain (0..64; negatives can't occur).
// Concrete final. Out-of-range values are rejected, not clamped.
#pragma once

namespace pchie {

class TraceStrength final {
 public:
  static constexpr int kMin = 0;
  static constexpr int kMax = 64;
  static bool ok(int s) { return s >= kMin && s <= kMax; }
};

}  // namespace pchie
