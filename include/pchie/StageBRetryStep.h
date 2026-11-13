// Bolt PCHIE failed-actuation retry - one Stage B door try under the opt-in retry path.
// source: "neural" (top-1, no trace), "injected" (top-1 after failure-trace injection),
// "tabu_next" (top-1 masked as a repeat; next-best neural candidate), "symbolic" (systematic
// catalog probe), "exhausted" (every candidate masked on this world; tabu cleared, no attempt).
#pragma once

#include <cstdint>

namespace pchie {

struct StageBRetryStep {
  int door = -1;
  int attempt = 0;  // 1-based per door; 0 for an "exhausted" marker
  const char* source = "";
  const char* neural_top = "";  // neural top-1 before masking ("" if none: freeze / no-op)
  const char* proposed = "";
  bool opened = false;
  const char* reason = "";
  bool m7_trigger = false;  // top-1 would have repeated a failed proposal on this world
  bool noop = false;        // no usable proposal (freeze / empty)
  int trace_blocks = 0;     // failure records injected into this pass
  int trace_tokens = 0;
  uint64_t world_fnv = 0;   // canonical world signature at decision time
};

}  // namespace pchie