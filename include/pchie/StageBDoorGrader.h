// Bolt PCHIE Stage B - door grader. Concrete final. Deterministic, no mutex.
// Same rules as SimTraceGrader::grade_against_key, but against the Stage B
// door key (opaque ids). Simulation is SimTraceGrader::grade_circuit, so
// out-of-range strengths and unknown components FAIL (rejected, not
// clamped). No official chain.
#pragma once

#include "pchie/SimTraceGrader.h"
#include "pchie/StageBDoorKey.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceGrade.h"
#include "pchie/TraceL1Task.h"

#include <cstring>

namespace pchie {

class StageBDoorGrader final {
 public:
  TraceGrade grade(const TraceCircuit& c, const TraceL1Task& task) const {
    const StageBDoorAnswer* key = StageBDoorKey::for_task(task.id);
    if (!key) {
      TraceGrade out;
      out.expected = task.goal;
      out.reason = "unknown task";
      return out;
    }
    if (!c.component || std::strcmp(c.component, key->component) != 0 || c.facing != key->facing) {
      TraceGrade out;
      out.expected = task.goal;
      out.reason = "wrong component";
      if (c.component) out.observed = SimTraceGrader().grade_circuit(c, task.goal).observed;
      return out;
    }
    TraceGrade g = SimTraceGrader().grade_circuit(c, task.goal);
    if (g.pass && g.observed != key->expected_output) {
      g.pass = false;
      g.reason = "key mismatch";
    }
    return g;
  }
};

}  // namespace pchie