// Bolt PCHIE - headless Trace sim grader.
// Concrete final. Deterministic, no mutex. Grades a probe or a placed
// circuit against a goal / draft L1 answer key. Out-of-range strengths
// and unknown components FAIL (rejected, not clamped). No official chain.
#pragma once

#include "pchie/Dividor.h"
#include "pchie/ITraceGrader.h"
#include "pchie/Multiplexor.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceL1AnswerKey.h"
#include "pchie/TraceL1Task.h"

#include <cstring>

namespace pchie {

class SimTraceGrader final : public ITraceGrader {
 public:
  TraceGrade grade(const TraceProbe& probe) const override {
    TraceGrade out;
    out.expected = probe.expected_north;
    if (!probe.component) {
      out.reason = "missing component";
      return out;
    }
    int observed = 0;
    if (std::strcmp(probe.component, Dividor::kName) == 0) {
      if (!Dividor::evaluate(probe.south, observed)) {
        out.reason = "invalid strength";
        return out;
      }
    } else if (std::strcmp(probe.component, Multiplexor::kName) == 0) {
      if (!Multiplexor::evaluate(probe.south, probe.west, probe.east, observed)) {
        out.reason = "invalid strength";
        return out;
      }
    } else {
      out.reason = "unknown component";
      return out;
    }
    out.observed = observed;
    out.pass = (out.observed == out.expected);
    out.reason = out.pass ? "ok" : "mismatch";
    return out;
  }

  TraceGrade grade_circuit(const TraceCircuit& c, int goal) const {
    TraceGrade out;
    out.expected = goal;
    if (!c.component) {
      out.reason = "missing component";
      return out;
    }
    int observed = 0;
    if (std::strcmp(c.component, Dividor::kName) == 0) {
      if (!Dividor::evaluate_facing(c.facing, c.south, c.west, c.east, c.north, observed)) {
        out.reason = "invalid strength";
        return out;
      }
    } else if (std::strcmp(c.component, Multiplexor::kName) == 0) {
      // L1 pinout is specified for facing north only.
      if (c.facing != TraceFacing::North) {
        out.reason = "multiplexor facing not north";
        return out;
      }
      if (!Multiplexor::evaluate(c.south, c.west, c.east, observed)) {
        out.reason = "invalid strength";
        return out;
      }
    } else {
      out.reason = "unknown component";
      return out;
    }
    out.observed = observed;
    out.pass = (out.observed == out.expected);
    out.reason = out.pass ? "ok" : "mismatch";
    return out;
  }

  TraceGrade grade_against_key(const TraceCircuit& c, const TraceL1Task& task) const {
    const auto* key = TraceL1AnswerKey::for_task(task.id);
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
      // Still report the simulated output when it is computable.
      const TraceGrade sim = grade_circuit(c, task.goal);
      out.observed = sim.observed;
      return out;
    }
    return grade_circuit(c, task.goal);
  }
};

}  // namespace pchie
