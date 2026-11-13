// Bolt PCHIE - Trace grader runtime boundary (virtual).
// Plan: ITraceGrader / SimTraceGrader through IEngineHost. The concrete
// grader is SimTraceGrader. No official chain append at this slice.
#pragma once

#include "pchie/TraceGrade.h"
#include "pchie/TraceProbe.h"

namespace pchie {

class ITraceGrader {
 public:
  virtual ~ITraceGrader() = default;
  virtual TraceGrade grade(const TraceProbe& probe) const = 0;
};

}  // namespace pchie
