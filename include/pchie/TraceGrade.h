// Bolt PCHIE - one grader verdict.
#pragma once

namespace pchie {

struct TraceGrade {
  bool pass = false;
  int expected = 0;
  int observed = 0;
  const char* reason = "";
};

}  // namespace pchie
