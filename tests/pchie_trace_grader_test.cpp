// Bolt PCHIE - Trace sim grader, the specified component rules only.
// Labeled unit-test fixtures; never a golden and never the official answer key
// (that key is not in the worktree). Curriculum L1 beyond Multiplexor/Dividor
// is not built here.
//
// Rulings in force:
//   - Components are deterministic like compiler keywords.
//   - Multiplexor: south=in1, west=in2, east=control (powered selects in2),
//     north=out.
//   - Dividor: halves rounding down, and 0 -> 0.
#include "pchie/Dividor.h"
#include "pchie/ITraceGrader.h"
#include "pchie/Multiplexor.h"
#include "pchie/SimTraceGrader.h"
#include "pchie/TraceGrade.h"
#include "pchie/TraceProbe.h"

#include <cstdio>
#include <string>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[TRACE-GRADER] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static void test_dividor_math() {
  int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0;
  check(Dividor::evaluate(0, a) && a == 0, "Dividor: 0 -> 0");
  check(Dividor::evaluate(1, b) && b == 0, "Dividor: 1 halves rounding down to 0");
  check(Dividor::evaluate(2, c) && c == 1, "Dividor: 2 -> 1");
  check(Dividor::evaluate(3, d) && d == 1, "Dividor: 3 halves rounding down to 1");
  check(Dividor::evaluate(5, e) && e == 2, "Dividor: 5 halves rounding down to 2");
  check(Dividor::evaluate(64, f) && f == 32, "Dividor: 64 -> 32");
  check(Dividor::evaluate(0, g) && Dividor::evaluate(0, h) && g == h,
        "Dividor: deterministic (compiler-keyword stable)");
}

static void test_multiplexor_rules() {
  check(Multiplexor::control_powered(0) == false, "Multiplexor: control 0 is not powered");
  check(Multiplexor::control_powered(1) == true, "Multiplexor: control > 0 is powered");
  check(Multiplexor::control_powered(64) == true, "Multiplexor: control 64 is powered");
  // unpowered: select in1 (south)
  int p = 0, q = 0, r = 0, s2 = 0, t1 = 0, t2 = 0;
  check(Multiplexor::evaluate(/*south*/7, /*west*/11, /*east*/0, p) && p == 7,
        "Multiplexor: unpowered east selects south (in1)");
  check(Multiplexor::evaluate(/*south*/7, /*west*/11, /*east*/1, q) && q == 11,
        "Multiplexor: powered east selects west (in2)");
  check(Multiplexor::evaluate(0, 64, 0, r) && r == 0, "Multiplexor: unpowered, in1=0");
  check(Multiplexor::evaluate(0, 64, 5, s2) && s2 == 64, "Multiplexor: powered, in2=64");
  check(Multiplexor::evaluate(3, 9, 1, t1) && Multiplexor::evaluate(3, 9, 1, t2) && t1 == t2,
        "Multiplexor: deterministic (compiler-keyword stable)");
}

static void test_grader_fixtures_and_negative() {
  SimTraceGrader g;
  TraceProbe div_ok;
  div_ok.component = "dividor";
  div_ok.south = 5;
  div_ok.expected_north = 2;
  const TraceGrade a = g.grade(div_ok);
  check(a.pass && a.observed == 2 && a.expected == 2, "grader: Dividor fixture 5 -> 2 PASSES");

  TraceProbe div_wrong;
  div_wrong.component = "dividor";
  div_wrong.south = 5;
  div_wrong.expected_north = 3;  // deliberately wrong expected (or wrong build)
  const TraceGrade b = g.grade(div_wrong);
  check(!b.pass && b.observed == 2 && b.expected == 3,
        "grader: deliberately wrong Dividor expected is FAIL (negative control)");

  TraceProbe mux_ok;
  mux_ok.component = "multiplexor";
  mux_ok.south = 7;
  mux_ok.west = 11;
  mux_ok.east = 1;
  mux_ok.expected_north = 11;
  const TraceGrade c = g.grade(mux_ok);
  check(c.pass && c.observed == 11, "grader: Multiplexor powered fixture selects in2 and PASSES");

  TraceProbe mux_wrong_build;
  mux_wrong_build.component = "dividor";  // wrong component for this I/O
  mux_wrong_build.south = 7;
  mux_wrong_build.west = 11;
  mux_wrong_build.east = 1;
  mux_wrong_build.expected_north = 11;
  const TraceGrade d = g.grade(mux_wrong_build);
  check(!d.pass && d.observed == 3,
        "grader: wrong component (Dividor used as a mux) is FAIL (negative control)");

  // Two runs, same fixtures, same verdicts (determinism).
  const TraceGrade a2 = g.grade(div_ok);
  check(a.pass == a2.pass && a.observed == a2.observed, "grader: fixture verdicts are deterministic");
}

int main() {
  std::printf("[TRACE-GRADER] Trace sim grader (specified Multiplexor/Dividor rules; unit fixtures only)\n");
  test_dividor_math();
  test_multiplexor_rules();
  test_grader_fixtures_and_negative();
  std::printf("[TRACE-GRADER] Trace grader %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
