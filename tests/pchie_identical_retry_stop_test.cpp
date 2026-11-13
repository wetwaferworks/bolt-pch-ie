// Bolt PCHIE - opt-in early stop on identical retries, default OFF.
// LABELED UNIT-TEST FIXTURE ONLY: Stage B labeled grid, scripted proposers, no model, null
// official chain. Never a golden or baseline. Official runners leave identical-retry stop off.
#include "pchie/Dividor.h"
#include "pchie/IdenticalRetryStop.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/Multiplexor.h"
#include "pchie/StageBDoorFactory.h"
#include "pchie/StageBDoorGrid.h"
#include "pchie/StageBDoorMaze.h"
#include "pchie/StageBDoorOracle.h"
#include "pchie/TraceL1Task.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[RETRY-STOP] %-77s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static const char* propose_right(void*, int door, const TraceL1Task&) {
  return door == 0 ? Dividor::kName : Multiplexor::kName;
}
static const char* propose_always_dividor(void*, int, const TraceL1Task&) { return Dividor::kName; }
// Door 1 wrong answers that alternate between two different wrong outcomes: dividor (graded
// wrong) and no proposal (nullptr). Never two identical attempts in a row.
static const char* propose_alternating_wrong(void* ctx, int door, const TraceL1Task&) {
  if (door == 0) return Dividor::kName;
  int& k = *static_cast<int*>(ctx);
  return (k++ % 2 == 0) ? Dividor::kName : nullptr;
}

static void report(const char* tag, bool ok, const StageBDoorResult& r) {
  std::printf("[RETRY-STOP] %s ok=%d win=%d door0=%d door1=%d attempts=%d/%d stalled_door=%d early=%d path=%d steps=%d cap=%d err=%s\n",
              tag, int(ok), int(r.win), int(r.door_open[0]), int(r.door_open[1]), r.attempts_per_door[0],
              r.attempts_per_door[1], r.stalled_door, int(r.early_stopped), r.path_length, r.steps_used,
              r.cap, r.err.c_str());
}

static void test_rule() {
  IdenticalRetryStop s;
  check(s.limit() == 0 && !s.enabled(), "default limit is 0 (OFF)");
  bool fired = false;
  for (int i = 0; i < 1000; ++i) fired = fired || s.observe("same");
  check(!fired && s.run_length() == 0, "OFF: 1000 identical observations never fire, no state kept");
  check(!s.set_limit(1) && !s.set_limit(-3) && s.limit() == 0, "limits 1 and negative are rejected, unchanged");
  check(s.set_limit(3) && s.enabled(), "limit 3 accepted");
  check(!s.observe("a") && !s.observe("a") && s.observe("a"), "limit 3: fires on the 3rd identical key in a row");
  s.reset();
  check(!s.observe("a") && !s.observe("b") && !s.observe("a") && !s.observe("a") && s.observe("a"),
        "a different key restarts the run at 1");
  s.reset();
  check(!s.observe("a") && s.run_length() == 1, "reset() clears the run");
  check(s.set_limit(0) && !s.observe("a") && !s.observe("a") && !s.observe("a"), "set_limit(0) turns it OFF again");
}

static void test_stage_b() {
  const StageBDoorGrid g = StageBDoorFactory::make();
  // Default OFF is the official behavior (unchanged): honest stall at the step budget.
  StageBDoorResult off;
  MirrorSimSideLog log_off;
  StageBDoorMaze m_off(g);
  m_off.enable_mirror_sim(&log_off);
  check(m_off.identical_retry_stop() == 0, "StageBDoorMaze default: identical-retry stop OFF");
  const bool ok_off = m_off.run_unit(&propose_always_dividor, nullptr, off);
  report("off  always-dividor", ok_off, off);
  check(!ok_off && off.err == "step budget" && off.stalled_door == 1 && off.steps_used == off.cap &&
            !off.early_stopped && off.attempts_per_door[1] > 3,
        "OFF: wrong door-1 answer retries until the step budget (unchanged behavior)");

  // ON (N=3): the 3rd identical failed attempt at door 1 ends that door's attempts.
  StageBDoorResult on;
  MirrorSimSideLog log_on;
  StageBDoorMaze m_on(g);
  m_on.enable_mirror_sim(&log_on);
  check(m_on.set_identical_retry_stop(3) && m_on.identical_retry_stop() == 3, "opt-in N=3 accepted");
  check(!m_on.set_identical_retry_stop(1) && m_on.identical_retry_stop() == 3, "N=1 rejected; setting unchanged");
  const bool ok_on = m_on.run_unit(&propose_always_dividor, nullptr, on);
  report("on3  always-dividor", ok_on, on);
  check(!ok_on && on.early_stopped && on.stalled_door == 1 && on.attempts_per_door[1] == 3 &&
            on.err == "identical retries (identical-retry stop)",
        "ON N=3: stops after 3 identical failed attempts at door 1 (honest stall, earlier)");
  check(on.door_open[0] && !on.door_open[1] && !on.win && !on.reached_exit, "ON: no door opened falsely, no win");
  check(on.steps_used < off.steps_used && on.actions == off.actions && on.h_world_sha == off.h_world_sha &&
            log_on.canonical_bytes() == log_off.canonical_bytes(),
        "ON: same moves, world, and side log as OFF up to the stall; fewer steps");

  // ON with correct answers: never fires; identical to OFF.
  StageBDoorResult r_off, r_on;
  StageBDoorMaze a(g), b(g);
  (void)b.set_identical_retry_stop(2);
  const bool oka = a.run_unit(&propose_right, nullptr, r_off);
  const bool okb = b.run_unit(&propose_right, nullptr, r_on);
  check(oka && okb && r_on.win && !r_on.early_stopped && r_on.actions == r_off.actions &&
            r_on.steps_used == r_off.steps_used && r_on.h_world_sha == r_off.h_world_sha,
        "ON N=2 with correct answers: never fires, identical to OFF");

  // ON with alternating (non-identical) wrong answers: never fires; runs to the step budget.
  int k1 = 0, k2 = 0;
  StageBDoorResult alt_off, alt_on;
  StageBDoorMaze c(g), d(g);
  (void)d.set_identical_retry_stop(2);
  const bool okc = c.run_unit(&propose_alternating_wrong, &k1, alt_off);
  const bool okd = d.run_unit(&propose_alternating_wrong, &k2, alt_on);
  report("on2  alternating", okd, alt_on);
  check(!okc && !okd && !alt_on.early_stopped && alt_on.err == "step budget" &&
            alt_on.steps_used == alt_off.steps_used && alt_on.attempts_per_door[1] == alt_off.attempts_per_door[1],
        "ON N=2 with alternating wrong answers: not identical, so no early stop");

  // Deterministic run to run.
  StageBDoorResult on2;
  StageBDoorMaze m_on2(g);
  (void)m_on2.set_identical_retry_stop(3);
  (void)m_on2.run_unit(&propose_always_dividor, nullptr, on2);
  check(on2.steps_used == on.steps_used && on2.attempts_per_door[1] == on.attempts_per_door[1] &&
            on2.actions == on.actions && on2.err == on.err,
        "ON is run-to-run deterministic");
}

int main() {
  std::printf("[RETRY-STOP] identical-retry early stop - LABELED UNIT-TEST FIXTURE ONLY; default OFF\n");
  test_rule();
  test_stage_b();
  std::printf("[RETRY-STOP] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
