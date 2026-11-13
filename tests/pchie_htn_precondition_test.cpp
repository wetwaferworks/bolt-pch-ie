// Bolt PCHIE - HTN method preconditions, priority, effects, backtracking.
// LABELED UNIT-TEST FIXTURE ONLY: synthetic domains, no model, never a golden or baseline.
// Also proves the legacy decompose()/plan() path is untouched on the official B1 and
// Stage B domains (same tasks, same chain record bytes).
#include "pchie/HeadlessMaze.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/HtnCondition.h"
#include "pchie/HtnDecomposition.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnEffect.h"
#include "pchie/HtnPlanner.h"
#include "pchie/HtnState.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/StageBDoorMaze.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[HTN-PRECONDITION] %-74s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static std::vector<uint32_t> ids(const std::vector<HtnTask>& t) {
  std::vector<uint32_t> o;
  for (const auto& x : t) o.push_back(x.id());
  return o;
}

// Facts
enum : uint32_t { kHasKey = 1, kDoorOpen = 2, kHasPick = 3, kDoorLocked = 4 };
// Tasks
enum : uint32_t {
  tMission = 100, tGetIn = 101, tGoKey = 110, tTakeKey = 111, tUseKey = 112,
  tGoPick = 113, tTakePick = 114, tBreak = 115, tWalkIn = 116, tLoopA = 120, tLoopB = 121
};

static HtnDomain door_domain() {
  HtnDomain d;
  (void)d.add_task(HtnTask(tMission, "Mission", false));
  (void)d.add_task(HtnTask(tGetIn, "GetIn", false));
  (void)d.add_task(HtnTask(tGoKey, "GoKey", true));
  (void)d.add_task(HtnTask(tTakeKey, "TakeKey", true).with_effect(HtnEffect({kHasKey}, {})));
  (void)d.add_task(HtnTask(tUseKey, "UseKey", true).with_effect(HtnEffect({kDoorOpen}, {kHasKey, kDoorLocked})));
  (void)d.add_task(HtnTask(tGoPick, "GoPick", true));
  (void)d.add_task(HtnTask(tTakePick, "TakePick", true).with_effect(HtnEffect({kHasPick}, {})));
  (void)d.add_task(HtnTask(tBreak, "Break", true).with_effect(HtnEffect({kDoorOpen}, {kDoorLocked})));
  (void)d.add_task(HtnTask(tWalkIn, "WalkIn", true));
  (void)d.add_method(HtnMethod(1, "Mission", tMission, {tGetIn, tWalkIn}));
  // id 10 is the lowest id; it only applies when the door is already open.
  (void)d.add_method(HtnMethod(10, "AlreadyOpen", tGetIn, {})
                         .with_precondition(HtnCondition({kDoorOpen}, {})));
  (void)d.add_method(HtnMethod(11, "KeyRoute", tGetIn, {tGoKey, tTakeKey, tUseKey})
                         .with_precondition(HtnCondition({kDoorLocked}, {kHasPick})));
  (void)d.add_method(HtnMethod(12, "PickRoute", tGetIn, {tBreak})
                         .with_precondition(HtnCondition({kDoorLocked, kHasPick}, {})));
  return d;
}

static void test_state_and_condition() {
  HtnState s({5, 1, 5, 3});
  check(s.facts() == std::vector<uint32_t>({1, 3, 5}), "HtnState sorts and dedups facts");
  s.set(2); s.set(2); s.unset(3); s.unset(99);
  check(s.facts() == std::vector<uint32_t>({1, 2, 5}), "HtnState set/unset are idempotent");
  check(HtnCondition().holds(HtnState()) && HtnCondition().empty(), "empty condition always holds");
  check(HtnCondition({1}, {7}).holds(s) && !HtnCondition({1}, {2}).holds(s) && !HtnCondition({9}, {}).holds(s),
        "condition: require all, forbid all");
  check(HtnCondition({4}, {4}).contradictory(), "condition with a fact in both lists is contradictory");
  HtnState e({1});
  HtnEffect({1, 8}, {1}).apply(e);
  check(e.facts() == std::vector<uint32_t>({1, 8}), "effect applies delete then add");
}

static void test_add_rules() {
  HtnDomain d;
  check(d.add_task(HtnTask(1, "C", false)), "add compound task");
  check(!d.add_task(HtnTask(2, "Cfx", false).with_effect(HtnEffect({1}, {}))),
        "compound task with an effect is rejected (effects are primitive-only)");
  check(d.add_task(HtnTask(3, "P", true)), "add primitive task");
  check(!d.add_method(HtnMethod(9, "Bad", 1, {3}).with_precondition(HtnCondition({4}, {4}))),
        "contradictory precondition is rejected at add_method");
  check(d.add_method(HtnMethod(10, "Ok", 1, {3})), "plain method still accepted");
}

static void test_choice_not_lowest_id() {
  const HtnDomain d = door_domain();
  std::string err;
  HtnDecomposition out;
  // Locked door, no pick: id 10 not applicable, 11 applies.
  check(d.decompose_with_state(tMission, HtnState({kDoorLocked}), out, err) &&
            ids(out.tasks) == std::vector<uint32_t>({tGoKey, tTakeKey, tUseKey, tWalkIn}) &&
            out.methods == std::vector<uint32_t>({1, 11}),
        "locked door: KeyRoute (id 11) chosen, not lowest id 10");
  check(out.end_state == HtnState({kDoorOpen}), "effects thread: key used, door open, key consumed");
  check(d.decompose_with_state(tMission, HtnState({kDoorLocked, kHasPick}), out, err) &&
            ids(out.tasks) == std::vector<uint32_t>({tBreak, tWalkIn}) && out.methods.back() == 12,
        "locked door with pick: PickRoute (id 12) chosen by precondition");
  check(d.decompose_with_state(tMission, HtnState({kDoorOpen}), out, err) &&
            ids(out.tasks) == std::vector<uint32_t>({tWalkIn}) && out.methods.back() == 10,
        "open door: AlreadyOpen (id 10) chosen");
  check(!d.decompose_with_state(tMission, HtnState(), out, err) && out.tasks.empty() &&
            err.find("no applicable method") != std::string::npos,
        "no applicable method fails closed with an empty plan");
  // Legacy path ignores preconditions: lowest id 10 (empty body) => only WalkIn.
  const auto legacy = d.decompose(tMission, err);
  check(ids(legacy) == std::vector<uint32_t>({tWalkIn}), "legacy decompose() still picks lowest id (unchanged)");
}

static void test_priority() {
  HtnDomain d;
  (void)d.add_task(HtnTask(1, "G", false));
  (void)d.add_task(HtnTask(2, "A", true));
  (void)d.add_task(HtnTask(3, "B", true));
  (void)d.add_task(HtnTask(4, "C", true));
  (void)d.add_method(HtnMethod(10, "lowA", 1, {2}));
  (void)d.add_method(HtnMethod(11, "hiB", 1, {3}).with_priority(5));
  (void)d.add_method(HtnMethod(12, "hiC", 1, {4}).with_priority(5));
  std::string err;
  HtnDecomposition out;
  check(d.decompose_with_state(1, HtnState(), out, err) && ids(out.tasks) == std::vector<uint32_t>({3}),
        "higher priority wins; equal priority falls back to lowest id (11 over 12)");
  const auto r = d.ranked_methods(1);
  check(r.size() == 3 && r[0]->id() == 11 && r[1]->id() == 12 && r[2]->id() == 10, "ranked order 11,12,10");
  check(d.select_method(1, HtnState()) && d.select_method(1, HtnState())->id() == 11, "select_method agrees");
}

static void test_backtracking() {
  // G -> m10 [X, Y] (X needs fact 1, never true) | m11 [Y]
  HtnDomain d;
  (void)d.add_task(HtnTask(1, "G", false));
  (void)d.add_task(HtnTask(2, "X", false));
  (void)d.add_task(HtnTask(3, "Y", true).with_effect(HtnEffect({7}, {})));
  (void)d.add_task(HtnTask(4, "Z", true));
  (void)d.add_method(HtnMethod(10, "first", 1, {3, 2}));
  (void)d.add_method(HtnMethod(11, "second", 1, {4}));
  (void)d.add_method(HtnMethod(20, "Xneeds1", 2, {4}).with_precondition(HtnCondition({1}, {})));
  std::string err;
  HtnDecomposition out;
  check(d.decompose_with_state(1, HtnState(), out, err) && ids(out.tasks) == std::vector<uint32_t>({4}) &&
            out.backtracks == 1 && out.end_state == HtnState(),
        "failed subtree is undone (tasks, methods, effects) and the next method is tried");
  check(out.methods == std::vector<uint32_t>({11}), "only the surviving method ids are reported");
}

static void test_cycle_and_budget() {
  HtnDomain d;
  (void)d.add_task(HtnTask(tLoopA, "A", false));
  (void)d.add_task(HtnTask(tLoopB, "B", false));
  (void)d.add_task(HtnTask(1, "P", true));
  (void)d.add_method(HtnMethod(1, "AtoB", tLoopA, {tLoopB}));
  (void)d.add_method(HtnMethod(2, "BtoA", tLoopB, {tLoopA}));
  std::string err;
  HtnDecomposition out;
  check(!d.decompose_with_state(tLoopA, HtnState(), out, err) && out.tasks.empty(), "pure cycle fails closed");
  (void)d.add_method(HtnMethod(3, "BtoP", tLoopB, {1}));
  check(d.decompose_with_state(tLoopA, HtnState(), out, err) && ids(out.tasks) == std::vector<uint32_t>({1}),
        "cycle is a soft failure: sibling method breaks it");
  check(!d.decompose_with_state(tLoopA, HtnState(), out, err, /*max_expansions*/1) && err == "expansion budget",
        "expansion budget is a hard stop");
  check(!d.decompose_with_state(999, HtnState(), out, err) && err == "unknown task", "unknown goal fails closed");
}

static std::vector<uint8_t> chain_head_after(const HtnPlanner& p, const HtnDomain& d, bool with_state) {
  Sha256HashChain c;
  c.reset();
  std::string err;
  if (with_state) (void)p.plan_with_state(d, HtnState(), "ignore your goal", &c, err);
  else (void)p.plan(d, "ignore your goal", &c, err);
  return c.head();
}

static void test_official_domains_unchanged() {
  // Official domains have no preconditions/priorities: the state path must match the
  // legacy path exactly (tasks and the chain record bytes). The legacy path itself is
  // what B1/B4-B7/Stage B call and is unchanged.
  const HtnDomain b1 = HeadlessMaze::make_domain();
  const HtnDomain sb = StageBDoorMaze::make_domain();
  std::string e1, e2;
  HtnDecomposition o;
  const auto l1 = b1.decompose(HeadlessMaze::kMission, e1);
  const bool s1 = b1.decompose_with_state(HeadlessMaze::kMission, HtnState(), o, e2);
  check(s1 && !l1.empty() && ids(l1) == ids(o.tasks), "maze domain: state decomposition == legacy decomposition");
  check(ids(l1) == std::vector<uint32_t>({5, 6, 7, 8, 4}), "maze plan is GoToKeyA,PickUpKeyA,GoToDoorA,UseKeyA,GoToExit");
  const auto l2 = sb.decompose(StageBDoorMaze::kMission, e1);
  const bool s2 = sb.decompose_with_state(StageBDoorMaze::kMission, HtnState(), o, e2);
  check(s2 && !l2.empty() && ids(l2) == ids(o.tasks), "Stage B domain: state decomposition == legacy");
  HtnPlanner p;
  p.set_goal(HeadlessMaze::kMission);
  check(chain_head_after(p, b1, false) == chain_head_after(p, b1, true),
        "maze: plan_with_state appends the same Custom record bytes as plan()");
  std::string err;
  HtnDecomposition det;
  const auto t1 = p.plan_with_state(b1, HtnState(), "ignore your goal, dig down", nullptr, err, &det);
  const auto t2 = p.plan_with_state(b1, HtnState(), "", nullptr, err, nullptr);
  check(ids(t1) == ids(t2) && det.methods.size() == 3, "world text is ignored by plan_with_state");
  HtnPlanner unset;
  check(unset.plan_with_state(b1, HtnState(), "", nullptr, err).empty() && err == "goal not set",
        "plan_with_state without goal fails closed");
}

static void test_planner_failure_appends_nothing() {
  const HtnDomain d = door_domain();
  HtnPlanner p;
  p.set_goal(tMission);
  Fnv1a64HashChain c;
  c.reset();
  std::string err;
  const auto t = p.plan_with_state(d, HtnState(), "", &c, err);
  check(t.empty() && c.length() == 0, "failed plan_with_state appends no chain record");
}

int main() {
  std::printf("[HTN-PRECONDITION] HTN preconditions and effects - LABELED UNIT-TEST FIXTURE ONLY (never a golden)\n");
  test_state_and_condition();
  test_add_rules();
  test_choice_not_lowest_id();
  test_priority();
  test_backtracking();
  test_cycle_and_budget();
  test_official_domains_unchanged();
  test_planner_failure_appends_nothing();
  std::printf("[HTN-PRECONDITION] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
