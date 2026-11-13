// Bolt PCHIE - HtnTask / HtnMethod / HtnDomain fixture decomposition.
// Labeled unit-test fixture (B1 maze-style tasks). No model, no CUDA, no golden.
#include "pchie/HtnDomain.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace pchie;

// Fixture task ids (maze B1 seed: get key A, open door A, go to exit).
static constexpr uint32_t kMission = 1;
static constexpr uint32_t kGetKeyA = 2;
static constexpr uint32_t kOpenDoorA = 3;
static constexpr uint32_t kGoToExit = 4;
static constexpr uint32_t kGoToKeyA = 5;
static constexpr uint32_t kPickUpKeyA = 6;
static constexpr uint32_t kGoToDoorA = 7;
static constexpr uint32_t kUseKeyA = 8;
static constexpr uint32_t kMethodMission = 10;
static constexpr uint32_t kMethodGetKeyA = 11;
static constexpr uint32_t kMethodOpenDoorA = 12;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[HTN-DOMAIN] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static bool fill_maze_fixture(HtnDomain& d) {
  bool ok = true;
  ok = ok && d.add_task(HtnTask(kMission, "Mission", false));
  ok = ok && d.add_task(HtnTask(kGetKeyA, "GetKeyA", false));
  ok = ok && d.add_task(HtnTask(kOpenDoorA, "OpenDoorA", false));
  ok = ok && d.add_task(HtnTask(kGoToExit, "GoToExit", true));
  ok = ok && d.add_task(HtnTask(kGoToKeyA, "GoToKeyA", true));
  ok = ok && d.add_task(HtnTask(kPickUpKeyA, "PickUpKeyA", true));
  ok = ok && d.add_task(HtnTask(kGoToDoorA, "GoToDoorA", true));
  ok = ok && d.add_task(HtnTask(kUseKeyA, "UseKeyA", true));
  ok = ok && d.add_method(HtnMethod(kMethodMission, "ClearMaze", kMission,
                                   {kGetKeyA, kOpenDoorA, kGoToExit}));
  ok = ok && d.add_method(HtnMethod(kMethodGetKeyA, "FetchKeyA", kGetKeyA,
                                   {kGoToKeyA, kPickUpKeyA}));
  ok = ok && d.add_method(HtnMethod(kMethodOpenDoorA, "UnlockDoorA", kOpenDoorA,
                                   {kGoToDoorA, kUseKeyA}));
  return ok;
}

static void test_task_method_values() {
  const HtnTask t(kGoToExit, "GoToExit", true);
  check(t.id() == kGoToExit && t.name() == "GoToExit" && t.primitive(),
        "HtnTask stores id, name, primitive");
  const HtnTask c(kMission, "Mission", false);
  check(c.primitive() == false, "compound HtnTask is not primitive");
  const HtnMethod m(kMethodMission, "ClearMaze", kMission, {kGetKeyA, kOpenDoorA, kGoToExit});
  check(m.id() == kMethodMission && m.head_task_id() == kMission && m.subtask_ids().size() == 3,
        "HtnMethod stores id, head, ordered subtask ids");
  check(m.subtask_ids()[0] == kGetKeyA && m.subtask_ids()[1] == kOpenDoorA &&
            m.subtask_ids()[2] == kGoToExit,
        "HtnMethod subtask order is registration order");
}

static void test_domain_register_and_lookup() {
  HtnDomain d;
  check(fill_maze_fixture(d), "maze fixture add_task/add_method all succeed");
  check(d.add_task(HtnTask(kMission, "MissionDup", false)) == false,
        "duplicate task id is rejected");
  check(d.add_method(HtnMethod(kMethodMission, "Dup", kMission, {kGoToExit})) == false,
        "duplicate method id is rejected");
  const HtnTask* t = d.find_task(kGetKeyA);
  check(t && t->name() == "GetKeyA" && t->primitive() == false, "find_task(GetKeyA) succeeds");
  check(d.find_task(99) == nullptr, "find_task unknown id returns nullptr");
}

static void test_methods_for_deterministic() {
  HtnDomain d;
  check(d.add_task(HtnTask(kMission, "Mission", false)), "register Mission");
  check(d.add_task(HtnTask(kGoToExit, "GoToExit", true)), "register GoToExit");
  check(d.add_task(HtnTask(kGoToKeyA, "GoToKeyA", true)), "register GoToKeyA");
  // Register higher id first; lookup must still return increasing method id.
  check(d.add_method(HtnMethod(20, "Late", kMission, {kGoToExit})), "add method id 20 first");
  check(d.add_method(HtnMethod(15, "Early", kMission, {kGoToKeyA})), "add method id 15 second");
  const auto ms = d.methods_for(kMission);
  check(ms.size() == 2 && ms[0] && ms[1] && ms[0]->id() == 15 && ms[1]->id() == 20,
        "methods_for returns methods sorted by method id");
}

static void test_expand_once_and_decompose() {
  HtnDomain d;
  check(fill_maze_fixture(d), "fixture ready for decompose");

  std::string err;
  const auto once_prim = d.expand_once(kGoToExit, err);
  check(err.empty() && once_prim.size() == 1 && once_prim[0].id() == kGoToExit,
        "expand_once(primitive) returns that task");

  const auto once_get = d.expand_once(kGetKeyA, err);
  check(err.empty() && once_get.size() == 2 && once_get[0].id() == kGoToKeyA &&
            once_get[1].id() == kPickUpKeyA,
        "expand_once(GetKeyA) => GoToKeyA, PickUpKeyA (lowest method id)");

  const auto once_mission = d.expand_once(kMission, err);
  check(err.empty() && once_mission.size() == 3 && once_mission[0].id() == kGetKeyA &&
            once_mission[1].id() == kOpenDoorA && once_mission[2].id() == kGoToExit,
        "expand_once(Mission) => GetKeyA, OpenDoorA, GoToExit");

  // Full fixture decomposition: primitives in method-body order, recursively.
  const auto plan = d.decompose(kMission, err);
  const std::vector<uint32_t> want = {kGoToKeyA, kPickUpKeyA, kGoToDoorA, kUseKeyA, kGoToExit};
  bool ids_ok = err.empty() && plan.size() == want.size();
  for (size_t i = 0; i < want.size() && ids_ok; ++i) {
    ids_ok = plan[i].id() == want[i] && plan[i].primitive();
  }
  check(ids_ok, "decompose(Mission) => GoToKeyA, PickUpKeyA, GoToDoorA, UseKeyA, GoToExit");

  std::string err2;
  const auto plan2 = d.decompose(kMission, err2);
  check(err2.empty() && plan2 == plan, "decompose is deterministic (same domain, same goal)");
}

static void test_decompose_failures() {
  HtnDomain d;
  (void)d.add_task(HtnTask(kMission, "Mission", false));
  std::string err;
  const auto none = d.decompose(kMission, err);
  check(none.empty() && !err.empty(), "decompose of compound with no method fails cleanly");

  std::string erru;
  const auto unk = d.decompose(99, erru);
  check(unk.empty() && !erru.empty(), "decompose of unknown task fails cleanly");
}

int main() {
  std::printf("[PCH] HtnDomain fixture decomposition tests\n");
  test_task_method_values();
  test_domain_register_and_lookup();
  test_methods_for_deterministic();
  test_expand_once_and_decompose();
  test_decompose_failures();
  std::printf("[PCH] HtnDomain %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}