// Bolt PCHIE Stage B - labeled sectional 2x2-corridor maze with two L1
// logic-gate doors (Dividor door, then Multiplexor door).
// Unit/labeled only, never a golden, official chain untouched (null chain).
// Built with the checklist defaults
//. No retune after a miss. Honest report.
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/Dividor.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/JoinJobDispatcher.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/Multiplexor.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"
#include "pchie/SimTraceGrader.h"
#include "pchie/StageBDoorFactory.h"
#include "pchie/StageBDoorGrader.h"
#include "pchie/StageBDoorGrid.h"
#include "pchie/StageBDoorKey.h"
#include "pchie/StageBDoorMaze.h"
#include "pchie/StageBDoorOracle.h"
#include "pchie/StageBLogicDoor.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceL1AnswerKey.h"
#include "pchie/TraceL1Curriculum.h"
#include "pchie/TraceL1Harness.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace pchie;

static const std::string& model_path() {
  static const std::string p = [] {
    std::string path, err;
    if (!pchie::resolve_model_path(path, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return std::string();
    }
    return path;
  }();
  return p;
}


static int g_fail = 0;
// Real-model runs use every available thread by default (--workers N to choose). The join dispatcher makes
// results independent of the thread count. Loop guard v2 is on by default; --loop-guard off gives the old walk.
static int g_workers = 1;
static bool g_loop_guard = true;
static StreamAuthFlags g_stream_auth;  // TESLA + attestation on by default
static void check(bool ok, const char* what) {
  std::printf("[STAGE-B] %-79s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static bool collide8(const std::vector<uint8_t>& sha, uint64_t fnv) {
  return sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
}

static void alert_collide(const std::vector<uint8_t>& sha, uint64_t fnv, const char* tag) {
  if (collide8(sha, fnv)) {
    std::printf("[STAGE-B] ALERT: FNV vs SHA disagreement FAILED on %s -- report it immediately\n",
                tag);
  }
}

static bool forks_ok(const MirrorSimSideLog& log, int* refused_world) {
  if (refused_world) *refused_world = 0;
  bool pred_ok = true;
  for (const auto& f : log.forks()) {
    if (!f.allowed) {
      if (!f.real_post_sha.empty()) {
        if (refused_world) ++*refused_world;
        pred_ok = false;
      }
      continue;
    }
    if (f.pred_sha != f.real_post_sha || f.pred_fnv != f.real_post_fnv) pred_ok = false;
  }
  return pred_ok;
}

// ---------------------------------------------------------------------------
// Official-hash invariance (must match the wander-fix table).
static const char* kB1WorldSha =
    "457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5";
static const uint64_t kB1WorldFnv = 0xD968C7EE5D48C403ull;
static const char* kTraceDraftKeySha =
    "32a5a92832e94043ef9252ebf581374e52ebf164520afaa616d1347fdeda520f";
static const uint64_t kTraceDraftKeyFnv = 0x4DA259404F3809EAull;

static void test_official_invariance() {
  MazeGrid def(1, 16, 16);
  const int bfs = MazeOracle::shortest_length(def);
  check(bfs == 50, "official seed=1 16x16 maze BFS is still 50");
  HeadlessMaze maze(1);
  MazeResult out;
  const bool ok = maze.run_unit("", nullptr, out);
  check(ok && out.reached_exit && out.path_length == 50,
        "official maze unit driver still reaches the exit in 50");
  check(GgufWeightsHash::to_hex(out.h_world_sha) == kB1WorldSha && out.h_world_fnv == kB1WorldFnv,
        "official maze H_world sha256/fnv unchanged (457668e7... / 0xD968C7EE5D48C403)");
  check(GgufWeightsHash::to_hex(TraceL1AnswerKey::sha256()) == kTraceDraftKeySha &&
            TraceL1AnswerKey::fnv() == kTraceDraftKeyFnv,
        "Trace L1 draft key sha256/fnv unchanged (Stage B ids not added to it)");
  const auto tasks = TraceL1Curriculum::tasks();
  check(tasks.size() == 10 &&
            TraceL1Harness::prompt_for(tasks[0]) ==
                "L1 task L1-D0. Place one component facing north. south=0 west=64 east=1. "
                "Goal output at north=0.",
        "recorded Trace L1 prompt format and ids unchanged");
}

// ---------------------------------------------------------------------------
// Factory.
static bool in_2x2_nonwall_block(const StageBDoorGrid& g, int x, int y) {
  const int ox[4] = {0, -1, 0, -1};
  const int oy[4] = {0, 0, -1, -1};
  for (int k = 0; k < 4; ++k) {
    const int x0 = x + ox[k], y0 = y + oy[k];
    bool ok = true;
    for (int dy = 0; dy < 2 && ok; ++dy)
      for (int dx = 0; dx < 2 && ok; ++dx)
        if (!g.in_bounds(x0 + dx, y0 + dy) || g.cell(x0 + dx, y0 + dy) == StageBCell::Wall)
          ok = false;
    if (ok) return true;
  }
  return false;
}

static void section_topology(const StageBDoorGrid& g, int section, int& rooms, int& dead_ends,
                             int& branches) {
  rooms = dead_ends = branches = 0;
  auto open = [&](int x, int y) {
    return g.in_bounds(x, y) && g.cell(x, y) != StageBCell::Wall &&
           g.cell(x, y) != StageBCell::Door;
  };
  for (int j = 0; j < StageBDoorFactory::kRoomsY; ++j) {
    for (int ii = 0; ii < StageBDoorFactory::kSectionRoomsX; ++ii) {
      const int i = section * StageBDoorFactory::kSectionRoomsX + ii;
      const int x = StageBDoorFactory::room_origin_x(i);
      const int y = StageBDoorFactory::room_origin_y(j);
      if (!open(x, y)) continue;
      ++rooms;
      int d = 0;
      if (open(x + 2, y) || open(x + 2, y + 1)) ++d;
      if (open(x - 1, y) || open(x - 1, y + 1)) ++d;
      if (open(x, y + 2) || open(x + 1, y + 2)) ++d;
      if (open(x, y - 1) || open(x + 1, y - 1)) ++d;
      if (d == 1) ++dead_ends;
      if (d >= 3) ++branches;
    }
  }
}

static void test_factory() {
  const StageBDoorGrid a = StageBDoorFactory::make();
  const StageBDoorGrid b = StageBDoorFactory::make();
  check(a == b && a.canonical_bytes() == b.canonical_bytes(),
        "Stage B factory is deterministic for the labeled seed");
  check(a.seed() == StageBDoorFactory::kLabeledSeed && StageBDoorFactory::kLabeledSeed == 7 &&
            a.width() == 28 && a.height() == 10,
        "labeled seed=7, 28x10 (3 sections x 3x3 rooms of 2x2)");
  const StageBDoorGrid other = StageBDoorFactory::make(8);
  check(!(other == a), "a different seed gives a different labeled layout");

  bool all2x2 = true;
  int bad_cells = 0, doors[2] = {0, 0}, starts = 0, exits = 0;
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      const StageBCell c = a.cell(x, y);
      if (c == StageBCell::Wall) continue;
      if (!in_2x2_nonwall_block(a, x, y)) all2x2 = false;
      if (c == StageBCell::Door) {
        const int k = a.door_at(x, y);
        if (k == 0 || k == 1) ++doors[k];
        else ++bad_cells;
      } else if (c == StageBCell::Start) {
        ++starts;
      } else if (c == StageBCell::Exit) {
        ++exits;
      } else if (c != StageBCell::Floor) {
        ++bad_cells;
      }
      if (c != StageBCell::Door && a.door_at(x, y) != -1) ++bad_cells;
    }
  }
  check(all2x2, "every non-wall cell sits in a 2x2 block (two-cell-wide corridors)");
  check(doors[0] == 2 && doors[1] == 2 && bad_cells == 0,
        "exactly two doors, each a 2-cell throat, ids 0 and 1");
  check(starts == 1 && exits == 1, "one start, one exit");
  check(a.x() == 1 && a.y() == 1 && a.cell(1, 1) == StageBCell::Start,
        "agent starts on the start cell in room (0,0)");
  check(!a.door_open(0) && !a.door_open(1), "both doors start closed");

  int total_rooms = 0, total_dead = 0;
  bool each_branchy = true;
  for (int s = 0; s < StageBDoorFactory::kSections; ++s) {
    int rooms = 0, dead = 0, br = 0;
    section_topology(a, s, rooms, dead, br);
    total_rooms += rooms;
    total_dead += dead;
    if (rooms != 9 || dead < 1) each_branchy = false;
  }
  check(total_rooms == 27 && each_branchy,
        "3 sections of 9 rooms, each section has at least one dead end (not a corridor)");

  const int all_open = StageBDoorOracle::shortest_length_all_open(a);
  check(all_open > 0, "all-doors-open BFS reaches the exit");
  check(StageBDoorOracle::shortest_length_to_exit(a) < 0,
        "with both doors closed the exit is unreachable");
  check(StageBDoorOracle::dist_to_subgoal(a, StageBDoorOracle::kGoToDoor0) > 0,
        "door 0 is reachable (adjacent cell) from the start with doors closed");
  check(StageBDoorOracle::dist_to_subgoal(a, StageBDoorOracle::kGoToDoor1) < 0,
        "door 1 is not reachable until door 0 opens (sectional)");
  std::printf("[STAGE-B] labeled seed=%llu size=%dx%d rooms=%d dead_ends=%d all_open_bfs=%d (unit-only, never a golden)\n",
              (unsigned long long)a.seed(), a.width(), a.height(), total_rooms, total_dead,
              all_open);
  for (int y = 0; y < a.height(); ++y) {
    std::string row;
    for (int x = 0; x < a.width(); ++x) {
      switch (a.cell(x, y)) {
        case StageBCell::Wall: row += '#'; break;
        case StageBCell::Floor: row += '.'; break;
        case StageBCell::Start: row += 'S'; break;
        case StageBCell::Exit: row += 'E'; break;
        case StageBCell::Door: row += char('0' + a.door_at(x, y)); break;
      }
    }
    std::printf("[STAGE-B]   %s\n", row.c_str());
  }
}

// ---------------------------------------------------------------------------
// Door tasks, key, grader.
static bool contains(const std::string& s, const char* needle) {
  return s.find(needle) != std::string::npos;
}

static void test_door_tasks_and_grader() {
  check(StageBDoorKey::door_count() == 2, "Stage B key has two door tasks");
  TraceL1Task t0, t1;
  check(StageBDoorKey::task_for_door(0, t0) && StageBDoorKey::task_for_door(1, t1),
        "each door has a task");
  const auto* k0 = StageBDoorKey::for_task(t0.id);
  const auto* k1 = StageBDoorKey::for_task(t1.id);
  check(k0 && k1 && std::strcmp(k0->component, Dividor::kName) == 0 &&
            std::strcmp(k1->component, Multiplexor::kName) == 0,
        "door 0 = Dividor (clearer first), door 1 = Multiplexor");
  check(t0.south == 12 && t0.west == 0 && t0.east == 0 && t0.goal == 6 &&
            t1.south == 9 && t1.west == 20 && t1.east == 3 && t1.goal == 20 &&
            t0.facing == TraceFacing::North && t1.facing == TraceFacing::North,
        "door tasks are the recorded ones (SB-3F9A 12/0/0->6, SB-7E2B 9/20/3->20)");
  bool opaque = true;
  for (const TraceL1Task* t : {&t0, &t1}) {
    const std::string id = t->id;
    const std::string p = TraceL1Harness::prompt_for(*t);
    if (contains(id, "D") || contains(id, "M") || contains(id, "d") || contains(id, "m"))
      opaque = false;
    if (contains(p, "dividor") || contains(p, "multiplexor") || contains(p, "Dividor") ||
        contains(p, "Multiplexor"))
      opaque = false;
    if (TraceL1AnswerKey::for_task(t->id) != nullptr) opaque = false;
  }
  check(opaque, "opaque ids: no D/M hint, prompt never names the answer, not in the Trace L1 key");

  // Distinguishing: right component passes, the other fails.
  bool dist = true;
  for (const TraceL1Task* t : {&t0, &t1}) {
    const auto* k = StageBDoorKey::for_task(t->id);
    TraceCircuit right;
    right.component = k->component;
    right.facing = t->facing;
    right.south = t->south;
    right.west = t->west;
    right.east = t->east;
    TraceCircuit wrong = right;
    wrong.component = std::strcmp(k->component, Dividor::kName) == 0 ? Multiplexor::kName
                                                                      : Dividor::kName;
    const TraceGrade gr = StageBDoorGrader().grade(right, *t);
    const TraceGrade gw = StageBDoorGrader().grade(wrong, *t);
    const TraceGrade sim_w = SimTraceGrader().grade_circuit(wrong, t->goal);
    if (!gr.pass || gr.observed != t->goal || gw.pass || sim_w.pass) dist = false;
  }
  check(dist, "tasks are distinguishing: key component PASSES, other component FAILS");

  TraceCircuit oob;
  oob.component = Dividor::kName;
  oob.facing = TraceFacing::North;
  oob.south = 65;
  TraceL1Task bad = t0;
  bad.south = 65;
  check(!StageBDoorGrader().grade(oob, bad).pass, "out-of-range strength is rejected, not clamped");
  TraceL1Task unknown = t0;
  unknown.id = "L1-D5";
  TraceCircuit c5;
  c5.component = Dividor::kName;
  c5.south = 5;
  check(!StageBDoorGrader().grade(c5, unknown).pass, "a Trace L1 task id is unknown to the Stage B key");

  const auto kb = StageBDoorKey::canonical_bytes();
  const auto ksha = StageBDoorKey::sha256();
  alert_collide(ksha, StageBDoorKey::fnv(), "Stage B door key");
  check(!kb.empty() && ksha.size() == 32 && !collide8(ksha, StageBDoorKey::fnv()),
        "Stage B door key has canonical bytes; FNV != first 8 of SHA-256");
  std::printf("[STAGE-B] door key sha256=%s fnv=0x%016llX (labeled, unit-only, never a golden)\n",
              GgufWeightsHash::to_hex(ksha).c_str(), (unsigned long long)StageBDoorKey::fnv());
}

// ---------------------------------------------------------------------------
// Door invariant: opens only on a correct, gate-passed answer, only that door.
static bool walk_to(StageBDoorGrid& g, uint32_t task) {
  for (int guard = 0; guard < 1000; ++guard) {
    if (StageBDoorOracle::goal_reached(g, task)) return true;
    const auto p = StageBDoorOracle::path_to_subgoal(g, task);
    if (p.empty()) return false;
    if (!g.apply(p[0])) return false;
  }
  return false;
}

static void test_door_invariant() {
  StageBDoorGrid g = StageBDoorFactory::make();
  TraceL1Task t0, t1;
  (void)StageBDoorKey::task_for_door(0, t0);
  (void)StageBDoorKey::task_for_door(1, t1);

  // Not adjacent: even a correct answer does nothing.
  auto a = StageBLogicDoor::resolve(g, 0, t0, Dividor::kName, true);
  check(!a.opened && !g.door_open(0), "correct answer while not adjacent does not open the door");

  check(walk_to(g, StageBDoorOracle::kGoToDoor0) && g.adjacent_to_door(0),
        "oracle walk reaches the cell next to door 0");
  const StageBDoorGrid before = g;
  a = StageBLogicDoor::resolve(g, 0, t0, Multiplexor::kName, true);
  check(!a.opened && !g.door_open(0) && g == before,
        "wrong component does not open door 0 and leaves the world unchanged");
  a = StageBLogicDoor::resolve(g, 0, t0, Dividor::kName, false);
  check(!a.opened && !g.door_open(0) && g == before,
        "gate-rejected (correct-named) proposal does not open door 0");
  a = StageBLogicDoor::resolve(g, 0, t0, nullptr, true);
  check(!a.opened && !g.door_open(0), "missing proposal does not open door 0");
  a = StageBLogicDoor::resolve(g, 0, t1, Multiplexor::kName, true);
  check(!a.opened && !g.door_open(0), "door 1's task cannot open door 0");
  a = StageBLogicDoor::resolve(g, 1, t1, Multiplexor::kName, true);
  check(!a.opened && !g.door_open(1), "door 1 cannot be answered from door 0's side");
  a = StageBLogicDoor::resolve(g, 0, t0, Dividor::kName, true);
  check(a.opened && g.door_open(0) && !g.door_open(1),
        "correct gate-passed answer opens door 0 ONLY (door 1 still closed)");
  check(!g.at_exit() && StageBDoorOracle::shortest_length_to_exit(g) < 0,
        "opening door 0 is not a win; exit still unreachable");
  a = StageBLogicDoor::resolve(g, 0, t0, Dividor::kName, true);
  check(!a.opened, "an already-open door is not re-opened (no double count)");

  check(walk_to(g, StageBDoorOracle::kGoToDoor1) && g.adjacent_to_door(1),
        "after door 0 opens, door 1 becomes reachable");
  a = StageBLogicDoor::resolve(g, 1, t1, Dividor::kName, true);
  check(!a.opened && !g.door_open(1), "wrong component (dividor) does not open door 1");
  a = StageBLogicDoor::resolve(g, 1, t1, Multiplexor::kName, true);
  check(a.opened && g.door_open(1), "correct answer opens door 1");
  check(!g.at_exit(), "both doors open is still not a win until the exit is reached");
  check(walk_to(g, StageBDoorOracle::kGoToExit) && g.at_exit(), "exit is reachable after both doors");
}

// ---------------------------------------------------------------------------
// Headless driver (unit, no model).
static const char* propose_right(void*, int door, const TraceL1Task&) {
  return door == 0 ? Dividor::kName : Multiplexor::kName;
}
static const char* propose_always_dividor(void*, int, const TraceL1Task&) { return Dividor::kName; }
static const char* propose_always_multiplexor(void*, int, const TraceL1Task&) {
  return Multiplexor::kName;
}

static void report(const char* tag, bool ok, const StageBDoorResult& r, double secs) {
  std::printf("[STAGE-B] %s ok=%d win=%d exit=%d door0=%d door1=%d attempts=%d/%d stalled_door=%d path=%d steps=%d cap=%d bfs=%d err=%s wall_clock_s=%.3f\n",
              tag, int(ok), int(r.win), int(r.reached_exit), int(r.door_open[0]),
              int(r.door_open[1]), r.attempts_per_door[0], r.attempts_per_door[1],
              r.stalled_door, r.path_length, r.steps_used, r.cap, r.bfs_length, r.err.c_str(),
              secs);
}

static void test_unit_driver() {
  const StageBDoorGrid g = StageBDoorFactory::make();
  const int bfs = StageBDoorOracle::shortest_length_all_open(g) + StageBDoorGrid::kDoors;
  const int cap = (bfs * 4 < 32) ? 32 : bfs * 4;

  {
    MirrorSimSideLog log, log2;
    StageBDoorMaze m(g), m2(g);
    m.enable_mirror_sim(&log);
    m2.enable_mirror_sim(&log2);
    StageBDoorResult r, r2;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = m.run_unit(&propose_right, nullptr, r);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const bool ok2 = m2.run_unit(&propose_right, nullptr, r2);
    report("unit right-answers", ok, r, s);
    check(r.bfs_length == bfs && r.cap == cap, "driver uses the existing default cap max(32, 4*bfs)");
    check(ok && r.win && r.reached_exit && r.door_open[0] && r.door_open[1],
          "unit, correct answers: both doors open and the exit is reached (win)");
    check(r.attempts_per_door[0] == 1 && r.attempts_per_door[1] == 1,
          "unit, correct answers: one attempt per door");
    int refused_world = 0;
    check(forks_ok(log, &refused_world) && refused_world == 0 && log.refused_count() == 0 &&
              log.allowed_count() == r.path_length,
          "unit: Mirror-Sim predicted == real for every commit, 0 refusals");
    check(ok2 == ok && r.actions == r2.actions && log.canonical_bytes() == log2.canonical_bytes(),
          "unit driver is run-to-run deterministic");
  }
  {
    MirrorSimSideLog log;
    StageBDoorMaze m(g);
    m.enable_mirror_sim(&log);
    StageBDoorResult r;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = m.run_unit(&propose_always_dividor, nullptr, r);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    report("unit always-dividor", ok, r, s);
    check(!ok && !r.win && !r.reached_exit && r.door_open[0] && !r.door_open[1],
          "unit, wrong door-1 answer: door 0 opens, door 1 stays closed, no win");
    check(r.stalled_door == 1 && r.err == "step budget" && r.steps_used == r.cap &&
              r.attempts_per_door[1] > 1,
          "unit, wrong door-1 answer: retries until the step budget, honest stall at door 1");
    int refused_world = 0;
    check(forks_ok(log, &refused_world) && refused_world == 0, "unit stall: predicted == real");
  }
  {
    StageBDoorMaze m(g);
    StageBDoorResult r;
    const bool ok = m.run_unit(&propose_always_multiplexor, nullptr, r);
    report("unit always-multiplexor", ok, r, 0.0);
    check(!ok && !r.win && !r.door_open[0] && !r.door_open[1] && r.stalled_door == 0 &&
              r.err == "step budget",
          "unit, wrong door-0 answer: door 0 stays closed, door 1 never reached, no win");
  }
}

// ---------------------------------------------------------------------------
// Real Little Nemo.
static void test_real_little_nemo() {
  const StageBDoorGrid g = StageBDoorFactory::make();
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight Stage B: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[STAGE-B] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/128, /*branches*/8);
  static JoinJobDispatcher disp(g_workers);
  if (g_workers > 1) m.set_dispatcher(&disp);
  std::printf("[STAGE-B] workers=%d (join dispatcher; results do not depend on the thread count) loop_guard=%s tesla=%s attest=%s\n", g_workers,
              g_loop_guard ? "on" : "off", g_stream_auth.tesla ? "on" : "off", g_stream_auth.attest ? "on" : "off");
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  MirrorSimSideLog log1, log2;
  StageBDoorResult a, b;
  auto run_one = [&](MirrorSimSideLog& log, StageBDoorResult& out) -> bool {
    StageBDoorMaze maze(g);
    maze.enable_loop_guard(g_loop_guard);
    maze.set_stream_auth(g_stream_auth);
    maze.enable_mirror_sim(&log);
    return maze.run(m, out);
  };
  const auto t0 = std::chrono::steady_clock::now();
  const bool oka = run_one(log1, a);
  const double secs_a = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto t1 = std::chrono::steady_clock::now();
  const bool okb = run_one(log2, b);
  const double secs_b = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  report("real Little Nemo run1", oka, a, secs_a);
  report("real Little Nemo run2", okb, b, secs_b);
  std::printf("[STAGE-B] loop guard: run1 triggers=%d run2 triggers=%d\n", a.loop_triggers, b.loop_triggers);
  for (const std::string& line : a.loop_log) std::printf("[STAGE-B]   %s\n", line.c_str());
  for (size_t i = 0; i < a.attempts.size(); ++i) {
    const auto& at = a.attempts[i];
    if (i < 3 || i + 2 >= a.attempts.size())
      std::printf("[STAGE-B]   attempt %zu door=%d task=%s proposed=%s gate=%d grade=%s observed=%d goal=%d opened=%d reason=%s\n",
                  i, at.door, at.task_id, at.proposed, int(at.gate_pass),
                  at.grade.pass ? "PASS" : "FAIL", at.grade.observed, at.grade.expected,
                  int(at.opened), at.reason);
  }
  bool same_attempts = a.attempts.size() == b.attempts.size();
  for (size_t i = 0; same_attempts && i < a.attempts.size(); ++i) {
    if (std::strcmp(a.attempts[i].proposed, b.attempts[i].proposed) != 0 ||
        a.attempts[i].opened != b.attempts[i].opened)
      same_attempts = false;
  }
  int refused_world = 0;
  const bool preds = forks_ok(log1, &refused_world);
  check(refused_world == 0 && preds, "real: predicted == real for every commit, nothing refused reached the world");
  check(oka == okb && a.actions == b.actions && same_attempts &&
            log1.canonical_bytes() == log2.canonical_bytes(),
        "real: proposals, door attempts, actions, side log run-to-run deterministic");
  check(a.win == a.reached_exit && (!a.win || (a.door_open[0] && a.door_open[1])),
        "real: win only at the exit, and only after both doors opened");
  std::printf("[STAGE-B] real side_log sha256=%s fnv=0x%016llX (provisional, unit-only, never a golden)\n",
              GgufWeightsHash::to_hex(log1.sha256()).c_str(), (unsigned long long)log1.fnv());
  alert_collide(log1.sha256(), log1.fnv(), "real Stage B side log");
  check(!collide8(log1.sha256(), log1.fnv()), "real: side-log FNV != first 8 of SHA-256");
  if (!a.h_world_sha.empty()) {
    alert_collide(a.h_world_sha, a.h_world_fnv, "real Stage B H_world");
    std::printf("[STAGE-B] real H_world sha256=%s fnv=0x%016llX (unit-only)\n",
                GgufWeightsHash::to_hex(a.h_world_sha).c_str(), (unsigned long long)a.h_world_fnv);
  }
  if (!(oka && a.win))
    std::printf("[STAGE-B] honest result: Little Nemo did not reach the exit (stall_door=%d, err=%s). Not retuned.\n",
                a.stalled_door, a.err.c_str());
  check(live_end == live0, "real: no KV page leak after runs");
  std::printf("[STAGE-B] live_pages start=%u end=%u\n", live0, live_end);
}

int main(int argc, char** argv) {
  bool unit_only = false;
  const unsigned hw = std::thread::hardware_concurrency();
  g_workers = hw == 0 ? 1 : (hw > unsigned(JoinJobDispatcher::kMaxWorkers) ? JoinJobDispatcher::kMaxWorkers : int(hw));
  for (int i = 1; i < argc; ++i) {
    if (!argv[i]) continue;
    if (std::strcmp(argv[i], "--unit") == 0) unit_only = true;
    else if (std::strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
      const int w = std::atoi(argv[++i]);
      if (w >= 1 && w <= JoinJobDispatcher::kMaxWorkers) g_workers = w;
    } else if (std::strcmp(argv[i], "--loop-guard") == 0 && i + 1 < argc) {
      const char* v = argv[++i];
      if (std::strcmp(v, "on") != 0 && std::strcmp(v, "off") != 0) { std::printf("[STAGE-B] --loop-guard takes on or off\n"); return 3; }
      g_loop_guard = std::strcmp(v, "on") == 0;
    }
  }
  const auto sa = parse_stream_auth_flags(argc, argv);
  if (!sa.ok) { std::printf("[STAGE-B] %s\n", sa.err.c_str()); return 3; }
  g_stream_auth = sa.flags;
  std::printf("[STAGE-B] Stage B labeled 2-door logic-gate maze (official chain untouched)%s\n",
              unit_only ? " [--unit]" : "");
  test_official_invariance();
  test_factory();
  test_door_tasks_and_grader();
  test_door_invariant();
  test_unit_driver();
  if (!unit_only) test_real_little_nemo();
  std::printf("[STAGE-B] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
