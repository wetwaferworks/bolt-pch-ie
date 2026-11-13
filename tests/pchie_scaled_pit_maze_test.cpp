// Bolt PCHIE Stage A - labeled 64x64 2x2-wide branching pit maze under Mirror-Sim.
// Unit/labeled only, never a golden, official chain untouched.
// Full branching maze, 2x2-wide corridors, 1x1 pits along
// one wall with a safe 1-cell lane, pits-as-walls BFS reaches the exit,
// real Little Nemo, honest report.
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
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
static void check(bool ok, const char* what) {
  std::printf("[SCALED-PIT-MAZE] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static bool collide8(const std::vector<uint8_t>& sha, uint64_t fnv) {
  return sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
}

static void alert_collide(const std::vector<uint8_t>& sha, uint64_t fnv, const char* tag) {
  if (collide8(sha, fnv)) {
    std::printf("[SCALED-PIT-MAZE] ALERT: FNV vs SHA disagreement FAILED on %s -- report it immediately\n",
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

static bool path_avoids_all_pits(const MazeGrid& start, const std::vector<MazeAction>& path) {
  MazeGrid g = start;
  if (g.in_pit()) return false;
  for (MazeAction a : path) {
    if (!g.apply(a)) return false;
    if (g.in_pit()) return false;
  }
  return g.at_exit();
}

static bool in_2x2_nonwall_block(const MazeGrid& g, int x, int y) {
  const int ox[4] = {0, -1, 0, -1};
  const int oy[4] = {0, 0, -1, -1};
  for (int k = 0; k < 4; ++k) {
    const int x0 = x + ox[k], y0 = y + oy[k];
    bool ok = true;
    for (int dy = 0; dy < 2 && ok; ++dy) {
      for (int dx = 0; dx < 2 && ok; ++dx) {
        if (!g.in_bounds(x0 + dx, y0 + dy) || g.cell(x0 + dx, y0 + dy) == MazeCell::Wall) ok = false;
      }
    }
    if (ok) return true;
  }
  return false;
}

static bool every_nonwall_in_2x2(const MazeGrid& g) {
  for (int y = 0; y < g.height(); ++y) {
    for (int x = 0; x < g.width(); ++x) {
      if (g.cell(x, y) == MazeCell::Wall) continue;
      if (!in_2x2_nonwall_block(g, x, y)) return false;
    }
  }
  return true;
}

static void room_topology(const MazeGrid& g, int& rooms, int& branches, int& dead_ends) {
  rooms = branches = dead_ends = 0;
  const int rx = (g.width() > 0) ? (g.width() - 1) / 3 : 0;
  const int ry = (g.height() > 0) ? (g.height() - 1) / 3 : 0;
  auto nonwall = [&](int x, int y) {
    return g.in_bounds(x, y) && g.cell(x, y) != MazeCell::Wall;
  };
  for (int j = 0; j < ry; ++j) {
    for (int i = 0; i < rx; ++i) {
      const int x = 1 + 3 * i;
      const int y = 1 + 3 * j;
      if (!nonwall(x, y)) continue;
      ++rooms;
      int d = 0;
      if (nonwall(x + 2, y) || nonwall(x + 2, y + 1)) ++d;
      if (nonwall(x - 1, y) || nonwall(x - 1, y + 1)) ++d;
      if (nonwall(x, y + 2) || nonwall(x + 1, y + 2)) ++d;
      if (nonwall(x, y - 1) || nonwall(x + 1, y - 1)) ++d;
      if (d >= 3) ++branches;
      if (d == 1) ++dead_ends;
    }
  }
}

static bool pit_has_safe_lane(const MazeGrid& g, int x, int y) {
  if (!g.is_pit(x, y)) return false;
  const int dxs[4] = {0, 0, 1, -1};
  const int dys[4] = {-1, 1, 0, 0};
  for (int k = 0; k < 4; ++k) {
    const int wx = x + dxs[k], wy = y + dys[k];
    const int lx = x - dxs[k], ly = y - dys[k];
    if (!g.in_bounds(wx, wy) || g.cell(wx, wy) != MazeCell::Wall) continue;
    if (!g.in_bounds(lx, ly)) continue;
    const MazeCell c = g.cell(lx, ly);
    if (c != MazeCell::Wall && c != MazeCell::Pit && c != MazeCell::DoorA) return true;
  }
  return false;
}

static void test_official_16x16_untouched() {
  MazeGrid def(1, 16, 16);
  int pits = 0;
  for (int y = 0; y < def.height(); ++y)
    for (int x = 0; x < def.width(); ++x)
      if (def.cell(x, y) == MazeCell::Pit) ++pits;
  const int bfs = MazeOracle::shortest_length(def);
  const int safe = MazeOracle::shortest_length_avoiding_pits(def);
  check(pits == 0 && bfs == 50 && safe == 50,
        "official seed=1 16x16 unchanged: zero pits, BFS == avoiding-pits BFS == 50");
}

static void test_factory_invariants() {
  const MazeGrid a = ScaledPitMaze::make();
  const MazeGrid b = ScaledPitMaze::make(ScaledPitMaze::kLabeledSeed, ScaledPitMaze::kWidth,
                                            ScaledPitMaze::kHeight);
  check(a == b, "64x64 factory is deterministic for the labeled seed");
  check(a.width() == 64 && a.height() == 64 && a.seed() == ScaledPitMaze::kLabeledSeed,
        "factory builds a labeled 64x64 maze");

  // Negative control: official generate() at 64x64 is still 1-wide carved
  // style and must not satisfy the 2x2 corridor invariant.
  const MazeGrid carved(ScaledPitMaze::kLabeledSeed, 64, 64);
  check(!every_nonwall_in_2x2(carved),
        "official generate() 64x64 is not 2x2-wide (negative control)");
  check(every_nonwall_in_2x2(a), "every non-wall cell sits in a 2x2 corridor/room block");

  int rooms = 0, branches = 0, dead_ends = 0;
  room_topology(a, rooms, branches, dead_ends);
  check(rooms == 21 * 21, "64x64 room grid is 21x21 (2x2 rooms, 1-wide walls)");
  check(branches >= 8 && dead_ends >= 8,
        "maze is branching (many junctions and dead ends), not a single corridor");

  const int pits = ScaledPitMaze::pit_count(a);
  check(pits > 0, "seeded random pit count is at least 1");
  bool pits_ok = true;
  int pits_checked = 0;
  for (int y = 0; y < a.height(); ++y) {
    for (int x = 0; x < a.width(); ++x) {
      if (a.cell(x, y) != MazeCell::Pit) continue;
      ++pits_checked;
      if (!pit_has_safe_lane(a, x, y)) pits_ok = false;
    }
  }
  check(pits_ok && pits_checked == pits,
        "every pit is 1x1 along one wall with a safe 1-cell lane opposite");

  const int safe = MazeOracle::shortest_length_avoiding_pits(a);
  const auto safe_path = MazeOracle::shortest_path_avoiding_pits(a);
  check(safe > 0 && int(safe_path.size()) == safe,
        "pits-as-walls BFS proves the exit is reachable");
  check(path_avoids_all_pits(a, safe_path),
        "pits-as-walls BFS path never enters a pit and reaches the exit");

  bool checked_fall = false;
  bool fall_ok = true;
  std::vector<MazeGrid> q;
  q.push_back(a);
  const int nstate = 4 * a.width() * a.height();
  std::vector<uint8_t> seen(size_t(nstate), 0);
  auto pack = [&](const MazeGrid& g) {
    const int layer = (g.door_open() ? 2 : 0) + (g.have_key() ? 1 : 0);
    return layer * a.width() * a.height() + g.y() * a.width() + g.x();
  };
  size_t qi = 0;
  while (qi < q.size() && qi < 16384 && !checked_fall) {
    const MazeGrid cur = q[qi++];
    const int idx = pack(cur);
    if (idx < 0 || idx >= nstate || seen[size_t(idx)]) continue;
    seen[size_t(idx)] = 1;
    for (MazeAction act : cur.legal_actions()) {
      MazeGrid nxt = cur;
      if (!nxt.apply(act)) continue;
      if (nxt.in_pit()) {
        checked_fall = true;
        if (nxt.at_exit() || !nxt.legal_actions().empty()) fall_ok = false;
        if (nxt.apply(MazeAction::North) || nxt.apply(MazeAction::South) || nxt.apply(MazeAction::East) ||
            nxt.apply(MazeAction::West) || nxt.apply(MazeAction::Pickup) || nxt.apply(MazeAction::Use))
          fall_ok = false;
        break;
      }
      q.push_back(std::move(nxt));
    }
  }
  check(checked_fall && fall_ok, "falling into a pit is fail/stuck, never a win");
  std::printf("[SCALED-PIT-MAZE] labeled seed=%llu size=%dx%d pits=%d avoid_bfs=%d rooms=%d branches=%d dead_ends=%d (unit-only, never a golden)\n",
              (unsigned long long)a.seed(), a.width(), a.height(), pits, safe, rooms, branches,
              dead_ends);
}

static void test_unit_driver() {
  const MazeGrid g = ScaledPitMaze::make();
  const int avoid = MazeOracle::shortest_length_avoiding_pits(g);
  MirrorSimSideLog log;
  HeadlessMaze maze(g);
  maze.enable_mirror_sim(&log);
  MazeResult out;
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = maze.run_unit("", nullptr, out);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  int refused_world = 0;
  const bool preds = forks_ok(log, &refused_world);
  check(refused_world == 0, "unit 64x64: no refused action reached the real world");
  check(preds, "unit 64x64: predicted == real H_world for every allowed commit");
  alert_collide(log.sha256(), log.fnv(), "unit 64x64 side log");
  check(!collide8(log.sha256(), log.fnv()), "unit 64x64: side-log FNV != first 8 of SHA-256");
  std::printf("[SCALED-PIT-MAZE] unit driver path=%d avoid_bfs=%d exit=%d ok=%d err=%s allowed=%d refused=%d exhaust=%d wall_clock_s=%.3f\n",
              out.path_length, avoid, int(out.reached_exit), int(ok), out.err.c_str(),
              log.allowed_count(), log.refused_count(), int(out.budget_exhausted), secs);
  if (!(ok && out.reached_exit)) {
    std::printf("[SCALED-PIT-MAZE] unit driver honest miss (stall or budget). Not retuned.\n");
  }
}

static void test_real_little_nemo() {
  const MazeGrid g = ScaledPitMaze::make();
  const int avoid = MazeOracle::shortest_length_avoiding_pits(g);
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight 64x64: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[SCALED-PIT-MAZE] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  MirrorSimSideLog log1, log2;
  MazeResult a, b;
  auto run_one = [&](MirrorSimSideLog& log, MazeResult& out) -> bool {
    HeadlessMaze maze(g);
    maze.enable_mirror_sim(&log);
    return maze.run(m, "", nullptr, out);
  };
  const auto t0 = std::chrono::steady_clock::now();
  const bool oka = run_one(log1, a);
  const double secs_a = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const auto t1 = std::chrono::steady_clock::now();
  const bool okb = run_one(log2, b);
  const double secs_b = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  int refused_world = 0;
  const bool preds = forks_ok(log1, &refused_world);
  check(refused_world == 0, "real 64x64: no refused action reached the real world");
  check(preds, "real 64x64: predicted == real H_world for every allowed commit");
  check(oka == okb && a.actions == b.actions && a.reached_exit == b.reached_exit &&
            log1.canonical_bytes() == log2.canonical_bytes(),
        "real 64x64: proposals, actions, and side log are run-to-run deterministic");

  std::printf("[SCALED-PIT-MAZE] real Little Nemo path=%d avoid_bfs=%d exit=%d ok=%d err=%s allowed=%d refused=%d exhaust=%d wall_clock_s=%.3f (run2=%.3f)\n",
              a.path_length, avoid, int(a.reached_exit), int(oka), a.err.c_str(),
              log1.allowed_count(), log1.refused_count(), int(a.budget_exhausted), secs_a, secs_b);
  std::printf("[SCALED-PIT-MAZE] real side_log sha256=%s fnv=0x%016llX (provisional, unit-only, never a golden)\n",
              GgufWeightsHash::to_hex(log1.sha256()).c_str(), (unsigned long long)log1.fnv());
  if (!(oka && a.reached_exit)) {
    std::printf("[SCALED-PIT-MAZE] honest miss: Little Nemo did not reach the exit (stall or budget). Not retuned.\n");
  } else {
    check(a.path_length >= avoid, "real 64x64: path length is at least pit-avoiding BFS");
  }
  alert_collide(log1.sha256(), log1.fnv(), "real 64x64 side log");
  if (!a.h_world_sha.empty()) alert_collide(a.h_world_sha, a.h_world_fnv, "real 64x64 H_world");
  check(!collide8(log1.sha256(), log1.fnv()), "real 64x64: side-log FNV != first 8 of SHA-256");
  if (!a.h_world_sha.empty())
    check(!collide8(a.h_world_sha, a.h_world_fnv), "real 64x64: H_world FNV != first 8 of SHA-256");
  check(live_end == live0, "real 64x64: no KV page leak after runs");
  std::printf("[SCALED-PIT-MAZE] live_pages start=%u end=%u\n", live0, live_end);
}

int main(int argc, char** argv) {
  bool unit_only = false;
  for (int i = 1; i < argc; ++i) {
    if (argv[i] && std::strcmp(argv[i], "--unit") == 0) unit_only = true;
  }
  std::printf("[SCALED-PIT-MAZE] Stage A labeled 64x64 2x2 branching pit maze (official chain untouched)%s\n",
              unit_only ? " [--unit]" : "");
  test_official_16x16_untouched();
  test_factory_invariants();
  test_unit_driver();
  if (!unit_only) test_real_little_nemo();
  std::printf("[SCALED-PIT-MAZE] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
