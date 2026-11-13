// Bolt PCHIE - regression for the B7 real-Nemo corridor wander (fixes 1 and 2).
// EXPECTED OUTPUTS (defined before the product change; not a golden lock):
// Unit-test fixture only (never feeds a golden or baseline):
//   - Seed=1 16x16, after the opening South, the agent is at (1,2) with
//     legal {N,S} and South a real 1-step maze-progress gain over North.
//   - The 0.05 logit mix must not flip that 1-step gap even with the recorded
//     Little Nemo GoToKeyA logits (N=-17.6087, S=-18.8737) that used to invert
//     the prior. Mix is allowed only on an exact maze_value tie (threshold
//     documented on HeadlessMaze). On an exact tie, the higher logit wins.
//   - pick_root_action: a visit lead of 1 or less is a tie and falls through
//     to Q, then maze_value, then lower action id. A lead of 2 or more still
//     wins. At the (1,2) fork, n_N=5 / n_S=4 with better Q and maze_value on
//     South must pick South.
// Real Little Nemo:
//   - Node caps 2, 10, 100 (odd new-node counts) and 3, 11, 17 (even) all
//     reach the exit, path within 1.2x BFS, no 2-cycle oscillation, legal
//     actions only. search_budget = max(16, node_budget), same as the B7
//     harness. Hashes provisional / unlocked. No B9.
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/NemotronModel.h"
#include "pchie/pchie_mcts.h"
#include "pchie/ModelPath.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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


// Recorded GoToKeyA action-word logits from the wander diagnosis (unit only).
static const float kLogitN = -17.6087f;
static const float kLogitS = -18.8737f;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[WANDER] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static const char* act_name(int32_t a) {
  switch (a) {
    case int32_t(MazeAction::North): return "N";
    case int32_t(MazeAction::South): return "S";
    case int32_t(MazeAction::East): return "E";
    case int32_t(MazeAction::West): return "W";
    case int32_t(MazeAction::Pickup): return "P";
    case int32_t(MazeAction::Use): return "U";
    default: return "?";
  }
}

static bool legal_action(int32_t a) {
  return a == int32_t(MazeAction::North) || a == int32_t(MazeAction::South) ||
         a == int32_t(MazeAction::East) || a == int32_t(MazeAction::West) ||
         a == int32_t(MazeAction::Pickup) || a == int32_t(MazeAction::Use);
}

static bool oscillating(const std::vector<int32_t>& a) {
  int run = 0;
  for (size_t i = 0; i + 3 < a.size(); ++i) {
    if (a[i] == a[i + 2] && a[i + 1] == a[i + 3] && a[i] != a[i + 1]) {
      ++run;
      if (run >= 3) return true;
    } else {
      run = 0;
    }
  }
  return false;
}

static uint32_t to_q20(double v) {
  if (v < 0.0) v = 0.0;
  if (v > 1.0) v = 1.0;
  return uint32_t(v * double(PUCT_Q20));
}

// Build a one-level tree at the current grid with given visit/Q on each root edge.
static void plant_child(PuctSearch& S, size_t edge, uint32_t n, int64_t q20) {
  PuctNode leaf;
  leaf.parent = 0;
  leaf.edge_in = uint32_t(edge);
  leaf.depth = 1;
  leaf.n = n;
  leaf.w = n ? (int64_t(n) * q20) : 0;
  leaf.expanded = true;
  const uint32_t id = uint32_t(S.nodes.size());
  S.nodes[0].edges[edge].child = id;
  S.nodes.push_back(leaf);
}

static void test_unit_fork_and_rules() {
  MazeGrid g(1, 16, 16);
  check(g.x() == 1 && g.y() == 1, "unit: seed=1 start is (1,1)");
  check(g.apply(MazeAction::South) && g.x() == 1 && g.y() == 2,
        "unit: opening South lands on the (1,2) fork");

  const uint32_t task = HeadlessMaze::kGoToKeyA;
  MazeGrid npos = g, spos = g;
  check(npos.apply(MazeAction::North) && spos.apply(MazeAction::South),
        "unit: North and South are both legal at (1,2)");
  const int d0 = MazeOracle::dist_to_subgoal(g, task);
  const int dN = MazeOracle::dist_to_subgoal(npos, task);
  const int dS = MazeOracle::dist_to_subgoal(spos, task);
  const double mvN = HeadlessMaze::maze_value(npos, task);
  const double mvS = HeadlessMaze::maze_value(spos, task);
  check(d0 == 25 && dN == 26 && dS == 24,
        "unit: South is a 1-step progress gain, North is a 1-step loss");
  check(mvS > mvN && (dN - dS) == 2, "unit: maze_value(South) > maze_value(North) by a 1-step gap");

  const double best = mvS > mvN ? mvS : mvN;
  // At the fork, pre-move distance is 25; a 1-step gap is 1/25 - 1/27 = 2/(25*27).
  const double gap_at_25 = 2.0 / (25.0 * 27.0);
  check(std::fabs((mvS - mvN) - gap_at_25) < 1e-12,
        "unit: 1-step maze_value gap at d=25 is 2/(25*27)");
  check(HeadlessMaze::kMazeValueTieEps == 0.0,
        "unit: documented mix threshold is exact maze_value equality (eps=0)");
  check(gap_at_25 > HeadlessMaze::kMazeValueTieEps,
        "unit: the 1-step gap is strictly above the mix threshold");

  // Fix 2: these logits used to flip the prior (0.05*?logit > 16*?mv).
  check(!HeadlessMaze::logit_mix_allowed(mvN, best),
        "unit: logit mix is not allowed on the backward 1-step action");
  check(HeadlessMaze::logit_mix_allowed(mvS, best),
        "unit: South is an exact-best maze_value (eligible only if a peer ties)");
  const float sN = HeadlessMaze::mixed_prior_score(mvN, best, kLogitN, /*n_same*/1);
  const float sS = HeadlessMaze::mixed_prior_score(mvS, best, kLogitS, /*n_same*/1);
  check(sS > sN, "unit: mixed prior keeps South ahead of North at the (1,2) fork");
  check(sS == float(mvS) * 16.0f,
        "unit: a unique best-progress action is not mixed (no tie to break)");
  // And the old always-on mix would have flipped (guard that we are testing the real gap).
  const float oldN = float(mvN) * 16.0f + 0.05f * kLogitN;
  const float oldS = float(mvS) * 16.0f + 0.05f * kLogitS;
  check(oldN > oldS, "unit: the recorded logits would flip a 1-step gap under the old always-on mix");

  // Exact tie: same maze_value, higher logit may break the tie.
  check(HeadlessMaze::logit_mix_allowed(mvS, mvS), "unit: mix is allowed on an exact maze_value tie");
  const float tie_hi = HeadlessMaze::mixed_prior_score(mvS, mvS, -10.0f, /*n_same*/2);
  const float tie_lo = HeadlessMaze::mixed_prior_score(mvS, mvS, -12.0f, /*n_same*/2);
  check(tie_hi > tie_lo, "unit: on an exact tie the 0.05 mix still breaks in favor of the higher logit");
  const float cap = HeadlessMaze::kMixCapFrac * 16.0f * float(gap_at_25);
  const float raw_hi = HeadlessMaze::kLogitMix * 6.25f;
  const float raw_lo = HeadlessMaze::kLogitMix * kLogitS;
  const float cap_best = HeadlessMaze::mixed_prior_score(mvS, mvS, 6.25f, 2, cap, raw_lo, raw_hi);
  const float cap_same = HeadlessMaze::mixed_prior_score(mvS, mvS, 6.25f, 2, cap, raw_hi, raw_hi);
  const float worse = HeadlessMaze::mixed_prior_score(mvN, mvS, 6.25f, 1, cap, 0.0f, 0.0f);
  check(cap_best > worse && cap_best <= float(mvS) * 16.0f + cap + 1e-6f,
        "unit: capped mix on a progress tie cannot jump a 1-step gap");
  check(cap_same == float(mvS) * 16.0f,
        "unit: capped mix with identical group logits does not add the raw term");

  // Fix 1: visit lead of 1 at the (1,2) fork must fall through to Q / maze_value.
  PuctSearch S;
  S.reset();
  PuctEdge eN, eS;
  eN.token = int32_t(MazeAction::North);
  eS.token = int32_t(MazeAction::South);
  S.expand(0, {eN, eS});
  plant_child(S, 0, /*n*/5, to_q20(mvN));
  plant_child(S, 1, /*n*/4, to_q20(mvS));
  const int32_t pick = HeadlessMaze::pick_root_action(S, g, task);
  check(pick == int32_t(MazeAction::South),
        "unit: visit lead of 1 at (1,2) falls through; South (better Q and maze_value) is chosen");

  // Lead of 2 still wins even with a worse Q / maze_value.
  PuctSearch S2;
  S2.reset();
  S2.expand(0, {eN, eS});
  plant_child(S2, 0, /*n*/6, to_q20(mvN));
  plant_child(S2, 1, /*n*/4, to_q20(mvS));
  const int32_t pick2 = HeadlessMaze::pick_root_action(S2, g, task);
  check(pick2 == int32_t(MazeAction::North),
        "unit: a visit lead of 2 still wins even when Q and maze_value prefer South");

  // Equal visits, equal Q: maze_value then lower id.
  PuctSearch S3;
  S3.reset();
  S3.expand(0, {eN, eS});
  plant_child(S3, 0, /*n*/4, to_q20(0.04));
  plant_child(S3, 1, /*n*/4, to_q20(0.04));
  const int32_t pick3 = HeadlessMaze::pick_root_action(S3, g, task);
  check(pick3 == int32_t(MazeAction::South),
        "unit: tied visits and Q fall through to maze_value (South)");

  // Equal visits, Q, and maze_value: lower action id (North=7 < South=8).
  // Use a synthetic grid-independent call by giving both the same v via identical children
  // on a state where we compare only id ? reuse spos as if both led to the same value
  // by checking the explicit token order on a fork with equal n, q, and forced equal v.
  // Here both children get the same Q and we temporarily plant on a dummy: if mv differs
  // this case is already covered. Construct equal-mv by picking two actions from a
  // later state is hard; compare two North/South edges with equal n/Q on a copy of
  // pick logic via a grid where both moves are illegal? Instead plant on start after
  // moving to a dead-end equivalent: skip if we cannot equalize mv. We test id order
  // by equal n, Q, and using a custom task-less trick: both edges same token? No.
  // Equal mv: from a position where N and S land at the same subgoal distance.
  // The start corridor at (1,2) does not have that. Test the id rule with equal
  // n and Q on two equally bad / equally good synthetic maze_value by using
  // Pickup/Use if not legal (v=-1 for both).
  PuctSearch S4;
  S4.reset();
  PuctEdge eU, eP;
  eP.token = int32_t(MazeAction::Pickup);
  eU.token = int32_t(MazeAction::Use);
  S4.expand(0, {eP, eU});
  plant_child(S4, 0, 3, to_q20(0.5));
  plant_child(S4, 1, 3, to_q20(0.5));
  const int32_t pick4 = HeadlessMaze::pick_root_action(S4, g, task);
  check(pick4 == int32_t(MazeAction::Pickup),
        "unit: tied visits, Q, and maze_value (both illegal here) fall through to lower action id");
}

static void test_real_node_caps() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[WANDER] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  const int caps[] = {2, 10, 100, 3, 11, 17};
  for (int cap : caps) {
    HeadlessMaze maze(1, 16, 16);
    maze.set_node_budget(cap);
    maze.set_search_budget(cap < 16 ? 16 : cap);
    MazeResult r;
    (void)maze.run(m, "", nullptr, r);
    bool all_legal = !r.actions.empty();
    for (int32_t a : r.actions)
      if (!legal_action(a)) all_legal = false;
    const bool osc = oscillating(r.actions);
    char label[160];
    std::snprintf(label, sizeof(label),
                  "real Little Nemo: node_budget=%d reaches exit within 1.2x BFS, no oscillation",
                  cap);
    check(r.ok && r.reached_exit && r.within_oracle && all_legal && !osc && r.path_length > 0 &&
              r.max_nodes > 0 && r.max_nodes <= cap,
          label);
    std::printf("[WANDER] cap=%d exit=%d path=%d bfs=%d max_nodes=%d osc=%d err=%s\n", cap,
                int(r.reached_exit), r.path_length, r.bfs_length, r.max_nodes, int(osc),
                r.err.c_str());
  }

  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == live0, "real Little Nemo: no KV page leak after wander-regression caps");
  std::printf("[WANDER] live_pages start=%u end=%u\n", live0, live_end);
}

int main() {
  std::printf("[WANDER] maze corridor-wander regression (fixes 1 and 2; hashes provisional)\n");
  test_unit_fork_and_rules();
  test_real_node_caps();
  std::printf("[WANDER] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
