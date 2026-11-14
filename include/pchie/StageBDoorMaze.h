// Bolt PCHIE Stage B - headless driver for the labeled 2-door maze.
// Concrete final. Serial, deterministic, no mutex, no atomic.
// Navigation uses the existing maze search unchanged in kind: PchStack /
// MctsSearch over legal grid moves, leaf value = gated maze progress
// 1/(1+d), the B1 exact-tie-only logit mix (HeadlessMaze::
// mixed_prior_score, kMixCapFrac) and the B1 root pick rule. At each door
// the Trace L1 harness answers through StageBLogicDoor. Default budgets are
// the existing ones (search 16, step cap max(32, 4*bfs)); nothing retuned.
// Off the official chain: there is no chain parameter; every planner /
// search call gets a null chain. Mirror-Sim is commit-time only and
// writes the side log; Stage B has no pits, so no unsafe rule applies.
// Win = reaching the exit only.
// Failed-actuation retry (opt-in, default OFF; official runners never enable it):
// enable_failed_actuation_retry(true) routes door tasks to drive_door_retry() - failure-trace
// injection on a fork of the primed door prompt, then a Tabu mask on repeats keyed by the
// canonical world signature (next-best neural candidate), then a symbolic catalog probe; a
// freeze / no-op proposal is replaced by a real try. With retry OFF, drive_door() is unchanged.
#pragma once

#include "pchie/HeadlessMaze.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnMethod.h"
#include "pchie/HtnTask.h"
#include "pchie/IModel.h"
#include "pchie/IdenticalRetryStop.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/ModelBranch.h"
#include "pchie/NodeSeed.h"
#include "pchie/PchStack.h"
#include "pchie/rl/MazeL1.h"  // opt-in reasoning layer
#include "pchie/neml/NemlStore.h"  // episodic memory receipt at teardown
#include "pchie/neml/NemlTrack.h"
#include "pchie/Dividor.h"
#include "pchie/FailureTrace.h"
#include "pchie/Multiplexor.h"
#include "pchie/PchGate.h"
#include "pchie/Sha256.h"
#include "pchie/StageBRetryStep.h"
#include "pchie/StageBDoorGrid.h"
#include "pchie/StageBDoorKey.h"
#include "pchie/StageBDoorOracle.h"
#include "pchie/StageBLogicDoor.h"
#include "pchie/TraceL1RetryHarness.h"
#include "pchie/TraceL1Task.h"
#include "pchie/UntrustedText.h"
#include "pchie/WorldTabu.h"
#include "pchie/pchie_mcts.h"
#include "pchie/pchie_tree.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pchie {

struct StageBDoorResult {
  std::vector<HtnTask> plan;
  std::vector<int32_t> actions;  // committed moves only
  std::vector<StageBDoorAttempt> attempts;
  int path_length = 0;
  int bfs_length = 0;  // all-doors-open move BFS + one answer per door
  int cap = 0;
  int steps_used = 0;  // moves + door attempts + Mirror-Sim refusals
  bool door_open[StageBDoorGrid::kDoors] = {false, false};
  int attempts_per_door[StageBDoorGrid::kDoors] = {0, 0};
  int stalled_door = -1;
  bool early_stopped = false;  // identical-retry stop (opt-in) ended a door
  bool reached_exit = false;
  bool win = false;
  bool ok = false;
  std::string err;
  std::vector<uint8_t> h_world_sha;
  uint64_t h_world_fnv = 0;
  int mirror_allowed = 0;
  int mirror_refused = 0;
  std::vector<uint8_t> mirror_log_sha;
  uint64_t mirror_log_fnv = 0;
  // Failed-actuation retry (opt-in; all zero / empty when OFF).
  int injections = 0;          // failure records added to the scratch context
  int tabu_masks = 0;          // candidates masked as repeats on the same world signature
  int symbolic_fallbacks = 0;  // symbolic catalog-probe tries
  int noop_recoveries = 0;     // freeze / no-op proposals replaced by a real try
  int exhausted_rounds = 0;    // every candidate masked on a world: tabu cleared, trying continues
  std::vector<StageBRetryStep> retry_steps;
  // Loop guard v2 (shared rl::LoopGuard, on by default; empty with the guard off or idle).
  int loop_triggers = 0;
  std::vector<std::string> loop_log;
  // TESLA stream authentication and attestation, on by default. Side-only.
  bool tesla_enabled = true;
  bool attest_enabled = true;
  std::vector<uint8_t> tesla_commitment;
  std::vector<uint8_t> tesla_commitment_id;
  uint64_t tesla_steps = 0;
  std::string tesla_err;
  bool attest_computed = false;
  AttestationStatus attest_status = AttestationStatus::Unattested;
  std::string attest_reason;
  // Episodic memory (NEML) receipt, side-only (never feeds the world hash or actions).
  uint32_t neml_receipt_id = 0;
  std::string neml_err;
  size_t neml_events = 0;
  int neml_forks = 0, neml_dead_ends = 0;
};

class StageBDoorMaze final {
 public:
  static constexpr uint32_t kMission = 20;
  static constexpr uint32_t kGoToDoor0 = StageBDoorOracle::kGoToDoor0;
  static constexpr uint32_t kAnswerDoor0 = StageBDoorOracle::kAnswerDoor0;
  static constexpr uint32_t kGoToDoor1 = StageBDoorOracle::kGoToDoor1;
  static constexpr uint32_t kAnswerDoor1 = StageBDoorOracle::kAnswerDoor1;
  static constexpr uint32_t kGoToExit = StageBDoorOracle::kGoToExit;
  static constexpr uint32_t kMethodMission = 30;

  // Unit-fixture answer source (no model). Returns a component name.
  using UnitProposer = const char* (*)(void* ctx, int door, const TraceL1Task& task);
  // Unit-fixture ranked source for the retry path: writes up to max names best-first, returns count.
  using UnitRanker = int (*)(void* ctx, int door, const TraceL1Task& task, int n_failed, const char** out,
                             int max);

  explicit StageBDoorMaze(StageBDoorGrid grid) : grid_(std::move(grid)), seed_(grid_.seed()) {}

  const StageBDoorGrid& grid() const { return grid_; }

  static HtnDomain make_domain() {
    HtnDomain d;
    (void)d.add_task(HtnTask(kMission, "StageBMission", false));
    (void)d.add_task(HtnTask(kGoToDoor0, "GoToDoor0", true));
    (void)d.add_task(HtnTask(kAnswerDoor0, "AnswerDoor0", true));
    (void)d.add_task(HtnTask(kGoToDoor1, "GoToDoor1", true));
    (void)d.add_task(HtnTask(kAnswerDoor1, "AnswerDoor1", true));
    (void)d.add_task(HtnTask(kGoToExit, "GoToExit", true));
    (void)d.add_method(HtnMethod(kMethodMission, "ClearSections", kMission,
                                {kGoToDoor0, kAnswerDoor0, kGoToDoor1, kAnswerDoor1, kGoToExit}));
    return d;
  }

  void set_search_budget(int b) { search_budget_ = b; }
  void set_step_budget(int b) { step_budget_ = b; }
  void enable_mirror_sim(MirrorSimSideLog* log) { mirror_log_ = log; }
  // Opt-in reasoning layer (default nullptr). Official runners never call this.
  void set_rl(rl::MazeRl* r) { rl_ = r; }
  // Episodic memory (NEML) recording, ON by default for model runs (see HeadlessMaze); recording only.
  void set_neml_record(bool on) { neml_record_ = on; }
  void set_neml_record_unit(bool on) { neml_record_unit_ = on; }
  void set_neml_config(const neml::Config* c) { neml_cfg_ = c; }
  // Loop guard v2, the shared reasoning-layer guard (include/pchie/rl/LoopGuard.h), ON by default.
  // It watches navigation moves; the state key is (cell, doors open). Door attempts do not move the player and are
  // not counted (the identical-retry stop and the failed-actuation retry cover repeated attempts).
  // enable_loop_guard(false) (--loop-guard off) restores the old default behavior exactly.
  void enable_loop_guard(bool on) { lg_.set_enabled(on); }
  void enable_tesla(bool on) { stream_auth_.tesla = on; }
  void enable_attest(bool on) { stream_auth_.attest = on; }
  void set_stream_auth(StreamAuthFlags f) { stream_auth_ = f; }
  StreamAuthFlags stream_auth() const { return stream_auth_; }
  bool loop_guard_enabled() const noexcept { return lg_.enabled(); }
  // identical-retry stop (opt-in, default OFF): end a door's attempts after N identical failed
  // attempts in a row (N >= 2; 0 = off). Official runners never call this.
  bool set_identical_retry_stop(int n) { return retry_stop_.set_limit(n); }
  int identical_retry_stop() const noexcept { return retry_stop_.limit(); }
  // Failed-actuation retry (opt-in, default OFF). Official runners never call this.
  // With retry ON, identical-retry stop (if set) is not a hard stop: an identical repeat is the Tabu trigger.
  void enable_failed_actuation_retry(bool on) { retry_on_ = on; }
  bool failed_actuation_retry() const noexcept { return retry_on_; }
  // Max failure-trace tokens injected per pass (newest whole records). Default fits a 128 cache.
  void set_retry_trace_budget(int tokens) { trace_budget_ = tokens > 0 ? tokens : 0; }

  static double maze_value(const StageBDoorGrid& g, uint32_t task_id) {
    const int d = StageBDoorOracle::dist_to_subgoal(g, task_id);
    if (d < 0) return 0.0;
    if (d == 0) return 1.0;
    return 1.0 / (1.0 + double(d));
  }

  bool run_unit(UnitProposer proposer, void* ctx, StageBDoorResult& out) {
    ranker_ = nullptr;
    return run_common(nullptr, proposer, ctx, out);
  }
  // Retry-path unit fixture with ranked candidates (meaningful only with retry ON).
  bool run_unit_ranked(UnitRanker ranker, void* ctx, StageBDoorResult& out) {
    ranker_ = ranker;
    const bool ok = run_common(nullptr, nullptr, ctx, out);
    ranker_ = nullptr;
    return ok;
  }

  // Caller has loaded the model and called init_cache.
  bool run(IModel& model, StageBDoorResult& out) {
    ranker_ = nullptr;
    return run_common(&model, nullptr, nullptr, out);
  }

 private:
  static uint32_t to_q20(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return uint32_t(v * double(PUCT_Q20));
  }

  static void hash_world(const StageBDoorGrid& g, std::vector<uint8_t>& sha, uint64_t& fnv) {
    const auto bytes = g.canonical_bytes();
    sha = Sha256::hash(bytes.data(), bytes.size());
    fnv = MirrorSimSideLog::fnv_bytes(bytes.data(), bytes.size());
  }

  static std::vector<int32_t> token_path(const PuctSearch& S, const PuctPick& p,
                                         const std::vector<int32_t>& prefix) {
    std::vector<int32_t> toks = prefix;
    for (size_t i = 1; i < p.path.size(); ++i) {
      const uint32_t id = p.path[i];
      if (id >= S.nodes.size()) break;
      const PuctNode& nd = S.nodes[id];
      if (nd.parent >= S.nodes.size()) break;
      const PuctNode& par = S.nodes[nd.parent];
      if (nd.edge_in >= par.edges.size()) break;
      toks.push_back(par.edges[nd.edge_in].token);
    }
    return toks;
  }

  static float action_logit(const std::vector<float>& lg, IModel& model, MazeAction a) {
    const char* word = "N";
    switch (a) {
      case MazeAction::North: word = "North"; break;
      case MazeAction::South: word = "South"; break;
      case MazeAction::East: word = "East"; break;
      case MazeAction::West: word = "West"; break;
      default: break;
    }
    const auto ids = model.tokenize_no_bos(word);  // vocab-id lookup, no BOS (BOS policy)
    if (ids.empty()) return 0.0f;
    const int32_t id = ids[0];
    if (id < 0 || size_t(id) >= lg.size()) return 0.0f;
    return lg[size_t(id)];
  }

  // Same scoring as HeadlessMaze::edges_from_actions, over the Stage B grid.
  static std::vector<PuctEdge> edges_from_actions(const std::vector<MazeAction>& acts,
                                                  const StageBDoorGrid& st, uint32_t task_id,
                                                  const std::vector<float>* logits,
                                                  IModel* model, float temp, uint64_t seed) {
    if (acts.empty()) return {};
    std::vector<double> mvs(acts.size(), 0.0);
    std::vector<unsigned char> ok(acts.size(), 0);
    double best_mv = -1.0e300;
    for (size_t i = 0; i < acts.size(); ++i) {
      StageBDoorGrid nxt = st;
      if (nxt.apply(acts[i])) {
        ok[i] = 1;
        mvs[i] = maze_value(nxt, task_id);
        if (mvs[i] > best_mv) best_mv = mvs[i];
      }
    }
    std::vector<int> n_same(acts.size(), 0);
    double min_gap = 0.0;
    bool have_gap = false;
    for (size_t i = 0; i < acts.size(); ++i) {
      if (!ok[i]) continue;
      for (size_t j = 0; j < acts.size(); ++j) {
        if (!ok[j]) continue;
        if (mvs[j] == mvs[i]) ++n_same[i];
        const double gp = mvs[i] > mvs[j] ? mvs[i] - mvs[j] : mvs[j] - mvs[i];
        if (gp > 0.0 && (!have_gap || gp < min_gap)) {
          min_gap = gp;
          have_gap = true;
        }
      }
    }
    const float mix_cap =
        have_gap ? (HeadlessMaze::kMixCapFrac * 16.0f * float(min_gap)) : 0.0f;
    std::vector<float> raws(acts.size(), 0.0f);
    for (size_t i = 0; i < acts.size(); ++i) {
      if (!ok[i] || n_same[i] < 2 || !(logits && model)) continue;
      raws[i] = HeadlessMaze::kLogitMix * action_logit(*logits, *model, acts[i]);
    }
    std::vector<float> lg;
    lg.reserve(acts.size());
    for (size_t i = 0; i < acts.size(); ++i) {
      float s = -1.0e9f;
      if (ok[i]) {
        float glo = 0.0f, ghi = 0.0f;
        bool any = false;
        if (n_same[i] >= 2 && logits && model) {
          for (size_t j = 0; j < acts.size(); ++j) {
            if (!ok[j] || mvs[j] != mvs[i]) continue;
            if (!any || raws[j] < glo) glo = raws[j];
            if (!any || raws[j] > ghi) ghi = raws[j];
            any = true;
          }
        }
        const float logit = (logits && model) ? action_logit(*logits, *model, acts[i]) : 0.0f;
        s = HeadlessMaze::mixed_prior_score(mvs[i], best_mv, logit, n_same[i], mix_cap, glo,
                                              ghi);
      }
      lg.push_back(s);
    }
    if (!(temp > 0.0f)) temp = 1.0f;
    auto raw = PuctSearch::priors_from_logits(lg, int(lg.size()), temp, seed);
    std::vector<PuctEdge> e;
    e.reserve(raw.size());
    for (PuctEdge ed : raw) {
      if (ed.token < 0 || size_t(ed.token) >= acts.size()) continue;
      ed.token = static_cast<int32_t>(acts[size_t(ed.token)]);
      e.push_back(ed);
    }
    std::sort(e.begin(), e.end(),
              [](const PuctEdge& a, const PuctEdge& b) { return a.token < b.token; });
    return e;
  }

  // Same rule as HeadlessMaze::pick_root_action, over the Stage B grid.
  static int32_t pick_root_action(const PuctSearch& S, const StageBDoorGrid& g, uint32_t task_id) {
    if (S.nodes.empty() || S.nodes[0].edges.empty()) return 0;
    const std::vector<PuctEdge>& edges = S.nodes[0].edges;
    int best = -1;
    uint32_t best_n = 0;
    int64_t best_q = 0;
    double best_v = -1.0;
    int32_t best_tok = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
      const PuctEdge& e = edges[i];
      uint32_t n = 0;
      int64_t q = 0;
      if (e.child != 0xFFFFFFFFu && e.child < S.nodes.size()) {
        const PuctNode& c = S.nodes[e.child];
        n = c.n;
        q = c.n ? (c.w / int64_t(c.n)) : 0;
      }
      StageBDoorGrid nxt = g;
      const double v = nxt.apply(static_cast<MazeAction>(e.token)) ? maze_value(nxt, task_id) : -1.0;
      const uint32_t lead = n > best_n ? n - best_n : best_n - n;
      const bool visit_tie = lead <= 1u;
      const bool better =
          best < 0 || (!visit_tie && n > best_n) ||
          (visit_tie && (q > best_q ||
                         (q == best_q && (v > best_v || (v == best_v && e.token < best_tok)))));
      if (better) {
        best = int(i);
        best_n = n;
        best_q = q;
        best_v = v;
        best_tok = e.token;
      }
    }
    return best < 0 ? 0 : edges[size_t(best)].token;
  }

  static const char* task_name(uint32_t id) {
    switch (id) {
      case kGoToDoor0: return "GoToDoor0";
      case kGoToDoor1: return "GoToDoor1";
      case kGoToExit: return "GoToExit";
      default: return "StageBMission";
    }
  }

  bool prime_logits(IModel* model, uint32_t task_id, std::vector<float>& logits, PchStack& stack,
                    std::string& err) {
    logits.clear();
    if (!model) return true;
    const auto ids = model->tokenize(task_name(task_id));
    if (ids.empty()) return true;
    ModelBranch br = model->new_branch();
    for (int32_t t : ids) model->forward(br, t);
    logits = model->logits();
    (void)stack.ncm().propose_from_logits(logits, 8);
    model->release(br);
    if (logits.empty()) {
      err = "no primitive logits";
      return false;
    }
    return true;
  }

  static uint64_t lg_key_(const StageBDoorGrid& g) {
    return (uint64_t(uint32_t(g.x())) << 32) | (uint64_t(uint32_t(g.y())) << 2) | (g.door_open(0) ? 1u : 0u) | (g.door_open(1) ? 2u : 0u);
  }
  // Loop guard reasoning check: a fresh forward on the loop note (same shape as prime_logits).
  bool prime_note_logits(IModel* model, const std::string& text, std::vector<float>& logits, PchStack& stack) {
    logits.clear();
    if (!model) return true;
    const auto ids = model->tokenize(text);
    if (ids.empty()) return true;
    ModelBranch br = model->new_branch();
    for (int32_t t : ids) model->forward(br, t);
    logits = model->logits();
    (void)stack.ncm().propose_from_logits(logits, 8);
    model->release(br);
    return !logits.empty();
  }

  bool expand_state(PchStack& stack, const StageBDoorGrid& st, uint32_t node,
                    const std::vector<float>* logits, IModel* model, std::string& err) {
    if (node >= stack.search().puct().nodes.size()) {
      err = "expand node oob";
      return false;
    }
    auto legal = st.legal_actions();
    if (node == 0)  // loop guard: the forced decision may not turn 180
      legal = lg_.root_filter(legal, [&](MazeAction a, uint64_t& k) {
        StageBDoorGrid c = st;
        if (!c.apply(a)) return false;
        k = lg_key_(c);
        return true;
      });
    if (legal.empty()) {
      err = "no legal actions";
      return false;
    }
    float temp = stack.search().puct().cfg.prior_temp;
    if (!(temp > 0.0f)) temp = 1.0f;
    const uint64_t seed = stack.search().puct().nodes[node].seed;
    auto edges = edges_from_actions(legal, st, stack.search().task_id(), logits, model, temp, seed);
    if (edges.empty()) {
      err = "no expansion edges";
      return false;
    }
    stack.search().expand(node, std::move(edges), nullptr);
    return true;
  }

  int steps_used(const StageBDoorResult& out) const {
    return int(out.actions.size()) + attempts_ + mirror_refusals_;
  }

  // Traits for rl::MazeRl (see HeadlessMaze::rl_traits).
  static std::string rl_obs_text(const StageBDoorGrid& g, uint32_t task_id) {
    int target = -1;  // -1 exit
    const char* tn = "Exit";
    if (task_id == kGoToDoor0) { target = 0; tn = "Door0"; }
    if (task_id == kGoToDoor1) { target = 1; tn = "Door1"; }
    int tx = -1, ty = -1;
    for (int y = 0; y < g.height() && tx < 0; ++y)
      for (int x = 0; x < g.width() && tx < 0; ++x) {
        if (target < 0 ? g.cell(x, y) == StageBCell::Exit : g.door_at(x, y) == target) { tx = x; ty = y; }
      }
    std::string o = "At x=" + std::to_string(g.x()) + " y=" + std::to_string(g.y()) + ". Target " + tn +
                    " at x=" + std::to_string(tx) + " y=" + std::to_string(ty) +
                    (target >= 0 ? " (stand next to it)" : "") +
                    ".\nView (# wall . floor D closed door E exit @ you):\n";
    for (int dy = -2; dy <= 2; ++dy) {
      for (int dx = -2; dx <= 2; ++dx) {
        const int x = g.x() + dx, y = g.y() + dy;
        char c = '#';
        if (dx == 0 && dy == 0) c = '@';
        else switch (g.cell(x, y)) {
          case StageBCell::Floor: case StageBCell::Start: c = '.'; break;
          case StageBCell::Wall: c = '#'; break;
          case StageBCell::Door: c = g.door_open(g.door_at(x, y)) ? '.' : 'D'; break;
          case StageBCell::Exit: c = 'E'; break;
        }
        o.push_back(c);
      }
      o.push_back('\n');
    }
    return o;
  }
  static rl::MazeTraits<StageBDoorGrid> rl_traits() {
    rl::MazeTraits<StageBDoorGrid> t;
    t.legal = [](const StageBDoorGrid& s) { return s.legal_actions(); };
    t.apply = [](StageBDoorGrid& s, MazeAction a) { return s.apply(a); };
    t.goal = [](const StageBDoorGrid& s, uint32_t k) { return StageBDoorOracle::goal_reached(s, k); };
    t.oracle_value = [](const StageBDoorGrid& s, uint32_t k) { return maze_value(s, k); };
    t.official_edges = [](const std::vector<MazeAction>& a, const StageBDoorGrid& s, uint32_t k,
                          const std::vector<float>* lg, IModel* m, float temp, uint64_t seed) {
      return edges_from_actions(a, s, k, lg, m, temp, seed);
    };
    t.official_pick = [](const PuctSearch& S, const StageBDoorGrid& s, uint32_t k) { return pick_root_action(S, s, k); };
    t.obs_text = [](const StageBDoorGrid& s, uint32_t k) { return rl_obs_text(s, k); };
    t.state_key = [](const StageBDoorGrid& s) { return lg_key_(s); };  // loop guard (no unsafe cells here)
    return t;
  }

  // Mirror on or off, one apply. No Stage B unsafe rule: this fixture has no pits.
  bool commit_nav(PchStack& stack, StageBDoorGrid& g, int32_t act, StageBDoorResult& out) {
    const StageBDoorGrid neml_pre = g;  // NEML: committed-step triggers (read-only)
    int32_t neml_top = -1;
    int neml_moves = 0;
    if (!stack.search().puct().nodes.empty()) {
      const auto& re = stack.search().puct().nodes[0].edges;
      neml_top = neml::top_prior_token(re);
      for (const auto& e : re) if (neml::dir_of_action(e.token) != neml::kDirNone) ++neml_moves;
    }
    if (mirror_log_) {
      MirrorSimFork fork;
      hash_world(g, fork.pre_sha, fork.pre_fnv);
      fork.action = act;
      StageBDoorGrid copy = g;
      if (!copy.apply(static_cast<MazeAction>(act))) {
        mirror_log_->append(std::move(fork));
        out.err = "illegal commit";
        return false;
      }
      hash_world(copy, fork.pred_sha, fork.pred_fnv);
      fork.allowed = true;
      fork.rule_fired = "";
      (void)stack.search().commit(nullptr);
      if (!g.apply(static_cast<MazeAction>(act))) {
        out.err = "illegal commit";
        return false;
      }
      hash_world(g, fork.real_post_sha, fork.real_post_fnv);
      mirror_log_->append(std::move(fork));
    } else {
      (void)stack.search().commit(nullptr);
      if (!g.apply(static_cast<MazeAction>(act))) {
        out.err = "illegal commit";
        return false;
      }
    }
    out.actions.push_back(act);
    neml_trk_.step(neml_pre, act, g, neml_top, neml_moves, int(out.actions.size()) - 1, [](const StageBDoorGrid& c) {
      for (int k = 0; k < StageBDoorGrid::kDoors; ++k) if (c.adjacent_to_door(k)) return true;
      return false;
    });
    return true;
  }

  bool drive_nav(PchStack& stack, StageBDoorGrid& g, uint32_t task_id, IModel* model, int cap,
                 StageBDoorResult& out) {
    std::vector<float> logits;
    std::string err;
    if (!prime_logits(model, task_id, logits, stack, err)) {
      out.err = err;
      return false;
    }
    const rl::MazeTraits<StageBDoorGrid> rl_tr = rl_ ? rl_traits() : rl::MazeTraits<StageBDoorGrid>();
    while (!StageBDoorOracle::goal_reached(g, task_id)) {
      if (steps_used(out) >= cap) {
        out.err = "step budget";
        return false;
      }
      uint64_t step_seed =
          NodeSeed::Splitmix64(seed_ ^ (uint64_t(task_id) << 32) ^ uint64_t(out.actions.size()));
      if (rl_ && rl_->engaged()) {  // L1/L2/sealed-tuple decide (opt-in; official runners never set rl_)
        int32_t act = 0;
        const int rbudget = search_budget_ > 0 ? search_budget_ : 16;
        rl_->loop_guard = &lg_;
        if (!rl_->decide<StageBDoorGrid>(stack, g, task_id, out.actions, logits.empty() ? nullptr : &logits,
                                         model, nullptr, rbudget, step_seed, rl_tr, act, err)) {
          out.err = err;
          return false;
        }
        if (!commit_nav(stack, g, act, out)) return false;
        lg_.after_commit(lg_key_(g), out.actions.size(), g.x(), g.y(), task_name(task_id));
        continue;
      }
      std::vector<float> lg_logits;  // loop guard: fresh reasoning-check forward (empty when the guard is idle)
      if (lg_.forcing()) {
        step_seed = lg_.reseed(step_seed);
        (void)prime_note_logits(model, lg_.note(), lg_logits, stack);
      }
      const std::vector<float>* step_lg = (lg_.forcing() && !lg_logits.empty()) ? &lg_logits : (logits.empty() ? nullptr : &logits);
      stack.search().reset(step_seed, task_id);
      if (stack.search().puct().nodes.empty()) {
        out.err = "empty tree";
        return false;
      }
      if (!expand_state(stack, g, 0, step_lg, model, err)) {
        out.err = err;
        return false;
      }
      const int budget = lg_.budget(search_budget_ > 0 ? search_budget_ : 16);  // loop guard reasoning check: 4x
      for (int i = 0; i < budget; ++i) {
        PuctPick p;
        if (!stack.search().select(p, nullptr)) {
          out.err = "select failed";
          return false;
        }
        const auto path = token_path(stack.search().puct(), p, out.actions);
        StageBDoorGrid scratch = g;
        bool legal_path = true;
        for (size_t k = out.actions.size(); k < path.size(); ++k) {
          if (!scratch.apply(static_cast<MazeAction>(path[k]))) {
            legal_path = false;
            break;
          }
        }
        const bool gate_pass = legal_path && stack.leaf().gate().Pass(ProposalTokens::from_proposal(path));  // action proposals only
        const bool term = !legal_path || StageBDoorOracle::goal_reached(scratch, task_id) || p.terminal;
        const double v = gate_pass ? maze_value(scratch, task_id) : 0.0;
        (void)stack.leaf().Evaluate(ProposalTokens::from_proposal(path), logits, uint32_t(path.size()), 0.0, nullptr, 0, term);
        const uint32_t vq = to_q20(v);
        if (term || !gate_pass) {
          stack.search().mark_terminal(p.leaf, vq);
        } else {
          if (p.leaf >= stack.search().puct().nodes.size()) {
            out.err = "leaf oob";
            return false;
          }
          if (!expand_state(stack, scratch, p.leaf, logits.empty() ? nullptr : &logits, model, err))
            stack.search().mark_terminal(p.leaf, vq);
        }
        stack.search().record_leaf(p.leaf, vq, gate_pass, stack.leaf().gate().LastRejectToken(),
                                   nullptr);
        stack.search().backup(p, vq, nullptr);
      }
      const auto& nodes = stack.search().puct().nodes;
      if (nodes.empty() || !nodes[0].expanded || nodes[0].edges.empty()) {
        out.err = "empty commit";
        return false;
      }
      const int32_t act = pick_root_action(stack.search().puct(), g, task_id);
      if (!commit_nav(stack, g, act, out)) return false;
      lg_.after_commit(lg_key_(g), out.actions.size(), g.x(), g.y(), task_name(task_id));
    }
    return true;
  }

  bool drive_door(StageBDoorGrid& g, int door, IModel* model, UnitProposer proposer, void* ctx,
                  int cap, StageBDoorResult& out) {
    retry_stop_.reset();
    while (!g.door_open(door)) {
      if (steps_used(out) >= cap) {
        out.err = "step budget";
        out.stalled_door = door;
        return false;
      }
      StageBDoorAttempt a;
      if (model) {
        std::string err;
        if (!StageBLogicDoor::attempt(*model, g, door, a, err)) {
          out.err = err.empty() ? "door attempt failed" : err;
          return false;
        }
      } else {
        TraceL1Task t;
        if (!StageBDoorKey::task_for_door(door, t)) {
          out.err = "bad door";
          return false;
        }
        // Unit path: no tokens to gate; gate rejection is unit-tested directly.
        const char* comp = proposer ? proposer(ctx, door, t) : nullptr;
        a = StageBLogicDoor::resolve(g, door, t, comp, true);
      }
      ++attempts_;
      ++out.attempts_per_door[door];
      out.attempts.push_back(a);
      // identical-retry stop: OFF by default (observe() returns false and keeps no state).
      if (!a.opened && retry_stop_.observe(attempt_key(a))) {
        out.err = "identical retries (identical-retry stop)";
        out.stalled_door = door;
        out.early_stopped = true;
        return false;
      }
    }
    return true;
  }

  // Failed-actuation retry (opt-in). Decide (inject -> rank -> Tabu mask -> symbolic) then
  // verify + actuate through StageBLogicDoor::resolve exactly as drive_door does. Never ends a
  // door early: only the existing step budget ends it (honest), so a stall becomes tries.
  bool drive_door_retry(StageBDoorGrid& g, int door, IModel* model, UnitProposer proposer, void* ctx,
                        int cap, StageBDoorResult& out) {
    TraceL1Task t;
    if (!StageBDoorKey::task_for_door(door, t)) {
      out.err = "bad door";
      return false;
    }
    TraceL1RetryHarness h;
    FailureTrace trace;
    int n_failed = 0;
    std::string err;
    if (model && !h.prime(*model, t, err)) {
      h.release(*model);
      out.err = err.empty() ? "door prime failed" : err;
      return false;
    }
    bool ok = true;
    while (!g.door_open(door)) {
      if (steps_used(out) >= cap) {
        out.err = "step budget";
        out.stalled_door = door;
        ok = false;
        break;
      }
      std::vector<uint8_t> wsha;
      uint64_t w = 0;
      hash_world(g, wsha, w);
      struct Cand {
        const char* name;
        std::vector<int32_t> path;
      };
      std::vector<Cand> cands;
      StageBRetryStep st;
      st.door = door;
      st.world_fnv = w;
      if (model) {
        const auto ids = trace.ids(size_t(trace_budget_));
        st.trace_tokens = int(ids.size());
        std::vector<TraceL1RetryHarness::Candidate> c;
        if (!h.rank(*model, ids, c, err)) {
          out.err = err.empty() ? "rank failed" : err;
          ok = false;
          break;
        }
        for (auto& x : c) cands.push_back(Cand{x.name, std::move(x.path)});
        st.trace_blocks = int(trace.blocks_fitting(size_t(trace_budget_)));  // records injected
      } else if (ranker_) {
        const char* names[4] = {nullptr, nullptr, nullptr, nullptr};
        int n = ranker_(ctx, door, t, n_failed, names, 4);
        if (n > 4) n = 4;
        for (int i = 0; i < n; ++i)
          if (names[i] && *names[i]) cands.push_back(Cand{names[i], {}});
        st.trace_blocks = n_failed;
      } else if (proposer) {
        const char* p = proposer(ctx, door, t);
        if (p && *p) cands.push_back(Cand{p, {}});
        st.trace_blocks = n_failed;
      }
      const char* pick = nullptr;
      std::vector<int32_t> path;
      if (cands.empty()) {
        st.noop = true;  // freeze / no-op: never resolves to standing still
        ++out.noop_recoveries;
      } else {
        st.neural_top = cands[0].name;
      }
      for (size_t i = 0; i < cands.size(); ++i) {
        if (tabu_.contains(w, door, cands[i].name)) {
          ++out.tabu_masks;
          if (i == 0) st.m7_trigger = true;  // the identical-retry condition, used as the trigger
          continue;
        }
        pick = cands[i].name;
        path = cands[i].path;
        st.source = i == 0 ? (n_failed ? "injected" : "neural") : "tabu_next";
        break;
      }
      if (!pick) {  // symbolic fallback: systematic probe over the component catalog
        const char* catalog[2] = {Dividor::kName, Multiplexor::kName};
        for (const char* nm : catalog)
          if (!tabu_.contains(w, door, nm)) {
            pick = nm;
            break;
          }
        if (pick) {
          ++out.symbolic_fallbacks;
          st.source = "symbolic";
          if (model) path = model->tokenize_no_bos(pick);  // vocab path for the gate
        }
      }
      if (!pick) {  // every candidate failed on this world: clear and keep trying (bounded by cap)
        ++out.exhausted_rounds;
        tabu_.clear_for(w, door);
        st.source = "exhausted";
        out.retry_steps.push_back(st);
        continue;
      }
      bool gate_pass = true;  // unit path: no tokens to gate (as drive_door)
      if (model) {
        PchGate gate;
        gate_pass = gate.Pass(ProposalTokens::from_proposal(path));
      }
      const StageBDoorAttempt a = StageBLogicDoor::resolve(g, door, t, pick, gate_pass);
      ++attempts_;
      ++out.attempts_per_door[door];
      out.attempts.push_back(a);
      st.attempt = out.attempts_per_door[door];
      st.proposed = a.proposed;
      st.opened = a.opened;
      st.reason = a.reason;
      out.retry_steps.push_back(st);
      if (!a.opened) {
        std::vector<uint8_t> sha2;
        uint64_t w2 = 0;
        hash_world(g, sha2, w2);
        (void)tabu_.add(w2, door, pick);
        if (model)
          trace.add_failure(*model, st.attempt, UntrustedText(a.proposed), UntrustedText(a.reason),
                            a.grade.observed, w2 != w);
        ++out.injections;
        ++n_failed;
      }
    }
    if (model) h.release(*model);
    return ok;
  }

  // identical-retry stop identity of one attempt: door, task, proposal, gate verdict, outcome reason. A failed
  // attempt does not change the world, so identical keys mean an identical retry.
  static std::string attempt_key(const StageBDoorAttempt& a) {
    std::string k = std::to_string(a.door);
    k += '|';
    k += a.task_id ? a.task_id : "";
    k += '|';
    k += a.proposed ? a.proposed : "";
    k += '|';
    k += a.gate_pass ? '1' : '0';
    k += '|';
    k += a.reason ? a.reason : "";
    return k;
  }

  void fill(const StageBDoorGrid& g, StageBDoorResult& out) const {
    out.loop_triggers = lg_.triggers();
    out.loop_log = lg_.log();
    out.path_length = int(out.actions.size());
    out.reached_exit = g.at_exit();
    out.win = out.reached_exit;
    for (int k = 0; k < StageBDoorGrid::kDoors; ++k) out.door_open[k] = g.door_open(k);
    out.steps_used = steps_used(out);
    hash_world(g, out.h_world_sha, out.h_world_fnv);
    if (mirror_log_) {
      out.mirror_allowed = mirror_log_->allowed_count();
      out.mirror_refused = mirror_log_->refused_count();
      out.mirror_log_sha = mirror_log_->sha256();
      out.mirror_log_fnv = mirror_log_->fnv();
    }
  }

  // Every run goes through here: the run itself, then the NEML receipt at teardown.
  bool run_common(IModel* model, UnitProposer proposer, void* ctx, StageBDoorResult& out) {
    neml_trk_.begin(grid_);
    const bool ok = run_inner(model, proposer, ctx, out);
    out.neml_events = neml_trk_.rec.all.size();
    out.neml_forks = neml_trk_.rec.forks;
    out.neml_dead_ends = neml_trk_.rec.dead_ends;
    const neml::Config& c = neml_cfg_ ? *neml_cfg_ : neml::process_config();
    if (neml_record_ && c.record && (model || neml_record_unit_)) {
      const auto cb = grid_.canonical_bytes();
      const auto h = Sha256::hash(cb.data(), cb.size());
      neml::ReceiptInfo info;
      for (int i = 7; i >= 0; --i) info.maze_key = (info.maze_key << 8) | h[size_t(i)];
      info.maze_seed = uint32_t(seed_);
      info.maze_w = grid_.width();
      info.maze_h = grid_.height();
      info.path_kind = neml::kPathStageB;
      info.flags = model ? neml::kFlagModel : 0;
      const std::string note = out.ok ? std::string("stage b ok") : ("stage b " + out.err);
      out.neml_receipt_id = neml::write_receipt(c, neml_trk_.rec, int(out.actions.size()),
                                                out.reached_exit || neml_trk_.exit_seen, note, info, &out.neml_err);
    }
    return ok;
  }

  bool run_inner(IModel* model, UnitProposer proposer, void* ctx, StageBDoorResult& out) {
    out = {};
    lg_.reset();
    StreamAuthSession auth_session(stream_auth_);
    if (stream_auth_.tesla) auth_session.start_tesla(seed_);
    struct AuthPublish {
      StreamAuthSession* s;
      StageBDoorResult* o;
      ~AuthPublish() {
        if (s->flags().tesla) s->on_action_seals(o->actions);
        s->publish(*o);
      }
    } auth_publish{&auth_session, &out};
    attempts_ = 0;
    mirror_refusals_ = 0;
    tabu_.clear();
    if (search_budget_ <= 0) {
      out.err = "search budget";
      return false;
    }
    PchStack stack;
    stack.set_goal(kMission);
    stack.set_root_seed(seed_);
    std::string err;
    const HtnDomain domain = make_domain();
    out.plan = stack.planner().plan(domain, "", nullptr, err);
    if (out.plan.empty()) {
      out.err = err.empty() ? "empty plan" : err;
      return false;
    }
    StageBDoorGrid g = grid_;
    const int moves = StageBDoorOracle::shortest_length_all_open(g);
    if (moves < 0) {
      out.err = "unsolvable";
      return false;
    }
    out.bfs_length = moves + StageBDoorGrid::kDoors;
    const int cap =
        step_budget_ > 0 ? step_budget_ : (out.bfs_length * 4 < 32 ? 32 : out.bfs_length * 4);
    out.cap = cap;
    for (const HtnTask& task : out.plan) {
      bool ok = true;
      if (task.id() == kAnswerDoor0)
        ok = retry_on_ ? drive_door_retry(g, 0, model, proposer, ctx, cap, out)
                       : drive_door(g, 0, model, proposer, ctx, cap, out);
      else if (task.id() == kAnswerDoor1)
        ok = retry_on_ ? drive_door_retry(g, 1, model, proposer, ctx, cap, out)
                       : drive_door(g, 1, model, proposer, ctx, cap, out);
      else ok = drive_nav(stack, g, task.id(), model, cap, out);
      if (!ok) {
        fill(g, out);
        return false;
      }
    }
    fill(g, out);
    if (!out.reached_exit) {
      out.err = "exit not reached";
      out.win = false;
      return false;
    }
    out.ok = true;
    return true;
  }

  StageBDoorGrid grid_;
  uint64_t seed_ = 0;
  rl::MazeRl* rl_ = nullptr;
  rl::LoopGuard lg_;  // loop guard v2, on by default (shared with B1, Stage A and the L1/null decide path)  // opt-in reasoning layer
  StreamAuthFlags stream_auth_;  // TESLA + attestation, on by default
  int search_budget_ = 16;
  int step_budget_ = 0;
  int attempts_ = 0;
  int mirror_refusals_ = 0;
  MirrorSimSideLog* mirror_log_ = nullptr;
  IdenticalRetryStop retry_stop_;  // identical-retry stop, OFF by default
  bool retry_on_ = false;          // failed-actuation retry, OFF by default
  int trace_budget_ = 80;          // failure-trace tokens per pass (36-token prompt + 80 < 128)
  UnitRanker ranker_ = nullptr;
  WorldTabu tabu_;
  bool neml_record_ = true, neml_record_unit_ = false;  // NEML recording (model runs), on by default
  const neml::Config* neml_cfg_ = nullptr;
  neml::GridTracker neml_trk_;
};

}  // namespace pchie