// Bolt PCHIE B1 - headless maze driver built on PchStack.
// Concrete final. Serial, deterministic, no mutex, no atomic.
// Flat-world grid only. Does not write .lldemo (deferred to the in-engine pilot). Does not lock a golden.
// MCTS expands legal maze actions (not vocab tokens). Leaf value is gated maze-progress
// (LogProbLeafEvaluator at beta=0 would be a constant 0.5 and would not seek the exit).
// Real-weight path: one Little Nemo forward per HTN primitive for action priors + NCM;
// rollouts stay on the grid. Official chain is append-only via the planner and MctsSearch.
#pragma once

#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/MirrorSimGate.h"
#include <unordered_map>
#include <set>
#include <cstdio>
#include "pchie/MirrorSimSideLog.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnMethod.h"
#include "pchie/HtnTask.h"
#include "pchie/ChainEpoch.h"
#include "pchie/ChainRecord.h"
#include "pchie/IHashChain.h"
#include "pchie/StreamAuth.h"
#include "pchie/IModel.h"
#include "pchie/ModelBranch.h"
#include "pchie/NcmConcept.h"
#include "pchie/NodeSeed.h"
#include "pchie/PchStack.h"
#include "pchie/Sha256.h"
#include "pchie/rl/MazeL1.h"  // opt-in; official runners never call set_rl()
#include "pchie/neml/NemlQuery.h"  // episodic memory: receipts at teardown, opt-in memory prior
#include "pchie/neml/NemlTrack.h"
#include "pchie/pchie_mcts.h"
#include "pchie/pchie_tree.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pchie {

struct MazeResult {
  std::vector<HtnTask> plan;
  std::vector<int32_t> actions;
  int path_length = 0;
  int bfs_length = 0;
  bool reached_exit = false;
  bool within_oracle = false;
  bool ok = false;
  std::string err;
  uint64_t h_world_fnv = 0;
  std::vector<uint8_t> h_world_sha;
  size_t chain_length = 0;
  std::vector<NcmConcept> last_ncm;
  bool budget_exhausted = false;
  int max_nodes = 0;
  int ticks_used = 0;
  int exhaust_records = 0;
  // B9 Mirror-Sim diagnostics. Filled only when the gate is enabled.
  // Sim/refusal records are not official chain records.
  int mirror_allowed = 0;
  int mirror_refused = 0;
  bool mirror_stalled = false;
  std::vector<uint8_t> mirror_log_sha;
  uint64_t mirror_log_fnv = 0;
  int bfs_length_avoiding_pits = 0;
  // Scratch tiptoe crossing check (opt-in, default off; zero when off).
  int tiptoe_checks = 0;        // search nodes at a crossing where a checked pit step was not taken
  int tiptoe_crossings = 0;     // committed steps taken from a pit-adjacent cell
  int tiptoe_footing_ok = 0;    // ... that ended on safe footing (verified after the step)
  int tiptoe_footing_fail = 0;  // committed steps that ended in a pit (must stay 0)
  // Loop guard v2 (shared rl::LoopGuard, on by default; empty with the guard off or idle).
  int loop_triggers = 0;
  std::vector<std::string> loop_log;
  // TESLA stream authentication and attestation, on by default. Side-only: these
  // fields never feed back into the official chain, h_world, or actions.
  bool tesla_enabled = true;
  bool attest_enabled = true;
  std::vector<uint8_t> tesla_commitment;
  std::vector<uint8_t> tesla_commitment_id;
  uint64_t tesla_steps = 0;
  std::string tesla_err;
  bool attest_computed = false;
  AttestationStatus attest_status = AttestationStatus::Unattested;
  std::string attest_reason;
  // Episodic memory (NEML). Side-only: recording never feeds the chain, h_world, or actions.
  uint32_t neml_receipt_id = 0;          // receipt written at teardown (0 = none; see neml_err)
  std::string neml_err;
  size_t neml_events = 0;                // triggers fired this run
  int neml_forks = 0, neml_dead_ends = 0;
  std::vector<uint8_t> neml_snapshot_sha;  // ledger snapshot this run read (empty = memory prior off)
  int neml_memory_nudges = 0;            // expansions whose priors the memory changed
};

class HeadlessMaze final {
 public:
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

  explicit HeadlessMaze(uint64_t seed, int width = MazeGrid::kDefaultWidth,
                          int height = MazeGrid::kDefaultHeight)
      : grid_(seed, width, height), seed_(seed) {}

  // Wraps an existing grid (labeled unit-test fixture only, e.g. closed_box).
  explicit HeadlessMaze(MazeGrid grid) : grid_(std::move(grid)), seed_(grid_.seed()) {}

  const MazeGrid& grid() const { return grid_; }
  uint64_t seed() const noexcept { return seed_; }

  static HtnDomain make_domain() {
    HtnDomain d;
    (void)d.add_task(HtnTask(kMission, "Mission", false));
    (void)d.add_task(HtnTask(kGetKeyA, "GetKeyA", false));
    (void)d.add_task(HtnTask(kOpenDoorA, "OpenDoorA", false));
    (void)d.add_task(HtnTask(kGoToExit, "GoToExit", true));
    (void)d.add_task(HtnTask(kGoToKeyA, "GoToKeyA", true));
    (void)d.add_task(HtnTask(kPickUpKeyA, "PickUpKeyA", true));
    (void)d.add_task(HtnTask(kGoToDoorA, "GoToDoorA", true));
    (void)d.add_task(HtnTask(kUseKeyA, "UseKeyA", true));
    (void)d.add_method(HtnMethod(kMethodMission, "ClearMaze", kMission,
                                {kGetKeyA, kOpenDoorA, kGoToExit}));
    (void)d.add_method(HtnMethod(kMethodGetKeyA, "FetchKeyA", kGetKeyA,
                                {kGoToKeyA, kPickUpKeyA}));
    (void)d.add_method(HtnMethod(kMethodOpenDoorA, "UnlockDoorA", kOpenDoorA,
                                {kGoToDoorA, kUseKeyA}));
    return d;
  }

  void set_search_budget(int b) { search_budget_ = b; }
  void set_step_budget(int b) { step_budget_ = b; }
  // B7: 0 means unlimited (current default). A cap never allows a search or
  // run to exceed it; exhaustion appends ChainRecordKind::BudgetExhausted and
  // still commits the best-so-far action. Tick budget is the deterministic
  // stand-in for wall-clock time (one tick per MCTS iteration started).
  void set_node_budget(int b) { node_budget_ = b; }
  void set_tick_budget(int b) { tick_budget_ = b; }
  void set_token_budget(int b) { token_budget_ = b; }
  int search_budget() const noexcept { return search_budget_; }
  int step_budget() const noexcept { return step_budget_; }
  int node_budget() const noexcept { return node_budget_; }
  int tick_budget() const noexcept { return tick_budget_; }
  int token_budget() const noexcept { return token_budget_; }

  // Optional B5 hook (nullptr = current behavior, no snapshot, no hash change).
  // After mid-rollout select, if the check returns true, the in-flight PUCT
  // tree and official chain are restored to the pre-select snapshot and the
  // iteration is retried once. iter_done fires only after a completed backup.
  using FreezeCheck = bool (*)(void* ctx, uint32_t task_id, int step, int iter);
  using IterDone = void (*)(void* ctx, uint32_t task_id, int step, int iter);
  void set_freeze_check(FreezeCheck fn, void* ctx = nullptr) {
    freeze_check_ = fn;
    freeze_ctx_ = ctx;
  }
  void set_iter_done(IterDone fn, void* ctx = nullptr) {
    iter_done_ = fn;
    if (ctx) freeze_ctx_ = ctx;
  }

  // B9 Mirror-Sim Action Gate (opt-in). Default is off so official B1/B4-B7
  // hashes cannot move. When enabled, only the action chosen at commit time
  // is simulated on a grid copy; rollouts are not gated. Default rule is
  // pit_trap. Sim/refusal records go to `log` (not the official chain).
  void enable_mirror_sim(MirrorSimSideLog* log) {
    mirror_log_ = log;
    mirror_gate_.use_default_pit_trap();
  }
  bool mirror_sim_enabled() const noexcept { return mirror_log_ != nullptr; }
  // Scratch 'tiptoe' crossing check. At a danger crossing (a state with a
  // pit next to it) Little Nemo CHECKS each candidate step on a grid copy before it is considered: a step that
  // would land in a pit is never taken (in the search or at commit), so a pit beside the lane no longer drags the
  // whole forward branch down and Nemo stops turning 180 at the bridge. It then steps one block (one committed
  // action, as always) and VERIFIES safe footing afterwards. Priors, model logits, values and the PUCT pick are
  // unchanged: Nemo still chooses; no BFS path is forced. Deliberately departs from the rule that rollouts are not gated
  // only when the flag is on. With the flag off nothing below runs.
  void enable_tiptoe(bool on) { tiptoe_ = on; }
  // Loop guard v2, the shared reasoning-layer guard in include/pchie/rl/LoopGuard.h. ON by default.
  // The state key is (cell, key, door). On the 4th entry of a state the next decision is a forced reasoning check
  // (loop note forward, fresh seed, 4x budget, no 180 turn), and the pit landing check stays on for the rest of the
  // run. enable_loop_guard(false) (--loop-guard off) restores the old default behavior exactly.
  void enable_loop_guard(bool on) { lg_.set_enabled(on); }
  bool loop_guard_enabled() const noexcept { return lg_.enabled(); }
  // TESLA stream authentication and hardware attestation, ON by default.
  // --tesla off / --attest off turn each one off for testing. TESLA wraps the official chain only as
  // a side stream (MAC after append). Attestation in this process never talks to the TPM: without a
  // quote the recorded verdict is Unattested, never Attested.
  void enable_tesla(bool on) { stream_auth_.tesla = on; }
  void enable_attest(bool on) { stream_auth_.attest = on; }
  void set_stream_auth(StreamAuthFlags f) { stream_auth_ = f; }
  StreamAuthFlags stream_auth() const { return stream_auth_; }
  // Opt-in reasoning layer. Default null. Official runners never call this.
  // An engaged MazeRl replaces the search below. It uses the same loop guard (through rl_traits' state key and
  // pit check), and tiptoe still records committed steps when it is enabled.
  void set_rl(rl::MazeRl* r) { rl_ = r; }
  // Episodic memory (NEML), include/pchie/neml. Recording is ON by default: every run with Little Nemo writes one
  // receipt at teardown (process switch neml::process_config().record, PCHIE_NEML=0 / --neml off turns it off).
  // Recording only watches committed steps; it never changes a decision or a hash.
  void set_neml_record(bool on) { neml_record_ = on; }            // per-maze off switch
  void set_neml_record_unit(bool on) { neml_record_unit_ = on; }  // tests: also record model-free fixture runs
  void set_neml_config(const neml::Config* c) { neml_cfg_ = c; }  // tests: a private ledger instead of the process one
  // Memory prior (opt-in per run; official runners and golden tests never call this). Root and tree expansions at a
  // cell divide the prior of a move that led into a recorded dead end from that cell (this maze only), so a
  // remembered dead end is tried later. The snapshot is hashed into the chain at run start (provenance), so a
  // memory-informed run's hashes differ from an official run's by design. The snapshot must outlive the run.
  void set_memory(const neml::Snapshot* s) { mem_ = s; }
  uint64_t maze_key() const { return neml::maze_key_of(grid_.canonical_bytes()); }
  bool tiptoe_enabled() const noexcept { return tiptoe_; }
  static bool pit_adjacent(const MazeGrid& g, int x, int y) {
    return g.is_pit(x, y - 1) || g.is_pit(x, y + 1) || g.is_pit(x + 1, y) || g.is_pit(x - 1, y);
  }
  // Rule registry for reviewer-judged extra rules (test rules in unit tests only).
  MirrorSimGate& mirror_gate() { return mirror_gate_; }

  // Fix 2: the 0.05 logit mix may only break exact maze_value ties. Threshold:
  // kMazeValueTieEps = 0, i.e. exact equality of post-action maze_value, which
  // is exact equality of integer dist_to_subgoal. A 1-step gap at distance d
  // is ?mv = 2/(d(d+2)) and is always a different integer d, so it can never
  // be flipped. A nonzero progress band is not used, because that gap shrinks
  // without bound as d grows and would eventually let a 1-step gap through.
  // The mix is a within-group tie-break only (n_same_mv >= 2). A unique
  // best-progress action is not mixed, so a large negative logit cannot drop
  // it below an unmixed worse action. When other maze_value groups exist, the
  // mix term is affinely mapped into [0, kMixCapFrac * min 16*?mv] so it
  // cannot cross a real progress gap (kMixCapFrac = 1/4).
  static constexpr double kMazeValueTieEps = 0.0;
  static constexpr float kLogitMix = 0.05f;
  static constexpr float kMixCapFrac = 0.25f;

  static double maze_value(const MazeGrid& g, uint32_t task_id) {
    const int d = MazeOracle::dist_to_subgoal(g, task_id);
    if (d < 0) return 0.0;
    if (d == 0) return 1.0;
    return 1.0 / (1.0 + double(d));
  }

  static bool logit_mix_allowed(double mv, double best_mv) {
    const double d = mv > best_mv ? mv - best_mv : best_mv - mv;
    return d <= kMazeValueTieEps;
  }

  // n_same_mv is the number of legal actions that share this action's
  // maze_value. Mix is applied only when that count is at least 2 (a real
  // tie). mix_cap > 0 caps the tie-break into [0, mix_cap] so it cannot
  // outweigh a progress gap; mix_cap <= 0 leaves the raw 0.05*logit term
  // (safe when every legal action shares the same maze_value).
  static float mixed_prior_score(double mv, double best_mv, float logit, int n_same_mv = 1,
                                 float mix_cap = 0.0f, float group_raw_lo = 0.0f,
                                 float group_raw_hi = 0.0f) {
    (void)best_mv;
    float s = float(mv) * 16.0f;
    if (n_same_mv < 2) return s;
    const float raw = kLogitMix * logit;
    if (mix_cap > 0.0f) {
      // Cap is active because other maze_value groups exist. Equal logits
      // in this group need no tie-break; do not add the raw term, which
      // could otherwise shift the whole group across a progress gap.
      if (group_raw_hi > group_raw_lo)
        s += mix_cap * (raw - group_raw_lo) / (group_raw_hi - group_raw_lo);
    } else {
      s += raw;
    }
    return s;
  }

  // Most visited at the root; a visit lead of 1 or less is a tie and falls
  // through to Q, then maze_value, then lower action id.
  static int32_t pick_root_action(const PuctSearch& S, const MazeGrid& g, uint32_t task_id) {
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
      MazeGrid nxt = g;
      const double v = nxt.apply(static_cast<MazeAction>(e.token)) ? maze_value(nxt, task_id) : -1.0;
      // Fix 1: a visit lead of 1 or less is a tie; fall through to Q, then
      // maze_value, then lower action id.
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

  // Labeled unit-test fixture path: no IModel, uniform action priors, grid leaf values.
  bool run_unit(const std::string& world_text, IHashChain* chain, MazeResult& out) {
    return run_common(nullptr, world_text, chain, out);
  }

  // Caller has loaded the model and called init_cache. One forward per HTN primitive.
  bool run(IModel& model, const std::string& world_text, IHashChain* chain, MazeResult& out) {
    return run_common(&model, world_text, chain, out);
  }

 private:
  static uint32_t to_q20(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return uint32_t(v * double(PUCT_Q20));
  }

  static uint64_t fnv_bytes(const uint8_t* p, size_t n) {
    uint64_t h = FNV_OFFSET;
    constexpr uint64_t P = 1099511628211ULL;
    for (size_t i = 0; i < n; ++i) {
      h ^= uint64_t(p[i]);
      h *= P;
    }
    return h;
  }

  static bool within_1_2(int path, int bfs) {
    if (bfs <= 0 || path < bfs) return false;
    return path * 5 <= bfs * 6;
  }

  static bool subgoal_done(const MazeGrid& g, uint32_t task_id) {
    return MazeOracle::dist_to_subgoal(g, task_id) == 0;
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

  // Heuristic score is maze progress. The 0.05 logit mix is applied only when
  // maze_value ties exactly (kMazeValueTieEps = 0); it must not flip a 1-step
  // gap (see kMazeValueTieEps).
  static std::vector<PuctEdge> edges_from_actions(const std::vector<MazeAction>& acts,
                                                  const MazeGrid& st, uint32_t task_id,
                                                  const std::vector<float>* logits,
                                                  IModel* model, float temp, uint64_t seed) {
    if (acts.empty()) return {};
    std::vector<double> mvs(acts.size(), 0.0);
    std::vector<unsigned char> ok(acts.size(), 0);
    double best_mv = -1.0e300;
    for (size_t i = 0; i < acts.size(); ++i) {
      MazeGrid nxt = st;
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
        const double g = mvs[i] > mvs[j] ? mvs[i] - mvs[j] : mvs[j] - mvs[i];
        if (g > 0.0 && (!have_gap || g < min_gap)) {
          min_gap = g;
          have_gap = true;
        }
      }
    }
    const float mix_cap = have_gap ? (kMixCapFrac * 16.0f * float(min_gap)) : 0.0f;
    std::vector<float> raws(acts.size(), 0.0f);
    for (size_t i = 0; i < acts.size(); ++i) {
      if (!ok[i] || n_same[i] < 2 || !(logits && model)) continue;
      raws[i] = kLogitMix * action_logit(*logits, *model, acts[i]);
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
        s = mixed_prior_score(mvs[i], best_mv, logit, n_same[i], mix_cap, glo, ghi);
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

  static float action_logit(const std::vector<float>& lg, IModel& model, MazeAction a) {
    const char* word = "N";
    switch (a) {
      case MazeAction::North: word = "North"; break;
      case MazeAction::South: word = "South"; break;
      case MazeAction::East: word = "East"; break;
      case MazeAction::West: word = "West"; break;
      case MazeAction::Pickup: word = "Pickup"; break;
      case MazeAction::Use: word = "Use"; break;
    }
    const auto ids = model.tokenize_no_bos(word);  // vocab-id lookup, no BOS (BOS policy)
    if (ids.empty()) return 0.0f;
    const int32_t id = ids[0];
    if (id < 0 || size_t(id) >= lg.size()) return 0.0f;
    return lg[size_t(id)];
  }

  static const char* task_name(uint32_t id) {
    switch (id) {
      case kGoToKeyA: return "GoToKeyA";
      case kPickUpKeyA: return "PickUpKeyA";
      case kGoToDoorA: return "GoToDoorA";
      case kUseKeyA: return "UseKeyA";
      case kGoToExit: return "GoToExit";
      default: return "Mission";
    }
  }

  bool prime_logits(IModel* model, uint32_t task_id, std::vector<float>& logits,
                    PchStack& stack, std::string& err) {
    logits.clear();
    if (!model) return true;
    const auto ids = model->tokenize(task_name(task_id));
    if (ids.empty()) return true;  // uniform priors
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

  bool expand_state(PchStack& stack, const MazeGrid& st, uint32_t node,
                    const std::vector<float>* logits, IModel* model, IHashChain* chain,
                    std::string& err) {
    if (node >= stack.search().puct().nodes.size()) {
      err = "expand node oob";
      return false;
    }
    auto legal = st.legal_actions();
    if (node == 0)  // loop guard: the forced decision may not turn 180
      legal = lg_.root_filter(legal, [&](MazeAction a, uint64_t& k) {
        MazeGrid c = st;
        if (!c.apply(a)) return false;
        k = lg_key_(c);
        return true;
      });
    if ((tiptoe_ || lg_.forcing() || lg_.cautious()) && !legal.empty()) {  // tiptoe: check each step's landing cell; a pit step is not a candidate
      std::vector<MazeAction> safe;
      safe.reserve(legal.size());
      for (MazeAction a : legal) {
        MazeGrid c = st;
        if (c.apply(a) && !c.in_pit()) safe.push_back(a);
      }
      if (!safe.empty() && safe.size() < legal.size()) {
        ++tiptoe_checks_;
        legal = std::move(safe);
      }
    }
    if (legal.empty()) {
      err = "no legal actions";
      return false;
    }
    float temp = stack.search().puct().cfg.prior_temp;
    if (!(temp > 0.0f)) temp = 1.0f;
    const uint64_t seed = stack.search().puct().nodes[node].seed;
    auto edges = edges_from_actions(legal, st, stack.search().task_id(), logits, model, temp, seed);
    mem_adjust_(edges, st);  // no-op unless set_memory()
    if (edges.empty()) {
      err = "no expansion edges";
      return false;
    }
    stack.search().expand(node, std::move(edges), chain);
    return true;
  }

  static void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
    b.push_back(uint8_t(v >> 16));
    b.push_back(uint8_t(v >> 24));
  }

  static constexpr uint32_t kBudgetNodes = 1;
  static constexpr uint32_t kBudgetTicks = 2;
  static constexpr uint32_t kBudgetTokens = 3;

  void append_budget_exhausted(IHashChain* chain, MazeResult& out, uint32_t reason,
                               uint32_t limit, uint32_t used) {
    out.budget_exhausted = true;
    ++out.exhaust_records;
    if (!chain) return;
    std::vector<uint8_t> p;
    put_u32(p, reason);
    put_u32(p, limit);
    put_u32(p, used);
    chain->append(ChainRecord{ChainRecordKind::BudgetExhausted, std::move(p)});
  }

  void note_nodes(const PchStack& stack, MazeResult& out) {
    const int n = int(stack.search().puct().nodes.size());
    if (n > out.max_nodes) out.max_nodes = n;
  }

  static bool rl_goal(const MazeGrid& g, uint32_t task_id) {
    switch (task_id) {
      case kGoToKeyA: return g.on_key() || g.have_key();
      case kPickUpKeyA: return g.have_key();
      case kGoToDoorA: return g.adjacent_to_door() || g.door_open();
      case kUseKeyA: return g.door_open();
      case kGoToExit: return g.at_exit();
      default: return g.at_exit();
    }
  }
  rl::MazeTraits<MazeGrid> rl_traits() const {
    rl::MazeTraits<MazeGrid> t;
    t.legal = [](const MazeGrid& s) { return s.legal_actions(); };
    t.apply = [](MazeGrid& s, MazeAction a) { return s.apply(a); };
    t.goal = [](const MazeGrid& s, uint32_t k) { return rl_goal(s, k); };
    t.oracle_value = [](const MazeGrid& s, uint32_t k) { return maze_value(s, k); };
    t.official_edges = [this](const std::vector<MazeAction>& a, const MazeGrid& s, uint32_t k,
                              const std::vector<float>* lg, IModel* m, float temp, uint64_t seed) {
      auto e = edges_from_actions(a, s, k, lg, m, temp, seed);
      mem_adjust_(e, s);  // no-op unless set_memory()
      return e;
    };
    t.official_pick = [](const PuctSearch& S, const MazeGrid& s, uint32_t k) { return pick_root_action(S, s, k); };
    t.obs_text = [](const MazeGrid& s, uint32_t k) { return rl::b1_obs_text(s, k); };
    t.state_key = [](const MazeGrid& s) { return lg_key_(s); };  // loop guard
    t.unsafe = [](const MazeGrid& s) { return s.in_pit(); };     // loop guard cautious landing check
    return t;
  }

  enum class CommitStep { Applied, Refused, Failed };

  // One commit tail. preset_act is the reasoning-layer choice; null picks at the
  // same point the official path always did (after Commit when the mirror is off,
  // before the sim when it is on) so B1/B4-B7 chain order stays put.
  CommitStep commit_chosen(PchStack& stack, MazeGrid& g, uint32_t task_id, const int32_t* preset_act,
                           IHashChain* chain, int cap, int& task_step, MazeResult& out) {
    auto empty_commit_fatal = [&]() -> bool {
      const auto& nodes = stack.search().puct().nodes;
      return nodes.empty() || !nodes[0].expanded || nodes[0].edges.empty();
    };
    const MazeGrid neml_pre = g;  // NEML: committed-step triggers (read-only)
    int32_t neml_top = -1;
    int neml_moves = 0;
    if (!stack.search().puct().nodes.empty()) {
      const auto& re = stack.search().puct().nodes[0].edges;
      neml_top = neml::top_prior_token(re);
      for (const auto& e : re) if (neml::dir_of_action(e.token) != neml::kDirNone) ++neml_moves;
    }
    if (mirror_log_) {
      if (empty_commit_fatal()) {
        out.err = "empty commit";
        return CommitStep::Failed;
      }
      const int32_t act = preset_act ? *preset_act : pick_root_action(stack.search().puct(), g, task_id);
      const int tt_pre_x = g.x(), tt_pre_y = g.y();
      // Commit-time Mirror-Sim only. Rollouts / root-child expansion above
      // are deliberately not gated.
      const auto dec = mirror_gate_.simulate(g, static_cast<MazeAction>(act));
      MirrorSimFork fork;
      fork.pre_sha = dec.pre_sha;
      fork.pre_fnv = dec.pre_fnv;
      fork.action = act;
      fork.pred_sha = dec.pred_sha;
      fork.pred_fnv = dec.pred_fnv;
      fork.allowed = dec.allowed;
      fork.rule_fired = dec.rule_fired ? dec.rule_fired : "";
      if (!dec.sim_applied) {
        mirror_log_->append(std::move(fork));
        out.err = "illegal commit";
        return CommitStep::Failed;
      }
      if (!dec.allowed) {
        // Refused: must not reach the real world, and must not write an
        // official Commit. Honest stall if the same action is picked again.
        mirror_log_->append(std::move(fork));
        ++mirror_refusals_;
        if (int(out.actions.size()) + mirror_refusals_ >= cap) {
          out.err = "step budget";
          out.mirror_stalled = true;
          return CommitStep::Failed;
        }
        return CommitStep::Refused;
      }
      const auto committed = stack.search().commit(chain);
      (void)committed;
      if (!g.apply(static_cast<MazeAction>(act))) {
        out.err = "illegal commit";
        return CommitStep::Failed;
      }
      MirrorSimSideLog::hash_world(g, fork.real_post_sha, fork.real_post_fnv);
      mirror_log_->append(std::move(fork));
      out.actions.push_back(act);
      ++task_step;
      neml_step_(neml_pre, act, g, neml_top, neml_moves, int(out.actions.size()) - 1);
      if (tiptoe_) tiptoe_verify_(tt_pre_x, tt_pre_y, g);
      lg_.after_commit(lg_key_(g), out.actions.size(), g.x(), g.y(), task_name(task_id));
      return CommitStep::Applied;
    }
    const auto committed = stack.search().commit(chain);
    if (committed.empty() && empty_commit_fatal()) {
      out.err = "empty commit";
      return CommitStep::Failed;
    }
    const int32_t act = preset_act ? *preset_act : pick_root_action(stack.search().puct(), g, task_id);
    const int tt_pre_x = g.x(), tt_pre_y = g.y();
    if (!g.apply(static_cast<MazeAction>(act))) {
      out.err = "illegal commit";
      return CommitStep::Failed;
    }
    out.actions.push_back(act);
    ++task_step;
    neml_step_(neml_pre, act, g, neml_top, neml_moves, int(out.actions.size()) - 1);
    if (tiptoe_) tiptoe_verify_(tt_pre_x, tt_pre_y, g);
    lg_.after_commit(lg_key_(g), out.actions.size(), g.x(), g.y(), task_name(task_id));
    return CommitStep::Applied;
  }

  bool drive_task(PchStack& stack, MazeGrid& g, uint32_t task_id, IModel* model,
                  IHashChain* chain, int cap, MazeResult& out) {
    std::vector<float> logits;
    std::string err;
    if (!prime_logits(model, task_id, logits, stack, err)) {
      out.err = err;
      return false;
    }
    if (!logits.empty()) out.last_ncm = stack.ncm().candidates();

    int task_step = 0;
    while (!subgoal_done(g, task_id)) {
      if (token_budget_ > 0 && int(out.actions.size()) >= token_budget_) {
        append_budget_exhausted(chain, out, kBudgetTokens, uint32_t(token_budget_),
                                uint32_t(out.actions.size()));
        return true;
      }
      if (tick_budget_ > 0 && ticks_used_ >= tick_budget_) {
        append_budget_exhausted(chain, out, kBudgetTicks, uint32_t(tick_budget_),
                                uint32_t(ticks_used_));
        return true;
      }
      if (int(out.actions.size()) + (mirror_log_ ? mirror_refusals_ : 0) >= cap) {
        out.err = "step budget";
        if (mirror_log_ && mirror_refusals_ > 0) out.mirror_stalled = true;
        return false;
      }
      uint64_t step_seed =
          NodeSeed::Splitmix64(seed_ ^ (uint64_t(task_id) << 32) ^ uint64_t(out.actions.size()));
      if (rl_ && rl_->engaged()) {
        int32_t act = 0;
        const int rbudget = search_budget_ > 0 ? search_budget_ : 16;
        const rl::MazeTraits<MazeGrid> rl_tr = rl_traits();
        rl_->loop_guard = &lg_;
        if (!rl_->decide<MazeGrid>(stack, g, task_id, out.actions, logits.empty() ? nullptr : &logits,
                                     model, chain, rbudget, step_seed, rl_tr, act, err)) {
          out.err = err;
          return false;
        }
        note_nodes(stack, out);
        const CommitStep stepped = commit_chosen(stack, g, task_id, &act, chain, cap, task_step, out);
        if (stepped == CommitStep::Failed) return false;
        if (stepped == CommitStep::Refused) continue;
      } else {
        std::vector<float> lg_logits;  // loop guard: fresh reasoning-check forward (empty when the guard is idle)
        if (lg_.forcing()) {
          step_seed = lg_.reseed(step_seed);
          if (model) { std::string e2; (void)prime_logits_text(model, lg_.note(), lg_logits, stack, e2); }
        }
        const std::vector<float>* step_lg = (lg_.forcing() && !lg_logits.empty()) ? &lg_logits : (logits.empty() ? nullptr : &logits);
        stack.search().reset(step_seed, task_id);
        if (stack.search().puct().nodes.empty()) {
          out.err = "empty tree";
          return false;
        }
        if (!expand_state(stack, g, 0, step_lg, model, chain, err)) {
          out.err = err;
          return false;
        }
        note_nodes(stack, out);
        int budget = search_budget_ > 0 ? search_budget_ : 16;
        budget = lg_.budget(budget);  // loop guard reasoning check: 4x
        for (int i = 0; i < budget; ++i) {
          if (tick_budget_ > 0 && ticks_used_ >= tick_budget_) {
            // Do not append here: commit the best-so-far action first, then
            // record exhaustion once after apply so we never double-append.
            break;
          }
          if (node_budget_ > 0 && int(stack.search().puct().nodes.size()) >= node_budget_) {
            append_budget_exhausted(chain, out, kBudgetNodes, uint32_t(node_budget_),
                                    uint32_t(stack.search().puct().nodes.size()));
            break;
          }
          if (tick_budget_ > 0) ++ticks_used_;
          bool retried = false;
          PuctPick p;
          for (;;) {
          std::vector<PuctNode> nodes_snap;
          std::vector<uint8_t> head_snap;
          size_t len_snap = 0;
          if (freeze_check_) {
            nodes_snap = stack.search().puct().nodes;
            if (chain) {
              head_snap = chain->head();
              len_snap = chain->length();
            }
          }
          p = PuctPick{};
          if (!stack.search().select(p, chain)) {
            out.err = "select failed";
            return false;
          }
          if (freeze_check_ && !retried &&
              freeze_check_(freeze_ctx_, task_id, task_step, i)) {
            stack.search().puct().nodes = std::move(nodes_snap);
            if (chain) chain->restore_head(head_snap, len_snap);
            // Resume point: a ratcheted chain opens a new epoch here; other chains are left as they were.
            reroot_if_ratcheted(chain, "resume", "b1.freeze-retry",
                                {uint64_t(task_id), uint64_t(int64_t(task_step)), uint64_t(int64_t(i))});
            retried = true;
            continue;
          }
          const auto path = token_path(stack.search().puct(), p, out.actions);
          MazeGrid scratch = g;
          bool legal_path = true;
          // Apply only the newly selected actions (after the already committed prefix).
          for (size_t k = out.actions.size(); k < path.size(); ++k) {
            if (!scratch.apply(static_cast<MazeAction>(path[k]))) {
              legal_path = false;
              break;
            }
          }
          const bool gate_pass = legal_path && stack.leaf().gate().Pass(ProposalTokens::from_proposal(path));  // action proposals only
          const bool term = !legal_path || subgoal_done(scratch, task_id) || p.terminal;
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
            if (!expand_state(stack, scratch, p.leaf, step_lg, model,
                              chain, err)) {
              // A leaf with no legal actions is terminal.
              stack.search().mark_terminal(p.leaf, vq);
            }
          }
          stack.search().record_leaf(p.leaf, vq, gate_pass, stack.leaf().gate().LastRejectToken(),
                                     chain);
          stack.search().backup(p, vq, chain);
          if (iter_done_) iter_done_(freeze_ctx_, task_id, task_step, i);
          break;
          }  // retry-once loop
          note_nodes(stack, out);
        }
        note_nodes(stack, out);
        // With the gate off, commit_chosen writes Commit first, then apply, so
        // B1/B4-B7 hashes cannot move. With the gate on it simulates first and
        // writes Commit only for an allowed action.
        const CommitStep stepped = commit_chosen(stack, g, task_id, nullptr, chain, cap, task_step, out);
        if (stepped == CommitStep::Failed) return false;
        if (stepped == CommitStep::Refused) continue;
      }
      if (token_budget_ > 0 && int(out.actions.size()) >= token_budget_) {
        append_budget_exhausted(chain, out, kBudgetTokens, uint32_t(token_budget_),
                                uint32_t(out.actions.size()));
        return true;
      }
      if (tick_budget_ > 0 && ticks_used_ >= tick_budget_) {
        append_budget_exhausted(chain, out, kBudgetTicks, uint32_t(tick_budget_),
                                uint32_t(ticks_used_));
        return true;
      }
    }
    return true;
  }

  // Every run goes through here: the run itself, then the NEML receipt at teardown.
  bool run_common(IModel* model, const std::string& world_text, IHashChain* chain, MazeResult& out) {
    neml_trk_.begin(grid_);
    mem_nudges_ = 0;
    mem_key_ = mem_ ? maze_key() : 0;
    const bool ok = run_inner(model, world_text, chain, out);
    if (mem_) out.neml_snapshot_sha = mem_->sha256;
    out.neml_memory_nudges = mem_nudges_;
    out.neml_events = neml_trk_.rec.all.size();
    out.neml_forks = neml_trk_.rec.forks;
    out.neml_dead_ends = neml_trk_.rec.dead_ends;
    neml_write_(model != nullptr, out);
    return ok;
  }

  bool run_inner(IModel* model, const std::string& world_text, IHashChain* chain, MazeResult& out) {
    out = {};
    ticks_used_ = 0;
    mirror_refusals_ = 0;
    tiptoe_checks_ = tiptoe_crossings_ = tiptoe_footing_ok_ = tiptoe_footing_fail_ = 0;
    lg_.reset();
    StreamAuthSession auth_session(stream_auth_);
    TeslaWrappingChain tesla_wrap;
    if (stream_auth_.tesla) auth_session.start_tesla(seed_);
    if (stream_auth_.tesla && chain) {
      tesla_wrap.bind(chain, &auth_session);
      chain = &tesla_wrap;
    }
    struct AuthPublish {
      StreamAuthSession* s;
      MazeResult* o;
      ~AuthPublish() { s->publish(*o); }
    } auth_publish{&auth_session, &out};
    if (mem_ && chain) {  // provenance: the ledger snapshot this run's decisions read (memory-informed runs only)
      std::vector<uint8_t> p{'n', 'e', 'm', 'l', '.', 's', 'n', 'a', 'p', '.', 'v', '1'};
      p.insert(p.end(), mem_->sha256.begin(), mem_->sha256.end());
      chain->append(ChainRecord{ChainRecordKind::Custom, std::move(p)});
    }
    if (search_budget_ <= 0) {
      out.err = "search budget";
      return false;
    }
    PchStack stack;
    stack.set_goal(kMission);
    stack.set_root_seed(seed_);
    std::string err;
    const HtnDomain domain = make_domain();
    out.plan = stack.planner().plan(domain, world_text, chain, err);
    if (out.plan.empty()) {
      out.err = err.empty() ? "empty plan" : err;
      return false;
    }
    MazeGrid g = grid_;
    out.bfs_length = MazeOracle::shortest_length(g);
    out.bfs_length_avoiding_pits = MazeOracle::shortest_length_avoiding_pits(g);
    if (out.bfs_length < 0) {
      out.err = "unsolvable";
      return false;
    }
    const int cap = step_budget_ > 0 ? step_budget_ : (out.bfs_length * 4 < 32 ? 32 : out.bfs_length * 4);
    for (const HtnTask& task : out.plan) {
      if (!drive_task(stack, g, task.id(), model, chain, cap, out)) {
        if (mirror_log_) fill_world_result(g, chain, out);
        out.loop_triggers = lg_.triggers();  // report-only; 0 and empty with --loop-guard off
        out.loop_log = lg_.log();
        return false;
      }
      if (out.budget_exhausted &&
          ((token_budget_ > 0 && int(out.actions.size()) >= token_budget_) ||
           (tick_budget_ > 0 && ticks_used_ >= tick_budget_)))
        break;
    }
    fill_world_result(g, chain, out);
    if (!out.reached_exit) {
      out.err = "exit not reached";
      return false;
    }
    if (!out.within_oracle) {
      out.err = "path longer than 1.2 BFS";
      return false;
    }
    out.ok = true;
    return true;
  }

  void mem_adjust_(std::vector<PuctEdge>& e, const MazeGrid& st) const {
    if (!mem_) return;
    const neml::Query q(*mem_);
    if (neml::apply_memory_prior(e, q.junction(mem_key_, st.x(), st.y()), neml::dir_of_action)) ++mem_nudges_;
  }
  void neml_step_(const MazeGrid& pre, int32_t act, const MazeGrid& post, int32_t top, int moves, int step) {
    neml_trk_.step(pre, act, post, top, moves, step, [](const MazeGrid& c) {
      return c.cell(c.x(), c.y()) == MazeCell::KeyA || c.adjacent_to_door() || c.in_pit();
    });
  }
  void neml_write_(bool model_run, MazeResult& out) {
    const neml::Config& c = neml_cfg_ ? *neml_cfg_ : neml::process_config();
    if (!neml_record_ || !c.record || !(model_run || neml_record_unit_)) return;
    neml::ReceiptInfo info;
    info.maze_key = maze_key();
    info.maze_seed = uint32_t(seed_);
    info.maze_w = grid_.width();
    info.maze_h = grid_.height();
    info.path_kind = neml::kPathMaze;
    info.flags = uint8_t((model_run ? neml::kFlagModel : 0) | (mem_ ? neml::kFlagMemoryPrior : 0));
    const std::string note = out.ok ? std::string("maze ok") : ("maze " + out.err);
    out.neml_receipt_id = neml::write_receipt(c, neml_trk_.rec, int(out.actions.size()),
                                              out.reached_exit || neml_trk_.exit_seen, note, info, &out.neml_err);
  }

  static uint64_t lg_key_(const MazeGrid& g) {
    return (uint64_t(uint32_t(g.x())) << 32) | (uint64_t(uint32_t(g.y())) << 2) | (g.have_key() ? 1u : 0u) | (g.door_open() ? 2u : 0u);
  }
  bool prime_logits_text(IModel* model, const std::string& text, std::vector<float>& logits, PchStack& stack,
                         std::string& err) {
    logits.clear();
    if (!model) return true;
    const auto ids = model->tokenize(text);
    if (ids.empty()) return true;
    ModelBranch br = model->new_branch();
    for (int32_t t : ids) model->forward(br, t);
    logits = model->logits();
    (void)stack.ncm().propose_from_logits(logits, 8);
    model->release(br);
    if (logits.empty()) err = "no loop-check logits";
    return !logits.empty();
  }
  void tiptoe_verify_(int pre_x, int pre_y, const MazeGrid& g) {  // verify footing after one careful step
    if (g.in_pit()) { ++tiptoe_footing_fail_; return; }
    if (pit_adjacent(g, pre_x, pre_y)) { ++tiptoe_crossings_; ++tiptoe_footing_ok_; }
  }

  void fill_world_result(const MazeGrid& g, IHashChain* chain, MazeResult& out) {
    out.tiptoe_checks = tiptoe_checks_;
    out.tiptoe_crossings = tiptoe_crossings_;
    out.tiptoe_footing_ok = tiptoe_footing_ok_;
    out.tiptoe_footing_fail = tiptoe_footing_fail_;
    out.loop_triggers = lg_.triggers();
    out.loop_log = lg_.log();
    out.path_length = int(out.actions.size());
    out.reached_exit = g.at_exit();
    out.within_oracle = out.reached_exit && within_1_2(out.path_length, out.bfs_length);
    const auto bytes = g.canonical_bytes();
    if (!bytes.empty()) {
      out.h_world_sha = Sha256::hash(bytes.data(), bytes.size());
      out.h_world_fnv = fnv_bytes(bytes.data(), bytes.size());
    }
    if (chain) out.chain_length = chain->length();
    out.ticks_used = ticks_used_;
    if (mirror_log_) {
      out.mirror_allowed = mirror_log_->allowed_count();
      out.mirror_refused = mirror_log_->refused_count();
      out.mirror_log_sha = mirror_log_->sha256();
      out.mirror_log_fnv = mirror_log_->fnv();
    }
  }

  MazeGrid grid_;
  uint64_t seed_ = 0;
  int search_budget_ = 16;
  int step_budget_ = 0;
  int node_budget_ = 0;
  int tick_budget_ = 0;
  int token_budget_ = 0;
  int ticks_used_ = 0;
  FreezeCheck freeze_check_ = nullptr;
  IterDone iter_done_ = nullptr;
  void* freeze_ctx_ = nullptr;
  MirrorSimGate mirror_gate_;
  MirrorSimSideLog* mirror_log_ = nullptr;
  int mirror_refusals_ = 0;
  rl::MazeRl* rl_ = nullptr;  // opt-in reasoning layer
  bool tiptoe_ = false;  // scratch tiptoe crossing check (opt-in)
  int tiptoe_checks_ = 0, tiptoe_crossings_ = 0, tiptoe_footing_ok_ = 0, tiptoe_footing_fail_ = 0;
  rl::LoopGuard lg_;  // loop guard v2, on by default (shared with Stage B and the L1/null decide path)
  StreamAuthFlags stream_auth_;  // TESLA + attestation, on by default
  // NEML episodic memory: recording on by default (model runs), memory prior opt-in.
  bool neml_record_ = true, neml_record_unit_ = false;
  const neml::Config* neml_cfg_ = nullptr;
  neml::GridTracker neml_trk_;
  const neml::Snapshot* mem_ = nullptr;
  uint64_t mem_key_ = 0;
  mutable int mem_nudges_ = 0;
};

}  // namespace pchie
