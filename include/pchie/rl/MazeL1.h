// L1 model-driven PUCT, L2 query-before-plan, and sealed decision tuples (query-before-plan without a store fails closed).
// DecisionTuple, wired into the grid-maze decide loop (B1 / Stage A / Stage B). Decide only:
// nothing here walks in-engine. Opt-in: a driver takes this path only when set_rl() was given
// an engaged MazeRl. Official runners never call set_rl, so their code path is unchanged.
//
// Honesty rules (plan §3, §4):
//  - With prior=model, the root/leaf priors are Little Nemo action-word logits on a KV fork of a
//    per-decision branch primed with a trusted header (tokenize) + the untrusted observation
//    (tokenize_untrusted). No maze_value / BFS term enters the priors.
//  - With value=model, the leaf value is Little Nemo's Q20 mass on " closer" vs " farther" /
//    " stuck" after the candidate moves. With value=null it is a constant 0.5 (control).
//    Environment terminal outcomes (subgoal reached = 1.0, dead/no legal move = 0.0) are game
//    rules, not distance oracles, and are applied identically to model and null runs.
//  - Every oracle (driver maze_value / pick tie-break) call goes through MazeRl::oracle(), which
//    counts calls and can be poisoned. Under model prior+value the count must stay 0.
//  - Legal-move masking (grid legal_actions) and the PchGate stay in force; a gate-rejected
//    edge gets prior 0 and its leaf is terminal with value 0 (fail closed).
//  - L2 hits are proposals only: a hit raises one root edge's prior after the gate and legality
//    checks pass. It never commits an action directly. No append while a run is live.
//  - Decision tuples are sealed on a scratch side chain (JSONL under scratch/rl/). The
//    official IHashChain is never written by this file except through the unchanged
//    MctsSearch calls the official loop already makes.
#pragma once

#include "pchie/MazeGrid.h"
#include "pchie/IHashChain.h"
#include "pchie/IModel.h"
#include "pchie/ModelBranch.h"
#include "pchie/PchStack.h"
#include "pchie/pchie_mcts.h"
#include "pchie/rl/ForkScore.h"
#include "pchie/rl/LoopGuard.h"
#include "pchie/rl/PlanMemory.h"
#include "pchie/rl/ResearchAdvisor.h"
#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlFlags.h"
#include "pchie/rl/RlSideChain.h"
#include "pchie/rl/UntrustedTokens.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

template <class G>
struct MazeTraits {
  std::function<std::vector<MazeAction>(const G&)> legal;
  std::function<bool(G&, MazeAction)> apply;
  std::function<bool(const G&, uint32_t)> goal;            // env termination, no distance
  std::function<double(const G&, uint32_t)> oracle_value;  // driver maze_value (BFS)
  std::function<std::vector<PuctEdge>(const std::vector<MazeAction>&, const G&, uint32_t,
                                      const std::vector<float>*, IModel*, float, uint64_t)>
      official_edges;                                       // driver edges_from_actions (BFS mix)
  std::function<int32_t(const PuctSearch&, const G&, uint32_t)> official_pick;  // BFS tie-break
  std::function<std::string(const G&, uint32_t)> obs_text;  // untrusted observation text
  std::function<uint64_t(const G&)> state_key;              // loop guard state key (empty: no root taboo)
  std::function<bool(const G&)> unsafe;                     // loop guard cautious landing check (empty: none)
  std::function<ResearchView(const G&)> research_view;      // what he sees, for research advisors (empty: none)
};

struct MazeRlStats {
  uint64_t decisions = 0;
  uint64_t oracle_calls = 0;       // value + prior + pick-tiebreak oracle consultations
  uint64_t model_leaf_evals = 0;   // distinct (decision, path) model evaluations
  uint64_t model_cache_hits = 0;
  uint64_t model_forwards = 0;
  uint64_t gate_fail_closed = 0;
  uint64_t mem_queries = 0, mem_hits = 0, mem_misses = 0, mem_rejected = 0;
  uint64_t tuples_sealed = 0;
  uint32_t max_ctx_used = 0;
  uint64_t tree_digest = 0;        // running mix of per-decision PUCT digests
  uint64_t research_proposals = 0, research_applied = 0, research_rejected = 0, research_capped = 0;
};

inline const char* action_word(MazeAction a) {
  switch (a) {
    case MazeAction::North: return " North";
    case MazeAction::South: return " South";
    case MazeAction::East: return " East";
    case MazeAction::West: return " West";
    case MazeAction::Pickup: return " Pickup";
    case MazeAction::Use: return " Use";
  }
  return " North";
}
inline int action_slot(int32_t tok) { return (tok >= 7 && tok <= 12) ? int(tok - 7) : -1; }

class MazeRl final {
 public:
  RlFlags flags{};
  IModel* model = nullptr;          // real Little Nemo; required for prior/value = model
  bool force_rl_loop = false;       // test-only: run this loop with official sources (fidelity)
  bool poison_oracle = false;       // test-only: oracle returns garbage
  FilePlanMemory* memory = nullptr; // L2 store (scratch/rl/mem/...)
  uint32_t domain_version = 1;
  int32_t mem_boost_q20 = kQ20 / 2; // share of root prior mass moved to a passing memory edge
  std::string tuple_log;            // decision-tuple JSONL (must be under scratch/rl/); empty = chain only
  std::string qbp_log;              // L2 hit/miss JSONL (scratch/rl/)
  std::string run_label = "rl";
  uint32_t max_ctx = 256;
  std::function<void(MazeRl&)> on_decision;  // test hook (e.g. mid-run append attempt)
  LoopGuard* loop_guard = nullptr;           // the driver's guard (set by the driver before each decide)
  IResearchAdvisor* research = nullptr;      // opt-in research layers (--research-*); null = unchanged

  RlSideChain tuple_chain;
  RlSideChain qbp_chain;
  std::vector<DecisionTuple> tuples;
  std::vector<std::array<uint8_t, 32>> seals;
  std::array<uint8_t, 32> weights_hash{};
  MazeRlStats st;
  std::string last_err;

  bool prior_model() const { return flags.puct_prior == PriorSrc::Model; }
  ValueSrc value_src() const { return flags.puct_value; }
  // L3/L4/L5/L6 (and the between-run L2 writer) are not wired into the maze loop in this pass.
  // If any of them is set, the driver still enters decide(), which fails closed instead of
  // silently running the official path (no fake identity result).
  bool unwired_flag_on() const {
    return flags.fork_score || flags.belief || flags.vision_only || flags.options || flags.symbolic_peer ||
           flags.symbolic_weight_q20 != 0 || flags.verifier;
  }
  bool research_on() const { return research && research->engaged(); }
  bool engaged() const {
    return force_rl_loop || prior_model() || value_src() != ValueSrc::Oracle || flags.qbp || unwired_flag_on() ||
           research_on();
  }
  bool needs_model() const { return prior_model() || value_src() == ValueSrc::Model; }

  void begin_run() {
    if (memory) memory->set_run_live(true);
    if (model) {
      const auto w = model->weights_sha256();
      if (w.size() >= 32) std::copy(w.begin(), w.begin() + 32, weights_hash.begin());
    }
  }
  void end_run() {
    if (research_on()) research->end_run();
    if (memory) memory->set_run_live(false);
    if (model && have_hdr_) {
      model->release(hdr_);
      have_hdr_ = false;
    }
  }

  template <class G>
  double oracle(const MazeTraits<G>& tr, const G& g, uint32_t task) {
    ++st.oracle_calls;
    if (poison_oracle) {
      uint64_t z = st.oracle_calls * 0x9E3779B97F4A7C15ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z ^= z >> 31;
      return double(z % 1000ULL) / 1000.0;
    }
    return tr.oracle_value(g, task);
  }

  // Main entry: search + pick for one decision. Does not commit (the driver commits).
  template <class G>
  bool decide(PchStack& stack, const G& g, uint32_t task_id, const std::vector<int32_t>& prefix,
              const std::vector<float>* task_logits, IModel* driver_model, IHashChain* chain,
              int budget, uint64_t step_seed, const MazeTraits<G>& tr, int32_t& act,
              std::string& err) {
    act = 0;
    if (unwired_flag_on()) {
      err = "rl: L3/L4/L5/L6 flag set but not wired into the maze loop (fail closed)";
      return false;
    }
    if (needs_model() && !model) {
      err = "rl: model prior/value requested without a model (fail closed)";
      return false;
    }
    // qbp on without a store used to skip L2 silently (an engaged run with no
    // L2 records). Fail closed instead, so a flags-on L2 result always means L2 actually queried.
    if (flags.qbp && !memory) {
      err = "rl: qbp set without a plan-memory store (fail closed)";
      return false;
    }
    // Loop guard: a forced decision is a reasoning check (fresh seed, 4x budget, loop note in the prompt,
    // no 180 turn at the root); cautious mode checks each landing. Disabled or idle: nothing changes.
    const bool lg_force = loop_guard && loop_guard->forcing();
    const bool lg_careful = loop_guard && (loop_guard->forcing() || loop_guard->cautious()) && bool(tr.unsafe);
    if (loop_guard) {
      step_seed = loop_guard->reseed(step_seed);
      budget = loop_guard->budget(budget);
    }
    stack.search().reset(step_seed, task_id);
    if (stack.search().puct().nodes.empty()) {
      err = "empty tree";
      return false;
    }
    cache_.clear();
    // Observation (untrusted) -> tokens.
    const std::string obs = tr.obs_text(g, task_id);
    std::vector<int32_t> obs_tokens;
    if (model) {
      obs_tokens = model->tokenize_untrusted(obs);
    } else {
      UntrustedTokenizeRequest rq;
      rq.text = obs;
      obs_tokens = tokenize_untrusted(rq).tokens;  // byte-level contract stand-in (no model)
    }
    const bool branch = needs_model();
    float root6[6] = {0, 0, 0, 0, 0, 0};
    if (branch) {
      std::vector<int32_t> prime_tokens = obs_tokens;  // L2 keys and tuple seals keep the plain observation
      if (lg_force && !loop_guard->note().empty()) {
        const auto nt = model->tokenize_untrusted("\n" + loop_guard->note());
        prime_tokens.insert(prime_tokens.end(), nt.begin(), nt.end());
      }
      if (!prime_decision(prime_tokens, err)) return false;
      copy6(root6);
    }
    auto legal = tr.legal(g);
    if (lg_force && tr.state_key)
      legal = loop_guard->root_filter(legal, [&](MazeAction a, uint64_t& k) {
        G c = g;
        if (!tr.apply(c, a)) return false;
        k = tr.state_key(c);
        return true;
      });
    if (lg_careful) legal = careful(tr, g, legal);
    if (legal.empty()) {
      release_dec();
      err = "no legal actions";
      return false;
    }
    float temp = stack.search().puct().cfg.prior_temp;
    if (!(temp > 0.0f)) temp = 1.0f;
    const uint64_t seed0 = stack.search().puct().nodes[0].seed;
    std::vector<PuctEdge> edges;
    if (prior_model()) {
      edges = model_edges(legal, root6, temp, seed0, stack.leaf().gate(), prefix);
    } else {
      st.oracle_calls += legal.size();
      edges = tr.official_edges(legal, g, task_id, task_logits, driver_model, temp, seed0);
    }
    if (edges.empty()) {
      release_dec();
      err = "no expansion edges";
      return false;
    }
    // L2: query before plan. Hit = proposal (prior boost) after gate + legality.
    std::array<uint8_t, 32> sit{};
    if (flags.qbp && memory) l2_query(task_id, obs_tokens, g, tr, prefix, stack.leaf().gate(), edges, sit);
    if (research_on() && !research_adjust(task_id, g, tr, prefix, stack.leaf().gate(), edges, err)) {
      release_dec();
      return false;
    }
    stack.search().expand(0, std::move(edges), chain);

    const std::vector<float> no_logits;
    const std::vector<float>& lg_for_eval = task_logits ? *task_logits : no_logits;
    for (int i = 0; i < budget; ++i) {
      PuctPick p;
      if (!stack.search().select(p, chain)) {
        release_dec();
        err = "select failed";
        return false;
      }
      const auto path = token_path(stack.search().puct(), p, prefix);
      G scratch = g;
      bool legal_path = true;
      std::vector<int32_t> rel;
      for (size_t k = prefix.size(); k < path.size(); ++k) {
        rel.push_back(path[k]);
        if (!tr.apply(scratch, static_cast<MazeAction>(path[k]))) {
          legal_path = false;
          break;
        }
      }
      const bool gate_pass = legal_path && stack.leaf().gate().Pass(ProposalTokens::from_proposal(path));
      const bool reached = legal_path && tr.goal(scratch, task_id);
      const bool term = !legal_path || reached || p.terminal;
      double v = 0.0;
      const Eval* ev = nullptr;
      if (gate_pass) {
        if (value_src() == ValueSrc::Oracle) {
          v = oracle(tr, scratch, task_id);
        } else if (reached) {
          v = 1.0;                       // environment reward: subgoal reached
        } else if (tr.legal(scratch).empty()) {
          v = 0.0;                       // environment: dead state (e.g. pit)
        } else if (value_src() == ValueSrc::Null) {
          v = 0.5;                       // null control
        } else {
          if (!(ev = eval(rel, /*need_value*/true, err))) {
            release_dec();
            return false;
          }
          v = double(ev->vq) / double(PUCT_Q20);
        }
      } else {
        ++st.gate_fail_closed;
      }
      (void)stack.leaf().Evaluate(ProposalTokens::from_proposal(path), lg_for_eval,
                                  uint32_t(path.size()), 0.0, nullptr, 0, term);
      const uint32_t vq = (value_src() != ValueSrc::Oracle && ev) ? ev->vq : to_q20(v);
      if (term || !gate_pass) {
        stack.search().mark_terminal(p.leaf, vq);
      } else {
        auto leg = tr.legal(scratch);
        if (lg_careful) leg = careful(tr, scratch, leg);
        bool expanded = false;
        if (!leg.empty()) {
          const uint64_t sd = stack.search().puct().nodes[p.leaf].seed;
          std::vector<PuctEdge> le;
          if (prior_model()) {
            if (!ev && !(ev = eval(rel, value_src() == ValueSrc::Model, err))) {
              release_dec();
              return false;
            }
            le = model_edges(leg, ev->lg6, temp, sd, stack.leaf().gate(), path);
          } else {
            st.oracle_calls += leg.size();
            le = tr.official_edges(leg, scratch, task_id, task_logits, driver_model, temp, sd);
          }
          if (!le.empty()) {
            stack.search().expand(p.leaf, std::move(le), chain);
            expanded = true;
          }
        }
        if (!expanded) stack.search().mark_terminal(p.leaf, vq);
      }
      stack.search().record_leaf(p.leaf, vq, gate_pass, stack.leaf().gate().LastRejectToken(), chain);
      stack.search().backup(p, vq, chain);
    }
    const auto& nodes = stack.search().puct().nodes;
    if (nodes.empty() || !nodes[0].expanded || nodes[0].edges.empty()) {
      release_dec();
      err = "empty commit";
      return false;
    }
    if (value_src() == ValueSrc::Oracle) {
      st.oracle_calls += nodes[0].edges.size();
      act = tr.official_pick(stack.search().puct(), g, task_id);
    } else {
      act = rl_pick(stack.search().puct());
    }
    const uint64_t d = stack.search().digest();
    st.tree_digest = st.tree_digest ? hash_mix64(st.tree_digest, d) : hash_mix64(FNV_OFFSET, d);
    seal(task_id, obs_tokens, stack.search().puct(), act);
    if (research_on()) {
      G after = g;
      (void)tr.apply(after, static_cast<MazeAction>(act));
      research->decided(act, tr.research_view(after));
    }
    ++st.decisions;
    release_dec();
    if (on_decision) on_decision(*this);
    return true;
  }

  // Research layers: caps first (they only lower priors), then proposals (legal + gate, like an L2 hit).
  template <class G>
  bool research_adjust(uint32_t task_id, const G& g, const MazeTraits<G>& tr, const std::vector<int32_t>& prefix,
                       const PchGate& gate, std::vector<PuctEdge>& edges, std::string& err) {
    if (!tr.research_view) {
      err = "rl: research layers need a view this maze does not provide (fail closed)";
      return false;
    }
    std::vector<int32_t> legal;
    for (const PuctEdge& e : edges) legal.push_back(e.token);
    std::vector<ResearchProposal> props;
    std::vector<ResearchCap> caps;
    if (!research->advise(task_id, tr.research_view(g), legal, props, caps, err)) return false;
    for (const ResearchCap& c : caps)
      for (PuctEdge& e : edges)
        if (e.token == c.token) {
          int64_t s = c.score_q20 < 0 ? 0 : (c.score_q20 > kQ20 ? kQ20 : c.score_q20);
          const uint64_t capped = (uint64_t(e.prior_q20) * uint64_t(s + kQ20)) >> 21;
          if (capped < e.prior_q20) {
            e.prior_q20 = uint32_t(capped);
            ++st.research_capped;
          }
        }
    for (const ResearchProposal& p : props) {
      ++st.research_proposals;
      PuctEdge* hit = nullptr;
      for (PuctEdge& e : edges)
        if (e.token == p.token) hit = &e;
      bool ok = hit != nullptr && hit->prior_q20 > 0;  // a gate-rejected edge (prior 0) stays rejected
      if (ok) {
        G probe = g;
        ok = tr.apply(probe, static_cast<MazeAction>(p.token));
      }
      if (ok) {
        std::vector<int32_t> pp = prefix;
        pp.push_back(p.token);
        ok = gate.Pass(ProposalTokens::from_proposal(pp));
      }
      const int32_t share = p.share_q20 < 0 ? 0 : (p.share_q20 > kQ20 ? kQ20 : p.share_q20);
      if (!ok || share == 0) {
        ++st.research_rejected;
        continue;
      }
      const uint64_t keep = uint64_t(kQ20 - share);
      for (PuctEdge& e : edges) e.prior_q20 = uint32_t((uint64_t(e.prior_q20) * keep) >> 20);
      hit->prior_q20 += uint32_t(share);
      ++st.research_applied;
    }
    return true;
  }

  // Loop guard cautious check: a move whose landing is unsafe is not a candidate while a safe one exists.
  template <class G>
  static std::vector<MazeAction> careful(const MazeTraits<G>& tr, const G& g, const std::vector<MazeAction>& legal) {
    std::vector<MazeAction> safe;
    for (MazeAction a : legal) {
      G c = g;
      if (tr.apply(c, a) && !tr.unsafe(c)) safe.push_back(a);
    }
    return (!safe.empty() && safe.size() < legal.size()) ? safe : legal;
  }

  // Model edges over legal actions. A gate-rejected token keeps prior 0 (fail closed).
  std::vector<PuctEdge> model_edges(const std::vector<MazeAction>& acts, const float* lg6, float temp,
                                    uint64_t seed, const PchGate& gate,
                                    const std::vector<int32_t>& prefix) {
    std::vector<float> lg;
    std::vector<int32_t> toks;
    for (MazeAction a : acts) {
      const int s = action_slot(int32_t(a));
      lg.push_back(s >= 0 ? lg6[s] : -1.0e9f);
      toks.push_back(int32_t(a));
    }
    auto raw = PuctSearch::priors_from_logits(lg, int(lg.size()), temp, seed);
    std::vector<PuctEdge> e;
    for (PuctEdge ed : raw) {
      if (ed.token < 0 || size_t(ed.token) >= toks.size()) continue;
      ed.token = toks[size_t(ed.token)];
      std::vector<int32_t> pp = prefix;
      pp.push_back(ed.token);
      if (!gate.Pass(ProposalTokens::from_proposal(pp))) {
        ed.prior_q20 = 0;
        ++st.gate_fail_closed;
      }
      e.push_back(ed);
    }
    std::sort(e.begin(), e.end(), [](const PuctEdge& a, const PuctEdge& b) { return a.token < b.token; });
    return e;
  }

  // Visits (lead > 1), then Q, then lower token. No oracle term.
  static int32_t rl_pick(const PuctSearch& S) {
    const auto& edges = S.nodes[0].edges;
    int best = -1;
    uint32_t best_n = 0;
    int64_t best_q = 0;
    int32_t best_tok = 0;
    for (size_t i = 0; i < edges.size(); ++i) {
      const PuctEdge& e = edges[i];
      uint32_t n = 0;
      int64_t q = 0;
      if (e.child != 0xFFFFFFFFu && e.child < S.nodes.size()) {
        n = S.nodes[e.child].n;
        q = n ? S.nodes[e.child].w / int64_t(n) : 0;
      }
      const uint32_t lead = n > best_n ? n - best_n : best_n - n;
      const bool tie = lead <= 1u;
      const bool better = best < 0 || (!tie && n > best_n) ||
                          (tie && (q > best_q || (q == best_q && e.token < best_tok)));
      if (better) {
        best = int(i);
        best_n = n;
        best_q = q;
        best_tok = e.token;
      }
    }
    return best < 0 ? 0 : edges[size_t(best)].token;
  }

  // ---- trusted prompt pieces (harness-authored) ----
  static const char* header_text() {
    return "You are Little Nemo, an agent in a grid maze. x grows to the East and y grows to the "
           "South. North means y-1, South means y+1, East means x+1, West means x-1. Pickup takes a "
           "key on your cell. Use opens an adjacent door when you hold the key. You will see your "
           "position, your target and the cells around you, then a list of moves. Answer whether "
           "those moves bring you closer to the target, farther from it, or leave you stuck.\n";
  }
  static const char* moves_text() { return "\nMoves:"; }
  static const char* verdict_text() { return "\nVerdict:"; }

 private:
  struct Eval {
    float lg6[6] = {0, 0, 0, 0, 0, 0};
    uint32_t vq = 0;
    bool has_value = false;
  };

  static uint32_t to_q20(double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    return uint32_t(v * double(PUCT_Q20));
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

  void fwd(ModelBranch& b, int32_t t) {
    model->forward(b, t);
    ++st.model_forwards;
    if (b.n_past > st.max_ctx_used) st.max_ctx_used = b.n_past;
  }

  void ensure_ids() {
    if (ids_ready_) return;
    for (int s = 0; s < 6; ++s) word_ids_[s] = model->tokenize_no_bos(action_word(MazeAction(7 + s)));
    const char* vw[3] = {" closer", " farther", " stuck"};
    for (int k = 0; k < 3; ++k) {
      const auto ids = model->tokenize_no_bos(vw[k]);
      verdict_id_[k] = ids.empty() ? -1 : ids[0];
    }
    moves_ids_ = model->tokenize_no_bos(moves_text());
    verdict_ids_ = model->tokenize_no_bos(verdict_text());
    ids_ready_ = true;
  }

  bool prime_decision(const std::vector<int32_t>& obs_tokens, std::string& err) {
    ensure_ids();
    if (!have_hdr_) {
      const auto h = model->tokenize(header_text());  // trusted framing, one BOS
      hdr_ = model->new_branch();
      for (int32_t t : h) fwd(hdr_, t);
      have_hdr_ = true;
    }
    const uint32_t need = hdr_.n_past + uint32_t(obs_tokens.size() + moves_ids_.size()) + 24;
    if (need > max_ctx) {
      err = "rl: context budget exceeded (fail closed)";
      return false;
    }
    dec_ = model->fork(hdr_);
    have_dec_ = true;
    for (int32_t t : obs_tokens) fwd(dec_, t);
    for (int32_t t : moves_ids_) fwd(dec_, t);
    return true;
  }

  void copy6(float* out) const {
    const auto& lg = model->logits();
    for (int s = 0; s < 6; ++s) {
      const auto& ids = word_ids_[s];
      out[s] = (!ids.empty() && ids[0] >= 0 && size_t(ids[0]) < lg.size()) ? lg[size_t(ids[0])] : -1.0e9f;
    }
  }

  void release_dec() {
    if (have_dec_) {
      model->release(dec_);
      have_dec_ = false;
    }
  }

  const Eval* eval(const std::vector<int32_t>& rel, bool need_value, std::string& err) {
    auto it = cache_.find(rel);
    if (it != cache_.end() && (it->second.has_value || !need_value)) {
      ++st.model_cache_hits;
      return &it->second;
    }
    if (!have_dec_) {
      err = "rl: no decision branch";
      return nullptr;
    }
    uint32_t extra = uint32_t(verdict_ids_.size());
    for (int32_t a : rel) {
      const int s = action_slot(a);
      extra += s >= 0 ? uint32_t(word_ids_[s].size()) : 1u;
    }
    if (dec_.n_past + extra > max_ctx) {
      err = "rl: context budget exceeded at leaf (fail closed)";
      return nullptr;
    }
    ModelBranch b = model->fork(dec_);
    for (int32_t a : rel) {
      const int s = action_slot(a);
      if (s < 0) continue;
      for (int32_t t : word_ids_[s]) fwd(b, t);
    }
    Eval e;
    copy6(e.lg6);
    if (need_value) {
      for (int32_t t : verdict_ids_) fwd(b, t);
      const auto& lg = model->logits();
      double l[3];
      for (int k = 0; k < 3; ++k)
        l[k] = (verdict_id_[k] >= 0 && size_t(verdict_id_[k]) < lg.size()) ? double(lg[size_t(verdict_id_[k])]) : -1.0e9;
      const double m = std::max(l[0], std::max(l[1], l[2]));
      const double e0 = std::exp(l[0] - m), e1 = std::exp(l[1] - m), e2 = std::exp(l[2] - m);
      const double closer = e0 / (e0 + e1 + e2);
      e.vq = to_q20(closer);
      e.has_value = true;
    }
    model->release(b);
    ++st.model_leaf_evals;
    cache_[rel] = e;
    return &cache_[rel];
  }

  template <class G>
  void l2_query(uint32_t task_id, const std::vector<int32_t>& obs_tokens, const G& g,
                const MazeTraits<G>& tr, const std::vector<int32_t>& prefix, const PchGate& gate,
                std::vector<PuctEdge>& edges, std::array<uint8_t, 32>& sit) {
    SituationKey key;
    key.task_id = task_id;
    key.obs_tokens = obs_tokens;
    key.weights_hash = weights_hash;
    key.domain_version = domain_version;
    sit = situation_hash(key);
    ++st.mem_queries;
    const auto row = memory->query_verified(key);
    std::string result = "miss";
    std::string row_hex;
    if (row && !row->plan_fragment.empty()) {
      row_hex = to_hex(row->row_hash);
      const int32_t a = row->plan_fragment[0];
      bool legal_ok = false;
      for (MazeAction la : tr.legal(g)) legal_ok = legal_ok || int32_t(la) == a;
      G probe = g;
      legal_ok = legal_ok && tr.apply(probe, static_cast<MazeAction>(a));
      std::vector<int32_t> pp = prefix;
      pp.push_back(a);
      const bool gate_ok = gate.Pass(ProposalTokens::from_proposal(pp));
      PuctEdge* hit = nullptr;
      for (PuctEdge& e : edges) if (e.token == a) hit = &e;
      if (legal_ok && gate_ok && hit) {
        // Move mem_boost_q20 of the mass onto the memory edge (integer, deterministic).
        const uint64_t keep = uint64_t(kQ20 - mem_boost_q20);
        for (PuctEdge& e : edges) e.prior_q20 = uint32_t((uint64_t(e.prior_q20) * keep) >> 20);
        hit->prior_q20 += uint32_t(mem_boost_q20);
        ++st.mem_hits;
        result = "hit";
      } else {
        ++st.mem_rejected;
        result = gate_ok ? "hit_rejected_illegal" : "hit_rejected_gate";
      }
    } else {
      ++st.mem_misses;
    }
    std::ostringstream o;
    o << "{\"run\":\"" << json_escape(run_label) << "\",\"decision\":" << st.decisions << ",\"task\":" << task_id
      << ",\"situation\":\"" << to_hex(sit) << "\",\"result\":\"" << result << "\",\"row\":\"" << row_hex << "\"}";
    if (!qbp_log.empty()) {
      std::string e2;
      if (!qbp_chain.append_jsonl_and_chain(qbp_log, o.str(), e2)) last_err = e2;
    } else {
      qbp_chain.append_string(o.str());
    }
  }

  void seal(uint32_t task_id, const std::vector<int32_t>& obs_tokens, const PuctSearch& S, int32_t act) {
    const PuctNode& root = S.nodes[0];
    DecisionTuple t;
    std::vector<uint8_t> b;
    append_u32_le(b, task_id);
    append_u32_le(b, uint32_t(obs_tokens.size()));
    for (int32_t x : obs_tokens) append_i32_le(b, x);
    b.insert(b.end(), 32, uint8_t(0));  // belief digest (L4 off)
    t.state_hash = sha256_bytes(b.data(), b.size());
    std::vector<uint32_t> n;
    uint64_t total = 0;
    for (const PuctEdge& e : root.edges) {
      t.options.push_back(e.token);
      const uint32_t c = (e.child != 0xFFFFFFFFu && e.child < S.nodes.size()) ? S.nodes[e.child].n : 0u;
      n.push_back(c);
      total += c;
    }
    int32_t conf = 0;
    for (uint32_t x : n) {
      const int32_t q = total ? int32_t((uint64_t(x) << 20) / total) : 0;
      t.distribution_q20.push_back(q);
      conf = std::max(conf, q);
    }
    t.decision = act;
    t.confidence_q20 = conf;
    t.weights_hash = weights_hash;
    t.blank_control_decision = -1;  // grid maze: no blank-view control
    t.verifier_verdict = 0xFF;      // L6 off
    const auto s = seal_decision_tuple(t);
    std::ostringstream o;
    o << "{\"run\":\"" << json_escape(run_label) << "\",\"decision\":" << st.decisions << ",\"task\":" << task_id
      << ",\"state_hash\":\"" << to_hex(t.state_hash) << "\",\"options\":[";
    for (size_t i = 0; i < t.options.size(); ++i) o << (i ? "," : "") << t.options[i];
    o << "],\"distribution_q20\":[";
    for (size_t i = 0; i < t.distribution_q20.size(); ++i) o << (i ? "," : "") << t.distribution_q20[i];
    o << "],\"decision_token\":" << act << ",\"confidence_q20\":" << conf << ",\"weights_hash\":\""
      << to_hex(t.weights_hash) << "\",\"blank_control\":-1,\"verifier\":255,\"seal\":\"" << to_hex(s) << "\"}";
    if (!tuple_log.empty()) {
      std::string e2;
      if (!tuple_chain.append_jsonl_and_chain(tuple_log, o.str(), e2)) last_err = e2;
    } else {
      tuple_chain.append_string(o.str());
    }
    tuples.push_back(t);
    seals.push_back(s);
    ++st.tuples_sealed;
  }

 private:
  ModelBranch hdr_{};
  ModelBranch dec_{};
  bool have_hdr_ = false;
  bool have_dec_ = false;
  bool ids_ready_ = false;
  std::vector<int32_t> word_ids_[6];
  int32_t verdict_id_[3] = {-1, -1, -1};
  std::vector<int32_t> moves_ids_, verdict_ids_;
  std::map<std::vector<int32_t>, Eval> cache_;
};

// ---- Observation text (untrusted by contract; tokenized only through tokenize_untrusted) ----
inline std::string b1_obs_text(const MazeGrid& g, uint32_t task_id) {
  MazeCell tc = MazeCell::Exit;
  const char* tn = "Exit";
  switch (task_id) {
    case 5: case 6: tc = MazeCell::KeyA; tn = "Key"; break;
    case 7: case 8: tc = MazeCell::DoorA; tn = "Door"; break;
    default: break;
  }
  int tx = -1, ty = -1;
  if (!g.find(tc, tx, ty) && tc == MazeCell::KeyA) { tx = g.x(); ty = g.y(); }
  std::ostringstream o;
  o << "At x=" << g.x() << " y=" << g.y() << ". " << tn << " at x=" << tx << " y=" << ty << ". Key held: "
    << (g.have_key() ? "yes" : "no") << ".\nView (# wall . floor K key D door E exit O pit @ you):\n";
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      const int x = g.x() + dx, y = g.y() + dy;
      char c = '#';
      if (dx == 0 && dy == 0) c = '@';
      else switch (g.cell(x, y)) {
        case MazeCell::Floor: case MazeCell::Start: c = '.'; break;
        case MazeCell::Wall: c = '#'; break;
        case MazeCell::KeyA: c = 'K'; break;
        case MazeCell::DoorA: c = g.door_open() ? '.' : 'D'; break;
        case MazeCell::Exit: c = 'E'; break;
        case MazeCell::Pit: c = 'O'; break;
      }
      o << c;
    }
    o << '\n';
  }
  return o.str();
}

}  // namespace rl
}  // namespace pchie
