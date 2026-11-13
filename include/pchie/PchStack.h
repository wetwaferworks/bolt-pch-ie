// Bolt PCHIE - PchStack: one coordinator-driven pass that wires
// HtnPlanner, NcmCandidateGenerator, MctsSearch, PchCpuLeafPath (PchGate),
// and DualPool step-back. Concrete final. Serial, deterministic.
// No mutex, no atomic. Official chain records are coordinator-only
// (planner + MctsSearch). DualPool rollback uses a scratch chain so completed
// official records are never rewound. The original AcceptAllGate
// program is not used and is not edited.
// , opt-in only (set_span_mode): NCM concepts, including multi-token
// spans, become PUCT edges through NcmSpanTrie, and HTN concept tasks bound by
// NcmMethodBinder become searches forced along the concept span. With span mode off
// (the default, and every official runner) the code path is the original path, unchanged.
// KV moves go through MctsJoinedStepBack::kv_reposition / kv_snapshot
// (KV only, same page operations as MctsDualPoolStepBack). An opt-in discard hook
// (set_discard_hook) exercises the joined checkpoint/step_back: tree + KV + scratch KV
// chain are restored together and the iteration is redone once. Scratch chain only
// Scratch chain only: the official chain is never rewound; records a discarded
// iteration appended stay on it. No official runner sets the hook.
#pragma once

#include "pchie/Fnv1a64HashChain.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnPlanner.h"
#include "pchie/HtnState.h"
#include "pchie/HtnTask.h"
#include "pchie/IHashChain.h"
#include "pchie/IModel.h"
#include "pchie/MctsJoinedStepBack.h"
#include "pchie/MctsSearch.h"
#include "pchie/ModelBranch.h"
#include "pchie/NcmCandidateGenerator.h"
#include "pchie/NcmConcept.h"
#include "pchie/NcmMethodBinder.h"
#include "pchie/NcmSpanTrie.h"
#include "pchie/PchCpuLeafPath.h"
#include "pchie/TurnBoundaryTerminalPolicy.h"
#include "pchie/pchie_mcts.h"
#include "pchie/pchie_tree.h"
#include "pchie/rl/RlFlags.h"  // flags default off; an engaged flag fail-closes until a driver calls set_rl

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

struct PchStackResult {
  std::vector<HtnTask> plan;
  std::vector<int32_t> committed;
  uint64_t puct_digest = 0;
  size_t chain_length = 0;
  bool ok = false;
  std::string err;
  uint64_t gate_rejects = 0;
  bool last_gate_pass = true;
  double last_value = 0.0;
  std::vector<NcmConcept> last_ncm;
  uint32_t last_task_id = 0;
  // the HTN and NCM extensions span-mode counters (always 0 with span mode off).
  size_t span_expansions = 0;     // node expansions taken from the NcmSpanTrie
  size_t concept_tasks = 0;       // HTN concept tasks searched along their span
  size_t span_stop_dropped = 0;   // longest-span policy drops (see NcmSpanTrie)
  size_t step_backs = 0;          // joined step-backs taken (discard hook)
};

class PchStack final {
 public:
  explicit PchStack(PchCpuLeafPath leaf = PchCpuLeafPath(), int budget = 2, int prior_k = 8)
      : leaf_(std::move(leaf)), budget_(budget), prior_k_(prior_k) {}

  void set_goal(uint32_t id) { planner_.set_goal(id); }
  void set_budget(int b) { budget_ = b; }
  void set_prior_k(int k) { prior_k_ = k; }
  void set_root_seed(uint64_t s) { root_seed_ = s; }
  // the HTN and NCM extensions (opt-in). root spans are extra NCM concepts (e.g. NcmSpanProposer output)
  // merged with the model's 1-token candidates at every task root. The binder (not owned,
  // must outlive run) resolves HTN concept tasks to their spans.
  void set_span_mode(bool on) { span_mode_ = on; }
  // opt-in: plan with preconditions/priority against a trusted start
  // state (HtnPlanner::plan_with_state). Off by default: the legacy plan() path is used.
  void set_plan_state(HtnState start) {
    plan_state_ = std::move(start);
    use_plan_state_ = true;
  }
  void clear_plan_state() { use_plan_state_ = false; plan_state_ = HtnState(); }
  void set_root_spans(std::vector<NcmConcept> spans) { root_spans_ = std::move(spans); }
  void set_concept_binder(const NcmMethodBinder* b) { binder_ = b; }
  bool span_mode() const noexcept { return span_mode_; }
  // opt-in, model path only: called after the leaf forward and before leaf
  // evaluation; returning true discards that iteration via the joined step_back and redoes
  // it once. Unset (default) => no checkpoint is ever taken. The step_back restores tree + KV
  // + scratch KV chain only; official chain records stay (append-only).
  using DiscardHook = bool (*)(void* ctx, uint32_t task_id, int iteration);
  void set_discard_hook(DiscardHook h, void* ctx) {
    discard_hook_ = h;
    discard_ctx_ = ctx;
  }

  // RL flags (opt-in, default off). Official runners never call this.
  // The live decide path is MazeRl::set_rl on the maze driver. Engaged flags here
  // fail closed: this stack does not apply them.
  void set_rl_flags(const rl::RlFlags& f) { rl_flags_ = f; }
  const rl::RlFlags& rl_flags() const noexcept { return rl_flags_; }
  bool rl_idle() const { return rl::official_runner_clean(rl_flags_); }

  int budget() const noexcept { return budget_; }
  int prior_k() const noexcept { return prior_k_; }
  uint64_t root_seed() const noexcept { return root_seed_; }

  HtnPlanner& planner() { return planner_; }
  const HtnPlanner& planner() const { return planner_; }
  NcmCandidateGenerator& ncm() { return ncm_; }
  const NcmCandidateGenerator& ncm() const { return ncm_; }
  MctsSearch& search() { return search_; }
  const MctsSearch& search() const { return search_; }
  PchCpuLeafPath& leaf() { return leaf_; }
  const PchCpuLeafPath& leaf() const { return leaf_; }

  // Full pass. Caller has already forwarded `prompt` on `root`. `root` is not
  // mutated or released. world_text is untrusted and is ignored by the planner.
  // root_logits / root_hidden must be the prompt-root vectors (copied before any
  // later forward). model.logits() is overwritten by leaf forwards and must not
  // be reused as the next run's root.
  bool run(IModel& model, const ModelBranch& root, const std::vector<int32_t>& prompt,
           const std::vector<float>& root_logits, const std::vector<float>& root_hidden,
           const HtnDomain& domain, const std::string& world_text, IHashChain* chain,
           PchStackResult& out) {
    out = {};
    if (!rl_idle()) {  // engaged RL flags are not applied on this stack
      out.err = "rl layer engaged but not wired (scratch)";
      return false;
    }
    if (!prepare_plan(domain, world_text, chain, out)) return false;
    if (root_logits.empty()) {
      out.err = "no root logits";
      return false;
    }

    Fnv1a64HashChain kv_chain;
    kv_chain.reset();
    ModelBranch work{};
    bool have_work = false;
    bool ok = false;
    {
      MctsJoinedStepBack step(model.pool_swa(), model.pool_glob());
      work = model.fork(root);
      have_work = true;
      step.kv_snapshot(0, work, kv_chain);
      ok = true;
      for (const HtnTask& task : out.plan) {
        if (!step.kv_reposition(0, work, kv_chain)) {
          out.err = "root rollback failed";
          ok = false;
          break;
        }
        if (!simulate_task(task.id(), root_logits, root_hidden, prompt, &model, &work, &step,
                           &kv_chain, chain, out)) {
          ok = false;
          break;
        }
      }
    }
    if (have_work) model.release(work);
    if (!ok) return false;
    if (chain) out.chain_length = chain->length();
    out.ok = true;
    return true;
  }

  // Labeled unit-test fixture only: no IModel, no DualPool, synthetic logits.
  bool run_unit_fixture(const HtnDomain& domain, const std::string& world_text,
                        const std::vector<float>& root_logits, const std::vector<int32_t>& prompt,
                        IHashChain* chain, PchStackResult& out) {
    out = {};
    if (!rl_idle()) {  // engaged RL flags are not applied on this stack
      out.err = "rl layer engaged but not wired (scratch)";
      return false;
    }
    if (!prepare_plan(domain, world_text, chain, out)) return false;
    if (root_logits.empty()) {
      out.err = "no root logits";
      return false;
    }
    const std::vector<float> no_hidden;
    for (const HtnTask& task : out.plan) {
      if (!simulate_task(task.id(), root_logits, no_hidden, prompt, nullptr, nullptr, nullptr,
                         nullptr, chain, out)) {
        return false;
      }
    }
    if (chain) out.chain_length = chain->length();
    out.ok = true;
    return true;
  }

 private:
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

  // 1-token NCM concepts become PUCT edges. Priors are softmax over the compact
  // candidate scores at prior_temp (then remapped back to token ids). Multi-token
  // concepts are skipped here.
  static std::vector<PuctEdge> edges_from_ncm(const NcmCandidateGenerator& ncm, float temp,
                                              uint64_t seed) {
    const auto& cs = ncm.candidates();
    std::vector<float> lg;
    std::vector<int32_t> toks;
    lg.reserve(cs.size());
    toks.reserve(cs.size());
    for (const auto& c : cs) {
      if (c.tokens().size() != 1) continue;
      lg.push_back(static_cast<float>(c.score()));
      toks.push_back(c.tokens()[0]);
    }
    if (lg.empty() || temp <= 0.0f) return {};
    const auto raw = PuctSearch::priors_from_logits(lg, int(lg.size()), temp, seed);
    std::vector<PuctEdge> e;
    e.reserve(raw.size());
    for (PuctEdge ed : raw) {
      if (ed.token < 0 || size_t(ed.token) >= toks.size()) continue;
      ed.token = toks[size_t(ed.token)];
      e.push_back(ed);
    }
    std::sort(e.begin(), e.end(),
              [](const PuctEdge& a, const PuctEdge& b) { return a.token < b.token; });
    return e;
  }

  bool prepare_plan(const HtnDomain& domain, const std::string& world_text, IHashChain* chain,
                    PchStackResult& out) {
    if (budget_ <= 0) {
      out.err = "budget";
      return false;
    }
    if (prior_k_ <= 0) {
      out.err = "prior_k";
      return false;
    }
    std::string err;
    out.plan = use_plan_state_ ? planner_.plan_with_state(domain, plan_state_, world_text, chain, err)
                               : planner_.plan(domain, world_text, chain, err);
    if (out.plan.empty()) {
      out.err = err.empty() ? "empty plan" : err;
      return false;
    }
    return true;
  }

  bool expand_node(uint32_t node, const std::vector<float>& logits, uint64_t seed,
                   IHashChain* chain, PchStackResult& out) {
    if (node >= search_.puct().nodes.size()) {
      out.err = "expand node oob";
      return false;
    }
    (void)ncm_.propose_from_logits(logits, prior_k_);
    float temp = search_.puct().cfg.prior_temp;
    if (!(temp > 0.0f)) temp = 1.0f;
    auto edges = edges_from_ncm(ncm_, temp, seed);
    if (edges.empty()) {
      out.err = "no expansion edges";
      return false;
    }
    search_.expand(node, std::move(edges), chain);
    out.last_ncm = ncm_.candidates();
    return true;
  }

  // task-root expansion in span mode.
  bool expand_span_root(uint32_t task_id, const std::vector<float>& root_logits, uint64_t seed,
                        IHashChain* chain, PchStackResult& out) {
    float temp = search_.puct().cfg.prior_temp;
    if (!(temp > 0.0f)) temp = 1.0f;
    std::vector<NcmConcept> cands;
    const NcmConcept* bound = binder_ ? binder_->concept_for_task(task_id) : nullptr;
    if (NcmMethodBinder::is_concept_task(task_id)) {
      if (!bound) {
        out.err = "unbound concept task";
        return false;
      }
      cands.push_back(*bound);
      ++out.concept_tasks;
    } else {
      (void)ncm_.propose_from_logits(root_logits, prior_k_);
      cands = ncm_.candidates();
      cands.insert(cands.end(), root_spans_.begin(), root_spans_.end());
    }
    if (!trie_.build(cands, temp, seed)) {
      out.err = "span trie build failed";
      return false;
    }
    out.span_stop_dropped += trie_.stop_mass_dropped();
    auto edges = trie_.edges_at({});
    if (edges.empty()) {
      out.err = "no expansion edges";
      return false;
    }
    search_.expand(0, std::move(edges), chain);
    out.last_ncm = std::move(cands);
    return true;
  }

  bool simulate_task(uint32_t task_id, const std::vector<float>& root_logits,
                     const std::vector<float>& root_hidden, const std::vector<int32_t>& prompt,
                     IModel* model, ModelBranch* work, MctsJoinedStepBack* step,
                     IHashChain* kv_chain, IHashChain* chain, PchStackResult& out) {
    search_.reset(root_seed_, task_id);
    out.last_task_id = task_id;
    if (search_.puct().nodes.empty()) {
      out.err = "empty tree";
      return false;
    }
    if (span_mode_) {
      if (!expand_span_root(task_id, root_logits, search_.puct().nodes[0].seed, chain, out)) return false;
    } else {
      if (NcmMethodBinder::is_concept_task(task_id)) {
        out.err = "concept task needs span mode";
        return false;
      }
      if (!expand_node(0, root_logits, search_.puct().nodes[0].seed, chain, out)) return false;
    }

    TurnBoundaryTerminalPolicy stop({1, 106});
    const bool joined = discard_hook_ && model && work && step && kv_chain;
    int retried_iteration = -1;
    for (int i = 0; i < budget_; ++i) {
      if (joined) {
        std::string cerr;
        // Label 0 is reused per iteration. Node 0's per-node KV (task root) is never
        // invalidated because a restored tree always has at least node 0.
        if (!step->checkpoint(0, search_, *work, *kv_chain, cerr)) {
          out.err = "checkpoint failed: " + cerr;
          return false;
        }
      }
      PuctPick p;
      if (!search_.select(p, chain)) {
        out.err = "select failed";
        return false;
      }
      const std::vector<float>* lg = &root_logits;
      const float* hid = root_hidden.empty() ? nullptr : root_hidden.data();
      size_t hid_n = root_hidden.size();
      std::vector<float> leaf_lg;
      std::vector<float> leaf_hid;
      if (model && work && step && kv_chain && !p.terminal) {
        if (!step->kv_reposition(p.parent, *work, *kv_chain)) {
          out.err = "rollback failed";
          return false;
        }
        model->forward(*work, p.token);
        step->kv_snapshot(p.leaf, *work, *kv_chain);
        leaf_lg = model->logits();
        leaf_hid = model->hidden_state();
        lg = &leaf_lg;
        hid = leaf_hid.empty() ? nullptr : leaf_hid.data();
        hid_n = leaf_hid.size();
      }
      if (joined && retried_iteration != i && discard_hook_(discard_ctx_, task_id, i)) {
        std::string serr;
        if (!step->step_back(0, search_, *work, *kv_chain, serr)) {
          out.err = "step_back failed: " + serr;
          return false;
        }
        ++out.step_backs;
        retried_iteration = i;
        --i;  // redo this iteration once from the restored tree/KV/scratch chain
        continue;
      }
      // Gate split (2026-11-11): the gate sees only the proposed edge tokens; the depth keeps
      // its old meaning (prompt + proposal length).
      auto proposal = ProposalTokens::from_proposal(token_path(search_.puct(), p, {}));
      const uint32_t depth = uint32_t(prompt.size() + proposal.size());
      const bool term = p.terminal || stop.IsTerminal(p.token);
      const double v = leaf_.Evaluate(proposal, *lg, depth, /*sum_log_prob*/0.0, hid,
                                      hid_n, term);
      out.last_value = v;
      out.last_gate_pass = leaf_.last_context().gate_pass;
      out.gate_rejects = leaf_.gate().RejectCount();
      const uint32_t vq = to_q20(v);
      if (term || !out.last_gate_pass) {
        search_.mark_terminal(p.leaf, vq);
      } else {
        if (p.leaf >= search_.puct().nodes.size()) {
          out.err = "leaf oob";
          return false;
        }
        bool from_trie = false;
        if (span_mode_) {
          const std::vector<int32_t> pre = token_path(search_.puct(), p, {});
          if (trie_.continues(pre)) {
            auto edges = trie_.edges_at(pre);
            if (edges.empty()) {
              out.err = "span trie returned no edges";
              return false;
            }
            search_.expand(p.leaf, std::move(edges), chain);
            ++out.span_expansions;
            from_trie = true;
          }
        }
        if (!from_trie &&
            !expand_node(p.leaf, *lg, search_.puct().nodes[p.leaf].seed, chain, out))
          return false;
      }
      search_.record_leaf(p.leaf, vq, out.last_gate_pass, leaf_.gate().LastRejectToken(), chain);
      search_.backup(p, vq, chain);
    }
    const std::vector<int32_t> committed = search_.commit(chain);
    if (span_mode_ && binder_) {
      const NcmConcept* c = binder_->concept_for_task(task_id);
      if (c && (committed.size() < c->tokens().size() ||
                !std::equal(c->tokens().begin(), c->tokens().end(), committed.begin()))) {
        out.err = "concept span not committed (budget below span length or gate stop)";
        return false;
      }
    }
    out.committed.insert(out.committed.end(), committed.begin(), committed.end());
    const uint64_t d = search_.digest();
    out.puct_digest = out.puct_digest ? hash_mix64(out.puct_digest, d) : hash_mix64(FNV_OFFSET, d);
    if (committed.empty() || d == 0) {
      out.err = "empty commit";
      return false;
    }
    return true;
  }

  HtnPlanner planner_;
  NcmCandidateGenerator ncm_;
  MctsSearch search_;
  PchCpuLeafPath leaf_;
  int budget_ = 2;
  int prior_k_ = 8;
  uint64_t root_seed_ = 42;
  bool span_mode_ = false;
  bool use_plan_state_ = false;
  HtnState plan_state_;
  std::vector<NcmConcept> root_spans_;
  const NcmMethodBinder* binder_ = nullptr;
  NcmSpanTrie trie_;
  DiscardHook discard_hook_ = nullptr;
  void* discard_ctx_ = nullptr;
  rl::RlFlags rl_flags_{};  // default off
};

}  // namespace pchie