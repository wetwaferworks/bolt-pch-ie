// Bolt PCHIE - MctsSearch wrapper around PuctSearch, coordinator-only opaque records.
// Synthetic prior path is a LABELED UNIT-TEST FIXTURE ONLY (never a golden or baseline).
// Real-weight path uses Little Nemo. CPU only, no CUDA, no golden lock.
#include "pchie/MctsSearch.h"
#include "pchie/CommitSelector.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchCpuLeafPath.h"
#include "pchie/TurnBoundaryTerminalPolicy.h"
#include "pchie/ChainRecord.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

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
  std::printf("[MCTS] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static uint64_t fnv_bytes(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = FNV_OFFSET;
  constexpr uint64_t P = 1099511628211ULL;
  for (size_t i = 0; i < n; ++i) { h ^= uint64_t(b[i]); h *= P; }
  return h;
}

// Test-only recorder (not a production class). Forwards to FNV for the head.
class RecordingChain final : public IHashChain {
 public:
  std::vector<ChainRecordKind> kinds;
  Fnv1a64HashChain inner;
  void reset() override { kinds.clear(); inner.reset(); }
  void append(const ChainRecord& r) override {
    kinds.push_back(r.kind);
    inner.append(r);
  }
  std::vector<uint8_t> head() const override { return inner.head(); }
  size_t length() const override { return inner.length(); }
  void restore_head(const std::vector<uint8_t>& h, size_t n) override { inner.restore_head(h, n); }
};

static uint32_t to_q20(double v) {
  if (v < 0.0) v = 0.0;
  if (v > 1.0) v = 1.0;
  return uint32_t(v * double(PUCT_Q20));
}

// Labeled unit-test fixture: two synthetic prior edges, fixed leaf values, no model.
static void run_fixture(MctsSearch& S, IHashChain* chain, int budget) {
  S.reset(/*root_seed*/42, /*task_id*/7);
  std::vector<PuctEdge> root = {
      PuctEdge{1, uint32_t(double(PUCT_Q20) * 0.7), 0xFFFFFFFFu},
      PuctEdge{2, uint32_t(double(PUCT_Q20) * 0.3), 0xFFFFFFFFu}};
  S.expand(0, root, chain);
  for (int i = 0; i < budget; ++i) {
    PuctPick p;
    if (!S.select(p, chain)) break;
    const uint32_t v = (p.token == 1) ? uint32_t(double(PUCT_Q20) * 0.7)
                                      : uint32_t(double(PUCT_Q20) * 0.2);
    if (p.leaf >= S.puct().nodes.size()) break;
    if (!p.terminal && !S.puct().nodes[p.leaf].expanded) {
      S.expand(p.leaf, {PuctEdge{9, PUCT_Q20 - 1, 0xFFFFFFFFu}}, chain);
    }
    S.record_leaf(p.leaf, v, /*gate_pass*/true, /*last_reject*/-1, chain);
    S.backup(p, v, chain);
  }
  (void)S.commit(chain);
}

static void test_fixture_synthetic_priors() {
  RecordingChain a, b;
  a.reset();
  b.reset();
  MctsSearch s1, s2;
  run_fixture(s1, &a, /*budget*/4);
  run_fixture(s2, &b, /*budget*/4);

  check(s1.task_id() == 7 && s2.task_id() == 7, "reset stores task_id (per-task search)");
  check(s1.puct().nodes.size() > 1 && s1.puct().nodes[0].expanded,
        "fixture expands the PUCT root via the wrapper");
  check(s1.digest() != 0 && s1.digest() == s2.digest(),
        "fixture digest is nonzero and run-to-run deterministic");

  const auto c1 = s1.commit(nullptr);
  const auto c2 = s2.commit(nullptr);
  check(!c1.empty() && c1 == c2, "fixture commit path is nonempty and deterministic");

  check(a.length() == b.length() && a.length() > 0 && a.head() == b.head(),
        "fixture coordinator chain length and FNV head are deterministic");

  // Expected kinds: Expansion (root), then per sim: Selection, [Expansion], LeafValue, GateVerdict, Backup; then Commit.
  check(!a.kinds.empty() && a.kinds.front() == ChainRecordKind::Expansion,
        "fixture chain starts with Expansion");
  check(!a.kinds.empty() && a.kinds.back() == ChainRecordKind::Commit,
        "fixture chain ends with Commit");
  auto has = [&](ChainRecordKind k) {
    for (auto x : a.kinds) if (x == k) return true;
    return false;
  };
  check(has(ChainRecordKind::Selection) && has(ChainRecordKind::LeafValue) &&
            has(ChainRecordKind::Backup) && has(ChainRecordKind::GateVerdict),
        "fixture chain records Selection, LeafValue, Backup, GateVerdict");

  // Reused PuctPick must not leak a stale terminal flag through the wrapper.
  MctsSearch s3;
  s3.reset(42, 7);
  s3.expand(0, {PuctEdge{1, PUCT_Q20 - 1, 0xFFFFFFFFu}}, nullptr);
  PuctPick p{};
  check(s3.select(p, nullptr) && p.terminal == false && p.leaf != 0,
        "first select creates a non-terminal leaf and zeros the pick");
  s3.mark_terminal(p.leaf, PUCT_Q20 / 2);
  s3.record_leaf(p.leaf, PUCT_Q20 / 2, true, -1, nullptr);
  s3.backup(p, PUCT_Q20 / 2, nullptr);
  check(s3.select(p, nullptr) && p.terminal == true && p.leaf != 0,
        "re-hit of a terminal sets pick.terminal");
  s3.backup(p, PUCT_Q20 / 2, nullptr);

  const auto bytes = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(c1.data()),
                                          reinterpret_cast<const uint8_t*>(c1.data()) + c1.size() * sizeof(int32_t));
  const auto sha = Sha256::hash(bytes.data(), bytes.size());
  const uint64_t fnv = fnv_bytes(bytes.data(), bytes.size());
  bool collide8 = sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
  check(!collide8, "commit-path FNV-1a64 != first 8 bytes of SHA-256 (no accidental agreement)");
  if (collide8)
    std::printf("[MCTS] ALERT: FNV vs SHA disagreement FAILED on commit path -- report it immediately\n");
  std::printf("[MCTS] fixture digest=0x%016llX commit_sha256=%s fnv=0x%016llX\n",
              (unsigned long long)s1.digest(), GgufWeightsHash::to_hex(sha).c_str(),
              (unsigned long long)fnv);
}

static std::vector<int32_t> token_path(const PuctSearch& S, const PuctPick& p,
                                       const std::vector<int32_t>& prefix) {
  std::vector<int32_t> toks = prefix;
  // p.path is node ids from root to leaf; skip the root (no inbound token).
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

static bool run_real_search(NemotronModel& m, const ModelBranch& root,
                            const std::vector<float>& root_logits,
                            const std::vector<int32_t>& prompt,
                            IHashChain* chain, uint64_t* digest,
                            std::vector<int32_t>* committed,
                            std::vector<ModelBranch>& live) {
  MctsSearch S;
  S.reset(/*root_seed*/123, /*task_id*/5);
  const int K = 8;
  const float tau = S.puct().cfg.prior_temp;
  // Guard: a stub/unreset tree has no root node; do not index nodes[0].
  if (S.puct().nodes.empty()) return false;
  auto e0 = PuctSearch::priors_from_logits(root_logits, K, tau, S.puct().nodes[0].seed);
  S.expand(0, std::move(e0), chain);
  live.clear();
  live.push_back(m.fork(root));
  PchCpuLeafPath ev(0.7, 0.0);
  TurnBoundaryTerminalPolicy stop({1, 106});
  const int budget = 2;
  for (int i = 0; i < budget; ++i) {
    PuctPick p;
    if (!S.select(p, chain)) return false;
    if (p.parent >= live.size()) return false;
    if (p.leaf >= live.size()) live.resize(p.leaf + 1);
    if (!p.terminal) {
      live[p.leaf] = m.fork(live[p.parent]);
      m.forward(live[p.leaf], p.token);
    }
    // Gate split (2026-11-11): gate sees proposal tokens only; depth keeps prompt + proposal.
    const auto path = token_path(S.puct(), p, {});
    const bool term = p.terminal || stop.IsTerminal(p.token);
    const double v = ev.Evaluate(ProposalTokens::from_proposal(path), m.logits(), /*depth*/uint32_t(prompt.size() + path.size()),
                                 /*sum_log_prob*/0.0, m.hidden_state().data(),
                                 m.hidden_state().size(), term);
    const uint32_t vq = to_q20(v);
    if (term) {
      S.mark_terminal(p.leaf, vq);
    } else {
      if (p.leaf >= S.puct().nodes.size()) return false;
      auto e = PuctSearch::priors_from_logits(m.logits(), K, tau, S.puct().nodes[p.leaf].seed);
      S.expand(p.leaf, std::move(e), chain);
    }
    S.record_leaf(p.leaf, vq, ev.last_context().gate_pass, ev.gate().LastRejectToken(), chain);
    S.backup(p, vq, chain);
  }
  *committed = S.commit(chain);
  *digest = S.digest();
  return !committed->empty() && *digest != 0;
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight MCTS: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[MCTS] load err: %s\n", err.c_str());
    return;
  }
  const auto ids = m.tokenize("Hi");
  if (ids.empty()) {
    check(false, "real-weight MCTS: tokenize(\"Hi\") nonempty");
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/16);
  ModelBranch root = m.new_branch();
  for (int32_t t : ids) m.forward(root, t);  // BOS policy: full prompt {BOS, Hi}
  const auto root_logits = m.logits();
  check(!root_logits.empty(), "real-weight MCTS: root logits present");
  // Gated path prefix excludes the runtime BOS anchor (stand-in reject set holds id 2).
  const auto gated = m.tokenize_no_bos("Hi");

  Fnv1a64HashChain fnv1, fnv2;
  fnv1.reset();
  fnv2.reset();
  uint64_t d1 = 0, d2 = 0;
  std::vector<int32_t> c1, c2;
  std::vector<ModelBranch> live1, live2;
  const bool ok1 = run_real_search(m, root, root_logits, gated, &fnv1, &d1, &c1, live1);
  const bool ok2 = run_real_search(m, root, root_logits, gated, &fnv2, &d2, &c2, live2);
  check(ok1 && ok2, "real Little Nemo two serial searches complete (budget 2)");
  check(d1 == d2 && d1 != 0, "real Little Nemo PUCT digest is run-to-run deterministic");
  check(c1 == c2 && !c1.empty(), "real Little Nemo commit path is run-to-run deterministic");
  check(fnv1.length() == fnv2.length() && fnv1.length() > 0 && fnv1.head() == fnv2.head(),
        "real Little Nemo FNV chain head is run-to-run deterministic");

  Sha256HashChain sha1, sha2;
  sha1.reset();
  sha2.reset();
  uint64_t ds1 = 0, ds2 = 0;
  std::vector<int32_t> cs1, cs2;
  std::vector<ModelBranch> ls1, ls2;
  (void)run_real_search(m, root, root_logits, gated, &sha1, &ds1, &cs1, ls1);
  (void)run_real_search(m, root, root_logits, gated, &sha2, &ds2, &cs2, ls2);
  check(sha1.length() == sha2.length() && sha1.head() == sha2.head(),
        "real Little Nemo SHA-256 chain head is run-to-run deterministic");
  check(ds1 == d1 && cs1 == c1, "real Little Nemo digest/commit match the FNV-chain runs");

  const auto bytes = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(c1.data()),
                                          reinterpret_cast<const uint8_t*>(c1.data()) + c1.size() * sizeof(int32_t));
  const auto sha = Sha256::hash(bytes.data(), bytes.size());
  const uint64_t fnv = fnv_bytes(bytes.data(), bytes.size());
  bool collide8 = sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
  check(!collide8, "real commit-path FNV-1a64 != first 8 bytes of SHA-256");
  if (collide8)
    std::printf("[MCTS] ALERT: FNV vs SHA disagreement FAILED on real commit path -- report it immediately\n");
  std::printf("[MCTS] real digest=0x%016llX commit_sha256=%s fnv=0x%016llX n=%zu\n",
              (unsigned long long)d1, GgufWeightsHash::to_hex(sha).c_str(),
              (unsigned long long)fnv, c1.size());

  for (auto& b : live1) m.release(b);
  for (auto& b : live2) m.release(b);
  for (auto& b : ls1) m.release(b);
  for (auto& b : ls2) m.release(b);
  m.release(root);
}

int main() {
  std::printf("[PCH] MctsSearch wrapper tests (synthetic fixture + Little Nemo)\n");
  test_fixture_synthetic_priors();
  test_real_little_nemo();
  std::printf("[PCH] MctsSearch %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
