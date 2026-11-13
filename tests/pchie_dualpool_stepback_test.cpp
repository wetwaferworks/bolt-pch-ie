// Bolt PCHIE - DualPool step-back on MctsSearch, AcceptAllGate (G=1) path.
// Also covers priors_from_logits when K exceeds the logit length.
// Synthetic prior checks are a LABELED UNIT-TEST FIXTURE ONLY (never a golden or baseline).
// Real-weight path uses Little Nemo. CPU only, no CUDA, no golden lock.
#include "pchie/MctsDualPoolStepBack.h"
#include "pchie/MctsSearch.h"
#include "pchie/AcceptAllGate.h"
#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/LogitStats.h"
#include "pchie/LeafContext.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/NemotronModel.h"
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
  std::printf("[DUALPOOL-STEPBACK] %-72s %s\n", what, ok ? "PASS" : "FAIL");
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

static bool edges_equal(const std::vector<PuctEdge>& a, const std::vector<PuctEdge>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].token != b[i].token || a[i].prior_q20 != b[i].prior_q20 || a[i].child != b[i].child)
      return false;
  }
  return true;
}
static uint32_t to_q20(double v) {
  if (v < 0.0) v = 0.0;
  if (v > 1.0) v = 1.0;
  return uint32_t(v * double(PUCT_Q20));
}

static void test_priors_k_exceeds_logit_length() {
  // Labeled unit-test fixture: tiny logit vector, no model.
  const std::vector<float> lg = {0.1f, 2.0f, -0.5f};
  const auto exact = PuctSearch::priors_from_logits(lg, /*K*/3, 1.25f, /*seed*/11);
  const auto over = PuctSearch::priors_from_logits(lg, /*K*/8, 1.25f, 11);
  const auto over2 = PuctSearch::priors_from_logits(lg, 8, 1.25f, 11);
  check(exact.size() == 3, "priors_from_logits K==len returns 3 edges");
  check(over.size() == 3, "priors_from_logits K>len clamps to logit length (no UB)");
  check(edges_equal(over, over2) && edges_equal(over, exact), "clamped priors match K==len and are deterministic");

  const auto empty = PuctSearch::priors_from_logits({}, 8, 1.25f, 11);
  check(empty.empty(), "priors_from_logits empty logits returns no edges");
  const auto zero = PuctSearch::priors_from_logits(lg, 0, 1.25f, 11);
  check(zero.empty(), "priors_from_logits K==0 returns no edges");
  const auto neg = PuctSearch::priors_from_logits(lg, -4, 1.25f, 11);
  check(neg.empty(), "priors_from_logits K<0 returns no edges");
}

static void test_rollback_unknown_node() {
  // Synthetic pool so this case does not need Little Nemo.
  // Labeled unit-test fixture only.
  KVPagePool swa(1, 1, 4, 8);
  KVPagePool glob(1, 1, 4, 8);
  MctsDualPoolStepBack sb(swa, glob);
  Fnv1a64HashChain chain;
  chain.reset();
  ModelBranch out;
  check(sb.rollback(0, out, chain) == false, "rollback of a missing node_id fails cleanly");
}

static bool eval_accept_all(const std::vector<int32_t>& path,
                            const std::vector<float>& logits,
                            uint32_t depth, bool terminal,
                            const float* hidden, size_t hidden_size,
                            uint32_t* value_q20) {
  AcceptAllGate gate;
  LogProbLeafEvaluator ev(0.7, 0.0);
  LeafContext lc;
  lc.depth = depth;
  lc.sum_log_prob = 0.0;
  lc.gate_pass = gate.Pass(path);
  lc.hidden_state = hidden;
  lc.hidden_size = hidden_size;
  if (terminal || logits.empty()) lc.entropy = 0.0;
  else lc.entropy = LogitStats(logits).Entropy();
  if (!lc.gate_pass) return false;
  *value_q20 = to_q20(ev.EvalLeaf(lc));
  return true;
}

static void test_real_little_nemo_stepback() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "step-back: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[DUALPOOL-STEPBACK] load err: %s\n", err.c_str());
    return;
  }
  const auto ids = m.tokenize("Hi");
  if (ids.empty()) {
    check(false, "step-back: tokenize(\"Hi\") nonempty");
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/16);
  ModelBranch root = m.new_branch();
  for (int32_t t : ids) m.forward(root, t);  // BOS policy: full prompt {BOS, Hi}
  const auto root_logits = m.logits();
  check(!root_logits.empty() && int(m.hidden_state().size()) == m.d_model(),
        "step-back: root logits and hidden present");

  const uint32_t live_after_prompt =
      m.pool_swa().live_pages() + m.pool_glob().live_pages();

  MctsSearch S;
  S.reset(/*root_seed*/123, /*task_id*/5);
  Fnv1a64HashChain chain;
  chain.reset();
  const int K = 8;
  auto e0 = PuctSearch::priors_from_logits(root_logits, K, S.puct().cfg.prior_temp,
                                           S.puct().nodes[0].seed);
  S.expand(0, std::move(e0), &chain);
  ModelBranch at_root = m.fork(root);
  const std::vector<uint8_t> head_at_snap = chain.head();
  const size_t len_at_snap = chain.length();

  uint32_t vroot = 0;
  check(eval_accept_all(ids, root_logits, /*depth*/1, /*terminal*/false,
                        m.hidden_state().data(), m.hidden_state().size(), &vroot) &&
            vroot > 0,
        "AcceptAllGate (G=1) leaf path yields V>0 on the clean prompt");

  {
  MctsDualPoolStepBack sb(m.pool_swa(), m.pool_glob());
  sb.snapshot(/*node_id*/0, at_root, chain);
  check(sb.store().find(0) != nullptr, "snapshot(0) is stored in DualPoolForkSnapshotStore");

  PuctPick p;
  if (!S.select(p, &chain) || p.parent != 0) {
    check(false, "step-back: select a child of the snapshotted root");
    m.release(at_root);
    m.release(root);
    return;
  }
  ModelBranch child = m.fork(at_root);
  m.forward(child, p.token);
  const auto after = m.logits();
  check(after != root_logits, "forward after snapshot changes logits");
  check(chain.length() > len_at_snap, "search after snapshot appends coordinator records");

  uint32_t vchild = 0;
  std::vector<int32_t> child_path = ids;
  child_path.push_back(p.token);
  check(eval_accept_all(child_path, after, /*depth*/2, false,
                        m.hidden_state().data(), m.hidden_state().size(), &vchild),
        "AcceptAllGate (G=1) still passes on the child path");

  ModelBranch restored;
  check(sb.rollback(0, restored, chain), "rollback(0) succeeds");
  check(chain.length() == len_at_snap && chain.head() == head_at_snap,
        "rollback restores the chain head and length");
  check(restored.n_past == at_root.n_past, "rollback restores n_past");

  m.release(child);
  ModelBranch replay = m.fork(restored);
  m.release(restored);
  m.forward(replay, p.token);
  check(m.logits() == after, "rollback + same next token reproduces post-snapshot logits");

  const auto logit_sha = Sha256::hash(reinterpret_cast<const uint8_t*>(after.data()),
                                      after.size() * sizeof(float));
  const uint64_t logit_fnv = fnv_bytes(after.data(), after.size() * sizeof(float));
  bool collide8 = logit_sha.size() >= 8 &&
                  std::memcmp(&logit_fnv, logit_sha.data(), 8) == 0;
  check(!collide8, "step-back logits FNV-1a64 != first 8 bytes of SHA-256");
  if (collide8)
    std::printf("[DUALPOOL-STEPBACK] ALERT: FNV vs SHA disagreement FAILED on step-back logits -- report it immediately\n");
  std::printf("[DUALPOOL-STEPBACK] logits sha256=%s fnv=0x%016llX token=%d\n",
              GgufWeightsHash::to_hex(logit_sha).c_str(),
              (unsigned long long)logit_fnv, int(p.token));

  // A few snapshot/rollback cycles must not leak pages once the store is gone.
  {
    MctsDualPoolStepBack loop(m.pool_swa(), m.pool_glob());
    for (int i = 0; i < 4; ++i) {
      loop.snapshot(1, replay, chain);
      ModelBranch tmp = m.fork(replay);
      m.forward(tmp, p.token);
      ModelBranch back;
      if (!loop.rollback(1, back, chain)) {
        check(false, "cycle rollback succeeds");
        m.release(tmp);
        break;
      }
      m.release(tmp);
      m.release(back);
    }
  }
  m.release(replay);
  m.release(at_root);
  }
  m.release(root);
  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == 0,
        "no KV page leak after snapshot/rollback cycles and release");
  std::printf("[DUALPOOL-STEPBACK] live_pages prompt=%u end=%u cow=%llu\n",
              live_after_prompt, live_end,
              (unsigned long long)(m.pool_swa().cow_copies() + m.pool_glob().cow_copies()));
}

int main() {
  std::printf("[PCH] DualPool step-back + priors K-guard tests\n");
  test_priors_k_exceeds_logit_length();
  test_rollback_unknown_node();
  test_real_little_nemo_stepback();
  std::printf("[PCH] DualPool step-back %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
