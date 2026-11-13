// Bolt PCHIE - new CPU leaf path (PchGate + LogProbLeafEvaluator).
// Tests first. Does not touch the original AcceptAllGate path. CPU only.
// Token-id fixture plus a real Little Nemo leaf (weights involved => real model).
#include "pchie/PchCpuLeafPath.h"
#include "pchie/NemotronModel.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/LogitStats.h"
#include "pchie/Sha256.h"
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
  std::printf("[LEAF-PATH] %-72s %s\n", what, ok ? "PASS" : "FAIL");
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

static void test_fixture_token_id_path() {
  // Labeled unit-test fixture: synthetic logits, no model. Not a golden / baseline.
  PchCpuLeafPath path(0.7, 0.0);
  const std::vector<float> logits = {0.0f, 1.0f, 0.25f, -0.5f};
  const float hidden[4] = {1.0f, 2.0f, 3.0f, 4.0f};

  const double v_clean = path.Evaluate(ProposalTokens::from_proposal(std::vector<int32_t>{1003}), logits, /*depth*/1,
                                       /*sum_log_prob*/0.0, hidden, 4, /*is_terminal*/false);
  check(v_clean == 0.5, "fixture clean path, beta=0, sum_log_prob=0 => V == sigmoid(0) == 0.5");
  check(path.last_context().gate_pass == true, "fixture clean path sets gate_pass true");
  check(path.last_context().hidden_state == hidden && path.last_context().hidden_size == 4,
        "fixture plumbs hidden_state into LeafContext");
  check(path.last_context().depth == 1 && path.last_context().sum_log_prob == 0.0,
        "fixture records depth and sum_log_prob");
  check(path.last_context().entropy != 0.0, "fixture non-terminal entropy comes from logits");

  const double v_rej = path.Evaluate(ProposalTokens::from_proposal(std::vector<int32_t>{1003, 63937}), logits, 1, 0.0, hidden, 4, false);
  check(v_rej == 0.0, "fixture reject path => V == 0 (G = 0)");
  check(path.last_context().gate_pass == false, "fixture reject path sets gate_pass false");
  check(path.gate().LastRejectToken() == 63937, "fixture reject records LastRejectToken 63937");

  const double v_term = path.Evaluate(ProposalTokens::from_proposal(std::vector<int32_t>{1003}), logits, 1, 0.0, hidden, 4, /*is_terminal*/true);
  check(v_term == 0.5, "fixture terminal clean path still has V == 0.5 when beta=0");
  check(path.last_context().entropy == 0.0, "fixture terminal forces entropy to 0");

  const double a = path.Evaluate(ProposalTokens::from_proposal(std::vector<int32_t>{1003}), logits, 1, -0.25, nullptr, 0, false);
  const double b = path.Evaluate(ProposalTokens::from_proposal(std::vector<int32_t>{1003}), logits, 1, -0.25, nullptr, 0, false);
  check(a == b && a > 0.0 && a < 0.5, "fixture run-to-run V is deterministic and uses sum_log_prob");
}

static void test_real_little_nemo_leaf() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight leaf: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[LEAF-PATH] load err: %s\n", err.c_str());
    return;
  }
  const auto ids = m.tokenize("Hi");
  if (ids.empty()) {
    check(false, "real-weight leaf: tokenize(\"Hi\") nonempty");
    return;
  }
  // BOS policy: tokenize honors add_bos_token=True -> {<bos>=2, Hi=10979}.
  check(ids.size() == 2 && ids[0] == 2 && ids[1] == 10979, "real-weight leaf: tokenize(\"Hi\") is {2, 10979} (BOS policy)");
  const auto word = m.tokenize_no_bos("Hi");
  check(word.size() == 1 && word[0] == 10979, "real-weight leaf: tokenize_no_bos(\"Hi\") is {10979} (clean id)");

  m.init_cache(/*max_ctx*/32, /*branches*/2);
  ModelBranch br = m.new_branch();
  for (int32_t t : ids) m.forward(br, t);  // full prompt {BOS, Hi}
  const std::vector<float> logits = m.logits();
  const std::vector<float> hidden = m.hidden_state();
  check(!logits.empty() && int(hidden.size()) == m.d_model(), "real-weight leaf: logits and hidden present");

  PchCpuLeafPath path(0.7, 0.0);
  // Gated path: the BOS anchor is not a proposal token (stand-in reject set holds id 2).
  const std::vector<int32_t> clean = word;
  const double v1 = path.Evaluate(ProposalTokens::from_proposal(clean), logits, /*depth*/1, /*sum_log_prob*/0.0,
                                  hidden.data(), hidden.size(), false);
  check(v1 == 0.5, "real Little Nemo clean path {10979} => V == 0.5 (beta=0, sum_lp=0)");
  check(path.last_context().gate_pass == true, "real Little Nemo clean path gate_pass true");
  check(path.last_context().hidden_state == hidden.data() &&
        path.last_context().hidden_size == hidden.size(),
        "real Little Nemo hidden_state is the model's post-output-RMSNorm vector");

  const double v2 = path.Evaluate(ProposalTokens::from_proposal(clean), logits, 1, 0.0, hidden.data(), hidden.size(), false);
  check(v1 == v2, "real Little Nemo leaf V is run-to-run deterministic");

  std::vector<int32_t> rejected = word;
  rejected.push_back(63937);
  const double v_rej = path.Evaluate(ProposalTokens::from_proposal(rejected), logits, 1, 0.0, hidden.data(), hidden.size(), false);
  check(v_rej == 0.0, "real Little Nemo path plus reject id 63937 => V == 0 (G = 0)");
  check(path.last_context().gate_pass == false, "real Little Nemo reject path gate_pass false");

  // SHA vs FNV: alert if FNV head equals first 8 bytes of SHA on hidden bits.
  const auto hidden_sha = Sha256::hash(reinterpret_cast<const uint8_t*>(hidden.data()),
                                       hidden.size() * sizeof(float));
  const uint64_t hidden_fnv = fnv_bytes(hidden.data(), hidden.size() * sizeof(float));
  bool collide8 = hidden_sha.size() >= 8 &&
                  std::memcmp(&hidden_fnv, hidden_sha.data(), 8) == 0;
  check(!collide8, "hidden FNV-1a64 != first 8 bytes of SHA-256 (no accidental agreement)");
  if (collide8)
    std::printf("[LEAF-PATH] ALERT: FNV vs SHA disagreement FAILED on hidden — report it immediately\n");
  std::printf("[LEAF-PATH] hidden sha256=%s fnv=0x%016llX V_clean=%.6f\n",
              GgufWeightsHash::to_hex(hidden_sha).c_str(),
              (unsigned long long)hidden_fnv, v1);

  m.release(br);
}

int main() {
  std::printf("[PCH] PchCpuLeafPath tests (new CPU leaf path; original AcceptAllGate untouched)\n");
  test_fixture_token_id_path();
  test_real_little_nemo_leaf();
  std::printf("[PCH] PchCpuLeafPath %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
