// Bolt PCHIE - PchStack coordinator glue.
// Synthetic logits / domains are a LABELED UNIT-TEST FIXTURE ONLY (never a golden or baseline).
// Real-weight path uses Little Nemo + PchGate (not AcceptAllGate).
// The original DualPool harness stays on AcceptAllGate and is not edited.
// CPU only, no CUDA, no golden lock, no B1 maze.
#include "pchie/PchStack.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchGate.h"
#include "pchie/HtnMethod.h"
#include "pchie/Sha256.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/pchie_tree.h"
#include "pchie/ChainRecord.h"
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
  std::printf("[PCH-STACK] %-72s %s\n", what, ok ? "PASS" : "FAIL");
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

// Tiny domain (NOT the B1 maze). Goal 1 -> primitive Utter 2.
static bool add_unit_domain(HtnDomain& d) {
  return d.add_task(HtnTask(1, "Goal", false)) &&
         d.add_task(HtnTask(2, "Utter", true)) &&
         d.add_method(HtnMethod(1, "DoUtter", 1, {2}));
}

class RecordingChain final : public IHashChain {
 public:
  Fnv1a64HashChain inner;
  std::vector<ChainRecordKind> kinds;
  void reset() override { inner.reset(); kinds.clear(); }
  void append(const ChainRecord& r) override {
    kinds.push_back(r.kind);
    inner.append(r);
  }
  std::vector<uint8_t> head() const override { return inner.head(); }
  size_t length() const override { return inner.length(); }
  void restore_head(const std::vector<uint8_t>& h, size_t n) override {
    inner.restore_head(h, n);
  }
};

static bool has_kind(const std::vector<ChainRecordKind>& ks, ChainRecordKind k) {
  for (auto x : ks) if (x == k) return true;
  return false;
}

static void test_unit_fixture() {
  HtnDomain domain;
  check(add_unit_domain(domain), "unit-test fixture domain registered (not maze maze)");

  std::string err;
  const auto expected = domain.decompose(1, err);
  check(expected.size() == 1 && expected[0].id() == 2, "domain decompose(Goal) => {Utter}");

  // Clean synthetic logits: peak at token 7 (not in the default PchGate reject set, not EOS).
  std::vector<float> lg(10, 0.0f);
  lg[7] = 3.0f;
  lg[9] = 1.5f;
  const std::vector<int32_t> prompt = {11};

  RecordingChain a, b;
  a.reset();
  b.reset();
  PchStack s1, s2;
  s1.set_goal(1);
  s2.set_goal(1);
  s1.set_root_seed(42);
  s2.set_root_seed(42);
  s1.set_budget(2);
  s2.set_budget(2);
  s1.set_prior_k(4);
  s2.set_prior_k(4);
  PchStackResult r1, r2;
  const bool ok1 = s1.run_unit_fixture(domain, "", lg, prompt, &a, r1);
  const bool ok2 = s2.run_unit_fixture(domain, "poisoned world / maze maze bait", lg, prompt, &b, r2);
  check(ok1 && ok2 && r1.ok && r2.ok, "unit fixture run_unit_fixture succeeds twice");
  check(r1.plan == expected && r2.plan == expected,
        "unit fixture plan equals domain decompose and ignores untrusted world text");
  check(r1.last_task_id == 2 && r2.last_task_id == 2, "unit fixture last task is Utter");
  check(!r1.committed.empty() && r1.committed == r2.committed,
        "unit fixture committed tokens are nonempty and deterministic");
  check(r1.puct_digest != 0 && r1.puct_digest == r2.puct_digest,
        "unit fixture PUCT digest is nonzero and deterministic");
  check(a.length() == b.length() && a.length() > 0 && a.head() == b.head(),
        "unit fixture FNV chain head and length are deterministic");
  check(!a.kinds.empty() && a.kinds.front() == ChainRecordKind::Custom,
        "unit fixture chain starts with the planner Custom record");
  check(!a.kinds.empty() && a.kinds.back() == ChainRecordKind::Commit,
        "unit fixture chain ends with Commit");
  check(has_kind(a.kinds, ChainRecordKind::Expansion) &&
            has_kind(a.kinds, ChainRecordKind::Selection) &&
            has_kind(a.kinds, ChainRecordKind::LeafValue) &&
            has_kind(a.kinds, ChainRecordKind::GateVerdict) &&
            has_kind(a.kinds, ChainRecordKind::Backup),
        "unit fixture coordinator records Expansion, Selection, LeafValue, GateVerdict, Backup");
  check(r1.last_gate_pass && r1.last_value > 0.0,
        "unit fixture clean path: PchGate passes and V > 0");
  check(!r1.last_ncm.empty(), "unit fixture NCM candidates were produced");

  // No-goal fails cleanly.
  PchStack nog;
  PchStackResult bad;
  check(nog.run_unit_fixture(domain, "", lg, prompt, nullptr, bad) == false && !bad.ok &&
            bad.err.find("goal") != std::string::npos,
        "unit fixture without set_goal fails cleanly");

  // Reject path: token 4 is in the default PchGate reject set.
  std::vector<float> bad_lg(10, 0.0f);
  bad_lg[4] = 5.0f;
  PchStack sj;
  sj.set_goal(1);
  sj.set_root_seed(42);
  sj.set_budget(2);
  sj.set_prior_k(4);
  PchStackResult rj;
  const bool okj = sj.run_unit_fixture(domain, "", bad_lg, prompt, nullptr, rj);
  check(okj && rj.ok, "unit fixture reject-path run still completes");
  check(rj.last_gate_pass == false && rj.last_value == 0.0 && rj.gate_rejects > 0,
        "unit fixture PchGate reject path zeros V (G=0)");
}

static void test_real_little_nemo() {
  HtnDomain domain;
  if (!add_unit_domain(domain)) {
    check(false, "real-weight stack: unit domain registered");
    return;
  }
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight stack: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[PCH-STACK] load err: %s\n", err.c_str());
    return;
  }
  const auto prompt = m.tokenize("Hi");
  if (prompt.empty()) {
    check(false, "real-weight stack: tokenize(\"Hi\") nonempty");
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/8);
  ModelBranch root = m.new_branch();
  for (int32_t t : prompt) m.forward(root, t);
  check(!m.logits().empty() && !m.hidden_state().empty(),
        "real-weight stack: root logits and hidden present");
  const std::vector<float> root_logits = m.logits();
  const std::vector<float> root_hidden = m.hidden_state();
  // BOS policy: root KV is {BOS, Hi}; the gated path prefix excludes the runtime BOS
  // anchor (stand-in reject set holds id 2; open item for Archie) -> {10979} as before.
  const auto gated = m.tokenize_no_bos("Hi");

  const uint32_t live_prompt = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  Fnv1a64HashChain fnv1, fnv2;
  Sha256HashChain sha1, sha2;
  fnv1.reset();
  fnv2.reset();
  sha1.reset();
  sha2.reset();
  PchStackResult a, b, c, d;
  bool oka = false, okb = false, okc = false, okd = false;
  {
    PchStack s1, s2, s3, s4;
    s1.set_goal(1);
    s2.set_goal(1);
    s3.set_goal(1);
    s4.set_goal(1);
    s1.set_root_seed(123);
    s2.set_root_seed(123);
    s3.set_root_seed(123);
    s4.set_root_seed(123);
    s1.set_budget(2);
    s2.set_budget(2);
    s3.set_budget(2);
    s4.set_budget(2);
    s1.set_prior_k(8);
    s2.set_prior_k(8);
    s3.set_prior_k(8);
    s4.set_prior_k(8);
    oka = s1.run(m, root, gated, root_logits, root_hidden, domain, "", &fnv1, a);
    okb = s2.run(m, root, gated, root_logits, root_hidden, domain, "untrusted world text", &fnv2, b);
    okc = s3.run(m, root, gated, root_logits, root_hidden, domain, "", &sha1, c);
    okd = s4.run(m, root, gated, root_logits, root_hidden, domain, "", &sha2, d);
  }
  check(oka && okb && okc && okd && a.ok && b.ok && c.ok && d.ok,
        "real Little Nemo four serial PchStack runs complete (budget 2)");
  check(a.plan == b.plan && a.plan.size() == 1 && a.plan[0].id() == 2,
        "real Little Nemo plan is {Utter} and ignores world text");
  check(!a.committed.empty() && a.committed == b.committed && a.committed == c.committed &&
            a.committed == d.committed,
        "real Little Nemo committed tokens are run-to-run deterministic");
  check(a.puct_digest != 0 && a.puct_digest == b.puct_digest && a.puct_digest == c.puct_digest &&
            a.puct_digest == d.puct_digest,
        "real Little Nemo PUCT digest is run-to-run deterministic");
  check(fnv1.length() == fnv2.length() && fnv1.length() > 0 && fnv1.head() == fnv2.head(),
        "real Little Nemo FNV chain head is run-to-run deterministic");
  check(sha1.length() == sha2.length() && sha1.length() > 0 && sha1.head() == sha2.head(),
        "real Little Nemo SHA-256 chain head is run-to-run deterministic");
  check(a.last_gate_pass && a.last_value > 0.0, "real Little Nemo clean path: PchGate passes, V > 0");
  check(!a.last_ncm.empty() && a.last_ncm.size() <= 8, "real Little Nemo NCM candidates present, within K");

  const auto bytes = std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(a.committed.data()),
      reinterpret_cast<const uint8_t*>(a.committed.data()) + a.committed.size() * sizeof(int32_t));
  const auto sha = Sha256::hash(bytes.data(), bytes.size());
  const uint64_t fnv = fnv_bytes(bytes.data(), bytes.size());
  bool collide8 = sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
  check(!collide8, "commit-path FNV-1a64 != first 8 bytes of SHA-256");
  if (collide8)
    std::printf("[PCH-STACK] ALERT: FNV vs SHA disagreement FAILED on commit path -- report it immediately\n");
  std::printf("[PCH-STACK] digest=0x%016llX commit_sha256=%s fnv=0x%016llX n=%zu V=%.6f\n",
              (unsigned long long)a.puct_digest, GgufWeightsHash::to_hex(sha).c_str(),
              (unsigned long long)fnv, a.committed.size(), a.last_value);

  // Reject the prompt token itself so every Evaluate path fails the gate.
  const int32_t hi = gated[0];  // 10979, as before the BOS policy
  PchStack rej(PchCpuLeafPath(0.7, 0.0, PchGate(std::vector<int32_t>{hi})));
  rej.set_goal(1);
  rej.set_root_seed(123);
  rej.set_budget(2);
  rej.set_prior_k(8);
  PchStackResult rr;
  const bool okr = rej.run(m, root, gated, root_logits, root_hidden, domain, "", nullptr, rr);
  check(okr && rr.ok, "real Little Nemo reject-path run completes");
  // Gate split (2026-11-11): the gate sees ONLY proposal tokens, so rejecting a
  // prompt token no longer zeros any leaf. Pre-split this asserted G=0 (gsplit\regress log keeps
  // the pre-update FAIL as evidence). Proposal-side G=0 stays covered by the unit-fixture reject
  // path (token 4 proposed) and by pchie_gate_split_test.
  check(rr.gate_rejects == 0 && rr.last_gate_pass == true && rr.last_value > 0.0,
        "gate split: rejecting a PROMPT token no longer zeros V (gate sees proposals only)");
  // OPEN ITEM pinned (BOS policy vs provisional stand-in gate): with the BOS anchor inside
  // the gated prefix, every leaf fails the stand-in gate (id 2 is in its reject set).
  {
    PchStack bos_gated;
    bos_gated.set_goal(1);
    bos_gated.set_root_seed(123);
    bos_gated.set_budget(2);
    bos_gated.set_prior_k(8);
    PchStackResult rb;
    const bool okb2 = bos_gated.run(m, root, prompt, root_logits, root_hidden, domain, "", nullptr, rb);
    // Gate split closes the pinned OPEN ITEM: BOS (id 2) in the trusted prompt is not gated.
    check(okb2 && rb.last_gate_pass == true && rb.last_value > 0.0 && rb.gate_rejects == 0,
          "gate split: BOS inside the trusted prompt no longer rejects leaves (OPEN ITEM closed)");
  }

  m.release(root);
  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == 0, "no KV page leak after PchStack runs and root release");
  std::printf("[PCH-STACK] live_pages prompt=%u end=%u\n", live_prompt, live_end);
}

int main() {
  std::printf("[PCH] PchStack coordinator glue tests (PchGate path; original AcceptAllGate untouched)\n");
  test_unit_fixture();
  test_real_little_nemo();
  std::printf("[PCH] PchStack %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
