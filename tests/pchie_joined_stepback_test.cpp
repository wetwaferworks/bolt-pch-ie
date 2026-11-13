// Bolt PCHIE - DualPool KV step-back joined with PUCT tree checkpoint/step-back.
// Synthetic KV pools / edges are a LABELED UNIT-TEST FIXTURE ONLY (never a golden or
// baseline). The live part uses real Little Nemo weights on CPU. No golden is locked.
// The joined step-back step-back (and the opt-in PchStack discard
// hook) is scratch-chain only. The official chain is append-only and is never rewound;
// records appended by a discarded iteration stay on it. Tests below pin that with a spy.
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/HtnDomain.h"
#include "pchie/IHashChain.h"
#include "pchie/MctsJoinedStepBack.h"
#include "pchie/MctsSearch.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchStack.h"
#include "pchie/Sha256HashChain.h"
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
  std::printf("[JOINED-STEPBACK] %-75s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

// Test-only official-chain spy (labeled fixture): forwards to a SHA-256 chain, counts every
// restore_head call and every append, and remembers the longest length seen. The joined step-back path must
// never call restore_head on the official chain, so restores() must stay 0.
class OfficialSpyChain final : public IHashChain {
 public:
  void reset() override {
    inner_.reset();
    restores_ = 0;
    appends_ = 0;
    max_len_ = 0;
  }
  void append(const ChainRecord& r) override {
    inner_.append(r);
    ++appends_;
    if (inner_.length() > max_len_) max_len_ = inner_.length();
  }
  std::vector<uint8_t> head() const override { return inner_.head(); }
  size_t length() const override { return inner_.length(); }
  void restore_head(const std::vector<uint8_t>& h, size_t n) override {
    ++restores_;
    inner_.restore_head(h, n);
  }
  int restores() const { return restores_; }
  size_t appends() const { return appends_; }
  size_t max_len() const { return max_len_; }

 private:
  Sha256HashChain inner_;
  int restores_ = 0;
  size_t appends_ = 0;
  size_t max_len_ = 0;
};

static std::vector<PuctEdge> edges(std::initializer_list<int32_t> toks) {
  std::vector<PuctEdge> e;
  const uint32_t p = PUCT_Q20 / uint32_t(toks.size());
  for (int32_t t : toks) { PuctEdge x; x.token = t; x.prior_q20 = p; e.push_back(x); }
  return e;
}

// One synthetic MCTS iteration: select, expand the new leaf, back up a fixed value; KV for the
// leaf is a fork of the parent's KV with one written token (labeled fixture KV writes).
static bool iterate(MctsSearch& S, MctsJoinedStepBack& J, KVPagePool& swa, KVPagePool& glob, ModelBranch& w,
                    Fnv1a64HashChain& kv, IHashChain* official, uint32_t value) {
  PuctPick p;
  if (!S.select(p, official)) return false;
  if (!J.kv_reposition(p.parent, w, kv)) return false;
  *swa.k_write(w.swa, 0, w.n_past) = float(p.token);
  *glob.k_write(w.glob, 0, w.n_past) = float(p.token);
  ++w.n_past;
  J.kv_snapshot(p.leaf, w, kv);
  S.expand(p.leaf, edges({p.token + 1, p.token + 2}), official);
  S.backup(p, value, official);
  return true;
}

static void test_unit_joined() {
  KVPagePool swa(1, 1, 4, 64), glob(1, 1, 4, 64);
  {
    MctsJoinedStepBack J(swa, glob);
    MctsSearch S;
    S.reset(77, /*task*/5);
    Fnv1a64HashChain kv;
    OfficialSpyChain off;
    kv.reset();
    off.reset();
    ModelBranch w{};
    *swa.k_write(w.swa, 0, 0) = 1.0f;
    *glob.k_write(w.glob, 0, 0) = 1.0f;
    w.n_past = 1;
    J.kv_snapshot(0, w, kv);
    S.expand(0, edges({10, 20, 30}), &off);
    bool ok = iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20 / 2) && iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20 / 4);
    std::string err;
    const size_t nodes_at_ckpt = S.puct().nodes.size();
    const uint64_t d_ckpt = S.digest();
    const size_t off_len_ckpt = off.length();
    const auto kv_head = kv.head();
    ok = ok && J.checkpoint(1, S, w, kv, err);
    check(ok, "checkpoint at a phase boundary (no virtual loss)");
    const uint32_t n_past_ckpt = w.n_past;
    // Advance: 3 more iterations (new nodes, more official records, new KV snapshots).
    ok = iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20) && iterate(S, J, swa, glob, w, kv, &off, 3) &&
         iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20 / 3);
    const uint64_t d_after = S.digest();
    const auto off_head_after = off.head();
    const size_t off_len_after = off.length();
    const size_t advance_records = off_len_after - off_len_ckpt;
    const size_t valid_before = J.node_kv().valid_count();
    const uint32_t live_before = swa.live_pages() + glob.live_pages();
    check(ok && S.puct().nodes.size() > nodes_at_ckpt && d_after != d_ckpt && advance_records > 0,
          "search advanced past the checkpoint (tree, official chain, KV)");
    // Fail-closed cases change nothing.
    MctsSearch other;
    other.reset(77, /*task*/6);
    other.expand(0, edges({1}), nullptr);
    check(!J.step_back(1, other, w, kv, err) && err == "task id mismatch", "step_back refuses a different task");
    check(!J.step_back(9, S, w, kv, err) && err == "unknown checkpoint", "step_back refuses an unknown label");
    check(S.digest() == d_after && off.head() == off_head_after && off.length() == off_len_after,
          "refused step_backs changed nothing");
    PuctPick inflight;
    (void)S.select(inflight, nullptr);
    check(!J.checkpoint(2, S, w, kv, err) && err.find("virtual loss") != std::string::npos,
          "checkpoint refuses while a pick holds virtual loss");
    S.backup(inflight, 0, nullptr);
    // Joined step back (scratch chain only).
    ok = J.step_back(1, S, w, kv, err);
    check(ok && S.puct().nodes.size() == nodes_at_ckpt && S.digest() == d_ckpt,
          "step_back restores the PUCT node table (size and digest)");
    check(off.head() == off_head_after && off.length() == off_len_after && off.restores() == 0,
          "step_back leaves the official chain untouched (no restore_head; records stay)");
    check(kv.head() == kv_head && w.n_past == n_past_ckpt, "step_back restores KV n_past and the scratch KV chain head");
    bool dropped = true, kept = true;
    for (uint32_t id = 0; id < 16; ++id) {
      const bool v = J.node_kv().find(id) != nullptr;
      if (id >= nodes_at_ckpt && v) dropped = false;
      if (id < nodes_at_ckpt && !v) kept = false;
    }
    check(dropped && kept && J.node_kv().valid_count() < valid_before,
          "per-node KV snapshots at node ids >= restored size are invalidated; older kept");
    check(swa.live_pages() + glob.live_pages() < live_before, "invalidated snapshots free their KV pages");
    // Redo the same three iterations: identical tree (deterministic join); the official chain
    // grows by the redo's records on top of the discarded ones (append-only history).
    ok = iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20) && iterate(S, J, swa, glob, w, kv, &off, 3) &&
         iterate(S, J, swa, glob, w, kv, &off, PUCT_Q20 / 3);
    check(ok && S.digest() == d_after, "redoing the same iterations after step_back reproduces the digest");
    check(off.length() == off_len_after + advance_records && off.head() != off_head_after &&
              off.max_len() == off.length() && off.restores() == 0,
          "official chain kept the discarded records and appended the redo (never shrank)");
    J.drop_checkpoint(1);
    check(!J.has_checkpoint(1) && !J.step_back(1, S, w, kv, err), "dropped checkpoint cannot be used");
    swa.release(w.swa);
    glob.release(w.glob);
  }
  check(swa.live_pages() + glob.live_pages() == 0, "no KV page leak after the joined store is destroyed");
}

static bool discard_at_1(void*, uint32_t, int it) { return it == 1; }
static bool discard_all(void*, uint32_t, int) { return true; }

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[JOINED-STEPBACK] load err: %s\n", err.c_str());
    return;
  }
  // BOS policy: the root KV is the full prompt (BOS + Hi). The gated path prefix handed to
  // PchStack is the prompt without the runtime BOS anchor (the stand-in reject set holds id 2;
  // open item for Archie), exactly the gated tokens of the pre-BOS baseline.
  const auto prompt = m.tokenize("Hi");
  const auto gated = m.tokenize_no_bos("Hi");
  check(prompt.size() == 2 && prompt[0] == 2 && prompt[1] == 10979 && gated.size() == 1 && gated[0] == 10979,
        "real: prompt is {2 BOS, 10979 Hi}; gated prefix is {10979}");
  m.init_cache(/*max_ctx*/32, /*branches*/16);
  ModelBranch root = m.new_branch();
  for (int32_t t : prompt) m.forward(root, t);
  const std::vector<float> root_logits = m.logits();
  const std::vector<float> root_hidden = m.hidden_state();
  HtnDomain d;
  (void)d.add_task(HtnTask(1, "G", false));
  (void)d.add_task(HtnTask(2, "A", true));
  (void)d.add_task(HtnTask(3, "B", true));
  (void)d.add_method(HtnMethod(10, "AB", 1, {2, 3}));
  auto run = [&](PchStack::DiscardHook h, PchStackResult& r, std::vector<uint8_t>& head, int& restores,
                 size_t& max_len) {
    PchStack s;
    s.set_goal(1);
    s.set_budget(3);
    if (h) s.set_discard_hook(h, nullptr);
    OfficialSpyChain c;
    c.reset();
    const bool ok = s.run(m, root, gated, root_logits, root_hidden, d, "", &c, r);
    head = c.head();
    restores = c.restores();
    max_len = c.max_len();
    return ok;
  };
  PchStackResult clean, one, all;
  std::vector<uint8_t> hc, h1, ha;
  int rc = -1, r1 = -1, ra = -1;
  size_t mc = 0, m1 = 0, ma = 0;
  const bool okc = run(nullptr, clean, hc, rc, mc);
  const bool ok1 = run(discard_at_1, one, h1, r1, m1);
  const bool oka = run(discard_all, all, ha, ra, ma);
  check(okc && clean.step_backs == 0 && rc == 0, "real: clean PchStack run (no hook) takes no step-back");
  // A discarded iteration appended exactly one official record (its Selection) before the
  // hook fired; that record stays. 2 tasks x 1 discard = +2; 2 tasks x 3 discards = +6.
  check(ok1 && one.step_backs == 2 && one.puct_digest == clean.puct_digest && one.committed == clean.committed,
        "real: discard at iteration 1 -> scratch step_back -> same digest and commit as clean");
  check(ok1 && r1 == 0 && m1 == one.chain_length && one.chain_length == clean.chain_length + 2 && h1 != hc,
        "real: discard at iteration 1 -> official chain never rewound (+2 kept Selection records)");
  check(oka && all.step_backs == 6 && all.puct_digest == clean.puct_digest && all.committed == clean.committed,
        "real: discard every iteration -> same digest and commit as clean");
  check(oka && ra == 0 && ma == all.chain_length && all.chain_length == clean.chain_length + 6 && ha != hc,
        "real: discard every iteration -> official chain never rewound (+6 kept records)");
  std::printf("[JOINED-STEPBACK] real clean chain_len=%zu digest=0x%016llX committed_len=%zu\n", clean.chain_length,
              (unsigned long long)clean.puct_digest, clean.committed.size());
  std::printf("[JOINED-STEPBACK] real hook chain_len: discard@1=%zu discard_all=%zu (append-only; test-only side values)\n",
              one.chain_length, all.chain_length);

  // Direct join with real forwards: checkpoint, advance, step back, redo -> same digest and logits;
  // the official chain keeps the discarded records and appends the redo.
  {
    MctsJoinedStepBack J(m.pool_swa(), m.pool_glob());
    MctsSearch S;
    S.reset(123, 9);
    Fnv1a64HashChain kv;
    OfficialSpyChain off;
    kv.reset();
    off.reset();
    ModelBranch w = m.fork(root);
    J.kv_snapshot(0, w, kv);
    S.expand(0, PuctSearch::priors_from_logits(root_logits, 4, 1.25f, S.puct().nodes[0].seed), &off);
    auto real_iter = [&](uint64_t& logit_fnv) {
      PuctPick p;
      if (!S.select(p, &off)) return false;
      if (!J.kv_reposition(p.parent, w, kv)) return false;
      m.forward(w, p.token);
      J.kv_snapshot(p.leaf, w, kv);
      const std::vector<float> lg = m.logits();
      for (float f : lg) { uint32_t u; std::memcpy(&u, &f, 4); logit_fnv = (logit_fnv ^ u) * 1099511628211ULL; }
      S.expand(p.leaf, PuctSearch::priors_from_logits(lg, 4, 1.25f, S.puct().nodes[p.leaf].seed), &off);
      S.backup(p, PUCT_Q20 / 2, &off);
      return true;
    };
    uint64_t f0 = 1469598103934665603ULL, fa = f0, fb = f0;
    bool ok = real_iter(f0);
    ok = ok && J.checkpoint(0, S, w, kv, err);
    const size_t l_ckpt = off.length();
    ok = ok && real_iter(fa) && real_iter(fa);
    const uint64_t da = S.digest();
    const auto oa = off.head();
    const size_t la = off.length();
    ok = ok && J.step_back(0, S, w, kv, err);
    const bool off_untouched = off.head() == oa && off.length() == la;
    ok = ok && real_iter(fb) && real_iter(fb);
    check(ok && S.digest() == da && fa == fb,
          "real: checkpoint, 2 forwards, step_back, redo -> same digest and leaf logits");
    check(ok && off_untouched && off.length() == la + (la - l_ckpt) && off.restores() == 0,
          "real: direct step_back leaves the official chain append-only");
    m.release(w);
  }
  m.release(root);
  check(m.pool_swa().live_pages() + m.pool_glob().live_pages() == 0, "real: no KV page leak");
}

int main(int argc, char** argv) {
  std::printf("[JOINED-STEPBACK] joined step-back - synthetic parts are LABELED UNIT-TEST FIXTURES ONLY\n");
  std::printf("[JOINED-STEPBACK] joined step-back is scratch-chain only; official chain append-only\n");
  test_unit_joined();
  if (!(argc > 1 && std::string(argv[1]) == "--unit")) test_real_little_nemo();
  else std::printf("[JOINED-STEPBACK] --unit: real Little Nemo part skipped\n");
  std::printf("[JOINED-STEPBACK] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
