// Bolt PCHIE - NCM concepts (incl. multi-token spans) become HTN methods
// and PUCT edges. Synthetic parts are a LABELED UNIT-TEST FIXTURE ONLY (never a golden or
// baseline). The live part uses real Little Nemo weights on CPU. No golden is locked.
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/HtnCondition.h"
#include "pchie/HtnDomain.h"
#include "pchie/HtnPlanner.h"
#include "pchie/HtnState.h"
#include "pchie/NcmMethodBinder.h"
#include "pchie/NcmSpanProposer.h"
#include "pchie/NcmSpanTrie.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchGate.h"
#include "pchie/PchStack.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/ModelPath.h"

#include <cmath>
#include <cstdio>
#include <limits>
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
  std::printf("[NCM-JOIN] %-74s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

// Verbatim copy of the P4 PchStack::edges_from_ncm algorithm (1-token concepts only), used
// to prove the trie reproduces it for 1-token sets.
static std::vector<PuctEdge> legacy_edges(const std::vector<NcmConcept>& cs, float temp, uint64_t seed) {
  std::vector<float> lg;
  std::vector<int32_t> toks;
  for (const auto& c : cs) {
    if (c.tokens().size() != 1) continue;
    lg.push_back(static_cast<float>(c.score()));
    toks.push_back(c.tokens()[0]);
  }
  if (lg.empty() || temp <= 0.0f) return {};
  const auto raw = PuctSearch::priors_from_logits(lg, int(lg.size()), temp, seed);
  std::vector<PuctEdge> e;
  for (PuctEdge ed : raw) {
    if (ed.token < 0 || size_t(ed.token) >= toks.size()) continue;
    ed.token = toks[size_t(ed.token)];
    e.push_back(ed);
  }
  std::sort(e.begin(), e.end(), [](const PuctEdge& a, const PuctEdge& b) { return a.token < b.token; });
  return e;
}
static bool same_edges(const std::vector<PuctEdge>& a, const std::vector<PuctEdge>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].token != b[i].token || a[i].prior_q20 != b[i].prior_q20 || a[i].child != b[i].child) return false;
  return true;
}

enum : uint32_t { tG = 1, tSay = 2, tWave = 3 };
static HtnDomain small_domain() {
  HtnDomain d;
  (void)d.add_task(HtnTask(tG, "Greet", false));
  (void)d.add_task(HtnTask(tSay, "Say", true));
  (void)d.add_task(HtnTask(tWave, "Wave", true));
  (void)d.add_method(HtnMethod(10, "SayHi", tG, {tSay}));
  return d;
}

static void test_binder() {
  HtnDomain d = small_domain();
  NcmMethodBinder b;
  std::string err;
  const NcmConcept c(7, "hello_there", {11, 22, 33}, 1.5);
  check(b.bind(d, c, tG, err), "bind 3-token concept as a method on compound Greet");
  const uint32_t tid = NcmMethodBinder::concept_task_id(7);
  check(d.find_task(tid) && d.find_task(tid)->primitive() && d.has_method(NcmMethodBinder::method_id(7)),
        "concept task (primitive) and method exist at reserved ids");
  check(b.concept_for_task(tid) && b.concept_for_task(tid)->tokens() == std::vector<int32_t>({11, 22, 33}),
        "binder resolves concept task -> span");
  // Default priority 0: authored lower id still wins (bound methods never silently override).
  std::string e2;
  HtnDecomposition o;
  check(d.decompose_with_state(tG, HtnState(), o, e2) && o.tasks.size() == 1 && o.tasks[0].id() == tSay,
        "priority 0 bound method loses to authored lower id");
  check(!b.bind(d, c, tG, err) && err == "method id taken", "rebinding the same concept fails closed");
  const NcmConcept c8(8, "", {44}, 0.0);
  check(b.bind(d, c8, tG, err, HtnCondition({5}, {}), /*priority*/3), "bind with precondition and priority 3");
  check(d.decompose_with_state(tG, HtnState({5}), o, e2) && o.tasks[0].id() == NcmMethodBinder::concept_task_id(8),
        "precondition holds + priority: concept method chosen over authored");
  check(d.decompose_with_state(tG, HtnState(), o, e2) && o.tasks[0].id() == tSay,
        "precondition false: authored method chosen");
  // Rejections leave the domain untouched.
  struct Bad { NcmConcept c; uint32_t head; const char* why; };
  std::vector<int32_t> seventeen(17, 1);
  const Bad bad[] = {
      {NcmConcept(20, "", {}, 0.0), tG, "empty span"},
      {NcmConcept(21, "", seventeen, 0.0), tG, "17-token span"},
      {NcmConcept(22, "", {-1}, 0.0), tG, "negative token"},
      {NcmConcept(23, "", {1}, std::numeric_limits<double>::quiet_NaN()), tG, "NaN score"},
      {NcmConcept(24, "", {1}, 0.0), tSay, "primitive head"},
      {NcmConcept(25, "", {1}, 0.0), 999, "unknown head"},
      {NcmConcept(26, "", {1}, 0.0), NcmMethodBinder::concept_task_id(7), "concept-task head"},
      {NcmConcept(0x20000000u, "", {1}, 0.0), tG, "concept id out of range"},
  };
  bool all_rejected = true, untouched = true;
  for (const Bad& x : bad) {
    if (b.bind(d, x.c, x.head, err)) all_rejected = false;
    if (x.c.id() <= NcmMethodBinder::kNcmMaxConceptId &&
        (d.find_task(NcmMethodBinder::concept_task_id(x.c.id())) || d.has_method(NcmMethodBinder::method_id(x.c.id()))))
      untouched = false;
  }
  check(all_rejected, "binder rejects empty/17-token/negative/NaN/primitive/unknown/concept head/id range");
  check(untouched, "rejected binds add no task and no method (fail closed, no partial mutation)");
  HtnDomain d2 = small_domain();
  check(!b.bind(d2, NcmConcept(7, "other", {99}, 0.0), tG, err), "same concept id with a different span is refused");
  check(!d.add_method(HtnMethod(NcmMethodBinder::method_id(9), "x", tG, {tWave})) ||
            !b.bind(d, NcmConcept(9, "", {1}, 0.0), tG, err),
        "a hand-added method in the reserved range blocks the bind");
}

static void test_trie_identity_and_spans() {
  // 1-token concepts: trie root == legacy edges_from_ncm, for several seeds/temps.
  std::vector<float> lg = {0.5f, 3.0f, -1.0f, 3.0f, 2.25f, -7.0f, 0.0f, 1.0f};
  NcmCandidateGenerator g;
  (void)g.propose_from_logits(lg, 6);
  bool same = true;
  for (uint64_t seed : {0ull, 7ull, 0xDEADBEEFull})
    for (float t : {1.0f, 1.25f, 0.5f}) {
      NcmSpanTrie tr;
      same &= tr.build(g.candidates(), t, seed) && same_edges(tr.edges_at({}), legacy_edges(g.candidates(), t, seed));
    }
  check(same, "1-token concepts: span-trie root edges == original edges_from_ncm (bit-exact)");

  // Multi-token spans.
  std::vector<NcmConcept> cs = {NcmConcept(1, "a", {5, 9, 2}, 2.0), NcmConcept(2, "b", {5, 9, 7}, 1.0),
                                NcmConcept(3, "c", {3}, 0.5)};
  NcmSpanTrie tr;
  check(tr.build(cs, 1.0f, 42), "trie builds from multi-token spans");
  const auto r = tr.edges_at({});
  const auto masses = legacy_edges({NcmConcept(1, "", {0}, 2.0), NcmConcept(2, "", {1}, 1.0), NcmConcept(3, "", {2}, 0.5)}, 1.0f, 42);
  check(r.size() == 2 && r[0].token == 3 && r[1].token == 5 && r[0].prior_q20 == masses[2].prior_q20 &&
            r[1].prior_q20 == masses[0].prior_q20 + masses[1].prior_q20,
        "root: one edge per first token; shared prefix sums concept masses");
  const auto n5 = tr.edges_at({5});
  check(n5.size() == 1 && n5[0].token == 9 && n5[0].prior_q20 == PUCT_Q20, "mid-span single continuation gets prior 1.0");
  const auto n59 = tr.edges_at({5, 9});
  check(n59.size() == 2 && n59[0].token == 2 && n59[1].token == 7 && n59[0].prior_q20 > n59[1].prior_q20 &&
            n59[0].prior_q20 + n59[1].prior_q20 <= PUCT_Q20 && n59[0].prior_q20 + n59[1].prior_q20 >= PUCT_Q20 - 2,
        "branch inside span: renormalized integer priors, higher score first");
  check(tr.continues({5}) && tr.continues({5, 9}) && !tr.continues({5, 9, 2}) && tr.ends({5, 9, 2}) && tr.ends({3}),
        "continues/ends report span structure");
  check(tr.edges_at({5, 9, 2}).empty() && tr.edges_at({4}).empty(), "complete or unknown prefix has no trie edges");
  NcmSpanTrie tr2;
  (void)tr2.build({NcmConcept(1, "", {5}, 1.0), NcmConcept(2, "", {5, 6}, 1.0)}, 1.0f, 1);
  check(tr2.stop_mass_dropped() == 1, "concept ending inside a longer one is counted (longest-span policy)");
  NcmSpanTrie bad;
  check(!bad.build({}, 1.0f, 0) && !bad.build(cs, 0.0f, 0) &&
            !bad.build({NcmConcept(1, "", std::vector<int32_t>(17, 1), 0.0)}, 1.0f, 0) &&
            !bad.build({NcmConcept(1, "", {1}, std::numeric_limits<double>::quiet_NaN())}, 1.0f, 0) &&
            !bad.build({NcmConcept(1, "", {-3}, 0.0)}, 1.0f, 0) && bad.empty(),
        "trie rejects empty set, temp<=0, 17 tokens, NaN, negative token");
}

static std::vector<float> fixture_logits() {
  std::vector<float> lg(64);
  for (size_t i = 0; i < lg.size(); ++i) lg[i] = float((i * 37) % 23) * 0.25f - 2.0f;
  return lg;
}

static void test_stack_unit_fixture() {
  // LABELED UNIT-TEST FIXTURE: no IModel, synthetic logits, leaves reuse root logits.
  HtnDomain d = small_domain();
  NcmMethodBinder b;
  std::string err;
  const NcmConcept span(40, "four", {12, 34, 56, 9}, 0.0);
  (void)b.bind(d, span, tG, err, HtnCondition(), /*priority*/1);
  const auto lg = fixture_logits();
  const std::vector<int32_t> prompt = {5, 6};  // not in the provisional PchGate reject set
  auto run = [&](bool span_mode, int budget, PchStackResult& r, std::vector<uint8_t>& head) {
    PchStack s;
    s.set_goal(tG);
    s.set_budget(budget);
    s.set_span_mode(span_mode);
    s.set_concept_binder(&b);
    s.set_plan_state(HtnState());  // H2: priority-1 concept method beats the authored one
    Sha256HashChain c;
    c.reset();
    const bool ok = s.run_unit_fixture(d, "", lg, prompt, &c, r);
    head = c.head();
    return ok;
  };
  PchStackResult a, a2, z, small;
  std::vector<uint8_t> ha, ha2, hz, hs;
  check(run(true, 6, a, ha) && a.ok && a.concept_tasks == 1, "span mode: concept task searched along its span");
  check(a.committed.size() >= 4 && std::equal(span.tokens().begin(), span.tokens().end(), a.committed.begin()),
        "committed tokens start with the concept span 12,34,56,9");
  check(a.span_expansions >= 3, "mid-span nodes were expanded from the trie");
  check(run(true, 6, a2, ha2) && ha == ha2 && a.puct_digest == a2.puct_digest && a.committed == a2.committed,
        "span mode is run-to-run deterministic (chain head, digest, commit)");
  check(!run(false, 6, z, hz) && z.err == "concept task needs span mode", "concept task without span mode fails closed");
  check(!run(true, 2, small, hs) && small.err.find("concept span not committed") != std::string::npos,
        "budget below span length fails closed (no partial span claimed)");
  {
    // A concept whose span contains a gate-rejected token (46) must not be claimed.
    HtnDomain dg = small_domain();
    NcmMethodBinder bg;
    (void)bg.bind(dg, NcmConcept(41, "bad", {12, 46, 9}, 0.0), tG, err, HtnCondition(), 1);
    PchStack s;
    s.set_goal(tG);
    s.set_budget(6);
    s.set_span_mode(true);
    s.set_concept_binder(&bg);
    s.set_plan_state(HtnState());
    PchStackResult rg;
    check(!s.run_unit_fixture(dg, "", lg, prompt, nullptr, rg) &&
              rg.err.find("concept span not committed") != std::string::npos && rg.gate_rejects > 0,
          "concept span through a PchGate-rejected token fails closed");
  }
  {
    // Legacy planning (no plan state) keeps lowest-id authored method even in span mode.
    PchStack s;
    s.set_goal(tG);
    s.set_budget(6);
    s.set_span_mode(true);
    s.set_concept_binder(&b);
    PchStackResult rl;
    check(s.run_unit_fixture(d, "", lg, prompt, nullptr, rl) && rl.plan.size() == 1 && rl.plan[0].id() == tSay &&
              rl.concept_tasks == 0,
          "without set_plan_state the legacy lowest-id plan is used (opt-in only)");
  }
  // Root spans merged with 1-token candidates on an authored task.
  HtnDomain d1 = small_domain();
  PchStack s;
  s.set_goal(tG);
  s.set_budget(6);
  s.set_span_mode(true);
  s.set_root_spans({NcmConcept(900, "hot", {7, 8, 9}, 50.0)});
  PchStackResult rs;
  check(s.run_unit_fixture(d1, "", lg, prompt, nullptr, rs) && rs.span_expansions >= 2 &&
            rs.committed.size() >= 3 && rs.committed[0] == 7 && rs.committed[1] == 8 && rs.committed[2] == 9,
        "root span with dominant score becomes a multi-token PUCT path and is committed");
}

// ---------------------------------------------------------------------------
// N2 live sanity. Test-only helpers.
// A span is degenerate ("pure repeat") if it is all one token (len >= 2) or period-2
// (t[i] == t[i-2] for every i >= 2, len >= 3), e.g. the old BOS-less "Hi Hi Hi" / "** Hi **".
static bool span_is_pure_repeat(const std::vector<int32_t>& t) {
  if (t.size() >= 2) {
    bool all = true;
    for (size_t i = 1; i < t.size(); ++i)
      if (t[i] != t[0]) all = false;
    if (all) return true;
  }
  if (t.size() >= 3) {
    bool p2 = true;
    for (size_t i = 2; i < t.size(); ++i)
      if (t[i] != t[i - 2]) p2 = false;
    if (p2) return true;
  }
  return false;
}

// The root is collapsed (copy loop) if its top-1 softmax probability is > 0.99 and the top-1
// token is the last prompt token. Lowest id wins ties.
static bool root_is_collapsed(const std::vector<float>& lg, const std::vector<int32_t>& prompt,
                              double* p_top, int32_t* top_id) {
  if (lg.empty()) return true;
  size_t best = 0;
  for (size_t i = 1; i < lg.size(); ++i)
    if (lg[i] > lg[best]) best = i;
  double sum = 0.0;
  for (float x : lg) sum += std::exp(double(x) - double(lg[best]));
  const double p = 1.0 / sum;
  if (p_top) *p_top = p;
  if (top_id) *top_id = int32_t(best);
  return p > 0.99 && !prompt.empty() && int32_t(best) == prompt.back();
}

// Collapsed span set: every multi-token span has the same second token (the old set all
// continued with "Hi").
static bool spans_share_one_continuation(const std::vector<NcmConcept>& s) {
  int32_t second = -1;
  size_t multi = 0;
  for (const auto& c : s) {
    if (c.tokens().size() < 2) continue;
    ++multi;
    if (second < 0) second = c.tokens()[1];
    else if (c.tokens()[1] != second) return false;
  }
  return multi >= 2;
}

static bool spans_sane(const std::vector<NcmConcept>& s) {
  if (spans_share_one_continuation(s)) return false;
  for (const auto& c : s)
    if (span_is_pure_repeat(c.tokens())) return false;
  return true;
}

// Labeled unit-test fixture: the four degenerate spans and the collapsed root the old
// BOS-less live test produced must be rejected by the sanity check.
static void test_n2_sanity_fixture() {
  const std::vector<NcmConcept> old = {NcmConcept(1, "a", {10979, 10979, 10979}, 7.1459),
                                       NcmConcept(2, "b", {1018, 10979, 1018}, -2.6395),
                                       NcmConcept(3, "c", {3586, 10979, 3586}, -2.9851),
                                       NcmConcept(4, "d", {236772, 10979, 236772}, -3.9935)};
  bool each = true;
  for (const auto& c : old) each = each && span_is_pure_repeat(c.tokens());
  check(each && !spans_sane(old), "N2 sanity (fixture): the old BOS-less spans are all rejected");
  const std::vector<NcmConcept> good = {NcmConcept(5, "e", {1, 2, 3}, 1.0), NcmConcept(6, "f", {4, 5}, 0.5),
                                        NcmConcept(7, "g", {8}, 0.1)};
  check(spans_sane(good) && !span_is_pure_repeat({7, 8, 7, 9}) && span_is_pure_repeat({7, 8, 7, 8}),
        "N2 sanity (fixture): varied spans pass; period-2 is caught, near-miss is not");
  std::vector<float> lg(16, 0.0f);
  lg[5] = 20.0f;  // p(top) ~ 1
  double p = 0.0;
  int32_t top = -1;
  check(root_is_collapsed(lg, {5}, &p, &top) && !root_is_collapsed(lg, {6}, &p, &top),
        "N2 sanity (fixture): collapsed root = top-1 p > 0.99 on the last prompt token");
}

static void print_span_text(const NemotronModel& m, const NcmConcept& c) {
  std::string s;
  for (int32_t t : c.tokens()) s += m.decode(t);
  std::printf("[NCM-JOIN] real span id=%u len=%zu score=%.4f tokens=", c.id(), c.tokens().size(), c.score());
  for (int32_t t : c.tokens()) std::printf("%d ", t);
  std::printf("text=\"");
  for (unsigned char ch : s) {
    if (ch < 0x20 || ch == '"' || ch == '\\') std::printf("\\x%02X", ch);
    else std::printf("%c", ch);
  }
  std::printf("\"\n");
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[NCM-JOIN] load err: %s\n", err.c_str());
    return;
  }
  //  decision 2: a well-formed prompt. tokenize now honors add_bos_token, so
  // "Hi" becomes {<bos>, Hi}. The chat template is not used here because its <|turn> (105) is
  // in the provisional stand-in reject set, which would gate-reject every PchStack path.
  // The gated path prefix handed to PchStack excludes the runtime BOS anchor (id 2 is also in
  // the stand-in reject set; open item for Archie).
  const auto prompt = m.tokenize("Hi");
  const auto gated = m.tokenize_no_bos("Hi");
  check(prompt.size() == 2 && prompt[0] == 2 && prompt[1] == 10979 && gated.size() == 1 && gated[0] == 10979,
        "real: N2 prompt is {2 BOS, 10979 Hi} (tokenize honors add_bos_token)");
  m.init_cache(/*max_ctx*/32, /*branches*/16);
  ModelBranch root = m.new_branch();
  for (int32_t t : prompt) m.forward(root, t);
  const std::vector<float> root_logits = m.logits();
  const std::vector<float> root_hidden = m.hidden_state();
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  double p_top = 0.0;
  int32_t top_id = -1;
  const bool collapsed = root_is_collapsed(root_logits, prompt, &p_top, &top_id);
  std::printf("[NCM-JOIN] real root top-1 id=%d p=%.4f\n", top_id, p_top);
  check(!collapsed, "real: N2 root is not collapsed (no copy loop on the last prompt token)");

  NcmSpanProposer prop(4, 3);
  std::vector<NcmConcept> s1, s2;
  const bool p1 = prop.propose(m, root, root_logits, 1000, s1, err);
  const bool p2 = prop.propose(m, root, root_logits, 1000, s2, err);
  size_t multi = 0;
  for (const auto& c : s1) if (c.tokens().size() >= 2) ++multi;
  check(p1 && p2 && s1.size() == 4 && s1 == s2, "real: NcmSpanProposer gives 4 spans, deterministic");
  check(multi >= 1, "real: at least one multi-token span from Little Nemo");
  check(m.pool_swa().live_pages() + m.pool_glob().live_pages() == live0, "real: proposer releases every fork");
  for (const auto& c : s1) print_span_text(m, c);
  check(spans_sane(s1), "real: N2 spans are not collapsed or pure-repeat (sanity check)");

  // Bind the best multi-token span that the stand-in gate passes as a method (priority 1) and
  // run the stack in span mode. (A span through a rejected token fails closed; that case is
  // unit-tested above, so it is not the live N1 subject.)
  const NcmConcept* best = nullptr;
  const PchGate span_gate;
  for (const auto& c : s1)
    if (c.tokens().size() >= 2 && span_gate.Pass(ProposalTokens::from_proposal(c.tokens())) && (!best || c.score() > best->score())) best = &c;
  HtnDomain d = small_domain();
  NcmMethodBinder b;
  if (!best || !b.bind(d, *best, tG, err, HtnCondition(), 1)) {
    check(false, "real: bind best gate-passing multi-token span");
    m.release(root);
    return;
  }
  std::printf("[NCM-JOIN] real N1 bound span id=%u\n", best->id());
  auto run = [&](PchStackResult& r, std::vector<uint8_t>& head, bool spans_at_root) {
    PchStack s;
    s.set_goal(tG);
    s.set_budget(4);
    s.set_span_mode(true);
    s.set_concept_binder(&b);
    s.set_plan_state(HtnState());
    if (spans_at_root) s.set_root_spans(s1);
    Sha256HashChain c;
    c.reset();
    const bool ok = s.run(m, root, gated, root_logits, root_hidden, spans_at_root ? small_domain() : d, "", &c, r);
    head = c.head();
    return ok;
  };
  PchStackResult r1, r2, q1, q2;
  std::vector<uint8_t> h1, h2, g1, g2;
  const bool ok1 = run(r1, h1, false), ok2 = run(r2, h2, false);
  if (!ok1) std::printf("[NCM-JOIN] real concept run err: %s\n", r1.err.c_str());
  check(ok1 && ok2 && r1.concept_tasks == 1 &&
            std::equal(best->tokens().begin(), best->tokens().end(), r1.committed.begin()),
        "real: HTN concept method -> forced PUCT span, committed with real forwards");
  check(h1 == h2 && r1.puct_digest == r2.puct_digest && r1.committed == r2.committed,
        "real: concept-method run is deterministic");
  const bool okq1 = run(q1, g1, true), okq2 = run(q2, g2, true);
  if (!okq1) std::printf("[NCM-JOIN] real root-span run err: %s\n", q1.err.c_str());
  check(okq1 && okq2 && g1 == g2 && q1.committed == q2.committed && q1.last_ncm.size() >= s1.size(),
        "real: root spans merged with model candidates; deterministic");
  std::printf("[NCM-JOIN] real root-span run: span_expansions=%zu committed_len=%zu stop_dropped=%zu\n",
              q1.span_expansions, q1.committed.size(), q1.span_stop_dropped);
  m.release(root);
  check(m.pool_swa().live_pages() + m.pool_glob().live_pages() == 0, "real: no KV page leak after runs and root release");
}

int main(int argc, char** argv) {
  std::printf("[NCM-JOIN] NCM spans joined to HTN and PUCT - synthetic parts are LABELED UNIT-TEST FIXTURES ONLY\n");
  test_binder();
  test_trie_identity_and_spans();
  test_stack_unit_fixture();
  test_n2_sanity_fixture();
  if (!(argc > 1 && std::string(argv[1]) == "--unit")) test_real_little_nemo();
  else std::printf("[NCM-JOIN] --unit: real Little Nemo part skipped\n");
  std::printf("[NCM-JOIN] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
