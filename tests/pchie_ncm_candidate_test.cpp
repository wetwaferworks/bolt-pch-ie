// Bolt PCHIE - NcmCandidateGenerator: score then id, 16-token cap.
// Synthetic concepts are a LABELED UNIT-TEST FIXTURE ONLY (never a golden or baseline).
// Real-weight path uses Little Nemo logits. CPU only, no CUDA, no golden lock.
#include "pchie/NcmCandidateGenerator.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
  std::printf("[NCM] %-72s %s\n", what, ok ? "PASS" : "FAIL");
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

static bool order_ok(const std::vector<NcmConcept>& c) {
  for (size_t i = 1; i < c.size(); ++i) {
    if (c[i - 1].score() > c[i].score()) continue;
    if (c[i - 1].score() == c[i].score() && c[i - 1].id() < c[i].id()) continue;
    return false;
  }
  return true;
}

static void test_concept_value_and_cap() {
  check(kNcmMaxSpan == 16 && NcmCandidateGenerator().max_span() == 16,
        "max_span is 16 tokens");
  const NcmConcept a(3, "latch", {10, 11, 12}, 0.5);
  check(a.id() == 3 && a.name() == "latch" && a.tokens().size() == 3 && a.score() == 0.5,
        "NcmConcept stores id, name, tokens, score");
  check(a.tokens()[0] == 10 && a.tokens()[2] == 12,
        "token expansion of a concept is the stored span, in order");

  NcmCandidateGenerator g;
  check(g.add(NcmConcept(1, "ok16", std::vector<int32_t>(16, 7), 1.0)) == true,
        "add accepts a 16-token span (at the cap)");
  check(g.add(NcmConcept(2, "too_long", std::vector<int32_t>(17, 7), 2.0)) == false,
        "add rejects a 17-token span (over the cap)");
  check(g.add(NcmConcept(3, "empty", {}, 1.0)) == false,
        "add rejects an empty token span");
  const auto& cs = g.candidates();
  bool only16 = cs.size() == 1 && cs[0].id() == 1 && cs[0].tokens().size() == 16;
  check(only16, "candidates() keeps only the in-cap concept");
}

static void test_score_then_id_and_ties() {
  // Labeled unit-test fixture: synthetic scores, no model.
  NcmCandidateGenerator g;
  check(g.add(NcmConcept(5, "b", {2}, 1.0)), "add id 5 score 1.0");
  check(g.add(NcmConcept(2, "a", {1}, 1.0)), "add id 2 score 1.0 (tie)");
  check(g.add(NcmConcept(9, "c", {3}, 2.0)), "add id 9 score 2.0");
  check(g.add(NcmConcept(1, "d", {4}, 0.25)), "add id 1 score 0.25");
  const auto& cs = g.candidates();
  check(cs.size() == 4, "four accepted concepts");
  check(cs.size() == 4 && cs[0].id() == 9 && cs[1].id() == 2 && cs[2].id() == 5 && cs[3].id() == 1,
        "order is score descending, then id ascending (2 before 5 on a 1.0 tie)");
  check(order_ok(cs), "pairwise score-then-id order holds");
}

static void test_negative_nan_inf_and_empty_short() {
  NcmCandidateGenerator g;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double pinf = std::numeric_limits<double>::infinity();
  const double ninf = -std::numeric_limits<double>::infinity();
  check(g.add(NcmConcept(4, "neg", {1}, -1.5)) == true, "add accepts a negative finite score");
  check(g.add(NcmConcept(3, "nan", {2}, nan)) == false, "add rejects a NaN score");
  check(g.add(NcmConcept(1, "pinf", {3}, pinf)) == true, "add accepts +Inf (sorts first)");
  check(g.add(NcmConcept(2, "ninf", {4}, ninf)) == true, "add accepts -Inf (sorts last)");
  check(g.add(NcmConcept(5, "zero", {5}, 0.0)) == true, "add accepts score 0");
  const auto& cs = g.candidates();
  // Expected: +Inf id 1, 0.0 id 5, -1.5 id 4, -Inf id 2. NaN dropped.
  bool ok = cs.size() == 4 &&
            cs[0].id() == 1 && std::isinf(cs[0].score()) && cs[0].score() > 0 &&
            cs[1].id() == 5 && cs[1].score() == 0.0 &&
            cs[2].id() == 4 && cs[2].score() == -1.5 &&
            cs[3].id() == 2 && std::isinf(cs[3].score()) && cs[3].score() < 0;
  check(ok, "NaN dropped; +Inf first, then 0, then -1.5, then -Inf");
  check(order_ok(cs) || cs.size() != 4, "finite/Inf order is score descending then id");

  NcmCandidateGenerator empty;
  check(empty.candidates().empty(), "empty generator has no candidates");
  check(empty.propose_from_logits({}, 8) == true, "propose_from_logits on empty logits succeeds");
  check(empty.candidates().empty(), "empty logits yields no candidates");

  NcmCandidateGenerator one;
  const std::vector<float> short_lg = {0.75f};
  check(one.propose_from_logits(short_lg, 0) == true, "propose_from_logits on a 1-logit vector succeeds");
  check(one.candidates().size() == 1 && one.candidates()[0].id() == 0 &&
            one.candidates()[0].tokens() == std::vector<int32_t>{0} &&
            one.candidates()[0].score() == 0.75,
        "short input: one 1-token concept, id=0, score=logit");
}

static void test_propose_from_logits_fixture() {
  // Labeled unit-test fixture: tiny logit vector, no model.
  const std::vector<float> lg = {0.1f, 2.0f, 2.0f, -0.5f};
  NcmCandidateGenerator a, b;
  check(a.propose_from_logits(lg, 0) && b.propose_from_logits(lg, 0),
        "two propose_from_logits calls succeed");
  check(a.candidates() == b.candidates() && a.candidates().size() == 4,
        "propose_from_logits is deterministic and keeps all finite logits");
  const auto& cs = a.candidates();
  // scores 2.0,2.0,0.1,-0.5 -> ids 1,2 (tie, lower id first), then 0, then 3
  check(cs.size() == 4 && cs[0].id() == 1 && cs[1].id() == 2 && cs[2].id() == 0 && cs[3].id() == 3,
        "logit ties break by lower token id; all spans have length 1");
  check(!cs.empty() && cs[0].tokens().size() == 1 && cs[0].tokens()[0] == 1, "concept tokens() is {id}");

  NcmCandidateGenerator top;
  check(top.propose_from_logits(lg, 2) == true && top.candidates().size() == 2 &&
            !top.candidates().empty() && top.candidates()[0].id() == 1 &&
            top.candidates().size() >= 2 && top.candidates()[1].id() == 2,
        "max_candidates=2 keeps the top two after score-then-id");

  std::vector<float> with_nan = {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.5f};
  NcmCandidateGenerator n;
  check(n.propose_from_logits(with_nan, 0) == true && n.candidates().size() == 2 &&
            n.candidates().size() >= 2 && n.candidates()[0].id() == 0 && n.candidates()[1].id() == 2,
        "NaN logit entries are skipped, remaining stay ordered");
}

static void test_real_little_nemo() {
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight NCM: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[NCM] load err: %s\n", err.c_str());
    return;
  }
  const auto ids = m.tokenize("Hi");
  if (ids.empty()) {
    check(false, "real-weight NCM: tokenize(\"Hi\") nonempty");
    return;
  }
  m.init_cache(/*max_ctx*/32, /*branches*/2);
  ModelBranch br = m.new_branch();
  for (int32_t t : ids) m.forward(br, t);  // BOS policy: full prompt {BOS, Hi}
  const auto logits = m.logits();
  check(!logits.empty(), "real-weight NCM: logits present");

  NcmCandidateGenerator a, b;
  check(a.propose_from_logits(logits, 8) && b.propose_from_logits(logits, 8),
        "real Little Nemo propose_from_logits(K=8) succeeds twice");
  check(a.candidates() == b.candidates() && a.candidates().size() == 8,
        "real Little Nemo candidates are run-to-run deterministic, size 8");
  check(order_ok(a.candidates()), "real Little Nemo order is score descending then id");
  bool spans_ok = true;
  for (const auto& c : a.candidates()) {
    if (c.tokens().size() != 1 || c.tokens().size() > kNcmMaxSpan) spans_ok = false;
  }
  check(spans_ok, "real Little Nemo proposed spans are length 1 and within the cap");

  std::vector<uint32_t> id_list;
  id_list.reserve(a.candidates().size());
  for (const auto& c : a.candidates()) id_list.push_back(c.id());
  const auto bytes = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(id_list.data()),
                                          reinterpret_cast<const uint8_t*>(id_list.data()) +
                                              id_list.size() * sizeof(uint32_t));
  const auto sha = Sha256::hash(bytes.data(), bytes.size());
  const uint64_t fnv = fnv_bytes(bytes.data(), bytes.size());
  bool collide8 = sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
  check(!collide8, "candidate-id FNV-1a64 != first 8 bytes of SHA-256");
  if (collide8)
    std::printf("[NCM] ALERT: FNV vs SHA disagreement FAILED on NCM ids -- report it immediately\n");
  std::printf("[NCM] ids sha256=%s fnv=0x%016llX first_id=%u score=%.6f\n",
              GgufWeightsHash::to_hex(sha).c_str(), (unsigned long long)fnv,
              a.candidates().empty() ? 0u : a.candidates()[0].id(),
              a.candidates().empty() ? 0.0 : a.candidates()[0].score());
  m.release(br);
}

int main() {
  std::printf("[PCH] NcmCandidateGenerator tests (score then id, max_span=16)\n");
  test_concept_value_and_cap();
  test_score_then_id_and_ties();
  test_negative_nan_inf_and_empty_short();
  test_propose_from_logits_fixture();
  test_real_little_nemo();
  std::printf("[PCH] NcmCandidateGenerator %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
