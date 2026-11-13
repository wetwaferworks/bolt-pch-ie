// Bolt PCHIE - failing tests for the real PchGate (token-id only, no model, no CUDA).
// Reference reject set: DefaultStrictStandInRejectIds(). Coordinator-thread only.
#include "pchie/PchGate.h"
#include "pchie/StrictStandInGate.h"
#include "pchie/AcceptAllGate.h"
#include "pchie/LogProbLeafEvaluator.h"
#include "pchie/LeafContext.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[PCH-GATE] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static const std::vector<int32_t> kRef = DefaultStrictStandInRejectIds();

static void test_default_uses_reference_reject_set() {
  PchGate g;
  check(g.IsBypass() == false, "default PchGate IsBypass()==false");
  const auto& r = g.RejectIds();
  auto has = [&](int32_t x) { return std::binary_search(r.begin(), r.end(), x); };
  bool all = (r.size() == kRef.size());
  for (int32_t x : kRef) all = all && has(x);
  check(all, "default PchGate reject set equals DefaultStrictStandInRejectIds (12 ids)");
  check(!has(1) && !has(106), "eos=1 and <turn|>=106 stay excluded from PchGate reject set");
}

static void test_pass_shape_and_membership() {
  PchGate g;
  AcceptAllGate accept;
  check(accept.Pass({}) == true, "AcceptAllGate::Pass({}) == true (control)");
  check(g.Pass(ProposalTokens::from_proposal({})) == true, "PchGate::Pass({}) == true (empty path / root)");

  for (int32_t id : kRef) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "PchGate::Pass({%d}) == false (reference reject-set member)", int(id));
    check(g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{id})) == false, buf);
  }
  check(g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003})) == true, "Pass({1003}) == true (not in reject set)");
  check(g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{63937})) == false, "Pass({63937}) == false (Data control)");
  check(g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003, 63937, 496})) == false, "Pass(path) false if any token is rejected");
  check(g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003, 496, 2364})) == true, "Pass(path) true if no token is rejected");
}

static void test_reject_count_deterministic() {
  PchGate g;
  check(g.RejectCount() == 0, "RejectCount starts at 0");
  (void)g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003}));
  check(g.RejectCount() == 0, "RejectCount unchanged on Pass==true");
  (void)g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{63937}));
  check(g.RejectCount() == 1, "RejectCount == 1 after first reject");
  check(g.LastRejectToken() == 63937, "LastRejectToken records first offending id");
  (void)g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{818, 1003}));
  check(g.RejectCount() == 2, "RejectCount increments on each Pass==false");
  check(g.LastRejectToken() == 818, "LastRejectToken is first hit in path order");

  PchGate g2;
  (void)g2.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003}));
  (void)g2.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{63937}));
  (void)g2.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{818, 1003}));
  check(g2.RejectCount() == g.RejectCount(), "RejectCount deterministic across identical Pass sequences");
}

static void test_reject_set_hash_matches_standin_reference() {
  PchGate a;
  PchGate b;
  StrictStandInGate ref(kRef);
  PchGate c({0, 2, 3});
  PchGate d({3, 2, 0, 0, -1});

  check(a.RejectSetHash() != 0, "RejectSetHash is nonzero for the reference set");
  check(a.RejectSetHash() == b.RejectSetHash(), "RejectSetHash equal for identical default sets");
  check(a.RejectSetHash() == ref.RejectSetHash(),
        "PchGate RejectSetHash matches StrictStandInGate on the reference set");
  check(a.RejectSetHash() != c.RejectSetHash(), "RejectSetHash changes when the set changes");
  check(c.RejectSetHash() == d.RejectSetHash(), "RejectSetHash ignores order, dupes, negatives");

  const uint64_t h0 = a.RejectSetHash();
  (void)a.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{63937}));
  (void)a.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{818}));
  check(a.RejectCount() >= 2, "precondition: RejectCount advanced");
  check(a.RejectSetHash() == h0, "reject_count does not feed RejectSetHash");
}

static void test_empty_bypass_negative_control() {
  PchGate bypass(std::vector<int32_t>{});
  PchGate strict;
  check(bypass.IsBypass() == true, "empty reject set IsBypass()==true");
  check(strict.IsBypass() == false, "default (reference) set IsBypass()==false");
  const bool run_ok = !bypass.IsBypass();
  check(run_ok == false, "bypassed (empty) reject set fails the run (negative control)");
}

static void test_gate_multiplies_leaf_value() {
  PchGate g;
  LogProbLeafEvaluator ev(0.7, 0.0);
  LeafContext ctx;
  ctx.depth = 1;
  ctx.sum_log_prob = -0.5;
  ctx.entropy = 0.0;
  ctx.gate_pass = g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{1003}));
  check(ev.EvalLeaf(ctx) > 0.0, "EvalLeaf > 0 when PchGate passes a clean path");
  ctx.gate_pass = g.Pass(ProposalTokens::from_proposal(std::vector<int32_t>{63937}));
  check(ev.EvalLeaf(ctx) == 0.0, "EvalLeaf == 0 when PchGate rejects (G = 0)");
}

int main() {
  std::printf("[PCH] PchGate failing tests (reference reject set = StrictStandInGate)\n");
  test_default_uses_reference_reject_set();
  test_pass_shape_and_membership();
  test_reject_count_deterministic();
  test_reject_set_hash_matches_standin_reference();
  test_empty_bypass_negative_control();
  test_gate_multiplies_leaf_value();
  std::printf("[PCH] PchGate %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}