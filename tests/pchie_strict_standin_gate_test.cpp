// Failing unit tests for StrictStandInGate (pre-hash-chain item 1). No model, no GPU, token-id only.
// TEST STAND-IN — not the PchGate safety gate. Written test-first against the stub header.
#include "pchie/StrictStandInGate.h"
#include "pchie/AcceptAllGate.h"
#include "pchie/pchie_tree.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("[STRICT] %-72s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}

static std::vector<int32_t> kRejectSet() {
    return DefaultStrictStandInRejectIds();
}

static void test_pass_shape_and_membership() {
    StrictStandInGate g(kRejectSet());
    AcceptAllGate accept;

    // Same Pass(const vector<int32_t>&) shape as AcceptAllGate (compiles via call sites).
    check(accept.Pass({}) == true, "AcceptAllGate::Pass({}) == true (control)");
    check(g.Pass({}) == true, "StrictStandInGate::Pass({}) == true (empty path / root)");

    // Known rejects (set A + set B).
    for (int32_t id : {0, 2, 3, 4, 46, 48, 50, 100, 105, 818, 63411, 63937}) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "Pass({%d}) == false (reject-set member)", id);
        check(g.Pass(std::vector<int32_t>{id}) == false, buf);
    }

    // Data control: Pass({1003})==true (1003 = about-token, not in set); Pass({63937})==false.
    check(g.Pass(std::vector<int32_t>{1003}) == true, "Pass({1003}) == true (not in reject set)");
    check(g.Pass(std::vector<int32_t>{63937}) == false, "Pass({63937}) == false (Data control)");

    // Any-member rule: path hits if one token is in the set.
    check(g.Pass(std::vector<int32_t>{1003, 63937, 496}) == false, "Pass(path) false if any token is rejected");
    check(g.Pass(std::vector<int32_t>{1003, 496, 2364}) == true, "Pass(path) true if no token is rejected");
}

static void test_reject_count_deterministic() {
    StrictStandInGate g(kRejectSet());
    check(g.RejectCount() == 0, "RejectCount starts at 0");

    (void)g.Pass(std::vector<int32_t>{1003});   // accept
    check(g.RejectCount() == 0, "RejectCount unchanged on Pass==true");

    (void)g.Pass(std::vector<int32_t>{63937});  // reject
    check(g.RejectCount() == 1, "RejectCount == 1 after first reject");
    check(g.LastRejectToken() == 63937, "LastRejectToken records first offending id");

    (void)g.Pass(std::vector<int32_t>{818, 1003});
    check(g.RejectCount() == 2, "RejectCount increments on each Pass==false");
    check(g.LastRejectToken() == 818, "LastRejectToken is first hit in path order");

    // Determinism: identical call sequence → identical count (no thread races in unit scope).
    StrictStandInGate g2(kRejectSet());
    (void)g2.Pass(std::vector<int32_t>{1003});
    (void)g2.Pass(std::vector<int32_t>{63937});
    (void)g2.Pass(std::vector<int32_t>{818, 1003});
    check(g2.RejectCount() == g.RejectCount(), "RejectCount deterministic across identical Pass sequences");
}

static void test_reject_set_hash() {
    StrictStandInGate a(kRejectSet());
    StrictStandInGate b(kRejectSet());
    StrictStandInGate c({0, 2, 3});                 // different set
    StrictStandInGate d({3, 2, 0, 0, -1});          // same members as c after sort/unique/drop neg

    check(a.RejectSetHash() != 0, "RejectSetHash is nonzero for a nonempty set");
    check(a.RejectSetHash() == b.RejectSetHash(), "RejectSetHash equal for identical sets");
    check(a.RejectSetHash() != c.RejectSetHash(), "RejectSetHash changes when the set changes");
    check(c.RejectSetHash() == d.RejectSetHash(), "RejectSetHash ignores order, dupes, negatives");

    // reject_count must NOT feed reject_set_hash (diagnostic only; not part of run hash inputs).
    const uint64_t h0 = a.RejectSetHash();
    (void)a.Pass(std::vector<int32_t>{63937});
    (void)a.Pass(std::vector<int32_t>{818});
    check(a.RejectCount() >= 2, "precondition: RejectCount advanced");
    check(a.RejectSetHash() == h0, "reject_count does not feed RejectSetHash / run-hash inputs");
}

static void test_empty_bypass_negative_control() {
    StrictStandInGate bypass({});
    StrictStandInGate strict(kRejectSet());

    check(bypass.IsBypass() == true, "empty reject set IsBypass()==true (strict-stand-in-bypass)");
    check(strict.IsBypass() == false, "nonempty reject set IsBypass()==false");

    // Negative control: a "strict" slot wired to empty set must fail the run.
    // Unit-level harness assertion pattern (no model):
    const bool run_ok = !bypass.IsBypass();   // would also require reject_count>0 after a live search
    check(run_ok == false, "bypassed (empty) reject set fails the run (negative control)");
}

static void test_default_set_contents() {
    auto ids = DefaultStrictStandInRejectIds();
    StrictStandInGate g(ids);
    // After sort/unique the committed set must contain A and B markers.
    const auto& r = g.RejectIds();
    auto has = [&](int32_t x) {
        return std::binary_search(r.begin(), r.end(), x);
    };
    bool all = true;
    for (int32_t x : {0, 2, 3, 4, 46, 48, 50, 100, 105, 818, 63411, 63937}) all = all && has(x);
    check(all && r.size() == 12, "DefaultStrictStandInRejectIds has A+B (12 ids)");
    // Terminal stops intentionally excluded from reject set.
    check(!has(1) && !has(106), "eos=1 and <turn|>=106 excluded from reject set");
}

int main() {
    test_default_set_contents();
    test_pass_shape_and_membership();
    test_reject_count_deterministic();
    test_reject_set_hash();
    test_empty_bypass_negative_control();
    std::printf("[PCHIE GATE] strict stand-in: %s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
