// Gate split test (2026-11-11).
// The gate sees ONLY proposal tokens. Compile-time: PchGate::Pass / PchCpuLeafPath::Evaluate no
// longer accept a raw token vector (so a prompt cannot be handed to the gate by accident), and
// ProposalTokens cannot be implicitly built from one. Run-time: a trusted prompt that carries BOS
// (id 2, in the reject set) no longer zeroes every leaf; a rejected PROPOSAL still does.
#include "pchie/PchCpuLeafPath.h"
#include "pchie/PchGate.h"

#include <cstdio>
#include <type_traits>
#include <utility>
#include <vector>

using namespace pchie;
static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  ok ? ++g_pass : ++g_fail;
}
template <class G, class = void> struct raw_pass : std::false_type {};
template <class G>
struct raw_pass<G, std::void_t<decltype(std::declval<const G&>().Pass(std::declval<const std::vector<int32_t>&>()))>> : std::true_type {};

int main() {
  std::printf("Gate split (gate sees proposal tokens only)\n");
  check(!raw_pass<PchGate>::value, "compile-time: PchGate::Pass(std::vector<int32_t>) is not callable");
  check(!std::is_convertible<std::vector<int32_t>, ProposalTokens>::value, "compile-time: no implicit vector -> ProposalTokens");

  const std::vector<int32_t> prompt = {2, 10979, 496};       // trusted prompt incl. BOS (id 2 is in the reject set)
  const std::vector<int32_t> clean_proposal = {1003};
  const std::vector<int32_t> bad_proposal = {1003, 63937};  // proposal containing a rejected id
  std::vector<int32_t> old_style = prompt;                   // what PchStack used to hand the gate
  old_style.insert(old_style.end(), clean_proposal.begin(), clean_proposal.end());

  PchGate g;
  check(g.Pass(ProposalTokens::from_proposal(clean_proposal)), "clean proposal passes");
  check(!g.Pass(ProposalTokens::from_proposal(bad_proposal)) && g.LastRejectToken() == 63937, "rejected proposal still rejected (63937)");
  check(!g.Pass(ProposalTokens::from_proposal(old_style)) && g.LastRejectToken() == 2,
        "pre-split behaviour reproduced: prompt+proposal is rejected on BOS (the leak the split removes)");

  std::vector<float> logits(4, 0.0f);
  logits[1] = 1.0f;
  PchCpuLeafPath leaf(0.7, 0.0);
  const uint32_t depth = uint32_t(prompt.size() + clean_proposal.size());
  const double v_split = leaf.Evaluate(ProposalTokens::from_proposal(clean_proposal), logits, depth, 0.0, nullptr, 0, false);
  const double v_old = leaf.Evaluate(ProposalTokens::from_proposal(old_style), logits, depth, 0.0, nullptr, 0, false);
  check(v_split > 0.0, "leaf value > 0 when only the proposal is gated (BOS in prompt is not seen)");
  check(v_old == 0.0, "old prompt-inclusive gating zeroed the same leaf (G = 0)");
  std::printf("  v_split=%.6f v_old=%.6f depth=%u\n", v_split, v_old, depth);
  std::printf("GATE-SPLIT: %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
