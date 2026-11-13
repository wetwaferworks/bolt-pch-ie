// Bolt PCHIE reasoning-layer flag parser. Flags default off.
// Official runners must never set these flags.
#include "pchie/rl/PchStackRlHooks.h"
#include "pchie/rl/RlFlags.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace pchie::rl;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[RL-FLAGS] %-72s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

int main() {
  const RlFlags d{};
  check(!d.any_on(), "default flags: any_on() is false");
  check(official_runner_clean(d), "default flags pass official_runner_clean");
  check(official_runner_clean(official_defaults()), "official_defaults() is clean");

  const std::string id0 = d.identity_line();
  check(id0 == official_defaults().identity_line(), "identity_line is stable for defaults");
  check(id0.find("puct_model=0") != std::string::npos, "identity_line records puct_model=0");
  check(id0.find("verifier_mode=record") != std::string::npos, "default verifier mode is record");
  check(id0.find("qbp_min_truth=sim") != std::string::npos, "proposed min-truth default is sim (still an open question)");

  auto empty = parse_rl_flags(std::vector<std::string>{});
  check(empty.ok && official_runner_clean(empty.flags), "parse of no args is official-clean");

  auto ignored = parse_rl_flags(std::vector<std::string>{"--budget=8", "maze.bin"});
  check(ignored.ok && official_runner_clean(ignored.flags), "non --rl-* args are ignored");

  auto m = parse_rl_flags(std::vector<std::string>{"--rl-puct-model"});
  check(m.ok && m.flags.puct_model, "--rl-puct-model sets puct_model");
  check(m.flags.puct_prior == PriorSrc::Model && m.flags.puct_value == ValueSrc::Model,
        "--rl-puct-model sets prior and value to model");
  check(m.flags.any_on() && !official_runner_clean(m.flags), "flags on fails official_runner_clean");

  auto v = parse_rl_flags(std::vector<std::string>{"--rl-puct-value=null"});
  check(v.ok && v.flags.puct_value == ValueSrc::Null && !v.flags.puct_model,
        "--rl-puct-value=null is a standalone null control");

  auto vis = parse_rl_flags(std::vector<std::string>{"--rl-vision-only"});
  check(vis.ok && vis.flags.vision_only && vis.flags.belief, "--rl-vision-only implies --rl-belief");

  auto hz = parse_rl_flags(std::vector<std::string>{"--rl-fork-score", "--rl-fork-horizon=3"});
  check(hz.ok && hz.flags.fork_score && hz.flags.fork_horizon == 3, "--rl-fork-horizon=3");

  auto bad = parse_rl_flags(std::vector<std::string>{"--rl-not-a-real-flag"});
  check(!bad.ok && !bad.err.empty(), "unknown --rl-* flag fails closed");

  auto badh = parse_rl_flags(std::vector<std::string>{"--rl-fork-horizon=0"});
  check(!badh.ok, "horizon 0 fails closed");

  auto all = parse_rl_flags(std::vector<std::string>{
      "--rl-qbp", "--rl-qbp-write", "--rl-qbp-min-truth=human", "--rl-fork-score",
      "--rl-belief", "--rl-options", "--rl-options-set=corridor,turn,backout,probe",
      "--rl-symbolic-peer", "--rl-symbolic-weight=262144", "--rl-verifier",
      "--rl-verifier-mode=enforce"});
  check(all.ok, "all documented flags parse");
  check(all.flags.qbp && all.flags.qbp_write && all.flags.qbp_min_truth == MinTruth::Human, "L2 flags");
  check(all.flags.options_set == "corridor,turn,backout,probe", "L5 options-set");
  check(all.flags.symbolic_weight_q20 == 262144 && all.flags.verifier_mode == VerifierMode::Enforce,
        "L6 weight and enforce mode");

  PchStackRlHooks hooks;
  check(flags_off_hooks_are_idle(d, hooks), "flags off and hooks idle");
  int hook_sentinel = 0;
  hooks.leaf_value_source = &hook_sentinel;
  check(!flags_off_hooks_are_idle(d, hooks), "an engaged hook is not idle");

  if (g_fail) {
    std::printf("[RL-FLAGS] %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("[RL-FLAGS] ALL PASS\n");
  return 0;
}
