// Contract fixture for the L1-L6 hook names. PchStack does not store this table.
// PchStack keeps RlFlags and fail-closes when they are not official-clean.
// An engaged layer runs through HeadlessMaze::set_rl / StageBDoorMaze::set_rl.
//
// Intended setters (from the 2026-11-11 queue plan; signatures to be checked
// against the live headers before any PchStack edit):
//   void set_leaf_value_source(ILeafValueSource*);   // nullptr = oracle (default)
//   void set_prior_source(PriorSource);              // default: current mix
//   void set_plan_memory(IPlanMemory*, ReusePolicy); // nullptr = off
//   void set_fork_score(const ForkScoreConfig*);     // nullptr = off
//   void set_options(const OptionSpec* specs, size_t n);  // n==0 = off
//   void add_proposal_source(IProposalSource*);
//   void set_verifier(IPlanVerifier*);               // both off by default
//
// Official chain: MctsJoinedStepBack::checkpoint / step_back must not be
// given the official IHashChain under L3. Use side-chain-only scoring (see ForkScore.h).
#pragma once

#include "pchie/rl/RlFlags.h"

#include <cstddef>

namespace pchie {
namespace rl {

struct PchStackRlHooks {
  void* leaf_value_source = nullptr;
  void* plan_memory = nullptr;
  const void* fork_score = nullptr;
  const void* option_specs = nullptr;
  size_t option_count = 0;
  void* proposal_source = nullptr;
  void* verifier = nullptr;

  bool any_engaged() const {
    return leaf_value_source || plan_memory || fork_score || option_specs || option_count ||
           proposal_source || verifier;
  }
};

// Flags off => hooks stay null. This is the compile-in, flags-off identity
// for the hook table itself. The 98/98 official-token regression is deferred
// until a concurrent visnav run is idle.
inline bool flags_off_hooks_are_idle(const RlFlags& f, const PchStackRlHooks& h) {
  return !f.any_on() && official_runner_clean(f) && !h.any_engaged();
}

}  // namespace rl
}  // namespace pchie
