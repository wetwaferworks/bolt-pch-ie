// Bolt PCHIE Mirror-Sim gate - Mirror-Sim Action Gate.
// Concrete final. Simulates a committed action on a deterministic grid copy
// (no CoW) at commit time only, not during rollouts or over root children.
// Unsafe actions are refused and never written to the real world.
#pragma once

#include "pchie/IUnsafeRule.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/PitTrapRule.h"

#include <string>
#include <vector>

namespace pchie {

struct MirrorSimDecision {
  bool sim_applied = false;
  bool allowed = false;
  const char* rule_fired = "";
  std::vector<uint8_t> pre_sha;
  uint64_t pre_fnv = 0;
  std::vector<uint8_t> pred_sha;
  uint64_t pred_fnv = 0;
};

class MirrorSimGate final {
 public:
  void clear_rules() { rules_.clear(); }

  // the maintainer judges new rules at first. Default registration is pit_trap only.
  void add_rule(const IUnsafeRule* rule) {
    if (rule) rules_.push_back(rule);
  }

  void use_default_pit_trap() {
    rules_.clear();
    rules_.push_back(&pit_trap_);
  }

  // Copy `pre`, apply `a` on the copy, run rules in registration order.
  // Does not mutate `pre`. Does not append to any official chain.
  MirrorSimDecision simulate(const MazeGrid& pre, MazeAction a) const {
    MirrorSimDecision d;
    MirrorSimSideLog::hash_world(pre, d.pre_sha, d.pre_fnv);
    MazeGrid copy = pre;
    if (!copy.apply(a)) {
      d.sim_applied = false;
      d.allowed = false;
      d.rule_fired = "";
      return d;
    }
    d.sim_applied = true;
    MirrorSimSideLog::hash_world(copy, d.pred_sha, d.pred_fnv);
    for (const IUnsafeRule* r : rules_) {
      if (r && r->is_unsafe(pre, a, copy)) {
        d.allowed = false;
        d.rule_fired = r->name();
        return d;
      }
    }
    d.allowed = true;
    d.rule_fired = "";
    return d;
  }

 private:
  PitTrapRule pit_trap_;
  std::vector<const IUnsafeRule*> rules_;
};

}  // namespace pchie
