// MuZero-style look-ahead (Schrittwieser et al., 2020) with a frozen dynamics model.
// A model learned offline from Little Nemo's own transitions predicts, for a state and an action,
// the next state and the reward. Once frozen it never changes, and its SHA-256 identifies it.
// The look-ahead searches every action sequence up to a small depth inside the model (no world
// access), scores the leaves with the frozen value table, and returns the best first action. Untried (state, action) pairs get an optimistic value,
// so the look-ahead also pulls him toward edges he has not tried.
// This tabular model is the reference; a trained network can replace it behind the same interface.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace pchie {
namespace research {

struct Transition {
  uint64_t state = 0;
  uint8_t action = 0;
  uint64_t next = 0;
  int32_t reward_q20 = 0;
};

struct DynamicsEntry {
  uint64_t next = 0;        // most frequent next state (ties: lower state)
  uint32_t count = 0;       // times this (state, action) was seen
  int32_t reward_q20 = 0;   // mean reward, rounded toward zero
};

struct LookaheadConfig {
  uint32_t depth = 3;
  uint32_t actions = 6;               // action ids 0..actions-1
  int32_t untried_value_q20 = 1 << 18;
  int32_t step_cost_q20 = 1 << 8;
  int32_t discount_num = 63;          // discount = num / den
  int32_t discount_den = 64;
};

struct LookaheadChoice {
  int action = -1;
  int64_t value_q20 = 0;
  uint64_t nodes = 0;
};

class FrozenDynamics;

class DynamicsTrainer {
 public:
  void add(const Transition& t);
  /// Freezes the model and computes its value table offline (integer value iteration over the frozen
  /// model, untried pairs worth 0), so the look-ahead can stand on it at its leaves like MuZero's
  /// value head. Both are covered by the model hash.
  FrozenDynamics freeze(const LookaheadConfig& value_cfg = LookaheadConfig{}) const;
  size_t size() const { return seen_.size(); }

 private:
  struct Tally {
    std::map<uint64_t, uint32_t> next;
    int64_t reward_sum = 0;
    uint32_t count = 0;
  };
  std::map<std::pair<uint64_t, uint8_t>, Tally> seen_;
};

class FrozenDynamics {
 public:
  bool predict(uint64_t state, uint8_t action, DynamicsEntry& out) const;
  LookaheadChoice best_action(uint64_t state, const LookaheadConfig& cfg) const;
  size_t size() const { return table_.size(); }
  uint32_t value_sweeps() const { return value_sweeps_; }
  const std::string& sha256() const { return sha_; }
  std::string canonical() const;

 private:
  friend class DynamicsTrainer;
  int64_t value(uint64_t state, uint32_t depth, const LookaheadConfig& cfg, uint64_t& nodes) const;
  int64_t leaf_value(uint64_t state) const;
  int64_t step_value(uint64_t state, uint32_t a, uint32_t depth, const LookaheadConfig& cfg, uint64_t& nodes) const;
  std::map<std::pair<uint64_t, uint8_t>, DynamicsEntry> table_;
  std::map<uint64_t, int64_t> values_;
  uint32_t value_sweeps_ = 0;
  std::string sha_;
};

}  // namespace research
}  // namespace pchie
