// L3. Fork-and-score look-ahead contracts, including the vision-only path.
// Official-chain step_back is refused by construction (type tag + runtime guard).
// Decision tuples are sealed on the scratch side chain only.
// Fold-by-stable-work-id is a local stand-in for DeterministicJoin (that header
// is not in the Nov 10 11:02 ET box snapshot; a later code-read is deferred).
#pragma once

#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlFlags.h"

#include <algorithm>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

struct ForkScoreConfig {
  uint32_t horizon = 2;
  uint32_t forks_per_decision = 4;
  uint32_t token_budget = 0;
};

inline ForkScoreConfig fork_score_from_flags(const RlFlags& f) {
  ForkScoreConfig c;
  c.horizon = f.fork_horizon;
  c.token_budget = f.fork_budget;
  return c;
}

enum class ChainKind : uint8_t { Official = 0, Scratch = 1 };

template <ChainKind K>
constexpr bool step_back_allowed_v = (K == ChainKind::Scratch);

static_assert(step_back_allowed_v<ChainKind::Scratch>, "scratch step_back must be allowed");
static_assert(!step_back_allowed_v<ChainKind::Official>,
              "official-chain step_back is forbidden under L3");

// Runtime twin of the compile-time guard. Passing Official always fails and
// leaves state unchanged.
inline bool may_step_back(ChainKind k, std::string& err) {
  if (k == ChainKind::Official) {
    err = "step_back with the official chain is refused by construction";
    return false;
  }
  err.clear();
  return true;
}

struct DecisionTuple {  // UML D2 contents
  std::array<uint8_t, 32> state_hash{};  // obs tokens + belief digest
  std::vector<int32_t> options;
  int32_t decision = 0;
  std::vector<int32_t> distribution_q20;
  int32_t confidence_q20 = 0;
  std::array<uint8_t, 32> weights_hash{};
  int32_t blank_control_decision = 0;  // vision path only
  uint8_t verifier_verdict = 0xFF;     // filled by L6 when on, else 0xFF
};

inline std::vector<uint8_t> decision_tuple_canonical(const DecisionTuple& t) {
  std::vector<uint8_t> b;
  b.insert(b.end(), t.state_hash.begin(), t.state_hash.end());
  append_u32_le(b, uint32_t(t.options.size()));
  for (int32_t x : t.options) append_i32_le(b, x);
  append_i32_le(b, t.decision);
  append_u32_le(b, uint32_t(t.distribution_q20.size()));
  for (int32_t x : t.distribution_q20) append_i32_le(b, x);
  append_i32_le(b, t.confidence_q20);
  b.insert(b.end(), t.weights_hash.begin(), t.weights_hash.end());
  append_i32_le(b, t.blank_control_decision);
  b.push_back(t.verifier_verdict);
  return b;
}

inline std::array<uint8_t, 32> seal_decision_tuple(const DecisionTuple& t) {
  const auto c = decision_tuple_canonical(t);
  return sha256_bytes(c.data(), c.size());
}

inline bool tuple_rehashes_to(const DecisionTuple& t, const std::array<uint8_t, 32>& sealed) {
  return seal_decision_tuple(t) == sealed;
}

// Deterministic fold of parallel fork results by stable work id.
// Arrival order is ignored (same invariant as pchie_join_order_test).
struct ForkWorkItem {
  uint64_t work_id = 0;  // intended encoding later: (decision, fork, horizon depth)
  int32_t score_q20 = 0;
};

inline std::vector<ForkWorkItem> fold_by_work_id(std::vector<ForkWorkItem> items) {
  std::sort(items.begin(), items.end(),
            [](const ForkWorkItem& a, const ForkWorkItem& b) { return a.work_id < b.work_id; });
  return items;
}

inline uint64_t fold_digest(const std::vector<ForkWorkItem>& items) {
  const auto folded = fold_by_work_id(items);
  std::vector<uint8_t> b;
  for (const auto& it : folded) {
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(it.work_id >> (8 * i)));
    append_i32_le(b, it.score_q20);
  }
  const auto h = sha256_bytes(b.data(), b.size());
  uint64_t u = 0;
  for (int i = 0; i < 8; ++i) u |= uint64_t(h[i]) << (8 * i);
  return u;
}

}  // namespace rl
}  // namespace pchie
