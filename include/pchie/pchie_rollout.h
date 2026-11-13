// Bolt PCHIE - deterministic stub: seeded fake token generator and
// N independent fork rollouts, statically partitioned across worker threads.
#pragma once
#include <cstdint>
#include <vector>

namespace pchie {

uint64_t splitmix64(uint64_t& state) noexcept;

struct RolloutResult {
    uint64_t tree_digest;
    uint64_t mainline_head_hash;
    uint32_t nodes;
    uint32_t forks;
    bool     kv_fully_released;
};

// One rollout: mainline of `mainline_len` tokens, `num_forks` sibling forks
// anchored at the same tick, each `fork_len` tokens; every page released at end.
RolloutResult run_single_rollout(uint64_t seed, uint32_t mainline_len, uint32_t num_forks, uint32_t fork_len);

// Rollout i -> worker floor(i * threads / n); results combined in rollout index order.
std::vector<RolloutResult> run_rollouts(uint32_t n, uint32_t threads, uint64_t base_seed);
uint64_t combine_digests(const std::vector<RolloutResult>& r) noexcept;

} // namespace pchie
