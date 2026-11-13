// Bolt PCHIE - tick-stamped variation tree and stepping API (clean-room).
#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

namespace pchie {

constexpr uint32_t INVALID_ID    = 0xFFFFFFFFu;
constexpr uint32_t MAINLINE_FORK = 0u;
constexpr uint32_t KV_BLOCK_SIZE = 16u; // tokens per physical KV page

// 64-bit FNV-1a, byte-wise little-endian fold: identical on every platform.
inline uint64_t hash_mix64(uint64_t h, uint64_t val) noexcept {
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    for (int i = 0; i < 8; ++i) {
        h ^= (val >> (i * 8)) & 0xFFULL;
        h *= FNV_PRIME;
    }
    return h;
}
constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;

struct TokenRecord {
    uint64_t handshake_tick   = 0;
    uint32_t node_id          = 0;
    uint32_t fork_id          = 0;
    uint32_t parent_node_id   = INVALID_ID; // unique parent (disambiguates siblings at same tick)
    uint64_t parent_tick      = 0;
    uint32_t token_id         = 0;
    uint32_t seq_pos          = 0;
    uint64_t step_seed        = 0;          // self-contained sampler state at this step
    uint64_t model_state_hash = 0;
    std::vector<uint32_t> kv_block_table;   // logical->physical KV pages at this token
    std::vector<uint32_t> child_node_ids;
};

struct ForkDescriptor {
    uint32_t fork_id;
    uint32_t origin_node_id;
    uint64_t origin_tick;
    uint32_t head_node_id;
};

class VariationTree {
public:
    std::vector<TokenRecord>    nodes;
    std::vector<ForkDescriptor> forks;
    uint32_t cursor_node_id = INVALID_ID;
    uint32_t active_fork_id = MAINLINE_FORK;

    void init_root(uint64_t initial_tick, uint32_t bos_token, uint64_t seed);
    static uint64_t compute_record_hash(const TokenRecord& rec, uint64_t logit_digest) noexcept;

    // Replay the recorded child on the active fork; INVALID_ID at the branch tip.
    uint32_t step_forward() noexcept;
    // Move to the parent node. The active fork is KEPT, so step_forward()
    // returns into the same variation (we do not jump to the parent's fork).
    uint32_t step_back() noexcept;
    uint32_t fork_at(uint64_t target_tick);
    uint32_t return_to_mainline() noexcept;
    bool     goto_position(uint64_t tick, uint32_t fork_id) noexcept;
    uint32_t append_generated_token(uint64_t handshake_tick, uint32_t token_id,
                                    uint64_t next_seed, uint64_t logit_digest,
                                    const std::vector<uint32_t>& kv_table_snapshot);
    // append a child under an arbitrary node (MCTS expansion). Fork bookkeeping is
    // left to the caller; the record's fork_id is the node's tree slot, not creation order.
    uint32_t append_child(uint32_t parent, uint32_t slot, uint64_t tick, uint32_t token_id,
                          uint64_t seed, uint64_t logit_digest);
    // Order-fixed digest over every node hash (node-id order).
    uint64_t tree_digest() const noexcept;

private:
    uint32_t find_node_in_active_lineage(uint64_t target_tick) const noexcept;
};

} // namespace pchie
