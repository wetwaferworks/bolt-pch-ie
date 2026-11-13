#include "pchie/pchie_tree.h"

namespace pchie {

void VariationTree::init_root(uint64_t initial_tick, uint32_t bos_token, uint64_t seed) {
    nodes.clear();
    forks.clear();
    TokenRecord root{};
    root.handshake_tick = initial_tick;
    root.token_id       = bos_token;
    root.step_seed      = seed;
    root.model_state_hash = compute_record_hash(root, 0ULL);
    nodes.push_back(root);
    forks.push_back({MAINLINE_FORK, 0, initial_tick, 0});
    cursor_node_id = 0;
    active_fork_id = MAINLINE_FORK;
}

uint64_t VariationTree::compute_record_hash(const TokenRecord& rec, uint64_t logit_digest) noexcept {
    uint64_t h = FNV_OFFSET;
    h = hash_mix64(h, rec.handshake_tick);
    h = hash_mix64(h, rec.fork_id);
    h = hash_mix64(h, rec.parent_node_id);
    h = hash_mix64(h, rec.token_id);
    h = hash_mix64(h, rec.seq_pos);
    h = hash_mix64(h, rec.step_seed);
    h = hash_mix64(h, logit_digest);
    for (uint32_t blk : rec.kv_block_table) h = hash_mix64(h, blk);
    return h;
}

uint32_t VariationTree::step_forward() noexcept {
    if (cursor_node_id == INVALID_ID) return INVALID_ID;
    for (uint32_t c : nodes[cursor_node_id].child_node_ids) {
        if (nodes[c].fork_id == active_fork_id) { cursor_node_id = c; return c; }
    }
    return INVALID_ID;
}

uint32_t VariationTree::step_back() noexcept {
    if (cursor_node_id == INVALID_ID) return INVALID_ID;
    uint32_t parent = nodes[cursor_node_id].parent_node_id;
    if (parent != INVALID_ID) cursor_node_id = parent; // active_fork_id intentionally unchanged
    return cursor_node_id;
}

uint32_t VariationTree::fork_at(uint64_t target_tick) {
    uint32_t anchor = find_node_in_active_lineage(target_tick);
    if (anchor == INVALID_ID) return INVALID_ID;
    uint32_t id = static_cast<uint32_t>(forks.size());
    forks.push_back({id, anchor, target_tick, anchor});
    cursor_node_id = anchor;
    active_fork_id = id;
    return id;
}

uint32_t VariationTree::return_to_mainline() noexcept {
    active_fork_id = MAINLINE_FORK;
    cursor_node_id = forks[MAINLINE_FORK].head_node_id;
    return cursor_node_id;
}

bool VariationTree::goto_position(uint64_t tick, uint32_t fork_id) noexcept {
    if (fork_id >= forks.size()) return false;
    for (uint32_t cur = forks[fork_id].head_node_id; cur != INVALID_ID; cur = nodes[cur].parent_node_id) {
        if (nodes[cur].handshake_tick == tick) {
            cursor_node_id = cur;
            active_fork_id = fork_id;
            return true;
        }
    }
    return false;
}

uint32_t VariationTree::append_generated_token(uint64_t tick, uint32_t token_id, uint64_t next_seed,
                                               uint64_t logit_digest, const std::vector<uint32_t>& kv) {
    assert(cursor_node_id != INVALID_ID);
    uint32_t new_id = static_cast<uint32_t>(nodes.size());
    TokenRecord rec{};
    rec.handshake_tick = tick;
    rec.node_id        = new_id;
    rec.fork_id        = active_fork_id;
    rec.parent_node_id = nodes[cursor_node_id].node_id;
    rec.parent_tick    = nodes[cursor_node_id].handshake_tick;
    rec.token_id       = token_id;
    rec.seq_pos        = nodes[cursor_node_id].seq_pos + 1;
    rec.step_seed      = next_seed;
    rec.kv_block_table = kv;
    rec.model_state_hash = compute_record_hash(rec, logit_digest);
    nodes.push_back(std::move(rec));
    nodes[cursor_node_id].child_node_ids.push_back(new_id);
    cursor_node_id = new_id;
    forks[active_fork_id].head_node_id = new_id;
    return new_id;
}

uint32_t VariationTree::append_child(uint32_t parent, uint32_t slot, uint64_t tick, uint32_t token_id,
                                     uint64_t seed, uint64_t logit_digest) {
    const uint32_t new_id = static_cast<uint32_t>(nodes.size());
    TokenRecord rec{};
    rec.handshake_tick = tick;
    rec.node_id        = new_id;
    rec.fork_id        = slot;
    rec.parent_node_id = parent;
    rec.parent_tick    = nodes[parent].handshake_tick;
    rec.token_id       = token_id;
    rec.seq_pos        = nodes[parent].seq_pos + 1;
    rec.step_seed      = seed;                 // kv_block_table stays empty: no physical pages in the hash
    rec.model_state_hash = compute_record_hash(rec, logit_digest);
    nodes.push_back(std::move(rec));
    nodes[parent].child_node_ids.push_back(new_id);
    return new_id;
}

uint64_t VariationTree::tree_digest() const noexcept {
    uint64_t h = FNV_OFFSET;
    for (const auto& n : nodes) h = hash_mix64(h, n.model_state_hash);
    return h;
}

uint32_t VariationTree::find_node_in_active_lineage(uint64_t t) const noexcept {
    for (uint32_t cur = cursor_node_id; cur != INVALID_ID; cur = nodes[cur].parent_node_id)
        if (nodes[cur].handshake_tick == t) return cur;
    return INVALID_ID;
}

} // namespace pchie
