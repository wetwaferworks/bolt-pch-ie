#include "pchie/pchie_rollout.h"
#include "pchie/pchie_tree.h"
#include "pchie/pchie_kv_cow.h"
#include <thread>

namespace pchie {

uint64_t splitmix64(uint64_t& s) noexcept {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void gen_step(VariationTree& t, PagedKVCache& kv, uint64_t tick, uint64_t salt) {
    const TokenRecord& p = t.nodes[t.cursor_node_id];
    auto table = kv.clone_block_table(p.kv_block_table);
    uint8_t* slot = kv.prepare_write_slot(table, p.seq_pos + 1);
    uint64_t rng = p.step_seed ^ salt;
    const uint32_t tok = static_cast<uint32_t>(splitmix64(rng) % 32000u);
    const uint64_t digest = splitmix64(rng);
    for (size_t i = 0; i < kv.bytes_per_token(); ++i) slot[i] = static_cast<uint8_t>(digest >> ((i & 7) * 8));
    t.append_generated_token(tick, tok, rng, digest, table);
}

RolloutResult run_single_rollout(uint64_t seed, uint32_t mainline_len, uint32_t num_forks, uint32_t fork_len) {
    constexpr uint64_t TICK0 = 1000, K = 8; // generation mode: K ticks per token
    VariationTree t;
    PagedKVCache kv(2, 2, 16, 4096);
    t.init_root(TICK0, 1, seed);
    kv.prepare_write_slot(t.nodes[0].kv_block_table, 0);
    for (uint32_t s = 1; s <= mainline_len; ++s) gen_step(t, kv, TICK0 + s * K, 0);

    const uint64_t anchor_tick = TICK0 + (mainline_len / 2) * K;
    for (uint32_t f = 0; f < num_forks; ++f) {
        t.return_to_mainline();
        const uint32_t fid = t.fork_at(anchor_tick);
        for (uint32_t s = 1; s <= fork_len; ++s)
            gen_step(t, kv, anchor_tick + s * K, static_cast<uint64_t>(fid) << 32);
        // exercise stepping: back 3, forward 3 must land on the same head
        const uint32_t head = t.cursor_node_id;
        for (int i = 0; i < 3; ++i) t.step_back();
        for (int i = 0; i < 3; ++i) t.step_forward();
        if (t.cursor_node_id != head) return {0, 0, 0, 0, false};
    }
    t.return_to_mainline();

    RolloutResult r{t.tree_digest(), t.nodes[t.cursor_node_id].model_state_hash,
                    static_cast<uint32_t>(t.nodes.size()), static_cast<uint32_t>(t.forks.size()), false};
    for (size_t i = t.nodes.size(); i-- > 0;) kv.release_block_table(t.nodes[i].kv_block_table);
    r.kv_fully_released = (kv.free_blocks() == 4096);
    return r;
}

std::vector<RolloutResult> run_rollouts(uint32_t n, uint32_t threads, uint64_t base_seed) {
    std::vector<RolloutResult> out(n);
    if (threads == 0) threads = 1;
    auto work = [&](uint32_t w) {
        const uint32_t begin = static_cast<uint32_t>(uint64_t(n) * w / threads);
        const uint32_t end   = static_cast<uint32_t>(uint64_t(n) * (w + 1) / threads);
        for (uint32_t i = begin; i < end; ++i) {
            uint64_t s = base_seed ^ (uint64_t(i) * 0x9e3779b97f4a7c15ULL);
            out[i] = run_single_rollout(splitmix64(s), 64, 3, 24);
        }
    };
    std::vector<std::thread> pool;
    for (uint32_t w = 1; w < threads; ++w) pool.emplace_back(work, w);
    work(0);
    for (auto& th : pool) th.join();
    return out;
}

uint64_t combine_digests(const std::vector<RolloutResult>& r) noexcept {
    uint64_t h = FNV_OFFSET;
    for (const auto& x : r) { h = hash_mix64(h, x.tree_digest); h = hash_mix64(h, x.mainline_head_hash); }
    return h;
}

} // namespace pchie
