// Bolt PCHIE gate: tree/stepping, CoW KV, rounding, Q4 dot, thread invariance.
#include "pchie/pchie_tree.h"
#include "pchie/pchie_kv_cow.h"
#include "pchie/pchie_q4_det.h"
#include "pchie/pchie_rollout.h"
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <cstring>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static void test_archie_scenario() {
    using namespace pchie;
    VariationTree tree;
    PagedKVCache kv(32, 8, 128, 256);
    uint64_t seed = 0xC0FFEE123456789ULL;
    tree.init_root(1000, 1, seed);
    kv.prepare_write_slot(tree.nodes[0].kv_block_table, 0);
    for (int s = 1; s <= 5; ++s) {
        const auto& p = tree.nodes[tree.cursor_node_id];
        auto tb = kv.clone_block_table(p.kv_block_table);
        kv.prepare_write_slot(tb, p.seq_pos + 1);
        uint64_t rng = p.step_seed;
        uint32_t tok = uint32_t(splitmix64(rng) % 32000);
        uint64_t d = splitmix64(rng);
        tree.append_generated_token(1000 + s * 8, tok, rng, d, tb);
    }
    const uint32_t main_head = tree.cursor_node_id;
    uint32_t fa = tree.fork_at(1024);
    CHECK(fa == 1);
    for (int s = 1; s <= 2; ++s) {
        const auto& p = tree.nodes[tree.cursor_node_id];
        auto tb = kv.clone_block_table(p.kv_block_table);
        kv.prepare_write_slot(tb, p.seq_pos + 1);
        uint64_t rng = p.step_seed ^ (uint64_t(fa) << 32);
        uint32_t tok = uint32_t(splitmix64(rng) % 32000);
        uint64_t d = splitmix64(rng);
        tree.append_generated_token(1024 + s * 8, tok, rng, d, tb);
    }
    // step_back keeps the active fork: back 2 lands on the anchor (a mainline node) but still fork A
    const uint32_t fork_head = tree.cursor_node_id;
    tree.step_back(); tree.step_back();
    CHECK(tree.active_fork_id == fa);
    CHECK(tree.nodes[tree.cursor_node_id].handshake_tick == 1024);
    tree.step_forward(); tree.step_forward();
    CHECK(tree.cursor_node_id == fork_head);
    // sibling fork at same tick is distinguishable by parent_node_id
    tree.return_to_mainline();
    CHECK(tree.cursor_node_id == main_head);
    std::printf("[PCHIE GATE] archie-scenario nodes=%zu forks=%zu mainline_head=0x%016llX tree=0x%016llX\n",
                tree.nodes.size(), tree.forks.size(),
                (unsigned long long)tree.nodes[tree.cursor_node_id].model_state_hash,
                (unsigned long long)tree.tree_digest());
}

static void test_kv_release() {
    using namespace pchie;
    PagedKVCache kv(1, 1, 8, 8);
    std::vector<uint32_t> a;
    for (uint32_t p = 0; p < 20; ++p) kv.prepare_write_slot(a, p); // pages 0,1
    CHECK(a.size() == 2 && a[0] == 0 && a[1] == 1);
    auto b = kv.clone_block_table(a);
    kv.prepare_write_slot(b, 20); // tail shared -> CoW to page 2
    CHECK(b[1] == 2 && kv.ref_count(0) == 2 && kv.ref_count(1) == 1);
    kv.release_block_table(a); // page 1 freed, page 0 still held by b
    CHECK(kv.free_blocks() == 6 && kv.ref_count(0) == 1);
    std::vector<uint32_t> c;
    kv.prepare_write_slot(c, 0);
    CHECK(c[0] == 1); // lowest free id reused
    kv.release_block_table(b);
    kv.release_block_table(c);
    CHECK(kv.free_blocks() == 8);
    std::vector<uint32_t> d;
    kv.prepare_write_slot(d, 0);
    CHECK(d[0] == 0);
}

static void test_rounding() {
    using pchie::round_half_even_f32;
    const float in[]  = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 2.4999998f, 2.5000002f, -126.5f, 126.5f, 0.0f, -0.0f};
    const int   out[] = {0, 2, 2, 0, -2, -2, 2, 3, -126, 126, 0, 0};
    for (size_t i = 0; i < sizeof(out) / sizeof(out[0]); ++i) CHECK(round_half_even_f32(in[i]) == out[i]);
}

static uint64_t matvec_hash(uint32_t threads) {
    using namespace pchie;
    constexpr int ROWS = 256, K = 1024, NB = K / Q_BLOCK_SIZE;
    std::vector<BlockQ4_0> w(ROWS * NB);
    uint64_t s = 42;
    for (auto& b : w) {
        b.d_fp16 = uint16_t(0x2000 + (splitmix64(s) & 0x0FFF));
        for (auto& q : b.qs) q = uint8_t(splitmix64(s));
    }
    std::vector<float> x(K);
    for (auto& v : x) v = float(int64_t(splitmix64(s) % 20001) - 10000) / 4096.0f;
    std::vector<BlockQ8_0> xq(NB);
    quantize_row_q8_0_deterministic(x.data(), xq.data(), K);
    std::vector<float> y(ROWS);
    auto work = [&](uint32_t t) {
        for (int r = int(ROWS * t / threads); r < int(ROWS * (t + 1) / threads); ++r)
            y[r] = dot_q4_0_q8_0_deterministic(&w[size_t(r) * NB], xq.data(), NB);
    };
    std::vector<std::thread> pool;
    for (uint32_t t = 1; t < threads; ++t) pool.emplace_back(work, t);
    work(0);
    for (auto& th : pool) th.join();
    uint64_t h = FNV_OFFSET;
    for (float v : y) { uint32_t u; std::memcpy(&u, &v, 4); h = hash_mix64(h, u); }
    return h;
}

int main() {
    test_archie_scenario();
    test_kv_release();
    test_rounding();

    const uint64_t m1 = matvec_hash(1), m2 = matvec_hash(2), m16 = matvec_hash(16);
    std::printf("[PCHIE GATE] q4xq8 matvec 1t=0x%016llX 2t=0x%016llX 16t=0x%016llX\n",
                (unsigned long long)m1, (unsigned long long)m2, (unsigned long long)m16);
    CHECK(m1 == m2 && m2 == m16);

    constexpr uint32_t N = 64;
    constexpr uint64_t SEED = 0xB017C41E5EEDULL;
    uint64_t ref = 0;
    for (int run = 1; run <= 3; ++run) {
        for (uint32_t th : {1u, 2u, 16u}) {
            auto r = pchie::run_rollouts(N, th, SEED);
            bool ok = true;
            for (auto& x : r) ok = ok && x.kv_fully_released && x.nodes == 1 + 64 + 3 * 24 && x.forks == 4;
            CHECK(ok);
            const uint64_t h = pchie::combine_digests(r);
            std::printf("[PCHIE GATE] rollouts N=%u run=%d threads=%2u digest=0x%016llX\n",
                        N, run, th, (unsigned long long)h);
            if (ref == 0) ref = h;
            CHECK(h == ref);
        }
    }
    std::printf(g_fail ? "STEP1 GATE: FAIL (%d)\n" : "STEP1 GATE: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
