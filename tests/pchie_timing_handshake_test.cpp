// Bolt PCHIE gate: Timing Handshake (K ticks/token), flat world, generation vs scrub.
#include "pchie/pchie_host.h"
#include "pchie/pchie_rollout.h"
#include <cstdio>
#include <thread>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static uint64_t session_digest(uint64_t seed, bool checks) {
    using namespace pchie;
    int calls = 0;
    HostSession s(1000, 8, seed, [&](const TokenRecord& p, uint32_t f) { ++calls; return stub_token_source(p, f); });
    for (int i = 0; i < 40; ++i) s.generate();
    const uint64_t anchor = 1000 + 20 * 8;
    for (int f = 0; f < 3; ++f) {
        s.tree.return_to_mainline();
        s.tree.fork_at(anchor);
        for (int i = 0; i < 12; ++i) s.generate();
    }
    const int gen_calls = calls;
    const uint64_t d_gen = s.world_digest();
    // Scrub mode: walk fork 2 back to its anchor and forward again; no inference calls.
    s.mode = HostMode::Scrub;
    s.tree.goto_position(anchor + 12 * 8, 2);
    for (int i = 0; i < 12; ++i) s.scrub_back();
    if (checks) CHECK(s.clock.now() == anchor);
    for (int i = 0; i < 12; ++i) s.scrub_forward();
    if (checks) {
        CHECK(calls == gen_calls);
        CHECK(s.clock.now() == anchor + 12 * 8);
        // every record's tick = parent tick + K
        for (size_t i = 1; i < s.tree.nodes.size(); ++i)
            CHECK(s.tree.nodes[i].handshake_tick == s.tree.nodes[s.tree.nodes[i].parent_node_id].handshake_tick + 8);
        // flat world: no procedural terrain
        CHECK(s.world.block_at(3, 3, 3) == 2 && s.world.block_at(40, 3, 9) == 2 && s.world.block_at(5, 4, 5) == 0);
        const RobotPose mid = pose_at_tick(s.tree, 2, anchor + 4, s.world);
        CHECK(mid.y == s.world.floor_y * 1024);
    }
    s.tree.goto_position(anchor + 12 * 8, 2);
    return hash_mix64(d_gen, s.world_digest());
}

int main() {
    uint64_t ref = 0;
    session_digest(7, true);
    for (int run = 1; run <= 3; ++run)
        for (uint32_t th : {1u, 2u, 16u}) {
            constexpr uint32_t N = 32;
            std::vector<uint64_t> d(N);
            auto work = [&](uint32_t w) {
                for (uint32_t i = N * w / th; i < N * (w + 1) / th; ++i) d[i] = session_digest(0xB0170000ULL + i, false);
            };
            std::vector<std::thread> pool;
            for (uint32_t w = 1; w < th; ++w) pool.emplace_back(work, w);
            work(0);
            for (auto& t : pool) t.join();
            uint64_t h = pchie::FNV_OFFSET;
            for (auto x : d) h = pchie::hash_mix64(h, x);
            std::printf("[PCHIE GATE] step2 sessions N=%u run=%d threads=%2u digest=0x%016llX\n", N, run, th, (unsigned long long)h);
            if (!ref) ref = h;
            CHECK(h == ref);
        }
    std::printf(g_fail ? "STEP2 GATE: FAIL (%d)\n" : "STEP2 GATE: PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
