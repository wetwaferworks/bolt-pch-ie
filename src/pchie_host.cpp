#include "pchie/pchie_host.h"
#include "pchie/pchie_rollout.h"
#include <stdexcept>

namespace pchie {

uint8_t FlatWorld::block_at(int32_t x, int32_t y, int32_t z) const noexcept {
    if (x < 0 || z < 0 || x >= size_x || z >= size_z || y < 0 || y >= size_y) return 0;
    if (y < floor_y - 1) return 1;   // stone
    if (y == floor_y - 1) return 2;  // top layer
    return 0;                        // air
}

uint64_t FlatWorld::digest() const noexcept {
    uint64_t h = FNV_OFFSET;
    h = hash_mix64(h, uint64_t(size_x)); h = hash_mix64(h, uint64_t(size_y));
    h = hash_mix64(h, uint64_t(size_z)); h = hash_mix64(h, uint64_t(floor_y));
    for (int32_t y = 0; y < size_y; ++y) h = hash_mix64(h, block_at(0, y, 0));
    return h;
}

static RobotPose step_pose(RobotPose p, uint32_t token, const FlatWorld& w) noexcept {
    static const int32_t dx[4] = {1024, 0, -1024, 0}, dz[4] = {0, 1024, 0, -1024};
    const uint32_t dir = token & 3u;
    p.yaw = int32_t(dir) * 90;
    const int32_t nx = p.x + dx[dir], nz = p.z + dz[dir];
    const int32_t lim_x = (w.size_x - 1) * 1024, lim_z = (w.size_z - 1) * 1024;
    if (nx >= 0 && nx <= lim_x) p.x = nx;
    if (nz >= 0 && nz <= lim_z) p.z = nz;
    p.y = w.floor_y * 1024; // flat: always standing on the floor
    return p;
}

RobotPose pose_from_lineage(const VariationTree& t, uint32_t node_id, const FlatWorld& w) noexcept {
    std::vector<uint32_t> path;
    for (uint32_t c = node_id; c != INVALID_ID; c = t.nodes[c].parent_node_id) path.push_back(c);
    RobotPose p{(w.size_x / 2) * 1024, w.floor_y * 1024, (w.size_z / 2) * 1024, 0};
    for (size_t i = path.size(); i-- > 1;) p = step_pose(p, t.nodes[path[i - 1]].token_id, w);
    return p;
}

RobotPose pose_at_tick(const VariationTree& t, uint32_t fork_id, uint64_t tick, const FlatWorld& w) noexcept {
    uint32_t c = t.forks[fork_id].head_node_id;
    while (c != INVALID_ID && t.nodes[c].handshake_tick > tick) c = t.nodes[c].parent_node_id;
    if (c == INVALID_ID) c = 0;
    const RobotPose a = pose_from_lineage(t, c, w);
    // next node on this fork's lineage after c
    uint32_t n = t.forks[fork_id].head_node_id, nxt = INVALID_ID;
    while (n != INVALID_ID && n != c) { nxt = n; n = t.nodes[n].parent_node_id; }
    if (nxt == INVALID_ID || n != c) return a;
    const RobotPose b = pose_from_lineage(t, nxt, w);
    const int64_t span = int64_t(t.nodes[nxt].handshake_tick - t.nodes[c].handshake_tick);
    const int64_t f = int64_t(tick - t.nodes[c].handshake_tick);
    auto lerp = [&](int32_t u, int32_t v) { return int32_t(u + (int64_t(v - u) * f) / span); };
    return {lerp(a.x, b.x), lerp(a.y, b.y), lerp(a.z, b.z), f * 2 < span ? a.yaw : b.yaw};
}

HostSession::HostSession(uint64_t start_tick, uint32_t k, uint64_t seed, TokenSource src)
    : clock(start_tick, k), kv(2, 2, 16, 4096), src_(std::move(src)) {
    tree.init_root(start_tick, 1, seed);
    kv.prepare_write_slot(tree.nodes[0].kv_block_table, 0);
}

uint32_t HostSession::generate() {
    if (mode != HostMode::Generation) throw std::logic_error("generate() in scrub mode");
    const TokenRecord& p = tree.nodes[tree.cursor_node_id];
    if (tree.step_forward() != INVALID_ID) throw std::logic_error("generate() off branch tip");
    auto table = kv.clone_block_table(p.kv_block_table);
    kv.prepare_write_slot(table, p.seq_pos + 1);
    const GenOut g = src_(p, tree.active_fork_id);
    clock.seek(p.handshake_tick);
    const uint64_t tick = clock.commit_token();
    return tree.append_generated_token(tick, g.token, g.next_seed, g.logit_digest, table);
}

uint32_t HostSession::scrub_forward() noexcept {
    const uint32_t n = tree.step_forward();
    if (n != INVALID_ID) clock.seek(tree.nodes[n].handshake_tick);
    return n;
}
uint32_t HostSession::scrub_back() noexcept {
    const uint32_t n = tree.step_back();
    clock.seek(tree.nodes[n].handshake_tick);
    return n;
}

uint64_t HostSession::world_digest() const noexcept {
    uint64_t h = hash_mix64(tree.tree_digest(), world.digest());
    const RobotPose p = pose_from_lineage(tree, tree.cursor_node_id, world);
    h = hash_mix64(h, uint32_t(p.x)); h = hash_mix64(h, uint32_t(p.y));
    h = hash_mix64(h, uint32_t(p.z)); h = hash_mix64(h, uint32_t(p.yaw));
    return h;
}

GenOut stub_token_source(const TokenRecord& p, uint32_t fork_id) noexcept {
    uint64_t rng = p.step_seed ^ (uint64_t(fork_id) << 32);
    const uint32_t tok = uint32_t(splitmix64(rng) % 32000u);
    const uint64_t d = splitmix64(rng);
    return {tok, rng, d};
}

void HandshakeLedger::snapshot(uint32_t node_id, uint64_t engine_digest) {
    if (snaps_.size() <= node_id) snaps_.resize(size_t(node_id) + 1);
    if (!snaps_[node_id].valid) ++count_;
    snaps_[node_id] = {clock_.now(), engine_digest, true};
}
bool HandshakeLedger::rollback(uint32_t node_id, uint64_t& out) noexcept {
    const EngineSnapshot* s = find(node_id);
    if (!s) return false;
    clock_.seek(s->tick); out = s->engine_digest; return true;
}
const EngineSnapshot* HandshakeLedger::find(uint32_t node_id) const noexcept {
    return (node_id < snaps_.size() && snaps_[node_id].valid) ? &snaps_[node_id] : nullptr;
}
uint64_t engine_digest_at(const VariationTree& t, uint32_t node_id, const FlatWorld& w) noexcept {
    uint64_t h = hash_mix64(w.digest(), node_id);
    h = hash_mix64(h, t.nodes[node_id].handshake_tick);
    const RobotPose p = pose_from_lineage(t, node_id, w);
    h = hash_mix64(h, uint32_t(p.x)); h = hash_mix64(h, uint32_t(p.y));
    h = hash_mix64(h, uint32_t(p.z)); h = hash_mix64(h, uint32_t(p.yaw));
    return h;
}

} // namespace pchie