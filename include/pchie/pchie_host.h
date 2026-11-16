// Bolt PCHIE - engine host seam: a clean-room Timing Handshake clock, a flat
// voxel world (no procedural generation), and generation/scrub modes over the variation tree.
#pragma once
#include "pchie/pchie_tree.h"
#include "pchie/pchie_kv_cow.h"
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace pchie {

// Monotonic simulation clock. Generation mode commits each token exactly K ticks
// after its parent, independent of wall-clock latency (slow CPU == fast GPU stamps).
class TimingHandshake {
public:
    explicit TimingHandshake(uint64_t start_tick, uint32_t ticks_per_token)
        : tick_(start_tick), k_(ticks_per_token) { if (k_ == 0) throw std::invalid_argument("ticks_per_token must be greater than zero"); }
    uint64_t now() const noexcept { return tick_; }
    uint32_t ticks_per_token() const noexcept { return k_; }
    uint64_t commit_token() noexcept { tick_ += k_; return tick_; }
    void     seek(uint64_t t) noexcept { tick_ = t; }
private:
    uint64_t tick_;
    uint32_t k_;
};

// Completely flat world: solid floor for y < floor_y, air above. No noise, no biomes.
struct FlatWorld {
    int32_t size_x = 64, size_y = 16, size_z = 64, floor_y = 4;
    uint8_t block_at(int32_t x, int32_t y, int32_t z) const noexcept;
    uint64_t digest() const noexcept;
};

// Robot pose in fixed point (1/1024 voxel), derived only from tokens: deterministic.
struct RobotPose { int32_t x, y, z, yaw; };
RobotPose pose_from_lineage(const VariationTree& t, uint32_t node_id, const FlatWorld& w) noexcept;
// Interpolated pose at an arbitrary tick between two committed tokens (integer lerp).
RobotPose pose_at_tick(const VariationTree& t, uint32_t fork_id, uint64_t tick, const FlatWorld& w) noexcept;

enum class HostMode { Generation, Scrub };

// Token source: (parent record, fork salt) -> {token, next_seed, logit_digest}.
struct GenOut { uint32_t token; uint64_t next_seed; uint64_t logit_digest; };
using TokenSource = std::function<GenOut(const TokenRecord& parent, uint32_t fork_id)>;

class HostSession {
public:
    HostSession(uint64_t start_tick, uint32_t k, uint64_t seed, TokenSource src);
    HostMode mode = HostMode::Generation;
    VariationTree tree;
    TimingHandshake clock;
    FlatWorld world;
    PagedKVCache kv;

    // Generation mode: run inference for one token on the active fork tip.
    uint32_t generate();
    // Scrub mode: never calls the token source; returns the recorded node or INVALID_ID.
    uint32_t scrub_forward() noexcept;
    uint32_t scrub_back() noexcept;
    uint64_t world_digest() const noexcept; // tree + world + pose at cursor
private:
    TokenSource src_;
};

GenOut stub_token_source(const TokenRecord& parent, uint32_t fork_id) noexcept;

// engine snapshot / rollback keyed by variation-tree node ID, over the public
// Timing Handshake only. A snapshot records the handshake tick and an engine digest
// (world + pose) when a node is committed; rollback(node) seeks the clock back to that
// tick and hands back the digest so the caller can verify the restored engine state.
struct EngineSnapshot { uint64_t tick = 0; uint64_t engine_digest = 0; bool valid = false; };
class HandshakeLedger {
public:
    explicit HandshakeLedger(TimingHandshake& clock) : clock_(clock) {}
    // Commit a child of `parent_tick` on the clock (tick = parent + K), returns the tick.
    uint64_t commit_child(uint64_t parent_tick) noexcept { clock_.seek(parent_tick); return clock_.commit_token(); }
    void snapshot(uint32_t node_id, uint64_t engine_digest);
    bool rollback(uint32_t node_id, uint64_t& engine_digest_out) noexcept;
    const EngineSnapshot* find(uint32_t node_id) const noexcept;
    size_t size() const noexcept { return count_; }
    void clear() noexcept { snaps_.clear(); count_ = 0; }
private:
    TimingHandshake& clock_;
    std::vector<EngineSnapshot> snaps_;   // indexed by node ID
    size_t count_ = 0;
};
// Engine digest at a node: flat world + robot pose derived from the node's lineage.
uint64_t engine_digest_at(const VariationTree& t, uint32_t node_id, const FlatWorld& w) noexcept;

} // namespace pchie