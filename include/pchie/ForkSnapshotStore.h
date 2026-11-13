// Bolt PCHIE - ForkSnapshotStore: CoW page refs + sim id + chain head (delta-log, no full copies).
// No std::mutex. Single-threaded / coordinator use (phase-boundary).
#pragma once
#include <cstdint>
#include <vector>
#include "pchie/IForkSnapshot.h"

namespace pchie {

class ForkSnapshotStore final : public IForkSnapshot {
 public:
  explicit ForkSnapshotStore(KVPagePool& pool) : pool_(pool) {}

  ~ForkSnapshotStore() override {
    for (ForkSnapshot& s : snaps_) {
      if (s.valid) { pool_.release(s.kv_refs); s.valid = false; }
    }
  }

  ForkSnapshotStore(const ForkSnapshotStore&) = delete;
  ForkSnapshotStore& operator=(const ForkSnapshotStore&) = delete;

  void snapshot(uint32_t node_id,
                const KVPagePool::Table& kv,
                uint64_t sim_snapshot_id,
                const std::vector<uint8_t>& chain_head,
                size_t chain_length) override {
    if (node_id >= snaps_.size()) snaps_.resize(node_id + 1);
    ForkSnapshot& s = snaps_[node_id];
    if (s.valid) pool_.release(s.kv_refs);
    s.node_id = node_id;
    s.kv_refs = pool_.fork(kv);  // bump refs; share pages (CoW)
    s.sim_snapshot_id = sim_snapshot_id;
    s.chain_head = chain_head;
    s.chain_length = chain_length;
    s.valid = true;
  }

  bool rollback(uint32_t node_id,
                KVPagePool::Table& kv_out,
                uint64_t& sim_snapshot_id_out,
                std::vector<uint8_t>& chain_head_out,
                size_t& chain_length_out) override {
    const ForkSnapshot* s = find(node_id);
    if (!s || !s->valid) return false;
    kv_out = pool_.fork(s->kv_refs);  // caller owns a shared view; stored snap kept
    sim_snapshot_id_out = s->sim_snapshot_id;
    chain_head_out = s->chain_head;
    chain_length_out = s->chain_length;
    return true;
  }

  const ForkSnapshot* find(uint32_t node_id) const noexcept override {
    if (node_id >= snaps_.size()) return nullptr;
    return snaps_[node_id].valid ? &snaps_[node_id] : nullptr;
  }

  size_t size() const noexcept override {
    size_t n = 0;
    for (const auto& s : snaps_) if (s.valid) ++n;
    return n;
  }

 private:
  KVPagePool& pool_;
  std::vector<ForkSnapshot> snaps_;
};

}  // namespace pchie