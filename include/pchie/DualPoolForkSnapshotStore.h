// Bolt PCHIE - CoW snapshot of two KV pools (SWA + global) plus chain head.
// Mirrors ForkSnapshotStore but covers both pools. No mutex; coordinator-thread only.
#pragma once

#include <cstdint>
#include <vector>
#include "pchie/ModelBranch.h"
#include "pchie/pchie_kv_pools.h"

namespace pchie {

struct DualPoolForkSnapshot {
  uint32_t node_id = 0;
  ModelBranch branch;
  std::vector<uint8_t> chain_head;
  size_t chain_length = 0;
  bool valid = false;
};

class DualPoolForkSnapshotStore final {
 public:
  DualPoolForkSnapshotStore(KVPagePool& swa, KVPagePool& glob) : swa_(swa), glob_(glob) {}

  ~DualPoolForkSnapshotStore() {
    for (DualPoolForkSnapshot& s : snaps_) release_snap(s);
  }

  DualPoolForkSnapshotStore(const DualPoolForkSnapshotStore&) = delete;
  DualPoolForkSnapshotStore& operator=(const DualPoolForkSnapshotStore&) = delete;

  void snapshot(uint32_t node_id, const ModelBranch& b,
                const std::vector<uint8_t>& chain_head, size_t chain_length) {
    if (node_id >= snaps_.size()) snaps_.resize(node_id + 1);
    DualPoolForkSnapshot& s = snaps_[node_id];
    release_snap(s);
    s.node_id = node_id;
    s.branch.swa = swa_.fork(b.swa);
    s.branch.glob = glob_.fork(b.glob);
    s.branch.n_past = b.n_past;
    s.chain_head = chain_head;
    s.chain_length = chain_length;
    s.valid = true;
  }

  bool rollback(uint32_t node_id, ModelBranch& out,
                std::vector<uint8_t>& chain_head_out, size_t& chain_length_out) {
    const DualPoolForkSnapshot* s = find(node_id);
    if (!s || !s->valid) return false;
    // out may already hold pages; drop them before overwriting the tables.
    swa_.release(out.swa);
    glob_.release(out.glob);
    out.swa = swa_.fork(s->branch.swa);
    out.glob = glob_.fork(s->branch.glob);
    out.n_past = s->branch.n_past;
    chain_head_out = s->chain_head;
    chain_length_out = s->chain_length;
    return true;
  }

  // drop one snapshot / every snapshot at node_id >= first. Used when the
  // PUCT tree is stepped back so node ids at or above the restored size will be reused;
  // their old KV snapshots must never be rolled back to (and their pages are freed).
  void invalidate(uint32_t node_id) {
    if (node_id < snaps_.size()) release_snap(snaps_[node_id]);
  }
  void invalidate_from(uint32_t first) {
    for (size_t i = first; i < snaps_.size(); ++i) release_snap(snaps_[i]);
  }
  size_t valid_count() const noexcept {
    size_t n = 0;
    for (const auto& s : snaps_) n += s.valid ? 1u : 0u;
    return n;
  }

  const DualPoolForkSnapshot* find(uint32_t node_id) const noexcept {
    if (node_id >= snaps_.size() || !snaps_[node_id].valid) return nullptr;
    return &snaps_[node_id];
  }

 private:
  void release_snap(DualPoolForkSnapshot& s) {
    if (!s.valid) return;
    swa_.release(s.branch.swa);
    glob_.release(s.branch.glob);
    s.valid = false;
  }

  KVPagePool& swa_;
  KVPagePool& glob_;
  std::vector<DualPoolForkSnapshot> snaps_;
};

}  // namespace pchie