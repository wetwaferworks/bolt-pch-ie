// Bolt PCHIE - DualPool step-back hooked to MctsSearch node ids.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// Restores KV page refs + n_past + chain head only (not the PUCT tree).
// Uses DualPoolForkSnapshotStore; does not own the model.
#pragma once

#include "pchie/ChainEpoch.h"
#include "pchie/DualPoolForkSnapshotStore.h"
#include "pchie/IHashChain.h"
#include "pchie/ModelBranch.h"
#include "pchie/pchie_kv_pools.h"

#include <cstdint>
#include <vector>

namespace pchie {

class MctsDualPoolStepBack final {
 public:
  MctsDualPoolStepBack(KVPagePool& swa, KVPagePool& glob) : store_(swa, glob) {}

  MctsDualPoolStepBack(const MctsDualPoolStepBack&) = delete;
  MctsDualPoolStepBack& operator=(const MctsDualPoolStepBack&) = delete;

  void snapshot(uint32_t node_id, const ModelBranch& b, const IHashChain& chain) {
    store_.snapshot(node_id, b, chain.head(), chain.length());
  }

  bool rollback(uint32_t node_id, ModelBranch& out, IHashChain& chain) {
    std::vector<uint8_t> head;
    size_t len = 0;
    if (!store_.rollback(node_id, out, head, len)) return false;
    chain.restore_head(head, len);
    // Fork point: a ratcheted chain opens a new epoch here; other chains are left as they were.
    reroot_if_ratcheted(&chain, "fork", "dualpool.rollback", {uint64_t(node_id), uint64_t(len)});
    return true;
  }

  DualPoolForkSnapshotStore& store() { return store_; }
  const DualPoolForkSnapshotStore& store() const { return store_; }

 private:
  DualPoolForkSnapshotStore store_;
};

}  // namespace pchie