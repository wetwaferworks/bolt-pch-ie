// Bolt PCHIE - fork snapshot runtime boundary.
#pragma once
#include <cstdint>
#include <vector>
#include "pchie/pchie_kv_pools.h"

namespace pchie {

struct ForkSnapshot {
  uint32_t node_id = 0;
  KVPagePool::Table kv_refs;     // CoW-shared page refs (not a full copy)
  uint64_t sim_snapshot_id = 0;  // handshake/engine snapshot id
  std::vector<uint8_t> chain_head;
  size_t chain_length = 0;
  bool valid = false;
};

class IForkSnapshot {
 public:
  virtual ~IForkSnapshot() = default;
  virtual void snapshot(uint32_t node_id,
                        const KVPagePool::Table& kv,
                        uint64_t sim_snapshot_id,
                        const std::vector<uint8_t>& chain_head,
                        size_t chain_length) = 0;
  virtual bool rollback(uint32_t node_id,
                        KVPagePool::Table& kv_out,
                        uint64_t& sim_snapshot_id_out,
                        std::vector<uint8_t>& chain_head_out,
                        size_t& chain_length_out) = 0;
  virtual const ForkSnapshot* find(uint32_t node_id) const noexcept = 0;
  virtual size_t size() const noexcept = 0;
};

}  // namespace pchie