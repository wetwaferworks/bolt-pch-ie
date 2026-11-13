// Bolt PCHIE - hash-chain runtime boundary (virtual). Hot-path appenders are
// concrete final classes (Fnv1a64HashChain / Sha256HashChain).
// CODE REVIEW NOTE (Archie): thin virtual IHashChain boundary with concrete
// final Fnv1a64HashChain / Sha256HashChain behind it is approved for the hash-chain and model layers.
// Coordinator thread only: worker threads never append (plan Archie amendment 3).
#pragma once
#include <cstdint>
#include <vector>
#include "pchie/ChainRecord.h"

namespace pchie {

class IHashChain {
 public:
  virtual ~IHashChain() = default;
  virtual void reset() = 0;
  virtual void append(const ChainRecord& record) = 0;
  // Current head bytes: 8 for FNV-1a64, 32 for SHA-256.
  virtual std::vector<uint8_t> head() const = 0;
  virtual size_t length() const = 0;  // number of appended records
  // Snapshot/rollback of the chain head (and length) without replaying.
  virtual void restore_head(const std::vector<uint8_t>& head, size_t length) = 0;
  // Side wrappers (TESLA, etc.) return the official inner chain so type-sensitive
  // helpers such as reroot_if_ratcheted still see a RatchetHashChain. Default is this.
  virtual IHashChain* official_chain() { return this; }
  virtual const IHashChain* official_chain() const { return this; }
};


}  // namespace pchie
