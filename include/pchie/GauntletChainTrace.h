// Bolt PCHIE B4 - recording hash chain: per-record kind + SHA-256 / FNV heads.
// Concrete final IHashChain. Official head() is SHA-256 (32). Coordinator-thread
// only. No mutex, no atomic. Used to name the first differing step and record type.
#pragma once

#include "pchie/ChainRecord.h"
#include "pchie/Fnv1a64HashChain.h"
#include "pchie/IHashChain.h"
#include "pchie/Sha256HashChain.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pchie {

class GauntletChainTrace final : public IHashChain {
 public:
  void reset() override {
    sha_.reset();
    fnv_.reset();
    kinds_.clear();
    heads_.clear();
    fnv_heads_.clear();
  }

  void append(const ChainRecord& r) override {
    sha_.append(r);
    fnv_.append(r);
    kinds_.push_back(r.kind);
    heads_.push_back(sha_.head());
    fnv_heads_.push_back(fnv_.head());
  }

  std::vector<uint8_t> head() const override { return sha_.head(); }
  size_t length() const override { return sha_.length(); }

  void restore_head(const std::vector<uint8_t>& h, size_t n) override {
    sha_.restore_head(h, n);
    if (n == 0 || n > heads_.size()) {
      if (n == 0) {
        fnv_.reset();
        kinds_.clear();
        heads_.clear();
        fnv_heads_.clear();
      } else {
        // Cannot reconstruct FNV / kinds past the recorded prefix (official B1
        // chain is append-only; this path should not run).
        fnv_.restore_head(std::vector<uint8_t>(Fnv1a64HashChain::kHeadBytes, 0), n);
      }
      return;
    }
    kinds_.resize(n);
    heads_.resize(n);
    fnv_heads_.resize(n);
    fnv_.restore_head(fnv_heads_.back(), n);
  }

  std::vector<uint8_t> fnv_head() const { return fnv_.head(); }
  const std::vector<ChainRecordKind>& kinds() const { return kinds_; }
  const std::vector<std::vector<uint8_t>>& heads() const { return heads_; }

 private:
  Sha256HashChain sha_;
  Fnv1a64HashChain fnv_;
  std::vector<ChainRecordKind> kinds_;
  std::vector<std::vector<uint8_t>> heads_;
  std::vector<std::vector<uint8_t>> fnv_heads_;
};

}  // namespace pchie
