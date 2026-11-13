// Bolt PCHIE B6 - recording chain that also keeps payloads for injection audit.
// Concrete final IHashChain. Official head() is SHA-256. Coordinator-thread
// only. No mutex, no atomic. Used to prove poison strings never enter records.
#pragma once

#include "pchie/GauntletChainTrace.h"
#include "pchie/ChainRecord.h"
#include "pchie/IHashChain.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class InjectionChainAudit final : public IHashChain {
 public:
  void reset() override {
    trace_.reset();
    payloads_.clear();
  }

  void append(const ChainRecord& r) override {
    trace_.append(r);
    payloads_.push_back(r.payload);
  }

  std::vector<uint8_t> head() const override { return trace_.head(); }
  size_t length() const override { return trace_.length(); }

  void restore_head(const std::vector<uint8_t>& h, size_t n) override {
    trace_.restore_head(h, n);
    if (n == 0) {
      payloads_.clear();
      return;
    }
    if (n <= payloads_.size()) payloads_.resize(n);
  }

  std::vector<uint8_t> fnv_head() const { return trace_.fnv_head(); }
  const GauntletChainTrace& trace() const { return trace_; }
  const std::vector<std::vector<uint8_t>>& payloads() const { return payloads_; }

  bool payload_contains(const std::string& needle) const {
    if (needle.empty()) return false;
    const auto* n = reinterpret_cast<const uint8_t*>(needle.data());
    const size_t nl = needle.size();
    for (const auto& p : payloads_) {
      if (p.size() < nl) continue;
      for (size_t i = 0; i + nl <= p.size(); ++i) {
        bool eq = true;
        for (size_t j = 0; j < nl; ++j) {
          if (p[i + j] != n[j]) {
            eq = false;
            break;
          }
        }
        if (eq) return true;
      }
    }
    return false;
  }

 private:
  GauntletChainTrace trace_;
  std::vector<std::vector<uint8_t>> payloads_;
};

}  // namespace pchie
