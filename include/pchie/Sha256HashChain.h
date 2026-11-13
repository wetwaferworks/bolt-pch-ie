// Bolt PCHIE - SHA-256 hash chain for exported logs (concrete final).
// h_0 = 32 zero bytes; h_n = SHA256(h_{n-1} || canonical_bytes(record_n)).
#pragma once
#include <cstdint>
#include <vector>
#include "pchie/IHashChain.h"
#include "pchie/Sha256.h"

namespace pchie {

class Sha256HashChain final : public IHashChain {
 public:
  static constexpr size_t kHeadBytes = Sha256::kDigestBytes;

  void reset() override {
    head_.assign(kHeadBytes, 0);
    length_ = 0;
  }

  void append(const ChainRecord& record) override {
    const std::vector<uint8_t> rec = record.canonical_bytes();
    std::vector<uint8_t> cat;
    cat.reserve(head_.size() + rec.size());
    cat.insert(cat.end(), head_.begin(), head_.end());
    cat.insert(cat.end(), rec.begin(), rec.end());
    head_ = Sha256::hash(cat.data(), cat.size());
    ++length_;
  }

  std::vector<uint8_t> head() const override { return head_; }
  size_t length() const override { return length_; }

  void restore_head(const std::vector<uint8_t>& head, size_t length) override {
    head_ = head;
    if (head_.size() != kHeadBytes) head_.resize(kHeadBytes, 0);
    length_ = length;
  }

 private:
  std::vector<uint8_t> head_ = std::vector<uint8_t>(kHeadBytes, 0);
  size_t length_ = 0;
};

}  // namespace pchie