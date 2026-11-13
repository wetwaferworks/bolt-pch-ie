// Bolt PCHIE - FNV-1a64 hash chain (hot path, concrete final).
// h_0 = FNV_OFFSET; h_n = FNV1a64_continue(h_{n-1}, canonical_bytes(record_n)).
#pragma once
#include <cstdint>
#include <vector>
#include "pchie/IHashChain.h"
#include "pchie/pchie_tree.h"

namespace pchie {

class Fnv1a64HashChain final : public IHashChain {
 public:
  static constexpr size_t kHeadBytes = 8;

  void reset() override {
    head_ = FNV_OFFSET;
    length_ = 0;
  }

  void append(const ChainRecord& record) override {
    const std::vector<uint8_t> bytes = record.canonical_bytes();
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    uint64_t h = head_;
    for (uint8_t b : bytes) {
      h ^= uint64_t(b);
      h *= FNV_PRIME;
    }
    head_ = h;
    ++length_;
  }

  std::vector<uint8_t> head() const override {
    std::vector<uint8_t> out(8);
    for (int i = 0; i < 8; ++i) out[size_t(i)] = uint8_t((head_ >> (8 * i)) & 0xFFu);
    return out;
  }

  size_t length() const override { return length_; }

  void restore_head(const std::vector<uint8_t>& head, size_t length) override {
    head_ = 0;
    for (size_t i = 0; i < head.size() && i < 8; ++i) head_ |= (uint64_t(head[i]) << (8 * i));
    length_ = length;
  }

  uint64_t head_u64() const noexcept { return head_; }

 private:
  uint64_t head_ = FNV_OFFSET;
  size_t length_ = 0;
};

}  // namespace pchie