// Bolt PCHIE - SHA-256 (one class). Deterministic, no locks, host CPU only.
#pragma once
#include <cstdint>
#include <vector>

namespace pchie {

class Sha256 final {
 public:
  static constexpr size_t kDigestBytes = 32;
  Sha256();
  void reset();
  void update(const uint8_t* data, size_t n);
  void final(uint8_t out[kDigestBytes]);
  static std::vector<uint8_t> hash(const uint8_t* data, size_t n);

 private:
  void transform(const uint8_t block[64]);
  uint32_t state_[8];
  uint64_t bitlen_;
  uint64_t total_bytes_;
  uint8_t buf_[64];
  uint32_t buflen_;
};

}  // namespace pchie