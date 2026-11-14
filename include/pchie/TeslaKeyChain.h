// Bolt PCHIE - TESLA-manner interval key chain for stream authentication, off by default.
//
// In the manner of RFC 4082 (TESLA), adapted so the chain's own order is the clock: there is no wall clock
// in the safety test. Interval keys authenticate transport packets. They are a separate chain from the
// ratcheted seal keys (Signal's Double Ratchet has no RFC; never put an RFC number on it). Never derive
// one chain from the other.
//
// Construction, for n interval keys K_0 .. K_{n-1}:
//   K_{n-1} is the secret seed (or is derived from it). K_i = F(K_{i+1}), with
//   F(k) = SHA-256("pchie.tesla.v1.chain" || k). The public commitment is C = F(K_0), one extra hash so
//   K_0 itself stays secret until it is disclosed. A disclosed K_i checks as F^{i+1}(K_i) == C.
//   The MAC key is K'_i = HMAC-SHA-256(K_i, "pchie.tesla.v1.mac"), never the chain key itself.
//
// Default schedule: I = 8 sealed steps per interval, disclosure lag d = 2.
// This layer is additive. Maze drivers turn TESLA on by default via StreamAuthSession;
// --tesla off turns it off for testing. Official seals are unchanged.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "pchie/HmacSha256.h"
#include "pchie/Sha256.h"

namespace pchie {

struct TeslaSchedule {
  static constexpr uint32_t kMaxI = 1u << 16;
  static constexpr uint32_t kMaxD = 1u << 10;
  uint32_t I = 8;  // sealed steps per interval
  uint32_t d = 2;  // disclosure lag, in intervals (RFC 4082: d = 1 does not work)
  void check() const {
    if (I == 0) throw std::invalid_argument("TESLA schedule: I must be at least 1");
    if (d < 2) throw std::invalid_argument("TESLA schedule: d must be at least 2");
    // Upper bounds keep index arithmetic far from overflow when a schedule comes from an untrusted file.
    if (I > kMaxI) throw std::invalid_argument("TESLA schedule: I is too large");
    if (d > kMaxD) throw std::invalid_argument("TESLA schedule: d is too large");
  }
};

inline constexpr const char* kTeslaChainTag = "pchie.tesla.v1.chain";
inline constexpr const char* kTeslaMacTag = "pchie.tesla.v1.mac";

inline std::vector<uint8_t> tesla_chain_step(const std::vector<uint8_t>& key) {
  if (key.size() != Sha256::kDigestBytes) throw std::invalid_argument("TESLA chain step: key must be 32 bytes");
  std::vector<uint8_t> b(kTeslaChainTag, kTeslaChainTag + 20);
  b.insert(b.end(), key.begin(), key.end());
  return Sha256::hash(b.data(), b.size());
}

inline std::vector<uint8_t> tesla_mac_key(const std::vector<uint8_t>& interval_key) {
  if (interval_key.size() != Sha256::kDigestBytes) throw std::invalid_argument("TESLA MAC key: interval key must be 32 bytes");
  const std::vector<uint8_t> tag(kTeslaMacTag, kTeslaMacTag + 18);
  return hmac_sha256(interval_key, tag);
}

// commitment_id is what an anchor record would carry: a hash of the public commitment, not the secret seed.
inline std::vector<uint8_t> tesla_commitment_id(const std::vector<uint8_t>& commitment) {
  const char* t = "pchie.tesla.v1.commitment-id";
  std::vector<uint8_t> b(t, t + 28);
  b.insert(b.end(), commitment.begin(), commitment.end());
  return Sha256::hash(b.data(), b.size());
}

class TeslaKeyChain {
 public:
  // seed is secret and must be at least 32 bytes. Only the first 32 bytes are used as K_{n-1}.
  // n is the number of interval keys (must cover every interval the run will emit, including a short last one).
  TeslaKeyChain(std::vector<uint8_t> seed, uint32_t n, TeslaSchedule sch = {}) : sch_(sch) {
    sch_.check();
    if (seed.size() < 32) throw std::invalid_argument("TESLA key chain: seed must be at least 32 bytes");
    if (n == 0) throw std::invalid_argument("TESLA key chain: need at least one interval");
    keys_.resize(n);
    keys_[n - 1].assign(seed.begin(), seed.begin() + 32);
    wipe_bytes(seed);
    for (uint32_t i = n - 1; i > 0; --i) keys_[i - 1] = tesla_chain_step(keys_[i]);
    commitment_ = tesla_chain_step(keys_[0]);
  }
  TeslaKeyChain(const TeslaKeyChain&) = delete;
  TeslaKeyChain& operator=(const TeslaKeyChain&) = delete;
  TeslaKeyChain(TeslaKeyChain&&) = default;
  TeslaKeyChain& operator=(TeslaKeyChain&&) = default;
  ~TeslaKeyChain() { for (auto& k : keys_) wipe_bytes(k); }

  const std::vector<uint8_t>& commitment() const { return commitment_; }
  uint32_t size() const { return uint32_t(keys_.size()); }
  TeslaSchedule schedule() const { return sch_; }

  // A copy of interval key i, for disclosure. The chain keeps its own copy until destruction.
  std::vector<uint8_t> key(uint32_t i) const {
    if (i >= keys_.size()) throw std::out_of_range("TESLA key chain: interval index out of range");
    return keys_[i];
  }
  std::vector<uint8_t> mac_key(uint32_t i) const {
    std::vector<uint8_t> k = key(i);
    std::vector<uint8_t> m = tesla_mac_key(k);
    wipe_bytes(k);
    return m;
  }

  // F^{i+1}(disclosed) == commitment.
  static bool key_belongs(const std::vector<uint8_t>& disclosed, uint32_t i, const std::vector<uint8_t>& commitment) {
    if (disclosed.size() != Sha256::kDigestBytes || commitment.size() != Sha256::kDigestBytes) return false;
    std::vector<uint8_t> x = disclosed;
    // 64-bit count: i = 0xFFFFFFFF must mean 2^32 steps, never wrap to zero steps (which would accept
    // the public commitment itself as a key). Callers cap i before calling; see TeslaWatcher.
    const uint64_t steps = uint64_t(i) + 1;
    for (uint64_t n = 0; n < steps; ++n) x = tesla_chain_step(x);
    const bool ok = x == commitment;
    wipe_bytes(x);
    return ok;
  }
  // Hash a later key back to an earlier one: K_j = F^{i-j}(K_i) for j < i.
  static std::vector<uint8_t> derive_earlier(const std::vector<uint8_t>& later_key, uint32_t later_i, uint32_t earlier_i) {
    if (earlier_i > later_i) throw std::invalid_argument("TESLA derive: earlier interval must be <= later");
    std::vector<uint8_t> x = later_key;
    for (uint32_t n = 0; n < later_i - earlier_i; ++n) x = tesla_chain_step(x);
    return x;
  }

 private:
  TeslaSchedule sch_;
  std::vector<std::vector<uint8_t>> keys_;
  std::vector<uint8_t> commitment_;
};

}  // namespace pchie
