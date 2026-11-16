// Bolt PCHIE - HMAC-SHA256 (RFC 2104) and HKDF-SHA256 (RFC 5869), written in-house on pchie::Sha256.
// One implementation for the whole tree: the ratcheted seal chain and the anchoring side tool both use it.
// Checked against the RFC 4231 and RFC 5869 test vectors in anchoring/tests/anchor_unit.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "pchie/Sha256.h"

namespace pchie {

inline void wipe_bytes(std::vector<uint8_t>& v);

// Scope guards also clear partially computed secret buffers on allocation failures.
struct CryptoWipeGuard {
  std::vector<uint8_t>& bytes;
  ~CryptoWipeGuard() { wipe_bytes(bytes); }
};

inline std::vector<uint8_t> hmac_sha256(const std::vector<uint8_t>& key, const std::vector<uint8_t>& msg) {
  std::vector<uint8_t> k = key;
  CryptoWipeGuard kg{k};
  if (k.size() > 64) { std::vector<uint8_t> hk = Sha256::hash(k.data(), k.size()); wipe_bytes(k); k = std::move(hk); }
  k.resize(64, 0);
  std::vector<uint8_t> inner, outer;
  CryptoWipeGuard ig{inner}, og{outer};
  if (msg.size() > inner.max_size() - 64) throw std::length_error("hmac message too large");
  inner.reserve(64 + msg.size()); outer.reserve(96);
  inner.resize(64); outer.resize(64);
  for (int i = 0; i < 64; ++i) { inner[i] = uint8_t(k[i] ^ 0x36); outer[i] = uint8_t(k[i] ^ 0x5c); }
  wipe_bytes(k);
  inner.reserve(64 + msg.size());
  inner.insert(inner.end(), msg.begin(), msg.end());
  std::vector<uint8_t> ih = Sha256::hash(inner.data(), inner.size());
  CryptoWipeGuard ihg{ih};
  outer.reserve(64 + ih.size());
  outer.insert(outer.end(), ih.begin(), ih.end());
  std::vector<uint8_t> mac = Sha256::hash(outer.data(), outer.size());
  // The pads are key material and the inner hash depends on the key; none of them outlive the call.
  wipe_bytes(inner); wipe_bytes(outer); wipe_bytes(ih);
  return mac;
}

// HKDF-Extract: an empty salt means 32 zero bytes, per RFC 5869 s2.2.
inline std::vector<uint8_t> hkdf_extract_sha256(const std::vector<uint8_t>& salt, const std::vector<uint8_t>& ikm) {
  return hmac_sha256(salt.empty() ? std::vector<uint8_t>(32, 0) : salt, ikm);
}

// RFC 5869 s2.3: at most 255 blocks of 32 bytes. Asking for more is refused instead of wrapping the counter.
constexpr size_t kHkdfSha256MaxLen = 255 * 32;
inline std::vector<uint8_t> hkdf_expand_sha256(const std::vector<uint8_t>& prk, const std::vector<uint8_t>& info, size_t len) {
  if (len > kHkdfSha256MaxLen) throw std::length_error("hkdf_expand_sha256: more than 255 * 32 bytes requested");
  std::vector<uint8_t> okm, t;
  CryptoWipeGuard tg{t};
  okm.reserve(len + 32);
  for (unsigned i = 1; okm.size() < len; ++i) {
    std::vector<uint8_t> m;
    CryptoWipeGuard mg{m};
    if (info.size() > m.max_size() - 33) throw std::length_error("hkdf info too large");
    m.reserve(33 + info.size());
    m.insert(m.end(), t.begin(), t.end());
    m.insert(m.end(), info.begin(), info.end());
    m.push_back(uint8_t(i));
    wipe_bytes(t);
    t = hmac_sha256(prk, m);
    wipe_bytes(m);
    okm.insert(okm.end(), t.begin(), t.end());
  }
  wipe_bytes(t);
  if (okm.size() > len) { volatile uint8_t* p = okm.data(); for (size_t j = len; j < okm.size(); ++j) p[j] = 0; }
  okm.resize(len);
  return okm;
}

// Overwrite key material before it is released. The volatile write keeps the compiler from dropping it.
inline void wipe_bytes(std::vector<uint8_t>& v) {
  volatile uint8_t* p = v.data();
  for (size_t i = 0; i < v.size(); ++i) p[i] = 0;
  v.clear();
}

// Compare a computed MAC or signature with a presented one in time that depends only on the lengths, so the
// position of the first differing byte is not visible. Lengths are not secret.
inline bool ct_equal(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  if (a.size() != b.size()) return false;
  volatile uint8_t acc = 0;
  for (size_t i = 0; i < a.size(); ++i) acc = uint8_t(acc | (a[i] ^ b[i]));
  return acc == 0;
}

}  // namespace pchie
