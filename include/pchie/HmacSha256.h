// Bolt PCHIE - HMAC-SHA256 (RFC 2104) and HKDF-SHA256 (RFC 5869), written in-house on pchie::Sha256.
// One implementation for the whole tree: the ratcheted seal chain and the anchoring side tool both use it.
// Checked against the RFC 4231 and RFC 5869 test vectors in anchoring/tests/anchor_unit.cpp.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "pchie/Sha256.h"

namespace pchie {

inline std::vector<uint8_t> hmac_sha256(const std::vector<uint8_t>& key, const std::vector<uint8_t>& msg) {
  std::vector<uint8_t> k = key;
  if (k.size() > 64) k = Sha256::hash(k.data(), k.size());
  k.resize(64, 0);
  std::vector<uint8_t> inner(64), outer(64);
  for (int i = 0; i < 64; ++i) { inner[i] = uint8_t(k[i] ^ 0x36); outer[i] = uint8_t(k[i] ^ 0x5c); }
  inner.insert(inner.end(), msg.begin(), msg.end());
  const std::vector<uint8_t> ih = Sha256::hash(inner.data(), inner.size());
  outer.insert(outer.end(), ih.begin(), ih.end());
  return Sha256::hash(outer.data(), outer.size());
}

// HKDF-Extract: an empty salt means 32 zero bytes, per RFC 5869 s2.2.
inline std::vector<uint8_t> hkdf_extract_sha256(const std::vector<uint8_t>& salt, const std::vector<uint8_t>& ikm) {
  return hmac_sha256(salt.empty() ? std::vector<uint8_t>(32, 0) : salt, ikm);
}

inline std::vector<uint8_t> hkdf_expand_sha256(const std::vector<uint8_t>& prk, const std::vector<uint8_t>& info, size_t len) {
  std::vector<uint8_t> okm, t;
  for (uint8_t i = 1; okm.size() < len; ++i) {
    std::vector<uint8_t> m = t;
    m.insert(m.end(), info.begin(), info.end());
    m.push_back(i);
    t = hmac_sha256(prk, m);
    okm.insert(okm.end(), t.begin(), t.end());
  }
  okm.resize(len);
  return okm;
}

// Overwrite key material before it is released. The volatile write keeps the compiler from dropping it.
inline void wipe_bytes(std::vector<uint8_t>& v) {
  volatile uint8_t* p = v.data();
  for (size_t i = 0; i < v.size(); ++i) p[i] = 0;
  v.clear();
}

}  // namespace pchie
