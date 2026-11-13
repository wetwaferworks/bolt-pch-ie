// Bolt PCHIE reasoning layers (opt-in scratch, 2026-11-11).
// Flags default off. Official runners must never set them. Nothing here is
// locked, committed, or official. Coordinator-thread only: no mutex, no atomic.
// Untrusted observations go through tokenize_untrusted (parse_special=false, no BOS).
// New records go to a scratch side chain only, never the official IHashChain.
#pragma once

#include "pchie/Sha256.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

constexpr int32_t kQ20 = 1 << 20;  // 1048576, same scale as PUCT_Q20
constexpr uint32_t kQ20u = 1u << 20;

inline std::string to_hex(const uint8_t* p, size_t n) {
  static const char k[] = "0123456789abcdef";
  std::string s(n * 2, '0');
  for (size_t i = 0; i < n; ++i) {
    s[2 * i] = k[(p[i] >> 4) & 0xF];
    s[2 * i + 1] = k[p[i] & 0xF];
  }
  return s;
}

inline std::string to_hex(const std::array<uint8_t, 32>& a) { return to_hex(a.data(), a.size()); }
inline std::string to_hex(const std::vector<uint8_t>& v) { return to_hex(v.data(), v.size()); }

inline bool from_hex(const std::string& hex, std::array<uint8_t, 32>& out) {
  if (hex.size() != 64) return false;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (size_t i = 0; i < 32; ++i) {
    const int hi = nib(hex[2 * i]);
    const int lo = nib(hex[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = uint8_t((hi << 4) | lo);
  }
  return true;
}

inline std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o.push_back(char(c));
        }
        break;
    }
  }
  return o;
}

inline std::array<uint8_t, 32> sha256_bytes(const uint8_t* p, size_t n) {
  const auto v = Sha256::hash(p, n);
  std::array<uint8_t, 32> a{};
  if (v.size() >= 32) std::memcpy(a.data(), v.data(), 32);
  return a;
}

inline std::array<uint8_t, 32> sha256_bytes(const std::string& s) {
  return sha256_bytes(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}

inline void append_u32_le(std::vector<uint8_t>& b, uint32_t v) {
  b.push_back(uint8_t(v));
  b.push_back(uint8_t(v >> 8));
  b.push_back(uint8_t(v >> 16));
  b.push_back(uint8_t(v >> 24));
}

inline void append_i32_le(std::vector<uint8_t>& b, int32_t v) {
  append_u32_le(b, uint32_t(v));
}

}  // namespace rl
}  // namespace pchie
