// Small SHA-256 helpers shared by the research layers.
#pragma once

#include "pchie/Sha256.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace research {

inline std::string to_hex(const std::vector<uint8_t>& bytes) {
  static const char* kHex = "0123456789abcdef";
  std::string s;
  s.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    s.push_back(kHex[b >> 4]);
    s.push_back(kHex[b & 15]);
  }
  return s;
}

inline std::string sha256_hex(const std::string& text) {
  return to_hex(Sha256::hash(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

inline std::string sha256_hex(const std::vector<uint8_t>& bytes) {
  return to_hex(Sha256::hash(bytes.data(), bytes.size()));
}

}  // namespace research
}  // namespace pchie
