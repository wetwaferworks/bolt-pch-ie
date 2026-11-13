// Scratch SHA-256 side chain for reasoning-layer records.
// h_0 = 32 zero bytes; h_n = SHA256(h_{n-1} || payload).
// This is not IHashChain and must never be passed to official runners.
#pragma once

#include "pchie/rl/RlCommon.h"

#include <fstream>

namespace pchie {
namespace rl {

class RlSideChain final {
 public:
  void reset() {
    head_.fill(0);
    length_ = 0;
  }

  void append_payload(const uint8_t* p, size_t n) {
    std::vector<uint8_t> cat;
    cat.reserve(32 + n);
    cat.insert(cat.end(), head_.begin(), head_.end());
    if (p && n) cat.insert(cat.end(), p, p + n);
    head_ = sha256_bytes(cat.data(), cat.size());
    ++length_;
  }

  void append_string(const std::string& s) {
    append_payload(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }

  const std::array<uint8_t, 32>& head() const { return head_; }
  std::string head_hex() const { return to_hex(head_); }
  size_t length() const { return length_; }

  // Append-only JSONL mirror of payloads, also under scratch/rl/ only.
  bool append_jsonl_and_chain(const std::string& path, const std::string& json_line, std::string& err) {
    err.clear();
    const bool ok_path = path.find("/scratch/rl/") != std::string::npos ||
                         path.find("\\scratch\\rl\\") != std::string::npos;
    if (!ok_path) {
      err = "side chain log must live under scratch/rl/";
      return false;
    }
    std::ofstream out(path, std::ios::app | std::ios::binary);
    if (!out) {
      err = "cannot open side chain log: " + path;
      return false;
    }
    out << json_line << '\n';
    if (!out) {
      err = "write failed";
      return false;
    }
    append_string(json_line);
    return true;
  }

 private:
  std::array<uint8_t, 32> head_{};
  size_t length_ = 0;
};

}  // namespace rl
}  // namespace pchie
