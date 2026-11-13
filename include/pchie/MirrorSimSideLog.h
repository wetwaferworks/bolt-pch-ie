// Bolt PCHIE Mirror-Sim gate - deterministic side log of Mirror-Sim saved forks.
// Not an official hash chain. Sim and refusal records stay here so B1/B4-B7
// official hashes do not move.
#pragma once

#include "pchie/MazeGrid.h"
#include "pchie/Sha256.h"
#include "pchie/pchie_tree.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

struct MirrorSimFork {
  std::vector<uint8_t> pre_sha;
  uint64_t pre_fnv = 0;
  int32_t action = 0;
  std::vector<uint8_t> pred_sha;
  uint64_t pred_fnv = 0;
  bool allowed = false;
  std::string rule_fired;
  std::vector<uint8_t> real_post_sha;
  uint64_t real_post_fnv = 0;
};

class MirrorSimSideLog final {
 public:
  static uint64_t fnv_bytes(const uint8_t* p, size_t n) {
    uint64_t h = FNV_OFFSET;
    constexpr uint64_t P = 1099511628211ULL;
    for (size_t i = 0; i < n; ++i) {
      h ^= uint64_t(p[i]);
      h *= P;
    }
    return h;
  }

  static void hash_world(const MazeGrid& g, std::vector<uint8_t>& sha, uint64_t& fnv) {
    const auto bytes = g.canonical_bytes();
    if (bytes.empty()) {
      sha.clear();
      fnv = 0;
      return;
    }
    sha = Sha256::hash(bytes.data(), bytes.size());
    fnv = fnv_bytes(bytes.data(), bytes.size());
  }

  void clear() { forks_.clear(); }

  void append(MirrorSimFork f) { forks_.push_back(std::move(f)); }

  const std::vector<MirrorSimFork>& forks() const { return forks_; }

  int allowed_count() const {
    int n = 0;
    for (const auto& f : forks_) {
      if (f.allowed) ++n;
    }
    return n;
  }

  int refused_count() const {
    int n = 0;
    for (const auto& f : forks_) {
      if (!f.allowed) ++n;
    }
    return n;
  }

  std::vector<uint8_t> canonical_bytes() const {
    std::vector<uint8_t> b;
    auto put_u64 = [&](uint64_t v) {
      for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    auto put_u32 = [&](uint32_t v) {
      for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
    };
    auto put_vec = [&](const std::vector<uint8_t>& v) {
      put_u32(uint32_t(v.size()));
      b.insert(b.end(), v.begin(), v.end());
    };
    put_u32(uint32_t(forks_.size()));
    for (const auto& f : forks_) {
      put_vec(f.pre_sha);
      put_u64(f.pre_fnv);
      put_u32(uint32_t(f.action));
      put_vec(f.pred_sha);
      put_u64(f.pred_fnv);
      b.push_back(f.allowed ? uint8_t(1) : uint8_t(0));
      put_u32(uint32_t(f.rule_fired.size()));
      b.insert(b.end(), f.rule_fired.begin(), f.rule_fired.end());
      put_vec(f.real_post_sha);
      put_u64(f.real_post_fnv);
    }
    return b;
  }

  std::vector<uint8_t> sha256() const {
    const auto b = canonical_bytes();
    return Sha256::hash(b.data(), b.size());
  }

  uint64_t fnv() const {
    const auto b = canonical_bytes();
    return fnv_bytes(b.data(), b.size());
  }

 private:
  std::vector<MirrorSimFork> forks_;
};

}  // namespace pchie
