// Bolt PCHIE LithoKraft testbeds (2026-11-10) - LithoKraft-namespace maze area hash.
// Concrete final. Pure, deterministic, no mutex, no atomic, no thread.
// Hashes the block ids the engine reports for a PCHIE maze (read back over the LithoLattice MCP
// world_get_block, in LkMazeVoxelMap::verify order: row y, then column x, then layer 0..2) together with
// the maze size and voxel origin, under the domain tag "LithoKraft.maze_area.v1".
// NAMESPACE: this is a LithoKraft/LithoLattice world record. It is never appended to a PCHIE chain,
// never compared as a PCHIE.run_hash, and is unrelated to the engine's T4 diamond sim-area golden.
// Provisional and unlocked.
#pragma once

#include "pchie/GgufWeightsHash.h"
#include "pchie/Sha256.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class LkMazeAreaHash final {
 public:
  static constexpr const char* kDomain = "LithoKraft.maze_area.v1";
  static constexpr int kLayers = 3;

  // Canonical bytes: domain line, "w h ox fy oz" line, then ids as little-endian u16.
  // Returns an empty vector (fail closed) when ids.size() != w * h * kLayers.
  static std::vector<uint8_t> canonical(int w, int h, int ox, int fy, int oz, const std::vector<uint16_t>& ids) {
    std::vector<uint8_t> out;
    if (w <= 0 || h <= 0 || ids.size() != size_t(w) * size_t(h) * size_t(kLayers)) return out;
    const std::string head = std::string(kDomain) + "\n" + std::to_string(w) + " " + std::to_string(h) + " " +
                             std::to_string(ox) + " " + std::to_string(fy) + " " + std::to_string(oz) + "\n";
    out.assign(head.begin(), head.end());
    out.reserve(out.size() + ids.size() * 2);
    for (uint16_t v : ids) {
      out.push_back(uint8_t(v & 0xFFu));
      out.push_back(uint8_t(v >> 8));
    }
    return out;
  }

  // Lower-case hex SHA-256 of canonical(); empty string when the input is malformed.
  static std::string sha256_hex(int w, int h, int ox, int fy, int oz, const std::vector<uint16_t>& ids) {
    const std::vector<uint8_t> b = canonical(w, h, ox, fy, oz, ids);
    if (b.empty()) return std::string();
    return GgufWeightsHash::to_hex(Sha256::hash(b.data(), b.size()));
  }
};

}  // namespace pchie
