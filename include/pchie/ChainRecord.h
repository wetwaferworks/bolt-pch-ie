// Bolt PCHIE - canonical hash-chain record (one class/struct per file).
// Wire format (little-endian): kind:u32 | payload_len:u32 | payload bytes.
// Payload schemas per kind are filled by later phases; the hash chain treats payload as opaque bytes.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace pchie {

enum class ChainRecordKind : uint32_t {
  Genesis = 0,
  Model = 1,
  Config = 2,
  Prompt = 3,
  Expansion = 4,
  Selection = 5,
  LeafValue = 6,
  Backup = 7,
  Commit = 8,
  Handshake = 9,
  GateVerdict = 10,
  BudgetExhausted = 11,
  Custom = 255
};

struct ChainRecord {
  ChainRecordKind kind = ChainRecordKind::Custom;
  std::vector<uint8_t> payload;

  // Canonical bytes fed to H(h_prev || record). Deterministic across platforms.
  std::vector<uint8_t> canonical_bytes() const {
    std::vector<uint8_t> out(8 + payload.size());
    const uint32_t k = static_cast<uint32_t>(kind);
    const uint32_t n = static_cast<uint32_t>(payload.size());
    out[0] = uint8_t(k); out[1] = uint8_t(k >> 8); out[2] = uint8_t(k >> 16); out[3] = uint8_t(k >> 24);
    out[4] = uint8_t(n); out[5] = uint8_t(n >> 8); out[6] = uint8_t(n >> 16); out[7] = uint8_t(n >> 24);
    if (!payload.empty()) std::memcpy(out.data() + 8, payload.data(), payload.size());
    return out;
  }
};

}  // namespace pchie