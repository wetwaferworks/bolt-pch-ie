// Bolt PCHIE - provisional manifest fields. Not a locked golden.
#pragma once

#include <cstdint>
#include <string>

namespace pchie {

struct ModelManifest {
  std::string path;
  std::string architecture;
  std::string name;
  std::string weights_sha256_hex;
  uint64_t directory_digest = 0;
  std::string prompt;
  std::string logits_sha256_hex;
  std::string hidden_sha256_hex;
  uint64_t logits_fnv1a64 = 0;
};

}  // namespace pchie