// Bolt PCHIE - SHA-256 streamed over GGUF file bytes (not the directory digest).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class GgufWeightsHash final {
 public:
  // Stream the file; never slurp the whole image. Empty on I/O error (err set).
  static std::vector<uint8_t> sha256_file(const std::string& path, std::string& err);
  static std::string to_hex(const std::vector<uint8_t>& digest);
};

}  // namespace pchie