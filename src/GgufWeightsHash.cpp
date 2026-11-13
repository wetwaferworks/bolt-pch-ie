// Bolt PCHIE - SHA-256 streamed over GGUF file bytes (never slurped).
#include "pchie/GgufWeightsHash.h"
#include "pchie/Sha256.h"

#include <fstream>
#include <vector>

namespace pchie {

std::vector<uint8_t> GgufWeightsHash::sha256_file(const std::string& path, std::string& err) {
  err.clear();
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    err = "cannot open " + path;
    return {};
  }
  Sha256 h;
  constexpr size_t kChunk = 1u << 20;  // 1 MiB
  std::vector<uint8_t> buf(kChunk);
  while (in) {
    in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size()));
    const auto n = in.gcount();
    if (n > 0) h.update(buf.data(), size_t(n));
  }
  if (in.bad()) {
    err = "read error: " + path;
    return {};
  }
  std::vector<uint8_t> out(Sha256::kDigestBytes);
  h.final(out.data());
  return out;
}

std::string GgufWeightsHash::to_hex(const std::vector<uint8_t>& digest) {
  static const char* k = "0123456789abcdef";
  std::string s;
  s.resize(digest.size() * 2);
  for (size_t i = 0; i < digest.size(); ++i) {
    s[2 * i] = k[digest[i] >> 4];
    s[2 * i + 1] = k[digest[i] & 15];
  }
  return s;
}

}  // namespace pchie