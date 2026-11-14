// Bolt PCHIE - secret bytes from the operating system's random generator (src/OsRandom.cpp).
// Windows: BCryptGenRandom with the system-preferred RNG. Elsewhere: getrandom(), then /dev/urandom.
// Returns false when no OS generator answers. Callers that need a secret must then refuse; they never fall back
// to anything derived from public inputs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pchie {

bool os_random_bytes(uint8_t* out, size_t n);

inline bool os_random_bytes(std::vector<uint8_t>& out, size_t n) {
  out.assign(n, 0);
  if (os_random_bytes(out.data(), n)) return true;
  out.clear();
  return false;
}

}  // namespace pchie
