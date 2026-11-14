#include "pchie/OsRandom.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#if defined(_MSC_VER)
#pragma comment(lib, "bcrypt.lib")
#endif
#else
#include <cerrno>
#include <cstdio>
#if defined(__linux__)
#include <sys/random.h>
#endif
#endif

namespace pchie {

bool os_random_bytes(uint8_t* out, size_t n) {
  if (n == 0) return true;
  if (!out) return false;
#if defined(_WIN32)
  while (n > 0) {
    const ULONG chunk = n > 0x10000000u ? 0x10000000u : ULONG(n);
    if (BCryptGenRandom(nullptr, out, chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) return false;
    out += chunk;
    n -= chunk;
  }
  return true;
#else
#if defined(__linux__)
  size_t got = 0;
  while (got < n) {
    const ssize_t r = getrandom(out + got, n - got, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      break;  // try /dev/urandom below
    }
    got += size_t(r);
  }
  if (got == n) return true;
#endif
  std::FILE* f = std::fopen("/dev/urandom", "rb");
  if (!f) return false;
  const size_t r = std::fread(out, 1, n, f);
  std::fclose(f);
  return r == n;
#endif
}

}  // namespace pchie
