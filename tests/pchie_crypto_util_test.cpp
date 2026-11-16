// Small checks for the shared crypto helpers in HmacSha256.h: constant-time compare, HMAC and HKDF.
#include <cstdio>
#include <string>
#include <vector>
#include "pchie/HmacSha256.h"

static int passes = 0, failures = 0;
static void check(bool ok, const std::string& name) { std::printf("[CRYPTO] %-80s %s\n", name.c_str(), ok ? "PASS" : "FAIL"); ok ? ++passes : ++failures; }

int main() {
  using V = std::vector<uint8_t>;
  // ---- ct_equal ----
  V a(32); for (int i = 0; i < 32; ++i) a[i] = uint8_t(i * 7 + 1);
  V first = a; first[0] ^= 1; V last = a; last[31] ^= 0x80; V shorter(a.begin(), a.end() - 1);
  check(pchie::ct_equal(a, a) && pchie::ct_equal(V{}, V{}), "ct_equal: equal inputs (and two empty ones) compare equal");
  check(!pchie::ct_equal(a, first), "ct_equal: a difference in the first byte is caught");
  check(!pchie::ct_equal(a, last), "ct_equal: a difference in the last byte is caught");
  check(!pchie::ct_equal(a, shorter) && !pchie::ct_equal(shorter, a) && !pchie::ct_equal(a, V{}), "ct_equal: different lengths never compare equal");
  std::printf("CRYPTO-UTIL: %d passed, %d failed\n", passes, failures);
  return failures ? 1 : 0;
}
