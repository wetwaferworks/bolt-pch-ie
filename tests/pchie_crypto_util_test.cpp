// Small checks for the shared crypto helpers in HmacSha256.h: constant-time compare, HMAC and HKDF.
#include <cstdio>
#include <stdexcept>
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
  // ---- HMAC and HKDF: RFC vectors unchanged, and the RFC 5869 length cap ----
  auto hx = [](const V& v) { static const char* d = "0123456789abcdef"; std::string o; for (uint8_t b : v) { o += d[b >> 4]; o += d[b & 15]; } return o; };
  check(hx(pchie::hmac_sha256(V(20, 0x0b), V{'H', 'i', ' ', 'T', 'h', 'e', 'r', 'e'})) == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7", "hmac: RFC 4231 test case 1");
  const std::string tc6s = "Test Using Larger Than Block-Size Key - Hash Key First"; const V tc6(tc6s.begin(), tc6s.end());
  check(hx(pchie::hmac_sha256(V(131, 0xaa), tc6)) ==
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54", "hmac: RFC 4231 test case 6 (key longer than a block)");
  const V ikm(22, 0x0b), salt{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, info{0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9};
  const V prk = pchie::hkdf_extract_sha256(salt, ikm);
  check(hx(pchie::hkdf_expand_sha256(prk, info, 42)) == "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865", "hkdf: RFC 5869 test case 1");
  check(pchie::hkdf_expand_sha256(prk, info, 8160).size() == 8160, "hkdf: the RFC maximum of 255 * 32 = 8160 bytes is allowed");
  bool threw = false; try { pchie::hkdf_expand_sha256(prk, info, 8161); } catch (const std::length_error&) { threw = true; }
  check(threw, "hkdf: 8161 bytes is refused instead of wrapping the block counter");
  { const V full = pchie::hkdf_expand_sha256(prk, info, 8160), part = pchie::hkdf_expand_sha256(prk, info, 100);
    check(V(full.begin(), full.begin() + 100) == part, "hkdf: a shorter output is a prefix of the longer one"); }
  std::printf("CRYPTO-UTIL: %d passed, %d failed\n", passes, failures);
  return failures ? 1 : 0;
}
