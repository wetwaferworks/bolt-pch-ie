/* Peachy SHA-256 (FIPS 180-4), in-house. MIT license. */
#ifndef PJ_SHA256_H
#define PJ_SHA256_H
#include <stddef.h>
#include <stdint.h>
typedef struct pj_sha256 {
  uint32_t h[8];
  uint64_t bits;
  uint8_t buf[64];
  size_t n;
} pj_sha256;
void pj_sha256_init(pj_sha256* c);
void pj_sha256_update(pj_sha256* c, const void* data, size_t len);
void pj_sha256_final(pj_sha256* c, uint8_t out[32]);
void pj_sha256_once(const void* data, size_t len, uint8_t out[32]);
#endif
