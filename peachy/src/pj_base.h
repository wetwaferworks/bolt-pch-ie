/* Peachy internals: arena, bounded writer, atomics and small helpers. MIT license. */
#ifndef PJ_BASE_H
#define PJ_BASE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---- atomics (no locks anywhere) ---- */
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
static __inline uint64_t pj_load_u64(volatile uint64_t* p) {
  return (uint64_t)_InterlockedCompareExchange64((volatile __int64*)p, 0, 0);
}
static __inline void pj_store_u64(volatile uint64_t* p, uint64_t v) { _InterlockedExchange64((volatile __int64*)p, (__int64)v); }
static __inline uint32_t pj_load_u32(volatile uint32_t* p) { return (uint32_t)_InterlockedCompareExchange((volatile long*)p, 0, 0); }
static __inline void pj_store_u32(volatile uint32_t* p, uint32_t v) { _InterlockedExchange((volatile long*)p, (long)v); }
#else
static inline uint64_t pj_load_u64(volatile uint64_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void pj_store_u64(volatile uint64_t* p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline uint32_t pj_load_u32(volatile uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void pj_store_u32(volatile uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#endif

/* ---- arena: one block taken at session open; nothing on the hot path asks the system for memory ---- */
typedef struct pj_arena {
  uint8_t* base;
  size_t cap;
  size_t used;
} pj_arena;

void* pj_arena_alloc(pj_arena* a, size_t n); /* 16-byte aligned, zeroed; NULL when full */

/* ---- bounded writer: never writes past cap, keeps the text NUL-terminated; overflow is sticky ---- */
typedef struct pj_w {
  char* p;
  size_t len;
  size_t cap;
  int overflow;
} pj_w;

void pj_w_init(pj_w* w, char* buf, size_t cap);
void pj_w_raw(pj_w* w, const char* s, size_t n);
void pj_w_cstr(pj_w* w, const char* s);
void pj_w_str(pj_w* w, const char* s, size_t n); /* JSON string literal; input must be valid UTF-8 */
void pj_w_cjstr(pj_w* w, const char* s);         /* JSON string literal of a C string */
void pj_w_i64(pj_w* w, int64_t v);
void pj_w_u64(pj_w* w, uint64_t v);
void pj_w_milli(pj_w* w, int64_t milli);         /* a fixed-point number with 3 decimals, locale free */
void pj_w_hex(pj_w* w, const uint8_t* b, size_t n);
void pj_w_bool(pj_w* w, int v);

/* ---- helpers ---- */
int pj_ct_equal(const uint8_t* a, const uint8_t* b, size_t n); /* constant time */
void pj_wipe(void* p, size_t n);                                /* not optimized away */
int pj_name_ok(const char* s, size_t max);                       /* [a-z0-9_-]{1,max} */
size_t pj_strnlen(const char* s, size_t max);
void pj_copy_cstr(char* dst, size_t cap, const char* src);

#endif
