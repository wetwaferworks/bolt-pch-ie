/* Peachy internals: arena, bounded writer and helpers. MIT license. */
#include "pj_base.h"

void* pj_arena_alloc(pj_arena* a, size_t n) {
  size_t off = (a->used + 15u) & ~(size_t)15u;
  if (off < a->used || n > a->cap || off > a->cap - n) return NULL;
  a->used = off + n;
  memset(a->base + off, 0, n);
  return a->base + off;
}

void pj_w_init(pj_w* w, char* buf, size_t cap) {
  w->p = buf;
  w->len = 0;
  w->cap = cap;
  w->overflow = 0;
  if (cap) buf[0] = 0;
}

void pj_w_raw(pj_w* w, const char* s, size_t n) {
  if (w->overflow) return;
  /* the text always stays NUL-terminated, so a finished writer is also a C string */
  if (w->cap == 0 || n > w->cap - 1 || w->len > w->cap - 1 - n) {
    w->overflow = 1;
    return;
  }
  memcpy(w->p + w->len, s, n);
  w->len += n;
  w->p[w->len] = 0;
}

void pj_w_cstr(pj_w* w, const char* s) { pj_w_raw(w, s, strlen(s)); }

void pj_w_str(pj_w* w, const char* s, size_t n) {
  static const char hex[] = "0123456789abcdef";
  size_t i;
  pj_w_raw(w, "\"", 1);
  for (i = 0; i < n && !w->overflow; ++i) {
    unsigned char c = (unsigned char)s[i];
    if (c == '"') pj_w_raw(w, "\\\"", 2);
    else if (c == '\\') pj_w_raw(w, "\\\\", 2);
    else if (c == '\n') pj_w_raw(w, "\\n", 2);
    else if (c == '\r') pj_w_raw(w, "\\r", 2);
    else if (c == '\t') pj_w_raw(w, "\\t", 2);
    else if (c < 0x20 || c == 0x7f) {
      char e[6] = {'\\', 'u', '0', '0', hex[c >> 4], hex[c & 15]};
      pj_w_raw(w, e, 6);
    } else {
      char ch = (char)c;
      pj_w_raw(w, &ch, 1);
    }
  }
  pj_w_raw(w, "\"", 1);
}

void pj_w_cjstr(pj_w* w, const char* s) { pj_w_str(w, s, strlen(s)); }

void pj_w_u64(pj_w* w, uint64_t v) {
  char b[24];
  int n = 0;
  char o[24];
  int k;
  do {
    b[n++] = (char)('0' + (v % 10u));
    v /= 10u;
  } while (v && n < 24);
  for (k = 0; k < n; ++k) o[k] = b[n - 1 - k];
  pj_w_raw(w, o, (size_t)n);
}

void pj_w_i64(pj_w* w, int64_t v) {
  if (v < 0) {
    pj_w_raw(w, "-", 1);
    pj_w_u64(w, (uint64_t)(-(v + 1)) + 1u);
  } else {
    pj_w_u64(w, (uint64_t)v);
  }
}

void pj_w_milli(pj_w* w, int64_t milli) {
  uint64_t m;
  char f[4];
  if (milli < 0) {
    pj_w_raw(w, "-", 1);
    m = (uint64_t)(-(milli + 1)) + 1u;
  } else {
    m = (uint64_t)milli;
  }
  pj_w_u64(w, m / 1000u);
  f[0] = '.';
  f[1] = (char)('0' + (m / 100u) % 10u);
  f[2] = (char)('0' + (m / 10u) % 10u);
  f[3] = (char)('0' + m % 10u);
  pj_w_raw(w, f, 4);
}

void pj_w_hex(pj_w* w, const uint8_t* b, size_t n) {
  static const char hex[] = "0123456789abcdef";
  size_t i;
  for (i = 0; i < n; ++i) {
    char two[2] = {hex[b[i] >> 4], hex[b[i] & 15]};
    pj_w_raw(w, two, 2);
  }
}

void pj_w_bool(pj_w* w, int v) { pj_w_cstr(w, v ? "true" : "false"); }

int pj_ct_equal(const uint8_t* a, const uint8_t* b, size_t n) {
  unsigned d = 0;
  size_t i;
  for (i = 0; i < n; ++i) d |= (unsigned)(a[i] ^ b[i]);
  return d == 0;
}

void pj_wipe(void* p, size_t n) {
  volatile unsigned char* q = (volatile unsigned char*)p;
  while (n--) *q++ = 0;
}

size_t pj_strnlen(const char* s, size_t max) {
  size_t n = 0;
  if (!s) return 0;
  while (n < max && s[n]) ++n;
  return n;
}

int pj_name_ok(const char* s, size_t max) {
  size_t n, i;
  if (!s) return 0;
  n = pj_strnlen(s, max + 1);
  if (n == 0 || n > max) return 0;
  for (i = 0; i < n; ++i) {
    char c = s[i];
    if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-')) return 0;
  }
  return 1;
}

void pj_copy_cstr(char* dst, size_t cap, const char* src) {
  size_t n;
  if (!cap) return;
  n = src ? pj_strnlen(src, cap - 1) : 0;
  if (n) memcpy(dst, src, n);
  dst[n] = 0;
}
