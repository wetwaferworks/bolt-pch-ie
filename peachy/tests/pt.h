/* Small shared helpers for the Peachy tests. MIT license. */
#ifndef PEACHY_PT_H
#define PEACHY_PT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#include <process.h>
#define pt_mkdir(p) _mkdir(p)
#define pt_getpid() _getpid()
#else
#include <sys/stat.h>
#include <unistd.h>
#define pt_mkdir(p) mkdir(p, 0700)
#define pt_getpid() getpid()
#endif

#include "peachy/peachy.h"
#include "pj_sha256.h"

static int pt_pass, pt_fail;

#define CHECK(cond, what)                                                                                      \
  do {                                                                                                         \
    if (cond) {                                                                                                \
      ++pt_pass;                                                                                               \
    } else {                                                                                                   \
      ++pt_fail;                                                                                               \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, what);                                        \
    }                                                                                                          \
  } while (0)

static int pt_done(const char* name) {
  printf("%s: %d passed, %d failed\n", name, pt_pass, pt_fail);
  return pt_fail ? 1 : 0;
}

static char pt_base[512];
static int pt_dir_counter;

/* A fresh directory under the test scratch root. */
static void pt_newdir(char* out, size_t cap, const char* tag) {
  snprintf(out, cap, "%s/%s-%d-%d", pt_base, tag, (int)pt_getpid(), ++pt_dir_counter);
  pt_mkdir(out);
}

static void pt_init(int argc, char** argv) {
  snprintf(pt_base, sizeof(pt_base), "%s", argc > 1 ? argv[1] : ".");
  pt_mkdir(pt_base);
}

static int pt_write_file(const char* path, const char* data, size_t n) {
  FILE* f = fopen(path, "wb");
  int ok;
  if (!f) return -1;
  ok = fwrite(data, 1, n, f) == n;
  return (fclose(f) == 0 && ok) ? 0 : -1;
}

/* Whole file, NUL-terminated; caller frees. */
static char* pt_read_file(const char* path, size_t* n_out) {
  FILE* f = fopen(path, "rb");
  char* b;
  long n;
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  n = ftell(f);
  fseek(f, 0, SEEK_SET);
  b = (char*)malloc((size_t)n + 1);
  if (!b) {
    fclose(f);
    return NULL;
  }
  if (fread(b, 1, (size_t)n, f) != (size_t)n) {
    free(b);
    fclose(f);
    return NULL;
  }
  b[n] = 0;
  fclose(f);
  if (n_out) *n_out = (size_t)n;
  return b;
}

static void pt_hex(const uint8_t* b, size_t n, char* out) {
  static const char* h = "0123456789abcdef";
  size_t i;
  for (i = 0; i < n; ++i) {
    out[2 * i] = h[b[i] >> 4];
    out[2 * i + 1] = h[b[i] & 15];
  }
  out[2 * n] = 0;
}

static void pt_token_sha_hex(const char* token, char out[65]) {
  uint8_t h[32];
  pj_sha256_once(token, strlen(token), h);
  pt_hex(h, 32, out);
}

/* ---- a scripted MCP client in pump mode ---- */
#define PT_MAX_OUT 512
typedef struct pt_client {
  char* msgs[PT_MAX_OUT];
  size_t lens[PT_MAX_OUT];
  int n;
  int fail_writes; /* make the write callback fail */
} pt_client;

static int32_t PEACHY_CALL pt_write(void* ctx, const char* msg, size_t len) {
  pt_client* c = (pt_client*)ctx;
  if (c->fail_writes) return -1;
  if (c->n >= PT_MAX_OUT) {
    free(c->msgs[0]);
    memmove(c->msgs, c->msgs + 1, sizeof(c->msgs[0]) * (PT_MAX_OUT - 1));
    memmove(c->lens, c->lens + 1, sizeof(c->lens[0]) * (PT_MAX_OUT - 1));
    --c->n;
  }
  c->msgs[c->n] = (char*)malloc(len + 1);
  memcpy(c->msgs[c->n], msg, len);
  c->msgs[c->n][len] = 0;
  c->lens[c->n] = len;
  ++c->n;
  return 0;
}

static void pt_client_clear(pt_client* c) {
  int i;
  for (i = 0; i < c->n; ++i) free(c->msgs[i]);
  c->n = 0;
}

static const char* pt_last(pt_client* c) { return c->n ? c->msgs[c->n - 1] : ""; }

/* A test clock the test moves by hand. */
typedef struct pt_clock {
  uint64_t t;
  uint64_t step; /* added on every read */
} pt_clock;
static uint64_t PEACHY_CALL pt_clock_fn(void* ctx) {
  pt_clock* c = (pt_clock*)ctx;
  c->t += c->step;
  return c->t;
}

#endif
