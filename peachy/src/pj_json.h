/* Peachy bounded JSON reader. MIT license.
 * Strict RFC 8259 grammar, explicit stack (no recursion), every limit checked while reading, output written
 * into a caller-given region. Strings must be valid UTF-8 with no NUL and no lone surrogates. Duplicate keys are
 * refused. Numbers keep an exact int64 when they are plain integers; other numbers are converted without the C
 * locale (not correctly rounded past about 15 significant digits, which no PMCP field needs). */
#ifndef PJ_JSON_H
#define PJ_JSON_H

#include <stddef.h>
#include <stdint.h>

enum { PJ_NULL = 0, PJ_BOOL, PJ_NUM, PJ_STR, PJ_ARR, PJ_OBJ, PJ_KEY };

typedef struct pj_tok {
  uint8_t type;
  uint8_t is_int; /* PJ_NUM: exact integer in i */
  uint8_t b;      /* PJ_BOOL */
  uint8_t pad;
  uint32_t len;   /* PJ_STR/PJ_KEY: bytes; PJ_ARR/PJ_OBJ: children */
  int32_t first;  /* PJ_ARR/PJ_OBJ: first child; PJ_KEY: its value; -1 none */
  int32_t next;   /* next sibling; -1 none */
  const char* s;  /* PJ_STR/PJ_KEY: decoded, NUL-terminated */
  int64_t i;
  double d;
} pj_tok;

typedef struct pj_limits {
  uint32_t max_bytes, max_depth, max_keys, max_items, max_string, max_values;
} pj_limits;

typedef struct pj_doc {
  pj_tok* t;
  uint32_t n;
  char err[48]; /* fixed text only */
} pj_doc;

/* Bytes of scratch space a parse needs for these limits. */
size_t pj_json_scratch_bytes(const pj_limits* lim);
/* 0 on success. On failure doc->err says why (fixed text). scratch must hold pj_json_scratch_bytes(lim). */
int pj_json_parse(const char* text, size_t len, const pj_limits* lim, void* scratch, size_t scratch_cap, pj_doc* doc);

int32_t pj_get(const pj_doc* d, int32_t obj, const char* key); /* value index or -1 */
int pj_is(const pj_doc* d, int32_t i, int type);
int pj_str_eq(const pj_doc* d, int32_t i, const char* s);
int pj_get_int(const pj_doc* d, int32_t i, int64_t lo, int64_t hi, int64_t* out); /* 0 on success */

#endif
