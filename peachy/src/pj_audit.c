/* Peachy audit log. MIT license. */
#include "pj_audit.h"

#include <stdlib.h>
#include <string.h>

#include "pj_base.h"
#include "pj_platform.h"
#include "pj_sha256.h"

#define PJ_AUDIT_MAX_LINE 8192

int pj_audit_open(pj_audit* a, const char* dir, const char* session_name) {
  char stamp[32];
  int k;
  memset(a, 0, sizeof(*a));
  if (!pj_dir_exists(dir) || !pj_name_ok(session_name, 32)) return -1;
  pj_compact_time(stamp, sizeof(stamp));
  for (k = 0; k < 100; ++k) {
    int n;
    if (k == 0)
      n = snprintf(a->path, sizeof(a->path), "%s/peachy-audit-%s-%s.log", dir, session_name, stamp);
    else
      n = snprintf(a->path, sizeof(a->path), "%s/peachy-audit-%s-%s-%02d.log", dir, session_name, stamp, k);
    if (n <= 0 || (size_t)n >= sizeof(a->path)) return -1;
    a->f = pj_fopen(a->path, "wbx"); /* exclusive: never overwrite */
    if (a->f) return 0;
  }
  return -1;
}

int pj_audit_append(pj_audit* a, uint64_t mono_ns, const char* body, size_t body_len) {
  char line[PJ_AUDIT_MAX_LINE];
  char lt[40];
  pj_w w;
  uint8_t h[32];
  pj_sha256 c;
  if (a->failed || !a->f) return -1;
  if (a->fail_after && a->seq + 1 >= a->fail_after) {
    a->failed = 1;
    return -1;
  }
  pj_local_time(lt, sizeof(lt));
  pj_w_init(&w, line, sizeof(line) - 80);
  pj_w_cstr(&w, "{\"v\":1,\"seq\":");
  pj_w_u64(&w, a->seq + 1);
  pj_w_cstr(&w, ",\"mono_ns\":");
  pj_w_u64(&w, mono_ns);
  pj_w_cstr(&w, ",\"local\":");
  pj_w_cjstr(&w, lt);
  if (body_len) {
    pj_w_raw(&w, ",", 1);
    pj_w_raw(&w, body, body_len);
  }
  pj_w_cstr(&w, ",\"prev\":\"");
  pj_w_hex(&w, a->head, 32);
  pj_w_cstr(&w, "\"}");
  if (w.overflow) {
    a->failed = 1;
    return -1;
  }
  pj_sha256_init(&c);
  pj_sha256_update(&c, a->head, 32);
  pj_sha256_update(&c, line, w.len);
  pj_sha256_final(&c, h);
  w.cap = sizeof(line);
  pj_w_raw(&w, " ", 1);
  pj_w_hex(&w, h, 32);
  pj_w_raw(&w, "\n", 1);
  if (w.overflow || fwrite(line, 1, w.len, a->f) != w.len || fflush(a->f) != 0) {
    a->failed = 1;
    return -1;
  }
  memcpy(a->head, h, 32);
  ++a->seq;
  return 0;
}

void pj_audit_close(pj_audit* a) {
  if (a->f) fclose(a->f);
  a->f = NULL;
}

static int unhex(const char* s, uint8_t* out, size_t n) {
  size_t i;
  for (i = 0; i < n; ++i) {
    int v[2], k;
    for (k = 0; k < 2; ++k) {
      char ch = s[2 * i + (size_t)k];
      if (ch >= '0' && ch <= '9') v[k] = ch - '0';
      else if (ch >= 'a' && ch <= 'f') v[k] = ch - 'a' + 10;
      else return -1;
    }
    out[i] = (uint8_t)((v[0] << 4) | v[1]);
  }
  return 0;
}

static int bad(pj_audit_report* r, uint64_t rec, const char* why) {
  r->first_bad = rec;
  pj_copy_cstr(r->why, sizeof(r->why), why);
  return 1;
}

/* Reads "seq":N and "prev":"hex" from the fixed positions Peachy writes. */
int pj_audit_verify(const char* path, pj_audit_report* r) {
  FILE* f;
  char* line;
  uint8_t prev[32];
  uint64_t rec = 0;
  int rc = 0;
  memset(r, 0, sizeof(*r));
  memset(prev, 0, sizeof(prev));
  f = pj_fopen(path, "rb");
  if (!f) return -1;
  line = (char*)malloc(PJ_AUDIT_MAX_LINE + 2);
  if (!line) {
    fclose(f);
    return -1;
  }
  for (;;) {
    size_t n = 0;
    int ch;
    const char *sp, *ps, *sq;
    uint8_t h[32], got[32], pv[32];
    pj_sha256 c;
    char want_seq[32];
    while ((ch = fgetc(f)) != EOF && ch != '\n') {
      if (n >= PJ_AUDIT_MAX_LINE) {
        rc = bad(r, rec + 1, "line too long");
        goto done;
      }
      line[n++] = (char)ch;
    }
    if (ch == EOF && n == 0) break;
    if (ch == EOF) {
      rc = bad(r, rec + 1, "last line cut short");
      goto done;
    }
    ++rec;
    line[n] = 0;
    if (n < 66 || line[n - 65] != ' ') {
      rc = bad(r, rec, "no record hash");
      goto done;
    }
    sp = line + n - 65;
    if (unhex(sp + 1, got, 32)) {
      rc = bad(r, rec, "bad record hash");
      goto done;
    }
    snprintf(want_seq, sizeof(want_seq), "{\"v\":1,\"seq\":%llu,", (unsigned long long)rec);
    sq = line;
    if (strncmp(sq, want_seq, strlen(want_seq)) != 0) {
      rc = bad(r, rec, "sequence out of order");
      goto done;
    }
    /* "prev":"<64 hex>"} sits right before the hash */
    if ((size_t)(sp - line) < 76) {
      rc = bad(r, rec, "record too short");
      goto done;
    }
    ps = sp - 2 - 64 - 9;
    if (strncmp(ps, ",\"prev\":\"", 9) != 0) {
      rc = bad(r, rec, "no prev field");
      goto done;
    }
    if (unhex(sp - 2 - 64, pv, 32) || sp[-2] != '"' || sp[-1] != '}') {
      rc = bad(r, rec, "bad prev field");
      goto done;
    }
    if (memcmp(pv, prev, 32) != 0) {
      rc = bad(r, rec, "prev does not match the record before");
      goto done;
    }
    pj_sha256_init(&c);
    pj_sha256_update(&c, prev, 32);
    pj_sha256_update(&c, line, (size_t)(sp - line));
    pj_sha256_final(&c, h);
    if (memcmp(h, got, 32) != 0) {
      rc = bad(r, rec, "record hash does not match its bytes");
      goto done;
    }
    memcpy(prev, h, 32);
  }
done:
  r->records = rec;
  memcpy(r->head, prev, 32);
  free(line);
  fclose(f);
  return rc;
}
