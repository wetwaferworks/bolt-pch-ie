/* Peachy allowlist: who may connect, over what, with which grants. MIT license.
 * {"version":1,"clients":[{"name":"grok","transports":["stdio"],"token_sha256":"<64 hex>",
 *   "tools":["observe","plan","propose","proposal_status","audit_head"],"rate_per_sec":8,"burst":16,
 *   "max_mode":"record"}]}
 * Only the SHA-256 of a token is stored, never the token. Unknown fields are refused. */
#include <stdlib.h>
#include <string.h>

#include "pj_internal.h"

#define PJ_ALLOW_MAX_BYTES 65536

const char* const pj_tool_names[PJ_T_COUNT] = {"peachy_info", "status", "audit_head", "observe",
                                               "plan", "propose", "proposal_status", "hold"};

int pj_tool_index(const char* name, size_t len) {
  int i;
  for (i = 0; i < PJ_T_COUNT; ++i)
    if (strlen(pj_tool_names[i]) == len && memcmp(pj_tool_names[i], name, len) == 0) return i;
  return -1;
}

static int why_set(char* why, size_t cap, const char* s) {
  if (why && cap) pj_copy_cstr(why, cap, s);
  return -1;
}

static int unhex32(const char* s, size_t n, uint8_t* out) {
  size_t i;
  if (n != 64) return -1;
  for (i = 0; i < 32; ++i) {
    int v = 0, k;
    for (k = 0; k < 2; ++k) {
      char c = s[2 * i + (size_t)k];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else return -1;
    }
    out[i] = (uint8_t)v;
  }
  return 0;
}

int pj_allow_parse(const char* text, size_t len, pj_allow* out, char* why, size_t why_cap) {
  pj_limits jl = {PJ_ALLOW_MAX_BYTES, 8, 16, 64, 256, 4096};
  size_t need = pj_json_scratch_bytes(&jl);
  void* scratch;
  pj_doc d;
  int32_t root, v, cl, j;
  int rc = -1;
  memset(out, 0, sizeof(*out));
  if (len > PJ_ALLOW_MAX_BYTES) return why_set(why, why_cap, "allowlist larger than 64 KiB");
  scratch = malloc(need);
  if (!scratch) return why_set(why, why_cap, "no memory");
  if (pj_json_parse(text, len, &jl, scratch, need, &d) != 0) {
    why_set(why, why_cap, "allowlist is not valid JSON within limits");
    goto done;
  }
  root = 0;
  if (!pj_is(&d, root, PJ_OBJ)) {
    why_set(why, why_cap, "allowlist must be an object");
    goto done;
  }
  for (j = d.t[root].first; j >= 0; j = d.t[j].next)
    if (strcmp(d.t[j].s, "version") && strcmp(d.t[j].s, "clients")) {
      why_set(why, why_cap, "allowlist has an unknown field");
      goto done;
    }
  {
    int64_t ver;
    if (pj_get_int(&d, pj_get(&d, root, "version"), 1, 1, &ver)) {
      why_set(why, why_cap, "allowlist version must be 1");
      goto done;
    }
  }
  cl = pj_get(&d, root, "clients");
  if (!pj_is(&d, cl, PJ_ARR) || d.t[cl].len > PJ_MAX_CLIENTS) {
    why_set(why, why_cap, "clients must be an array of at most 32");
    goto done;
  }
  for (v = d.t[cl].first; v >= 0; v = d.t[v].next) {
    pj_client* c = &out->c[out->n];
    int32_t f, k;
    int64_t num;
    uint32_t q;
    if (!pj_is(&d, v, PJ_OBJ)) {
      why_set(why, why_cap, "each client must be an object");
      goto done;
    }
    for (k = d.t[v].first; k >= 0; k = d.t[k].next) {
      const char* n = d.t[k].s;
      if (strcmp(n, "name") && strcmp(n, "transports") && strcmp(n, "token_sha256") && strcmp(n, "tools") &&
          strcmp(n, "rate_per_sec") && strcmp(n, "burst") && strcmp(n, "max_mode")) {
        why_set(why, why_cap, "client has an unknown field");
        goto done;
      }
    }
    f = pj_get(&d, v, "name");
    if (!pj_is(&d, f, PJ_STR) || !pj_name_ok(d.t[f].s, 32)) {
      why_set(why, why_cap, "client name must be [a-z0-9_-]{1,32}");
      goto done;
    }
    for (q = 0; q < out->n; ++q)
      if (!strcmp(out->c[q].name, d.t[f].s)) {
        why_set(why, why_cap, "client name listed twice");
        goto done;
      }
    pj_copy_cstr(c->name, sizeof(c->name), d.t[f].s);
    f = pj_get(&d, v, "transports");
    if (!pj_is(&d, f, PJ_ARR)) {
      why_set(why, why_cap, "transports must be an array");
      goto done;
    }
    for (k = d.t[f].first; k >= 0; k = d.t[k].next) {
      if (pj_str_eq(&d, k, "stdio")) c->transports |= PJ_TR_STDIO;
      else if (pj_str_eq(&d, k, "in_process")) c->transports |= PJ_TR_IN_PROCESS;
      else {
        why_set(why, why_cap, "unknown transport (stdio, in_process)");
        goto done;
      }
    }
    f = pj_get(&d, v, "token_sha256");
    if (f >= 0) {
      if (!pj_is(&d, f, PJ_STR) || unhex32(d.t[f].s, d.t[f].len, c->token_sha)) {
        why_set(why, why_cap, "token_sha256 must be 64 lowercase hex characters");
        goto done;
      }
      c->has_token = 1;
    }
    f = pj_get(&d, v, "tools");
    if (f >= 0) {
      if (!pj_is(&d, f, PJ_ARR)) {
        why_set(why, why_cap, "tools must be an array");
        goto done;
      }
      for (k = d.t[f].first; k >= 0; k = d.t[k].next) {
        int t = pj_is(&d, k, PJ_STR) ? pj_tool_index(d.t[k].s, d.t[k].len) : -1;
        if (t < 0 || !(PJ_BIT(t) & PJ_GRANTABLE_TOOLS)) {
          why_set(why, why_cap, "tools lists something that cannot be granted");
          goto done;
        }
        c->tools |= PJ_BIT(t);
      }
    }
    c->rate_per_sec = 8;
    c->burst = 16;
    f = pj_get(&d, v, "rate_per_sec");
    if (f >= 0) {
      if (pj_get_int(&d, f, 1, 256, &num)) {
        why_set(why, why_cap, "rate_per_sec must be 1..256");
        goto done;
      }
      c->rate_per_sec = (uint32_t)num;
    }
    f = pj_get(&d, v, "burst");
    if (f >= 0) {
      if (pj_get_int(&d, f, 1, 512, &num)) {
        why_set(why, why_cap, "burst must be 1..512");
        goto done;
      }
      c->burst = (uint32_t)num;
    }
    c->max_mode = PEACHY_MODE_RECORD;
    f = pj_get(&d, v, "max_mode");
    if (f >= 0) {
      if (pj_str_eq(&d, f, "record")) c->max_mode = PEACHY_MODE_RECORD;
      else if (pj_str_eq(&d, f, "advisory")) c->max_mode = PEACHY_MODE_ADVISORY;
      else {
        why_set(why, why_cap, "max_mode must be record or advisory in this stage");
        goto done;
      }
    }
    ++out->n;
  }
  rc = 0;
done:
  free(scratch);
  return rc;
}

int pj_allow_load(const char* path, pj_allow* out, char* why, size_t why_cap) {
  FILE* f;
  char* buf;
  size_t n;
  int rc;
  memset(out, 0, sizeof(*out));
  if (!path) return why_set(why, why_cap, "no allowlist path");
  f = pj_fopen(path, "rb");
  if (!f) return why_set(why, why_cap, "cannot open the allowlist");
  buf = (char*)malloc(PJ_ALLOW_MAX_BYTES + 1);
  if (!buf) {
    fclose(f);
    return why_set(why, why_cap, "no memory");
  }
  n = fread(buf, 1, PJ_ALLOW_MAX_BYTES + 1, f);
  fclose(f);
  if (n > PJ_ALLOW_MAX_BYTES) {
    free(buf);
    return why_set(why, why_cap, "allowlist larger than 64 KiB");
  }
  rc = pj_allow_parse(buf, n, out, why, why_cap);
  free(buf);
  return rc;
}

int pj_allow_write(const pj_allow* a, char* buf, size_t cap, size_t* len) {
  pj_w w;
  uint32_t i;
  int t;
  pj_w_init(&w, buf, cap);
  pj_w_cstr(&w, "{\n  \"version\": 1,\n  \"clients\": [");
  for (i = 0; i < a->n; ++i) {
    const pj_client* c = &a->c[i];
    int first = 1;
    pj_w_cstr(&w, i ? ",\n    {" : "\n    {");
    pj_w_cstr(&w, "\"name\": ");
    pj_w_cjstr(&w, c->name);
    pj_w_cstr(&w, ", \"transports\": [");
    if (c->transports & PJ_TR_STDIO) {
      pj_w_cstr(&w, "\"stdio\"");
      first = 0;
    }
    if (c->transports & PJ_TR_IN_PROCESS) pj_w_cstr(&w, first ? "\"in_process\"" : ", \"in_process\"");
    pj_w_cstr(&w, "]");
    if (c->has_token) {
      pj_w_cstr(&w, ", \"token_sha256\": \"");
      pj_w_hex(&w, c->token_sha, 32);
      pj_w_cstr(&w, "\"");
    }
    pj_w_cstr(&w, ", \"tools\": [");
    first = 1;
    for (t = 0; t < PJ_T_COUNT; ++t)
      if (c->tools & PJ_BIT(t)) {
        if (!first) pj_w_cstr(&w, ", ");
        pj_w_cjstr(&w, pj_tool_names[t]);
        first = 0;
      }
    pj_w_cstr(&w, "], \"rate_per_sec\": ");
    pj_w_u64(&w, c->rate_per_sec);
    pj_w_cstr(&w, ", \"burst\": ");
    pj_w_u64(&w, c->burst);
    pj_w_cstr(&w, ", \"max_mode\": ");
    pj_w_cstr(&w, c->max_mode == PEACHY_MODE_ADVISORY ? "\"advisory\"" : "\"record\"");
    pj_w_cstr(&w, "}");
  }
  pj_w_cstr(&w, a->n ? "\n  ]\n}\n" : "]\n}\n");
  if (w.overflow) return -1;
  *len = w.len;
  return 0;
}

int pj_allow_save(const char* path, const pj_allow* a) {
  char* buf = (char*)malloc(PJ_ALLOW_MAX_BYTES);
  char tmp[600];
  size_t n = 0;
  FILE* f;
  int ok;
  if (!buf) return -1;
  if (pj_allow_write(a, buf, PJ_ALLOW_MAX_BYTES, &n) || snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) {
    free(buf);
    return -1;
  }
  f = pj_fopen(tmp, "wb");
  if (!f) {
    free(buf);
    return -1;
  }
  ok = fwrite(buf, 1, n, f) == n;
  ok = (fclose(f) == 0) && ok;
  free(buf);
  if (!ok) return -1;
  return pj_replace_file(tmp, path);
}
