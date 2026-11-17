/* Peachy fuzzing. MIT license.
 *   peachy_fuzz <json|rpc|propose|allow|audit|all> <runs> <seed> [tmp dir]
 *   peachy_fuzz seeds <dir>   (writes the seed corpus, for libFuzzer)
 * An in-house mutation fuzzer (seed corpus, bit flips, byte edits, splices, a JSON dictionary) that runs on any
 * compiler, plus libFuzzer entry points when built with PEACHY_LIBFUZZER and PEACHY_FUZZ_TARGET_<name>.
 * Each target checks properties, not just "no crash": a tighter parse never accepts what a looser one refuses,
 * every reply Peachy sends is valid JSON, an allowlist survives a write and re-read, a changed audit log never
 * verifies as the original. Build with sanitizers for memory and undefined-behavior checks. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "peachy/peachy.h"
#include "pj_audit.h"
#include "pj_internal.h"
#include "pj_json.h"
#include "pj_sha256.h"

#define MAX_IN 4096

static void die(const char* why) {
  fprintf(stderr, "peachy_fuzz: property failed: %s\n", why);
  abort();
}

/* ---------------------------------------------------------------- json */
static unsigned char g_scratch[1 << 22];

static void check_doc(const pj_doc* d) {
  uint32_t k;
  for (k = 0; k < d->n; ++k) {
    const pj_tok* t = &d->t[k];
    if (t->type > PJ_KEY) die("token type");
    if ((t->first < -1 || t->first >= (int32_t)d->n) || (t->next < -1 || t->next >= (int32_t)d->n)) die("token link");
    if ((t->type == PJ_STR || t->type == PJ_KEY) && (!t->s || strlen(t->s) != t->len)) die("string length");
  }
}

static void target_json(const uint8_t* data, size_t n) {
  pj_limits loose = {MAX_IN * 2, 64, 4096, 65536, 65536, 65536};
  pj_limits tight = {MAX_IN * 2, 4, 4, 4, 8, 32};
  pj_doc a, b;
  int ra = pj_json_parse((const char*)data, n, &loose, g_scratch, pj_json_scratch_bytes(&loose), &a);
  if (ra == 0) check_doc(&a);
  {
    int rb = pj_json_parse((const char*)data, n, &tight, g_scratch, pj_json_scratch_bytes(&tight), &b);
    if (rb == 0 && ra != 0) die("tight limits accepted what loose limits refused");
    if (rb == 0) check_doc(&b);
  }
}

/* ---------------------------------------------------------------- rpc and propose */
static char g_dir[512] = ".";
static char g_allow[600];
static const char* k_token = "fuzz-token-0123456789abcdef012345";
static const peachy_api* A;
static PeachySession* g_s;
static uint64_t g_clock;
static unsigned long g_session_inputs;
static char g_logpath[600];
const char* pj_test_audit_path(PeachySession* s);

static uint64_t PEACHY_CALL fz_clock(void* ctx) {
  (void)ctx;
  g_clock += 1000000ull; /* 1 ms per read: rate buckets refill, deadlines hold */
  return g_clock;
}

static int32_t PEACHY_CALL fz_write(void* ctx, const char* msg, size_t len) {
  static unsigned char sc[1 << 22];
  pj_limits l = {1u << 20, 64, 4096, 65536, 1u << 20, 65536};
  pj_doc d;
  (void)ctx;
  if (memchr(msg, 0, len)) die("reply holds a NUL byte");
  if (pj_json_parse(msg, len, &l, sc, pj_json_scratch_bytes(&l), &d) != 0) die("reply is not valid JSON");
  if (!pj_is(&d, 0, PJ_OBJ) || !pj_str_eq(&d, pj_get(&d, 0, "jsonrpc"), "2.0")) die("reply is not JSON-RPC 2.0");
  return 0;
}

static void write_allow(void) {
  char h[65], doc[1024];
  uint8_t b[32];
  int i, n;
  FILE* f;
  pj_sha256_once(k_token, strlen(k_token), b);
  for (i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", b[i]);
  n = snprintf(doc, sizeof(doc),
               "{\"version\":1,\"clients\":[{\"name\":\"fuzz\",\"transports\":[\"stdio\"],\"token_sha256\":\"%s\",\"tools\":"
               "[\"observe\",\"plan\",\"propose\",\"proposal_status\",\"audit_head\"],\"rate_per_sec\":256,\"burst\":512,"
               "\"max_mode\":\"advisory\"}]}",
               h);
  snprintf(g_allow, sizeof(g_allow), "%s/fuzz-allow.json", g_dir);
  f = fopen(g_allow, "wb");
  if (!f) die("cannot write the fuzz allowlist");
  fwrite(doc, 1, (size_t)n, f);
  fclose(f);
}

static void session_reset(int initialized) {
  peachy_session_config c;
  peachy_limits L;
  char err[128];
  static unsigned counter;
  char name[33];
  if (g_s) {
    A->session_close(g_s);
    remove(g_logpath);
    g_s = NULL;
  }
  if (!A) {
    peachy_get_api(PEACHY_ABI_MAJOR, &A);
    write_allow();
  }
  memset(&L, 0, sizeof(L));
  L.struct_size = sizeof(L);
  L.max_message_bytes = 65536;
  L.bad_message_limit = 1000;
  L.arena_bytes = 4u << 20;
  memset(&c, 0, sizeof(c));
  c.struct_size = sizeof(c);
  c.limits = &L;
  c.clock = fz_clock;
  c.write = fz_write;
  c.allowlist_path = g_allow;
  c.audit_dir = g_dir;
  snprintf(name, sizeof(name), "fuzz-%u", ++counter % 100000u);
  c.session_name = name;
  c.client_name = "fuzz";
  c.client_token = (const uint8_t*)k_token;
  c.client_token_len = strlen(k_token);
  if (A->session_open(&c, &g_s, err, sizeof(err)) != PEACHY_OK) {
    fprintf(stderr, "open: %s\n", err);
    die("fuzz session refused");
  }
  snprintf(g_logpath, sizeof(g_logpath), "%s", pj_test_audit_path(g_s));
  if (initialized) {
    const char* init = "{\"jsonrpc\":\"2.0\",\"id\":0,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}\n";
    A->feed(g_s, init, strlen(init), NULL);
    A->pump(g_s, 1000000000ull, NULL);
  }
  g_session_inputs = 0;
}

static void feed_message(const uint8_t* data, size_t n) {
  peachy_status_info st;
  size_t acc = 0, off = 0;
  int guard = 0;
  while (off < n && guard++ < 64) {
    peachy_status r = A->feed(g_s, (const char*)data + off, n - off, &acc);
    off += acc;
    A->pump(g_s, 1000000000ull, NULL);
    if (r == PEACHY_E_DENIED) break;
  }
  A->feed(g_s, "\n", 1, &acc);
  A->pump(g_s, 1000000000ull, NULL);
  st.struct_size = sizeof(st);
  if (A->status(g_s, &st) != PEACHY_OK) die("status failed");
  if (st.arena_used > st.arena_bytes) die("arena over its ceiling");
}

static void target_rpc(const uint8_t* data, size_t n) {
  if (!g_s || ++g_session_inputs > 4000) session_reset(g_session_inputs & 1);
  {
    peachy_status_info st;
    st.struct_size = sizeof(st);
    A->status(g_s, &st);
    if (st.client_cut_off) session_reset(1);
  }
  feed_message(data, n);
}

static void target_propose(const uint8_t* data, size_t n) {
  static const char prefix[] = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":";
  static char line[MAX_IN + 256];
  size_t k = sizeof(prefix) - 1, total, i;
  if (!g_s || ++g_session_inputs > 4000) session_reset(1);
  {
    peachy_status_info st;
    st.struct_size = sizeof(st);
    A->status(g_s, &st);
    if (st.client_cut_off || st.held) session_reset(1);
  }
  if (n > MAX_IN) n = MAX_IN;
  memcpy(line, prefix, k);
  memcpy(line + k, data, n);
  memcpy(line + k + n, "}}", 2);
  total = k + n + 2;
  /* newlines inside the arguments would split the message; keep it one line */
  for (i = 0; i < total; ++i)
    if (line[i] == '\n' || line[i] == '\r') line[i] = ' ';
  feed_message((const uint8_t*)line, total);
}

/* ---------------------------------------------------------------- allowlist */
static void target_allow(const uint8_t* data, size_t n) {
  pj_allow a, b;
  char why[96];
  static char buf[65536];
  size_t len = 0;
  if (pj_allow_parse((const char*)data, n, &a, why, sizeof(why)) != 0) return;
  if (pj_allow_write(&a, buf, sizeof(buf), &len) != 0) die("accepted allowlist does not write back");
  if (pj_allow_parse(buf, len, &b, why, sizeof(why)) != 0) die("written allowlist does not read back");
  if (a.n != b.n) die("allowlist round trip lost a client");
  {
    uint32_t k;
    for (k = 0; k < a.n; ++k)
      if (strcmp(a.c[k].name, b.c[k].name) || a.c[k].tools != b.c[k].tools || a.c[k].transports != b.c[k].transports ||
          a.c[k].has_token != b.c[k].has_token || memcmp(a.c[k].token_sha, b.c[k].token_sha, 32) ||
          a.c[k].rate_per_sec != b.c[k].rate_per_sec || a.c[k].burst != b.c[k].burst || a.c[k].max_mode != b.c[k].max_mode)
        die("allowlist round trip changed a client");
  }
}

/* ---------------------------------------------------------------- audit */
static char* g_audit_orig;
static size_t g_audit_len;
static pj_audit_report g_audit_rep;
static char g_audit_path[600];

static void audit_setup(void) {
  pj_audit a;
  int k;
  FILE* f;
  long n;
  if (g_audit_orig) return;
  if (pj_audit_open(&a, g_dir, "fuzzaudit") != 0) die("audit open");
  for (k = 0; k < 6; ++k) {
    char body[64];
    int m = snprintf(body, sizeof(body), "\"what\":\"r%d\"", k);
    pj_audit_append(&a, (uint64_t)k, body, (size_t)m);
  }
  snprintf(g_audit_path, sizeof(g_audit_path), "%s", a.path);
  pj_audit_close(&a);
  f = fopen(g_audit_path, "rb");
  fseek(f, 0, SEEK_END);
  n = ftell(f);
  fseek(f, 0, SEEK_SET);
  g_audit_orig = (char*)malloc((size_t)n);
  g_audit_len = fread(g_audit_orig, 1, (size_t)n, f);
  fclose(f);
  pj_audit_verify(g_audit_path, &g_audit_rep);
}

static void target_audit(const uint8_t* data, size_t n) {
  char path[640];
  FILE* f;
  pj_audit_report r;
  int rc;
  audit_setup();
  snprintf(path, sizeof(path), "%s/fuzz-audit-case.log", g_dir);
  f = fopen(path, "wb");
  if (!f) die("audit case write");
  fwrite(data, 1, n, f);
  fclose(f);
  rc = pj_audit_verify(path, &r);
  if (rc == 0 && r.records == g_audit_rep.records && memcmp(r.head, g_audit_rep.head, 32) == 0 &&
      (n != g_audit_len || memcmp(data, g_audit_orig, n) != 0))
    die("a changed audit log verified as the original");
}

#if defined(PEACHY_LIBFUZZER)
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size);
int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > MAX_IN) return 0;
#if defined(PEACHY_FUZZ_TARGET_json)
  target_json(data, size);
#elif defined(PEACHY_FUZZ_TARGET_rpc)
  target_rpc(data, size);
#elif defined(PEACHY_FUZZ_TARGET_propose)
  target_propose(data, size);
#elif defined(PEACHY_FUZZ_TARGET_allow)
  target_allow(data, size);
#endif
  return 0;
}
#else
/* ---------------------------------------------------------------- seeds and mutation */
static const char* const k_json_seeds[] = {
    "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}",
    "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}",
    "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"steps\":2}]}}}",
    "{\"jsonrpc\":\"2.0\",\"id\":\"s\",\"method\":\"tools/call\",\"params\":{\"name\":\"observe\",\"arguments\":{\"radius\":2}}}",
    "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"proposal_status\",\"arguments\":{\"id\":1}}}",
    "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"hold\"}}",
    "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}",
    "[1,2.5e-3,-0,\"\\u00e9\\ud83d\\ude00\",true,false,null,{\"a\":{\"b\":[]}}]",
    "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\"}",
    NULL};
static const char* const k_propose_seeds[] = {
    "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\"}]}",
    "{\"actions\":[{\"kind\":\"look\",\"turn\":\"around\"},{\"kind\":\"action\",\"which\":\"use\"}]}",
    "{\"actions\":[{\"kind\":\"move\",\"dir\":\"west\",\"steps\":4},{\"kind\":\"move\",\"dir\":\"east\",\"steps\":1}]}",
    "{}", NULL};
static const char* const k_allow_seeds[] = {
    "{\"version\":1,\"clients\":[{\"name\":\"grok\",\"transports\":[\"stdio\",\"in_process\"],\"token_sha256\":"
    "\"00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff\",\"tools\":[\"observe\",\"propose\"],"
    "\"rate_per_sec\":8,\"burst\":16,\"max_mode\":\"advisory\"},{\"name\":\"b\",\"transports\":[]}]}",
    "{\"version\":1,\"clients\":[]}", NULL};
static const char* const k_dict[] = {"{", "}", "[", "]", ",", ":", "\"", "\\", "\\u0000", "\\ud800", "\\udc00", "1e999",
                                     "-0", "0.0", "null", "true", "false", "\"jsonrpc\"", "\"2.0\"", "\"id\"", "\"method\"",
                                     "\"params\"", "\"tools/call\"", "\"name\"", "\"arguments\"", "\"propose\"",
                                     "\"actions\"", "\"kind\"", "\"move\"", "\"dir\"", "\"steps\"", "\"version\"",
                                     "\"clients\"", "\"tools\"", "[[[[[[[[[[[[[[[[", "\xff", "\xc0\x80", "\xed\xa0\x80",
                                     "\xf4\x90\x80\x80", "9223372036854775808", "-9223372036854775809", "\n", "\r\n", NULL};

static uint64_t g_rng;
static uint64_t rnd(void) {
  g_rng ^= g_rng >> 12;
  g_rng ^= g_rng << 25;
  g_rng ^= g_rng >> 27;
  return g_rng * 2685821657736338717ull;
}

static size_t mutate(uint8_t* b, size_t n, size_t cap, const char* const* seeds) {
  int m, rounds = 1 + (int)(rnd() % 8);
  for (m = 0; m < rounds; ++m) {
    size_t pos = n ? (size_t)(rnd() % (n + 1)) : 0;
    switch (rnd() % 8) {
      case 0:
        if (n) b[rnd() % n] ^= (uint8_t)(1u << (rnd() % 8));
        break;
      case 1:
        if (n) b[rnd() % n] = (uint8_t)rnd();
        break;
      case 2:
        if (n < cap) {
          memmove(b + pos + 1, b + pos, n - pos);
          b[pos] = (uint8_t)rnd();
          ++n;
        }
        break;
      case 3:
        if (n) {
          size_t len = 1 + (size_t)(rnd() % 16);
          if (pos >= n) pos = n - 1;
          if (pos + len > n) len = n - pos;
          memmove(b + pos, b + pos + len, n - pos - len);
          n -= len;
        }
        break;
      case 4: {
        int nd = 0;
        const char* t;
        size_t tl;
        while (k_dict[nd]) ++nd;
        t = k_dict[rnd() % (uint64_t)nd];
        tl = strlen(t);
        if (n + tl <= cap) {
          memmove(b + pos + tl, b + pos, n - pos);
          memcpy(b + pos, t, tl);
          n += tl;
        }
        break;
      }
      case 5:
        if (n) {
          size_t src = (size_t)(rnd() % n), len = 1 + (size_t)(rnd() % 32);
          if (src + len > n) len = n - src;
          if (n + len <= cap) {
            memmove(b + pos + len, b + pos, n - pos);
            memmove(b + pos, b + (src >= pos ? src + len : src), len);
            n += len;
          }
        }
        break;
      case 6: {
        int ns = 0;
        const char* s;
        size_t sl, cut;
        while (seeds[ns]) ++ns;
        s = seeds[rnd() % (uint64_t)ns];
        sl = strlen(s);
        cut = (size_t)(rnd() % (sl + 1));
        if (pos + (sl - cut) <= cap) {
          memcpy(b + pos, s + cut, sl - cut);
          n = pos + (sl - cut);
        }
        break;
      }
      default:
        if (n) n = (size_t)(rnd() % n);
        break;
    }
  }
  return n;
}

typedef void (*target_fn)(const uint8_t*, size_t);

static unsigned long run_target(const char* name, target_fn fn, const char* const* seeds, unsigned long runs, int audit_mode) {
  static uint8_t buf[MAX_IN * 2];
  unsigned long k;
  int ns = 0;
  while (seeds && seeds[ns]) ++ns;
  for (k = 0; k < runs; ++k) {
    size_t n;
    if (audit_mode) {
      audit_setup();
      n = g_audit_len < sizeof(buf) ? g_audit_len : sizeof(buf);
      memcpy(buf, g_audit_orig, n);
      if (rnd() % 16) n = mutate(buf, n, sizeof(buf), k_json_seeds);
    } else {
      const char* s = seeds[k % (unsigned long)ns];
      n = strlen(s);
      memcpy(buf, s, n);
      if (k >= (unsigned long)ns) n = mutate(buf, n, MAX_IN, seeds);
    }
    fn(buf, n);
  }
  printf("%s: %lu inputs, no failures\n", name, runs);
  fflush(stdout);
  return runs;
}

static void write_seeds(const char* dir) {
  static const char* const* sets[3] = {k_json_seeds, k_propose_seeds, k_allow_seeds};
  static const char* const names[3] = {"json", "propose", "allow"};
  int s, k;
  for (s = 0; s < 3; ++s)
    for (k = 0; sets[s][k]; ++k) {
      char path[700];
      FILE* f;
      snprintf(path, sizeof(path), "%s/%s-%d.seed", dir, names[s], k);
      f = fopen(path, "wb");
      if (!f) continue;
      fwrite(sets[s][k], 1, strlen(sets[s][k]), f);
      fclose(f);
    }
}

int main(int argc, char** argv) {
  const char* which = argc > 1 ? argv[1] : "all";
  unsigned long runs = argc > 2 ? strtoul(argv[2], NULL, 10) : 10000;
  unsigned long total = 0;
  int all = !strcmp(which, "all");
  if (!strcmp(which, "seeds")) {
    write_seeds(argc > 2 ? argv[2] : ".");
    return 0;
  }
  g_rng = (argc > 3 ? strtoull(argv[3], NULL, 10) : 1) * 0x9E3779B97F4A7C15ull + 1;
  if (argc > 4) snprintf(g_dir, sizeof(g_dir), "%s", argv[4]);
  if (all || !strcmp(which, "json")) total += run_target("json", target_json, k_json_seeds, runs, 0);
  if (all || !strcmp(which, "allow")) total += run_target("allow", target_allow, k_allow_seeds, runs, 0);
  if (all || !strcmp(which, "rpc")) total += run_target("rpc", target_rpc, k_json_seeds, all ? runs / 4 + 1 : runs, 0);
  if (all || !strcmp(which, "propose")) total += run_target("propose", target_propose, k_propose_seeds, all ? runs / 4 + 1 : runs, 0);
  if (all || !strcmp(which, "audit")) total += run_target("audit", target_audit, NULL, all ? runs / 20 + 1 : runs, 1);
  if (g_s) {
    A->session_close(g_s);
    remove(g_logpath);
  }
  if (g_audit_orig) {
    char path[640];
    remove(g_audit_path);
    snprintf(path, sizeof(path), "%s/fuzz-audit-case.log", g_dir);
    remove(path);
    free(g_audit_orig);
  }
  printf("total: %lu inputs\n", total);
  return total ? 0 : 1;
}
#endif
