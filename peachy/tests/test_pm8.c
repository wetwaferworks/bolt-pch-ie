/* PM-8 stubs: the live frontier-model hookup is not done in this release. These tests pin what is ready
 * (client profiles that parse and carry no secrets, the example allowlist) and what must stay off until it is
 * approved (actuation, at every door). MIT license. */
#include "pt.h"
#include "pj_internal.h"
#include "pj_json.h"

static unsigned char scratch[1 << 22];

static int parse_file(const char* path, pj_doc* d, char** text) {
  pj_limits l = {1u << 20, 32, 256, 4096, 65536, 65536};
  size_t n;
  *text = pt_read_file(path, &n);
  if (!*text) return -1;
  return pj_json_parse(*text, n, &l, scratch, pj_json_scratch_bytes(&l), d);
}

static const pj_tok* find_args(const pj_doc* d, int32_t* out) {
  int32_t root = 0, servers = pj_get(d, root, "mcpServers"), pm, a;
  if (servers < 0) servers = pj_get(d, root, "mcp_servers");
  pm = pj_get(d, servers, "pmcp");
  a = pj_get(d, pm, "args");
  *out = a;
  return a >= 0 ? &d->t[a] : NULL;
}

static int arr_has(const pj_doc* d, int32_t arr, const char* s) {
  int32_t k;
  if (!pj_is(d, arr, PJ_ARR)) return 0;
  for (k = d->t[arr].first; k >= 0; k = d->t[k].next)
    if (pj_str_eq(d, k, s)) return 1;
  return 0;
}

static int arr_after(const pj_doc* d, int32_t arr, const char* key, const char* want) {
  int32_t k;
  for (k = d->t[arr].first; k >= 0; k = d->t[k].next)
    if (pj_str_eq(d, k, key)) return d->t[k].next >= 0 && pj_str_eq(d, d->t[k].next, want);
  return 0;
}

int main(int argc, char** argv) {
  static const char* const profiles[] = {"grok", "grokbot", "gemini", "muse", "claude-desktop", "generic-stdio", NULL};
  const char* dir;
  int i;
  pt_init(argc, argv);
  if (argc < 3) return 2;
  dir = argv[2];
  for (i = 0; profiles[i]; ++i) {
    char path[800], what[160];
    pj_doc d;
    char* text = NULL;
    int32_t a = -1;
    snprintf(path, sizeof(path), "%s/clients/%s.json", dir, profiles[i]);
    snprintf(what, sizeof(what), "%s profile parses", profiles[i]);
    CHECK(parse_file(path, &d, &text) == 0, what);
    find_args(&d, &a);
    snprintf(what, sizeof(what), "%s profile serves PMCP with a token file in record mode", profiles[i]);
    CHECK(a >= 0 && arr_has(&d, a, "serve") && arr_has(&d, a, "--token-file") && arr_after(&d, a, "--mode", "record"), what);
    snprintf(what, sizeof(what), "%s profile loads code only from the protected install folder", profiles[i]);
    CHECK(a >= 0 && arr_has(&d, a, "--install") && !arr_has(&d, a, "--lib"), what);
    snprintf(what, sizeof(what), "%s profile holds no secret", profiles[i]);
    CHECK(text && !strstr(text, "\"token\"") && !strstr(text, "authorization") && !strstr(text, "Bearer") &&
              !strstr(text, "token_sha256") && !strstr(text, "actuate"),
          what);
    free(text);
  }
  {
    char path[800], why[96];
    pj_allow a;
    uint32_t k;
    int zero = 1;
    snprintf(path, sizeof(path), "%s/allowlist.example.json", dir);
    CHECK(pj_allow_load(path, &a, why, sizeof(why)) == 0 && a.n == 6, "example allowlist loads, six clients");
    for (k = 0; k < a.n; ++k) {
      int b;
      for (b = 0; b < 32; ++b)
        if (a.c[k].token_sha[b]) zero = 0;
      if (a.c[k].max_mode != PEACHY_MODE_RECORD) zero = 0;
    }
    CHECK(zero, "example hashes are zero (match no token) and every client is record-only");
  }
  /* actuation stays off at every door */
  {
    pj_allow a;
    char why[96];
    const char* doc = "{\"version\":1,\"clients\":[{\"name\":\"x\",\"transports\":[\"stdio\"],\"max_mode\":\"actuate\"}]}";
    CHECK(pj_allow_parse(doc, strlen(doc), &a, why, sizeof(why)) != 0, "allowlist cannot grant actuate");
  }
  {
    const peachy_api* A;
    peachy_session_config c;
    PeachySession* s = NULL;
    char err[128];
    peachy_get_api(PEACHY_ABI_MAJOR, &A);
    memset(&c, 0, sizeof(c));
    c.struct_size = sizeof(c);
    c.requested_mode = PEACHY_MODE_ACTUATE;
    c.session_name = "pm8";
    CHECK(A->session_open(&c, &s, err, sizeof(err)) == PEACHY_E_UNSUPPORTED && !s, "session cannot open in actuate mode");
  }
  printf("PM-8 live hookup: not wired in this release (needs sign-ins and a go)\n");
  return pt_done("peachy_pm8_stubs");
}
