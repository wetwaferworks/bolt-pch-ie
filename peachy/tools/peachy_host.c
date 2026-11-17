/* peachy_host: a small console host for Peachy (PMCP). MIT license.
 *
 *   peachy_host serve --lib <peachy library> --allowlist <file> --audit-dir <dir> --client <name>
 *                     --token-file <file> [--session <name>] [--mode record|advisory]
 *                     [--planner none|pchie|mjolnir] [--pchie <library>] [--mjolnir <library>]
 *                     [--game-exe <program> [--game-arg <arg>]... [--game-workdir <dir>] [--game-stderr <file>]]
 *       Serves PMCP over stdio: one JSON-RPC message per line on stdin, replies on stdout. The token is read from
 *       the file, compared by SHA-256 inside the library, and never printed.
 *   peachy_host token-new --out <file>
 *       Makes a random token in a new file (never overwrites) and prints only its SHA-256 for the allowlist.
 *   peachy_host allow-add --allowlist <file> --client <name> (--token-sha256 <hex> | --token-file <file>)
 *                     [--transport stdio|in_process] [--max-mode record|advisory] [--rate <n>] [--burst <n>]
 *   peachy_host grant|revoke --allowlist <file> --client <name> <tool>...
 *   peachy_host show --allowlist <file>
 *   peachy_host version --lib <peachy library>
 *
 * Grants only change in the allowlist file, by the person at this console. Nothing here reads environment
 * variables. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "peachy/peachy.h"
#include "pj_base.h"
#include "pj_internal.h"
#include "pj_platform.h"
#include "pj_sha256.h"

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <windows.h>
static long read_stdin(char* buf, size_t cap) {
  DWORD got = 0;
  if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), buf, (DWORD)cap, &got, NULL)) return 0;
  return (long)got;
}
#else
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
static long read_stdin(char* buf, size_t cap) {
  for (;;) {
    ssize_t n = read(0, buf, cap);
    if (n < 0 && errno == EINTR) continue;
    return n < 0 ? 0 : (long)n;
  }
}
#endif

static const char* opt(int argc, char** argv, const char* name) {
  int i;
  for (i = 2; i + 1 < argc; ++i)
    if (!strcmp(argv[i], name)) return argv[i + 1];
  return NULL;
}

static int usage(void) {
  fprintf(stderr,
          "usage: peachy_host serve|token-new|allow-add|grant|revoke|show|version [options]\n"
          "       see the comment at the top of tools/peachy_host.c or peachy/README.md\n");
  return 2;
}

static const peachy_api* load_api(const char* lib_path, void** lib) {
  peachy_get_api_fn get;
  const peachy_api* api = NULL;
  *lib = lib_path ? pj_dl_open(lib_path) : NULL;
  if (!*lib) {
    fprintf(stderr, "peachy_host: cannot load the Peachy library\n");
    return NULL;
  }
  get = (peachy_get_api_fn)pj_dl_fn(*lib, "peachy_get_api");
  if (!get || get(PEACHY_ABI_MAJOR, &api) != PEACHY_OK || !api || api->struct_size < sizeof(peachy_api)) {
    fprintf(stderr, "peachy_host: the library's interface version does not match this host\n");
    pj_dl_close(*lib);
    *lib = NULL;
    return NULL;
  }
  return api;
}

/* Reads a token file into buf (trailing whitespace dropped). Returns length or -1. */
static long read_token(const char* path, uint8_t* buf, size_t cap) {
  FILE* f = path ? pj_fopen(path, "rb") : NULL;
  size_t n;
  if (!f) return -1;
  n = fread(buf, 1, cap, f);
  fclose(f);
  if (n == cap) {
    pj_wipe(buf, cap);
    return -1;
  }
  while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r' || buf[n - 1] == ' ' || buf[n - 1] == '\t')) --n;
  return (long)n;
}

/* ---------------------------------------------------------------- serve */

typedef struct reader {
  const peachy_api* api;
  PeachySession* s;
  volatile uint32_t eof;
} reader;

static void reader_main(void* arg) {
  reader* r = (reader*)arg;
  static char buf[65536];
  for (;;) {
    long n = read_stdin(buf, sizeof(buf));
    if (n <= 0) break;
    for (;;) {
      peachy_status st = r->api->post_input(r->s, buf, (size_t)n);
      if (st == PEACHY_OK) break;
      if (st != PEACHY_E_LIMIT) {
        pj_store_u32(&r->eof, 1);
        return;
      }
      pj_sleep_ms(1); /* ring full: wait for the worker */
    }
  }
  pj_store_u32(&r->eof, 1);
}

static void PEACHY_CALL host_log(void* ctx, int32_t level, const char* msg, size_t len) {
  (void)ctx;
  (void)level;
  fprintf(stderr, "%.*s\n", (int)len, msg);
}

static int drain(const peachy_api* api, PeachySession* s, char* out, size_t cap) {
  int any = 0;
  for (;;) {
    size_t n = 0;
    if (api->take_output(s, out, cap, &n) != PEACHY_OK || n == 0) break;
    fwrite(out, 1, n, stdout);
    fputc('\n', stdout);
    any = 1;
  }
  if (any) fflush(stdout);
  return any;
}

static int cmd_serve(int argc, char** argv) {
  void* lib;
  const peachy_api* api = load_api(opt(argc, argv, "--lib"), &lib);
  peachy_session_config c;
  PeachySession* s = NULL;
  char err[128];
  static uint8_t token[1100];
  long tl;
  const char* game_argv[34];
  uint32_t game_argc = 0;
  const char* mode = opt(argc, argv, "--mode");
  const char* planner = opt(argc, argv, "--planner");
  reader rd;
  pj_thread th;
  static char out[600 * 1024];
  peachy_status st;
  int i;
  uint64_t quiet_since = 0;
  if (!api) return 1;
#if defined(_WIN32)
  _setmode(_fileno(stdout), _O_BINARY);
#endif
  memset(&c, 0, sizeof(c));
  c.struct_size = sizeof(c);
  c.log = host_log;
  c.allowlist_path = opt(argc, argv, "--allowlist");
  c.audit_dir = opt(argc, argv, "--audit-dir");
  c.session_name = opt(argc, argv, "--session") ? opt(argc, argv, "--session") : "stdio";
  c.client_name = opt(argc, argv, "--client");
  c.client_transport = PEACHY_TRANSPORT_STDIO;
  c.requested_mode = (mode && !strcmp(mode, "advisory")) ? PEACHY_MODE_ADVISORY : PEACHY_MODE_RECORD;
  if (mode && strcmp(mode, "advisory") && strcmp(mode, "record")) {
    fprintf(stderr, "peachy_host: --mode must be record or advisory\n");
    return 2;
  }
  if (!planner || !strcmp(planner, "none")) c.planner_kind = PEACHY_PLANNER_NONE;
  else if (!strcmp(planner, "pchie")) c.planner_kind = PEACHY_PLANNER_PCHIE_DIRECT;
  else if (!strcmp(planner, "mjolnir")) c.planner_kind = PEACHY_PLANNER_MJOLNIR;
  else {
    fprintf(stderr, "peachy_host: --planner must be none, pchie or mjolnir\n");
    return 2;
  }
  c.pchie_path = opt(argc, argv, "--pchie");
  c.mjolnir_path = opt(argc, argv, "--mjolnir");
  if (opt(argc, argv, "--game-exe")) {
    game_argv[game_argc++] = opt(argc, argv, "--game-exe");
    for (i = 2; i + 1 < argc && game_argc < 33; ++i)
      if (!strcmp(argv[i], "--game-arg")) game_argv[game_argc++] = argv[++i];
    game_argv[game_argc] = NULL;
    c.game_kind = PEACHY_GAME_SPAWN_STDIO;
    c.game_argv = game_argv;
    c.game_argc = game_argc;
    c.game_workdir = opt(argc, argv, "--game-workdir");
    c.game_stderr_log = opt(argc, argv, "--game-stderr");
  }
  tl = read_token(opt(argc, argv, "--token-file"), token, sizeof(token));
  if (tl < 0) {
    fprintf(stderr, "peachy_host: cannot read the token file\n");
    return 2;
  }
  c.client_token = token;
  c.client_token_len = (size_t)tl;
  st = api->session_open(&c, &s, err, sizeof(err));
  pj_wipe(token, sizeof(token));
  if (st != PEACHY_OK) {
    fprintf(stderr, "peachy_host: session refused: %s (%s)\n", err, api->status_name(st));
    pj_dl_close(lib);
    return 1;
  }
  fprintf(stderr, "peachy_host: PMCP serving on stdio for client %s\n", c.client_name);
  if (api->thread_start(s) != PEACHY_OK) {
    api->session_close(s);
    pj_dl_close(lib);
    return 1;
  }
  memset(&rd, 0, sizeof(rd));
  rd.api = api;
  rd.s = s;
  if (pj_thread_start(&th, reader_main, &rd) != 0) {
    api->session_close(s);
    pj_dl_close(lib);
    return 1;
  }
  for (;;) {
    int any = drain(api, s, out, sizeof(out));
    if (pj_load_u32(&rd.eof)) {
      /* stdin closed: let the worker finish what is queued, then stop */
      if (any) quiet_since = 0;
      else if (!quiet_since) quiet_since = pj_mono_ns();
      else if (pj_mono_ns() - quiet_since > 500000000ull) break;
    }
    if (!any) pj_sleep_ms(1);
  }
  api->thread_stop(s);
  drain(api, s, out, sizeof(out));
  api->session_close(s);
  /* the reader thread has returned (it set eof) */
  pj_thread_join(&th);
  pj_dl_close(lib);
  return 0;
}

/* ---------------------------------------------------------------- tokens and the allowlist */

static int cmd_token_new(int argc, char** argv) {
  const char* path = opt(argc, argv, "--out");
  uint8_t raw[32], h[32];
  char hex[65];
  static const char* digits = "0123456789abcdef";
  int i;
  FILE* f;
  if (!path) return usage();
  if (pj_os_random(raw, sizeof(raw)) != 0) {
    fprintf(stderr, "peachy_host: the system random source failed\n");
    return 1;
  }
  for (i = 0; i < 32; ++i) {
    hex[2 * i] = digits[raw[i] >> 4];
    hex[2 * i + 1] = digits[raw[i] & 15];
  }
  hex[64] = 0;
#if defined(_WIN32)
  f = pj_fopen(path, "wbx");
#else
  {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    f = fd >= 0 ? fdopen(fd, "wb") : NULL;
  }
#endif
  if (!f) {
    pj_wipe(raw, sizeof(raw));
    pj_wipe(hex, sizeof(hex));
    fprintf(stderr, "peachy_host: cannot create the token file (it must not exist yet)\n");
    return 1;
  }
  fwrite(hex, 1, 64, f);
  fclose(f);
  pj_sha256_once(hex, 64, h);
  pj_wipe(raw, sizeof(raw));
  pj_wipe(hex, sizeof(hex));
  for (i = 0; i < 32; ++i) printf("%c%c", digits[h[i] >> 4], digits[h[i] & 15]);
  printf("\n");
  fprintf(stderr, "peachy_host: token written; the line above is its SHA-256 for the allowlist\n");
  return 0;
}

static int load_or_empty(const char* path, pj_allow* a) {
  char why[96];
  FILE* f = pj_fopen(path, "rb");
  if (!f) {
    memset(a, 0, sizeof(*a));
    return 0;
  }
  fclose(f);
  if (pj_allow_load(path, a, why, sizeof(why)) != 0) {
    fprintf(stderr, "peachy_host: allowlist refused: %s\n", why);
    return -1;
  }
  return 0;
}

static int unhex32(const char* s, uint8_t* out) {
  int i;
  if (!s || strlen(s) != 64) return -1;
  for (i = 0; i < 64; ++i) {
    char ch = s[i];
    int v = (ch >= '0' && ch <= '9') ? ch - '0' : (ch >= 'a' && ch <= 'f') ? ch - 'a' + 10 : -1;
    if (v < 0) return -1;
    if (i & 1) out[i / 2] = (uint8_t)(out[i / 2] | v);
    else out[i / 2] = (uint8_t)(v << 4);
  }
  return 0;
}

static int cmd_allow_add(int argc, char** argv) {
  const char* path = opt(argc, argv, "--allowlist");
  const char* name = opt(argc, argv, "--client");
  const char* tr = opt(argc, argv, "--transport");
  const char* mm = opt(argc, argv, "--max-mode");
  pj_allow a;
  pj_client* c;
  uint32_t k;
  if (!path || !name || !pj_name_ok(name, 32)) return usage();
  if (load_or_empty(path, &a)) return 1;
  for (k = 0; k < a.n; ++k)
    if (!strcmp(a.c[k].name, name)) {
      fprintf(stderr, "peachy_host: client already listed\n");
      return 1;
    }
  if (a.n >= PJ_MAX_CLIENTS) return 1;
  c = &a.c[a.n];
  memset(c, 0, sizeof(*c));
  pj_copy_cstr(c->name, sizeof(c->name), name);
  c->transports = (tr && !strcmp(tr, "in_process")) ? PJ_TR_IN_PROCESS : PJ_TR_STDIO;
  c->max_mode = (mm && !strcmp(mm, "advisory")) ? PEACHY_MODE_ADVISORY : PEACHY_MODE_RECORD;
  c->rate_per_sec = opt(argc, argv, "--rate") ? (uint32_t)atoi(opt(argc, argv, "--rate")) : 8;
  c->burst = opt(argc, argv, "--burst") ? (uint32_t)atoi(opt(argc, argv, "--burst")) : 16;
  if (c->rate_per_sec < 1 || c->rate_per_sec > 256 || c->burst < 1 || c->burst > 512) return usage();
  if (opt(argc, argv, "--token-sha256")) {
    if (unhex32(opt(argc, argv, "--token-sha256"), c->token_sha)) return usage();
  } else if (opt(argc, argv, "--token-file")) {
    uint8_t tok[1100];
    long n = read_token(opt(argc, argv, "--token-file"), tok, sizeof(tok));
    if (n < 16) {
      pj_wipe(tok, sizeof(tok));
      fprintf(stderr, "peachy_host: token file missing or too short\n");
      return 1;
    }
    pj_sha256_once(tok, (size_t)n, c->token_sha);
    pj_wipe(tok, sizeof(tok));
  } else {
    return usage();
  }
  c->has_token = 1;
  ++a.n;
  if (pj_allow_save(path, &a) != 0) {
    fprintf(stderr, "peachy_host: cannot write the allowlist\n");
    return 1;
  }
  printf("added %s (no tools granted yet)\n", name);
  return 0;
}

static int cmd_grant(int argc, char** argv, int grant) {
  const char* path = opt(argc, argv, "--allowlist");
  const char* name = opt(argc, argv, "--client");
  pj_allow a;
  pj_client* c = NULL;
  uint32_t k, bits = 0;
  int i;
  if (!path || !name) return usage();
  if (load_or_empty(path, &a)) return 1;
  for (k = 0; k < a.n; ++k)
    if (!strcmp(a.c[k].name, name)) c = &a.c[k];
  if (!c) {
    fprintf(stderr, "peachy_host: client not listed\n");
    return 1;
  }
  for (i = 2; i < argc; ++i) {
    int t;
    if (argv[i][0] == '-' && argv[i][1] == '-') {
      ++i;
      continue;
    }
    t = pj_tool_index(argv[i], strlen(argv[i]));
    if (t < 0 || !(PJ_BIT(t) & PJ_GRANTABLE_TOOLS)) {
      fprintf(stderr, "peachy_host: %s cannot be granted\n", argv[i]);
      return 1;
    }
    bits |= PJ_BIT(t);
  }
  if (!bits) return usage();
  if (grant) c->tools |= bits;
  else c->tools &= ~bits;
  if (pj_allow_save(path, &a) != 0) {
    fprintf(stderr, "peachy_host: cannot write the allowlist\n");
    return 1;
  }
  printf("%s %s; restart the session for it to take effect\n", grant ? "granted to" : "revoked from", name);
  return 0;
}

static int cmd_show(int argc, char** argv) {
  const char* path = opt(argc, argv, "--allowlist");
  pj_allow a;
  static char buf[65536];
  size_t n = 0;
  char why[96];
  if (!path) return usage();
  if (pj_allow_load(path, &a, why, sizeof(why)) != 0) {
    fprintf(stderr, "peachy_host: allowlist refused: %s\n", why);
    return 1;
  }
  if (pj_allow_write(&a, buf, sizeof(buf), &n) != 0) return 1;
  fwrite(buf, 1, n, stdout);
  return 0;
}

static int cmd_version(int argc, char** argv) {
  void* lib;
  const peachy_api* api = load_api(opt(argc, argv, "--lib"), &lib);
  peachy_version_info vi;
  if (!api) return 1;
  memset(&vi, 0, sizeof(vi));
  vi.struct_size = sizeof(vi);
  api->get_version(&vi);
  printf("%s (interface %u.%u%s%s)\n", vi.build_id, vi.abi_major, vi.abi_minor, vi.alpha ? ", alpha" : "",
         vi.checking_build ? ", checking build" : "");
  pj_dl_close(lib);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  if (!strcmp(argv[1], "serve")) return cmd_serve(argc, argv);
  if (!strcmp(argv[1], "token-new")) return cmd_token_new(argc, argv);
  if (!strcmp(argv[1], "allow-add")) return cmd_allow_add(argc, argv);
  if (!strcmp(argv[1], "grant")) return cmd_grant(argc, argv, 1);
  if (!strcmp(argv[1], "revoke")) return cmd_grant(argc, argv, 0);
  if (!strcmp(argv[1], "show")) return cmd_show(argc, argv);
  if (!strcmp(argv[1], "version")) return cmd_version(argc, argv);
  return usage();
}
