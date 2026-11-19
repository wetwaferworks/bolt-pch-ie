/* peachy_host: a small console host for Peachy (PMCP). MIT license.
 *
 *   peachy_host serve --install <folder> [--lib <peachy library>] --allowlist <file> --audit-dir <dir> --client <name>
 *                     --token-file <file> [--session <name>] [--mode record|advisory]
 *                     [--planner none|pchie|mjolnir] [--pchie <library>] [--mjolnir <library>]
 *                     [--game-exe <program> [--game-arg <arg>]... [--game-workdir <dir>] [--game-stderr <file>]]
 *       Serves PMCP over stdio: one JSON-RPC message per line on stdin, replies on stdout. The token is read from
 *       the file, compared by SHA-256 inside the library, and never printed. Every library and program it loads or
 *       starts (--lib, default the Peachy library in the install folder; --pchie; --mjolnir; --game-exe) must sit in
 *       the install folder and match its manifest, checked before anything is loaded (see src/pj_install.h).
 *   peachy_host install --to <folder> --file <path> [--file <path>]...
 *       Copies the files into a new install folder, writes its SHA-256 manifest and locks the folder down (owned by
 *       Administrators or root, read and run only for this account). Runs only elevated or as root. On Windows the
 *       folder belongs under Program Files, e.g. "C:\Program Files\BoltPCHIE\Peachy".
 *   peachy_host verify-install --install <folder>
 *       Says whether serve would accept the install folder: the guard on every file, nothing extra in it, and every
 *       file's SHA-256 against the manifest.
 *   peachy_host token-new --out <file>
 *       Makes a random token in a new file (never overwrites) and prints only its SHA-256 for the allowlist.
 *   peachy_host allow-add --allowlist <file> --client <name> (--token-sha256 <hex> | --token-file <file>)
 *                     [--transport stdio|in_process] [--max-mode record|advisory] [--rate <n>] [--burst <n>]
 *   peachy_host grant|revoke --allowlist <file> --client <name> <tool>...
 *   peachy_host protect --allowlist <file>
 *       Locks the allowlist and its folder down: owned by Administrators (root), read-only for this account.
 *   peachy_host check --allowlist <file> [--token-file <file>]
 *       Says whether serve would accept these files, and if not, why.
 *   peachy_host show --allowlist <file>
 *   peachy_host version --lib <peachy library>
 *
 * Grants only change in the allowlist file, by the person at this console. The outside models run under the same
 * account as the person, and their own tools can run commands, so "the person at this console" is made real by the
 * operating system rather than by this program: allow-add, grant, revoke and protect run only elevated (Windows) or
 * as root (POSIX), and serve refuses an allowlist this account could change, refuses a token file other accounts
 * could read, and never runs elevated itself. Nothing here reads environment variables. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "peachy/peachy.h"
#include "pj_base.h"
#include "pj_guard.h"
#include "pj_install.h"
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
          "usage: peachy_host serve|install|verify-install|token-new|allow-add|grant|revoke|protect|check|show|version [options]\n"
          "       see the comment at the top of tools/peachy_host.c or peachy/README.md\n");
  return 2;
}

/* ---------------------------------------------------------------- the file guard and elevation (pj_guard.h) */

#if defined(_WIN32)
#define PJ_ELEVATED_TEXT "an elevated (Run as administrator) prompt"
#else
#define PJ_ELEVATED_TEXT "root (sudo)"
#endif

#if defined(PEACHY_TEST_UNGUARDED)
/* The end-to-end test's build: every check passes. Never ship this binary. */
static int guard_file(const char* path, int kind, const char* what) {
  (void)path;
  (void)kind;
  (void)what;
  return 0;
}
static int need_elevated(const char* cmd) {
  (void)cmd;
  return 0;
}
static int serve_not_elevated(void) { return 0; }
static int protect_after_write(const char* path) {
  (void)path;
  return 0;
}
#else
/* 0 if serve may use the file; otherwise prints why (fixed text, never the file's contents) and returns -1. */
static int guard_file(const char* path, int kind, const char* what) {
  pj_file_facts f;
  char why[96];
  if (!path) return 0; /* a missing option is reported by whatever needs it */
  pj_file_facts_get(path, &f);
  if (pj_guard_check(&f, kind, why, sizeof(why)) == 0) return 0;
  fprintf(stderr, "peachy_host: %s refused: %s (see \"Protecting the files\" in peachy/configs/README.md)\n", what,
          why);
  return -1;
}

/* Changing grants needs the operating system's say-so, not just a command line any process could type. */
static int need_elevated(const char* cmd) {
  if (pj_process_elevated() == 1) return 0;
  fprintf(stderr, "peachy_host: %s changes who may do what, so it runs only from %s\n", cmd, PJ_ELEVATED_TEXT);
  return -1;
}

/* An elevated process could rewrite its own allowlist, so serving from one would undo the guard. */
static int serve_not_elevated(void) {
  if (pj_process_elevated() == 0) return 0;
  fprintf(stderr, "peachy_host: serve never runs elevated or as root: such a process could change its own grants\n");
  return -1;
}

static int protect_after_write(const char* path) {
  if (pj_protect_path(path, PJ_GUARD_ALLOWLIST, 0) == 0) return 0;
  fprintf(stderr, "peachy_host: the allowlist was written but could not be locked down; run protect\n");
  return -1;
}
#endif

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

/* ---------------------------------------------------------------- the install folder (pj_install.h) */

#if defined(_WIN32)
#define PJ_DEFAULT_LIB "peachy.dll"
#else
#define PJ_DEFAULT_LIB "libpeachy.so"
#endif

/* Every file in the install folder, held open after its check, and the checked paths serve may load. */
typedef struct code_set {
  pj_install_manifest m;
  pj_held_file held[PJ_INSTALL_MAX_FILES];
  uint32_t n_held;
  char lib[1024], pchie[1024], mjolnir[1024], game[1024];
} code_set;

static void code_release(code_set* c) {
  uint32_t i;
  for (i = 0; i < c->n_held; ++i) pj_held_close(&c->held[i]);
  c->n_held = 0;
}

/* Checks the whole install folder: the guard on the manifest and every file, nothing in the folder that the
 * manifest does not name, and every file's SHA-256 through a handle that stays open. 0 on success. */
static int check_install(const char* dir, code_set* c, int verbose) {
  char p[1100], why[160];
  uint32_t i;
  if (snprintf(p, sizeof(p), "%s/%s", dir, PJ_INSTALL_MANIFEST) >= (int)sizeof(p)) return -1;
  if (guard_file(p, PJ_GUARD_CODE, "the install manifest")) return -1;
  if (pj_install_read_manifest(dir, &c->m, why, sizeof(why)) != 0 ||
      pj_install_check_contents(dir, &c->m, why, sizeof(why)) != 0) {
    fprintf(stderr, "peachy_host: the install folder refused: %s\n", why);
    return -1;
  }
  for (i = 0; i < c->m.n; ++i) {
    if (snprintf(p, sizeof(p), "%s/%s", dir, c->m.e[i].name) >= (int)sizeof(p)) return -1;
    if (guard_file(p, PJ_GUARD_CODE, c->m.e[i].name)) return -1;
    if (pj_install_open_verified(dir, p, &c->m, &c->held[c->n_held], why, sizeof(why)) != 0) {
      fprintf(stderr, "peachy_host: %s refused: %s\n", c->m.e[i].name, why);
      return -1;
    }
    ++c->n_held;
    if (verbose) printf("ok  %s\n", c->m.e[i].name);
  }
  return 0;
}

/* Maps one path given on the command line to the checked file it names: it must resolve to a file directly in
 * the install folder that matches the manifest. out gets the checked location. 0 on success. */
static int resolve_code(const char* dir, code_set* c, const char* given, const char* what, char* out, size_t cap) {
  pj_held_file h;
  char why[160];
  if (!given) {
    out[0] = 0;
    return 0;
  }
  if (pj_install_open_verified(dir, given, &c->m, &h, why, sizeof(why)) != 0) {
    fprintf(stderr, "peachy_host: %s refused: %s\n", what, why);
    return -1;
  }
  pj_copy_cstr(out, cap, h.path);
  pj_held_close(&h); /* the folder-wide check above holds every file open already */
  return 0;
}

/* Everything serve loads or starts is checked here, before the first library is loaded. */
static int check_code(int argc, char** argv, code_set* c) {
  const char* dir = opt(argc, argv, "--install");
  char dflt[1100];
  const char* lib = opt(argc, argv, "--lib");
  memset(c, 0, sizeof(*c));
  if (!dir) {
#if defined(PEACHY_TEST_UNGUARDED)
    /* the test build may still serve straight from the build tree */
    pj_copy_cstr(c->lib, sizeof(c->lib), lib ? lib : "");
    if (opt(argc, argv, "--pchie")) pj_copy_cstr(c->pchie, sizeof(c->pchie), opt(argc, argv, "--pchie"));
    if (opt(argc, argv, "--mjolnir")) pj_copy_cstr(c->mjolnir, sizeof(c->mjolnir), opt(argc, argv, "--mjolnir"));
    if (opt(argc, argv, "--game-exe")) pj_copy_cstr(c->game, sizeof(c->game), opt(argc, argv, "--game-exe"));
    return 0;
#else
    fprintf(stderr, "peachy_host: serve loads code only from a protected install folder: pass --install <folder> "
                    "(made by peachy_host install)\n");
    return -1;
#endif
  }
  if (check_install(dir, c, 0)) goto bad;
  if (!lib) {
    if (snprintf(dflt, sizeof(dflt), "%s/%s", dir, PJ_DEFAULT_LIB) >= (int)sizeof(dflt)) goto bad;
    lib = dflt;
  }
  if (resolve_code(dir, c, lib, "the Peachy library", c->lib, sizeof(c->lib)) ||
      resolve_code(dir, c, opt(argc, argv, "--pchie"), "the PCHIE library", c->pchie, sizeof(c->pchie)) ||
      resolve_code(dir, c, opt(argc, argv, "--mjolnir"), "the Mjolnir library", c->mjolnir, sizeof(c->mjolnir)) ||
      resolve_code(dir, c, opt(argc, argv, "--game-exe"), "the game program", c->game, sizeof(c->game)))
    goto bad;
  return 0;
bad:
  code_release(c);
  return -1;
}

static int cmd_verify_install(int argc, char** argv) {
  const char* dir = opt(argc, argv, "--install");
  static code_set c;
  if (!dir) return usage();
  memset(&c, 0, sizeof(c));
  if (check_install(dir, &c, 1)) {
    code_release(&c);
    printf("REFUSED (the reason is on stderr)\n");
    return 1;
  }
  printf("OK %u files match the install manifest\n", c.m.n);
  code_release(&c);
  return 0;
}

/* Copies src to a new file dst (never overwrites). 0 on success. */
static int copy_new(const char* src, const char* dst) {
  FILE *in = pj_fopen(src, "rb"), *out = NULL;
  static char buf[1 << 16];
  size_t n;
  int ok = 1;
  if (!in) return -1;
  out = pj_fopen(dst, "wbx");
  if (!out) {
    fclose(in);
    return -1;
  }
  while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
    if (fwrite(buf, 1, n, out) != n) {
      ok = 0;
      break;
    }
  if (ferror(in)) ok = 0;
  fclose(in);
  if (fflush(out) != 0) ok = 0;
  if (fclose(out) != 0) ok = 0;
  return ok ? 0 : -1;
}

static const char* last_part(const char* p) {
  const char* b = p;
  for (; *p; ++p)
    if (*p == '/' || *p == '\\') b = p + 1;
  return b;
}

static int cmd_install(int argc, char** argv) {
  const char* dir = opt(argc, argv, "--to");
  static pj_install_manifest m;
  static char text[PJ_INSTALL_MANIFEST_MAX + 1];
  char p[1100];
  size_t tn;
  uint32_t i;
  int k;
  if (!dir) return usage();
  if (need_elevated("install")) return 1;
  memset(&m, 0, sizeof(m));
  /* first every source: a good name, readable, hashed */
  for (k = 2; k + 1 < argc; ++k) {
    uint8_t sha[32];
    uint64_t size = 0;
    if (strcmp(argv[k], "--file")) continue;
    ++k;
    if (pj_install_hash_file(argv[k], sha, &size) != 0 || pj_install_add(&m, last_part(argv[k]), sha, size) != 0) {
      fprintf(stderr, "peachy_host: install refused: %s cannot be read, has an unusual name, or is listed twice\n",
              last_part(argv[k]));
      return 1;
    }
  }
  if (m.n == 0) return usage();
  tn = pj_install_format(&m, text, sizeof(text));
  if (tn == 0) return 1;
  if (pj_dir_exists(dir)) {
    fprintf(stderr, "peachy_host: install makes a new folder; %s already exists\n", dir);
    return 1;
  }
#if defined(_WIN32)
  wchar_t wdir[1100];
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, dir, -1, wdir, 1100) <= 0 || !CreateDirectoryW(wdir, NULL)) {
#else
  if (mkdir(dir, 0755) != 0) {
#endif
    fprintf(stderr, "peachy_host: cannot make the install folder\n");
    return 1;
  }
  for (k = 2; k + 1 < argc; ++k) {
    uint8_t sha[32];
    uint64_t size = 0;
    const pj_install_entry* e;
    if (strcmp(argv[k], "--file")) continue;
    ++k;
    e = pj_install_find(&m, last_part(argv[k]));
    if (!e || snprintf(p, sizeof(p), "%s/%s", dir, e->name) >= (int)sizeof(p) || copy_new(argv[k], p) != 0 ||
        pj_install_hash_file(p, sha, &size) != 0 || size != e->size || memcmp(sha, e->sha256, 32) != 0) {
      fprintf(stderr, "peachy_host: install failed copying %s; the folder is incomplete, remove it and try again\n",
              last_part(argv[k]));
      return 1;
    }
  }
  if (snprintf(p, sizeof(p), "%s/%s", dir, PJ_INSTALL_MANIFEST) >= (int)sizeof(p)) return 1;
  {
    FILE* f = pj_fopen(p, "wbx");
    int ok = f && fwrite(text, 1, tn, f) == tn;
    if (f) ok = (fclose(f) == 0) && ok;
    if (!ok) {
      fprintf(stderr, "peachy_host: cannot write the install manifest\n");
      return 1;
    }
  }
#if !defined(PEACHY_TEST_UNGUARDED)
  /* lock down: the folder first, then each file */
  if (pj_protect_path(dir, PJ_GUARD_CODE, 1) != 0 || pj_protect_path(p, PJ_GUARD_CODE, 0) != 0) {
    fprintf(stderr, "peachy_host: the install folder could not be locked down\n");
    return 1;
  }
  for (i = 0; i < m.n; ++i) {
    char f[1100];
    if (snprintf(f, sizeof(f), "%s/%s", dir, m.e[i].name) >= (int)sizeof(f) || pj_protect_path(f, PJ_GUARD_CODE, 0) != 0) {
      fprintf(stderr, "peachy_host: %s could not be locked down\n", m.e[i].name);
      return 1;
    }
  }
#else
  (void)i;
#endif
  printf("installed %u files into %s; from a normal prompt, peachy_host verify-install --install <folder> "
         "confirms serve accepts it\n", m.n, dir);
  return 0;
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
  void* lib = NULL;
  const peachy_api* api = NULL;
  static code_set code;
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
  /* Nothing is loaded until every check has passed: a library runs code the moment it loads, before any check
   * inside it could. First this process and the files it trusts, then the code itself. */
  if (serve_not_elevated() || guard_file(opt(argc, argv, "--token-file"), PJ_GUARD_TOKEN, "the token file") ||
      guard_file(opt(argc, argv, "--allowlist"), PJ_GUARD_ALLOWLIST, "the allowlist"))
    return 2;
  if (check_code(argc, argv, &code)) return 2;
  api = load_api(code.lib[0] ? code.lib : NULL, &lib);
  if (!api) {
    code_release(&code);
    return 1;
  }
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
  c.pchie_path = code.pchie[0] ? code.pchie : NULL;
  c.mjolnir_path = code.mjolnir[0] ? code.mjolnir : NULL;
  if (code.game[0]) {
    game_argv[game_argc++] = code.game;
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
  /* make the file empty, close it to this account (plus SYSTEM and Administrators), then write the token */
  f = pj_fopen(path, "wbx");
  if (f) {
    int ok = fclose(f) == 0 && pj_protect_path(path, PJ_GUARD_TOKEN, 0) == 0;
    f = ok ? pj_fopen(path, "wb") : NULL;
    if (!ok) {
      fprintf(stderr, "peachy_host: the new token file could not be made private; delete it and try again\n");
      pj_wipe(raw, sizeof(raw));
      pj_wipe(hex, sizeof(hex));
      return 1;
    }
  }
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
  {
    int ok = fwrite(hex, 1, 64, f) == 64;
    ok = (fclose(f) == 0) && ok;
    if (!ok) {
      pj_wipe(raw, sizeof(raw));
      pj_wipe(hex, sizeof(hex));
      fprintf(stderr, "peachy_host: the token file could not be written completely; delete it and try again\n");
      return 1;
    }
  }
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

/* A whole decimal number in [lo, hi]: digits only, no sign, no spaces, no trailing text, no overflow. A missing
 * option keeps dflt. 0 ok, -1 refused. */
static int opt_u32(int argc, char** argv, const char* name, uint32_t lo, uint32_t hi, uint32_t dflt, uint32_t* out) {
  const char* t = opt(argc, argv, name);
  uint64_t v = 0;
  if (!t) {
    *out = dflt;
    return 0;
  }
  if (!*t) return -1;
  for (; *t; ++t) {
    if (*t < '0' || *t > '9') return -1;
    v = v * 10u + (uint64_t)(*t - '0');
    if (v > hi) return -1;
  }
  if (v < lo) return -1;
  *out = (uint32_t)v;
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
  if (need_elevated("allow-add")) return 1;
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
  /* every option is checked before anything is written, so a typo never produces an allowlist that won't load */
  if (tr && strcmp(tr, "stdio") && strcmp(tr, "in_process")) {
    fprintf(stderr, "peachy_host: --transport must be stdio or in_process\n");
    return usage();
  }
  if (mm && strcmp(mm, "record") && strcmp(mm, "advisory")) {
    fprintf(stderr, "peachy_host: --max-mode must be record or advisory\n");
    return usage();
  }
  c->transports = (tr && !strcmp(tr, "in_process")) ? PJ_TR_IN_PROCESS : PJ_TR_STDIO;
  c->max_mode = (mm && !strcmp(mm, "advisory")) ? PEACHY_MODE_ADVISORY : PEACHY_MODE_RECORD;
  if (opt_u32(argc, argv, "--rate", 1, 256, 8, &c->rate_per_sec)) {
    fprintf(stderr, "peachy_host: --rate must be a whole number from 1 to 256\n");
    return usage();
  }
  if (opt_u32(argc, argv, "--burst", 1, 512, 16, &c->burst)) {
    fprintf(stderr, "peachy_host: --burst must be a whole number from 1 to 512\n");
    return usage();
  }
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
  if (protect_after_write(path)) return 1;
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
  if (need_elevated(grant ? "grant" : "revoke")) return 1;
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
  if (protect_after_write(path)) return 1;
  printf("%s %s; restart the session for it to take effect\n", grant ? "granted to" : "revoked from", name);
  return 0;
}

/* The folder part of path, or -1 if there is none (protect never guesses at the current folder). */
static int folder_of(const char* path, char* out, size_t cap) {
  size_t n = strlen(path);
  while (n && path[n - 1] != '/' && path[n - 1] != '\\') --n;
  if (n < 2 || n >= cap) return -1; /* no folder, or the root itself */
  memcpy(out, path, n - 1);
  out[n - 1] = 0;
  if (out[0] == 0 || (out[1] == ':' && out[2] == 0)) return -1;
  return 0;
}

/* protect locks the whole folder down, so the folder must hold nothing but this allowlist (and its .tmp). */
#if defined(_WIN32)
static int folder_holds_only(const char* dir, const char* base) {
  wchar_t pat[1100];
  WIN32_FIND_DATAW fd;
  HANDLE h;
  char name[600];
  int ok = 1;
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, dir, -1, pat, 1000) <= 0) return 0;
  wcscat(pat, L"\\*");
  h = FindFirstFileW(pat, &fd);
  if (h == INVALID_HANDLE_VALUE) return 0;
  do {
    if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
    if (WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name, (int)sizeof(name), NULL, NULL) <= 0) ok = 0;
    else if (strcmp(name, base) && !(strncmp(name, base, strlen(base)) == 0 && !strcmp(name + strlen(base), ".tmp")))
      ok = 0;
  } while (ok && FindNextFileW(h, &fd));
  FindClose(h);
  return ok;
}
static int make_folder(const char* dir) { return CreateDirectoryA(dir, NULL) ? 0 : -1; }
#else
#include <dirent.h>
static int folder_holds_only(const char* dir, const char* base) {
  DIR* d = opendir(dir);
  struct dirent* e;
  int ok = 1;
  if (!d) return 0;
  while (ok && (e = readdir(d)) != NULL) {
    if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
    if (strcmp(e->d_name, base) &&
        !(strncmp(e->d_name, base, strlen(base)) == 0 && !strcmp(e->d_name + strlen(base), ".tmp")))
      ok = 0;
  }
  closedir(d);
  return ok;
}
static int make_folder(const char* dir) { return mkdir(dir, 0755) == 0 ? 0 : -1; }
#endif

static int cmd_protect(int argc, char** argv) {
  const char* path = opt(argc, argv, "--allowlist");
  char dir[600];
  const char* base;
  FILE* f;
  if (!path) return usage();
  if (folder_of(path, dir, sizeof(dir)) != 0) {
    fprintf(stderr, "peachy_host: give the allowlist's full path, in a folder of its own\n");
    return 2;
  }
  base = path + strlen(dir) + 1;
  if (need_elevated("protect")) return 1;
  if (!pj_dir_exists(dir) && make_folder(dir) != 0) {
    fprintf(stderr, "peachy_host: cannot make the allowlist's folder\n");
    return 1;
  }
  if (!folder_holds_only(dir, base)) {
    fprintf(stderr, "peachy_host: protect locks the whole folder down, so it must hold only the allowlist; "
                    "move everything else out first\n");
    return 1;
  }
  if (pj_protect_path(dir, PJ_GUARD_ALLOWLIST, 1) != 0) {
    fprintf(stderr, "peachy_host: cannot lock the allowlist's folder down\n");
    return 1;
  }
  f = pj_fopen(path, "rb");
  if (f) {
    fclose(f);
    if (pj_protect_path(path, PJ_GUARD_ALLOWLIST, 0) != 0) {
      fprintf(stderr, "peachy_host: cannot lock the allowlist down\n");
      return 1;
    }
  }
  printf("protected %s; from a normal prompt, peachy_host check --allowlist <file> confirms serve accepts it\n", dir);
  return 0;
}

static int cmd_check(int argc, char** argv) {
  const char* path = opt(argc, argv, "--allowlist");
  const char* tok = opt(argc, argv, "--token-file");
  int bad = 0;
  if (!path && !tok) return usage();
  if (serve_not_elevated()) bad = 1;
  if (tok && guard_file(tok, PJ_GUARD_TOKEN, "the token file")) bad = 1;
  if (path && guard_file(path, PJ_GUARD_ALLOWLIST, "the allowlist")) bad = 1;
  if (bad) {
    printf("REFUSED: serve would not start with these files (the reason is on stderr)\n");
    return 1;
  }
  printf("OK: serve would accept these files\n");
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
  const peachy_api* api;
  peachy_version_info vi;
  if (serve_not_elevated()) return 2; /* it loads a library, so never elevated either */
  api = load_api(opt(argc, argv, "--lib"), &lib);
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
  if (!strcmp(argv[1], "protect")) return cmd_protect(argc, argv);
  if (!strcmp(argv[1], "check")) return cmd_check(argc, argv);
  if (!strcmp(argv[1], "show")) return cmd_show(argc, argv);
  if (!strcmp(argv[1], "version")) return cmd_version(argc, argv);
  if (!strcmp(argv[1], "install")) return cmd_install(argc, argv);
  if (!strcmp(argv[1], "verify-install")) return cmd_verify_install(argc, argv);
  return usage();
}
