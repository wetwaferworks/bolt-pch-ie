/* peachy_host end to end: token making, allowlist edits, serving PMCP over stdio with the fake game, and
 * peachy_auditcheck on the resulting log. MIT license. */
#include "pt.h"
#include "pj_guard.h"
#include "pj_platform.h"

#if defined(_WIN32)
#include <windows.h>
static int find_log(const char* dir, char* out, size_t cap) {
  char pat[700];
  WIN32_FIND_DATAA fd;
  HANDLE h;
  snprintf(pat, sizeof(pat), "%s\\peachy-audit-*.log", dir);
  h = FindFirstFileA(pat, &fd);
  if (h == INVALID_HANDLE_VALUE) return -1;
  snprintf(out, cap, "%s/%s", dir, fd.cFileName);
  FindClose(h);
  return 0;
}
#else
#include <dirent.h>
static int find_log(const char* dir, char* out, size_t cap) {
  DIR* d = opendir(dir);
  struct dirent* e;
  int rc = -1;
  if (!d) return -1;
  while ((e = readdir(d)) != NULL)
    if (!strncmp(e->d_name, "peachy-audit-", 13)) {
      snprintf(out, cap, "%s/%s", dir, e->d_name);
      rc = 0;
      break;
    }
  closedir(d);
  return rc;
}
#endif

static const char *g_host, *g_lib, *g_fake, *g_check, *g_guarded, *g_marker;
static char g_dir[600];

/* Runs a program to completion; collects stdout. */
static int run(const char* const* argv, uint32_t argc, char* out, size_t cap, const char* stderr_log) {
  pj_proc p;
  size_t len = 0;
  uint64_t t_end = pj_mono_ns() + 30000000000ull;
  memset(&p, 0, sizeof(p));
  if (pj_proc_spawn(&p, argv, argc, NULL, stderr_log) != 0) return -1;
  for (;;) {
    size_t n = 0;
    if (len + 1 >= cap || pj_mono_ns() > t_end) break;
    if (pj_proc_read(&p, out + len, cap - 1 - len, &n, 100) != 0) break;
    len += n;
  }
  out[len] = 0;
  pj_proc_stop(&p, 5000);
  return (int)len;
}

/* Sends one line and waits for one reply line. */
static int ask(pj_proc* p, const char* line, char* buf, size_t cap, size_t* have) {
  uint64_t t_end = pj_mono_ns() + 30000000000ull;
  if (line) {
    if (pj_proc_write(p, line, strlen(line)) || pj_proc_write(p, "\n", 1)) return -1;
  }
  for (;;) {
    char* nl = *have ? (char*)memchr(buf, '\n', *have) : NULL;
    size_t n = 0;
    if (nl) {
      size_t k = (size_t)(nl - buf);
      memmove(buf + cap / 2, buf, k); /* reply copied to the upper half */
      buf[cap / 2 + k] = 0;
      memmove(buf, nl + 1, *have - k - 1);
      *have -= k + 1;
      return 0;
    }
    if (pj_mono_ns() > t_end || *have + 1 >= cap / 2) return -1;
    if (pj_proc_read(p, buf + *have, cap / 2 - 1 - *have, &n, 100) != 0) return -1;
    *have += n;
  }
}

static int err_has(const char* errlog, const char* what) {
  char* e = pt_read_file(errlog, NULL);
  int r = e && strstr(e, what) != NULL;
  free(e);
  return r;
}

/* The real host, with the same scratch files: grants can't change from an ordinary prompt, and serve won't start
 * on an allowlist this account could rewrite (review 2026-11-16, H1). */
static void guarded(const char* tok, const char* allow, const char* auditdir, const char* errlog) {
  char out[4096];
  size_t before_n = 0, after_n = 0;
  char *before, *after;
  int n, elevated = pj_process_elevated();
  before = pt_read_file(allow, &before_n);
  if (!elevated) {
    const char* g[] = {g_guarded, "grant", "--allowlist", allow, "--client", "grok", "plan", "audit_head"};
    const char* r[] = {g_guarded, "revoke", "--allowlist", allow, "--client", "grok", "propose"};
    const char* a[] = {g_guarded, "allow-add", "--allowlist", allow, "--client", "muse", "--token-file", tok};
    const char* p[] = {g_guarded, "protect", "--allowlist", allow};
    n = run(g, 8, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "runs only from"), "guarded grant refused without elevation");
    n = run(r, 7, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "runs only from"), "guarded revoke refused without elevation");
    n = run(a, 8, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "runs only from"), "guarded allow-add refused without elevation");
    n = run(p, 4, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "runs only from"), "guarded protect refused without elevation");
  } else {
    printf("note: running elevated, so the guarded grant refusals are skipped\n");
  }
  after = pt_read_file(allow, &after_n);
  CHECK(before && after && before_n == after_n && !memcmp(before, after, before_n),
        "the allowlist is byte for byte unchanged");
  free(before);
  free(after);
  {
    const char* c[] = {g_guarded, "check", "--allowlist", allow, "--token-file", tok};
    run(c, 6, out, sizeof(out), errlog);
    CHECK(strstr(out, "REFUSED") && err_has(errlog, "the allowlist refused") &&
              err_has(errlog, elevated ? "elevated" : "can change it"),
          "check: a scratch allowlist is refused, and says why");
  }
  {
    const char* s[] = {g_guarded, "serve", "--lib", g_lib, "--allowlist", allow, "--audit-dir", auditdir, "--client",
                       "grok", "--token-file", tok};
    n = run(s, 12, out, sizeof(out), errlog);
    CHECK(n == 0 && (err_has(errlog, "the allowlist refused") || err_has(errlog, "never runs elevated")),
          "guarded serve refuses a scratch allowlist and serves nothing");
    CHECK(!err_has(errlog, "PMCP serving"), "guarded serve never started a session");
  }
}

static const char* last_part(const char* p) {
  const char* b = p;
  for (; *p; ++p)
    if (*p == '/' || *p == '\\') b = p + 1;
  return b;
}

static int exists(const char* p) {
  FILE* f = fopen(p, "rb");
  if (f) fclose(f);
  return f != NULL;
}

/* The install folder (review 2026-11-19, H-1): serve loads code only from a folder whose manifest it checked, and
 * nothing is loaded before every check has passed. The marker library leaves a file behind the moment it loads. */
static void install_flow(const char* tok, const char* allow, const char* auditdir, const char* errlog) {
  char inst[800], marker_log[900], p[1000], fake_in[1000], lib_in[1000], marker_in[1000], out[4096];
  int n;
  snprintf(inst, sizeof(inst), "%s/inst", g_dir);
  /* the marker library writes next to the test folder base (fixed at build time) */
  snprintf(marker_log, sizeof(marker_log), "%s", pt_base);
  {
    size_t l = strlen(marker_log);
    snprintf(marker_log + l, sizeof(marker_log) - l, "/marker-loaded.txt");
  }
  remove(marker_log);
  snprintf(fake_in, sizeof(fake_in), "%s/%s", inst, last_part(g_fake));
  snprintf(lib_in, sizeof(lib_in), "%s/%s", inst, last_part(g_lib));
  snprintf(marker_in, sizeof(marker_in), "%s/%s", inst, last_part(g_marker));
  {
    const char* a[] = {g_host, "install", "--to", inst, "--file", g_lib, "--file", g_fake, "--file", g_marker};
    run(a, 10, out, sizeof(out), errlog);
    CHECK(strstr(out, "installed 3 files") != NULL, "install copies the files and writes the manifest");
    n = run(a, 10, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "already exists"), "install never reuses an existing folder");
  }
  {
    const char* v[] = {g_host, "verify-install", "--install", inst};
    run(v, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "OK 3 files") != NULL, "verify-install accepts what install wrote");
  }
  /* serve from the install folder, with the default library name */
  {
    const char* a[] = {g_host, "serve", "--install", inst, "--allowlist", allow, "--audit-dir", auditdir, "--client", "grok",
                       "--token-file", tok, "--session", "insttest", "--game-exe", fake_in};
    static char io[1 << 16];
    size_t have = 0;
    pj_proc pr;
    memset(&pr, 0, sizeof(pr));
    CHECK(pj_proc_spawn(&pr, a, 16, NULL, errlog) == 0, "serve --install starts");
    CHECK(ask(&pr, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}", io,
              sizeof(io), &have) == 0,
          "serve --install answers initialize");
    pj_proc_stop(&pr, 10000);
  }
  /* a library from outside the folder, even with the same bytes */
  {
    const char* a[] = {g_host, "serve", "--install", inst, "--lib", g_lib, "--allowlist", allow, "--audit-dir", auditdir,
                       "--client", "grok", "--token-file", tok};
    n = run(a, 14, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "not directly inside") && !err_has(errlog, "PMCP serving"),
          "serve refuses a library outside the install folder");
  }
  /* one changed byte in a planner library: refused, and it never loaded */
  {
    size_t len = 0;
    char* b = pt_read_file(marker_in, &len);
    const char* a[] = {g_host, "serve", "--install", inst, "--allowlist", allow, "--audit-dir", auditdir, "--client", "grok",
                       "--token-file", tok, "--planner", "pchie", "--pchie", marker_in};
    if (b && len > 100) {
      b[len - 1] ^= 1;
      pt_write_file(marker_in, b, len);
    }
    n = run(a, 16, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "does not match the install manifest") && !err_has(errlog, "PMCP serving"),
          "serve refuses a tampered library");
    CHECK(!exists(marker_log), "the tampered library was never loaded");
    {
      const char* v[] = {g_host, "verify-install", "--install", inst};
      run(v, 4, out, sizeof(out), errlog);
      CHECK(strstr(out, "REFUSED") != NULL, "verify-install refuses the tampered folder");
    }
    if (b && len > 100) {
      b[len - 1] ^= 1;
      pt_write_file(marker_in, b, len);
    }
    free(b);
  }
  /* a stray file next to the programs (a planted library the game could pick up) */
  {
    const char* v[] = {g_host, "verify-install", "--install", inst};
    snprintf(p, sizeof(p), "%s/version.dll", inst);
    pt_write_file(p, "MZ", 2);
    run(v, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "REFUSED") && err_has(errlog, "version.dll"), "a stray file in the install folder is refused");
    remove(p);
    run(v, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "OK 3 files") != NULL, "and accepted again once it is gone");
  }
  /* the real host: a scratch folder this account can change is not a protected install */
  {
    const char* v[] = {g_guarded, "verify-install", "--install", inst};
    run(v, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "REFUSED") && (err_has(errlog, "can change it") || err_has(errlog, "folder above")),
          "guarded verify-install refuses a folder this account can change");
  }
  if (!pj_process_elevated()) {
    char inst2[900];
    const char* a[] = {g_guarded, "install", "--to", inst2, "--file", g_lib};
    snprintf(inst2, sizeof(inst2), "%s/inst2", g_dir);
    n = run(a, 6, out, sizeof(out), errlog);
    CHECK(n == 0 && err_has(errlog, "runs only from") && !exists(inst2), "guarded install runs only elevated");
  }
  {
    /* the review's repro: point --lib at a library that acts when loaded. The old host loaded it before any check. */
    const char* a[] = {g_guarded, "serve", "--lib", g_marker, "--allowlist", allow, "--audit-dir", auditdir, "--client",
                       "grok", "--token-file", tok};
    const char* b2[] = {g_guarded, "serve", "--install", inst, "--lib", marker_in, "--allowlist", allow, "--audit-dir",
                        auditdir, "--client", "grok", "--token-file", tok};
    run(a, 12, out, sizeof(out), errlog);
    CHECK(!exists(marker_log), "guarded serve loads nothing before its checks (no install folder)");
    run(b2, 14, out, sizeof(out), errlog);
    CHECK(!exists(marker_log) && !err_has(errlog, "PMCP serving"),
          "guarded serve loads nothing from a folder this account can change");
  }
  {
    /* control: the marker does leave its file when something does load it, so the checks above mean something */
    const char* v[] = {g_host, "version", "--lib", g_marker};
    run(v, 4, out, sizeof(out), errlog);
    CHECK(exists(marker_log), "control: loading the marker library leaves its file");
    remove(marker_log);
  }
}

int main(int argc, char** argv) {
  char tok[900], allow[900], maze[900], errlog[900], out[8192], sha[80], logp[1200], auditdir[900];
  static char io[1 << 20];
  size_t have = 0;
  pj_proc p;
  int n;
  pt_init(argc, argv);
  if (argc < 8) return 2;
  g_host = argv[2]; /* the unguarded test build: this test's scratch folder is not a protected place */
  g_lib = argv[3];
  g_fake = argv[4];
  g_check = argv[5];
  g_guarded = argv[6]; /* the real host, for the refusals */
  g_marker = argv[7];  /* a library that leaves a file behind when it loads */
  pt_newdir(g_dir, sizeof(g_dir), "host");
  snprintf(tok, sizeof(tok), "%s/grok.token", g_dir);
  snprintf(allow, sizeof(allow), "%s/allow.json", g_dir);
  snprintf(maze, sizeof(maze), "%s/maze.txt", g_dir);
  snprintf(errlog, sizeof(errlog), "%s/host-stderr.txt", g_dir);
  snprintf(auditdir, sizeof(auditdir), "%s/audit", g_dir);
  pt_mkdir(auditdir);
  pt_write_file(maze, "#####\n#...#\n#.S.#\n#...#\n#####\n", 30);
  {
    const char* a[] = {g_host, "token-new", "--out", tok};
    char* t;
    char want[65];
    n = run(a, 4, out, sizeof(out), errlog);
    CHECK(n >= 64, "token-new prints a hash");
    memcpy(sha, out, 64);
    sha[64] = 0;
    t = pt_read_file(tok, NULL);
    CHECK(t && strlen(t) == 64, "token file holds a 64-character token");
    if (t) {
      pt_token_sha_hex(t, want);
      CHECK(strcmp(want, sha) == 0, "the printed hash is the token's SHA-256");
      CHECK(strstr(out, t) == NULL, "the token itself is never printed");
      free(t);
    }
    n = run(a, 4, out, sizeof(out), errlog);
    CHECK(n == 0, "token-new never overwrites an existing file");
  }
  {
    const char* a[] = {g_host, "allow-add", "--allowlist", allow, "--client", "grok", "--token-file", tok,
                       "--max-mode", "advisory", "--rate", "64", "--burst", "64"};
    const char* g[] = {g_host, "grant", "--allowlist", allow, "--client", "grok", "observe", "propose", "proposal_status"};
    const char* bad[] = {g_host, "grant", "--allowlist", allow, "--client", "grok", "hold"};
    const char* show[] = {g_host, "show", "--allowlist", allow};
    run(a, 14, out, sizeof(out), errlog);
    CHECK(strstr(out, "added grok") != NULL, "allow-add");
    run(g, 9, out, sizeof(out), errlog);
    CHECK(strstr(out, "granted to grok") != NULL, "grant");
    n = run(bad, 7, out, sizeof(out), errlog);
    CHECK(n == 0, "an always-on tool cannot be granted");
    run(show, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, sha) && strstr(out, "\"observe\"") && strstr(out, "\"propose\""), "show lists the hash and grants");
  }
  /* allow-add refuses a malformed number or name before it writes anything (review 2026-11-16, L8) */
  {
    static const char* const bad[][2] = {{"--rate", "5x"},      {"--rate", "-1"},          {"--rate", "4294967297"},
                                         {"--rate", ""},        {"--rate", "0"},           {"--burst", "513"},
                                         {"--burst", " 16"},    {"--transport", "http"},   {"--max-mode", "actuate"}};
    const char* show[] = {g_host, "show", "--allowlist", allow};
    char* before = pt_read_file(allow, NULL);
    char* after;
    size_t k;
    int quiet = 1;
    for (k = 0; k < sizeof(bad) / sizeof(bad[0]); ++k) {
      const char* a[] = {g_host, "allow-add", "--allowlist", allow, "--client", "gem", "--token-file", tok, bad[k][0], bad[k][1]};
      run(a, 10, out, sizeof(out), errlog);
      if (strstr(out, "added")) {
        quiet = 0;
        fprintf(stderr, "  allow-add took %s %s\n", bad[k][0], bad[k][1]);
      }
    }
    CHECK(quiet, "allow-add refuses malformed numbers, transports and modes");
    after = pt_read_file(allow, NULL);
    CHECK(before && after && !strcmp(before, after), "the allowlist is unchanged after refused allow-adds");
    free(before);
    free(after);
    run(show, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "\"gem\"") == NULL, "no half-made client entry");
  }
  {
    const char* v[] = {g_host, "version", "--lib", g_lib};
    run(v, 4, out, sizeof(out), errlog);
    CHECK(strstr(out, "peachy 0.1.0") && strstr(out, "alpha"), "version through the library");
  }
  /* serve */
  {
    const char* a[] = {g_host, "serve", "--lib", g_lib, "--allowlist", allow, "--audit-dir", auditdir, "--client", "grok",
                       "--token-file", tok, "--session", "hosttest", "--mode", "advisory", "--game-exe", g_fake,
                       "--game-arg", "--maze", "--game-arg", maze};
    char* reply = io + sizeof(io) / 2;
    memset(&p, 0, sizeof(p));
    CHECK(pj_proc_spawn(&p, a, 22, NULL, errlog) == 0, "serve starts");
    CHECK(ask(&p, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}", io,
              sizeof(io), &have) == 0 &&
              strstr(reply, "\"protocolVersion\":\"2025-11-25\""),
          "initialize over stdio");
    {
      const char* note = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n";
      pj_proc_write(&p, note, strlen(note));
    }
    CHECK(ask(&p, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}", io, sizeof(io), &have) == 0 &&
              strstr(reply, "\"propose\"") && !strstr(reply, "\"plan\""),
          "tools/list over stdio shows the grants");
    CHECK(ask(&p, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"observe\",\"arguments\":{\"radius\":1}}}",
              io, sizeof(io), &have) == 0 &&
              strstr(reply, "\"legal_moves\":[\"north\",\"east\",\"south\",\"west\"]"),
          "observe over stdio");
    CHECK(ask(&p, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
                  "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\"}]}}}",
              io, sizeof(io), &have) == 0 &&
              strstr(reply, "\"disposition\":\"advisory\"") && strstr(reply, "\"ran\":false"),
          "propose over stdio: advisory, nothing ran");
    pj_proc_stop(&p, 10000);
  }
  {
    char* e = pt_read_file(errlog, NULL);
    char* t = pt_read_file(tok, NULL);
    CHECK(e && t && !strstr(e, t), "the token never appears on stderr");
    free(e);
    free(t);
  }
  CHECK(find_log(auditdir, logp, sizeof(logp)) == 0, "serve wrote an audit log");
  {
    const char* a[] = {g_check, logp};
    run(a, 2, out, sizeof(out), errlog);
    CHECK(strncmp(out, "OK ", 3) == 0, "peachy_auditcheck: the log is whole");
  }
  {
    size_t len;
    char* t = pt_read_file(logp, &len);
    char bad[800];
    const char* a[] = {g_check, bad};
    snprintf(bad, sizeof(bad), "%s/tampered.log", g_dir);
    if (t) {
      char* q = strstr(t, "\"verdict\":\"ok\"");
      if (q) q[11] = 'x';
      pt_write_file(bad, t, len);
      free(t);
    }
    run(a, 2, out, sizeof(out), errlog);
    CHECK(strncmp(out, "BROKEN", 6) == 0, "peachy_auditcheck: a tampered log is caught");
  }
  /* a wrong token is refused and nothing is served */
  {
    char badtok[700];
    const char* a[] = {g_host, "serve", "--lib", g_lib, "--allowlist", allow, "--audit-dir", auditdir, "--client", "grok",
                       "--token-file", badtok};
    snprintf(badtok, sizeof(badtok), "%s/bad.token", g_dir);
    pt_write_file(badtok, "not-the-right-token-at-all-0000000000", 37);
    n = run(a, 12, out, sizeof(out), errlog);
    CHECK(n == 0, "wrong token: nothing served");
    {
      char* e = pt_read_file(errlog, NULL);
      CHECK(e && strstr(e, "session refused") && !strstr(e, "not-the-right-token"), "wrong token: refused, token not echoed");
      free(e);
    }
  }
  guarded(tok, allow, auditdir, errlog);
  install_flow(tok, allow, auditdir, errlog);
  return pt_done("peachy_host_e2e");
}
