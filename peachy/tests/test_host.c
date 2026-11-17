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

static const char *g_host, *g_lib, *g_fake, *g_check, *g_guarded;
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

int main(int argc, char** argv) {
  char tok[900], allow[900], maze[900], errlog[900], out[8192], sha[80], logp[1200], auditdir[900];
  static char io[1 << 20];
  size_t have = 0;
  pj_proc p;
  int n;
  pt_init(argc, argv);
  if (argc < 7) return 2;
  g_host = argv[2]; /* the unguarded test build: this test's scratch folder is not a protected place */
  g_lib = argv[3];
  g_fake = argv[4];
  g_check = argv[5];
  g_guarded = argv[6]; /* the real host, for the refusals */
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
  return pt_done("peachy_host_e2e");
}
