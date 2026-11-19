/* The install folder: manifest parsing and the checked open (review 2026-11-19, H-1). MIT license. */
#include "pt.h"
#include "pj_install.h"
#include "pj_platform.h"

#if !defined(_WIN32)
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

static void sha_of(const char* data, size_t n, uint8_t out[32]) { pj_sha256_once(data, n, out); }

static void manifest_text(void) {
  pj_install_manifest m, m2;
  uint8_t h[32];
  char out[PJ_INSTALL_MANIFEST_MAX + 1];
  size_t n;
  memset(&m, 0, sizeof(m));
  sha_of("b", 1, h);
  CHECK(pj_install_add(&m, "b.dll", h, 1) == 0, "add b");
  sha_of("a", 1, h);
  CHECK(pj_install_add(&m, "a.exe", h, 1) == 0, "add a (sorted in front)");
  CHECK(pj_install_add(&m, "a.exe", h, 1) != 0, "a duplicate name is refused");
  CHECK(pj_install_add(&m, "../evil.dll", h, 1) != 0 && pj_install_add(&m, "sub/x.dll", h, 1) != 0 &&
            pj_install_add(&m, ".hidden", h, 1) != 0 && pj_install_add(&m, "", h, 1) != 0 &&
            pj_install_add(&m, PJ_INSTALL_MANIFEST, h, 1) != 0 && pj_install_add(&m, "a b.dll", h, 1) != 0,
        "names with separators, dots in front, spaces, the manifest's own name or nothing are refused");
  n = pj_install_format(&m, out, sizeof(out));
  CHECK(n > 0 && !strncmp(out, "PJINSTALL1\n", 11), "formatted");
  CHECK(pj_install_parse(out, n, &m2) == 0 && m2.n == 2 && !strcmp(m2.e[0].name, "a.exe") && m2.e[1].size == 1,
        "round trip");
  {
    char bad[PJ_INSTALL_MANIFEST_MAX + 64];
    size_t k;
    CHECK(pj_install_parse(out, n - 1, &m2) != 0, "a missing final newline is refused");
    memcpy(bad, out, n);
    bad[11] = 'A'; /* upper-case hex */
    CHECK(pj_install_parse(bad, n, &m2) != 0, "upper-case hex is refused (canonical form only)");
    /* swap the two lines: not ascending */
    {
      const char* l2 = strchr(out + 11, '\n') + 1;
      size_t a = (size_t)(l2 - (out + 11)), b = n - 11 - a;
      memcpy(bad, out, 11);
      memcpy(bad + 11, l2, b);
      memcpy(bad + 11 + b, out + 11, a);
      CHECK(pj_install_parse(bad, n, &m2) != 0, "names out of order are refused");
    }
    memcpy(bad, out, n);
    for (k = 11 + 64 + 2; bad[k] != ' '; ++k) {
    }
    memmove(bad + 11 + 64 + 3, bad + 11 + 64 + 2, n - (11 + 64 + 2));
    bad[11 + 64 + 2] = '0';
    CHECK(pj_install_parse(bad, n + 1, &m2) != 0, "a size with a leading zero is refused");
    CHECK(pj_install_parse("PJINSTALL1\n", 11, &m2) != 0, "an empty manifest is refused");
    CHECK(pj_install_parse("PJINSTALL2\n", 11, &m2) != 0, "another version is refused");
  }
}

static void write_bytes(const char* path, const char* data, size_t n) {
  remove(path);
  pt_write_file(path, data, n);
}

static void checked_open(void) {
  char dir[600], other[700], p[800], q[800], why[160];
  pj_install_manifest m;
  pj_held_file h;
  uint8_t sh[32];
  pt_newdir(dir, sizeof(dir), "install");
  snprintf(other, sizeof(other), "%s/elsewhere", dir);
  pt_mkdir(other);
  snprintf(p, sizeof(p), "%s/inst", dir);
  pt_mkdir(p);
  memcpy(dir, p, strlen(p) + 1); /* the install folder */
  memset(&m, 0, sizeof(m));
  snprintf(p, sizeof(p), "%s/game.exe", dir);
  write_bytes(p, "GAME-BYTES", 10);
  sha_of("GAME-BYTES", 10, sh);
  pj_install_add(&m, "game.exe", sh, 10);
  snprintf(p, sizeof(p), "%s/peachy.dll", dir);
  write_bytes(p, "LIB-BYTES", 9);
  sha_of("LIB-BYTES", 9, sh);
  pj_install_add(&m, "peachy.dll", sh, 9);
  {
    char t[1024];
    size_t n = pj_install_format(&m, t, sizeof(t));
    snprintf(q, sizeof(q), "%s/%s", dir, PJ_INSTALL_MANIFEST);
    write_bytes(q, t, n);
  }
  {
    pj_install_manifest r;
    CHECK(pj_install_read_manifest(dir, &r, why, sizeof(why)) == 0 && r.n == 2, "manifest read back");
  }
  CHECK(pj_install_check_contents(dir, &m, why, sizeof(why)) == 0, "the folder holds exactly the manifest's files");
  CHECK(pj_install_open_verified(dir, p, &m, &h, why, sizeof(why)) == 0 && h.h != -1, "a matching file is accepted");
#if defined(_WIN32)
  {
    /* while held, nobody can write it, replace it or delete it */
    wchar_t w[800];
    HANDLE x;
    MultiByteToWideChar(CP_UTF8, 0, p, -1, w, 800);
    x = CreateFileW(w, GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
    CHECK(x == INVALID_HANDLE_VALUE, "Windows: a held file cannot be opened for writing");
    if (x != INVALID_HANDLE_VALUE) CloseHandle(x);
    CHECK(!DeleteFileW(w), "Windows: a held file cannot be deleted");
    /* and it still loads: execute access shares with the held reader */
    x = CreateFileW(w, GENERIC_READ | GENERIC_EXECUTE, FILE_SHARE_READ | FILE_SHARE_DELETE, NULL, OPEN_EXISTING, 0, NULL);
    CHECK(x != INVALID_HANDLE_VALUE, "Windows: a held file can still be opened to run");
    if (x != INVALID_HANDLE_VALUE) CloseHandle(x);
  }
#endif
  pj_held_close(&h);
  /* one changed byte */
  write_bytes(p, "LIB-BYTEZ", 9);
  CHECK(pj_install_open_verified(dir, p, &m, &h, why, sizeof(why)) != 0 && strstr(why, "does not match") && h.h == -1,
        "a changed byte is refused");
  write_bytes(p, "LIB-BYTES!", 10);
  CHECK(pj_install_open_verified(dir, p, &m, &h, why, sizeof(why)) != 0 && strstr(why, "does not match"),
        "an extra byte is refused");
  write_bytes(p, "LIB-BYTES", 9);
  /* the same bytes under the same name in another folder */
  snprintf(q, sizeof(q), "%s/peachy.dll", other);
  write_bytes(q, "LIB-BYTES", 9);
  CHECK(pj_install_open_verified(dir, q, &m, &h, why, sizeof(why)) != 0 && strstr(why, "not directly inside"),
        "a matching copy outside the install folder is refused");
  /* a name the manifest does not list */
  snprintf(q, sizeof(q), "%s/planner.dll", dir);
  write_bytes(q, "X", 1);
  CHECK(pj_install_open_verified(dir, q, &m, &h, why, sizeof(why)) != 0 && strstr(why, "not listed"),
        "a file the manifest does not list is refused");
  CHECK(pj_install_check_contents(dir, &m, why, sizeof(why)) != 0 && strstr(why, "planner.dll"),
        "a stray file in the install folder is refused by name");
  remove(q);
  /* a missing file */
  snprintf(q, sizeof(q), "%s/game.exe", dir);
  remove(q);
  CHECK(pj_install_check_contents(dir, &m, why, sizeof(why)) != 0 && strstr(why, "missing"), "a missing file is refused");
  write_bytes(q, "GAME-BYTES", 10);
#if !defined(_WIN32)
  /* a symlink in place of the file, and a second name for it */
  {
    char real[800];
    snprintf(real, sizeof(real), "%s/peachy.real", other);
    write_bytes(real, "LIB-BYTES", 9);
    remove(p);
    CHECK(symlink(real, p) == 0, "make a symlink");
    CHECK(pj_install_open_verified(dir, p, &m, &h, why, sizeof(why)) != 0, "a symlink in the install folder is refused");
    CHECK(pj_install_check_contents(dir, &m, why, sizeof(why)) != 0, "and the folder check refuses it too");
    remove(p);
    write_bytes(p, "LIB-BYTES", 9);
    snprintf(q, sizeof(q), "%s/second-name.dll", other);
    remove(q);
    CHECK(link(p, q) == 0, "make a hard link");
    CHECK(pj_install_open_verified(dir, p, &m, &h, why, sizeof(why)) != 0 && strstr(why, "single name"),
          "a file with a second name somewhere else is refused");
    remove(q);
    /* a path through a symlinked folder that points into the install folder resolves to the real location */
    snprintf(q, sizeof(q), "%s/via", other);
    remove(q);
    {
      char absdir[4096];
      CHECK(realpath(dir, absdir) != NULL && symlink(absdir, q) == 0, "make a folder symlink");
    }
    snprintf(q, sizeof(q), "%s/via/peachy.dll", other);
    CHECK(pj_install_open_verified(dir, q, &m, &h, why, sizeof(why)) == 0 && strstr(h.path, "/inst/peachy.dll"),
          "a path through a folder link is judged at its real location, and that location is what gets loaded");
    pj_held_close(&h);
  }
#endif
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  manifest_text();
  checked_open();
  return pt_done("peachy_install");
}
