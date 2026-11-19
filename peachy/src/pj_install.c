/* Peachy install folder: manifest parsing and the checked open. MIT license. See pj_install.h. */
#if !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE 1
#endif
#include "pj_install.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pj_base.h"
#include "pj_platform.h"
#include "pj_sha256.h"

static int refuse(char* why, size_t cap, const char* s) {
  if (why && cap) pj_copy_cstr(why, cap, s);
  return -1;
}

int pj_install_name_ok(const char* s) {
  size_t i;
  if (!s || !s[0] || s[0] == '.' || s[0] == '-') return 0;
  for (i = 0; s[i]; ++i) {
    char c = s[i];
    if (i >= PJ_INSTALL_NAME_MAX) return 0;
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
          c == '-'))
      return 0;
  }
  return 1;
}

static int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

int pj_install_parse(const char* buf, size_t n, pj_install_manifest* m) {
  static const char head[] = "PJINSTALL1\n";
  size_t pos = sizeof(head) - 1;
  if (!buf || !m || n > PJ_INSTALL_MANIFEST_MAX || n < pos || memcmp(buf, head, pos) != 0) return -1;
  memset(m, 0, sizeof(*m));
  while (pos < n) {
    pj_install_entry* e;
    size_t i, k, start;
    uint64_t v = 0;
    if (m->n >= PJ_INSTALL_MAX_FILES || n - pos < 64 + 2 + 1 + 2 + 1 + 1) return -1;
    e = &m->e[m->n];
    for (i = 0; i < 32; ++i) {
      int hi = hexval(buf[pos + 2 * i]), lo = hexval(buf[pos + 2 * i + 1]);
      if (hi < 0 || lo < 0) return -1;
      e->sha256[i] = (uint8_t)(hi * 16 + lo);
    }
    pos += 64;
    if (buf[pos] != ' ' || buf[pos + 1] != ' ') return -1;
    pos += 2;
    start = pos;
    while (pos < n && buf[pos] >= '0' && buf[pos] <= '9') {
      if (pos - start >= 20) return -1;
      v = v * 10u + (uint64_t)(buf[pos] - '0');
      if (v > PJ_INSTALL_FILE_MAX) return -1;
      ++pos;
    }
    if (pos == start || (buf[start] == '0' && pos - start > 1)) return -1; /* no leading zeros */
    e->size = v;
    if (pos + 2 > n || buf[pos] != ' ' || buf[pos + 1] != ' ') return -1;
    pos += 2;
    for (k = 0; pos < n && buf[pos] != '\n'; ++pos, ++k) {
      if (k >= PJ_INSTALL_NAME_MAX) return -1;
      e->name[k] = buf[pos];
    }
    if (pos >= n) return -1; /* every line ends with a newline */
    e->name[k] = 0;
    ++pos;
    if (strlen(e->name) != k || !pj_install_name_ok(e->name) || !strcmp(e->name, PJ_INSTALL_MANIFEST)) return -1;
    if (m->n > 0 && strcmp(m->e[m->n - 1].name, e->name) >= 0) return -1; /* strictly ascending: canonical, unique */
    ++m->n;
  }
  return m->n > 0 ? 0 : -1;
}

size_t pj_install_format(const pj_install_manifest* m, char* out, size_t cap) {
  static const char* digits = "0123456789abcdef";
  size_t o = 0;
  uint32_t i;
  int k;
  if (!m || !out || m->n == 0 || m->n > PJ_INSTALL_MAX_FILES || cap < 12) return 0;
  memcpy(out, "PJINSTALL1\n", 11);
  o = 11;
  for (i = 0; i < m->n; ++i) {
    const pj_install_entry* e = &m->e[i];
    char num[24];
    int nl = snprintf(num, sizeof(num), "%llu", (unsigned long long)e->size);
    size_t name_len = strlen(e->name);
    if (!pj_install_name_ok(e->name) || nl <= 0 || e->size > PJ_INSTALL_FILE_MAX) return 0;
    if (i > 0 && strcmp(m->e[i - 1].name, e->name) >= 0) return 0;
    if (o + 64 + 2 + (size_t)nl + 2 + name_len + 1 + 1 > cap || o + 64 + 2 + (size_t)nl + 2 + name_len + 1 > PJ_INSTALL_MANIFEST_MAX)
      return 0;
    for (k = 0; k < 32; ++k) {
      out[o++] = digits[e->sha256[k] >> 4];
      out[o++] = digits[e->sha256[k] & 15];
    }
    out[o++] = ' ';
    out[o++] = ' ';
    memcpy(out + o, num, (size_t)nl);
    o += (size_t)nl;
    out[o++] = ' ';
    out[o++] = ' ';
    memcpy(out + o, e->name, name_len);
    o += name_len;
    out[o++] = '\n';
  }
  out[o] = 0;
  return o;
}

int pj_install_add(pj_install_manifest* m, const char* name, const uint8_t sha[32], uint64_t size) {
  uint32_t i, at;
  if (!m || !pj_install_name_ok(name) || !strcmp(name, PJ_INSTALL_MANIFEST) || m->n >= PJ_INSTALL_MAX_FILES ||
      size > PJ_INSTALL_FILE_MAX)
    return -1;
  for (at = 0; at < m->n; ++at) {
    int c = strcmp(m->e[at].name, name);
    if (c == 0) return -1;
    if (c > 0) break;
  }
  for (i = m->n; i > at; --i) m->e[i] = m->e[i - 1];
  memset(&m->e[at], 0, sizeof(m->e[at]));
  pj_copy_cstr(m->e[at].name, sizeof(m->e[at].name), name);
  memcpy(m->e[at].sha256, sha, 32);
  m->e[at].size = size;
  ++m->n;
  return 0;
}

const pj_install_entry* pj_install_find(const pj_install_manifest* m, const char* name) {
  uint32_t i;
  if (!m || !name) return NULL;
  for (i = 0; i < m->n; ++i)
    if (!strcmp(m->e[i].name, name)) return &m->e[i];
  return NULL;
}

/* The last path element (after '/' or '\'). */
static const char* base_name(const char* p) {
  const char* b = p;
  for (; *p; ++p)
    if (*p == '/' || *p == '\\') b = p + 1;
  return b;
}

/* ------------------------------------------------------------------------------------------------ platform */
#if defined(_WIN32)
#include <windows.h>

static wchar_t* widen_path(const char* s) {
  int n, i;
  wchar_t* w;
  if (!s) return NULL;
  n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
  if (n <= 0 || n > 32768) return NULL;
  w = (wchar_t*)malloc((size_t)n * sizeof(wchar_t));
  if (!w) return NULL;
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n) != n) {
    free(w);
    return NULL;
  }
  for (i = 0; i < n; ++i)
    if (w[i] == L'/') w[i] = L'\\';
  return w;
}

/* The final path of an open handle (links, junctions and short names resolved), without the \\?\ prefix. */
static int final_path(HANDLE h, wchar_t* out, DWORD cap) {
  DWORD n = GetFinalPathNameByHandleW(h, out, cap, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  if (n == 0 || n >= cap) return -1;
  if (n >= 4 && !wcsncmp(out, L"\\\\?\\", 4)) memmove(out, out + 4, (n - 4 + 1) * sizeof(wchar_t));
  return 0;
}

static int hash_handle(HANDLE h, uint8_t sha[32], uint64_t* size) {
  pj_sha256 c;
  LARGE_INTEGER sz;
  uint64_t done = 0;
  uint8_t* buf;
  const DWORD chunk = 1u << 20;
  if (!GetFileSizeEx(h, &sz) || sz.QuadPart < 0) return -1;
  buf = (uint8_t*)malloc(chunk);
  if (!buf) return -1;
  pj_sha256_init(&c);
  while (done < (uint64_t)sz.QuadPart) {
    OVERLAPPED o;
    DWORD want = (uint64_t)sz.QuadPart - done > chunk ? chunk : (DWORD)((uint64_t)sz.QuadPart - done), got = 0;
    memset(&o, 0, sizeof(o));
    o.Offset = (DWORD)done;
    o.OffsetHigh = (DWORD)(done >> 32);
    if (!ReadFile(h, buf, want, &got, &o) || got == 0) {
      free(buf);
      return -1;
    }
    pj_sha256_update(&c, buf, got);
    done += got;
  }
  free(buf);
  pj_sha256_final(&c, sha);
  *size = done;
  return 0;
}

static HANDLE open_checked(const wchar_t* w) {
  /* readers may share; writers and deletion may not, for as long as the handle is open */
  return CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
}

int pj_install_open_verified(const char* install_dir, const char* path, const pj_install_manifest* m,
                             pj_held_file* held, char* why, size_t why_cap) {
  wchar_t *wd = NULL, *wp = NULL;
  wchar_t* fd_path = (wchar_t*)malloc(32768 * sizeof(wchar_t));
  wchar_t* fp_path = (wchar_t*)malloc(32768 * sizeof(wchar_t));
  HANDLE hd = INVALID_HANDLE_VALUE, hf = INVALID_HANDLE_VALUE;
  BY_HANDLE_FILE_INFORMATION fi;
  const pj_install_entry* e;
  const char* name = path ? base_name(path) : NULL;
  uint8_t sha[32];
  uint64_t size = 0;
  size_t dl;
  int rc = -1;
  if (!held || !fd_path || !fp_path) {
    refuse(why, why_cap, "out of memory");
    goto out;
  }
  held->h = -1;
  held->path[0] = 0;
  if (!install_dir || !path) {
    refuse(why, why_cap, "no install folder or no file");
    goto out;
  }
  e = pj_install_find(m, name);
  if (!e) {
    refuse(why, why_cap, "it is not listed in the install manifest");
    goto out;
  }
  wd = widen_path(install_dir);
  wp = widen_path(path);
  if (!wd || !wp) goto bad_name;
  hd = CreateFileW(wd, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                   FILE_FLAG_BACKUP_SEMANTICS, NULL);
  if (hd == INVALID_HANDLE_VALUE || final_path(hd, fd_path, 32768) != 0) {
    refuse(why, why_cap, "the install folder cannot be opened");
    goto out;
  }
  hf = open_checked(wp);
  if (hf == INVALID_HANDLE_VALUE) {
    refuse(why, why_cap, "it cannot be opened without sharing write access");
    goto out;
  }
  if (!GetFileInformationByHandle(hf, &fi) ||
      (fi.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) || fi.nNumberOfLinks != 1) {
    refuse(why, why_cap, "it is not an ordinary file with a single name");
    goto out;
  }
  if (final_path(hf, fp_path, 32768) != 0) {
    refuse(why, why_cap, "its real location cannot be read");
    goto out;
  }
  /* the real location must be <install folder>\<name>, compared without regard to case as Windows does */
  dl = wcslen(fd_path);
  while (dl > 0 && fd_path[dl - 1] == L'\\') --dl;
  {
    wchar_t wname[PJ_INSTALL_NAME_MAX + 1];
    int wn = MultiByteToWideChar(CP_UTF8, 0, name, -1, wname, PJ_INSTALL_NAME_MAX + 1);
    if (wn <= 0 || wcslen(fp_path) != dl + 1 + wcslen(wname) ||
        CompareStringOrdinal(fp_path, (int)dl, fd_path, (int)dl, TRUE) != CSTR_EQUAL || fp_path[dl] != L'\\' ||
        CompareStringOrdinal(fp_path + dl + 1, -1, wname, -1, TRUE) != CSTR_EQUAL) {
      refuse(why, why_cap, "it is not directly inside the install folder");
      goto out;
    }
  }
  if (hash_handle(hf, sha, &size) != 0) {
    refuse(why, why_cap, "it cannot be read");
    goto out;
  }
  if (size != e->size || !pj_ct_equal(sha, e->sha256, 32)) {
    refuse(why, why_cap, "its SHA-256 or size does not match the install manifest");
    goto out;
  }
  /* what loads is the checked location itself: absolute, backslashes, links resolved */
  if (WideCharToMultiByte(CP_UTF8, 0, fp_path, -1, held->path, (int)sizeof(held->path), NULL, NULL) <= 0) {
    held->path[0] = 0;
    refuse(why, why_cap, "its real location is too long");
    goto out;
  }
  held->h = (intptr_t)hf;
  hf = INVALID_HANDLE_VALUE;
  rc = 0;
  goto out;
bad_name:
  refuse(why, why_cap, "the path is not valid UTF-8");
out:
  if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
  if (hd != INVALID_HANDLE_VALUE) CloseHandle(hd);
  free(wd);
  free(wp);
  free(fd_path);
  free(fp_path);
  return rc;
}

void pj_held_close(pj_held_file* held) {
  if (held && held->h != -1) {
    CloseHandle((HANDLE)held->h);
    held->h = -1;
  }
}

int pj_install_hash_file(const char* path, uint8_t sha[32], uint64_t* size) {
  wchar_t* w = widen_path(path);
  HANDLE h;
  int rc;
  if (!w) return -1;
  h = open_checked(w);
  free(w);
  if (h == INVALID_HANDLE_VALUE) return -1;
  rc = hash_handle(h, sha, size);
  CloseHandle(h);
  return rc;
}

int pj_install_check_contents(const char* install_dir, const pj_install_manifest* m, char* why, size_t why_cap) {
  char pat[1100];
  wchar_t* wpat;
  WIN32_FIND_DATAW fd;
  HANDLE h;
  uint32_t seen = 0;
  int rc = 0, have_manifest = 0;
  if (snprintf(pat, sizeof(pat), "%s/*", install_dir) >= (int)sizeof(pat)) return refuse(why, why_cap, "path too long");
  wpat = widen_path(pat);
  if (!wpat) return refuse(why, why_cap, "the path is not valid UTF-8");
  h = FindFirstFileW(wpat, &fd);
  free(wpat);
  if (h == INVALID_HANDLE_VALUE) return refuse(why, why_cap, "the install folder cannot be listed");
  do {
    char name[300];
    if (!wcscmp(fd.cFileName, L".") || !wcscmp(fd.cFileName, L"..")) continue;
    if (WideCharToMultiByte(CP_UTF8, 0, fd.cFileName, -1, name, (int)sizeof(name), NULL, NULL) <= 0 ||
        (fd.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT))) {
      rc = refuse(why, why_cap, "the install folder holds a folder, link or unreadable name");
      break;
    }
    if (!strcmp(name, PJ_INSTALL_MANIFEST)) {
      have_manifest = 1;
      continue;
    }
    if (!pj_install_find(m, name)) {
      if (why && why_cap) snprintf(why, why_cap, "a file not in the manifest: %.64s", name);
      rc = -1;
      break;
    }
    ++seen;
  } while (FindNextFileW(h, &fd));
  FindClose(h);
  if (rc == 0 && (!have_manifest || seen != m->n)) rc = refuse(why, why_cap, "a file in the manifest is missing");
  return rc;
}

#else /* POSIX */
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

static int hash_fd(int fd, uint8_t sha[32], uint64_t* size) {
  pj_sha256 c;
  uint8_t* buf = (uint8_t*)malloc(1u << 20);
  uint64_t done = 0;
  if (!buf) return -1;
  pj_sha256_init(&c);
  for (;;) {
    ssize_t n = pread(fd, buf, 1u << 20, (off_t)done);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) {
      free(buf);
      return -1;
    }
    if (n == 0) break;
    pj_sha256_update(&c, buf, (size_t)n);
    done += (uint64_t)n;
  }
  free(buf);
  pj_sha256_final(&c, sha);
  *size = done;
  return 0;
}

int pj_install_open_verified(const char* install_dir, const char* path, const pj_install_manifest* m,
                             pj_held_file* held, char* why, size_t why_cap) {
  char rd[PATH_MAX], rp[PATH_MAX];
  struct stat sf, sr;
  const pj_install_entry* e;
  const char* name = path ? base_name(path) : NULL;
  uint8_t sha[32];
  uint64_t size = 0;
  size_t dl;
  int fd;
  if (!held) return refuse(why, why_cap, "no place to hold the file");
  held->h = -1;
  held->path[0] = 0;
  if (!install_dir || !path) return refuse(why, why_cap, "no install folder or no file");
  e = pj_install_find(m, name);
  if (!e) return refuse(why, why_cap, "it is not listed in the install manifest");
  if (!realpath(install_dir, rd)) return refuse(why, why_cap, "the install folder cannot be resolved");
  fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) return refuse(why, why_cap, "it cannot be opened (or it is a link)");
  if (fstat(fd, &sf) != 0 || !S_ISREG(sf.st_mode) || sf.st_nlink != 1) {
    close(fd);
    return refuse(why, why_cap, "it is not an ordinary file with a single name");
  }
  if (!realpath(path, rp) || stat(rp, &sr) != 0 || sr.st_dev != sf.st_dev || sr.st_ino != sf.st_ino) {
    close(fd);
    return refuse(why, why_cap, "its real location cannot be confirmed");
  }
  dl = strlen(rd);
  while (dl > 1 && rd[dl - 1] == '/') --dl;
  if (strncmp(rp, rd, dl) != 0 || rp[dl] != '/' || strcmp(rp + dl + 1, e->name) != 0) {
    close(fd);
    return refuse(why, why_cap, "it is not directly inside the install folder");
  }
  if (hash_fd(fd, sha, &size) != 0) {
    close(fd);
    return refuse(why, why_cap, "it cannot be read");
  }
  if (size != e->size || !pj_ct_equal(sha, e->sha256, 32)) {
    close(fd);
    return refuse(why, why_cap, "its SHA-256 or size does not match the install manifest");
  }
  held->h = fd;
  snprintf(held->path, sizeof(held->path), "%.*s/%s", (int)dl, rd, e->name);
  return 0;
}

void pj_held_close(pj_held_file* held) {
  if (held && held->h != -1) {
    close((int)held->h);
    held->h = -1;
  }
}

int pj_install_hash_file(const char* path, uint8_t sha[32], uint64_t* size) {
  int fd = open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW), rc;
  if (fd < 0) return -1;
  rc = hash_fd(fd, sha, size);
  close(fd);
  return rc;
}

int pj_install_check_contents(const char* install_dir, const pj_install_manifest* m, char* why, size_t why_cap) {
  DIR* d = opendir(install_dir);
  struct dirent* de;
  uint32_t seen = 0;
  int rc = 0, have_manifest = 0;
  if (!d) return refuse(why, why_cap, "the install folder cannot be listed");
  while ((de = readdir(d)) != NULL) {
    char full[1200];
    struct stat st;
    if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
    snprintf(full, sizeof(full), "%s/%s", install_dir, de->d_name);
    if (lstat(full, &st) != 0 || !S_ISREG(st.st_mode)) {
      rc = refuse(why, why_cap, "the install folder holds a folder, link or special file");
      break;
    }
    if (!strcmp(de->d_name, PJ_INSTALL_MANIFEST)) {
      have_manifest = 1;
      continue;
    }
    if (!pj_install_find(m, de->d_name)) {
      if (why && why_cap) snprintf(why, why_cap, "a file not in the manifest: %.64s", de->d_name);
      rc = -1;
      break;
    }
    ++seen;
  }
  closedir(d);
  if (rc == 0 && (!have_manifest || seen != m->n)) rc = refuse(why, why_cap, "a file in the manifest is missing");
  return rc;
}
#endif

int pj_install_read_manifest(const char* install_dir, pj_install_manifest* m, char* why, size_t why_cap) {
  char p[1100];
  FILE* f;
  char* buf;
  size_t n;
  int rc;
  if (!install_dir || snprintf(p, sizeof(p), "%s/%s", install_dir, PJ_INSTALL_MANIFEST) >= (int)sizeof(p))
    return refuse(why, why_cap, "no install folder");
  f = pj_fopen(p, "rb");
  if (!f) return refuse(why, why_cap, "the install manifest cannot be read");
  buf = (char*)malloc(PJ_INSTALL_MANIFEST_MAX + 1);
  if (!buf) {
    fclose(f);
    return refuse(why, why_cap, "out of memory");
  }
  n = fread(buf, 1, PJ_INSTALL_MANIFEST_MAX + 1, f);
  fclose(f);
  if (n > PJ_INSTALL_MANIFEST_MAX) rc = refuse(why, why_cap, "the install manifest is too large");
  else if (pj_install_parse(buf, n, m) != 0) rc = refuse(why, why_cap, "the install manifest is malformed");
  else rc = 0;
  free(buf);
  return rc;
}
