/* Peachy platform layer, Windows. MIT license.
 * The child-process code follows the proven PCHIE MCP stdio client: polled pipes (no reader thread), the
 * child's stderr straight to a file, and a careful stop order (close stdin, WM_CLOSE to the child's own
 * windows, wait; TerminateProcess only on our own child's handle, only after the grace period). */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <process.h>
#include <share.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pj_platform.h"

uint64_t pj_mono_ns(void) {
  LARGE_INTEGER f, c;
  QueryPerformanceFrequency(&f);
  QueryPerformanceCounter(&c);
  return (uint64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart);
}

void pj_sleep_ms(uint32_t ms) { Sleep(ms); }

int pj_os_random(uint8_t* out, size_t n) {
  while (n) {
    ULONG take = n > 0x10000 ? 0x10000 : (ULONG)n;
    if (BCryptGenRandom(NULL, out, take, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) return -1;
    out += take;
    n -= take;
  }
  return 0;
}

void pj_local_time(char* out, size_t cap) {
  SYSTEMTIME lt;
  TIME_ZONE_INFORMATION tz;
  DWORD r = GetTimeZoneInformation(&tz);
  long bias = tz.Bias + (r == TIME_ZONE_ID_DAYLIGHT ? tz.DaylightBias : (r == TIME_ZONE_ID_STANDARD ? tz.StandardBias : 0));
  long off = -bias;
  GetLocalTime(&lt);
  _snprintf_s(out, cap, _TRUNCATE, "%04u-%02u-%02uT%02u:%02u:%02u%c%02ld:%02ld", lt.wYear, lt.wMonth, lt.wDay, lt.wHour,
              lt.wMinute, lt.wSecond, off < 0 ? '-' : '+', (off < 0 ? -off : off) / 60, (off < 0 ? -off : off) % 60);
}

void pj_compact_time(char* out, size_t cap) {
  SYSTEMTIME lt;
  GetLocalTime(&lt);
  _snprintf_s(out, cap, _TRUNCATE, "%04u%02u%02u-%02u%02u%02u", lt.wYear, lt.wMonth, lt.wDay, lt.wHour, lt.wMinute,
              lt.wSecond);
}

uint32_t pj_pid(void) { return (uint32_t)GetCurrentProcessId(); }

static wchar_t* widen(const char* s) {
  int n;
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
  return w;
}

FILE* pj_fopen(const char* p, const char* mode) {
  wchar_t* wp = widen(p);
  wchar_t* wm = widen(mode);
  FILE* f = NULL;
  if (wp && wm) f = _wfsopen(wp, wm, _SH_DENYWR);
  free(wp);
  free(wm);
  return f;
}

FILE* pj_fopen_read_shared(const char* p) {
  wchar_t* wp = widen(p);
  FILE* f = NULL;
  /* Read-only and sharing everything, so the resume check and peachy_auditcheck can re-read an audit log that a
   * session still holds open for writing (that handle denies other writers, which a reader asking to deny writers
   * would collide with). Every other open keeps denying writers. */
  if (wp) f = _wfsopen(wp, L"rb", _SH_DENYNO);
  free(wp);
  return f;
}

int pj_dir_exists(const char* p) {
  wchar_t* wp = widen(p);
  DWORD a;
  if (!wp) return 0;
  a = GetFileAttributesW(wp);
  free(wp);
  return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

int pj_replace_file(const char* tmp, const char* fin) {
  wchar_t* a = widen(tmp);
  wchar_t* b = widen(fin);
  int ok = a && b && MoveFileExW(a, b, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
  free(a);
  free(b);
  return ok ? 0 : -1;
}

void* pj_dl_open(const char* p) {
  wchar_t* wp = widen(p);
  HMODULE h = NULL;
  if (!wp) return NULL;
  /* Absolute paths only; dependencies come from the library's own folder and System32, never the current dir. */
  h = LoadLibraryExW(wp, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  free(wp);
  return (void*)h;
}

pj_fnptr pj_dl_fn(void* lib, const char* name) { return lib ? (pj_fnptr)GetProcAddress((HMODULE)lib, name) : NULL; }

void pj_dl_close(void* lib) {
  if (lib) FreeLibrary((HMODULE)lib);
}

/* Builds a Windows command line with the quoting rules CommandLineToArgvW reverses. */
static wchar_t* build_cmdline(const char* const* argv, uint32_t argc) {
  size_t cap = 2, len = 0, i;
  wchar_t* out;
  uint32_t a;
  for (a = 0; a < argc; ++a) cap += strlen(argv[a]) * 2 + 3;
  if (cap > 32767) return NULL;
  out = (wchar_t*)malloc(cap * sizeof(wchar_t));
  if (!out) return NULL;
  for (a = 0; a < argc; ++a) {
    wchar_t* w = widen(argv[a]);
    size_t wl, bs = 0;
    if (!w) {
      free(out);
      return NULL;
    }
    wl = wcslen(w);
    if (a) out[len++] = L' ';
    out[len++] = L'"';
    for (i = 0; i < wl; ++i) {
      if (w[i] == L'\\') {
        ++bs;
        out[len++] = L'\\';
      } else if (w[i] == L'"') {
        size_t k;
        for (k = 0; k < bs + 1; ++k) out[len++] = L'\\';
        out[len++] = L'"';
        bs = 0;
      } else {
        bs = 0;
        out[len++] = w[i];
      }
    }
    for (i = 0; i < bs; ++i) out[len++] = L'\\';
    out[len++] = L'"';
    free(w);
  }
  out[len] = 0;
  return out;
}

int pj_proc_spawn(pj_proc* p, const char* const* argv, uint32_t argc, const char* workdir, const char* stderr_log) {
  SECURITY_ATTRIBUTES sa;
  HANDLE in_r = NULL, in_w = NULL, out_r = NULL, out_w = NULL, log;
  STARTUPINFOW si;
  PROCESS_INFORMATION pi;
  wchar_t *cl, *wd = NULL, *wl;
  uint32_t i;
  BOOL ok;
  memset(p, 0, sizeof(*p));
  p->fd_in = p->fd_out = -1;
  if (!argv || argc == 0 || argc > 64) return -1;
  for (i = 0; i < argc; ++i)
    if (!argv[i]) return -1;
  memset(&sa, 0, sizeof(sa));
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;
  if (!CreatePipe(&in_r, &in_w, &sa, 1 << 16)) return -1;
  if (!CreatePipe(&out_r, &out_w, &sa, 1 << 20)) {
    CloseHandle(in_r);
    CloseHandle(in_w);
    return -1;
  }
  SetHandleInformation(in_w, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(out_r, HANDLE_FLAG_INHERIT, 0);
  wl = widen(stderr_log ? stderr_log : "NUL");
  log = wl ? CreateFileW(wl, GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL)
           : INVALID_HANDLE_VALUE;
  free(wl);
  cl = build_cmdline(argv, argc);
  if (workdir) wd = widen(workdir);
  if (log == INVALID_HANDLE_VALUE || !cl || (workdir && !wd)) {
    if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
    CloseHandle(in_r); CloseHandle(in_w); CloseHandle(out_r); CloseHandle(out_w);
    free(cl);
    free(wd);
    return -1;
  }
  memset(&si, 0, sizeof(si));
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdInput = in_r;
  si.hStdOutput = out_w;
  si.hStdError = log;
  memset(&pi, 0, sizeof(pi));
  ok = CreateProcessW(NULL, cl, NULL, NULL, TRUE, CREATE_NO_WINDOW, NULL, wd, &si, &pi);
  CloseHandle(in_r);
  CloseHandle(out_w);
  CloseHandle(log);
  free(cl);
  free(wd);
  if (!ok) {
    CloseHandle(in_w);
    CloseHandle(out_r);
    return -1;
  }
  CloseHandle(pi.hThread);
  p->h_proc = pi.hProcess;
  p->h_in = in_w;
  p->h_out = out_r;
  p->pid = (int)pi.dwProcessId;
  p->started = 1;
  return 0;
}

int pj_proc_write(pj_proc* p, const char* data, size_t len) {
  size_t done = 0;
  if (!p->started || !p->h_in) return -1;
  while (done < len) {
    DWORD n = 0, take = (DWORD)((len - done) > 0x100000 ? 0x100000 : (len - done));
    if (!WriteFile((HANDLE)p->h_in, data + done, take, &n, NULL) || n == 0) return -1;
    done += n;
  }
  return 0;
}

int pj_proc_read(pj_proc* p, char* buf, size_t cap, size_t* n, uint32_t timeout_ms) {
  ULONGLONG t0 = GetTickCount64();
  *n = 0;
  if (!p->started || !p->h_out) return -1;
  for (;;) {
    DWORD avail = 0, got = 0;
    if (!PeekNamedPipe((HANDLE)p->h_out, NULL, 0, NULL, &avail, NULL)) return -1;
    if (avail) {
      DWORD take = avail > cap ? (DWORD)cap : avail;
      if (!ReadFile((HANDLE)p->h_out, buf, take, &got, NULL)) return -1;
      *n = got;
      return 0;
    }
    if (GetTickCount64() - t0 >= timeout_ms) return 0;
    Sleep(1);
  }
}

int pj_proc_alive(pj_proc* p) {
  DWORD code = 0;
  if (!p->started || !p->h_proc) return 0;
  return GetExitCodeProcess((HANDLE)p->h_proc, &code) && code == STILL_ACTIVE;
}

static BOOL CALLBACK close_own_windows(HWND h, LPARAM pid) {
  DWORD q = 0;
  GetWindowThreadProcessId(h, &q);
  if (q == (DWORD)pid) PostMessageW(h, WM_CLOSE, 0, 0);
  return TRUE;
}

void pj_proc_stop(pj_proc* p, uint32_t grace_ms) {
  if (!p->started) return;
  if (p->h_in) CloseHandle((HANDLE)p->h_in);
  p->h_in = NULL;
  if (p->h_proc) {
    if (WaitForSingleObject((HANDLE)p->h_proc, grace_ms / 2) == WAIT_TIMEOUT) {
      EnumWindows(close_own_windows, (LPARAM)p->pid);
      if (WaitForSingleObject((HANDLE)p->h_proc, grace_ms / 2 + 1) == WAIT_TIMEOUT)
        TerminateProcess((HANDLE)p->h_proc, 1); /* our own child's handle only */
      WaitForSingleObject((HANDLE)p->h_proc, 5000);
    }
    CloseHandle((HANDLE)p->h_proc);
    p->h_proc = NULL;
  }
  if (p->h_out) CloseHandle((HANDLE)p->h_out);
  p->h_out = NULL;
  p->started = 0;
}

typedef struct thr_box {
  pj_thread_fn fn;
  void* arg;
} thr_box;

static unsigned __stdcall thr_main(void* a) {
  thr_box b = *(thr_box*)a;
  free(a);
  b.fn(b.arg);
  return 0;
}

int pj_thread_start(pj_thread* t, pj_thread_fn fn, void* arg) {
  thr_box* b = (thr_box*)malloc(sizeof(thr_box));
  uintptr_t h;
  if (!b) return -1;
  b->fn = fn;
  b->arg = arg;
  h = _beginthreadex(NULL, 0, thr_main, b, 0, NULL);
  if (!h) {
    free(b);
    return -1;
  }
  t->h = (void*)h;
  t->started = 1;
  return 0;
}

void pj_thread_join(pj_thread* t) {
  if (!t->started) return;
  WaitForSingleObject((HANDLE)t->h, INFINITE);
  CloseHandle((HANDLE)t->h);
  t->started = 0;
}
#else
typedef int pj_platform_win_unused;
#endif
