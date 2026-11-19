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
#include <aclapi.h>
#include <bcrypt.h>
#include <process.h>
#include <share.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>

#include "pj_platform.h"
#include "pj_guard.h"

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
  HANDLE inherit[3];
  LPPROC_THREAD_ATTRIBUTE_LIST al = NULL;
  SIZE_T al_size = 0;
  STARTUPINFOEXW si;
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
  /* The child inherits exactly its three standard handles and nothing else. Without a handle list, every
   * inheritable handle in this process would go too, including the host's own stdio pipes to the MCP client. If
   * the list can't be built, the spawn fails rather than falling back to inheriting everything. */
  inherit[0] = in_r;
  inherit[1] = out_w;
  inherit[2] = log;
  InitializeProcThreadAttributeList(NULL, 1, 0, &al_size);
  al = al_size ? (LPPROC_THREAD_ATTRIBUTE_LIST)malloc(al_size) : NULL;
  if (!al || !InitializeProcThreadAttributeList(al, 1, 0, &al_size)) {
    free(al);
    al = NULL;
  } else if (!UpdateProcThreadAttribute(al, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof(inherit), NULL, NULL)) {
    DeleteProcThreadAttributeList(al);
    free(al);
    al = NULL;
  }
  memset(&si, 0, sizeof(si));
  si.StartupInfo.cb = sizeof(si);
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = in_r;
  si.StartupInfo.hStdOutput = out_w;
  si.StartupInfo.hStdError = log;
  si.lpAttributeList = al;
  memset(&pi, 0, sizeof(pi));
  ok = al ? CreateProcessW(NULL, cl, NULL, NULL, TRUE, CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, NULL, wd,
                           &si.StartupInfo, &pi)
          : FALSE;
  if (al) {
    DeleteProcThreadAttributeList(al);
    free(al);
  }
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

/* ---------------------------------------------------------------- file guard (see pj_guard.h) */

typedef struct win_sids {
  HANDLE imp; /* impersonation copy of this process's token, for AccessCheck */
  BYTE user_buf[128];
  PSID user;
  BYTE system[SECURITY_MAX_SID_SIZE];
  BYTE admins[SECURITY_MAX_SID_SIZE];
} win_sids;

static int win_sids_open(win_sids* w) {
  HANDLE tok = NULL;
  DWORD len = 0, sz;
  int ok = 0;
  memset(w, 0, sizeof(*w));
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_DUPLICATE, &tok)) return -1;
  if (GetTokenInformation(tok, TokenUser, w->user_buf, sizeof(w->user_buf), &len) &&
      DuplicateToken(tok, SecurityImpersonation, &w->imp)) {
    w->user = ((TOKEN_USER*)w->user_buf)->User.Sid;
    sz = sizeof(w->system);
    ok = CreateWellKnownSid(WinLocalSystemSid, NULL, w->system, &sz) != 0;
    sz = sizeof(w->admins);
    ok = ok && CreateWellKnownSid(WinBuiltinAdministratorsSid, NULL, w->admins, &sz) != 0;
  }
  CloseHandle(tok);
  if (!ok && w->imp) {
    CloseHandle(w->imp);
    w->imp = NULL;
  }
  return ok ? 0 : -1;
}

static void win_sids_close(win_sids* w) {
  if (w->imp) CloseHandle(w->imp);
  w->imp = NULL;
}

int pj_process_elevated(void) {
  HANDLE tok = NULL;
  TOKEN_ELEVATION e;
  DWORD len = 0;
  int r = 1; /* unknown counts as elevated */
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &tok)) return 1;
  if (GetTokenInformation(tok, TokenElevation, &e, sizeof(e), &len)) r = e.TokenIsElevated ? 1 : 0;
  CloseHandle(tok);
  return r;
}

/* What this process's token may do to p, by the file's own security descriptor. 0 on success. */
static int win_granted(const wchar_t* p, const win_sids* w, ACCESS_MASK* granted) {
  PSECURITY_DESCRIPTOR sd = NULL;
  GENERIC_MAPPING gm = {FILE_GENERIC_READ, FILE_GENERIC_WRITE, FILE_GENERIC_EXECUTE, FILE_ALL_ACCESS};
  union {
    PRIVILEGE_SET ps;
    BYTE b[256];
  } priv;
  DWORD pl = sizeof(priv);
  BOOL status = FALSE;
  int rc = -1;
  *granted = 0;
  if (GetNamedSecurityInfoW(p, SE_FILE_OBJECT,
                            OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, NULL,
                            NULL, NULL, NULL, &sd) != ERROR_SUCCESS)
    return -1;
  if (AccessCheck(sd, w->imp, MAXIMUM_ALLOWED, &gm, &priv.ps, &pl, granted, &status)) rc = 0;
  LocalFree(sd);
  return rc;
}

static int win_sid_is_ours(PSID sid, const win_sids* w) {
  return EqualSid(sid, w->user) || EqualSid(sid, (PSID)w->system) || EqualSid(sid, (PSID)w->admins);
}

/* 1 if an account other than ours, SYSTEM and Administrators could read p (or re-permission it), 0 if not, -1 if
 * the descriptor could not be read. A missing DACL lets everyone in; an allow entry of a kind we don't parse
 * counts as someone else. */
static int win_others_can_read(const wchar_t* p, const win_sids* w) {
  PSECURITY_DESCRIPTOR sd = NULL;
  PSID owner = NULL;
  PACL dacl = NULL;
  ACL_SIZE_INFORMATION info;
  DWORD k;
  memset(&info, 0, sizeof(info));
  int r = 0;
  const ACCESS_MASK reads = FILE_READ_DATA | GENERIC_READ | GENERIC_ALL | WRITE_DAC | WRITE_OWNER | MAXIMUM_ALLOWED;
  if (GetNamedSecurityInfoW(p, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION, &owner, NULL,
                            &dacl, NULL, &sd) != ERROR_SUCCESS)
    return -1;
  if (!owner || !win_sid_is_ours(owner, w) || !dacl ||
      !GetAclInformation(dacl, &info, sizeof(info), AclSizeInformation))
    r = 1;
  for (k = 0; r == 0 && k < info.AceCount; ++k) {
    ACE_HEADER* h = NULL;
    if (!GetAce(dacl, k, (LPVOID*)&h) || !h) {
      r = 1;
      break;
    }
    if (h->AceFlags & INHERIT_ONLY_ACE) continue;
    if (h->AceType == ACCESS_DENIED_ACE_TYPE) continue;
    if (h->AceType != ACCESS_ALLOWED_ACE_TYPE) {
      r = 1;
      break;
    }
    {
      ACCESS_ALLOWED_ACE* a = (ACCESS_ALLOWED_ACE*)h;
      if (!win_sid_is_ours((PSID)&a->SidStart, w) && (a->Mask & reads)) r = 1;
    }
  }
  LocalFree(sd);
  return r;
}

void pj_file_facts_get(const char* path, pj_file_facts* f) {
  wchar_t* wp = widen(path);
  wchar_t full[1024];
  DWORD n, attr;
  win_sids w;
  ACCESS_MASK g;
  const ACCESS_MASK file_change =
      FILE_WRITE_DATA | FILE_APPEND_DATA | FILE_WRITE_EA | FILE_WRITE_ATTRIBUTES | DELETE | WRITE_DAC | WRITE_OWNER;
  /* DELETE on a folder lets it be renamed away; the drive root can't be, so there only the rest count */
  const ACCESS_MASK dir_change = FILE_DELETE_CHILD | DELETE | WRITE_DAC | WRITE_OWNER;
  const ACCESS_MASK root_change = FILE_DELETE_CHILD | WRITE_DAC | WRITE_OWNER;
  int o;
  memset(f, 0, sizeof(*f));
  f->elevated = pj_process_elevated();
  if (!wp) return;
  n = GetFullPathNameW(wp, (DWORD)(sizeof(full) / sizeof(full[0])), full, NULL);
  free(wp);
  /* only plain drive paths (C:\...): no UNC, no \\?\ device paths */
  if (n == 0 || n >= sizeof(full) / sizeof(full[0]) || n < 4 || full[1] != L':' || full[2] != L'\\') return;
  if (win_sids_open(&w) != 0) return;
  attr = GetFileAttributesW(full);
  if (attr == INVALID_FILE_ATTRIBUTES) goto out;
  f->regular = !(attr & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE));
  if (attr & FILE_ATTRIBUTE_REPARSE_POINT) f->link_in_path = 1;
  if (win_granted(full, &w, &g) != 0) goto out;
  f->changeable = f->elevated || (g & file_change) != 0;
  o = win_others_can_read(full, &w);
  if (o < 0) goto out;
  f->others_can_read = o;
  /* every folder above: no reparse points, and nothing that would let this account delete, rename or
   * re-permission what's below it */
  for (;;) {
    wchar_t* slash = wcsrchr(full, L'\\');
    if (!slash || slash < full + 2) goto out;
    if (slash == full + 2) slash[1] = 0; /* the drive root, C:\ */
    else *slash = 0;
    attr = GetFileAttributesW(full);
    if (attr == INVALID_FILE_ATTRIBUTES || !(attr & FILE_ATTRIBUTE_DIRECTORY)) goto out;
    if (attr & FILE_ATTRIBUTE_REPARSE_POINT) f->link_in_path = 1;
    if (win_granted(full, &w, &g) != 0) goto out;
    if (f->elevated || (g & (full[3] == 0 ? root_change : dir_change))) f->path_changeable = 1;
    if (full[3] == 0) break;
  }
  f->known = 1;
out:
  win_sids_close(&w);
}

int pj_protect_path(const char* path, int kind, int is_dir) {
  wchar_t* wp = widen(path);
  win_sids w;
  EXPLICIT_ACCESS_W ea[3];
  PACL acl = NULL;
  DWORD inherit = is_dir ? SUB_CONTAINERS_AND_OBJECTS_INHERIT : NO_INHERITANCE;
  SECURITY_INFORMATION si = DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION;
  int k, rc = -1;
  if (!wp) return -1;
  if (win_sids_open(&w) != 0) {
    free(wp);
    return -1;
  }
  memset(ea, 0, sizeof(ea));
  for (k = 0; k < 3; ++k) {
    ea[k].grfAccessMode = SET_ACCESS;
    ea[k].grfInheritance = inherit;
    ea[k].Trustee.TrusteeForm = TRUSTEE_IS_SID;
    ea[k].Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
  }
  ea[0].grfAccessPermissions = GENERIC_ALL;
  ea[0].Trustee.ptstrName = (LPWSTR)w.system;
  ea[1].grfAccessPermissions = GENERIC_ALL;
  ea[1].Trustee.ptstrName = (LPWSTR)w.admins;
  ea[2].Trustee.ptstrName = (LPWSTR)w.user;
  if (kind == PJ_GUARD_ALLOWLIST || kind == PJ_GUARD_CODE) {
    ea[2].grfAccessPermissions = GENERIC_READ | GENERIC_EXECUTE;
    si |= OWNER_SECURITY_INFORMATION;
  } else if (kind == PJ_GUARD_TOKEN) {
    ea[2].grfAccessPermissions = GENERIC_READ | GENERIC_WRITE | DELETE;
  } else {
    goto out;
  }
  if (SetEntriesInAclW(3, ea, NULL, &acl) != ERROR_SUCCESS) goto out;
  if (SetNamedSecurityInfoW(wp, SE_FILE_OBJECT, si, kind != PJ_GUARD_TOKEN ? (PSID)w.admins : NULL, NULL, acl,
                            NULL) == ERROR_SUCCESS)
    rc = 0;
out:
  if (acl) LocalFree(acl);
  win_sids_close(&w);
  free(wp);
  return rc;
}
#else
typedef int pj_platform_win_unused;
#endif
