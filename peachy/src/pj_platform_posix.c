/* Peachy platform layer, POSIX. MIT license. */
#if !defined(_WIN32)
#define _GNU_SOURCE 1
#include "pj_platform.h"
#include "pj_guard.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/random.h>
#endif

uint64_t pj_mono_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void pj_sleep_ms(uint32_t ms) {
  struct timespec ts;
  ts.tv_sec = ms / 1000u;
  ts.tv_nsec = (long)(ms % 1000u) * 1000000L;
  while (nanosleep(&ts, &ts) != 0 && errno == EINTR) {
  }
}

int pj_os_random(uint8_t* out, size_t n) {
#if defined(__linux__)
  size_t got = 0;
  while (got < n) {
    ssize_t r = getrandom(out + got, n - got, 0);
    if (r < 0) {
      if (errno == EINTR) continue;
      return -1;
    }
    got += (size_t)r;
  }
  return 0;
#else
  size_t got = 0;
  while (got < n) {
    size_t take = n - got > 256 ? 256 : n - got;
    if (getentropy(out + got, take) != 0) return -1;
    got += take;
  }
  return 0;
#endif
}

void pj_local_time(char* out, size_t cap) {
  time_t t = time(NULL);
  struct tm lt;
  char z[24];
  long off;
  localtime_r(&t, &lt);
  off = lt.tm_gmtoff / 60;
  snprintf(z, sizeof(z), "%c%02ld:%02ld", off < 0 ? '-' : '+', (off < 0 ? -off : off) / 60, (off < 0 ? -off : off) % 60);
  snprintf(out, cap, "%04d-%02d-%02dT%02d:%02d:%02d%s", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour,
           lt.tm_min, lt.tm_sec, z);
}

void pj_compact_time(char* out, size_t cap) {
  time_t t = time(NULL);
  struct tm lt;
  localtime_r(&t, &lt);
  snprintf(out, cap, "%04d%02d%02d-%02d%02d%02d", lt.tm_year + 1900, lt.tm_mon + 1, lt.tm_mday, lt.tm_hour, lt.tm_min,
           lt.tm_sec);
}

uint32_t pj_pid(void) { return (uint32_t)getpid(); }

FILE* pj_fopen(const char* p, const char* mode) { return fopen(p, mode); }
FILE* pj_fopen_read_shared(const char* p) { return fopen(p, "rb"); }

int pj_dir_exists(const char* p) {
  struct stat st;
  return p && stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

int pj_replace_file(const char* tmp, const char* fin) { return rename(tmp, fin) == 0 ? 0 : -1; }

void* pj_dl_open(const char* p) { return dlopen(p, RTLD_NOW | RTLD_LOCAL); }
pj_fnptr pj_dl_fn(void* lib, const char* name) {
  void* p = lib ? dlsym(lib, name) : NULL;
  pj_fnptr f = NULL;
  if (p) memcpy(&f, &p, sizeof(f));
  return f;
}
void pj_dl_close(void* lib) {
  if (lib) dlclose(lib);
}

int pj_proc_spawn(pj_proc* p, const char* const* argv, uint32_t argc, const char* workdir, const char* stderr_log) {
  int in_p[2], out_p[2], err_fd;
  posix_spawn_file_actions_t fa;
  char* av[65];
  uint32_t i;
  pid_t pid;
  int rc;
  memset(p, 0, sizeof(*p));
  p->fd_in = p->fd_out = -1;
  if (!argv || argc == 0 || argc > 64) return -1;
  for (i = 0; i < argc; ++i) {
    if (!argv[i]) return -1;
    av[i] = (char*)argv[i];
  }
  av[argc] = NULL;
  if (pipe(in_p) != 0) return -1;
  if (pipe(out_p) != 0) {
    close(in_p[0]);
    close(in_p[1]);
    return -1;
  }
  err_fd = open(stderr_log ? stderr_log : "/dev/null", O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (err_fd < 0) {
    close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]);
    return -1;
  }
  fcntl(in_p[1], F_SETFD, FD_CLOEXEC);
  fcntl(out_p[0], F_SETFD, FD_CLOEXEC);
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, in_p[0], 0);
  posix_spawn_file_actions_adddup2(&fa, out_p[1], 1);
  posix_spawn_file_actions_adddup2(&fa, err_fd, 2);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 29))
  if (workdir) posix_spawn_file_actions_addchdir_np(&fa, workdir);
#else
  if (workdir) {
    posix_spawn_file_actions_destroy(&fa);
    close(in_p[0]); close(in_p[1]); close(out_p[0]); close(out_p[1]); close(err_fd);
    return -1;
  }
#endif
  rc = posix_spawn(&pid, av[0], &fa, NULL, av, NULL);
  posix_spawn_file_actions_destroy(&fa);
  close(in_p[0]);
  close(out_p[1]);
  close(err_fd);
  if (rc != 0) {
    close(in_p[1]);
    close(out_p[0]);
    return -1;
  }
  fcntl(out_p[0], F_SETFL, fcntl(out_p[0], F_GETFL) | O_NONBLOCK);
  p->pid = (int)pid;
  p->fd_in = in_p[1];
  p->fd_out = out_p[0];
  p->started = 1;
  return 0;
}

/* Writes with SIGPIPE blocked on this thread, so a dead child turns into an error, not a signal. The
 * process-wide signal setup is the host's and is left alone. */
int pj_proc_write(pj_proc* p, const char* data, size_t len) {
  sigset_t block, old;
  size_t done = 0;
  int ok = 0;
  if (!p->started || p->fd_in < 0) return -1;
  sigemptyset(&block);
  sigaddset(&block, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &block, &old);
  while (done < len) {
    ssize_t w = write(p->fd_in, data + done, len - done);
    if (w < 0) {
      if (errno == EINTR) continue;
      if (errno == EPIPE) {
        struct timespec zero = {0, 0};
        sigtimedwait(&block, NULL, &zero); /* drop the pending SIGPIPE */
      }
      ok = -1;
      break;
    }
    done += (size_t)w;
  }
  pthread_sigmask(SIG_SETMASK, &old, NULL);
  return ok;
}

int pj_proc_read(pj_proc* p, char* buf, size_t cap, size_t* n, uint32_t timeout_ms) {
  struct pollfd pf;
  ssize_t r;
  *n = 0;
  if (!p->started || p->fd_out < 0) return -1;
  pf.fd = p->fd_out;
  pf.events = POLLIN;
  pf.revents = 0;
  if (poll(&pf, 1, (int)timeout_ms) <= 0) return 0;
  r = read(p->fd_out, buf, cap);
  if (r > 0) {
    *n = (size_t)r;
    return 0;
  }
  if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return 0;
  return -1;
}

int pj_proc_alive(pj_proc* p) {
  int st;
  if (!p->started || p->pid <= 0) return 0;
  return waitpid((pid_t)p->pid, &st, WNOHANG) == 0;
}

void pj_proc_stop(pj_proc* p, uint32_t grace_ms) {
  int st;
  uint32_t waited = 0;
  if (!p->started) return;
  if (p->fd_in >= 0) close(p->fd_in);
  p->fd_in = -1;
  if (p->pid > 0) {
    while (waitpid((pid_t)p->pid, &st, WNOHANG) == 0 && waited < grace_ms) {
      pj_sleep_ms(5);
      waited += 5;
    }
    if (waitpid((pid_t)p->pid, &st, WNOHANG) == 0) {
      kill((pid_t)p->pid, SIGTERM); /* our own child only */
      waited = 0;
      while (waitpid((pid_t)p->pid, &st, WNOHANG) == 0 && waited < 2000) {
        pj_sleep_ms(5);
        waited += 5;
      }
      if (waitpid((pid_t)p->pid, &st, WNOHANG) == 0) {
        kill((pid_t)p->pid, SIGKILL);
        waitpid((pid_t)p->pid, &st, 0);
      }
    }
  }
  if (p->fd_out >= 0) close(p->fd_out);
  p->fd_out = -1;
  p->pid = 0;
  p->started = 0;
}

typedef struct thr_box {
  pj_thread_fn fn;
  void* arg;
} thr_box;

static void* thr_main(void* a) {
  thr_box b = *(thr_box*)a;
  free(a);
  b.fn(b.arg);
  return NULL;
}

int pj_thread_start(pj_thread* t, pj_thread_fn fn, void* arg) {
  pthread_t th;
  thr_box* b = (thr_box*)malloc(sizeof(thr_box));
  if (!b) return -1;
  b->fn = fn;
  b->arg = arg;
  if (pthread_create(&th, NULL, thr_main, b) != 0) {
    free(b);
    return -1;
  }
  t->t = (uintptr_t)th;
  t->started = 1;
  return 0;
}

void pj_thread_join(pj_thread* t) {
  if (!t->started) return;
  pthread_join((pthread_t)t->t, NULL);
  t->started = 0;
}

/* ---------------------------------------------------------------- file guard (see pj_guard.h) */

int pj_process_elevated(void) { return geteuid() == 0 ? 1 : 0; }

/* Could this process change the entry at p (it owns it, so it could chmod it, or it may write to it)? */
static int posix_changeable(const char* p, const struct stat* st) {
  return st->st_uid == geteuid() || faccessat(AT_FDCWD, p, W_OK, AT_EACCESS) == 0;
}

void pj_file_facts_get(const char* path, pj_file_facts* f) {
  char abs[4096];
  struct stat st;
  size_t n;
  memset(f, 0, sizeof(*f));
  f->elevated = pj_process_elevated();
  if (!path || !path[0]) return;
  if (path[0] == '/') {
    if (strlen(path) >= sizeof(abs)) return;
    memcpy(abs, path, strlen(path) + 1);
  } else {
    size_t c;
    if (!getcwd(abs, sizeof(abs))) return;
    c = strlen(abs);
    if (c + 1 + strlen(path) >= sizeof(abs)) return;
    abs[c] = '/';
    memcpy(abs + c + 1, path, strlen(path) + 1);
  }
  /* the folders above are walked as text, so "." and ".." components and doubled slashes are refused outright */
  if (strstr(abs, "//")) return;
  {
    const char* q = abs;
    while (*q) {
      const char* e;
      while (*q == '/') ++q;
      e = q;
      while (*e && *e != '/') ++e;
      if ((e - q == 1 && q[0] == '.') || (e - q == 2 && q[0] == '.' && q[1] == '.')) return;
      q = e;
    }
  }
  if (lstat(abs, &st) != 0) return;
  f->regular = S_ISREG(st.st_mode);
  f->link_in_path = S_ISLNK(st.st_mode);
  f->changeable = f->elevated || posix_changeable(abs, &st);
  f->others_can_read = st.st_uid != geteuid() || (st.st_mode & 077) != 0;
  n = strlen(abs);
  while (n > 1 && abs[n - 1] == '/') abs[--n] = 0;
  for (;;) {
    char* slash = strrchr(abs, '/');
    if (!slash) return;
    if (slash == abs) slash[1] = 0; /* the root itself */
    else *slash = 0;
    if (lstat(abs, &st) != 0) return;
    if (S_ISLNK(st.st_mode)) f->link_in_path = 1;
    else if (!S_ISDIR(st.st_mode)) return;
    if (f->elevated || posix_changeable(abs, &st)) f->path_changeable = 1;
    if (abs[1] == 0) break;
  }
  f->known = 1;
}

int pj_protect_path(const char* path, int kind, int is_dir) {
  if (!path) return -1;
  if (kind == PJ_GUARD_TOKEN) return chmod(path, 0600) == 0 ? 0 : -1;
  if ((kind != PJ_GUARD_ALLOWLIST && kind != PJ_GUARD_CODE) || geteuid() != 0) return -1;
  if (chown(path, 0, 0) != 0) return -1;
  return chmod(path, (is_dir || kind == PJ_GUARD_CODE) ? 0755 : 0644) == 0 ? 0 : -1;
}
#else
typedef int pj_platform_posix_unused;
#endif
