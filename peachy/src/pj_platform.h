/* Peachy thin platform layer: Windows and POSIX. MIT license.
 * No locks, no environment variables, no global state. */
#ifndef PJ_PLATFORM_H
#define PJ_PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

uint64_t pj_mono_ns(void);
void pj_sleep_ms(uint32_t ms);
int pj_os_random(uint8_t* out, size_t n);           /* 0 on success; never falls back */
void pj_local_time(char* out, size_t cap);          /* 2026-11-16T19:30:01-05:00 */
void pj_compact_time(char* out, size_t cap);        /* 20261116-193001 */
uint32_t pj_pid(void);
FILE* pj_fopen(const char* utf8_path, const char* mode);
/* Read-only open that shares with a writer already holding the file (Windows); plain fopen "rb" elsewhere. */
FILE* pj_fopen_read_shared(const char* utf8_path);
int pj_dir_exists(const char* utf8_path);
int pj_replace_file(const char* tmp_path, const char* final_path); /* 0 on success */

void* pj_dl_open(const char* utf8_path);
typedef void (*pj_fnptr)(void);
pj_fnptr pj_dl_fn(void* lib, const char* name);
void pj_dl_close(void* lib);

typedef struct pj_proc {
  void* h_proc;
  void* h_in;  /* our write end of the child's stdin */
  void* h_out; /* our read end of the child's stdout */
  int pid;
  int fd_in, fd_out;
  int started;
} pj_proc;

int pj_proc_spawn(pj_proc* p, const char* const* argv, uint32_t argc, const char* workdir, const char* stderr_log);
int pj_proc_write(pj_proc* p, const char* data, size_t len);            /* 0 on success */
int pj_proc_read(pj_proc* p, char* buf, size_t cap, size_t* n, uint32_t timeout_ms); /* 0 ok (n may be 0), -1 closed */
int pj_proc_alive(pj_proc* p);
void pj_proc_stop(pj_proc* p, uint32_t grace_ms);

typedef struct pj_thread {
  void* h;
  uintptr_t t;
  int started;
} pj_thread;
typedef void (*pj_thread_fn)(void* arg);
int pj_thread_start(pj_thread* t, pj_thread_fn fn, void* arg);
void pj_thread_join(pj_thread* t);

#endif
