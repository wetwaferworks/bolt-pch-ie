/* Bolt PCHIE - file, error and lock-free thread helpers for the ternary converter (C17). */
#include "pchie_ternary_convert_internal.h"

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#include <process.h>
#include <io.h>
#else
#define _FILE_OFFSET_BITS 64
#include <unistd.h>
#include <sys/types.h>
#if defined(__STDC_NO_THREADS__) || !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
#include <pthread.h>
#define TERNARY_USE_PTHREADS 1
#else
#include <threads.h>
#endif
#endif

int ternary_multiply_overflows(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0 && b > ~UINT64_C(0) / a) return 1;
    *out = a * b;
    return 0;
}

int ternary_seek64(FILE *file, uint64_t offset) {
#if defined(_WIN32)
    return _fseeki64(file, (long long)offset, SEEK_SET);
#else
    return fseeko(file, (off_t)offset, SEEK_SET);
#endif
}

uint64_t ternary_file_length(FILE *file) {
#if defined(_WIN32)
    if (_fseeki64(file, 0, SEEK_END) != 0) return 0;
    {
        const long long n = _ftelli64(file);
        return n < 0 ? 0 : (uint64_t)n;
    }
#else
    if (fseeko(file, 0, SEEK_END) != 0) return 0;
    {
        const off_t n = ftello(file);
        return n < 0 ? 0 : (uint64_t)n;
    }
#endif
}

FILE *ternary_open_file_read_only(const char *path) {
    FILE *file = NULL;
    if (!path) return NULL;
#if defined(_WIN32)
    if (fopen_s(&file, path, "rb") != 0) return NULL;
    return file;
#else
    file = fopen(path, "rb");
    return file;
#endif
}

FILE *ternary_open_file_write_new(const char *path) {
    FILE *file = NULL;
    if (!path) return NULL;
#if defined(_WIN32)
    if (fopen_s(&file, path, "wb") != 0) return NULL;
    return file;
#else
    file = fopen(path, "wb");
    return file;
#endif
}

void ternary_set_error(char *error, size_t error_capacity, const char *message) {
    if (!error || error_capacity == 0) return;
    if (!message) message = "";
    {
        size_t n = strlen(message);
        if (n >= error_capacity) n = error_capacity - 1;
        memcpy(error, message, n);
        error[n] = 0;
    }
}

void ternary_set_errorf(char *error, size_t error_capacity, const char *format, ...) {
    va_list args;
    if (!error || error_capacity == 0) return;
    va_start(args, format);
    vsnprintf(error, error_capacity, format, args);
    va_end(args);
    error[error_capacity - 1] = 0;
}

char *ternary_duplicate_string(const char *text, size_t length) {
    char *copy = (char *)malloc(length + 1);
    if (!copy) return NULL;
    if (length) memcpy(copy, text, length);
    copy[length] = 0;
    return copy;
}

int ternary_source_file_fail(ternary_source_file *out, char *error, size_t error_capacity, const char *message) {
    ternary_source_file_clear(out);
    ternary_set_error(error, error_capacity, message);
    return 0;
}

int ternary_pick_thread_count(int thread_count, uint64_t item_count) {
    int t = thread_count;
    if (t <= 0) {
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        t = (int)info.dwNumberOfProcessors;
#else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        t = n > 0 ? (int)n : 1;
#endif
    }
    if (t < 1) t = 1;
    if (t > 64) t = 64;
    if ((uint64_t)t > item_count) t = (int)(item_count ? item_count : 1);
    return t;
}

typedef struct ternary_worker_job {
    ternary_worker_fn work;
    void *context;
    uint64_t begin;
    uint64_t end;
    int worker_index;
} ternary_worker_job;

#if defined(_WIN32)
static unsigned __stdcall ternary_worker_entry(void *arg) {
    ternary_worker_job *job = (ternary_worker_job *)arg;
    job->work(job->begin, job->end, job->worker_index, job->context);
    return 0;
}
#elif defined(TERNARY_USE_PTHREADS)
static void *ternary_worker_entry_pthread(void *arg) {
    ternary_worker_job *job = (ternary_worker_job *)arg;
    job->work(job->begin, job->end, job->worker_index, job->context);
    return NULL;
}
#else
static int ternary_worker_entry_c11(void *arg) {
    ternary_worker_job *job = (ternary_worker_job *)arg;
    job->work(job->begin, job->end, job->worker_index, job->context);
    return 0;
}
#endif

void ternary_run_in_parallel(uint64_t item_count, int thread_count, ternary_worker_fn work, void *context) {
    const int workers = ternary_pick_thread_count(thread_count, item_count);
    ternary_worker_job *jobs;
    int i;
    if (!work) return;
    if (workers <= 1 || item_count == 0) {
        work(0, item_count, 0, context);
        return;
    }
    jobs = (ternary_worker_job *)calloc((size_t)workers, sizeof(*jobs));
    if (!jobs) {
        work(0, item_count, 0, context);
        return;
    }
    for (i = 0; i < workers; ++i) {
        jobs[i].work = work;
        jobs[i].context = context;
        jobs[i].begin = item_count * (uint64_t)i / (uint64_t)workers;
        jobs[i].end = item_count * (uint64_t)(i + 1) / (uint64_t)workers;
        jobs[i].worker_index = i;
    }

#if defined(_WIN32)
    {
        HANDLE handles[64];
        int started = 0;
        int launched = 0;
        for (i = 0; i < workers; ++i) {
            uintptr_t thread = _beginthreadex(NULL, 0, ternary_worker_entry, &jobs[i], 0, NULL);
            if (!thread) break;
            handles[started++] = (HANDLE)thread;
            launched = i + 1;
        }
        if (launched < workers) {
            /* A later worker failed to start. Wait for the ones that did, then finish the rest here. */
            if (started) WaitForMultipleObjects((DWORD)started, handles, TRUE, INFINITE);
            for (i = 0; i < started; ++i) CloseHandle(handles[i]);
            for (i = launched; i < workers; ++i)
                work(jobs[i].begin, jobs[i].end, jobs[i].worker_index, context);
        } else {
            WaitForMultipleObjects((DWORD)started, handles, TRUE, INFINITE);
            for (i = 0; i < started; ++i) CloseHandle(handles[i]);
        }
    }
#elif defined(TERNARY_USE_PTHREADS)
    {
        pthread_t threads[64];
        int started = 0;
        for (i = 0; i < workers; ++i) {
            if (pthread_create(&threads[started], NULL, ternary_worker_entry_pthread, &jobs[i]) != 0) break;
            started++;
        }
        for (i = 0; i < started; ++i) pthread_join(threads[i], NULL);
        for (i = started; i < workers; ++i)
            work(jobs[i].begin, jobs[i].end, jobs[i].worker_index, context);
    }
#else
    {
        thrd_t threads[64];
        int started = 0;
        for (i = 0; i < workers; ++i) {
            if (thrd_create(&threads[started], ternary_worker_entry_c11, &jobs[i]) != thrd_success) break;
            started++;
        }
        for (i = 0; i < started; ++i) thrd_join(threads[i], NULL);
        for (i = started; i < workers; ++i)
            work(jobs[i].begin, jobs[i].end, jobs[i].worker_index, context);
    }
#endif
    free(jobs);
}
