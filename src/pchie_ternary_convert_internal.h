/* Shared helpers for the ternary converter (C17). Not a public header. */
#ifndef PCHIE_TERNARY_CONVERT_INTERNAL_H
#define PCHIE_TERNARY_CONVERT_INTERNAL_H

#include "pchie/pchie_ternary_convert.h"
#if !defined(_WIN32) && !defined(_FILE_OFFSET_BITS)
#define _FILE_OFFSET_BITS 64
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

#ifdef __cplusplus
extern "C" {
#endif

int ternary_multiply_overflows(uint64_t a, uint64_t b, uint64_t *out);
int ternary_seek64(FILE *file, uint64_t offset);
uint64_t ternary_file_length(FILE *file);
FILE *ternary_open_file_read_only(const char *path);
FILE *ternary_open_file_write_new(const char *path);

void ternary_set_error(char *error, size_t error_capacity, const char *message);
void ternary_set_errorf(char *error, size_t error_capacity, const char *format, ...);

/* Runs work(begin, end, worker_index, context) over [0, item_count) cut into one contiguous range per worker;
 * joins before returning. Workers never share an output, so there is nothing to lock. */
typedef void (*ternary_worker_fn)(uint64_t begin, uint64_t end, int worker_index, void *context);
void ternary_run_in_parallel(uint64_t item_count, int thread_count, ternary_worker_fn work, void *context);
int ternary_pick_thread_count(int thread_count, uint64_t item_count);

char *ternary_duplicate_string(const char *text, size_t length);
int ternary_source_file_fail(ternary_source_file *out, char *error, size_t error_capacity, const char *message);

#ifdef __cplusplus
}
#endif

#endif
