/* Bolt PCHIE - ternary converter (clean-room, C17): reads a GGUF or safetensors model one tensor at a time
 * and turns its weight matrices into PTN1 ternary tensors (I2 packing, one scale per 128-weight block).
 *
 * Inputs are untrusted. Both header readers check every count, shape, offset and extent against the file size
 * before anything is allocated or read, refuse overlapping tensors, and fail closed on anything they don't know.
 * The model file is opened read-only and never written.
 *
 * The fit runs on the CPU (threads over disjoint groups, joined at the end) or on CUDA. Both run the same
 * ternary_group_fit() from pchie_ternary_group_fit.h, so they give the same codes and scales bit for bit; the
 * tests check it. Nothing here is wired into inference. See docs/ternary/README.md, "Converter".
 *
 * The method is Microsoft's BitNet research, written from the published papers, not from their source code.
 */
#ifndef PCHIE_TERNARY_CONVERT_H
#define PCHIE_TERNARY_CONVERT_H

#include "pchie/pchie_ternary_group_fit.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TERNARY_CONVERT_ERROR_CAPACITY = 512 };
enum { TERNARY_SOURCE_MAX_DIMENSIONS = 8 };
enum { TERNARY_SOURCE_MAX_NAME_BYTES = 4096 };
enum { TERNARY_SOURCE_MAX_TENSORS = 1 << 20 };
enum { TERNARY_SAFETENSORS_MAX_HEADER_BYTES = 16 << 20 };
enum { TERNARY_HARD_MAX_ROWS = 1 << 20 };
enum { TERNARY_HARD_MAX_COLUMNS = 1 << 20 };
enum { TERNARY_I2_BLOCK_BYTES = 32 };
enum { TERNARY_PTN1_HEADER_BYTES = 96 };

/* Same numeric values as pchie::ternary::Status, so C++ tests can compare them directly. */
typedef enum ternary_convert_status {
    TERNARY_CONVERT_OK = 0,
    TERNARY_CONVERT_BAD_ARGUMENT = 1,
    TERNARY_CONVERT_TOO_LARGE = 2,
    TERNARY_CONVERT_TRUNCATED = 3,
    TERNARY_CONVERT_BAD_MAGIC = 4,
    TERNARY_CONVERT_BAD_VERSION = 5,
    TERNARY_CONVERT_BAD_HEADER = 6,
    TERNARY_CONVERT_BAD_CHECKSUM = 7,
    TERNARY_CONVERT_BAD_CODE = 8,
    TERNARY_CONVERT_BAD_SCALE = 9,
    TERNARY_CONVERT_NON_FINITE = 10
} ternary_convert_status;

const char *ternary_convert_status_name(ternary_convert_status status);

typedef enum ternary_source_weight_type {
    TERNARY_SOURCE_WEIGHT_F32 = 0,
    TERNARY_SOURCE_WEIGHT_F16,
    TERNARY_SOURCE_WEIGHT_BF16,
    TERNARY_SOURCE_WEIGHT_Q8_0,
    TERNARY_SOURCE_WEIGHT_Q4_K,
    TERNARY_SOURCE_WEIGHT_Q6_K,
    TERNARY_SOURCE_WEIGHT_OTHER
} ternary_source_weight_type;

const char *ternary_source_weight_type_name(ternary_source_weight_type type);

typedef struct ternary_source_tensor {
    char *name;
    char type_name[16];                          /* as the file names it ("BF16", "Q4_K", ...) */
    ternary_source_weight_type type;             /* OTHER: listed and bounds-checked, but this converter can't decode it */
    uint64_t shape[TERNARY_SOURCE_MAX_DIMENSIONS]; /* outermost first (GGUF dims are reversed into this order) */
    size_t dimension_count;
    uint64_t rows;
    uint64_t columns;                            /* row-major view: columns is the innermost (contiguous) dimension */
    uint64_t file_offset;                        /* absolute file offset of the first byte */
    uint64_t byte_count;                         /* byte extent in the file */
    uint64_t bytes_per_row;                      /* 0 for OTHER */
} ternary_source_tensor;

typedef enum ternary_source_format {
    TERNARY_SOURCE_FORMAT_GGUF = 1,
    TERNARY_SOURCE_FORMAT_SAFETENSORS = 2
} ternary_source_format;

typedef struct ternary_source_file {
    ternary_source_format format;
    char *path;
    uint64_t file_size;
    uint64_t data_begin;
    uint64_t data_end;                           /* the region every tensor must sit inside */
    ternary_source_tensor *tensors;              /* in the order the file lists them */
    size_t tensor_count;
} ternary_source_file;

void ternary_source_file_clear(ternary_source_file *file);

/* Parses a safetensors header: json is the header text (header_length bytes, after the 8-byte length) of a file
 * of file_size bytes. Checks: header_length <= TERNARY_SAFETENSORS_MAX_HEADER_BYTES and fits the file; the JSON
 * is an object with no duplicate keys; "__metadata__", if present, maps strings to strings; every other entry has
 * exactly "dtype" (a known dtype), "shape" (at most TERNARY_SOURCE_MAX_DIMENSIONS whole numbers >= 0) and
 * "data_offsets" ([begin, end], whole numbers, begin <= end <= data size); end - begin equals the element count
 * times the dtype size (overflow checked); no two tensors overlap. On failure out is empty and error says why. */
int ternary_parse_safetensors_header(const char *json, uint64_t header_length, uint64_t file_size, const char *path,
                                     ternary_source_file *out, char *error, size_t error_capacity);
int ternary_open_safetensors(const char *path, ternary_source_file *out, char *error, size_t error_capacity);
/* GGUF through this converter's own C reader, with the same bounds checks as the repo reader plus refusals for
 * unsized tensor types and overlapping tensors. */
int ternary_open_gguf(const char *path, ternary_source_file *out, char *error, size_t error_capacity);
/* Picks the format from the first bytes ("GGUF" magic, else safetensors). */
int ternary_open_source(const char *path, ternary_source_file *out, char *error, size_t error_capacity);
/* The shared layout check: every extent inside [data_begin, data_end), no overlaps. Used by both readers. */
int ternary_check_tensor_layout(const ternary_source_file *file, char *error, size_t error_capacity);

/* Whether a tensor can be converted (2-D or more, columns a multiple of 128, a decodable type, within the hard
 * ceilings). reason gets the reason when it can't (may be NULL). */
int ternary_tensor_is_convertible(const ternary_source_tensor *tensor, char *reason, size_t reason_capacity);

/* Decodes n rows of raw file bytes (n * tensor->bytes_per_row) to float, row-major. */
ternary_convert_status ternary_decode_rows(const ternary_source_tensor *tensor, const uint8_t *raw, uint64_t row_count,
                                           float *out);
/* Reads and decodes rows [first_row, first_row + row_count). raw_buffer is grown as needed (*raw_capacity is the
 * current allocation). Fails on a short read or a non-finite value. */
int ternary_read_rows(FILE *file, const ternary_source_file *source, const ternary_source_tensor *tensor,
                      uint64_t first_row, uint64_t row_count, uint8_t **raw_buffer, size_t *raw_capacity,
                      float *out, char *error, size_t error_capacity);

/* n_groups groups of TERNARY_GROUP_WEIGHT_COUNT weights, contiguous in weights. codes gets n_groups * 128 codes,
 * scales n_groups scales. Any non-finite weight gives NON_FINITE with codes and scales zeroed. thread_count <= 0
 * means the hardware count. */
ternary_convert_status ternary_fit_groups_cpu(const float *weights, uint64_t group_count, ternary_group_fit_kind fit_kind,
                                              int thread_count, int8_t *codes, float *scales);

typedef struct ternary_cuda_info {
    int built;                                   /* compiled with the CUDA backend */
    int device_count;
    char device_name[256];
    uint64_t total_bytes;
} ternary_cuda_info;

typedef struct ternary_cuda_counters {
    uint64_t own_bytes_peak;                     /* device memory this converter held at most */
    uint64_t device_used_peak;                   /* whole-device use (all processes) seen right after our allocations */
    double kernel_milliseconds;
    double copy_milliseconds;
    uint64_t call_count;
} ternary_cuda_counters;

ternary_cuda_info ternary_query_cuda_info(void);
/* Same contract and same bits as ternary_fit_groups_cpu. Device buffers are kept between calls and grow as needed;
 * ternary_cuda_release() frees them. Returns OK, NON_FINITE (outputs zeroed) or BAD_ARGUMENT; error has CUDA errors.
 * One caller at a time: the buffers are process-wide, so a call made while another thread's call is running is
 * refused (BAD_ARGUMENT). Do not call ternary_cuda_release() while a fit is running. */
ternary_convert_status ternary_fit_groups_cuda(const float *weights, uint64_t group_count, ternary_group_fit_kind fit_kind,
                                               int8_t *codes, float *scales, char *error, size_t error_capacity);
const ternary_cuda_counters *ternary_query_cuda_counters(void);
void ternary_cuda_release(void);

typedef struct ternary_reconstruction_error {
    uint64_t weight_count;
    uint64_t zero_codes;
    uint64_t zero_groups;
    uint64_t group_count;
    double sum_weight_squared;
    double sum_error_squared;
    double sum_weight_times_reconstruction;
    double sum_reconstruction_squared;
    double max_absolute_error;
} ternary_reconstruction_error;

double ternary_reconstruction_relative_error(const ternary_reconstruction_error *stats);
double ternary_reconstruction_cosine(const ternary_reconstruction_error *stats);
double ternary_reconstruction_rmse(const ternary_reconstruction_error *stats);
void ternary_reconstruction_error_add(ternary_reconstruction_error *into, const ternary_reconstruction_error *other);

/* Adds rows x columns weights and their reconstruction scale * code to stats, row by row in order (any thread
 * count gives the same sums). */
void ternary_accumulate_reconstruction_error(const float *weights, const int8_t *codes, const float *scales,
                                             uint64_t rows, uint64_t columns, int thread_count,
                                             ternary_reconstruction_error *stats);

typedef enum ternary_fit_backend {
    TERNARY_FIT_BACKEND_CPU = 0,
    TERNARY_FIT_BACKEND_CUDA = 1
} ternary_fit_backend;

typedef struct ternary_convert_options {
    ternary_group_fit_kind fit_kind;
    ternary_fit_backend backend;
    int check_parity;                            /* also run the other backend and compare every code and scale */
    int thread_count;
    uint64_t chunk_weights;                      /* weights per streamed chunk (default 32 Mi, 128 MiB of float) */
} ternary_convert_options;

void ternary_convert_options_init(ternary_convert_options *options);

/* An owned I2, per-block ternary matrix produced by the converter. */
typedef struct ternary_packed_tensor {
    uint64_t rows;
    uint64_t columns;
    uint8_t *packed_codes;
    size_t packed_codes_bytes;
    float *scales;
    size_t scale_count;
} ternary_packed_tensor;

void ternary_packed_tensor_clear(ternary_packed_tensor *tensor);

typedef struct ternary_convert_result {
    ternary_packed_tensor output;
    ternary_reconstruction_error stats;
    double read_milliseconds;
    double fit_milliseconds;
    double other_fit_milliseconds;
    double stats_milliseconds;
    uint64_t parity_groups_checked;
    uint64_t parity_groups_different;
} ternary_convert_result;

void ternary_convert_result_clear(ternary_convert_result *result);

/* Streams one tensor in row chunks: read and decode, fit, accumulate the error, pack. */
int ternary_convert_tensor(FILE *file, const ternary_source_file *source, const ternary_source_tensor *tensor,
                           const ternary_convert_options *options, ternary_convert_result *result,
                           char *error, size_t error_capacity);

/* I2 pack / unpack of one row of ternary values (int8 in {-1,0,1}, length a multiple of 128). */
ternary_convert_status ternary_pack_row_i2(const int8_t *codes, uint64_t columns, uint8_t *packed);
ternary_convert_status ternary_unpack_row_i2(const uint8_t *packed, uint64_t columns, int8_t *codes);
uint64_t ternary_i2_row_bytes(uint64_t columns);

/* PTN1 (I2, one scale per 128-weight block). serialize mallocs *bytes; the caller frees it.
 * parse validates everything and on success points view_codes / view_scales into the buffer. */
typedef struct ternary_ptn1_view {
    uint64_t rows;
    uint64_t columns;
    const uint8_t *packed_codes;
    const uint8_t *scales_raw;                   /* little-endian float32, possibly unaligned */
    uint64_t bytes_per_row;
    uint64_t scale_count;
} ternary_ptn1_view;

int ternary_serialize_ptn1(const ternary_packed_tensor *tensor, uint8_t **bytes, size_t *length,
                           char *error, size_t error_capacity);
ternary_convert_status ternary_parse_ptn1(const uint8_t *buffer, size_t length, uint64_t max_bytes,
                                          ternary_ptn1_view *view);
uint64_t ternary_fnv1a64(const uint8_t *bytes, size_t length, uint64_t seed);

#ifdef __cplusplus
}
#endif

#endif /* PCHIE_TERNARY_CONVERT_H */
