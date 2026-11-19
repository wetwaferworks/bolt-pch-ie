/* Bolt PCHIE - ternary converter: source readers, CPU fit, packing, error statistics, streaming (clean-room, C17).
 * See include/pchie/pchie_ternary_convert.h and docs/ternary/README.md, "Converter".
 */
#include "pchie/pchie_ternary_convert.h"
#include "pchie_ternary_convert_internal.h"
#include "pchie_safetensors_json.h"
#include "pchie_weight_dequantize.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

const char *ternary_convert_status_name(ternary_convert_status status) {
    switch (status) {
        case TERNARY_CONVERT_OK: return "ok";
        case TERNARY_CONVERT_BAD_ARGUMENT: return "bad argument";
        case TERNARY_CONVERT_TOO_LARGE: return "too large";
        case TERNARY_CONVERT_TRUNCATED: return "truncated";
        case TERNARY_CONVERT_BAD_MAGIC: return "bad magic";
        case TERNARY_CONVERT_BAD_VERSION: return "bad version";
        case TERNARY_CONVERT_BAD_HEADER: return "bad header";
        case TERNARY_CONVERT_BAD_CHECKSUM: return "bad checksum";
        case TERNARY_CONVERT_BAD_CODE: return "bad code";
        case TERNARY_CONVERT_BAD_SCALE: return "bad scale";
        case TERNARY_CONVERT_NON_FINITE: return "non-finite";
        default: return "unknown";
    }
}

const char *ternary_source_weight_type_name(ternary_source_weight_type type) {
    switch (type) {
        case TERNARY_SOURCE_WEIGHT_F32: return "F32";
        case TERNARY_SOURCE_WEIGHT_F16: return "F16";
        case TERNARY_SOURCE_WEIGHT_BF16: return "BF16";
        case TERNARY_SOURCE_WEIGHT_Q8_0: return "Q8_0";
        case TERNARY_SOURCE_WEIGHT_Q4_K: return "Q4_K";
        case TERNARY_SOURCE_WEIGHT_Q6_K: return "Q6_K";
        default: return "other";
    }
}

void ternary_source_file_clear(ternary_source_file *file) {
    size_t i;
    if (!file) return;
    free(file->path);
    if (file->tensors) {
        for (i = 0; i < file->tensor_count; ++i) free(file->tensors[i].name);
        free(file->tensors);
    }
    memset(file, 0, sizeof(*file));
}

void ternary_packed_tensor_clear(ternary_packed_tensor *tensor) {
    if (!tensor) return;
    free(tensor->packed_codes);
    free(tensor->scales);
    memset(tensor, 0, sizeof(*tensor));
}

void ternary_convert_result_clear(ternary_convert_result *result) {
    if (!result) return;
    ternary_packed_tensor_clear(&result->output);
    memset(result, 0, sizeof(*result));
}

void ternary_convert_options_init(ternary_convert_options *options) {
    if (!options) return;
    memset(options, 0, sizeof(*options));
    options->fit_kind = TERNARY_GROUP_FIT_LEAST_SQUARES;
    options->backend = TERNARY_FIT_BACKEND_CPU;
    options->chunk_weights = UINT64_C(1) << 25;
}

static double monotonic_milliseconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER now;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
#endif
}

/* ------------------------------------------------------------------------------------------------ layout */

typedef struct tensor_order_item {
    uint64_t offset;
    size_t index;
} tensor_order_item;

static int compare_tensor_order(const void *a, const void *b) {
    const tensor_order_item *ia = (const tensor_order_item *)a;
    const tensor_order_item *ib = (const tensor_order_item *)b;
    if (ia->offset < ib->offset) return -1;
    if (ia->offset > ib->offset) return 1;
    if (ia->index < ib->index) return -1;
    if (ia->index > ib->index) return 1;
    return 0;
}

int ternary_check_tensor_layout(const ternary_source_file *file, char *error, size_t error_capacity) {
    tensor_order_item *order;
    size_t i, used = 0;
    if (!file) {
        ternary_set_error(error, error_capacity, "data region outside the file");
        return 0;
    }
    if (file->data_begin > file->data_end || file->data_end > file->file_size) {
        ternary_set_error(error, error_capacity, "data region outside the file");
        return 0;
    }
    order = (tensor_order_item *)calloc(file->tensor_count ? file->tensor_count : 1, sizeof(*order));
    if (!order) {
        ternary_set_error(error, error_capacity, "out of memory");
        return 0;
    }
    for (i = 0; i < file->tensor_count; ++i) {
        const ternary_source_tensor *t = &file->tensors[i];
        if (t->file_offset < file->data_begin || t->file_offset > file->data_end ||
            t->byte_count > file->data_end - t->file_offset) {
            ternary_set_errorf(error, error_capacity, "tensor %s: extent outside the data region",
                               t->name ? t->name : "");
            free(order);
            return 0;
        }
        if (t->byte_count > 0) {
            order[used].offset = t->file_offset;
            order[used].index = i;
            used++;
        }
    }
    qsort(order, used, sizeof(*order), compare_tensor_order);
    for (i = 1; i < used; ++i) {
        const ternary_source_tensor *prev = &file->tensors[order[i - 1].index];
        const ternary_source_tensor *cur = &file->tensors[order[i].index];
        if (prev->file_offset + prev->byte_count > cur->file_offset) {
            ternary_set_errorf(error, error_capacity, "tensors %s and %s overlap",
                               prev->name ? prev->name : "", cur->name ? cur->name : "");
            free(order);
            return 0;
        }
    }
    free(order);
    return 1;
}

/* ------------------------------------------------------------------------------------------------ safetensors */

typedef struct safetensors_dtype {
    const char *name;
    uint64_t size;
    ternary_source_weight_type type;
} safetensors_dtype;

static const safetensors_dtype k_safetensors_dtypes[] = {
    {"F64", 8, TERNARY_SOURCE_WEIGHT_OTHER},
    {"F32", 4, TERNARY_SOURCE_WEIGHT_F32},
    {"F16", 2, TERNARY_SOURCE_WEIGHT_F16},
    {"BF16", 2, TERNARY_SOURCE_WEIGHT_BF16},
    {"I64", 8, TERNARY_SOURCE_WEIGHT_OTHER},
    {"I32", 4, TERNARY_SOURCE_WEIGHT_OTHER},
    {"I16", 2, TERNARY_SOURCE_WEIGHT_OTHER},
    {"I8", 1, TERNARY_SOURCE_WEIGHT_OTHER},
    {"U64", 8, TERNARY_SOURCE_WEIGHT_OTHER},
    {"U32", 4, TERNARY_SOURCE_WEIGHT_OTHER},
    {"U16", 2, TERNARY_SOURCE_WEIGHT_OTHER},
    {"U8", 1, TERNARY_SOURCE_WEIGHT_OTHER},
    {"BOOL", 1, TERNARY_SOURCE_WEIGHT_OTHER},
    {"F8_E4M3", 1, TERNARY_SOURCE_WEIGHT_OTHER},
    {"F8_E5M2", 1, TERNARY_SOURCE_WEIGHT_OTHER},
};

static int json_whole_number(const ternary_json_value *value, uint64_t *out) {
    double d;
    if (!value || value->type != TERNARY_JSON_NUMBER) return 0;
    d = value->number;
    if (!(d >= 0.0 && d <= 9007199254740992.0) || d != floor(d)) return 0;
    *out = (uint64_t)d;
    return 1;
}


/* Formats the error first, then frees the JSON tree and the partial file so the message never
 * holds a pointer into freed memory. */
static int safetensors_header_fail(ternary_json_value *root, ternary_source_file *file, char *error,
                                   size_t error_capacity, const char *format, ...) {
    va_list args;
    if (error && error_capacity) {
        va_start(args, format);
        vsnprintf(error, error_capacity, format, args);
        va_end(args);
        error[error_capacity - 1] = 0;
    }
    ternary_json_value_clear(root);
    ternary_source_file_clear(file);
    return 0;
}

int ternary_parse_safetensors_header(const char *json, uint64_t header_length, uint64_t file_size, const char *path,
                                     ternary_source_file *out, char *error, size_t error_capacity) {
    char *text = NULL;
    size_t text_length;
    ternary_json_value root;
    ternary_source_file file;
    uint64_t data_size;
    size_t m, dti;

    if (out) memset(out, 0, sizeof(*out));
    memset(&root, 0, sizeof(root));
    memset(&file, 0, sizeof(file));
    if (!json) return ternary_source_file_fail(out, error, error_capacity, "safetensors: no header");
    if (file_size < 8 || header_length > file_size - 8)
        return ternary_source_file_fail(out, error, error_capacity, "safetensors: header longer than the file");
    if (header_length < 2 || header_length > TERNARY_SAFETENSORS_MAX_HEADER_BYTES)
        return ternary_source_file_fail(out, error, error_capacity, "safetensors: header length out of range");
    if (json[0] != '{')
        return ternary_source_file_fail(out, error, error_capacity, "safetensors: header is not a JSON object");

    text_length = (size_t)header_length;
    text = (char *)malloc(text_length + 1);
    if (!text) return ternary_source_file_fail(out, error, error_capacity, "safetensors: out of memory");
    memcpy(text, json, text_length);
    text[text_length] = 0;
    while (text_length && text[text_length - 1] == ' ') text[--text_length] = 0;

    {
        char json_error[TERNARY_CONVERT_ERROR_CAPACITY];
        json_error[0] = 0;
        if (!ternary_json_parse(text, text_length, 4, TERNARY_SAFETENSORS_MAX_HEADER_BYTES, 1, &root, json_error,
                                sizeof(json_error))) {
            free(text);
            ternary_set_errorf(error, error_capacity, "safetensors: %s", json_error);
            if (out) ternary_source_file_clear(out);
            return 0;
        }
    }
    free(text);
    if (root.type != TERNARY_JSON_OBJECT) {
        ternary_json_value_clear(&root);
        return ternary_source_file_fail(out, error, error_capacity, "safetensors: header is not a JSON object");
    }
    if (root.member_count > TERNARY_SOURCE_MAX_TENSORS + 1) {
        ternary_json_value_clear(&root);
        return ternary_source_file_fail(out, error, error_capacity, "safetensors: too many tensors");
    }

    file.format = TERNARY_SOURCE_FORMAT_SAFETENSORS;
    file.path = ternary_duplicate_string(path ? path : "", path ? strlen(path) : 0);
    file.file_size = file_size;
    file.data_begin = 8 + header_length;
    file.data_end = file_size;
    data_size = file.data_end - file.data_begin;
    if (root.member_count) {
        file.tensors = (ternary_source_tensor *)calloc(root.member_count, sizeof(*file.tensors));
        if (!file.tensors) {
            ternary_json_value_clear(&root);
            ternary_source_file_clear(&file);
            return ternary_source_file_fail(out, error, error_capacity, "safetensors: out of memory");
        }
    }

    for (m = 0; m < root.member_count; ++m) {
        const ternary_json_member *member = &root.members[m];
        const ternary_json_value *v = member->value;
        const ternary_json_value *dtype, *shape, *offsets;
        const safetensors_dtype *info = NULL;
        ternary_source_tensor *tensor;
        uint64_t count = 1, bytes = 0, begin = 0, end = 0, dim, s;
        size_t n;
        char tensor_name[TERNARY_SOURCE_MAX_NAME_BYTES + 1];
        if (member->key_length == 12 && memcmp(member->key, "__metadata__", 12) == 0) {
            size_t k;
            if (!v || v->type != TERNARY_JSON_OBJECT) {
                ternary_json_value_clear(&root);
                ternary_source_file_clear(&file);
                return ternary_source_file_fail(out, error, error_capacity, "safetensors: __metadata__ is not an object");
            }
            for (k = 0; k < v->member_count; ++k)
                if (!v->members[k].value || v->members[k].value->type != TERNARY_JSON_STRING) {
                    ternary_json_value_clear(&root);
                    ternary_source_file_clear(&file);
                    return ternary_source_file_fail(out, error, error_capacity,
                                                    "safetensors: __metadata__ values must be strings");
                }
            continue;
        }
        if (member->key_length == 0 || member->key_length > TERNARY_SOURCE_MAX_NAME_BYTES ||
            memchr(member->key, 0, member->key_length)) {
            return safetensors_header_fail(&root, &file, error, error_capacity, "safetensors: bad tensor name");
        }
        memcpy(tensor_name, member->key, member->key_length);
        tensor_name[member->key_length] = 0;
        if (!v || v->type != TERNARY_JSON_OBJECT || v->member_count != 3) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: expected dtype, shape and data_offsets",
                                           tensor_name);
        }
        dtype = ternary_json_object_get(v, "dtype");
        shape = ternary_json_object_get(v, "shape");
        offsets = ternary_json_object_get(v, "data_offsets");
        if (!dtype || !shape || !offsets) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: expected dtype, shape and data_offsets",
                                           tensor_name);
        }
        if (dtype->type != TERNARY_JSON_STRING) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: dtype is not a string", tensor_name);
        }
        for (dti = 0; dti < sizeof(k_safetensors_dtypes) / sizeof(k_safetensors_dtypes[0]); ++dti)
            if (dtype->string_length == strlen(k_safetensors_dtypes[dti].name) &&
                memcmp(dtype->string_bytes, k_safetensors_dtypes[dti].name, dtype->string_length) == 0)
                info = &k_safetensors_dtypes[dti];
        if (!info) {
            char shown[33];
            size_t show = dtype->string_length < 32 ? dtype->string_length : 32;
            memcpy(shown, dtype->string_bytes ? dtype->string_bytes : "", show);
            shown[show] = 0;
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: unknown dtype %s", tensor_name, shown);
        }
        if (shape->type != TERNARY_JSON_ARRAY || shape->element_count > TERNARY_SOURCE_MAX_DIMENSIONS) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: bad shape", tensor_name);
        }
        tensor = &file.tensors[file.tensor_count];
        memset(tensor, 0, sizeof(*tensor));
        tensor->name = ternary_duplicate_string(member->key, member->key_length);
        n = strlen(info->name);
        if (n > sizeof(tensor->type_name) - 1) n = sizeof(tensor->type_name) - 1;
        memcpy(tensor->type_name, info->name, n);
        tensor->type_name[n] = 0;
        tensor->type = info->type;
        for (s = 0; s < shape->element_count; ++s) {
            if (!json_whole_number(&shape->elements[s], &dim)) {
                return safetensors_header_fail(&root, &file, error, error_capacity,
                                               "safetensors: tensor %s: a dimension is not a whole number >= 0",
                                               tensor_name);
            }
            if (ternary_multiply_overflows(count, dim, &count)) {
                return safetensors_header_fail(&root, &file, error, error_capacity,
                                               "safetensors: tensor %s: element count overflows", tensor_name);
            }
            tensor->shape[tensor->dimension_count++] = dim;
        }
        if (ternary_multiply_overflows(count, info->size, &bytes)) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: size overflows", tensor_name);
        }
        if (offsets->type != TERNARY_JSON_ARRAY || offsets->element_count != 2) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: bad data_offsets", tensor_name);
        }
        if (!json_whole_number(&offsets->elements[0], &begin) || !json_whole_number(&offsets->elements[1], &end)) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: data_offsets are not whole numbers >= 0",
                                           tensor_name);
        }
        if (begin > end) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: data_offsets begin after end", tensor_name);
        }
        if (end > data_size) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: data past the end of the file", tensor_name);
        }
        if (end - begin != bytes) {
            return safetensors_header_fail(&root, &file, error, error_capacity,
                                           "safetensors: tensor %s: data_offsets do not match dtype and shape",
                                           tensor_name);
        }
        tensor->file_offset = file.data_begin + begin;
        tensor->byte_count = bytes;
        tensor->columns = tensor->dimension_count == 0 ? 1 : tensor->shape[tensor->dimension_count - 1];
        tensor->rows = tensor->columns ? count / tensor->columns : 0;
        tensor->bytes_per_row = tensor->columns * info->size;
        file.tensor_count++;
    }

    if (!ternary_check_tensor_layout(&file, error, error_capacity)) {
        char prefixed[TERNARY_CONVERT_ERROR_CAPACITY];
        ternary_set_errorf(prefixed, sizeof(prefixed), "safetensors: %s", error ? error : "");
        ternary_json_value_clear(&root);
        ternary_source_file_clear(&file);
        ternary_set_error(error, error_capacity, prefixed);
        return 0;
    }
    ternary_json_value_clear(&root);
    *out = file;
    return 1;
}

int ternary_open_safetensors(const char *path, ternary_source_file *out, char *error, size_t error_capacity) {
    FILE *file;
    uint64_t size, header_length = 0;
    uint8_t length_bytes[8];
    char *header;
    int ok;
    int i;
    if (out) memset(out, 0, sizeof(*out));
    file = ternary_open_file_read_only(path);
    if (!file) {
        ternary_set_errorf(error, error_capacity, "cannot open %s", path ? path : "");
        return 0;
    }
    size = ternary_file_length(file);
    if (size < 8 || ternary_seek64(file, 0) != 0) {
        fclose(file);
        ternary_set_error(error, error_capacity, "safetensors: file too short");
        return 0;
    }
    if (fread(length_bytes, 1, 8, file) != 8) {
        fclose(file);
        ternary_set_error(error, error_capacity, "safetensors: cannot read the header length");
        return 0;
    }
    for (i = 7; i >= 0; --i) header_length = (header_length << 8) | length_bytes[i];
    if (header_length > size - 8 || header_length < 2 || header_length > TERNARY_SAFETENSORS_MAX_HEADER_BYTES) {
        fclose(file);
        ternary_set_error(error, error_capacity, "safetensors: header length out of range");
        return 0;
    }
    header = (char *)malloc((size_t)header_length);
    if (!header) {
        fclose(file);
        ternary_set_error(error, error_capacity, "safetensors: out of memory");
        return 0;
    }
    if (fread(header, 1, (size_t)header_length, file) != (size_t)header_length) {
        free(header);
        fclose(file);
        ternary_set_error(error, error_capacity, "safetensors: short header");
        return 0;
    }
    fclose(file);
    ok = ternary_parse_safetensors_header(header, header_length, size, path, out, error, error_capacity);
    free(header);
    return ok;
}

int ternary_open_source(const char *path, ternary_source_file *out, char *error, size_t error_capacity) {
    FILE *file;
    uint8_t magic[4] = {0, 0, 0, 0};
    if (out) memset(out, 0, sizeof(*out));
    file = ternary_open_file_read_only(path);
    if (!file) {
        ternary_set_errorf(error, error_capacity, "cannot open %s", path ? path : "");
        return 0;
    }
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        ternary_set_errorf(error, error_capacity, "file too short: %s", path ? path : "");
        return 0;
    }
    fclose(file);
    if (memcmp(magic, "GGUF", 4) == 0) return ternary_open_gguf(path, out, error, error_capacity);
    return ternary_open_safetensors(path, out, error, error_capacity);
}

int ternary_tensor_is_convertible(const ternary_source_tensor *tensor, char *reason, size_t reason_capacity) {
    if (!tensor) {
        ternary_set_error(reason, reason_capacity, "type not decodable here");
        return 0;
    }
    if (tensor->type == TERNARY_SOURCE_WEIGHT_OTHER) {
        ternary_set_error(reason, reason_capacity, "type not decodable here");
        return 0;
    }
    if (tensor->dimension_count < 2) {
        ternary_set_error(reason, reason_capacity, "not a matrix");
        return 0;
    }
    if (tensor->rows == 0 || tensor->columns == 0) {
        ternary_set_error(reason, reason_capacity, "empty");
        return 0;
    }
    if (tensor->columns % (uint64_t)TERNARY_GROUP_WEIGHT_COUNT != 0) {
        ternary_set_error(reason, reason_capacity, "row length is not a multiple of 128");
        return 0;
    }
    if (tensor->rows > TERNARY_HARD_MAX_ROWS || tensor->columns > TERNARY_HARD_MAX_COLUMNS) {
        ternary_set_error(reason, reason_capacity, "over the ternary size ceiling");
        return 0;
    }
    if (tensor->bytes_per_row == 0 || tensor->bytes_per_row * tensor->rows != tensor->byte_count) {
        ternary_set_error(reason, reason_capacity, "size does not match the shape");
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------------------------------ decode */

ternary_convert_status ternary_decode_rows(const ternary_source_tensor *tensor, const uint8_t *raw, uint64_t row_count,
                                           float *out) {
    uint64_t r, c, columns;
    int column_count;
    if (!tensor || !raw || !out || tensor->columns == 0 || tensor->columns > TERNARY_HARD_MAX_COLUMNS)
        return TERNARY_CONVERT_BAD_ARGUMENT;
    columns = tensor->columns;
    column_count = (int)columns;
    for (r = 0; r < row_count; ++r) {
        const uint8_t *src = raw + r * tensor->bytes_per_row;
        float *dst = out + r * columns;
        switch (tensor->type) {
            case TERNARY_SOURCE_WEIGHT_F32:
                memcpy(dst, src, (size_t)columns * 4);
                break;
            case TERNARY_SOURCE_WEIGHT_F16:
                for (c = 0; c < columns; ++c)
                    dst[c] = ternary_float16_to_float32((uint16_t)(src[2 * c] | (src[2 * c + 1] << 8)));
                break;
            case TERNARY_SOURCE_WEIGHT_BF16:
                for (c = 0; c < columns; ++c)
                    dst[c] = ternary_bfloat16_to_float32((uint16_t)(src[2 * c] | (src[2 * c + 1] << 8)));
                break;
            case TERNARY_SOURCE_WEIGHT_Q8_0:
                if (columns % 32) return TERNARY_CONVERT_BAD_ARGUMENT;
                ternary_dequantize_q8_0_row((const ternary_block_q8_0 *)src, dst, column_count);
                break;
            case TERNARY_SOURCE_WEIGHT_Q4_K:
                if (columns % TERNARY_KQUANT_SUPERBLOCK_WEIGHTS) return TERNARY_CONVERT_BAD_ARGUMENT;
                ternary_dequantize_q4_k_row((const ternary_block_q4_k *)src, dst, column_count);
                break;
            case TERNARY_SOURCE_WEIGHT_Q6_K:
                if (columns % TERNARY_KQUANT_SUPERBLOCK_WEIGHTS) return TERNARY_CONVERT_BAD_ARGUMENT;
                ternary_dequantize_q6_k_row((const ternary_block_q6_k *)src, dst, column_count);
                break;
            default:
                return TERNARY_CONVERT_BAD_ARGUMENT;
        }
        for (c = 0; c < columns; ++c)
            if (!isfinite(dst[c])) return TERNARY_CONVERT_NON_FINITE;
    }
    return TERNARY_CONVERT_OK;
}

int ternary_read_rows(FILE *file, const ternary_source_file *source, const ternary_source_tensor *tensor,
                      uint64_t first_row, uint64_t row_count, uint8_t **raw_buffer, size_t *raw_capacity,
                      float *out, char *error, size_t error_capacity) {
    char why[TERNARY_CONVERT_ERROR_CAPACITY];
    uint64_t offset, length;
    why[0] = 0;
    if (!file || !out) {
        ternary_set_error(error, error_capacity, "read_rows: bad argument");
        return 0;
    }
    if (!ternary_tensor_is_convertible(tensor, why, sizeof(why))) {
        ternary_set_errorf(error, error_capacity, "%s: %s", tensor && tensor->name ? tensor->name : "", why);
        return 0;
    }
    if (first_row > tensor->rows || row_count > tensor->rows - first_row) {
        ternary_set_errorf(error, error_capacity, "%s: rows out of range", tensor->name ? tensor->name : "");
        return 0;
    }
    offset = tensor->file_offset + first_row * tensor->bytes_per_row;
    length = row_count * tensor->bytes_per_row;
    if (offset < source->data_begin || offset > source->data_end || length > source->data_end - offset) {
        ternary_set_errorf(error, error_capacity, "%s: read outside the data region", tensor->name ? tensor->name : "");
        return 0;
    }
    if (!raw_buffer || !raw_capacity) {
        ternary_set_error(error, error_capacity, "read_rows: bad argument");
        return 0;
    }
    if (*raw_capacity < (size_t)length) {
        uint8_t *grown = (uint8_t *)realloc(*raw_buffer, (size_t)length ? (size_t)length : 1);
        if (!grown) {
            ternary_set_error(error, error_capacity, "out of memory");
            return 0;
        }
        *raw_buffer = grown;
        *raw_capacity = (size_t)length;
    }
    if (ternary_seek64(file, offset) != 0 || fread(*raw_buffer, 1, (size_t)length, file) != (size_t)length) {
        ternary_set_errorf(error, error_capacity, "%s: short read (file changed?)", tensor->name ? tensor->name : "");
        return 0;
    }
    {
        const ternary_convert_status status = ternary_decode_rows(tensor, *raw_buffer, row_count, out);
        if (status != TERNARY_CONVERT_OK) {
            ternary_set_errorf(error, error_capacity, "%s: decode failed: %s", tensor->name ? tensor->name : "",
                               ternary_convert_status_name(status));
            return 0;
        }
    }
    return 1;
}

/* ------------------------------------------------------------------------------------------------ fit */

typedef struct fit_worker_context {
    const float *weights;
    int8_t *codes;
    float *scales;
    ternary_group_fit_kind fit_kind;
    uint8_t *bad_flags;
} fit_worker_context;

static void fit_worker(uint64_t begin, uint64_t end, int worker_index, void *context) {
    fit_worker_context *ctx = (fit_worker_context *)context;
    float sorted[TERNARY_GROUP_WEIGHT_COUNT];
    uint64_t group;
    for (group = begin; group < end; ++group) {
        const float *group_weights = ctx->weights + group * TERNARY_GROUP_WEIGHT_COUNT;
        int i, ok = 1;
        for (i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) ok = ok && isfinite(group_weights[i]);
        if (!ok) {
            ctx->bad_flags[worker_index] = 1;
            continue;
        }
        ctx->scales[group] = ternary_group_fit(group_weights, ctx->fit_kind, sorted,
                                               ctx->codes + group * TERNARY_GROUP_WEIGHT_COUNT);
    }
}

ternary_convert_status ternary_fit_groups_cpu(const float *weights, uint64_t group_count, ternary_group_fit_kind fit_kind,
                                              int thread_count, int8_t *codes, float *scales) {
    int workers;
    uint8_t *bad;
    int i;
    fit_worker_context ctx;
    if (!weights || !codes || !scales || group_count == 0 || (unsigned)fit_kind > 1)
        return TERNARY_CONVERT_BAD_ARGUMENT;
    workers = ternary_pick_thread_count(thread_count, group_count);
    bad = (uint8_t *)calloc((size_t)workers, 1);
    if (!bad) return TERNARY_CONVERT_BAD_ARGUMENT;
    ctx.weights = weights;
    ctx.codes = codes;
    ctx.scales = scales;
    ctx.fit_kind = fit_kind;
    ctx.bad_flags = bad;
    ternary_run_in_parallel(group_count, workers, fit_worker, &ctx);
    for (i = 0; i < workers; ++i) {
        if (bad[i]) {
            memset(codes, 0, (size_t)(group_count * TERNARY_GROUP_WEIGHT_COUNT));
            memset(scales, 0, (size_t)group_count * sizeof(float));
            free(bad);
            return TERNARY_CONVERT_NON_FINITE;
        }
    }
    free(bad);
    return TERNARY_CONVERT_OK;
}

/* ------------------------------------------------------------------------------------------------ stats */

double ternary_reconstruction_relative_error(const ternary_reconstruction_error *stats) {
    if (!stats || !(stats->sum_weight_squared > 0)) return 0.0;
    return sqrt(stats->sum_error_squared / stats->sum_weight_squared);
}

double ternary_reconstruction_cosine(const ternary_reconstruction_error *stats) {
    if (!stats || !(stats->sum_weight_squared > 0 && stats->sum_reconstruction_squared > 0)) return 0.0;
    return stats->sum_weight_times_reconstruction / sqrt(stats->sum_weight_squared * stats->sum_reconstruction_squared);
}

double ternary_reconstruction_rmse(const ternary_reconstruction_error *stats) {
    if (!stats || !stats->weight_count) return 0.0;
    return sqrt(stats->sum_error_squared / (double)stats->weight_count);
}

void ternary_reconstruction_error_add(ternary_reconstruction_error *into, const ternary_reconstruction_error *other) {
    if (!into || !other) return;
    into->weight_count += other->weight_count;
    into->zero_codes += other->zero_codes;
    into->zero_groups += other->zero_groups;
    into->group_count += other->group_count;
    into->sum_weight_squared += other->sum_weight_squared;
    into->sum_error_squared += other->sum_error_squared;
    into->sum_weight_times_reconstruction += other->sum_weight_times_reconstruction;
    into->sum_reconstruction_squared += other->sum_reconstruction_squared;
    if (other->max_absolute_error > into->max_absolute_error) into->max_absolute_error = other->max_absolute_error;
}

typedef struct stats_worker_context {
    const float *weights;
    const int8_t *codes;
    const float *scales;
    uint64_t columns;
    uint64_t blocks_per_row;
    ternary_reconstruction_error *per_row;
} stats_worker_context;

static void stats_worker(uint64_t begin, uint64_t end, int worker_index, void *context) {
    stats_worker_context *ctx = (stats_worker_context *)context;
    uint64_t row, block, i;
    (void)worker_index;
    for (row = begin; row < end; ++row) {
        ternary_reconstruction_error *row_stats = &ctx->per_row[row];
        for (block = 0; block < ctx->blocks_per_row; ++block) {
            const uint64_t group = row * ctx->blocks_per_row + block;
            const double gamma = (double)ctx->scales[group];
            if (ctx->scales[group] == 0.0f) row_stats->zero_groups++;
            row_stats->group_count++;
            for (i = 0; i < (uint64_t)TERNARY_GROUP_WEIGHT_COUNT; ++i) {
                const uint64_t k = group * TERNARY_GROUP_WEIGHT_COUNT + i;
                const double weight = (double)ctx->weights[k];
                const double reconstruction = gamma * (double)ctx->codes[k];
                const double err = weight - reconstruction;
                const double abs_err = fabs(err);
                row_stats->sum_weight_squared += weight * weight;
                row_stats->sum_error_squared += err * err;
                row_stats->sum_weight_times_reconstruction += weight * reconstruction;
                row_stats->sum_reconstruction_squared += reconstruction * reconstruction;
                if (abs_err > row_stats->max_absolute_error) row_stats->max_absolute_error = abs_err;
                if (ctx->codes[k] == 0) row_stats->zero_codes++;
            }
            row_stats->weight_count += TERNARY_GROUP_WEIGHT_COUNT;
        }
    }
}

void ternary_accumulate_reconstruction_error(const float *weights, const int8_t *codes, const float *scales,
                                             uint64_t rows, uint64_t columns, int thread_count,
                                             ternary_reconstruction_error *stats) {
    ternary_reconstruction_error *per_row;
    stats_worker_context ctx;
    uint64_t r;
    if (!weights || !codes || !scales || !stats || rows == 0 || columns == 0 ||
        columns % TERNARY_GROUP_WEIGHT_COUNT)
        return;
    per_row = (ternary_reconstruction_error *)calloc((size_t)rows, sizeof(*per_row));
    if (!per_row) return;
    ctx.weights = weights;
    ctx.codes = codes;
    ctx.scales = scales;
    ctx.columns = columns;
    ctx.blocks_per_row = columns / TERNARY_GROUP_WEIGHT_COUNT;
    ctx.per_row = per_row;
    ternary_run_in_parallel(rows, thread_count, stats_worker, &ctx);
    for (r = 0; r < rows; ++r) ternary_reconstruction_error_add(stats, &per_row[r]); /* fixed row order */
    free(per_row);
}

/* ------------------------------------------------------------------------------------------------ packing */

uint64_t ternary_i2_row_bytes(uint64_t columns) {
    if (columns == 0 || columns % TERNARY_GROUP_WEIGHT_COUNT != 0 || columns > TERNARY_HARD_MAX_COLUMNS) return 0;
    return columns / 4;
}

ternary_convert_status ternary_pack_row_i2(const int8_t *codes, uint64_t columns, uint8_t *packed) {
    uint64_t block, blocks;
    int j;
    if (!codes || !packed || ternary_i2_row_bytes(columns) == 0) return TERNARY_CONVERT_BAD_ARGUMENT;
    blocks = columns / TERNARY_GROUP_WEIGHT_COUNT;
    for (block = 0; block < blocks; ++block) {
        const int8_t *group = codes + block * TERNARY_GROUP_WEIGHT_COUNT;
        for (j = 0; j < TERNARY_GROUP_WEIGHT_COUNT; ++j)
            if ((unsigned)((uint8_t)(group[j] + 1)) > 2u) return TERNARY_CONVERT_BAD_CODE;
    }
    /* Four codes per byte: weights j, j+32, j+64, j+96 in bit pairs 0-1, 2-3, 4-5, 6-7, each stored as q+1. */
    for (block = 0; block < blocks; ++block) {
        const int8_t *group = codes + block * TERNARY_GROUP_WEIGHT_COUNT;
        uint8_t *dest = packed + block * TERNARY_I2_BLOCK_BYTES;
        for (j = 0; j < TERNARY_I2_BLOCK_BYTES; ++j) {
            const unsigned c0 = (unsigned)(uint8_t)(group[j] + 1);
            const unsigned c1 = (unsigned)(uint8_t)(group[j + 32] + 1);
            const unsigned c2 = (unsigned)(uint8_t)(group[j + 64] + 1);
            const unsigned c3 = (unsigned)(uint8_t)(group[j + 96] + 1);
            dest[j] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
        }
    }
    return TERNARY_CONVERT_OK;
}

ternary_convert_status ternary_unpack_row_i2(const uint8_t *packed, uint64_t columns, int8_t *codes) {
    uint64_t block, blocks;
    int j;
    if (!packed || !codes || ternary_i2_row_bytes(columns) == 0) return TERNARY_CONVERT_BAD_ARGUMENT;
    blocks = columns / TERNARY_GROUP_WEIGHT_COUNT;
    for (block = 0; block < blocks; ++block) {
        const uint8_t *src = packed + block * TERNARY_I2_BLOCK_BYTES;
        int8_t *group = codes + block * TERNARY_GROUP_WEIGHT_COUNT;
        for (j = 0; j < TERNARY_I2_BLOCK_BYTES; ++j) {
            const uint8_t byte = src[j];
            /* A 2-bit field is 11 (invalid code 3) iff both bits of the pair are set: (byte & (byte >> 1) & 0x55). */
            if ((byte & (byte >> 1) & 0x55u) != 0) return TERNARY_CONVERT_BAD_CODE;
            group[j] = (int8_t)((byte & 3u) - 1u);
            group[j + 32] = (int8_t)(((byte >> 2) & 3u) - 1u);
            group[j + 64] = (int8_t)(((byte >> 4) & 3u) - 1u);
            group[j + 96] = (int8_t)(((byte >> 6) & 3u) - 1u);
        }
    }
    return TERNARY_CONVERT_OK;
}

/* ------------------------------------------------------------------------------------------------ PTN1 */

enum { TERNARY_PTN1_VERSION = 1, TERNARY_PTN1_PACKING_I2 = 1, TERNARY_PTN1_SCALE_PER_BLOCK = 2,
       TERNARY_PTN1_ACTIVATION_ABSMAX = 1 };

static void put_u16_le(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32_le(uint8_t *p, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void put_u64_le(uint8_t *p, uint64_t v) {
    int i;
    for (i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint16_t get_u16_le(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t get_u32_le(const uint8_t *p) {
    uint32_t v = 0;
    int i;
    for (i = 3; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static uint64_t get_u64_le(const uint8_t *p) {
    uint64_t v = 0;
    int i;
    for (i = 7; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}
static uint64_t round_up_32(uint64_t v) { return (v + 31) & ~UINT64_C(31); }

uint64_t ternary_fnv1a64(const uint8_t *bytes, size_t length, uint64_t seed) {
    size_t i;
    uint64_t h = seed;
    if (!bytes && length) return seed;
    for (i = 0; i < length; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
    }
    return h;
}

static ternary_convert_status check_scales_le(const uint8_t *raw, uint64_t count) {
    uint64_t i;
    for (i = 0; i < count; ++i) {
        uint32_t bits = get_u32_le(raw + 4 * i);
        float scale;
        memcpy(&scale, &bits, 4);
        if (!isfinite(scale) || scale < 0.0f) return TERNARY_CONVERT_BAD_SCALE;
        if (scale != 0.0f && scale < ternary_float_from_bits(0x00800000u)) return TERNARY_CONVERT_BAD_SCALE;
    }
    return TERNARY_CONVERT_OK;
}

static ternary_convert_status validate_i2_codes(const uint8_t *codes, uint64_t rows, uint64_t columns) {
    const uint64_t row_bytes = ternary_i2_row_bytes(columns);
    uint64_t r, b, j, blocks;
    if (!row_bytes) return TERNARY_CONVERT_BAD_ARGUMENT;
    blocks = columns / TERNARY_GROUP_WEIGHT_COUNT;
    for (r = 0; r < rows; ++r) {
        const uint8_t *row = codes + r * row_bytes;
        for (b = 0; b < blocks; ++b)
            for (j = 0; j < TERNARY_I2_BLOCK_BYTES; ++j) {
                const uint8_t byte = row[b * TERNARY_I2_BLOCK_BYTES + j];
                if ((byte & (byte >> 1) & 0x55u) != 0) return TERNARY_CONVERT_BAD_CODE;
            }
    }
    return TERNARY_CONVERT_OK;
}

int ternary_serialize_ptn1(const ternary_packed_tensor *tensor, uint8_t **bytes, size_t *length,
                           char *error, size_t error_capacity) {
    uint64_t scale_count, codes_offset, total;
    uint8_t *buf;
    uint64_t i;
    if (bytes) *bytes = NULL;
    if (length) *length = 0;
    if (!tensor || !bytes || !length) {
        ternary_set_error(error, error_capacity, "serialize: bad argument");
        return 0;
    }
    scale_count = tensor->scale_count;
    codes_offset = round_up_32(TERNARY_PTN1_HEADER_BYTES + 4 * scale_count);
    total = codes_offset + tensor->packed_codes_bytes;
    buf = (uint8_t *)calloc((size_t)total, 1);
    if (!buf) {
        ternary_set_error(error, error_capacity, "serialize: out of memory");
        return 0;
    }
    memcpy(buf, "PTN1", 4);
    put_u16_le(buf + 4, TERNARY_PTN1_VERSION);
    buf[6] = TERNARY_PTN1_PACKING_I2;
    buf[7] = TERNARY_PTN1_SCALE_PER_BLOCK;
    put_u32_le(buf + 8, TERNARY_PTN1_HEADER_BYTES);
    put_u32_le(buf + 12, TERNARY_PTN1_ACTIVATION_ABSMAX);
    put_u64_le(buf + 16, tensor->rows);
    put_u64_le(buf + 24, tensor->columns);
    put_u64_le(buf + 32, TERNARY_PTN1_HEADER_BYTES);
    put_u64_le(buf + 40, scale_count);
    put_u64_le(buf + 48, codes_offset);
    put_u64_le(buf + 56, tensor->packed_codes_bytes);
    for (i = 0; i < scale_count; ++i) {
        uint32_t bits;
        memcpy(&bits, &tensor->scales[i], 4);
        put_u32_le(buf + TERNARY_PTN1_HEADER_BYTES + 4 * i, bits);
    }
    if (tensor->packed_codes_bytes && tensor->packed_codes)
        memcpy(buf + codes_offset, tensor->packed_codes, tensor->packed_codes_bytes);
    put_u64_le(buf + 64, ternary_fnv1a64(buf + TERNARY_PTN1_HEADER_BYTES, (size_t)(total - TERNARY_PTN1_HEADER_BYTES),
                                         1469598103934665603ull));
    *bytes = buf;
    *length = (size_t)total;
    return 1;
}

ternary_convert_status ternary_parse_ptn1(const uint8_t *buffer, size_t length, uint64_t max_bytes,
                                          ternary_ptn1_view *view) {
    uint64_t rows, columns, scales_offset, scale_count, codes_offset, codes_bytes, row_bytes, scales_end;
    ternary_convert_status status;
    int i;
    if (view) memset(view, 0, sizeof(*view));
    if (!buffer) return TERNARY_CONVERT_BAD_ARGUMENT;
    if ((uint64_t)length > max_bytes) return TERNARY_CONVERT_TOO_LARGE;
    if (length < TERNARY_PTN1_HEADER_BYTES) return TERNARY_CONVERT_TRUNCATED;
    if (memcmp(buffer, "PTN1", 4) != 0) return TERNARY_CONVERT_BAD_MAGIC;
    if (get_u16_le(buffer + 4) != TERNARY_PTN1_VERSION) return TERNARY_CONVERT_BAD_VERSION;
    if (buffer[6] != TERNARY_PTN1_PACKING_I2 || buffer[7] != TERNARY_PTN1_SCALE_PER_BLOCK)
        return TERNARY_CONVERT_BAD_HEADER;
    if (get_u32_le(buffer + 8) != TERNARY_PTN1_HEADER_BYTES ||
        get_u32_le(buffer + 12) != TERNARY_PTN1_ACTIVATION_ABSMAX)
        return TERNARY_CONVERT_BAD_HEADER;
    for (i = 72; i < 96; ++i)
        if (buffer[i] != 0) return TERNARY_CONVERT_BAD_HEADER;
    rows = get_u64_le(buffer + 16);
    columns = get_u64_le(buffer + 24);
    scales_offset = get_u64_le(buffer + 32);
    scale_count = get_u64_le(buffer + 40);
    codes_offset = get_u64_le(buffer + 48);
    codes_bytes = get_u64_le(buffer + 56);
    if (rows == 0 || columns == 0 || columns % TERNARY_GROUP_WEIGHT_COUNT != 0) return TERNARY_CONVERT_BAD_HEADER;
    if (rows > TERNARY_HARD_MAX_ROWS || columns > TERNARY_HARD_MAX_COLUMNS) return TERNARY_CONVERT_TOO_LARGE;
    row_bytes = ternary_i2_row_bytes(columns);
    if (scales_offset != TERNARY_PTN1_HEADER_BYTES || scale_count != rows * (columns / TERNARY_GROUP_WEIGHT_COUNT))
        return TERNARY_CONVERT_BAD_HEADER;
    scales_end = scales_offset + 4 * scale_count;
    if (codes_offset != round_up_32(scales_end) || codes_bytes != rows * row_bytes) return TERNARY_CONVERT_BAD_HEADER;
    if (codes_offset > (uint64_t)length || codes_bytes != (uint64_t)length - codes_offset)
        return ((uint64_t)length < codes_offset + codes_bytes) ? TERNARY_CONVERT_TRUNCATED : TERNARY_CONVERT_BAD_HEADER;
    for (i = 0; (uint64_t)i + scales_end < codes_offset; ++i)
        if (buffer[scales_end + (uint64_t)i] != 0) return TERNARY_CONVERT_BAD_HEADER;
    if (get_u64_le(buffer + 64) !=
        ternary_fnv1a64(buffer + TERNARY_PTN1_HEADER_BYTES, length - TERNARY_PTN1_HEADER_BYTES, 1469598103934665603ull))
        return TERNARY_CONVERT_BAD_CHECKSUM;
    status = check_scales_le(buffer + scales_offset, scale_count);
    if (status != TERNARY_CONVERT_OK) return status;
    status = validate_i2_codes(buffer + codes_offset, rows, columns);
    if (status != TERNARY_CONVERT_OK) return status;
    if (view) {
        view->rows = rows;
        view->columns = columns;
        view->packed_codes = buffer + codes_offset;
        view->scales_raw = buffer + scales_offset;
        view->bytes_per_row = row_bytes;
        view->scale_count = scale_count;
    }
    return TERNARY_CONVERT_OK;
}

static ternary_convert_status validate_packed_tensor(const ternary_packed_tensor *tensor) {
    uint64_t row_bytes, expected_scales;
    ternary_convert_status status;
    if (!tensor || !tensor->packed_codes || !tensor->scales) return TERNARY_CONVERT_BAD_ARGUMENT;
    if (tensor->rows == 0 || tensor->columns == 0 || tensor->columns % TERNARY_GROUP_WEIGHT_COUNT != 0)
        return TERNARY_CONVERT_BAD_ARGUMENT;
    if (tensor->rows > TERNARY_HARD_MAX_ROWS || tensor->columns > TERNARY_HARD_MAX_COLUMNS)
        return TERNARY_CONVERT_TOO_LARGE;
    row_bytes = ternary_i2_row_bytes(tensor->columns);
    expected_scales = tensor->rows * (tensor->columns / TERNARY_GROUP_WEIGHT_COUNT);
    if (tensor->packed_codes_bytes != tensor->rows * row_bytes || tensor->scale_count != expected_scales)
        return TERNARY_CONVERT_BAD_ARGUMENT;
    /* Scales are native-endian floats in memory; check them directly. */
    {
        uint64_t i;
        for (i = 0; i < tensor->scale_count; ++i) {
            const float scale = tensor->scales[i];
            if (!isfinite(scale) || scale < 0.0f) return TERNARY_CONVERT_BAD_SCALE;
            if (scale != 0.0f && scale < ternary_float_from_bits(0x00800000u)) return TERNARY_CONVERT_BAD_SCALE;
        }
    }
    status = validate_i2_codes(tensor->packed_codes, tensor->rows, tensor->columns);
    return status;
}

/* ------------------------------------------------------------------------------------------------ convert */

static ternary_convert_status run_fit(ternary_fit_backend backend, const float *weights, uint64_t group_count,
                                      ternary_group_fit_kind fit_kind, int thread_count, int8_t *codes, float *scales,
                                      char *error, size_t error_capacity) {
    if (backend == TERNARY_FIT_BACKEND_CUDA)
        return ternary_fit_groups_cuda(weights, group_count, fit_kind, codes, scales, error, error_capacity);
    return ternary_fit_groups_cpu(weights, group_count, fit_kind, thread_count, codes, scales);
}

int ternary_convert_tensor(FILE *file, const ternary_source_file *source, const ternary_source_tensor *tensor,
                           const ternary_convert_options *options, ternary_convert_result *result,
                           char *error, size_t error_capacity) {
    char why[TERNARY_CONVERT_ERROR_CAPACITY];
    uint64_t columns, rows, blocks_per_row, row_bytes, chunk_rows, first_row;
    uint8_t *raw = NULL;
    size_t raw_capacity = 0;
    float *weights = NULL;
    int8_t *codes = NULL, *parity_codes = NULL;
    float *parity_scales = NULL;
    ternary_convert_options opt;
    why[0] = 0;
    if (result) memset(result, 0, sizeof(*result));
    if (!result || !options) {
        ternary_set_error(error, error_capacity, "convert: bad argument");
        return 0;
    }
    if (!ternary_tensor_is_convertible(tensor, why, sizeof(why))) {
        ternary_set_errorf(error, error_capacity, "%s: %s", tensor && tensor->name ? tensor->name : "", why);
        return 0;
    }
    opt = *options;
    if (opt.chunk_weights == 0) opt.chunk_weights = UINT64_C(1) << 25;
    columns = tensor->columns;
    rows = tensor->rows;
    blocks_per_row = columns / TERNARY_GROUP_WEIGHT_COUNT;
    row_bytes = ternary_i2_row_bytes(columns);
    chunk_rows = opt.chunk_weights / columns;
    if (chunk_rows < 1) chunk_rows = 1;
    if (chunk_rows > rows) chunk_rows = rows;

    result->output.rows = rows;
    result->output.columns = columns;
    result->output.packed_codes_bytes = (size_t)(rows * row_bytes);
    result->output.scale_count = (size_t)(rows * blocks_per_row);
    result->output.packed_codes = (uint8_t *)calloc(result->output.packed_codes_bytes ? result->output.packed_codes_bytes : 1, 1);
    result->output.scales = (float *)calloc(result->output.scale_count ? result->output.scale_count : 1, sizeof(float));
    weights = (float *)malloc((size_t)(chunk_rows * columns) * sizeof(float));
    codes = (int8_t *)malloc((size_t)(chunk_rows * columns));
    if (!result->output.packed_codes || !result->output.scales || !weights || !codes) {
        free(weights);
        free(codes);
        ternary_convert_result_clear(result);
        ternary_set_error(error, error_capacity, "convert: out of memory");
        return 0;
    }
    if (opt.check_parity) {
        parity_codes = (int8_t *)malloc((size_t)(chunk_rows * columns));
        parity_scales = (float *)malloc((size_t)(chunk_rows * blocks_per_row) * sizeof(float));
        if (!parity_codes || !parity_scales) {
            free(weights);
            free(codes);
            free(parity_codes);
            free(parity_scales);
            ternary_convert_result_clear(result);
            ternary_set_error(error, error_capacity, "convert: out of memory");
            return 0;
        }
    }

    for (first_row = 0; first_row < rows; first_row += chunk_rows) {
        const uint64_t n = (chunk_rows < rows - first_row) ? chunk_rows : (rows - first_row);
        const uint64_t groups = n * blocks_per_row;
        double t0 = monotonic_milliseconds();
        float *group_scales = result->output.scales + first_row * blocks_per_row;
        char fit_error[TERNARY_CONVERT_ERROR_CAPACITY];
        ternary_convert_status status;
        uint64_t r, g;
        fit_error[0] = 0;
        if (!ternary_read_rows(file, source, tensor, first_row, n, &raw, &raw_capacity, weights, error, error_capacity)) {
            free(weights);
            free(codes);
            free(parity_codes);
            free(parity_scales);
            free(raw);
            ternary_convert_result_clear(result);
            return 0;
        }
        result->read_milliseconds += monotonic_milliseconds() - t0;

        t0 = monotonic_milliseconds();
        status = run_fit(opt.backend, weights, groups, opt.fit_kind, opt.thread_count, codes, group_scales,
                         fit_error, sizeof(fit_error));
        result->fit_milliseconds += monotonic_milliseconds() - t0;
        if (status != TERNARY_CONVERT_OK) {
            ternary_set_errorf(error, error_capacity, "%s: fit failed: %s%s%s", tensor->name ? tensor->name : "",
                               ternary_convert_status_name(status), fit_error[0] ? " (" : "",
                               fit_error[0] ? fit_error : "");
            if (fit_error[0]) {
                /* already included */
            }
            free(weights);
            free(codes);
            free(parity_codes);
            free(parity_scales);
            free(raw);
            ternary_convert_result_clear(result);
            return 0;
        }

        if (opt.check_parity) {
            const ternary_fit_backend other =
                (opt.backend == TERNARY_FIT_BACKEND_CUDA) ? TERNARY_FIT_BACKEND_CPU : TERNARY_FIT_BACKEND_CUDA;
            t0 = monotonic_milliseconds();
            status = run_fit(other, weights, groups, opt.fit_kind, opt.thread_count, parity_codes, parity_scales,
                             fit_error, sizeof(fit_error));
            result->other_fit_milliseconds += monotonic_milliseconds() - t0;
            if (status != TERNARY_CONVERT_OK) {
                ternary_set_errorf(error, error_capacity, "%s: parity fit failed: %s%s%s",
                                   tensor->name ? tensor->name : "", ternary_convert_status_name(status),
                                   fit_error[0] ? " (" : "", fit_error[0] ? fit_error : "");
                free(weights);
                free(codes);
                free(parity_codes);
                free(parity_scales);
                free(raw);
                ternary_convert_result_clear(result);
                return 0;
            }
            for (g = 0; g < groups; ++g) {
                const int same =
                    memcmp(codes + g * TERNARY_GROUP_WEIGHT_COUNT, parity_codes + g * TERNARY_GROUP_WEIGHT_COUNT,
                           TERNARY_GROUP_WEIGHT_COUNT) == 0 &&
                    memcmp(group_scales + g, parity_scales + g, sizeof(float)) == 0;
                if (!same) result->parity_groups_different++;
            }
            result->parity_groups_checked += groups;
        }

        t0 = monotonic_milliseconds();
        ternary_accumulate_reconstruction_error(weights, codes, group_scales, n, columns, opt.thread_count,
                                                &result->stats);
        result->stats_milliseconds += monotonic_milliseconds() - t0;
        for (r = 0; r < n; ++r) {
            status = ternary_pack_row_i2(codes + r * columns, columns,
                                         result->output.packed_codes + (first_row + r) * row_bytes);
            if (status != TERNARY_CONVERT_OK) {
                ternary_set_errorf(error, error_capacity, "%s: pack failed: %s", tensor->name ? tensor->name : "",
                                   ternary_convert_status_name(status));
                free(weights);
                free(codes);
                free(parity_codes);
                free(parity_scales);
                free(raw);
                ternary_convert_result_clear(result);
                return 0;
            }
        }
    }

    free(weights);
    free(codes);
    free(parity_codes);
    free(parity_scales);
    free(raw);
    {
        const ternary_convert_status status = validate_packed_tensor(&result->output);
        if (status != TERNARY_CONVERT_OK) {
            ternary_set_errorf(error, error_capacity, "%s: packed tensor did not validate: %s",
                               tensor->name ? tensor->name : "", ternary_convert_status_name(status));
            ternary_convert_result_clear(result);
            return 0;
        }
    }
    return 1;
}
