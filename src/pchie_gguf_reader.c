/* Bolt PCHIE - read-only GGUF v3 reader for the ternary converter (clean-room, C17).
 *
 * Parses the header, skips metadata (except general.alignment) and reads the tensor directory. It never writes
 * to the file. Counts, string lengths, the alignment, and every tensor's shape, offset and byte extent are
 * checked against the file size before anything is allocated. A tensor of unknown size, or two tensors that
 * overlap, is a refusal (those could hide an overlap).
 */
#include "pchie_ternary_convert_internal.h"

#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

static const uint32_t k_gguf_scalar_size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

int ternary_ggml_type_block(uint32_t type, uint64_t *block_elements, uint64_t *block_bytes) {
    uint64_t elements = 0, bytes = 0;
    switch (type) {
        case 0: elements = 1; bytes = 4; break;       /* F32 */
        case 1: elements = 1; bytes = 2; break;       /* F16 */
        case 2: elements = 32; bytes = 18; break;     /* Q4_0 */
        case 3: elements = 32; bytes = 20; break;     /* Q4_1 */
        case 6: elements = 32; bytes = 22; break;     /* Q5_0 */
        case 7: elements = 32; bytes = 24; break;     /* Q5_1 */
        case 8: elements = 32; bytes = 34; break;     /* Q8_0 */
        case 9: elements = 32; bytes = 36; break;     /* Q8_1 */
        case 10: elements = 256; bytes = 84; break;   /* Q2_K */
        case 11: elements = 256; bytes = 110; break;  /* Q3_K */
        case 12: elements = 256; bytes = 144; break;  /* Q4_K */
        case 13: elements = 256; bytes = 176; break;  /* Q5_K */
        case 14: elements = 256; bytes = 210; break;  /* Q6_K */
        case 15: elements = 256; bytes = 292; break;  /* Q8_K */
        case 24: elements = 1; bytes = 1; break;      /* I8 */
        case 25: elements = 1; bytes = 2; break;      /* I16 */
        case 26: elements = 1; bytes = 4; break;      /* I32 */
        case 27: elements = 1; bytes = 8; break;      /* I64 */
        case 28: elements = 1; bytes = 8; break;      /* F64 */
        case 30: elements = 1; bytes = 2; break;      /* BF16 */
        default:
            *block_elements = 0;
            *block_bytes = 0;
            return 0;
    }
    *block_elements = elements;
    *block_bytes = bytes;
    return 1;
}

const char *ternary_ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 8: return "Q8_0";
        case 12: return "Q4_K";
        case 14: return "Q6_K";
        case 30: return "BF16";
        default: return "other";
    }
}

typedef struct gguf_reader {
    FILE *file;
    uint64_t file_size;
    int ok;
} gguf_reader;

static uint64_t gguf_tell(gguf_reader *reader) {
#if defined(_WIN32)
    const long long p = _ftelli64(reader->file);
#else
    const off_t p = ftello(reader->file);
#endif
    return p < 0 ? ~UINT64_C(0) : (uint64_t)p;
}

static uint64_t gguf_remaining(gguf_reader *reader) {
    const uint64_t pos = gguf_tell(reader);
    return (pos == ~UINT64_C(0) || pos > reader->file_size) ? 0 : reader->file_size - pos;
}

static int gguf_read(gguf_reader *reader, void *dest, size_t bytes) {
    if (!reader->ok) return 0;
    if (bytes && fread(dest, 1, bytes, reader->file) != bytes) {
        reader->ok = 0;
        return 0;
    }
    return 1;
}

static uint8_t gguf_u8(gguf_reader *reader) {
    uint8_t v = 0;
    gguf_read(reader, &v, 1);
    return v;
}
static int8_t gguf_i8(gguf_reader *reader) { return (int8_t)gguf_u8(reader); }
static uint16_t gguf_u16(gguf_reader *reader) {
    uint8_t b[2];
    gguf_read(reader, b, 2);
    return (uint16_t)(b[0] | (b[1] << 8));
}
static int16_t gguf_i16(gguf_reader *reader) { return (int16_t)gguf_u16(reader); }
static uint32_t gguf_u32(gguf_reader *reader) {
    uint8_t b[4];
    uint32_t v = 0;
    int i;
    gguf_read(reader, b, 4);
    for (i = 3; i >= 0; --i) v = (v << 8) | b[i];
    return v;
}
static int32_t gguf_i32(gguf_reader *reader) { return (int32_t)gguf_u32(reader); }
static uint64_t gguf_u64(gguf_reader *reader) {
    uint8_t b[8];
    uint64_t v = 0;
    int i;
    gguf_read(reader, b, 8);
    for (i = 7; i >= 0; --i) v = (v << 8) | b[i];
    return v;
}
static int64_t gguf_i64(gguf_reader *reader) { return (int64_t)gguf_u64(reader); }
static float gguf_f32(gguf_reader *reader) {
    uint32_t bits = gguf_u32(reader);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}
static double gguf_f64(gguf_reader *reader) {
    uint64_t bits = gguf_u64(reader);
    double v;
    memcpy(&v, &bits, 8);
    return v;
}

static int gguf_skip(gguf_reader *reader, uint64_t n) {
    if (!reader->ok) return 0;
#if defined(_WIN32)
    if (_fseeki64(reader->file, (long long)n, SEEK_CUR) != 0) { reader->ok = 0; return 0; }
#else
    if (fseeko(reader->file, (off_t)n, SEEK_CUR) != 0) { reader->ok = 0; return 0; }
#endif
    return 1;
}

/* Reads a GGUF string. If dest is NULL the bytes are skipped. dest_length receives the length. */
static int gguf_string(gguf_reader *reader, char **dest, size_t *dest_length) {
    const uint64_t n = gguf_u64(reader);
    if (!reader->ok || n > (UINT64_C(1) << 24) || n > gguf_remaining(reader)) {
        reader->ok = 0;
        if (dest) *dest = NULL;
        if (dest_length) *dest_length = 0;
        return 0;
    }
    if (dest_length) *dest_length = (size_t)n;
    if (dest) {
        *dest = ternary_duplicate_string("", 0);
        if (n) {
            char *s = (char *)malloc((size_t)n + 1);
            if (!s) { reader->ok = 0; return 0; }
            if (!gguf_read(reader, s, (size_t)n)) { free(s); return 0; }
            s[n] = 0;
            free(*dest);
            *dest = s;
        }
        return reader->ok;
    }
    return n ? gguf_skip(reader, n) : 1;
}

static double gguf_read_number(gguf_reader *reader, uint32_t type) {
    switch (type) {
        case 0: return (double)gguf_u8(reader);
        case 1: return (double)gguf_i8(reader);
        case 2: return (double)gguf_u16(reader);
        case 3: return (double)gguf_i16(reader);
        case 4: return (double)gguf_u32(reader);
        case 5: return (double)gguf_i32(reader);
        case 6: return (double)gguf_f32(reader);
        case 7: return (double)gguf_u8(reader);
        case 10: return (double)gguf_u64(reader);
        case 11: return (double)gguf_i64(reader);
        case 12: return gguf_f64(reader);
        default:
            reader->ok = 0;
            return 0;
    }
}

int ternary_open_gguf(const char *path, ternary_source_file *out, char *error, size_t error_capacity) {
    gguf_reader reader;
    FILE *file;
    uint32_t magic, version;
    uint64_t tensor_count, key_count, i, alignment = 32, pos, data_offset, data_size;
    int have_alignment = 0;
    double alignment_value = 0;
    ternary_source_file file_info;

    memset(&file_info, 0, sizeof(file_info));
    if (out) memset(out, 0, sizeof(*out));
    file = ternary_open_file_read_only(path);
    if (!file) {
        ternary_set_errorf(error, error_capacity, "cannot open %s", path ? path : "");
        return 0;
    }
    memset(&reader, 0, sizeof(reader));
    reader.file = file;
    reader.ok = 1;
    reader.file_size = ternary_file_length(file);
    if (reader.file_size < 24 || ternary_seek64(file, 0) != 0) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: truncated or malformed GGUF");
    }
    reader.file_size = reader.file_size; /* already measured from the start after length() left us at SEEK_END */

    magic = gguf_u32(&reader);
    if (magic != 0x46554747u) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: bad magic");
    }
    version = gguf_u32(&reader);
    if (version != 3) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: unsupported GGUF version");
    }
    tensor_count = gguf_u64(&reader);
    key_count = gguf_u64(&reader);
    if (!reader.ok || key_count > gguf_remaining(&reader) / 12 || tensor_count > gguf_remaining(&reader) / 24 ||
        tensor_count > (UINT64_C(1) << 20) || key_count > (UINT64_C(1) << 20)) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: GGUF header counts exceed the file");
    }

    for (i = 0; i < key_count && reader.ok; ++i) {
        char *key = NULL;
        size_t key_length = 0;
        uint32_t type;
        if (!gguf_string(&reader, &key, &key_length)) {
            free(key);
            break;
        }
        type = gguf_u32(&reader);
        if (!reader.ok) { free(key); break; }
        if (type == 8) {
            if (!gguf_string(&reader, NULL, NULL)) { free(key); break; }
        } else if (type == 9) {
            const uint32_t element_type = gguf_u32(&reader);
            const uint64_t n = gguf_u64(&reader);
            uint64_t j;
            if (!reader.ok) { free(key); break; }
            if (n > (UINT64_C(1) << 26)) { reader.ok = 0; free(key); break; }
            if (element_type == 8) {
                if (n > gguf_remaining(&reader) / 8) { reader.ok = 0; free(key); break; }
                for (j = 0; j < n && reader.ok; ++j)
                    if (!gguf_string(&reader, NULL, NULL)) break;
            } else if (element_type < 13 && k_gguf_scalar_size[element_type]) {
                if (n > gguf_remaining(&reader) / k_gguf_scalar_size[element_type]) {
                    reader.ok = 0;
                    free(key);
                    break;
                }
                if (!gguf_skip(&reader, n * k_gguf_scalar_size[element_type])) { free(key); break; }
            } else {
                reader.ok = 0;
                free(key);
                break;
            }
        } else {
            const double number = gguf_read_number(&reader, type);
            if (key && key_length == 17 && memcmp(key, "general.alignment", 17) == 0) {
                have_alignment = 1;
                alignment_value = number;
            }
        }
        free(key);
    }
    if (!reader.ok) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: truncated or malformed GGUF");
    }
    if (have_alignment) {
        if (!(alignment_value >= 1.0 && alignment_value <= (double)(1u << 20)) ||
            alignment_value != (double)(uint64_t)alignment_value) {
            fclose(file);
            return ternary_source_file_fail(out, error, error_capacity, "GGUF: GGUF alignment out of range");
        }
        alignment = (uint64_t)alignment_value;
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: GGUF alignment must be a nonzero power of two");
    }
    if (tensor_count > gguf_remaining(&reader) / 24) {
        fclose(file);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: GGUF tensor count exceeds the file");
    }

    file_info.format = TERNARY_SOURCE_FORMAT_GGUF;
    file_info.path = ternary_duplicate_string(path ? path : "", path ? strlen(path) : 0);
    file_info.file_size = reader.file_size;
    file_info.tensors = NULL;
    file_info.tensor_count = 0;
    if (tensor_count) {
        file_info.tensors = (ternary_source_tensor *)calloc((size_t)tensor_count, sizeof(*file_info.tensors));
        if (!file_info.tensors) {
            fclose(file);
            ternary_source_file_clear(&file_info);
            return ternary_source_file_fail(out, error, error_capacity, "GGUF: out of memory");
        }
    }

    for (i = 0; i < tensor_count; ++i) {
        ternary_source_tensor *tensor = &file_info.tensors[i];
        uint32_t dimension_count, type;
        uint64_t dims[8];
        uint32_t d;
        size_t name_length = 0;
        if (!gguf_string(&reader, &tensor->name, &name_length)) break;
        dimension_count = gguf_u32(&reader);
        if (!reader.ok || dimension_count == 0 || dimension_count > 4) { reader.ok = 0; break; }
        for (d = 0; d < dimension_count; ++d) dims[d] = gguf_u64(&reader);
        type = gguf_u32(&reader);
        tensor->file_offset = gguf_u64(&reader); /* relative for now; adjusted after data_offset is known */
        if (!reader.ok) break;
        tensor->dimension_count = dimension_count;
        /* GGUF dims[0] is innermost; we store outermost first. */
        for (d = 0; d < dimension_count; ++d) tensor->shape[d] = dims[dimension_count - 1 - d];
        {
            const char *type_name = ternary_ggml_type_name(type);
            size_t n = strlen(type_name);
            if (n > sizeof(tensor->type_name) - 1) n = sizeof(tensor->type_name) - 1;
            memcpy(tensor->type_name, type_name, n);
            tensor->type_name[n] = 0;
        }
        switch (type) {
            case 0: tensor->type = TERNARY_SOURCE_WEIGHT_F32; break;
            case 1: tensor->type = TERNARY_SOURCE_WEIGHT_F16; break;
            case 8: tensor->type = TERNARY_SOURCE_WEIGHT_Q8_0; break;
            case 12: tensor->type = TERNARY_SOURCE_WEIGHT_Q4_K; break;
            case 14: tensor->type = TERNARY_SOURCE_WEIGHT_Q6_K; break;
            case 30: tensor->type = TERNARY_SOURCE_WEIGHT_BF16; break;
            default: tensor->type = TERNARY_SOURCE_WEIGHT_OTHER; break;
        }
        /* Stash the relative offset in file_offset; type in unused bytes_per_row until sized below. */
        tensor->bytes_per_row = type; /* temporary */
        file_info.tensor_count = (size_t)(i + 1);
    }
    if (!reader.ok) {
        fclose(file);
        ternary_source_file_clear(&file_info);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: truncated or malformed GGUF");
    }

    pos = gguf_tell(&reader);
    fclose(file);
    if (pos == ~UINT64_C(0) || pos > reader.file_size) {
        ternary_source_file_clear(&file_info);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: truncated or malformed GGUF");
    }
    data_offset = (pos + alignment - 1) / alignment * alignment;
    if (data_offset < pos || data_offset > reader.file_size) {
        ternary_source_file_clear(&file_info);
        return ternary_source_file_fail(out, error, error_capacity, "GGUF: GGUF data section starts past the end of the file");
    }
    data_size = reader.file_size - data_offset;
    file_info.data_begin = data_offset;
    file_info.data_end = reader.file_size;

    for (i = 0; i < file_info.tensor_count; ++i) {
        ternary_source_tensor *tensor = &file_info.tensors[i];
        const uint32_t type = (uint32_t)tensor->bytes_per_row;
        uint64_t n = 1, d, relative = tensor->file_offset, block_elements = 0, block_bytes = 0;
        int size_known;
        tensor->bytes_per_row = 0;
        for (d = 0; d < tensor->dimension_count; ++d) {
            /* dims were reversed; GGUF product uses the original dims, which is the same set. Zero is refused. */
            if (tensor->shape[d] == 0 || ternary_multiply_overflows(n, tensor->shape[d], &n)) {
                ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: bad shape", tensor->name ? tensor->name : "");
                ternary_source_file_clear(&file_info);
                return 0;
            }
        }
        if (relative % alignment != 0) {
            ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: offset is not aligned", tensor->name ? tensor->name : "");
            ternary_source_file_clear(&file_info);
            return 0;
        }
        if (relative > data_size) {
            ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: offset past the end of the file",
                               tensor->name ? tensor->name : "");
            ternary_source_file_clear(&file_info);
            return 0;
        }
        size_known = ternary_ggml_type_block(type, &block_elements, &block_bytes);
        if (!size_known) {
            ternary_set_errorf(error, error_capacity, "GGUF: tensor %s has a type of unknown size (%s); refusing the file",
                               tensor->name ? tensor->name : "", ternary_ggml_type_name(type));
            ternary_source_file_clear(&file_info);
            return 0;
        }
        /* Innermost dimension is shape[dimension_count-1], which is GGUF dims[0]. */
        {
            const uint64_t innermost = tensor->shape[tensor->dimension_count - 1];
            uint64_t bytes = 0;
            if (innermost % block_elements != 0) {
                ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: row is not a whole number of blocks",
                                   tensor->name ? tensor->name : "");
                ternary_source_file_clear(&file_info);
                return 0;
            }
            if (ternary_multiply_overflows(n / block_elements, block_bytes, &bytes)) {
                ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: size overflows", tensor->name ? tensor->name : "");
                ternary_source_file_clear(&file_info);
                return 0;
            }
            if (bytes > data_size - relative) {
                ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: extends past the end of the file",
                                   tensor->name ? tensor->name : "");
                ternary_source_file_clear(&file_info);
                return 0;
            }
            tensor->byte_count = bytes;
            tensor->columns = innermost;
            tensor->rows = tensor->columns ? n / tensor->columns : 0;
            if (block_elements != 0 && tensor->columns % block_elements == 0)
                tensor->bytes_per_row = tensor->columns / block_elements * block_bytes;
            if (tensor->rows != 0 && tensor->bytes_per_row * tensor->rows != tensor->byte_count) {
                ternary_set_errorf(error, error_capacity, "GGUF: tensor %s: size does not match its shape",
                                   tensor->name ? tensor->name : "");
                ternary_source_file_clear(&file_info);
                return 0;
            }
        }
        tensor->file_offset = data_offset + relative;
    }

    if (!ternary_check_tensor_layout(&file_info, error, error_capacity)) {
        char prefixed[TERNARY_CONVERT_ERROR_CAPACITY];
        ternary_set_errorf(prefixed, sizeof(prefixed), "GGUF: %s", error ? error : "");
        ternary_source_file_clear(&file_info);
        ternary_set_error(error, error_capacity, prefixed);
        return 0;
    }
    *out = file_info;
    return 1;
}
