/* Bolt PCHIE - packed ternary storage: B3 rows, compact scales, rANS trit streams, PTT1 tensors and PTP1 packs
 * (clean-room, C17). See include/pchie/pchie_ternary_pack.h for the formats. Every parser treats its input as
 * untrusted: sizes are checked before use, all arithmetic on sizes is overflow-checked, and any failure leaves
 * the outputs zeroed. */
#include "pchie/pchie_ternary_pack.h"
#include "pchie_ternary_convert_internal.h"
#include "pchie_ternary_pack_tables.h"
#include "pchie_weight_dequantize.h"
#include "pj_sha256.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

enum { PTT1_VERSION = 1, PTP1_VERSION = 1 };
#define RANS_LOW (UINT32_C(1) << 23)

const char *ternary_trit_encoding_name(int encoding) {
    switch (encoding) {
        case TERNARY_TRITS_B3: return "b3";
        case TERNARY_TRITS_RANS: return "rans";
        default: return "-";
    }
}

const char *ternary_scale_encoding_name(int encoding) {
    switch (encoding) {
        case TERNARY_SCALES_F32: return "f32";
        case TERNARY_SCALES_F16: return "f16";
        case TERNARY_SCALES_U8ROW: return "u8row";
        default: return "-";
    }
}

/* ------------------------------------------------------------------------------------------------ bytes */

static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static void put64(uint8_t *p, uint64_t v) {
    int i;
    for (i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i));
}
static uint16_t get16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t get64(const uint8_t *p) { return (uint64_t)get32(p) | ((uint64_t)get32(p + 4) << 32); }
static float f32_from_bits(uint32_t b) {
    float f;
    memcpy(&f, &b, 4);
    return f;
}
static uint32_t bits_from_f32(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    return b;
}
static int add_overflows(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 1;
    *out = a + b;
    return 0;
}

void ternary_sha256(const uint8_t *data, size_t length, uint8_t out[32]) { pj_sha256_once(data, length, out); }

/* A scale a reader accepts: finite, >= 0, and 0 or normal. */
static int scale_ok(float s) { return s == 0.0f || (s >= FLT_MIN && s <= FLT_MAX); }

uint16_t ternary_float32_to_float16(float value) {
    const uint32_t x = bits_from_f32(value);
    const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    const uint32_t a = x & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return (uint16_t)(sign | (a > 0x7F800000u ? 0x7E00u : 0x7C00u));
    if (a >= 0x477FF000u) return (uint16_t)(sign | 0x7C00u); /* rounds to 65536 or more */
    if (a < 0x38800000u) {                                     /* below 2^-14: half subnormal or zero */
        const uint32_t e = a >> 23;
        uint32_t m, q, rem, half, shift;
        if (a <= 0x33000000u) return sign;                     /* <= 2^-25 rounds to zero (2^-25 is a tie) */
        m = (a & 0x7FFFFFu) | 0x800000u;
        shift = 126u - e;                                      /* 14 .. 24 */
        q = m >> shift;
        rem = m & ((1u << shift) - 1u);
        half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (q & 1u))) ++q;
        return (uint16_t)(sign | q);
    }
    {
        const uint32_t h = a - 0x38000000u;
        return (uint16_t)(sign | ((h + 0xFFFu + ((h >> 13) & 1u)) >> 13));
    }
}

/* ------------------------------------------------------------------------------------------------ B3 */

uint64_t ternary_b3_row_bytes(uint64_t columns) {
    if (columns == 0 || columns > TERNARY_HARD_MAX_COLUMNS) return 0;
    return (columns + 4) / 5;
}

ternary_convert_status ternary_pack_row_b3(const int8_t *codes, uint64_t columns, uint8_t *packed) {
    const uint64_t n = ternary_b3_row_bytes(columns);
    uint64_t i;
    if (!codes || !packed || !n) return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < columns; ++i)
        if (codes[i] < -1 || codes[i] > 1) return TERNARY_CONVERT_BAD_CODE;
    for (i = 0; i < n; ++i) {
        unsigned value = 0, power = 1;
        int k;
        for (k = 0; k < 5; ++k) {
            const uint64_t w = i * 5 + (uint64_t)k;
            const int digit = w < columns ? codes[w] + 1 : 1;
            value += (unsigned)digit * power;
            power *= 3;
        }
        packed[i] = (uint8_t)value;
    }
    return TERNARY_CONVERT_OK;
}

ternary_convert_status ternary_unpack_row_b3(const uint8_t *packed, uint64_t columns, int8_t *codes) {
    const uint64_t n = ternary_b3_row_bytes(columns);
    const uint64_t full = columns / 5;
    uint64_t i;
    if (!packed || !codes || !n) return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < full; ++i) {
        const uint8_t b = packed[i];
        if (b > 242) {
            memset(codes, 0, (size_t)columns);
            return TERNARY_CONVERT_BAD_CODE;
        }
        memcpy(codes + i * 5, kB3Digits[b], 5);
    }
    if (full < n) {
        const uint8_t b = packed[full];
        const unsigned tail = (unsigned)(columns - full * 5);
        unsigned k;
        if (b > 242) {
            memset(codes, 0, (size_t)columns);
            return TERNARY_CONVERT_BAD_CODE;
        }
        for (k = tail; k < 5; ++k)
            if (kB3Digits[b][k] != 0) {
                memset(codes, 0, (size_t)columns);
                return TERNARY_CONVERT_BAD_CODE;
            }
        memcpy(codes + full * 5, kB3Digits[b], tail);
    }
    return TERNARY_CONVERT_OK;
}

/* Trit w of an I2 row (block of 128: byte j holds weights j, j+32, j+64, j+96 in bit pairs 0..3). */
static int i2_code_at(const uint8_t *i2, uint64_t w) {
    const uint64_t block = w / TERNARY_GROUP_WEIGHT_COUNT, in = w % TERNARY_GROUP_WEIGHT_COUNT;
    const uint8_t byte = i2[block * TERNARY_I2_BLOCK_BYTES + (in % 32)];
    return (byte >> (2 * (in / 32))) & 3;
}

ternary_convert_status ternary_b3_from_i2_row(const uint8_t *i2, uint64_t columns, uint8_t *b3) {
    const uint64_t n = ternary_b3_row_bytes(columns);
    uint64_t i;
    if (!i2 || !b3 || !n || ternary_i2_row_bytes(columns) == 0) return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < n; ++i) {
        unsigned value = 0, power = 1;
        int k;
        for (k = 0; k < 5; ++k) {
            const uint64_t w = i * 5 + (uint64_t)k;
            int digit = 1;
            if (w < columns) {
                digit = i2_code_at(i2, w);
                if (digit == 3) return TERNARY_CONVERT_BAD_CODE;
            }
            value += (unsigned)digit * power;
            power *= 3;
        }
        b3[i] = (uint8_t)value;
    }
    return TERNARY_CONVERT_OK;
}

ternary_convert_status ternary_i2_from_b3_row(const uint8_t *b3, uint64_t columns, uint8_t *i2) {
    const uint64_t n = ternary_b3_row_bytes(columns);
    const uint64_t i2_bytes = ternary_i2_row_bytes(columns);
    uint64_t w;
    if (!b3 || !i2 || !n || !i2_bytes) return TERNARY_CONVERT_BAD_ARGUMENT;
    memset(i2, 0, (size_t)i2_bytes);
    for (w = 0; w < n; ++w)
        if (b3[w] > 242) {
            memset(i2, 0, (size_t)i2_bytes);
            return TERNARY_CONVERT_BAD_CODE;
        }
    {
        const uint64_t full = columns / 5;
        if (full < n) {
            unsigned k;
            for (k = (unsigned)(columns - full * 5); k < 5; ++k)
                if (kB3Digits[b3[full]][k] != 0) return TERNARY_CONVERT_BAD_CODE;
        }
    }
    /* 640 weights are 128 B3 bytes and 5 I2 blocks: decode them through the table, then pack the blocks. */
    {
        int8_t t[640];
        uint64_t base;
        for (base = 0; base < columns; base += 640) {
            const uint64_t n = columns - base < 640 ? columns - base : 640; /* a multiple of 128 */
            const uint64_t bytes = (n + 4) / 5;
            uint64_t k, blk;
            for (k = 0; k < bytes; ++k) memcpy(t + 5 * k, kB3Digits[b3[base / 5 + k]], 5);
            for (blk = 0; blk < n / TERNARY_GROUP_WEIGHT_COUNT; ++blk) {
                const int8_t *q = t + blk * TERNARY_GROUP_WEIGHT_COUNT;
                uint8_t *dst = i2 + (base / TERNARY_GROUP_WEIGHT_COUNT + blk) * TERNARY_I2_BLOCK_BYTES;
                int j;
                for (j = 0; j < 32; ++j)
                    dst[j] = (uint8_t)((q[j] + 1) | ((q[j + 32] + 1) << 2) | ((q[j + 64] + 1) << 4) | ((q[j + 96] + 1) << 6));
            }
        }
    }
    (void)w;
    return TERNARY_CONVERT_OK;
}

/* ------------------------------------------------------------------------------------------------ scales */

uint64_t ternary_groups_per_row(uint64_t columns) {
    if (columns == 0 || columns > TERNARY_HARD_MAX_COLUMNS) return 0;
    return (columns + TERNARY_GROUP_WEIGHT_COUNT - 1) / TERNARY_GROUP_WEIGHT_COUNT;
}

uint64_t ternary_scale_section_bytes(ternary_scale_encoding encoding, uint64_t rows, uint64_t groups_per_row) {
    uint64_t n, bytes, extra;
    if (rows == 0 || groups_per_row == 0 || ternary_multiply_overflows(rows, groups_per_row, &n)) return 0;
    switch (encoding) {
        case TERNARY_SCALES_F32:
            return ternary_multiply_overflows(n, 4, &bytes) ? 0 : bytes;
        case TERNARY_SCALES_F16:
            if (ternary_multiply_overflows(n, 2, &bytes) || add_overflows(bytes, 4, &bytes)) return 0;
            return bytes;
        case TERNARY_SCALES_U8ROW:
            if (ternary_multiply_overflows(rows, 4, &extra) || add_overflows(n, extra, &bytes)) return 0;
            return bytes;
        default:
            return 0;
    }
}

static float u8row_decode(float ref, unsigned u) { return (float)(((double)ref * (double)u) / 255.0); }

static int f16_shift_for(float max_scale) {
    int e = 0;
    if (!(max_scale > 0.0f)) return 0;
    (void)frexpf(max_scale, &e); /* max_scale = m * 2^e, m in [0.5, 1) */
    e = 15 - e;                  /* max_scale * 2^shift < 2^15 */
    if (e < -128) e = -128;
    if (e > 127) e = 127;
    return e;
}

static float f16_decode(uint16_t h, int shift) { return ldexpf(ternary_float16_to_float32(h), -shift); }

ternary_convert_status ternary_encode_scales(ternary_scale_encoding encoding, const float *scales, uint64_t rows,
                                             uint64_t groups_per_row, uint8_t *out, float *decoded) {
    const uint64_t bytes = ternary_scale_section_bytes(encoding, rows, groups_per_row);
    const uint64_t n = rows * groups_per_row;
    uint64_t i, r, g;
    if (!scales || !out || !bytes) return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < n; ++i)
        if (!scale_ok(scales[i])) return TERNARY_CONVERT_BAD_SCALE;
    memset(out, 0, (size_t)bytes);
    switch (encoding) {
        case TERNARY_SCALES_F32:
            for (i = 0; i < n; ++i) put32(out + 4 * i, bits_from_f32(scales[i]));
            if (decoded) memcpy(decoded, scales, (size_t)n * sizeof(float));
            return TERNARY_CONVERT_OK;
        case TERNARY_SCALES_F16: {
            float max_scale = 0.0f;
            int shift;
            for (i = 0; i < n; ++i)
                if (scales[i] > max_scale) max_scale = scales[i];
            shift = f16_shift_for(max_scale);
            out[0] = (uint8_t)(int8_t)shift;
            for (i = 0; i < n; ++i) {
                uint16_t h = ternary_float32_to_float16(ldexpf(scales[i], shift));
                float d = f16_decode(h, shift);
                if (!scale_ok(d)) {
                    h = 0;
                    d = 0.0f;
                }
                put16(out + 4 + 2 * i, h);
                if (decoded) decoded[i] = d;
            }
            return TERNARY_CONVERT_OK;
        }
        case TERNARY_SCALES_U8ROW:
            for (r = 0; r < rows; ++r) {
                const float *row = scales + r * groups_per_row;
                uint8_t *codes = out + rows * 4 + r * groups_per_row;
                float ref = 0.0f;
                for (g = 0; g < groups_per_row; ++g)
                    if (row[g] > ref) ref = row[g];
                put32(out + 4 * r, bits_from_f32(ref));
                for (g = 0; g < groups_per_row; ++g) {
                    const float s = row[g];
                    unsigned best = 0;
                    float best_d = 0.0f;
                    if (s > 0.0f && ref > 0.0f) {
                        const double t = ((double)s * 255.0) / (double)ref;
                        unsigned u0 = (unsigned)floor(t), k;
                        double best_e;
                        if (u0 > 255) u0 = 255;
                        best_e = fabs((double)s);
                        for (k = u0; k <= u0 + 1 && k <= 255; ++k) {
                            const float d = u8row_decode(ref, k);
                            const double e = fabs((double)d - (double)s);
                            if (!scale_ok(d)) continue;
                            if (e < best_e) {
                                best_e = e;
                                best = k;
                                best_d = d;
                            }
                        }
                    }
                    codes[g] = (uint8_t)best;
                    if (decoded) decoded[r * groups_per_row + g] = best_d;
                }
            }
            return TERNARY_CONVERT_OK;
        default:
            return TERNARY_CONVERT_BAD_ARGUMENT;
    }
}

ternary_convert_status ternary_decode_scales(ternary_scale_encoding encoding, const uint8_t *section,
                                             uint64_t section_bytes, uint64_t rows, uint64_t groups_per_row,
                                             float *out) {
    const uint64_t bytes = ternary_scale_section_bytes(encoding, rows, groups_per_row);
    const uint64_t n = rows * groups_per_row;
    uint64_t i, r, g;
    if (!section || !out || !bytes) return TERNARY_CONVERT_BAD_ARGUMENT;
    if (section_bytes != bytes) return TERNARY_CONVERT_TRUNCATED;
    switch (encoding) {
        case TERNARY_SCALES_F32:
            for (i = 0; i < n; ++i) {
                out[i] = f32_from_bits(get32(section + 4 * i));
                if (!scale_ok(out[i])) goto bad;
            }
            return TERNARY_CONVERT_OK;
        case TERNARY_SCALES_F16: {
            const int shift = (int)(int8_t)section[0];
            if (section[1] || section[2] || section[3]) {
                memset(out, 0, (size_t)n * sizeof(float));
                return TERNARY_CONVERT_BAD_HEADER;
            }
            for (i = 0; i < n; ++i) {
                const uint16_t h = get16(section + 4 + 2 * i);
                if (h & 0x8000u) goto bad;
                out[i] = f16_decode(h, shift);
                if (!scale_ok(out[i])) goto bad;
            }
            return TERNARY_CONVERT_OK;
        }
        case TERNARY_SCALES_U8ROW:
            for (r = 0; r < rows; ++r) {
                const float ref = f32_from_bits(get32(section + 4 * r));
                if (!scale_ok(ref)) goto bad;
                for (g = 0; g < groups_per_row; ++g) {
                    const float d = u8row_decode(ref, section[rows * 4 + r * groups_per_row + g]);
                    if (!scale_ok(d)) goto bad;
                    out[r * groups_per_row + g] = d;
                }
            }
            return TERNARY_CONVERT_OK;
        default:
            return TERNARY_CONVERT_BAD_ARGUMENT;
    }
bad:
    memset(out, 0, (size_t)n * sizeof(float));
    return TERNARY_CONVERT_BAD_SCALE;
}

/* ------------------------------------------------------------------------------------------------ rANS */

enum { RANS_HEADER = 20 };

static void rans_frequencies(const int8_t *codes, uint64_t count, uint16_t freq[3]) {
    uint64_t c[3] = {0, 0, 0}, i;
    int s, big = 0;
    int32_t f[3], sum = 0;
    for (i = 0; i < count; ++i) c[codes[i] + 1]++;
    for (s = 0; s < 3; ++s) {
        f[s] = (int32_t)(((double)c[s] * TERNARY_RANS_PROB_TOTAL) / (double)count + 0.5);
        if (f[s] < 1) f[s] = 1;
        sum += f[s];
        if (c[s] > c[big]) big = s;
    }
    f[big] += TERNARY_RANS_PROB_TOTAL - sum; /* the largest share absorbs the rounding; it stays >= 1 */
    for (s = 0; s < 3; ++s) freq[s] = (uint16_t)f[s];
}

ternary_convert_status ternary_rans_encode(const int8_t *codes, uint64_t rows, uint64_t columns,
                                           uint32_t rows_per_chunk, uint8_t **section, size_t *section_bytes) {
    uint64_t total, chunks, worst, c, data_used = 0, i, base;
    uint16_t freq[3];
    uint32_t cum[3];
    uint8_t *buf = NULL, *scratch = NULL;
    if (section) *section = NULL;
    if (section_bytes) *section_bytes = 0;
    if (!codes || !section || !section_bytes || rows == 0 || columns == 0 || rows_per_chunk == 0 ||
        columns > TERNARY_HARD_MAX_COLUMNS || ternary_multiply_overflows(rows, columns, &total))
        return TERNARY_CONVERT_BAD_ARGUMENT;
    for (i = 0; i < total; ++i)
        if (codes[i] < -1 || codes[i] > 1) return TERNARY_CONVERT_BAD_CODE;
    chunks = (rows + rows_per_chunk - 1) / rows_per_chunk;
    if (chunks > UINT32_MAX) return TERNARY_CONVERT_TOO_LARGE;
    rans_frequencies(codes, total, freq);
    cum[0] = 0;
    cum[1] = freq[0];
    cum[2] = (uint32_t)freq[0] + freq[1];
    /* Worst case: 12 bits per trit plus the 4-byte state per chunk. */
    if (ternary_multiply_overflows(total, 12, &worst)) return TERNARY_CONVERT_TOO_LARGE;
    worst = worst / 8 + 1 + 8 * chunks + RANS_HEADER + 8 * chunks;
    if (worst > (uint64_t)SIZE_MAX) return TERNARY_CONVERT_TOO_LARGE;
    buf = (uint8_t *)malloc((size_t)worst);
    {
        const uint64_t chunk_trits = (rows < rows_per_chunk ? rows : (uint64_t)rows_per_chunk) * columns;
        scratch = (uint8_t *)malloc((size_t)(chunk_trits * 12 / 8 + 16));
    }
    if (!buf || !scratch) {
        free(buf);
        free(scratch);
        return TERNARY_CONVERT_TOO_LARGE;
    }
    memset(buf, 0, RANS_HEADER);
    put16(buf, freq[0]);
    put16(buf + 2, freq[1]);
    put16(buf + 4, freq[2]);
    put32(buf + 8, rows_per_chunk);
    put32(buf + 12, (uint32_t)chunks);
    base = RANS_HEADER + 8 * chunks;
    for (c = 0; c < chunks; ++c) {
        const uint64_t first = c * rows_per_chunk;
        const uint64_t nrows = (rows - first < rows_per_chunk) ? rows - first : rows_per_chunk;
        const int8_t *p = codes + first * columns;
        uint64_t n = nrows * columns, k, emitted = 0;
        uint32_t x = RANS_LOW;
        for (k = n; k-- > 0;) {
            const int s = p[k] + 1;
            const uint32_t f = freq[s];
            const uint32_t x_max = ((RANS_LOW >> TERNARY_RANS_PROB_BITS) << 8) * f;
            while (x >= x_max) {
                scratch[emitted++] = (uint8_t)x;
                x >>= 8;
            }
            x = ((x / f) << TERNARY_RANS_PROB_BITS) + (x % f) + cum[s];
        }
        scratch[emitted++] = (uint8_t)(x >> 24);
        scratch[emitted++] = (uint8_t)(x >> 16);
        scratch[emitted++] = (uint8_t)(x >> 8);
        scratch[emitted++] = (uint8_t)x;
        for (k = 0; k < emitted; ++k) buf[base + data_used + k] = scratch[emitted - 1 - k];
        data_used += emitted;
        put64(buf + RANS_HEADER + 8 * c, data_used);
    }
    free(scratch);
    *section = buf;
    *section_bytes = (size_t)(base + data_used);
    return TERNARY_CONVERT_OK;
}

typedef struct rans_layout {
    uint16_t freq[3];
    uint32_t rows_per_chunk;
    uint32_t chunk_count;
    uint64_t base;
} rans_layout;

static ternary_convert_status rans_layout_of(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                             uint64_t columns, rans_layout *out) {
    uint64_t base, prev = 0, c, expected_chunks, total;
    if (!section || rows == 0 || columns == 0 || columns > TERNARY_HARD_MAX_COLUMNS ||
        ternary_multiply_overflows(rows, columns, &total))
        return TERNARY_CONVERT_BAD_ARGUMENT;
    if (section_bytes < RANS_HEADER) return TERNARY_CONVERT_TRUNCATED;
    out->freq[0] = get16(section);
    out->freq[1] = get16(section + 2);
    out->freq[2] = get16(section + 4);
    out->rows_per_chunk = get32(section + 8);
    out->chunk_count = get32(section + 12);
    if (get16(section + 6) || get32(section + 16)) return TERNARY_CONVERT_BAD_HEADER;
    if (!out->freq[0] || !out->freq[1] || !out->freq[2] ||
        (uint32_t)out->freq[0] + out->freq[1] + out->freq[2] != TERNARY_RANS_PROB_TOTAL)
        return TERNARY_CONVERT_BAD_HEADER;
    if (out->rows_per_chunk == 0) return TERNARY_CONVERT_BAD_HEADER;
    expected_chunks = (rows + out->rows_per_chunk - 1) / out->rows_per_chunk;
    if (out->chunk_count != expected_chunks) return TERNARY_CONVERT_BAD_HEADER;
    base = RANS_HEADER + 8 * (uint64_t)out->chunk_count;
    if (base > section_bytes) return TERNARY_CONVERT_TRUNCATED;
    for (c = 0; c < out->chunk_count; ++c) {
        const uint64_t end = get64(section + RANS_HEADER + 8 * c);
        if (end < prev + 4 || end > section_bytes - base) return TERNARY_CONVERT_BAD_HEADER;
        prev = end;
    }
    if (prev != section_bytes - base) return TERNARY_CONVERT_BAD_HEADER;
    out->base = base;
    return TERNARY_CONVERT_OK;
}

ternary_convert_status ternary_rans_check(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                          uint64_t columns) {
    rans_layout layout;
    return rans_layout_of(section, section_bytes, rows, columns, &layout);
}

uint32_t ternary_rans_chunk_count(const uint8_t *section, uint64_t section_bytes) {
    if (!section || section_bytes < RANS_HEADER) return 0;
    return get32(section + 12);
}

ternary_convert_status ternary_rans_decode_chunk(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                                 uint64_t columns, uint32_t chunk, int8_t *codes,
                                                 uint64_t *first_row, uint64_t *row_count) {
    rans_layout L;
    ternary_convert_status st;
    uint64_t begin, end, first, nrows, n, k;
    const uint8_t *p, *stop;
    uint32_t x;
    uint8_t lut[TERNARY_RANS_PROB_TOTAL];
    uint32_t cum[3];
    if (first_row) *first_row = 0;
    if (row_count) *row_count = 0;
    if (!codes) return TERNARY_CONVERT_BAD_ARGUMENT;
    st = rans_layout_of(section, section_bytes, rows, columns, &L);
    if (st != TERNARY_CONVERT_OK) return st;
    if (chunk >= L.chunk_count) return TERNARY_CONVERT_BAD_ARGUMENT;
    begin = chunk ? get64(section + RANS_HEADER + 8 * ((uint64_t)chunk - 1)) : 0;
    end = get64(section + RANS_HEADER + 8 * (uint64_t)chunk);
    first = (uint64_t)chunk * L.rows_per_chunk;
    nrows = (rows - first < L.rows_per_chunk) ? rows - first : L.rows_per_chunk;
    n = nrows * columns;
    cum[0] = 0;
    cum[1] = L.freq[0];
    cum[2] = (uint32_t)L.freq[0] + L.freq[1];
    memset(lut, 0, L.freq[0]);
    memset(lut + cum[1], 1, L.freq[1]);
    memset(lut + cum[2], 2, L.freq[2]);
    p = section + L.base + begin;
    stop = section + L.base + end;
    x = get32(p);
    p += 4;
    for (k = 0; k < n; ++k) {
        const uint32_t slot = x & (TERNARY_RANS_PROB_TOTAL - 1);
        const int s = lut[slot];
        x = (uint32_t)L.freq[s] * (x >> TERNARY_RANS_PROB_BITS) + slot - cum[s];
        codes[k] = (int8_t)(s - 1);
        while (x < RANS_LOW) {
            if (p >= stop) goto bad;
            x = (x << 8) | *p++;
        }
    }
    if (x != RANS_LOW || p != stop) goto bad;
    if (first_row) *first_row = first;
    if (row_count) *row_count = nrows;
    return TERNARY_CONVERT_OK;
bad:
    memset(codes, 0, (size_t)n);
    return TERNARY_CONVERT_BAD_CODE;
}

ternary_convert_status ternary_rans_decode(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                           uint64_t columns, int8_t *codes) {
    rans_layout L;
    ternary_convert_status st;
    uint32_t c;
    if (!codes) return TERNARY_CONVERT_BAD_ARGUMENT;
    st = rans_layout_of(section, section_bytes, rows, columns, &L);
    if (st != TERNARY_CONVERT_OK) return st;
    for (c = 0; c < L.chunk_count; ++c) {
        st = ternary_rans_decode_chunk(section, section_bytes, rows, columns, c,
                                       codes + (uint64_t)c * L.rows_per_chunk * columns, NULL, NULL);
        if (st != TERNARY_CONVERT_OK) {
            memset(codes, 0, (size_t)(rows * columns));
            return st;
        }
    }
    return TERNARY_CONVERT_OK;
}

/* ------------------------------------------------------------------------------------------------ PTT1 */

static uint64_t raw_dtype_bytes(int dtype) {
    switch (dtype) {
        case TERNARY_RAW_F32: return 4;
        case TERNARY_RAW_F16: return 2;
        case TERNARY_RAW_BF16: return 2;
        default: return 0;
    }
}

/* Checks a description and returns the exact section sizes it needs (data_bytes is the exact size, except for
 * RANS, where it is 0 and the section is checked separately). */
static ternary_convert_status check_desc(const ternary_ptt1_desc *d, uint64_t max_bytes, uint64_t *scales_bytes,
                                         uint64_t *data_bytes) {
    uint64_t product = 1, count, gpr, bytes;
    unsigned i;
    if (d->dimension_count < 1 || d->dimension_count > TERNARY_SOURCE_MAX_DIMENSIONS) return TERNARY_CONVERT_BAD_HEADER;
    for (i = 0; i < TERNARY_SOURCE_MAX_DIMENSIONS; ++i) {
        if (i >= d->dimension_count) {
            if (d->dims[i]) return TERNARY_CONVERT_BAD_HEADER;
            continue;
        }
        if (d->dims[i] == 0) return TERNARY_CONVERT_BAD_HEADER;
        if (ternary_multiply_overflows(product, d->dims[i], &product)) return TERNARY_CONVERT_TOO_LARGE;
    }
    if (d->columns != d->dims[d->dimension_count - 1] || d->rows == 0 ||
        ternary_multiply_overflows(d->rows, d->columns, &count) || count != product)
        return TERNARY_CONVERT_BAD_HEADER;
    switch (d->kind) {
        case TERNARY_KIND_TERNARY:
            if (d->raw_dtype) return TERNARY_CONVERT_BAD_HEADER;
            if (d->rows > TERNARY_HARD_MAX_ROWS || d->columns > TERNARY_HARD_MAX_COLUMNS) return TERNARY_CONVERT_TOO_LARGE;
            gpr = ternary_groups_per_row(d->columns);
            if (d->scale_encoding < TERNARY_SCALES_F32 || d->scale_encoding > TERNARY_SCALES_U8ROW)
                return TERNARY_CONVERT_BAD_HEADER;
            *scales_bytes = ternary_scale_section_bytes((ternary_scale_encoding)d->scale_encoding, d->rows, gpr);
            if (!*scales_bytes) return TERNARY_CONVERT_TOO_LARGE;
            if (d->trit_encoding == TERNARY_TRITS_B3) {
                if (ternary_multiply_overflows(d->rows, ternary_b3_row_bytes(d->columns), data_bytes))
                    return TERNARY_CONVERT_TOO_LARGE;
            } else if (d->trit_encoding == TERNARY_TRITS_RANS) {
                *data_bytes = 0;
            } else {
                return TERNARY_CONVERT_BAD_HEADER;
            }
            break;
        case TERNARY_KIND_Q8:
            if (d->trit_encoding || d->raw_dtype || d->scale_encoding != TERNARY_SCALES_F32) return TERNARY_CONVERT_BAD_HEADER;
            if (d->rows > (UINT64_C(1) << 32) || d->columns > TERNARY_HARD_MAX_COLUMNS) return TERNARY_CONVERT_TOO_LARGE;
            gpr = ternary_groups_per_row(d->columns);
            *scales_bytes = ternary_scale_section_bytes(TERNARY_SCALES_F32, d->rows, gpr);
            if (!*scales_bytes) return TERNARY_CONVERT_TOO_LARGE;
            *data_bytes = count;
            break;
        case TERNARY_KIND_RAW:
            if (d->trit_encoding || d->scale_encoding || !raw_dtype_bytes(d->raw_dtype)) return TERNARY_CONVERT_BAD_HEADER;
            *scales_bytes = 0;
            if (ternary_multiply_overflows(count, raw_dtype_bytes(d->raw_dtype), data_bytes)) return TERNARY_CONVERT_TOO_LARGE;
            break;
        default:
            return TERNARY_CONVERT_BAD_HEADER;
    }
    if (add_overflows(*scales_bytes, *data_bytes, &bytes) || bytes > max_bytes) return TERNARY_CONVERT_TOO_LARGE;
    return TERNARY_CONVERT_OK;
}

int ternary_build_ptt1(const ternary_ptt1_desc *desc, const uint8_t *scales, uint64_t scales_bytes,
                       const uint8_t *data, uint64_t data_bytes, uint8_t **bytes, size_t *length,
                       char *error, size_t error_capacity) {
    uint64_t want_scales = 0, want_data = 0, total;
    ternary_convert_status st;
    uint8_t *buf;
    unsigned i;
    if (bytes) *bytes = NULL;
    if (length) *length = 0;
    if (!desc || !bytes || !length || (scales_bytes && !scales) || (data_bytes && !data)) {
        ternary_set_error(error, error_capacity, "ptt1: bad argument");
        return 0;
    }
    st = check_desc(desc, UINT64_MAX, &want_scales, &want_data);
    if (st == TERNARY_CONVERT_OK && desc->kind == TERNARY_KIND_TERNARY && desc->trit_encoding == TERNARY_TRITS_RANS)
        st = ternary_rans_check(data, data_bytes, desc->rows, desc->columns);
    else if (st == TERNARY_CONVERT_OK && want_data != data_bytes)
        st = TERNARY_CONVERT_BAD_HEADER;
    if (st == TERNARY_CONVERT_OK && want_scales != scales_bytes) st = TERNARY_CONVERT_BAD_HEADER;
    if (st != TERNARY_CONVERT_OK) {
        ternary_set_errorf(error, error_capacity, "ptt1: %s", ternary_convert_status_name(st));
        return 0;
    }
    total = TERNARY_PTT1_HEADER_BYTES + scales_bytes + data_bytes;
    if (total > (uint64_t)SIZE_MAX || !(buf = (uint8_t *)calloc((size_t)total, 1))) {
        ternary_set_error(error, error_capacity, "ptt1: out of memory");
        return 0;
    }
    memcpy(buf, "PTT1", 4);
    put16(buf + 4, PTT1_VERSION);
    buf[6] = (uint8_t)desc->kind;
    buf[7] = (uint8_t)desc->trit_encoding;
    buf[8] = (uint8_t)desc->scale_encoding;
    buf[9] = (uint8_t)desc->raw_dtype;
    buf[10] = (uint8_t)desc->dimension_count;
    put32(buf + 12, TERNARY_PTT1_HEADER_BYTES);
    for (i = 0; i < TERNARY_SOURCE_MAX_DIMENSIONS; ++i) put64(buf + 16 + 8 * i, desc->dims[i]);
    put64(buf + 80, desc->rows);
    put64(buf + 88, desc->columns);
    put64(buf + 96, TERNARY_PTT1_HEADER_BYTES);
    put64(buf + 104, scales_bytes);
    put64(buf + 112, TERNARY_PTT1_HEADER_BYTES + scales_bytes);
    put64(buf + 120, data_bytes);
    if (scales_bytes) memcpy(buf + TERNARY_PTT1_HEADER_BYTES, scales, (size_t)scales_bytes);
    if (data_bytes) memcpy(buf + TERNARY_PTT1_HEADER_BYTES + scales_bytes, data, (size_t)data_bytes);
    put64(buf + 128, ternary_fnv1a64(buf + TERNARY_PTT1_HEADER_BYTES, (size_t)(total - TERNARY_PTT1_HEADER_BYTES),
                                     1469598103934665603ull));
    *bytes = buf;
    *length = (size_t)total;
    return 1;
}

ternary_convert_status ternary_parse_ptt1(const uint8_t *buffer, size_t length, uint64_t max_bytes,
                                          ternary_ptt1_view *view) {
    ternary_ptt1_desc d;
    uint64_t want_scales = 0, want_data = 0, so, sb, dof, db, end;
    ternary_convert_status st;
    unsigned i;
    if (view) memset(view, 0, sizeof(*view));
    if (!buffer || !view) return TERNARY_CONVERT_BAD_ARGUMENT;
    if (length < TERNARY_PTT1_HEADER_BYTES) return TERNARY_CONVERT_TRUNCATED;
    if (memcmp(buffer, "PTT1", 4) != 0) return TERNARY_CONVERT_BAD_MAGIC;
    if (get16(buffer + 4) != PTT1_VERSION) return TERNARY_CONVERT_BAD_VERSION;
    if (get32(buffer + 12) != TERNARY_PTT1_HEADER_BYTES || buffer[11]) return TERNARY_CONVERT_BAD_HEADER;
    for (i = 136; i < TERNARY_PTT1_HEADER_BYTES; ++i)
        if (buffer[i]) return TERNARY_CONVERT_BAD_HEADER;
    memset(&d, 0, sizeof(d));
    d.kind = (ternary_tensor_kind)buffer[6];
    d.trit_encoding = buffer[7];
    d.scale_encoding = buffer[8];
    d.raw_dtype = buffer[9];
    d.dimension_count = buffer[10];
    for (i = 0; i < TERNARY_SOURCE_MAX_DIMENSIONS; ++i) d.dims[i] = get64(buffer + 16 + 8 * i);
    d.rows = get64(buffer + 80);
    d.columns = get64(buffer + 88);
    st = check_desc(&d, max_bytes, &want_scales, &want_data);
    if (st != TERNARY_CONVERT_OK) return st;
    so = get64(buffer + 96);
    sb = get64(buffer + 104);
    dof = get64(buffer + 112);
    db = get64(buffer + 120);
    if (so != TERNARY_PTT1_HEADER_BYTES || sb != want_scales || add_overflows(so, sb, &end) || dof != end ||
        add_overflows(dof, db, &end) || end != (uint64_t)length)
        return TERNARY_CONVERT_BAD_HEADER;
    if (!(d.kind == TERNARY_KIND_TERNARY && d.trit_encoding == TERNARY_TRITS_RANS) && db != want_data)
        return TERNARY_CONVERT_BAD_HEADER;
    if (db > max_bytes || sb > max_bytes - db) return TERNARY_CONVERT_TOO_LARGE;
    if (get64(buffer + 128) != ternary_fnv1a64(buffer + TERNARY_PTT1_HEADER_BYTES,
                                               length - TERNARY_PTT1_HEADER_BYTES, 1469598103934665603ull))
        return TERNARY_CONVERT_BAD_CHECKSUM;
    if (d.kind == TERNARY_KIND_TERNARY && d.trit_encoding == TERNARY_TRITS_RANS) {
        st = ternary_rans_check(buffer + dof, db, d.rows, d.columns);
        if (st != TERNARY_CONVERT_OK) return st;
    }
    view->desc = d;
    view->scales = buffer + so;
    view->scales_bytes = sb;
    view->data = buffer + dof;
    view->data_bytes = db;
    return TERNARY_CONVERT_OK;
}

ternary_convert_status ternary_ptt1_decode_ternary(const ternary_ptt1_view *view, int8_t *codes, float *scales) {
    const ternary_ptt1_desc *d;
    ternary_convert_status st = TERNARY_CONVERT_OK;
    uint64_t r;
    if (!view || view->desc.kind != TERNARY_KIND_TERNARY) return TERNARY_CONVERT_BAD_ARGUMENT;
    d = &view->desc;
    if (scales) {
        st = ternary_decode_scales((ternary_scale_encoding)d->scale_encoding, view->scales, view->scales_bytes,
                                   d->rows, ternary_groups_per_row(d->columns), scales);
        if (st != TERNARY_CONVERT_OK) return st;
    }
    if (codes) {
        if (d->trit_encoding == TERNARY_TRITS_RANS) {
            st = ternary_rans_decode(view->data, view->data_bytes, d->rows, d->columns, codes);
        } else {
            const uint64_t rb = ternary_b3_row_bytes(d->columns);
            for (r = 0; r < d->rows && st == TERNARY_CONVERT_OK; ++r)
                st = ternary_unpack_row_b3(view->data + r * rb, d->columns, codes + r * d->columns);
            if (st != TERNARY_CONVERT_OK) memset(codes, 0, (size_t)(d->rows * d->columns));
        }
        if (st != TERNARY_CONVERT_OK && scales)
            memset(scales, 0, (size_t)(d->rows * ternary_groups_per_row(d->columns)) * sizeof(float));
    }
    return st;
}

/* ------------------------------------------------------------------------------------------------ PTP1 */

void ternary_ptp1_view_clear(ternary_ptp1_view *view) {
    if (!view) return;
    free(view->entries);
    memset(view, 0, sizeof(*view));
}

void ternary_build_ptp1_header(uint32_t tensor_count, uint64_t index_offset, const uint8_t *index,
                               size_t index_length, uint8_t header[TERNARY_PTP1_HEADER_BYTES]) {
    memset(header, 0, TERNARY_PTP1_HEADER_BYTES);
    memcpy(header, "PTP1", 4);
    put16(header + 4, PTP1_VERSION);
    put32(header + 8, TERNARY_PTP1_HEADER_BYTES);
    put32(header + 12, tensor_count);
    put64(header + 16, index_offset);
    put64(header + 24, (uint64_t)index_length);
    ternary_sha256(index, index_length, header + 32);
}

ternary_convert_status ternary_parse_ptp1(const uint8_t *header, size_t header_length, const uint8_t *index,
                                          size_t index_length, uint64_t file_size, ternary_ptp1_view *view) {
    uint64_t index_offset, index_bytes, pos = 0, expect = TERNARY_PTP1_HEADER_BYTES, end;
    uint32_t count, i;
    uint8_t digest[32];
    if (view) memset(view, 0, sizeof(*view));
    if (!header || !view || (index_length && !index)) return TERNARY_CONVERT_BAD_ARGUMENT;
    if (header_length < TERNARY_PTP1_HEADER_BYTES) return TERNARY_CONVERT_TRUNCATED;
    if (memcmp(header, "PTP1", 4) != 0) return TERNARY_CONVERT_BAD_MAGIC;
    if (get16(header + 4) != PTP1_VERSION) return TERNARY_CONVERT_BAD_VERSION;
    if (get16(header + 6) || get32(header + 8) != TERNARY_PTP1_HEADER_BYTES) return TERNARY_CONVERT_BAD_HEADER;
    count = get32(header + 12);
    index_offset = get64(header + 16);
    index_bytes = get64(header + 24);
    if (count > TERNARY_SOURCE_MAX_TENSORS || index_offset < TERNARY_PTP1_HEADER_BYTES ||
        add_overflows(index_offset, index_bytes, &end) || end != file_size || index_bytes != (uint64_t)index_length)
        return TERNARY_CONVERT_BAD_HEADER;
    ternary_sha256(index, index_length, digest);
    if (memcmp(digest, header + 32, 32) != 0) return TERNARY_CONVERT_BAD_CHECKSUM;
    view->entries = (ternary_ptp1_entry *)calloc(count ? count : 1, sizeof(ternary_ptp1_entry));
    if (!view->entries) return TERNARY_CONVERT_TOO_LARGE;
    for (i = 0; i < count; ++i) {
        ternary_ptp1_entry *e = &view->entries[i];
        uint32_t name_length;
        if (index_length - pos < 4) goto bad;
        name_length = get32(index + pos);
        pos += 4;
        if (name_length == 0 || name_length > TERNARY_PACK_MAX_NAME_BYTES || index_length - pos < (uint64_t)name_length + 48)
            goto bad;
        if (memchr(index + pos, 0, name_length)) goto bad;
        e->name = (const char *)(index + pos);
        e->name_length = name_length;
        pos += name_length;
        e->offset = get64(index + pos);
        e->length = get64(index + pos + 8);
        e->sha256 = index + pos + 16;
        pos += 48;
        if (e->offset != expect || e->length < TERNARY_PTT1_HEADER_BYTES || add_overflows(e->offset, e->length, &end) ||
            end > index_offset)
            goto bad;
        expect = end;
    }
    if (pos != index_length || expect != index_offset) goto bad;
    view->tensor_count = count;
    view->index_offset = index_offset;
    view->index_bytes = index_bytes;
    return TERNARY_CONVERT_OK;
bad:
    ternary_ptp1_view_clear(view);
    return TERNARY_CONVERT_BAD_HEADER;
}

/* ------------------------------------------------------------------------------------------------ padded fit */

ternary_convert_status ternary_fit_rows_padded(const float *weights, uint64_t rows, uint64_t cols, int thread_count,
                                               int8_t *codes, float *scales) {
    const uint64_t gpr = ternary_groups_per_row(cols);
    uint64_t groups, padded_count, r, c;
    float *padded;
    int8_t *pc;
    ternary_convert_status st;
    if (!weights || !codes || !scales || rows == 0 || !gpr || rows > TERNARY_HARD_MAX_ROWS ||
        ternary_multiply_overflows(rows, gpr, &groups) ||
        ternary_multiply_overflows(groups, TERNARY_GROUP_WEIGHT_COUNT, &padded_count) ||
        padded_count > (uint64_t)SIZE_MAX / sizeof(float))
        return TERNARY_CONVERT_BAD_ARGUMENT;
    padded = (float *)calloc((size_t)padded_count, sizeof(float));
    pc = (int8_t *)malloc((size_t)padded_count);
    if (!padded || !pc) {
        free(padded);
        free(pc);
        return TERNARY_CONVERT_TOO_LARGE;
    }
    for (r = 0; r < rows; ++r) memcpy(padded + r * gpr * TERNARY_GROUP_WEIGHT_COUNT, weights + r * cols, (size_t)cols * sizeof(float));
    st = ternary_fit_groups_cpu(padded, groups, TERNARY_GROUP_FIT_LEAST_SQUARES, thread_count, pc, scales);
    if (st == TERNARY_CONVERT_OK)
        for (r = 0; r < rows; ++r) memcpy(codes + r * cols, pc + r * gpr * TERNARY_GROUP_WEIGHT_COUNT, (size_t)cols);
    else {
        memset(codes, 0, (size_t)(rows * cols));
        memset(scales, 0, (size_t)groups * sizeof(float));
    }
    (void)c;
    free(padded);
    free(pc);
    return st;
}
