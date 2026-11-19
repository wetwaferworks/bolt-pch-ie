/* Bolt PCHIE - packed storage for ternary models (clean-room, C17).
 *
 * PTN1 (pchie_ternary_convert.h) stores each ternary weight in 2 bits (I2) and one float32 scale per 128-weight
 * group: 2.25 bits per weight. This file adds a storage format, PTT1, that holds exactly the same ternary values in
 * less space, plus PTP1, one file that holds many PTT1 tensors (one pack per source shard). I2 stays the compute
 * format: a reader unpacks PTT1 to I2 (or int8) once, at load time.
 *
 * Trit encodings (lossless; unpacking gives back the I2 codes exactly, bit for bit)
 *   B3    five weights per byte, byte = sum_k (q_k + 1) * 3^k for weights 5i .. 5i+4 of a row, in order; bytes
 *         243..255 are invalid and padding weights past the row end must be 0. ceil(cols / 5) bytes per row,
 *         so 1.6 bits per weight. Same definition as Packing::B3 in pchie_ternary.h.
 *   RANS  the row-major trit stream entropy-coded with a static three-symbol rANS model (frequencies out of
 *         4096, stored with the tensor). The stream is cut into chunks of whole rows that decode independently.
 *         Typically 1.48 - 1.54 bits per weight on these models; slower to unpack than B3.
 *
 * Scale encodings (one scale per 128-weight group; the last group of a row may be shorter, see below)
 *   F32   float32 as in PTN1: lossless.
 *   F16   IEEE half of scale * 2^shift with one int8 shift for the whole tensor (so tiny scales keep their
 *         precision); decoded scale = half_to_float(h) * 2^-shift. 2 bytes per group.
 *   U8ROW per row one float32 ref (the row's largest scale) and per group one byte u;
 *         decoded scale = (float)(((double)ref * u) / 255.0). 1 byte per group plus 4 per row.
 *   F16 and U8ROW change the scale only, never the codes; the converter uses them only where the measured
 *   reconstruction error stays within a stated tolerance of F32, and records which one each tensor uses.
 *   Every decoded scale is finite, >= 0 and either 0 or a normal float, or the tensor is refused.
 *
 * Groups and rows
 *   A row of cols weights has ceil(cols / 128) groups; the last one holds cols % 128 weights when that is not 0.
 *   PTN1 needs cols to be a multiple of 128; PTT1 does not (this is how short rows, e.g. Mamba conv1d kernels,
 *   can be stored).
 *
 * Other tensor kinds in a PTT1, so a pack can hold a whole model
 *   RAW   the source bytes, unchanged (F32, F16 or BF16).
 *   Q8    int8 per weight with one float32 scale per 128-weight group (symmetric, scale = absmax / 127).
 *
 * PTT1 layout (little-endian; every field checked on parse; inputs are untrusted)
 *    0 "PTT1"   4 u16 version (1)   6 u8 kind (1 ternary, 2 raw, 3 q8)   7 u8 trit encoding (1 B3, 2 RANS; else 0)
 *    8 u8 scale encoding (1 F32, 2 F16, 3 U8ROW; q8: 1; raw: 0)   9 u8 raw dtype (1 F32, 2 F16, 3 BF16; else 0)
 *   10 u8 dimension count (1..8)   11 u8 0   12 u32 header bytes (160)   16 u64 dims[8] (outermost first, unused 0)
 *   80 u64 rows   88 u64 cols (rows * cols = product of dims; cols = innermost dim)
 *   96 u64 scales offset  104 u64 scales bytes  112 u64 data offset  120 u64 data bytes
 *  128 u64 FNV-1a 64 of every byte after the header   136..159 zero
 *   Sections follow the header back to back: scales, then data. Nothing else may be in the buffer.
 *   F16 scales section: int8 shift, 3 zero bytes, then u16 per group (sign bit clear).
 *   RANS data section: u16 freq[3] (each >= 1, sum 4096), u16 0, u32 rows per chunk, u32 chunk count,
 *   u32 0, u64 end offset of each chunk (relative to the first chunk byte, increasing), then the chunks.
 *   Each chunk starts with the 4-byte final encoder state; decoding must end with state 2^23 exactly at the
 *   chunk's end.
 *
 * PTP1 layout
 *    0 "PTP1"  4 u16 version (1)  6 u16 0  8 u32 header bytes (64)  12 u32 tensor count
 *   16 u64 index offset  24 u64 index bytes  32 sha256 of the index bytes
 *   Payloads (PTT1) sit between the header and the index, in order, back to back, without overlap.
 *   Index entry: u32 name length (1..4096), name bytes (no NUL), u64 payload offset, u64 payload length,
 *   sha256 of the payload. The index must end exactly at the end of the file.
 */
#ifndef PCHIE_TERNARY_PACK_H
#define PCHIE_TERNARY_PACK_H

#include "pchie/pchie_ternary_convert.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { TERNARY_PTT1_HEADER_BYTES = 160 };
enum { TERNARY_PTP1_HEADER_BYTES = 64 };
enum { TERNARY_RANS_PROB_BITS = 12 };
enum { TERNARY_RANS_PROB_TOTAL = 1 << 12 };
enum { TERNARY_PACK_MAX_NAME_BYTES = 4096 };

typedef enum ternary_tensor_kind {
    TERNARY_KIND_TERNARY = 1,
    TERNARY_KIND_RAW = 2,
    TERNARY_KIND_Q8 = 3
} ternary_tensor_kind;

typedef enum ternary_trit_encoding {
    TERNARY_TRITS_B3 = 1,
    TERNARY_TRITS_RANS = 2
} ternary_trit_encoding;

typedef enum ternary_scale_encoding {
    TERNARY_SCALES_F32 = 1,
    TERNARY_SCALES_F16 = 2,
    TERNARY_SCALES_U8ROW = 3
} ternary_scale_encoding;

typedef enum ternary_raw_dtype {
    TERNARY_RAW_F32 = 1,
    TERNARY_RAW_F16 = 2,
    TERNARY_RAW_BF16 = 3
} ternary_raw_dtype;

const char *ternary_trit_encoding_name(int encoding);
const char *ternary_scale_encoding_name(int encoding);

/* ---- B3 rows -------------------------------------------------------------------------------------------- */
/* ceil(columns / 5); 0 if columns is 0 or over TERNARY_HARD_MAX_COLUMNS. */
uint64_t ternary_b3_row_bytes(uint64_t columns);
ternary_convert_status ternary_pack_row_b3(const int8_t *codes, uint64_t columns, uint8_t *packed);
/* Refuses (BAD_CODE) a byte over 242 or a non-zero padding weight; codes is then zero-filled. */
ternary_convert_status ternary_unpack_row_b3(const uint8_t *packed, uint64_t columns, int8_t *codes);
/* Direct I2 <-> B3 transcoding of one row (columns a multiple of 128). Same checks as the unpackers. */
ternary_convert_status ternary_b3_from_i2_row(const uint8_t *i2, uint64_t columns, uint8_t *b3);
ternary_convert_status ternary_i2_from_b3_row(const uint8_t *b3, uint64_t columns, uint8_t *i2);

/* ---- groups and scales ---------------------------------------------------------------------------------- */
/* ceil(columns / 128); 0 if columns is 0 or over the ceiling. */
uint64_t ternary_groups_per_row(uint64_t columns);
/* Bytes of a scales section (0 on a bad argument or overflow). */
uint64_t ternary_scale_section_bytes(ternary_scale_encoding encoding, uint64_t rows, uint64_t groups_per_row);
/* scales: rows * groups_per_row values, each finite, >= 0 and 0 or normal (BAD_SCALE otherwise). out gets
 * ternary_scale_section_bytes() bytes. decoded (may be NULL) gets the scales a reader will see. */
ternary_convert_status ternary_encode_scales(ternary_scale_encoding encoding, const float *scales, uint64_t rows,
                                             uint64_t groups_per_row, uint8_t *out, float *decoded);
/* Validates and decodes a scales section of exactly section_bytes bytes into out (rows * groups_per_row). */
ternary_convert_status ternary_decode_scales(ternary_scale_encoding encoding, const uint8_t *section,
                                             uint64_t section_bytes, uint64_t rows, uint64_t groups_per_row,
                                             float *out);

/* ---- rANS trit streams ---------------------------------------------------------------------------------- */
/* Encodes rows * columns trits (int8 in {-1, 0, 1}, row-major) into a malloc'd RANS data section (the caller
 * frees *section). rows_per_chunk >= 1. */
ternary_convert_status ternary_rans_encode(const int8_t *codes, uint64_t rows, uint64_t columns,
                                           uint32_t rows_per_chunk, uint8_t **section, size_t *section_bytes);
/* Structure check of a RANS section for rows x columns trits (no decoding). */
ternary_convert_status ternary_rans_check(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                          uint64_t columns);
/* Decodes chunk `chunk` (rows [chunk * rows_per_chunk, ...)) into codes (that many rows * columns); *first_row
 * and *row_count say which rows those are. Every byte and the end state are checked. */
ternary_convert_status ternary_rans_decode_chunk(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                                 uint64_t columns, uint32_t chunk, int8_t *codes,
                                                 uint64_t *first_row, uint64_t *row_count);
/* Whole tensor. */
ternary_convert_status ternary_rans_decode(const uint8_t *section, uint64_t section_bytes, uint64_t rows,
                                           uint64_t columns, int8_t *codes);
uint32_t ternary_rans_chunk_count(const uint8_t *section, uint64_t section_bytes);

/* ---- PTT1 tensors --------------------------------------------------------------------------------------- */
typedef struct ternary_ptt1_desc {
    ternary_tensor_kind kind;
    int trit_encoding;                           /* ternary only */
    int scale_encoding;                          /* ternary and q8 */
    int raw_dtype;                               /* raw only */
    uint64_t dims[TERNARY_SOURCE_MAX_DIMENSIONS];
    unsigned dimension_count;
    uint64_t rows;
    uint64_t columns;
} ternary_ptt1_desc;

typedef struct ternary_ptt1_view {
    ternary_ptt1_desc desc;
    const uint8_t *scales;
    uint64_t scales_bytes;
    const uint8_t *data;
    uint64_t data_bytes;
} ternary_ptt1_view;

/* Builds a PTT1 from already-encoded sections (malloc'd, the caller frees *bytes). */
int ternary_build_ptt1(const ternary_ptt1_desc *desc, const uint8_t *scales, uint64_t scales_bytes,
                       const uint8_t *data, uint64_t data_bytes, uint8_t **bytes, size_t *length,
                       char *error, size_t error_capacity);
/* Validates the header, the sections' sizes, the checksum and (for RANS) the section structure. Contents of
 * the trit stream are checked by the decoders. */
ternary_convert_status ternary_parse_ptt1(const uint8_t *buffer, size_t length, uint64_t max_bytes,
                                          ternary_ptt1_view *view);
/* Ternary tensors: decodes all codes (rows * columns int8) and all scales (rows * groups_per_row floats).
 * Either output may be NULL. */
ternary_convert_status ternary_ptt1_decode_ternary(const ternary_ptt1_view *view, int8_t *codes, float *scales);

/* ---- PTP1 packs ----------------------------------------------------------------------------------------- */
typedef struct ternary_ptp1_entry {
    const char *name;                            /* not NUL-terminated; name_length bytes */
    uint32_t name_length;
    uint64_t offset;
    uint64_t length;
    const uint8_t *sha256;                       /* 32 bytes inside the index */
} ternary_ptp1_entry;

typedef struct ternary_ptp1_view {
    uint32_t tensor_count;
    ternary_ptp1_entry *entries;                 /* malloc'd; ternary_ptp1_view_clear frees it */
    uint64_t index_offset;
    uint64_t index_bytes;
} ternary_ptp1_view;

void ternary_ptp1_view_clear(ternary_ptp1_view *view);
/* header: the first TERNARY_PTP1_HEADER_BYTES bytes; index: the index bytes it points to; file_size: the whole
 * file. Checks the index hash, every entry, ordering and extents. Payload hashes are checked by the caller. */
ternary_convert_status ternary_parse_ptp1(const uint8_t *header, size_t header_length, const uint8_t *index,
                                          size_t index_length, uint64_t file_size, ternary_ptp1_view *view);
/* Fills a PTP1 header for the given index. */
void ternary_build_ptp1_header(uint32_t tensor_count, uint64_t index_offset, const uint8_t *index,
                               size_t index_length, uint8_t header[TERNARY_PTP1_HEADER_BYTES]);

/* ---- fit of rows of any length ---------------------------------------------------------------------- */
/* Least-squares ternary fit of rows x cols weights (row-major), one scale per 128-weight group, the last group
 * of a row holding cols % 128 weights when that is not 0. Each group is zero-padded to 128 and given to the same
 * group fit as the converter: for least squares that is exactly the fit of the real weights (a zero never joins
 * the chosen set, since it would lower P_k^2 / k, and always gets code 0). codes: rows * cols, scales:
 * rows * ceil(cols / 128). */
ternary_convert_status ternary_fit_rows_padded(const float *weights, uint64_t rows, uint64_t cols, int thread_count,
                                               int8_t *codes, float *scales);

/* ---- small shared helpers ------------------------------------------------------------------------------- */
uint16_t ternary_float32_to_float16(float value);   /* round to nearest even; overflow gives infinity */
void ternary_sha256(const uint8_t *data, size_t length, uint8_t out[32]);

#ifdef __cplusplus
}
#endif

#endif /* PCHIE_TERNARY_PACK_H */
