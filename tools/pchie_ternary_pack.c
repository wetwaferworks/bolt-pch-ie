/* Bolt PCHIE - pchie_ternary_pack: packs a ternary model into PTP1 files (one per source shard) with B3 or rANS
 * trits and compact scales, checks a pack against the existing PTN1 conversion, re-encodes packs losslessly and
 * measures decode speed (clean-room, C17).
 *
 *   pack    <model.safetensors|gguf> --out FILE.ptp [--reference PTN_DIR] [--scales tight|f32|f16|u8row]
 *           [--tolerance REL] [--trits b3|rans] [--embed q8|raw|ternary|skip] [--conv1d raw|ternary|skip]
 *           [--extras raw|skip] [--threads N] [--chunk-mweights M]
 *   from-ptn PTN_DIR --out FILE.ptp [--trits b3|rans]      (lossless: the PTN1 codes and float32 scales, re-packed)
 *   recode  IN.ptp --out OUT.ptp --trits b3|rans           (lossless: only the trit encoding changes)
 *   verify  FILE.ptp [--reference PTN_DIR]                 (hashes, full decode, exact I2 parity with PTN1)
 *   bench   FILE.ptp [--max-mweights M]                    (single-thread unpack speed)
 *
 * The model, the reference folder and the input pack are opened read-only. Output is written to FILE.partial
 * (created new; an existing file is refused) and renamed to FILE only after it is complete.
 * "tight" picks, per tensor, the first of u8row, f16, f32 whose reconstruction error against the source weights
 * is within the tolerance (default 1e-4, relative) of the float32 scales; f32 always qualifies. Codes never
 * change, so the ternary values always match the PTN1 conversion exactly.
 */
#include "pchie/pchie_ternary_pack.h"
#include "pchie_ternary_convert_internal.h"
#include "pj_sha256.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(precise, on)
#pragma fp_contract(off)
#endif

static double now_ms(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static int usage(void) {
    fprintf(stderr,
            "usage:\n"
            "  pchie_ternary_pack pack MODEL --out FILE.ptp [--reference PTN_DIR] [--scales tight|f32|f16|u8row]\n"
            "       [--tolerance REL] [--trits b3|rans] [--embed q8|raw|ternary|skip] [--conv1d raw|ternary|skip]\n"
            "       [--extras raw|skip] [--threads N] [--chunk-mweights M]\n"
            "  pchie_ternary_pack from-ptn PTN_DIR --out FILE.ptp [--trits b3|rans]\n"
            "  pchie_ternary_pack recode IN.ptp --out OUT.ptp --trits b3|rans\n"
            "  pchie_ternary_pack verify FILE.ptp [--reference PTN_DIR]\n"
            "  pchie_ternary_pack bench FILE.ptp [--max-mweights M]\n");
    return 2;
}

static void hex32(const uint8_t d[32], char out[65]) {
    static const char *x = "0123456789abcdef";
    int i;
    for (i = 0; i < 32; ++i) {
        out[2 * i] = x[d[i] >> 4];
        out[2 * i + 1] = x[d[i] & 15];
    }
    out[64] = 0;
}

static int contains(const char *s, const char *needle) { return s && strstr(s, needle) != NULL; }
static int ends_with(const char *s, const char *suffix) {
    size_t a = strlen(s), b = strlen(suffix);
    return a >= b && strcmp(s + a - b, suffix) == 0;
}

typedef struct strbuf {
    char *p;
    size_t len, cap;
} strbuf;

static void sb_printf(strbuf *b, const char *fmt, ...) {
    va_list ap, ap2;
    int n;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) {
        va_end(ap2);
        return;
    }
    if (b->len + (size_t)n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        char *q;
        while (b->len + (size_t)n + 1 > cap) cap *= 2;
        q = (char *)realloc(b->p, cap);
        if (!q) {
            va_end(ap2);
            return;
        }
        b->p = q;
        b->cap = cap;
    }
    vsnprintf(b->p + b->len, (size_t)n + 1, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

/* Absolute form of a path with links and junctions resolved (for "is the output inside that folder" checks).
 * A path that does not exist yet is resolved through its parent folder, which must exist; its last part may not
 * be "." or "..". */
static int last_part_ok(const char *name) { return name[0] && strcmp(name, ".") != 0 && strcmp(name, "..") != 0; }
#if defined(_WIN32)
static int final_path_win(const char *in, char *out, size_t cap) {
    HANDLE h = CreateFileA(in, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, NULL);
    DWORD n;
    char *s;
    if (h == INVALID_HANDLE_VALUE) return 0;
    n = GetFinalPathNameByHandleA(h, out, (DWORD)cap, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    CloseHandle(h);
    if (n == 0 || n >= cap) return 0;
    if (strncmp(out, "\\\\?\\", 4) == 0) memmove(out, out + 4, strlen(out + 4) + 1);
    for (s = out; *s; ++s) {
        if (*s == '/') *s = '\\';
        if (*s >= 'A' && *s <= 'Z') *s = (char)(*s - 'A' + 'a');
    }
    return 1;
}
#endif
static int full_path(const char *in, char *out, size_t cap) {
    char tmp[4096], parent[4096];
    char *slash;
#if defined(_WIN32)
    char *b;
    if (final_path_win(in, out, cap)) return 1;
    if (strlen(in) >= sizeof(tmp)) return 0;
    strcpy(tmp, in);
    slash = strrchr(tmp, '/');
    b = strrchr(tmp, '\\');
    if (b && (!slash || b > slash)) slash = b;
    if (!slash) {
        if (!last_part_ok(tmp) || !final_path_win(".", parent, sizeof(parent))) return 0;
        return snprintf(out, cap, "%s\\%s", parent, tmp) < (int)cap;
    }
    *slash = 0;
    if (!last_part_ok(slash + 1) || !final_path_win(tmp[0] ? tmp : "\\", parent, sizeof(parent))) return 0;
    for (b = slash + 1; *b; ++b)
        if (*b >= 'A' && *b <= 'Z') *b = (char)(*b - 'A' + 'a');
    return snprintf(out, cap, "%s\\%s", parent, slash + 1) < (int)cap;
#else
    if (realpath(in, out)) return 1;
    if (strlen(in) >= sizeof(tmp)) return 0;
    strcpy(tmp, in);
    slash = strrchr(tmp, '/');
    if (!slash) {
        if (!last_part_ok(tmp) || !getcwd(parent, sizeof(parent))) return 0;
        return snprintf(out, cap, "%s/%s", parent, tmp) < (int)cap;
    }
    *slash = 0;
    if (!last_part_ok(slash + 1) || !realpath(tmp[0] ? tmp : "/", parent)) return 0;
    return snprintf(out, cap, "%s/%s", parent, slash + 1) < (int)cap;
#endif
}

static void parent_of(const char *path, char *out, size_t cap) {
    const char *a = strrchr(path, '/');
#if defined(_WIN32)
    const char *b = strrchr(path, '\\');
    if (b && (!a || b > a)) a = b;
#endif
    if (!a) {
        snprintf(out, cap, ".");
        return;
    }
    snprintf(out, cap, "%.*s", (int)(a - path), path);
}

static int inside(const char *child, const char *dir) {
    size_t n = strlen(dir);
    if (n == 0 || strncmp(child, dir, n) != 0) return 0;
    return child[n] == 0 || child[n] == '/' || child[n] == '\\';
}

/* Creates a new file for writing; refuses one that already exists. */
static FILE *create_new(const char *path) {
#if defined(_WIN32)
    int fd = -1;
    if (_sopen_s(&fd, path, _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _SH_DENYWR, _S_IREAD | _S_IWRITE) != 0) return NULL;
    return _fdopen(fd, "wb");
#else
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    FILE *f;
    if (fd < 0) return NULL;
    f = fdopen(fd, "wb");
    if (!f) close(fd);
    return f;
#endif
}

static int rename_new(const char *from, const char *to) {
#if defined(_WIN32)
    return MoveFileExA(from, to, MOVEFILE_WRITE_THROUGH) ? 0 : -1; /* no REPLACE_EXISTING: never overwrites */
#else
    if (access(to, F_OK) == 0) return -1;
    return rename(from, to);
#endif
}

static int file_exists(const char *path) {
    FILE *f = ternary_open_file_read_only(path);
    if (!f) return 0;
    fclose(f);
    return 1;
}

static uint8_t *read_whole(const char *path, size_t *length, uint64_t max) {
    FILE *f = ternary_open_file_read_only(path);
    uint64_t n;
    uint8_t *buf;
    *length = 0;
    if (!f) return NULL;
    n = ternary_file_length(f);
    if (n == 0 || n > max || n > (uint64_t)SIZE_MAX || ternary_seek64(f, 0) != 0) {
        fclose(f);
        return NULL;
    }
    buf = (uint8_t *)malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *length = (size_t)n;
    return buf;
}

static int read_at(FILE *f, uint64_t offset, uint8_t *buf, uint64_t n) {
    if (ternary_seek64(f, offset) != 0) return 0;
    return fread(buf, 1, (size_t)n, f) == (size_t)n;
}

static int sha256_file(const char *path, char out[65]) {
    FILE *f = ternary_open_file_read_only(path);
    pj_sha256 c;
    uint8_t d[32];
    uint8_t *buf;
    size_t n;
    if (!f) return 0;
    buf = (uint8_t *)malloc(1 << 20);
    if (!buf) {
        fclose(f);
        return 0;
    }
    pj_sha256_init(&c);
    while ((n = fread(buf, 1, 1 << 20, f)) > 0) pj_sha256_update(&c, buf, n);
    fclose(f);
    free(buf);
    pj_sha256_final(&c, d);
    hex32(d, out);
    return 1;
}

/* ------------------------------------------------------------------------------------------------ PTN1 reference */

typedef struct ref_entry {
    char *name;
    char *file;
} ref_entry;

typedef struct ref_dir {
    char dir[4096];
    ref_entry *items;
    size_t count;
} ref_dir;

static void ref_dir_free(ref_dir *r) {
    size_t i;
    for (i = 0; i < r->count; ++i) {
        free(r->items[i].name);
        free(r->items[i].file);
    }
    free(r->items);
    memset(r, 0, sizeof(*r));
}

static char *dup_n(const char *s, size_t n) {
    char *p = (char *)malloc(n + 1);
    if (!p) return NULL;
    memcpy(p, s, n);
    p[n] = 0;
    return p;
}

/* manifest.tsv: a header line naming the columns, then one line per tensor; we need "name" and "file". */
static int ref_dir_load(const char *dir, ref_dir *r) {
    char path[4600];
    size_t len, pos = 0, line_no = 0;
    int name_col = -1, file_col = -1;
    uint8_t *text;
    memset(r, 0, sizeof(*r));
    if (strlen(dir) >= sizeof(r->dir)) return 0;
    strcpy(r->dir, dir);
    snprintf(path, sizeof(path), "%s/manifest.tsv", dir);
    text = read_whole(path, &len, UINT64_C(64) << 20);
    if (!text) return 0;
    while (pos < len) {
        size_t end = pos, col = 0, start;
        const char *fname = NULL, *ffile = NULL;
        size_t name_len = 0, file_len = 0;
        while (end < len && text[end] != '\n') ++end;
        start = pos;
        for (;;) {
            size_t fe = start;
            while (fe < end && text[fe] != '\t' && text[fe] != '\r') ++fe;
            if (line_no == 0) {
                if (fe - start == 4 && memcmp(text + start, "name", 4) == 0) name_col = (int)col;
                if (fe - start == 4 && memcmp(text + start, "file", 4) == 0) file_col = (int)col;
            } else {
                if ((int)col == name_col) {
                    fname = (const char *)text + start;
                    name_len = fe - start;
                }
                if ((int)col == file_col) {
                    ffile = (const char *)text + start;
                    file_len = fe - start;
                }
            }
            ++col;
            if (fe >= end || text[fe] == '\r') break;
            start = fe + 1;
        }
        if (line_no == 0 && (name_col < 0 || file_col < 0)) {
            free(text);
            return 0;
        }
        if (line_no > 0 && fname && ffile && name_len && file_len) {
            ref_entry *grown;
            if (memchr(ffile, '/', file_len) || memchr(ffile, '\\', file_len) || memchr(ffile, ':', file_len) ||
                (file_len >= 2 && ffile[0] == '.' && ffile[1] == '.')) {
                free(text);
                return 0; /* file names only, never paths */
            }
            grown = (ref_entry *)realloc(r->items, (r->count + 1) * sizeof(ref_entry));
            if (!grown) {
                free(text);
                return 0;
            }
            r->items = grown;
            r->items[r->count].name = dup_n(fname, name_len);
            r->items[r->count].file = dup_n(ffile, file_len);
            r->count++;
        }
        ++line_no;
        pos = end + 1;
    }
    free(text);
    return 1;
}

static const ref_entry *ref_find(const ref_dir *r, const char *name, size_t name_len) {
    size_t i;
    for (i = 0; i < r->count; ++i)
        if (r->items[i].name && strlen(r->items[i].name) == name_len && memcmp(r->items[i].name, name, name_len) == 0)
            return &r->items[i];
    return NULL;
}

/* Loads one reference PTN1; *buffer is the caller's to free. */
static int ref_load(const ref_dir *r, const ref_entry *e, uint8_t **buffer, ternary_ptn1_view *view) {
    char path[8800];
    size_t len;
    snprintf(path, sizeof(path), "%s/%s", r->dir, e->file);
    *buffer = read_whole(path, &len, UINT64_C(4) << 30);
    if (!*buffer) return 0;
    if (ternary_parse_ptn1(*buffer, len, UINT64_C(4) << 30, view) != TERNARY_CONVERT_OK) {
        free(*buffer);
        *buffer = NULL;
        return 0;
    }
    return 1;
}

static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/* ------------------------------------------------------------------------------------------------ pack writer */

typedef struct pack_writer {
    FILE *file;
    char partial[4200];
    char final_path[4096];
    uint64_t offset;
    strbuf index;
    uint32_t count;
} pack_writer;

static int pack_open(pack_writer *w, const char *path) {
    uint8_t zero[TERNARY_PTP1_HEADER_BYTES];
    memset(w, 0, sizeof(*w));
    if (strlen(path) >= sizeof(w->final_path)) return 0;
    strcpy(w->final_path, path);
    snprintf(w->partial, sizeof(w->partial), "%s.partial", path);
    if (file_exists(path)) {
        fprintf(stderr, "error: %s already exists\n", path);
        return 0;
    }
    w->file = create_new(w->partial);
    if (!w->file) {
        fprintf(stderr, "error: cannot create %s (it may exist already)\n", w->partial);
        return 0;
    }
    memset(zero, 0, sizeof(zero));
    if (fwrite(zero, 1, sizeof(zero), w->file) != sizeof(zero)) return 0;
    w->offset = TERNARY_PTP1_HEADER_BYTES;
    return 1;
}

static void pack_abandon(pack_writer *w) {
    if (w->file) {
        fclose(w->file);
        w->file = NULL;
        remove(w->partial);
    }
    free(w->index.p);
    w->index.p = NULL;
}

static int pack_add(pack_writer *w, const char *name, const uint8_t *payload, size_t length, char sha_hex[65]) {
    uint8_t digest[32];
    size_t name_len = strlen(name), need = 4 + name_len + 48;
    uint8_t *p;
    int i;
    if (!name_len || name_len > TERNARY_PACK_MAX_NAME_BYTES) return 0;
    ternary_sha256(payload, length, digest);
    if (sha_hex) hex32(digest, sha_hex);
    if (fwrite(payload, 1, length, w->file) != length) return 0;
    if (w->index.len + need > w->index.cap) {
        size_t cap = w->index.cap ? w->index.cap : 65536;
        char *q;
        while (w->index.len + need > cap) cap *= 2;
        q = (char *)realloc(w->index.p, cap);
        if (!q) return 0;
        w->index.p = q;
        w->index.cap = cap;
    }
    p = (uint8_t *)w->index.p + w->index.len;
    for (i = 0; i < 4; ++i) p[i] = (uint8_t)(name_len >> (8 * i));
    memcpy(p + 4, name, name_len);
    p += 4 + name_len;
    for (i = 0; i < 8; ++i) p[i] = (uint8_t)(w->offset >> (8 * i));
    for (i = 0; i < 8; ++i) p[8 + i] = (uint8_t)((uint64_t)length >> (8 * i));
    memcpy(p + 16, digest, 32);
    w->index.len += need;
    w->offset += length;
    w->count++;
    return 1;
}

static int pack_close(pack_writer *w, char file_sha[65]) {
    uint8_t header[TERNARY_PTP1_HEADER_BYTES];
    int ok = 1;
    ternary_build_ptp1_header(w->count, w->offset, (const uint8_t *)w->index.p, w->index.len, header);
    if (w->index.len && fwrite(w->index.p, 1, w->index.len, w->file) != w->index.len) ok = 0;
    if (ok && (ternary_seek64(w->file, 0) != 0 || fwrite(header, 1, sizeof(header), w->file) != sizeof(header))) ok = 0;
    if (fflush(w->file) != 0) ok = 0;
    if (fclose(w->file) != 0) ok = 0;
    w->file = NULL;
    free(w->index.p);
    w->index.p = NULL;
    if (!ok) {
        remove(w->partial);
        return 0;
    }
    if (file_sha && !sha256_file(w->partial, file_sha)) return 0;
    if (rename_new(w->partial, w->final_path) != 0) {
        fprintf(stderr, "error: cannot rename %s to %s\n", w->partial, w->final_path);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------------------------------ pack reader */

typedef struct pack_reader {
    FILE *file;
    uint8_t *index;
    ternary_ptp1_view view;
    uint64_t size;
} pack_reader;

static void pack_reader_close(pack_reader *r) {
    if (r->file) fclose(r->file);
    free(r->index);
    ternary_ptp1_view_clear(&r->view);
    memset(r, 0, sizeof(*r));
}

static int pack_reader_open(pack_reader *r, const char *path) {
    uint8_t header[TERNARY_PTP1_HEADER_BYTES];
    uint64_t index_offset = 0, index_bytes = 0;
    ternary_convert_status st;
    int i;
    memset(r, 0, sizeof(*r));
    r->file = ternary_open_file_read_only(path);
    if (!r->file) return 0;
    r->size = ternary_file_length(r->file);
    if (r->size < TERNARY_PTP1_HEADER_BYTES || !read_at(r->file, 0, header, sizeof(header))) goto bad;
    for (i = 0; i < 8; ++i) {
        index_offset |= (uint64_t)header[16 + i] << (8 * i);
        index_bytes |= (uint64_t)header[24 + i] << (8 * i);
    }
    if (index_offset > r->size || index_bytes != r->size - index_offset || index_bytes > (UINT64_C(256) << 20)) goto bad;
    r->index = (uint8_t *)malloc(index_bytes ? (size_t)index_bytes : 1);
    if (!r->index || (index_bytes && !read_at(r->file, index_offset, r->index, index_bytes))) goto bad;
    st = ternary_parse_ptp1(header, sizeof(header), r->index, (size_t)index_bytes, r->size, &r->view);
    if (st != TERNARY_CONVERT_OK) {
        fprintf(stderr, "error: %s: %s\n", path, ternary_convert_status_name(st));
        goto bad;
    }
    return 1;
bad:
    pack_reader_close(r);
    return 0;
}

/* Reads and hash-checks payload i (malloc'd). */
static uint8_t *pack_payload(pack_reader *r, uint32_t i, int *hash_ok) {
    const ternary_ptp1_entry *e = &r->view.entries[i];
    uint8_t *buf, d[32];
    *hash_ok = 0;
    if (e->length > (uint64_t)SIZE_MAX) return NULL;
    buf = (uint8_t *)malloc((size_t)e->length);
    if (!buf || !read_at(r->file, e->offset, buf, e->length)) {
        free(buf);
        return NULL;
    }
    ternary_sha256(buf, (size_t)e->length, d);
    *hash_ok = memcmp(d, e->sha256, 32) == 0;
    return buf;
}

static void entry_name(const ternary_ptp1_entry *e, char out[TERNARY_PACK_MAX_NAME_BYTES + 1]) {
    memcpy(out, e->name, e->name_length);
    out[e->name_length] = 0;
}

/* ------------------------------------------------------------------------------------------------ error stats */

typedef ternary_reconstruction_error rstats;

/* Any column count, partial last group: reconstruction = scale[group] * value. Sums in row order. */
static void stats_add_rows(rstats *s, const float *w, const int8_t *q, const float *scales, uint64_t rows,
                           uint64_t cols) {
    const uint64_t gpr = ternary_groups_per_row(cols);
    uint64_t r, c;
    for (r = 0; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            const double sc = (double)scales[r * gpr + c / TERNARY_GROUP_WEIGHT_COUNT];
            const double wv = (double)w[r * cols + c];
            const double rec = sc * (double)q[r * cols + c];
            const double e = wv - rec;
            s->sum_weight_squared += wv * wv;
            s->sum_error_squared += e * e;
            s->sum_weight_times_reconstruction += wv * rec;
            s->sum_reconstruction_squared += rec * rec;
            if (fabs(e) > s->max_absolute_error) s->max_absolute_error = fabs(e);
            if (q[r * cols + c] == 0) s->zero_codes++;
            s->weight_count++;
        }
        for (c = 0; c < gpr; ++c) {
            s->group_count++;
            if (scales[r * gpr + c] == 0.0f) s->zero_groups++;
        }
    }
}

static double rel_of(const rstats *s) { return ternary_reconstruction_relative_error(s); }
static double cos_of(const rstats *s) { return ternary_reconstruction_cosine(s); }

typedef struct par_stats_ctx {
    const float *w;
    const int8_t *q;
    const float *scales;
    uint64_t cols;
    rstats *per_row;
} par_stats_ctx;

static void par_stats_worker(uint64_t begin, uint64_t end, int worker_index, void *vctx) {
    par_stats_ctx *c = (par_stats_ctx *)vctx;
    const uint64_t gpr = ternary_groups_per_row(c->cols);
    uint64_t r;
    (void)worker_index;
    for (r = begin; r < end; ++r)
        stats_add_rows(&c->per_row[r], c->w + r * c->cols, c->q + r * c->cols, c->scales + r * gpr, 1, c->cols);
}

/* Same sums as stats_add_rows, rows spread over threads, added up in row order (any thread count, same bits). */
static int stats_add_rows_parallel(rstats *s, const float *w, const int8_t *q, const float *scales, uint64_t rows,
                                   uint64_t cols, int threads) {
    par_stats_ctx c;
    uint64_t r;
    c.per_row = (rstats *)calloc((size_t)rows, sizeof(rstats));
    if (!c.per_row) return 0;
    c.w = w;
    c.q = q;
    c.scales = scales;
    c.cols = cols;
    ternary_run_in_parallel(rows, threads, par_stats_worker, &c);
    for (r = 0; r < rows; ++r) ternary_reconstruction_error_add(s, &c.per_row[r]);
    free(c.per_row);
    return 1;
}

/* ------------------------------------------------------------------------------------------------ pack mode */

typedef enum policy { POLICY_SKIP = 0, POLICY_RAW, POLICY_Q8, POLICY_TERNARY } policy;

typedef struct pack_options {
    const char *model;
    const char *out;
    const char *reference;
    int scales; /* 0 = tight, else a ternary_scale_encoding */
    double tolerance;
    int trits;
    policy embed;
    policy conv1d;
    policy extras;
    int threads;
    uint64_t chunk_weights;
} pack_options;

typedef struct totals {
    uint64_t tensors[5], weights[5], bytes[5]; /* by kind: 1 ternary 2 raw 3 q8 4 skipped */
    uint64_t trit_bytes, scale_bytes;
    uint64_t scales_used[4];
    rstats base, chosen, u8, f16, q8, ext_embed, ext_conv1d;
    uint64_t parity_checked, parity_code_mismatch, parity_scale_mismatch, parity_missing;
} totals;

static int source_raw_dtype(const ternary_source_tensor *t) {
    switch (t->type) {
        case TERNARY_SOURCE_WEIGHT_F32: return TERNARY_RAW_F32;
        case TERNARY_SOURCE_WEIGHT_F16: return TERNARY_RAW_F16;
        case TERNARY_SOURCE_WEIGHT_BF16: return TERNARY_RAW_BF16;
        default: return 0;
    }
}

static void desc_from_source(const ternary_source_tensor *t, ternary_ptt1_desc *d) {
    size_t i;
    memset(d, 0, sizeof(*d));
    if (t->dimension_count == 0) {
        d->dims[0] = 1;
        d->dimension_count = 1;
    } else {
        for (i = 0; i < t->dimension_count && i < TERNARY_SOURCE_MAX_DIMENSIONS; ++i) d->dims[i] = t->shape[i];
        d->dimension_count = (unsigned)t->dimension_count;
    }
    d->rows = t->rows ? t->rows : 1;
    d->columns = t->columns ? t->columns : 1;
}

static uint32_t rans_rows_per_chunk(uint64_t rows, uint64_t cols) {
    uint64_t r = (UINT64_C(1) << 20) / cols; /* about 1 Mi trits per chunk, whole rows */
    if (r < 1) r = 1;
    if (r > rows) r = rows;
    if (r > UINT32_MAX) r = UINT32_MAX;
    return (uint32_t)r;
}

static int encode_trits(int trits, const int8_t *codes, uint64_t rows, uint64_t cols, uint8_t **data,
                        size_t *data_bytes) {
    const uint64_t rb = ternary_b3_row_bytes(cols);
    uint64_t r;
    *data = NULL;
    *data_bytes = 0;
    if (trits == TERNARY_TRITS_RANS)
        return ternary_rans_encode(codes, rows, cols, rans_rows_per_chunk(rows, cols), data, data_bytes) ==
               TERNARY_CONVERT_OK;
    *data = (uint8_t *)malloc(rows * rb > 0 ? (size_t)(rows * rb) : 1);
    if (!*data || !rb) {
        free(*data);
        *data = NULL;
        return 0;
    }
    for (r = 0; r < rows; ++r)
        if (ternary_pack_row_b3(codes + r * cols, cols, *data + r * rb) != TERNARY_CONVERT_OK) {
            free(*data);
            *data = NULL;
            return 0;
        }
    *data_bytes = (size_t)(rows * rb);
    return 1;
}

/* Reads a source tensor's rows in chunks and calls fn with float weights. */
typedef int (*rows_fn)(const float *w, uint64_t first_row, uint64_t row_count, void *ctx);

static int for_each_rows(FILE *f, const ternary_source_file *src, const ternary_source_tensor *t, uint64_t chunk_weights,
                         rows_fn fn, void *ctx) {
    uint64_t per = chunk_weights / t->columns, r;
    uint8_t *raw = NULL;
    float *w = NULL;
    int ok = 1;
    (void)src;
    if (t->bytes_per_row == 0 || t->bytes_per_row * t->rows != t->byte_count || t->columns > TERNARY_HARD_MAX_COLUMNS) {
        fprintf(stderr, "error: %s: rows can't be decoded here\n", t->name);
        return 0;
    }
    if (per < 1) per = 1;
    if (per > t->rows) per = t->rows;
    w = (float *)malloc((size_t)(per * t->columns) * sizeof(float));
    raw = (uint8_t *)malloc((size_t)(per * t->bytes_per_row));
    if (!w || !raw) ok = 0;
    for (r = 0; r < t->rows && ok; r += per) {
        const uint64_t n = (t->rows - r < per) ? t->rows - r : per;
        ternary_convert_status st;
        if (!read_at(f, t->file_offset + r * t->bytes_per_row, raw, n * t->bytes_per_row)) {
            fprintf(stderr, "error: %s: short read\n", t->name);
            ok = 0;
            break;
        }
        st = ternary_decode_rows(t, raw, n, w);
        if (st != TERNARY_CONVERT_OK) {
            fprintf(stderr, "error: %s: %s\n", t->name, ternary_convert_status_name(st));
            ok = 0;
            break;
        }
        ok = fn(w, r, n, ctx);
    }
    free(w);
    free(raw);
    return ok;
}

typedef struct variant_ctx {
    const int8_t *codes;
    const float *scales[3]; /* decoded f32 (= the fit), f16, u8row */
    uint64_t cols;
    int threads;
    rstats st[3];
} variant_ctx;

static int variant_rows(const float *w, uint64_t first, uint64_t n, void *vctx) {
    variant_ctx *c = (variant_ctx *)vctx;
    const uint64_t gpr = ternary_groups_per_row(c->cols);
    int v;
    for (v = 0; v < 3; ++v)
        if (!stats_add_rows_parallel(&c->st[v], w, c->codes + first * c->cols, c->scales[v] + first * gpr, n, c->cols,
                                     c->threads))
            return 0;
    return 1;
}

typedef struct collect_ctx {
    float *all;
    uint64_t cols;
} collect_ctx;

static int collect_rows(const float *w, uint64_t first, uint64_t n, void *vctx) {
    collect_ctx *c = (collect_ctx *)vctx;
    memcpy(c->all + first * c->cols, w, (size_t)(n * c->cols) * sizeof(float));
    return 1;
}

/* Symmetric int8 per 128-weight group (the last may be shorter): scale = absmax / 127, q = round-half-even(w / s). */
typedef struct q8_ctx {
    int8_t *q;
    float *scales;
    uint64_t cols;
    int threads;
    rstats st;
} q8_ctx;

static void q8_quantize_row(const float *row, uint64_t cols, int8_t *qr, float *sr) {
    const uint64_t gpr = ternary_groups_per_row(cols);
    uint64_t g, k;
    for (g = 0; g < gpr; ++g) {
        const uint64_t b = g * TERNARY_GROUP_WEIGHT_COUNT;
        const uint64_t e = (b + TERNARY_GROUP_WEIGHT_COUNT < cols) ? b + TERNARY_GROUP_WEIGHT_COUNT : cols;
        float amax = 0.0f, scale;
        for (k = b; k < e; ++k)
            if (fabsf(row[k]) > amax) amax = fabsf(row[k]);
        scale = amax / 127.0f;
        if (!(scale >= FLT_MIN)) scale = 0.0f;
        sr[g] = scale;
        for (k = b; k < e; ++k) {
            float v = scale > 0.0f ? nearbyintf(row[k] / scale) : 0.0f;
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            qr[k] = (int8_t)v;
        }
    }
}

static int q8_rows(const float *w, uint64_t first, uint64_t n, void *vctx) {
    q8_ctx *c = (q8_ctx *)vctx;
    const uint64_t gpr = ternary_groups_per_row(c->cols);
    uint64_t r;
    for (r = 0; r < n; ++r)
        q8_quantize_row(w + r * c->cols, c->cols, c->q + (first + r) * c->cols, c->scales + (first + r) * gpr);
    return stats_add_rows_parallel(&c->st, w, c->q + first * c->cols, c->scales + first * gpr, n, c->cols, c->threads);
}

/* klass: 1 ternary (as in the PTN1 conversion), 2 input embedding, 3 Mamba conv1d kernel, 4 anything else. */
static policy class_policy(const pack_options *o, const ternary_source_tensor *t, int *klass) {
    const int default_skip = contains(t->name, "embd") || contains(t->name, "embed") || contains(t->name, "vision") ||
                             contains(t->name, "audio");
    if (ternary_tensor_is_convertible(t, NULL, 0) && !default_skip) {
        *klass = 1;
        return POLICY_TERNARY;
    }
    if ((contains(t->name, "embed") || contains(t->name, "embd")) && !contains(t->name, "vision") &&
        !contains(t->name, "audio") && t->dimension_count == 2 && t->type != TERNARY_SOURCE_WEIGHT_OTHER) {
        *klass = 2;
        return o->embed;
    }
    if (ends_with(t->name, "conv1d.weight") && t->type != TERNARY_SOURCE_WEIGHT_OTHER && t->dimension_count >= 2) {
        *klass = 3;
        return o->conv1d;
    }
    *klass = 4;
    return o->extras;
}

static const char *policy_name(policy p) {
    switch (p) {
        case POLICY_SKIP: return "skip";
        case POLICY_RAW: return "raw";
        case POLICY_Q8: return "q8";
        case POLICY_TERNARY: return "ternary";
    }
    return "?";
}

/* Measure-only: the ternary error of a tensor kept in another form. */
static void measure_as_ternary(FILE *mf, const ternary_source_file *src, const ternary_source_tensor *t,
                               const pack_options *o, uint64_t chunk_weights, rstats *into) {
    const uint64_t weights = t->rows * t->columns, gpr = ternary_groups_per_row(t->columns);
    float *w, *sc;
    int8_t *codes;
    collect_ctx cc;
    if (!gpr || t->rows > TERNARY_HARD_MAX_ROWS) return;
    w = (float *)malloc((size_t)weights * sizeof(float));
    sc = (float *)malloc((size_t)(t->rows * gpr) * sizeof(float));
    codes = (int8_t *)malloc((size_t)weights);
    cc.all = w;
    cc.cols = t->columns;
    if (w && sc && codes && for_each_rows(mf, src, t, chunk_weights, collect_rows, &cc) &&
        ternary_fit_rows_padded(w, t->rows, t->columns, o->threads, codes, sc) == TERNARY_CONVERT_OK)
        stats_add_rows_parallel(into, w, codes, sc, t->rows, t->columns, o->threads);
    free(w);
    free(sc);
    free(codes);
}

/* One tensor converted exactly as the PTN1 converter does it, then stored with compact scales. */
typedef struct ternary_out {
    uint8_t *scales_sec;
    uint64_t scales_len;
    uint8_t *data;
    size_t data_len;
    int scale_encoding;
    double rel_f32, rel_stored, rel_f16, rel_u8, cos_stored;
    const char *parity;
} ternary_out;

static int pack_ternary_tensor(FILE *mf, const ternary_source_file *src, const ternary_source_tensor *t,
                               const pack_options *o, const ternary_convert_options *copt, const ref_dir *ref,
                               int have_ref, totals *T, ternary_out *out) {
    static const int enc[3] = {TERNARY_SCALES_F32, TERNARY_SCALES_F16, TERNARY_SCALES_U8ROW};
    ternary_convert_result res;
    const uint64_t weights = t->rows * t->columns;
    const uint64_t gpr = t->columns / TERNARY_GROUP_WEIGHT_COUNT, groups = t->rows * gpr;
    int8_t *codes = NULL;
    float *dec[3] = {NULL, NULL, NULL};
    uint8_t *sec[3] = {NULL, NULL, NULL};
    char error[TERNARY_CONVERT_ERROR_CAPACITY];
    uint64_t r;
    int ok = 1, chosen = 0;
    memset(&res, 0, sizeof(res));
    out->parity = "-";
    if (!ternary_convert_tensor(mf, src, t, copt, &res, error, sizeof(error))) {
        fprintf(stderr, "error: %s: %s\n", t->name, error);
        return 0;
    }
    codes = (int8_t *)malloc((size_t)weights);
    for (r = 0; r < 3; ++r) dec[r] = (float *)malloc((size_t)groups * sizeof(float));
    if (!codes || !dec[0] || !dec[1] || !dec[2]) ok = 0;
    for (r = 0; ok && r < t->rows; ++r)
        ok = ternary_unpack_row_i2(res.output.packed_codes + r * (t->columns / 4), t->columns, codes + r * t->columns) ==
             TERNARY_CONVERT_OK;
    if (ok && have_ref) {
        const ref_entry *e = ref_find(ref, t->name, strlen(t->name));
        uint8_t *rb = NULL;
        ternary_ptn1_view rv;
        T->parity_checked++;
        if (!e || !ref_load(ref, e, &rb, &rv)) {
            T->parity_missing++;
            out->parity = "missing";
        } else {
            const int codes_same = rv.rows == t->rows && rv.columns == t->columns &&
                                   rv.rows * rv.bytes_per_row == res.output.packed_codes_bytes &&
                                   memcmp(rv.packed_codes, res.output.packed_codes, res.output.packed_codes_bytes) == 0;
            int scales_same = rv.scale_count == res.output.scale_count;
            uint64_t k;
            for (k = 0; scales_same && k < rv.scale_count; ++k) {
                uint32_t bits;
                memcpy(&bits, &res.output.scales[k], 4);
                scales_same = le32(rv.scales_raw + 4 * k) == bits;
            }
            if (!codes_same) T->parity_code_mismatch++;
            if (!scales_same) T->parity_scale_mismatch++;
            out->parity = codes_same && scales_same ? "same" : (codes_same ? "scales-differ" : "codes-differ");
            free(rb);
        }
    }
    for (r = 0; r < 3 && ok; ++r) {
        const uint64_t n = ternary_scale_section_bytes((ternary_scale_encoding)enc[r], t->rows, gpr);
        sec[r] = (uint8_t *)malloc((size_t)n);
        ok = sec[r] && ternary_encode_scales((ternary_scale_encoding)enc[r], res.output.scales, t->rows, gpr, sec[r],
                                             dec[r]) == TERNARY_CONVERT_OK;
    }
    if (ok) {
        variant_ctx vc;
        memset(&vc, 0, sizeof(vc));
        vc.codes = codes;
        vc.cols = t->columns;
        vc.threads = o->threads;
        for (r = 0; r < 3; ++r) vc.scales[r] = dec[r];
        ok = for_each_rows(mf, src, t, copt->chunk_weights, variant_rows, &vc);
        if (ok) {
            const double base = rel_of(&vc.st[0]);
            const double limit = base * (1.0 + o->tolerance) + 1e-12;
            out->rel_f32 = base;
            out->rel_f16 = rel_of(&vc.st[1]);
            out->rel_u8 = rel_of(&vc.st[2]);
            if (o->scales == 0) chosen = out->rel_u8 <= limit ? 2 : (out->rel_f16 <= limit ? 1 : 0);
            else chosen = o->scales == TERNARY_SCALES_U8ROW ? 2 : (o->scales == TERNARY_SCALES_F16 ? 1 : 0);
            out->rel_stored = rel_of(&vc.st[chosen]);
            out->cos_stored = cos_of(&vc.st[chosen]);
            ternary_reconstruction_error_add(&T->base, &vc.st[0]);
            ternary_reconstruction_error_add(&T->f16, &vc.st[1]);
            ternary_reconstruction_error_add(&T->u8, &vc.st[2]);
            ternary_reconstruction_error_add(&T->chosen, &vc.st[chosen]);
        }
    }
    if (ok) {
        out->scale_encoding = enc[chosen];
        out->scales_len = ternary_scale_section_bytes((ternary_scale_encoding)enc[chosen], t->rows, gpr);
        out->scales_sec = sec[chosen];
        sec[chosen] = NULL;
        ok = encode_trits(o->trits, codes, t->rows, t->columns, &out->data, &out->data_len);
    }
    if (ok) T->scales_used[chosen + 1]++;
    free(codes);
    for (r = 0; r < 3; ++r) {
        free(dec[r]);
        free(sec[r]);
    }
    ternary_convert_result_clear(&res);
    return ok;
}

static int run_pack(const pack_options *o) {
    ternary_source_file src;
    ternary_convert_options copt;
    ref_dir ref;
    pack_writer pw;
    totals T;
    strbuf tsv = {0, 0, 0}, summary = {0, 0, 0};
    FILE *mf = NULL;
    char error[TERNARY_CONVERT_ERROR_CAPACITY], file_sha[65] = "";
    size_t i;
    int ok = 1, have_ref = 0;
    const double t0 = now_ms();
    memset(&T, 0, sizeof(T));
    memset(&src, 0, sizeof(src));
    memset(&ref, 0, sizeof(ref));
    memset(&pw, 0, sizeof(pw));
    error[0] = 0;
    {
        char model_full[4096], model_dir[4096], out_full[4096], ref_full[4096];
        if (!full_path(o->model, model_full, sizeof(model_full)) || !full_path(o->out, out_full, sizeof(out_full))) {
            fprintf(stderr, "error: cannot resolve paths\n");
            return 1;
        }
        parent_of(model_full, model_dir, sizeof(model_dir));
        if (inside(out_full, model_dir)) {
            fprintf(stderr, "error: the output may not be inside the model's folder\n");
            return 1;
        }
        if (o->reference) {
            if (!full_path(o->reference, ref_full, sizeof(ref_full)) || inside(out_full, ref_full)) {
                fprintf(stderr, "error: the output may not be inside the reference folder\n");
                return 1;
            }
        }
    }
    if (o->reference) {
        if (!ref_dir_load(o->reference, &ref)) {
            fprintf(stderr, "error: cannot read %s/manifest.tsv\n", o->reference);
            return 1;
        }
        have_ref = 1;
    }
    if (!ternary_open_source(o->model, &src, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        ref_dir_free(&ref);
        return 1;
    }
    mf = ternary_open_file_read_only(o->model);
    if (!mf || !pack_open(&pw, o->out)) {
        if (mf) fclose(mf);
        pack_abandon(&pw);
        ternary_source_file_clear(&src);
        ref_dir_free(&ref);
        return 1;
    }
    ternary_convert_options_init(&copt);
    copt.fit_kind = TERNARY_GROUP_FIT_LEAST_SQUARES;
    copt.backend = TERNARY_FIT_BACKEND_CPU;
    copt.thread_count = o->threads;
    if (o->chunk_weights) copt.chunk_weights = o->chunk_weights;

    sb_printf(&tsv, "index\tname\tclass\tstored\trows\tcols\ttrits\tscales\tpayload_bytes\trel_f32\trel_stored\trel_f16\t"
                    "rel_u8row\tcos_stored\tparity\tsha256\n");
    for (i = 0; i < src.tensor_count && ok; ++i) {
        const ternary_source_tensor *t = &src.tensors[i];
        int klass = 4;
        const policy pol = class_policy(o, t, &klass);
        const uint64_t weights = t->rows * t->columns;
        ternary_ptt1_desc d;
        ternary_out to;
        uint8_t *payload = NULL;
        size_t payload_len = 0;
        char sha[65] = "-";
        memset(&to, 0, sizeof(to));
        to.rel_f32 = to.rel_stored = to.rel_f16 = to.rel_u8 = to.cos_stored = -1.0;
        to.parity = "-";
        desc_from_source(t, &d);
        error[0] = 0;

        if (pol == POLICY_SKIP) {
            T.tensors[4]++;
            T.weights[4] += weights;
            sb_printf(&tsv, "%zu\t%s\t%d\tskipped\t%llu\t%llu\t-\t-\t0\t-\t-\t-\t-\t-\t-\t-\n", i, t->name, klass,
                      (unsigned long long)t->rows, (unsigned long long)t->columns);
            if (klass == 2 || klass == 3)
                measure_as_ternary(mf, &src, t, o, copt.chunk_weights, klass == 2 ? &T.ext_embed : &T.ext_conv1d);
            continue;
        }
        if (pol == POLICY_TERNARY && klass == 1) {
            ok = pack_ternary_tensor(mf, &src, t, o, &copt, &ref, have_ref, &T, &to);
            d.kind = TERNARY_KIND_TERNARY;
            d.trit_encoding = o->trits;
            d.scale_encoding = to.scale_encoding;
        } else if (pol == POLICY_TERNARY) {
            /* embedding or conv1d through the padded path, float32 scales */
            const uint64_t gpr = ternary_groups_per_row(t->columns), groups = t->rows * gpr;
            float *w = (float *)malloc((size_t)weights * sizeof(float));
            float *sc = (float *)malloc((size_t)groups * sizeof(float));
            int8_t *codes = (int8_t *)malloc((size_t)weights);
            collect_ctx cc;
            rstats st;
            memset(&st, 0, sizeof(st));
            cc.all = w;
            cc.cols = t->columns;
            ok = w && sc && codes && gpr && t->rows <= TERNARY_HARD_MAX_ROWS &&
                 for_each_rows(mf, &src, t, copt.chunk_weights, collect_rows, &cc) &&
                 ternary_fit_rows_padded(w, t->rows, t->columns, o->threads, codes, sc) == TERNARY_CONVERT_OK &&
                 stats_add_rows_parallel(&st, w, codes, sc, t->rows, t->columns, o->threads);
            if (ok) {
                to.rel_f32 = to.rel_stored = rel_of(&st);
                to.cos_stored = cos_of(&st);
                ternary_reconstruction_error_add(klass == 2 ? &T.ext_embed : &T.ext_conv1d, &st);
                d.kind = TERNARY_KIND_TERNARY;
                d.trit_encoding = o->trits;
                d.scale_encoding = TERNARY_SCALES_F32;
                to.scales_len = ternary_scale_section_bytes(TERNARY_SCALES_F32, t->rows, gpr);
                to.scales_sec = (uint8_t *)malloc((size_t)to.scales_len);
                ok = to.scales_sec &&
                     ternary_encode_scales(TERNARY_SCALES_F32, sc, t->rows, gpr, to.scales_sec, NULL) == TERNARY_CONVERT_OK &&
                     encode_trits(o->trits, codes, t->rows, t->columns, &to.data, &to.data_len);
                if (ok) T.scales_used[1]++;
            }
            free(w);
            free(sc);
            free(codes);
        } else if (pol == POLICY_Q8) {
            const uint64_t gpr = ternary_groups_per_row(t->columns);
            q8_ctx qc;
            memset(&qc, 0, sizeof(qc));
            qc.cols = t->columns;
            qc.threads = o->threads;
            qc.q = (int8_t *)malloc((size_t)weights);
            qc.scales = (float *)malloc((size_t)(t->rows * gpr) * sizeof(float));
            ok = qc.q && qc.scales && gpr && for_each_rows(mf, &src, t, copt.chunk_weights, q8_rows, &qc);
            if (ok) {
                to.rel_f32 = to.rel_stored = rel_of(&qc.st);
                to.cos_stored = cos_of(&qc.st);
                ternary_reconstruction_error_add(&T.q8, &qc.st);
                d.kind = TERNARY_KIND_Q8;
                d.scale_encoding = TERNARY_SCALES_F32;
                to.scales_len = ternary_scale_section_bytes(TERNARY_SCALES_F32, t->rows, gpr);
                to.scales_sec = (uint8_t *)malloc((size_t)to.scales_len);
                ok = to.scales_sec && ternary_encode_scales(TERNARY_SCALES_F32, qc.scales, t->rows, gpr, to.scales_sec,
                                                            NULL) == TERNARY_CONVERT_OK;
                to.data = (uint8_t *)qc.q;
                qc.q = NULL;
                to.data_len = (size_t)weights;
            }
            free(qc.q);
            free(qc.scales);
        } else { /* raw: the source bytes unchanged */
            d.kind = TERNARY_KIND_RAW;
            d.raw_dtype = source_raw_dtype(t);
            if (!d.raw_dtype) {
                fprintf(stderr, "error: %s: type %s can't be stored raw\n", t->name, t->type_name);
                ok = 0;
            } else {
                to.data_len = (size_t)t->byte_count;
                to.data = (uint8_t *)malloc(to.data_len ? to.data_len : 1);
                ok = to.data && read_at(mf, t->file_offset, to.data, t->byte_count);
            }
        }
        if (ok)
            ok = ternary_build_ptt1(&d, to.scales_sec, to.scales_len, to.data, to.data_len, &payload, &payload_len, error,
                                    sizeof(error));
        if (!ok && error[0]) fprintf(stderr, "error: %s: %s\n", t->name, error);
        if (ok) ok = pack_add(&pw, t->name, payload, payload_len, sha);
        if (ok) {
            const int kind = (int)d.kind;
            T.tensors[kind]++;
            T.weights[kind] += weights;
            T.bytes[kind] += payload_len;
            if (kind == TERNARY_KIND_TERNARY) {
                T.trit_bytes += to.data_len;
                T.scale_bytes += to.scales_len;
            }
            sb_printf(&tsv, "%zu\t%s\t%d\t%s\t%llu\t%llu\t%s\t%s\t%zu\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\t%s\n", i, t->name,
                      klass, kind == TERNARY_KIND_TERNARY ? "ternary" : (kind == TERNARY_KIND_Q8 ? "q8" : "raw"),
                      (unsigned long long)t->rows, (unsigned long long)t->columns,
                      ternary_trit_encoding_name(d.trit_encoding), ternary_scale_encoding_name(d.scale_encoding),
                      payload_len, to.rel_f32, to.rel_stored, to.rel_f16, to.rel_u8, to.cos_stored, to.parity, sha);
            if (pol != POLICY_TERNARY && (klass == 2 || klass == 3))
                measure_as_ternary(mf, &src, t, o, copt.chunk_weights, klass == 2 ? &T.ext_embed : &T.ext_conv1d);
        }
        free(payload);
        free(to.scales_sec);
        free(to.data);
    }
    fclose(mf);
    if (ok) ok = pack_close(&pw, file_sha);
    else pack_abandon(&pw);
    if (ok) {
        char tsv_path[4200];
        FILE *f;
        const uint64_t tw = T.weights[1];
        sb_printf(&summary, "Pack summary\nsource: %s (opened read-only)\nout: %s\nsha256: %s\n", o->model, o->out, file_sha);
        sb_printf(&summary, "options: scales %s, tolerance %g, trits %s, embed %s, conv1d %s, extras %s, threads %d\n",
                  o->scales ? ternary_scale_encoding_name(o->scales) : "tight", o->tolerance,
                  ternary_trit_encoding_name(o->trits), policy_name(o->embed), policy_name(o->conv1d),
                  policy_name(o->extras), o->threads);
        sb_printf(&summary, "ternary: tensors %llu weights %llu payload %llu trits %llu scales %llu bits/weight %.4f "
                            "(trits %.4f, scales %.4f)\n",
                  (unsigned long long)T.tensors[1], (unsigned long long)tw, (unsigned long long)T.bytes[1],
                  (unsigned long long)T.trit_bytes, (unsigned long long)T.scale_bytes,
                  tw ? 8.0 * (double)T.bytes[1] / (double)tw : 0.0, tw ? 8.0 * (double)T.trit_bytes / (double)tw : 0.0,
                  tw ? 8.0 * (double)T.scale_bytes / (double)tw : 0.0);
        sb_printf(&summary, "scales used: f32 %llu f16 %llu u8row %llu\n", (unsigned long long)T.scales_used[1],
                  (unsigned long long)T.scales_used[2], (unsigned long long)T.scales_used[3]);
        sb_printf(&summary, "q8: tensors %llu weights %llu payload %llu\n", (unsigned long long)T.tensors[3],
                  (unsigned long long)T.weights[3], (unsigned long long)T.bytes[3]);
        sb_printf(&summary, "raw: tensors %llu weights %llu payload %llu\n", (unsigned long long)T.tensors[2],
                  (unsigned long long)T.weights[2], (unsigned long long)T.bytes[2]);
        sb_printf(&summary, "skipped: tensors %llu weights %llu\n", (unsigned long long)T.tensors[4],
                  (unsigned long long)T.weights[4]);
#define ERRLINE(label, s)                                                                                             \
    sb_printf(&summary, "error\t%s\t%llu\t%.6f\t%.6f\t%.17g\t%.17g\n", label, (unsigned long long)(s).weight_count,     \
              rel_of(&(s)), cos_of(&(s)), (s).sum_weight_squared, (s).sum_error_squared)
        sb_printf(&summary, "error\tset\tweights\trel_error\tcosine\tsum_w2\tsum_e2\n");
        ERRLINE("ternary_f32", T.base);
        ERRLINE("ternary_stored", T.chosen);
        ERRLINE("ternary_all_f16", T.f16);
        ERRLINE("ternary_all_u8row", T.u8);
        ERRLINE("q8_stored", T.q8);
        ERRLINE("embedding_as_ternary", T.ext_embed);
        ERRLINE("conv1d_as_ternary", T.ext_conv1d);
#undef ERRLINE
        if (have_ref)
            sb_printf(&summary, "parity vs PTN1: checked %llu, codes differ %llu, scales differ %llu, missing %llu\n",
                      (unsigned long long)T.parity_checked, (unsigned long long)T.parity_code_mismatch,
                      (unsigned long long)T.parity_scale_mismatch, (unsigned long long)T.parity_missing);
        sb_printf(&summary, "time: %.1f s\n", (now_ms() - t0) / 1000.0);
        snprintf(tsv_path, sizeof(tsv_path), "%s.tsv", o->out);
        f = create_new(tsv_path);
        if (f) {
            if (tsv.p) fwrite(tsv.p, 1, tsv.len, f);
            if (summary.p) fwrite(summary.p, 1, summary.len, f);
            fclose(f);
        } else {
            fprintf(stderr, "warning: cannot create %s\n", tsv_path);
        }
        if (summary.p) fputs(summary.p, stdout);
        if (have_ref && (T.parity_code_mismatch || T.parity_scale_mismatch || T.parity_missing)) {
            fprintf(stderr, "error: the codes or scales differ from the PTN1 conversion\n");
            ok = 0;
        }
    }
    free(tsv.p);
    free(summary.p);
    ternary_source_file_clear(&src);
    ref_dir_free(&ref);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------------------------------------ from-ptn */

static int run_from_ptn(const char *dir, const char *out, int trits) {
    ref_dir ref;
    pack_writer pw;
    size_t i;
    int ok = 1;
    uint64_t weights = 0, trit_bytes = 0, scale_bytes = 0, payload_bytes = 0;
    char file_sha[65] = "";
    {
        char out_full[4096], dir_full[4096];
        if (!full_path(out, out_full, sizeof(out_full)) || !full_path(dir, dir_full, sizeof(dir_full)) ||
            inside(out_full, dir_full)) {
            fprintf(stderr, "error: the output may not be inside the reference folder\n");
            return 1;
        }
    }
    if (!ref_dir_load(dir, &ref)) {
        fprintf(stderr, "error: cannot read %s/manifest.tsv\n", dir);
        return 1;
    }
    if (!pack_open(&pw, out)) {
        pack_abandon(&pw);
        ref_dir_free(&ref);
        return 1;
    }
    for (i = 0; i < ref.count && ok; ++i) {
        uint8_t *rb = NULL, *data = NULL, *payload = NULL;
        size_t data_len = 0, payload_len = 0;
        ternary_ptn1_view v;
        ternary_ptt1_desc d;
        int8_t *codes = NULL;
        uint64_t r;
        char error[TERNARY_CONVERT_ERROR_CAPACITY];
        error[0] = 0;
        if (!ref_load(&ref, &ref.items[i], &rb, &v)) {
            fprintf(stderr, "error: cannot load %s\n", ref.items[i].file);
            ok = 0;
            break;
        }
        codes = (int8_t *)malloc((size_t)(v.rows * v.columns));
        ok = codes != NULL;
        for (r = 0; ok && r < v.rows; ++r)
            ok = ternary_unpack_row_i2(v.packed_codes + r * v.bytes_per_row, v.columns, codes + r * v.columns) ==
                 TERNARY_CONVERT_OK;
        if (ok) ok = encode_trits(trits, codes, v.rows, v.columns, &data, &data_len);
        memset(&d, 0, sizeof(d));
        d.kind = TERNARY_KIND_TERNARY;
        d.trit_encoding = trits;
        d.scale_encoding = TERNARY_SCALES_F32;
        d.dims[0] = v.rows;
        d.dims[1] = v.columns;
        d.dimension_count = 2;
        d.rows = v.rows;
        d.columns = v.columns;
        if (ok)
            ok = ternary_build_ptt1(&d, v.scales_raw, v.scale_count * 4, data, data_len, &payload, &payload_len, error,
                                    sizeof(error));
        if (!ok && error[0]) fprintf(stderr, "error: %s: %s\n", ref.items[i].name, error);
        if (ok) ok = pack_add(&pw, ref.items[i].name, payload, payload_len, NULL);
        if (ok) {
            weights += v.rows * v.columns;
            trit_bytes += data_len;
            scale_bytes += v.scale_count * 4;
            payload_bytes += payload_len;
        }
        free(codes);
        free(data);
        free(payload);
        free(rb);
    }
    if (ok) ok = pack_close(&pw, file_sha);
    else pack_abandon(&pw);
    if (ok)
        printf("from-ptn: %zu tensors, %llu weights, payload %llu (trits %llu, scales %llu), %.4f bits/weight, "
               "sha256 %s\n",
               ref.count, (unsigned long long)weights, (unsigned long long)payload_bytes, (unsigned long long)trit_bytes,
               (unsigned long long)scale_bytes, weights ? 8.0 * (double)payload_bytes / (double)weights : 0.0, file_sha);
    ref_dir_free(&ref);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------------------------------------ recode */

static int run_recode(const char *in, const char *out, int trits) {
    pack_reader pr;
    pack_writer pw;
    uint32_t i;
    int ok = 1;
    uint64_t trit_in = 0, trit_out = 0, weights = 0;
    char file_sha[65] = "";
    {
        char a[4096], b[4096];
        if (!full_path(in, a, sizeof(a)) || !full_path(out, b, sizeof(b)) || strcmp(a, b) == 0) {
            fprintf(stderr, "error: the output must be a different file\n");
            return 1;
        }
    }
    if (!pack_reader_open(&pr, in)) {
        fprintf(stderr, "error: cannot open %s\n", in);
        return 1;
    }
    if (!pack_open(&pw, out)) {
        pack_abandon(&pw);
        pack_reader_close(&pr);
        return 1;
    }
    for (i = 0; i < pr.view.tensor_count && ok; ++i) {
        const ternary_ptp1_entry *e = &pr.view.entries[i];
        int hash_ok;
        uint8_t *buf = pack_payload(&pr, i, &hash_ok), *payload = NULL, *data = NULL;
        size_t payload_len = 0, data_len = 0;
        ternary_ptt1_view v;
        char name[TERNARY_PACK_MAX_NAME_BYTES + 1], error[TERNARY_CONVERT_ERROR_CAPACITY];
        entry_name(e, name);
        error[0] = 0;
        if (!buf || !hash_ok || ternary_parse_ptt1(buf, (size_t)e->length, e->length, &v) != TERNARY_CONVERT_OK) {
            fprintf(stderr, "error: %s: payload damaged\n", name);
            free(buf);
            ok = 0;
            break;
        }
        if (v.desc.kind == TERNARY_KIND_TERNARY && v.desc.trit_encoding != trits) {
            int8_t *codes = (int8_t *)malloc((size_t)(v.desc.rows * v.desc.columns));
            ternary_ptt1_desc d = v.desc;
            ok = codes && ternary_ptt1_decode_ternary(&v, codes, NULL) == TERNARY_CONVERT_OK &&
                 encode_trits(trits, codes, d.rows, d.columns, &data, &data_len);
            d.trit_encoding = trits;
            if (ok)
                ok = ternary_build_ptt1(&d, v.scales, v.scales_bytes, data, data_len, &payload, &payload_len, error,
                                        sizeof(error));
            if (ok) {
                trit_in += v.data_bytes;
                trit_out += data_len;
                weights += d.rows * d.columns;
                ok = pack_add(&pw, name, payload, payload_len, NULL);
            }
            if (!ok) fprintf(stderr, "error: %s: cannot re-encode %s\n", name, error);
            free(codes);
        } else {
            if (v.desc.kind == TERNARY_KIND_TERNARY) {
                trit_in += v.data_bytes;
                trit_out += v.data_bytes;
                weights += v.desc.rows * v.desc.columns;
            }
            ok = pack_add(&pw, name, buf, (size_t)e->length, NULL);
        }
        free(buf);
        free(payload);
        free(data);
    }
    pack_reader_close(&pr);
    if (ok) ok = pack_close(&pw, file_sha);
    else pack_abandon(&pw);
    if (ok)
        printf("recode: trits %llu -> %llu bytes (%.4f -> %.4f bits/weight over %llu ternary weights), sha256 %s\n",
               (unsigned long long)trit_in, (unsigned long long)trit_out,
               weights ? 8.0 * (double)trit_in / (double)weights : 0.0,
               weights ? 8.0 * (double)trit_out / (double)weights : 0.0, (unsigned long long)weights, file_sha);
    return ok ? 0 : 1;
}

/* ------------------------------------------------------------------------------------------------ verify */

static int run_verify(const char *path, const char *reference) {
    pack_reader pr;
    ref_dir ref;
    uint32_t i;
    uint64_t bad_hash = 0, bad_parse = 0, bad_decode = 0, checked = 0, code_diff = 0, scale_diff = 0, missing = 0,
             scale_lossy = 0, by_kind[4] = {0, 0, 0, 0};
    double max_scale_rel = 0.0;
    int have_ref = 0;
    char file_sha[65];
    if (!sha256_file(path, file_sha)) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return 1;
    }
    if (!pack_reader_open(&pr, path)) {
        fprintf(stderr, "error: %s is not a valid pack\n", path);
        return 1;
    }
    memset(&ref, 0, sizeof(ref));
    if (reference) {
        if (!ref_dir_load(reference, &ref)) {
            fprintf(stderr, "error: cannot read %s/manifest.tsv\n", reference);
            pack_reader_close(&pr);
            return 1;
        }
        have_ref = 1;
    }
    for (i = 0; i < pr.view.tensor_count; ++i) {
        const ternary_ptp1_entry *e = &pr.view.entries[i];
        int hash_ok;
        uint8_t *buf = pack_payload(&pr, i, &hash_ok);
        ternary_ptt1_view v;
        if (!buf || !hash_ok) {
            bad_hash++;
            free(buf);
            continue;
        }
        if (ternary_parse_ptt1(buf, (size_t)e->length, e->length, &v) != TERNARY_CONVERT_OK) {
            bad_parse++;
            free(buf);
            continue;
        }
        by_kind[v.desc.kind]++;
        if (v.desc.kind == TERNARY_KIND_TERNARY) {
            const uint64_t n = v.desc.rows * v.desc.columns, gpr = ternary_groups_per_row(v.desc.columns);
            int8_t *codes = (int8_t *)malloc((size_t)n);
            float *sc = (float *)malloc((size_t)(v.desc.rows * gpr) * sizeof(float));
            const ref_entry *re = have_ref ? ref_find(&ref, e->name, e->name_length) : NULL;
            if (!codes || !sc || ternary_ptt1_decode_ternary(&v, codes, sc) != TERNARY_CONVERT_OK) {
                bad_decode++;
            } else if (re) {
                uint8_t *rb = NULL;
                ternary_ptn1_view rv;
                if (!ref_load(&ref, re, &rb, &rv)) {
                    missing++;
                } else if (rv.rows != v.desc.rows || rv.columns != v.desc.columns) {
                    checked++;
                    code_diff++;
                } else {
                    const uint64_t rbytes = ternary_i2_row_bytes(rv.columns);
                    uint8_t *i2 = (uint8_t *)malloc((size_t)rbytes);
                    uint64_t r, k;
                    int same = i2 != NULL;
                    checked++;
                    for (r = 0; same && r < rv.rows; ++r)
                        same = ternary_pack_row_i2(codes + r * rv.columns, rv.columns, i2) == TERNARY_CONVERT_OK &&
                               memcmp(i2, rv.packed_codes + r * rv.bytes_per_row, (size_t)rbytes) == 0;
                    if (!same) code_diff++;
                    free(i2);
                    if (v.desc.scale_encoding != TERNARY_SCALES_F32) scale_lossy++;
                    for (k = 0; k < rv.scale_count; ++k) {
                        float ref_s;
                        const uint32_t bits = le32(rv.scales_raw + 4 * k);
                        memcpy(&ref_s, &bits, 4);
                        if (v.desc.scale_encoding == TERNARY_SCALES_F32) {
                            if (memcmp(&ref_s, &sc[k], 4) != 0) {
                                scale_diff++;
                                break;
                            }
                        } else if (ref_s > 0.0f) {
                            const double rel = fabs((double)sc[k] - (double)ref_s) / (double)ref_s;
                            if (rel > max_scale_rel) max_scale_rel = rel;
                        }
                    }
                }
                free(rb);
            } else if (have_ref) {
                missing++; /* a ternary tensor the PTN1 conversion did not have (e.g. conv1d kept ternary) */
            }
            free(codes);
            free(sc);
        } else if (v.desc.kind == TERNARY_KIND_Q8) {
            const uint64_t gpr = ternary_groups_per_row(v.desc.columns);
            float *sc = (float *)malloc((size_t)(v.desc.rows * gpr) * sizeof(float));
            if (!sc || ternary_decode_scales(TERNARY_SCALES_F32, v.scales, v.scales_bytes, v.desc.rows, gpr, sc) !=
                           TERNARY_CONVERT_OK) {
                bad_decode++;
            } else {
                uint64_t k;
                for (k = 0; k < v.data_bytes; ++k)
                    if ((int8_t)v.data[k] == -128) {
                        bad_decode++;
                        break;
                    }
            }
            free(sc);
        }
        free(buf);
    }
    printf("verify %s: sha256 %s, %u tensors (ternary %llu, raw %llu, q8 %llu); hash failures %llu, parse failures "
           "%llu, decode failures %llu\n",
           path, file_sha, pr.view.tensor_count, (unsigned long long)by_kind[1], (unsigned long long)by_kind[2],
           (unsigned long long)by_kind[3], (unsigned long long)bad_hash, (unsigned long long)bad_parse,
           (unsigned long long)bad_decode);
    if (have_ref)
        printf("parity vs PTN1: %llu ternary tensors compared, I2 codes differ in %llu, float32 scales differ in %llu, "
               "not in PTN1 %llu; %llu use compact scales (largest scale change %.3g relative)\n",
               (unsigned long long)checked, (unsigned long long)code_diff, (unsigned long long)scale_diff,
               (unsigned long long)missing, (unsigned long long)scale_lossy, max_scale_rel);
    pack_reader_close(&pr);
    ref_dir_free(&ref);
    return (bad_hash || bad_parse || bad_decode || code_diff || scale_diff) ? 1 : 0;
}

/* ------------------------------------------------------------------------------------------------ bench */

static int run_bench(const char *path, uint64_t max_weights) {
    pack_reader pr;
    uint32_t i;
    uint64_t weights = 0, b3_bytes = 0, rans_bytes = 0, b3_w = 0, rans_w = 0, scale_groups = 0;
    double ms_b3 = 0, ms_b3_i2 = 0, ms_rans = 0, ms_scales = 0;
    volatile uint64_t sink = 0;
    if (!pack_reader_open(&pr, path)) {
        fprintf(stderr, "error: %s is not a valid pack\n", path);
        return 1;
    }
    for (i = 0; i < pr.view.tensor_count && weights < max_weights; ++i) {
        const ternary_ptp1_entry *e = &pr.view.entries[i];
        int hash_ok;
        uint8_t *buf = pack_payload(&pr, i, &hash_ok);
        ternary_ptt1_view v;
        if (buf && hash_ok && ternary_parse_ptt1(buf, (size_t)e->length, e->length, &v) == TERNARY_CONVERT_OK &&
            v.desc.kind == TERNARY_KIND_TERNARY) {
            const uint64_t n = v.desc.rows * v.desc.columns, gpr = ternary_groups_per_row(v.desc.columns);
            int8_t *codes = (int8_t *)malloc((size_t)n);
            float *sc = (float *)malloc((size_t)(v.desc.rows * gpr) * sizeof(float));
            double t;
            if (codes && sc) {
                t = now_ms();
                ternary_decode_scales((ternary_scale_encoding)v.desc.scale_encoding, v.scales, v.scales_bytes,
                                      v.desc.rows, gpr, sc);
                ms_scales += now_ms() - t;
                scale_groups += v.desc.rows * gpr;
                if (v.desc.trit_encoding == TERNARY_TRITS_B3) {
                    const uint64_t rb = ternary_b3_row_bytes(v.desc.columns), ib = ternary_i2_row_bytes(v.desc.columns);
                    uint64_t r;
                    t = now_ms();
                    for (r = 0; r < v.desc.rows; ++r)
                        ternary_unpack_row_b3(v.data + r * rb, v.desc.columns, codes + r * v.desc.columns);
                    ms_b3 += now_ms() - t;
                    if (ib) {
                        uint8_t *i2 = (uint8_t *)malloc((size_t)ib);
                        if (i2) {
                            t = now_ms();
                            for (r = 0; r < v.desc.rows; ++r) {
                                ternary_i2_from_b3_row(v.data + r * rb, v.desc.columns, i2);
                                sink += i2[r % ib];
                            }
                            ms_b3_i2 += now_ms() - t;
                            free(i2);
                        }
                    }
                    b3_bytes += v.data_bytes;
                    b3_w += n;
                } else {
                    t = now_ms();
                    ternary_rans_decode(v.data, v.data_bytes, v.desc.rows, v.desc.columns, codes);
                    ms_rans += now_ms() - t;
                    rans_bytes += v.data_bytes;
                    rans_w += n;
                }
                sink += (uint64_t)(uint8_t)codes[n - 1];
                weights += n;
            }
            free(codes);
            free(sc);
        }
        free(buf);
    }
    printf("bench %s (one thread): %llu ternary weights\n", path, (unsigned long long)weights);
    if (b3_w && ms_b3 > 0)
        printf("  b3 -> int8: %.1f Mweights/s (%.1f MB/s of packed input); b3 -> I2: %.1f Mweights/s\n",
               (double)b3_w / (ms_b3 * 1000.0), (double)b3_bytes / (ms_b3 * 1000.0),
               ms_b3_i2 > 0 ? (double)b3_w / (ms_b3_i2 * 1000.0) : 0.0);
    if (rans_w && ms_rans > 0)
        printf("  rans -> int8: %.1f Mweights/s (%.1f MB/s of packed input)\n", (double)rans_w / (ms_rans * 1000.0),
               (double)rans_bytes / (ms_rans * 1000.0));
    if (scale_groups && ms_scales > 0)
        printf("  scales: %.1f Mgroups/s\n", (double)scale_groups / (ms_scales * 1000.0));
    (void)sink;
    pack_reader_close(&pr);
    return 0;
}

/* ------------------------------------------------------------------------------------------------ main */

static int parse_trits(const char *s) {
    if (strcmp(s, "b3") == 0) return TERNARY_TRITS_B3;
    if (strcmp(s, "rans") == 0) return TERNARY_TRITS_RANS;
    return 0;
}

static int parse_policy(const char *s, policy *p, int allow_q8, int allow_ternary) {
    if (strcmp(s, "skip") == 0) *p = POLICY_SKIP;
    else if (strcmp(s, "raw") == 0) *p = POLICY_RAW;
    else if (allow_q8 && strcmp(s, "q8") == 0) *p = POLICY_Q8;
    else if (allow_ternary && strcmp(s, "ternary") == 0) *p = POLICY_TERNARY;
    else return 0;
    return 1;
}

int main(int argc, char **argv) {
    const char *mode, *input, *out = NULL, *reference = NULL;
    pack_options o;
    int argi, trits = TERNARY_TRITS_B3, trits_given = 0;
    uint64_t max_mweights = 512;
    if (argc < 3) return usage();
    mode = argv[1];
    input = argv[2];
    memset(&o, 0, sizeof(o));
    o.tolerance = 1e-4;
    o.embed = POLICY_Q8;
    o.conv1d = POLICY_RAW;
    o.extras = POLICY_RAW;
    for (argi = 3; argi < argc; argi += 2) {
        const char *a = argv[argi];
        const char *v = argi + 1 < argc ? argv[argi + 1] : NULL;
        if (!v) return usage();
        if (strcmp(a, "--out") == 0) out = v;
        else if (strcmp(a, "--reference") == 0) reference = v;
        else if (strcmp(a, "--trits") == 0) {
            trits = parse_trits(v);
            trits_given = 1;
            if (!trits) return usage();
        } else if (strcmp(a, "--scales") == 0) {
            if (strcmp(v, "tight") == 0) o.scales = 0;
            else if (strcmp(v, "f32") == 0) o.scales = TERNARY_SCALES_F32;
            else if (strcmp(v, "f16") == 0) o.scales = TERNARY_SCALES_F16;
            else if (strcmp(v, "u8row") == 0) o.scales = TERNARY_SCALES_U8ROW;
            else return usage();
        } else if (strcmp(a, "--tolerance") == 0) {
            char *end = NULL;
            o.tolerance = strtod(v, &end);
            if (!end || *end || !(o.tolerance >= 0.0) || o.tolerance > 1.0) return usage();
        } else if (strcmp(a, "--embed") == 0) {
            if (!parse_policy(v, &o.embed, 1, 1)) return usage();
        } else if (strcmp(a, "--conv1d") == 0) {
            if (!parse_policy(v, &o.conv1d, 0, 1)) return usage();
        } else if (strcmp(a, "--extras") == 0) {
            if (!parse_policy(v, &o.extras, 0, 0)) return usage();
        } else if (strcmp(a, "--threads") == 0) {
            o.threads = atoi(v);
            if (o.threads < 0 || o.threads > 64) return usage();
        } else if (strcmp(a, "--chunk-mweights") == 0) {
            const long m = atol(v);
            if (m < 1 || m > 1024) return usage();
            o.chunk_weights = (uint64_t)m << 20;
        } else if (strcmp(a, "--max-mweights") == 0) {
            const long m = atol(v);
            if (m < 1 || m > (1L << 20)) return usage();
            max_mweights = (uint64_t)m;
        } else return usage();
    }
    o.trits = trits;
    if (strcmp(mode, "pack") == 0) {
        if (!out) return usage();
        o.model = input;
        o.out = out;
        o.reference = reference;
        return run_pack(&o);
    }
    if (strcmp(mode, "from-ptn") == 0) return out ? run_from_ptn(input, out, trits) : usage();
    if (strcmp(mode, "recode") == 0) return (out && trits_given) ? run_recode(input, out, trits) : usage();
    if (strcmp(mode, "verify") == 0) return run_verify(input, reference);
    if (strcmp(mode, "bench") == 0) return run_bench(input, max_mweights << 20);
    return usage();
}
