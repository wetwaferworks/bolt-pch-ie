/* Bolt PCHIE - ternary converter command line (C17).
 *
 *   pchie_ternary_convert MODEL --list
 *   pchie_ternary_convert MODEL --out DIR [--fit lsq|absmean] [--backend cpu|cuda] [--check-parity]
 *                         [--match TEXT]... [--skip TEXT]... [--no-default-skips] [--limit N] [--threads N]
 *                         [--chunk-mweights N]
 *
 * MODEL is a GGUF or safetensors file. It is opened read-only. DIR must be new or empty and may not sit inside the
 * model's folder; nothing in it is ever overwritten. For each converted matrix the tool writes NNNN_name.ptn (a
 * PTN1 ternary tensor, I2 packing, one scale per 128 weights), and at the end manifest.tsv (one line per tensor:
 * shape, source type, error statistics, timings, SHA-256 of the .ptn), layers.tsv (error per layer) and
 * summary.txt. By default tensors whose names contain embd, embed, vision or audio are skipped.
 *
 * The error statistics compare each weight with gamma * q: relative error ||W - R|| / ||W||, cosine, RMSE,
 * largest absolute error and the share of zero codes. They say how close the ternary matrix is to the source
 * matrix; they say nothing about model quality.
 */
#include "pchie/pchie_ternary_convert.h"
#include "pchie_ternary_convert_internal.h"
#include "pj_sha256.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#if defined(_WIN32)
#include <windows.h>
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#endif

#if defined(_WIN32)
#include <windows.h>
static double tool_now_ms(void) {
    static LARGE_INTEGER frequency = {0};
    LARGE_INTEGER now;
    if (frequency.QuadPart == 0) QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart * 1000.0 / (double)frequency.QuadPart;
}
#else
#include <time.h>
static double tool_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}
#endif

static int usage(void) {
    fprintf(stderr,
            "usage: pchie_ternary_convert MODEL --list\n"
            "       pchie_ternary_convert MODEL --out DIR [--fit lsq|absmean] [--backend cpu|cuda] [--check-parity]\n"
            "                             [--match TEXT]... [--skip TEXT]... [--no-default-skips] [--limit N]\n"
            "                             [--threads N] [--chunk-mweights N]\n");
    return 2;
}

static void sha256_hex(const uint8_t *data, size_t length, char out[65]) {
    uint8_t digest[32];
    static const char *hex = "0123456789abcdef";
    int i;
    pj_sha256_once(data, length, digest);
    for (i = 0; i < 32; ++i) {
        out[2 * i] = hex[digest[i] >> 4];
        out[2 * i + 1] = hex[digest[i] & 15];
    }
    out[64] = 0;
}

static void safe_name(const char *name, char *out, size_t out_capacity) {
    size_t n = 0;
    if (!out_capacity) return;
    for (; name && *name && n + 1 < out_capacity && n < 160; ++name) {
        unsigned char c = (unsigned char)*name;
        out[n++] = (isalnum(c) || c == '.' || c == '-' || c == '_') ? (char)c : '_';
    }
    out[n] = 0;
}

static int copy_string(char *destination, size_t destination_capacity, const char *source) {
    size_t n;
    if (!destination || !destination_capacity) return 0;
    if (!source) source = "";
    n = strlen(source);
    if (n >= destination_capacity) return 0;
    memcpy(destination, source, n + 1);
    return 1;
}

/* "blk.12.ffn_up.weight" or "model.language_model.layers.12.mlp.up_proj.weight" -> "12"; anything else -> "other". */
static void layer_of(const char *name, char *out, size_t out_capacity) {
    const char *keys[] = {"blk.", "layers.", NULL};
    int k;
    if (!out_capacity) return;
    if (!copy_string(out, out_capacity, "other")) return;
    if (!name) return;
    for (k = 0; keys[k]; ++k) {
        const char *p = name;
        const size_t key_len = strlen(keys[k]);
        while ((p = strstr(p, keys[k])) != NULL) {
            const char *i = p + key_len, *j = i;
            while (*j && isdigit((unsigned char)*j)) ++j;
            if (j > i && *j == '.') {
                size_t n = (size_t)(j - i);
                if (n >= out_capacity) n = out_capacity - 1;
                memcpy(out, i, n);
                out[n] = 0;
                return;
            }
            p += 1;
        }
    }
}

static int write_new_file(const char *path, const uint8_t *bytes, size_t length, char *error, size_t error_capacity) {
    char tmp[4096];
    FILE *file;
    int ok, closed;
#if defined(_WIN32)
    if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) {
        ternary_set_errorf(error, error_capacity, "refusing to overwrite %s", path);
        return 0;
    }
#else
    if (access(path, F_OK) == 0) {
        ternary_set_errorf(error, error_capacity, "refusing to overwrite %s", path);
        return 0;
    }
#endif
    if (strlen(path) + 9 >= sizeof(tmp)) {
        ternary_set_error(error, error_capacity, "path too long");
        return 0;
    }
    snprintf(tmp, sizeof(tmp), "%s.partial", path);
#if defined(_WIN32)
    if (GetFileAttributesA(tmp) != INVALID_FILE_ATTRIBUTES) {
        ternary_set_errorf(error, error_capacity, "refusing to overwrite %s", tmp);
        return 0;
    }
#else
    if (access(tmp, F_OK) == 0) {
        ternary_set_errorf(error, error_capacity, "refusing to overwrite %s", tmp);
        return 0;
    }
#endif
    file = ternary_open_file_write_new(tmp);
    if (!file) {
        ternary_set_errorf(error, error_capacity, "cannot create %s", tmp);
        return 0;
    }
    ok = fwrite(bytes, 1, length, file) == length;
    closed = fclose(file) == 0;
    if (!ok || !closed) {
        ternary_set_errorf(error, error_capacity, "write failed: %s", tmp);
        return 0;
    }
#if defined(_WIN32)
    if (!MoveFileA(tmp, path)) {
        ternary_set_errorf(error, error_capacity, "rename failed: %s", path);
        return 0;
    }
#else
    if (rename(tmp, path) != 0) {
        ternary_set_errorf(error, error_capacity, "rename failed: %s", path);
        return 0;
    }
#endif
    return 1;
}

static int write_text(const char *path, const char *text, char *error, size_t error_capacity) {
    return write_new_file(path, (const uint8_t *)text, strlen(text), error, error_capacity);
}

static void path_to_lower(char *s) {
#if defined(_WIN32)
    for (; *s; ++s) *s = (char)tolower((unsigned char)*s);
#else
    (void)s;
#endif
}

static int canonicalize_path(const char *in, char *out, size_t out_capacity) {
#if defined(_WIN32)
    DWORD n = GetFullPathNameA(in, (DWORD)out_capacity, out, NULL);
    if (n == 0 || n >= out_capacity) return 0;
    return 1;
#else
    char cwd[PATH_MAX];
    if (in[0] == '/') {
        if (!copy_string(out, out_capacity, in)) return 0;
        return 1;
    }
    if (!getcwd(cwd, sizeof(cwd))) return 0;
    if (strlen(cwd) + 1 + strlen(in) >= out_capacity) return 0;
    snprintf(out, out_capacity, "%s/%s", cwd, in);
    return 1;
#endif
}

static int path_is_inside(const char *child, const char *parent) {
    char a[4096], b[4096];
    size_t parent_len;
    if (!canonicalize_path(child, a, sizeof(a)) || !canonicalize_path(parent, b, sizeof(b))) return 0;
    path_to_lower(a);
    path_to_lower(b);
    parent_len = strlen(b);
    while (parent_len && (b[parent_len - 1] == '/' || b[parent_len - 1] == '\\')) b[--parent_len] = 0;
    if (strncmp(a, b, parent_len) != 0) return 0;
    return a[parent_len] == 0 || a[parent_len] == '/' || a[parent_len] == '\\';
}

static int directory_exists(const char *path) {
#if defined(_WIN32)
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

static int path_exists(const char *path) {
#if defined(_WIN32)
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    return access(path, F_OK) == 0;
#endif
}

static int directory_is_empty(const char *path) {
#if defined(_WIN32)
    char pattern[4096];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    int empty = 1;
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 1;
    do {
        if (strcmp(fd.cFileName, ".") && strcmp(fd.cFileName, "..")) empty = 0;
    } while (empty && FindNextFileA(h, &fd));
    FindClose(h);
    return empty;
#else
    DIR *dir = opendir(path);
    struct dirent *ent;
    int empty = 1;
    if (!dir) return 0;
    while ((ent = readdir(dir)) != NULL) {
        if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..")) {
            empty = 0;
            break;
        }
    }
    closedir(dir);
    return empty;
#endif
}

static int make_directories(const char *path) {
#if defined(_WIN32)
    return CreateDirectoryA(path, NULL) || GetLastError() == ERROR_ALREADY_EXISTS;
#else
    return mkdir(path, 0777) == 0 || errno == EEXIST;
#endif
}

static void parent_directory(const char *path, char *out, size_t out_capacity) {
    size_t n;
    const char *slash;
    if (!canonicalize_path(path, out, out_capacity)) {
        out[0] = 0;
        return;
    }
    n = strlen(out);
    slash = out;
    {
        char *p = out + n;
        while (p > out && p[-1] != '/' && p[-1] != '\\') --p;
        if (p > out) p[-1] = 0;
    }
    (void)slash;
}

static int contains_text(const char *name, const char *needle) {
    return name && needle && strstr(name, needle) != NULL;
}

typedef struct string_list {
    char **items;
    size_t count;
    size_t capacity;
} string_list;

static int string_list_add(string_list *list, const char *text) {
    char *copy;
    if (list->count == list->capacity) {
        size_t cap = list->capacity ? list->capacity * 2 : 8;
        char **grown = (char **)realloc(list->items, cap * sizeof(*grown));
        if (!grown) return 0;
        list->items = grown;
        list->capacity = cap;
    }
    copy = ternary_duplicate_string(text, strlen(text));
    if (!copy) return 0;
    list->items[list->count++] = copy;
    return 1;
}

static void string_list_free(string_list *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) free(list->items[i]);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

typedef struct named_stats {
    char *key;
    ternary_reconstruction_error stats;
} named_stats;

typedef struct named_stats_list {
    named_stats *items;
    size_t count;
    size_t capacity;
} named_stats_list;

static named_stats *named_stats_get(named_stats_list *list, const char *key) {
    size_t i;
    for (i = 0; i < list->count; ++i)
        if (strcmp(list->items[i].key, key) == 0) return &list->items[i];
    if (list->count == list->capacity) {
        size_t cap = list->capacity ? list->capacity * 2 : 8;
        named_stats *grown = (named_stats *)realloc(list->items, cap * sizeof(*grown));
        if (!grown) return NULL;
        list->items = grown;
        list->capacity = cap;
    }
    list->items[list->count].key = ternary_duplicate_string(key, strlen(key));
    memset(&list->items[list->count].stats, 0, sizeof(ternary_reconstruction_error));
    if (!list->items[list->count].key) return NULL;
    return &list->items[list->count++];
}

static void named_stats_free(named_stats_list *list) {
    size_t i;
    for (i = 0; i < list->count; ++i) free(list->items[i].key);
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int compare_layer_entries(const void *a, const void *b) {
    const named_stats *ia = (const named_stats *)a;
    const named_stats *ib = (const named_stats *)b;
    long la = strcmp(ia->key, "other") == 0 ? (1L << 30) : atol(ia->key);
    long lb = strcmp(ib->key, "other") == 0 ? (1L << 30) : atol(ib->key);
    if (la < lb) return -1;
    if (la > lb) return 1;
    return strcmp(ia->key, ib->key);
}

static int compare_kind_entries(const void *a, const void *b) {
    const named_stats *ia = (const named_stats *)a;
    const named_stats *ib = (const named_stats *)b;
    return strcmp(ia->key, ib->key);
}

static char *append_fmt(char **buf, size_t *len, size_t *cap, const char *fmt, ...) {
    va_list args;
    char stack[1024];
    int n;
    va_start(args, fmt);
    n = vsnprintf(stack, sizeof(stack), fmt, args);
    va_end(args);
    if (n < 0) return *buf;
    if (*len + (size_t)n + 1 > *cap) {
        size_t cap2 = *cap ? *cap * 2 : 4096;
        char *grown;
        while (cap2 < *len + (size_t)n + 1) cap2 *= 2;
        grown = (char *)realloc(*buf, cap2);
        if (!grown) return *buf;
        *buf = grown;
        *cap = cap2;
    }
    memcpy(*buf + *len, stack, (size_t)n + 1);
    *len += (size_t)n;
    return *buf;
}

int main(int argc, char **argv) {
    const char *model;
    char out_dir[4096] = "";
    const char *fit_s = "lsq", *backend_s = "cpu";
    int list = 0, parity = 0, default_skips = 1;
    string_list match = {0}, skip = {0};
    long limit = -1;
    ternary_convert_options options;
    ternary_source_file source;
    char error[TERNARY_CONVERT_ERROR_CAPACITY];
    size_t *picked = NULL, picked_count = 0, i, k;
    int argi;
    error[0] = 0;
    memset(&source, 0, sizeof(source));
    ternary_convert_options_init(&options);
    if (argc < 3) return usage();
    model = argv[1];
    for (argi = 2; argi < argc; ++argi) {
        const char *a = argv[argi];
        const char *v = NULL;
        if (strcmp(a, "--list") == 0) list = 1;
        else if (strcmp(a, "--check-parity") == 0) parity = 1;
        else if (strcmp(a, "--no-default-skips") == 0) default_skips = 0;
        else if (strcmp(a, "--out") == 0) {
            if (argi + 1 >= argc) return usage();
            if (strlen(argv[++argi]) >= sizeof(out_dir)) return usage();
            if (!copy_string(out_dir, sizeof(out_dir), argv[argi])) return usage();
        } else if (strcmp(a, "--fit") == 0) {
            if (argi + 1 >= argc) return usage();
            fit_s = argv[++argi];
        } else if (strcmp(a, "--backend") == 0) {
            if (argi + 1 >= argc) return usage();
            backend_s = argv[++argi];
        } else if (strcmp(a, "--match") == 0) {
            if (argi + 1 >= argc) return usage();
            if (!string_list_add(&match, argv[++argi])) return 1;
        } else if (strcmp(a, "--skip") == 0) {
            if (argi + 1 >= argc) return usage();
            if (!string_list_add(&skip, argv[++argi])) return 1;
        } else if (strcmp(a, "--limit") == 0) {
            if (argi + 1 >= argc) return usage();
            v = argv[++argi];
            limit = atol(v);
            if (limit < 0) return usage();
        } else if (strcmp(a, "--threads") == 0) {
            if (argi + 1 >= argc) return usage();
            options.thread_count = atoi(argv[++argi]);
            if (options.thread_count < 0 || options.thread_count > 64) return usage();
        } else if (strcmp(a, "--chunk-mweights") == 0) {
            long m;
            if (argi + 1 >= argc) return usage();
            m = atol(argv[++argi]);
            if (m < 1 || m > 1024) return usage();
            options.chunk_weights = (uint64_t)m << 20;
        } else return usage();
    }
    if (strcmp(fit_s, "lsq") == 0) options.fit_kind = TERNARY_GROUP_FIT_LEAST_SQUARES;
    else if (strcmp(fit_s, "absmean") == 0) options.fit_kind = TERNARY_GROUP_FIT_ABSMEAN;
    else return usage();
    if (strcmp(backend_s, "cpu") == 0) options.backend = TERNARY_FIT_BACKEND_CPU;
    else if (strcmp(backend_s, "cuda") == 0) options.backend = TERNARY_FIT_BACKEND_CUDA;
    else return usage();
    options.check_parity = parity;
    if (default_skips) {
        string_list_add(&skip, "embd");
        string_list_add(&skip, "embed");
        string_list_add(&skip, "vision");
        string_list_add(&skip, "audio");
    }

    if (!ternary_open_source(model, &source, error, sizeof(error))) {
        fprintf(stderr, "error: %s\n", error);
        string_list_free(&match);
        string_list_free(&skip);
        return 1;
    }
    picked = (size_t *)calloc(source.tensor_count ? source.tensor_count : 1, sizeof(*picked));
    if (!picked) {
        ternary_source_file_clear(&source);
        string_list_free(&match);
        string_list_free(&skip);
        return 1;
    }
    for (i = 0; i < source.tensor_count; ++i) {
        const ternary_source_tensor *t = &source.tensors[i];
        int selected;
        size_t s;
        if (!ternary_tensor_is_convertible(t, NULL, 0)) continue;
        selected = match.count == 0;
        for (s = 0; s < match.count; ++s) if (contains_text(t->name, match.items[s])) selected = 1;
        for (s = 0; s < skip.count; ++s) if (contains_text(t->name, skip.items[s])) selected = 0;
        if (selected) picked[picked_count++] = i;
    }
    if (limit >= 0 && picked_count > (size_t)limit) picked_count = (size_t)limit;

    if (list) {
        size_t p;
        printf("%s: %s, %zu tensors, %zu selected\n", model,
               source.format == TERNARY_SOURCE_FORMAT_GGUF ? "GGUF" : "safetensors", source.tensor_count, picked_count);
        for (i = 0; i < source.tensor_count; ++i) {
            const ternary_source_tensor *t = &source.tensors[i];
            char why[TERNARY_CONVERT_ERROR_CAPACITY];
            int selected = 0;
            why[0] = 0;
            ternary_tensor_is_convertible(t, why, sizeof(why));
            for (p = 0; p < picked_count; ++p) if (picked[p] == i) selected = 1;
            printf("%c %-60s %-6s rows %8llu cols %6llu  %s\n", selected ? '*' : ' ', t->name ? t->name : "",
                   t->type_name, (unsigned long long)t->rows, (unsigned long long)t->columns, selected ? "" : why);
        }
        free(picked);
        ternary_source_file_clear(&source);
        string_list_free(&match);
        string_list_free(&skip);
        return 0;
    }
    if (!out_dir[0]) {
        free(picked);
        ternary_source_file_clear(&source);
        string_list_free(&match);
        string_list_free(&skip);
        return usage();
    }
    if (!picked_count) {
        fprintf(stderr, "error: no tensor selected\n");
        free(picked);
        ternary_source_file_clear(&source);
        string_list_free(&match);
        string_list_free(&skip);
        return 1;
    }

    {
        char model_dir[4096], out_canon[4096];
        parent_directory(model, model_dir, sizeof(model_dir));
        if (!canonicalize_path(out_dir, out_canon, sizeof(out_canon))) {
            fprintf(stderr, "error: cannot resolve %s\n", out_dir);
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        if (path_is_inside(out_canon, model_dir)) {
            fprintf(stderr, "error: the output folder may not be inside the model's folder\n");
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        if (path_exists(out_canon) && !(directory_exists(out_canon) && directory_is_empty(out_canon))) {
            fprintf(stderr, "error: %s exists and is not an empty folder\n", out_canon);
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        if (!make_directories(out_canon)) {
            fprintf(stderr, "error: cannot create %s\n", out_canon);
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        memcpy(out_dir, out_canon, sizeof(out_dir) - 1);
        out_dir[sizeof(out_dir) - 1] = 0;
    }

    if (options.backend == TERNARY_FIT_BACKEND_CUDA || options.check_parity) {
        const ternary_cuda_info info = ternary_query_cuda_info();
        if (!info.built || info.device_count == 0) {
            fprintf(stderr, "error: CUDA %s\n", info.built ? "device not found" : "backend not built");
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        printf("CUDA device: %s (%.0f MiB)\n", info.device_name, (double)info.total_bytes / 1048576.0);
    }

    {
        FILE *model_file = ternary_open_file_read_only(model);
        char *manifest = NULL, *layers = NULL, *kinds = NULL, *summary = NULL;
        size_t manifest_len = 0, manifest_cap = 0, layers_len = 0, layers_cap = 0, kinds_len = 0, kinds_cap = 0,
               summary_len = 0, summary_cap = 0;
        named_stats_list per_layer = {0}, per_kind = {0};
        ternary_reconstruction_error all;
        double started, read_ms = 0, fit_ms = 0, other_ms = 0, stats_ms = 0, write_ms = 0, total_ms;
        uint64_t parity_checked = 0, parity_diff = 0, bytes_out = 0, weights_in = 0;
        const ternary_cuda_counters *cc;
        memset(&all, 0, sizeof(all));
        if (!model_file) {
            fprintf(stderr, "error: cannot open %s\n", model);
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return 1;
        }
        append_fmt(&manifest, &manifest_len, &manifest_cap,
                   "index\tname\tsource_type\trows\tcols\tfit\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\t"
                   "zero_groups\tread_ms\tfit_ms\tother_fit_ms\tstats_ms\tparity_groups\tparity_different\tptn_bytes\t"
                   "sha256\tfile\n");
        started = tool_now_ms();
        for (k = 0; k < picked_count; ++k) {
            const ternary_source_tensor *t = &source.tensors[picked[k]];
            ternary_convert_result result;
            uint8_t *blob = NULL;
            size_t blob_length = 0;
            ternary_ptn1_view view;
            char fname[512], sha[65], idx[16], layer[64], kind[512];
            char out_path[4600];
            double tw;
            named_stats *slot;
            memset(&result, 0, sizeof(result));
            if (!ternary_convert_tensor(model_file, &source, t, &options, &result, error, sizeof(error))) {
                fprintf(stderr, "error: %s\n", error);
                fclose(model_file);
                free(picked);
                ternary_source_file_clear(&source);
                string_list_free(&match);
                string_list_free(&skip);
                named_stats_free(&per_layer);
                named_stats_free(&per_kind);
                free(manifest);
                return 1;
            }
            tw = tool_now_ms();
            if (!ternary_serialize_ptn1(&result.output, &blob, &blob_length, error, sizeof(error))) {
                fprintf(stderr, "error: %s\n", error);
                ternary_convert_result_clear(&result);
                fclose(model_file);
                return 1;
            }
            if (ternary_parse_ptn1(blob, blob_length, UINT64_C(1) << 31, &view) != TERNARY_CONVERT_OK) {
                fprintf(stderr, "error: %s: output did not re-parse\n", t->name ? t->name : "");
                free(blob);
                ternary_convert_result_clear(&result);
                fclose(model_file);
                return 1;
            }
            snprintf(idx, sizeof(idx), "%04zu", k);
            safe_name(t->name ? t->name : "", kind, sizeof(kind));
            snprintf(fname, sizeof(fname), "%s_%s.ptn", idx, kind);
            snprintf(out_path, sizeof(out_path), "%s/%s", out_dir, fname);
            if (!write_new_file(out_path, blob, blob_length, error, sizeof(error))) {
                fprintf(stderr, "error: %s\n", error);
                free(blob);
                ternary_convert_result_clear(&result);
                fclose(model_file);
                return 1;
            }
            sha256_hex(blob, blob_length, sha);
            write_ms += tool_now_ms() - tw;
            read_ms += result.read_milliseconds;
            fit_ms += result.fit_milliseconds;
            other_ms += result.other_fit_milliseconds;
            stats_ms += result.stats_milliseconds;
            parity_checked += result.parity_groups_checked;
            parity_diff += result.parity_groups_different;
            bytes_out += blob_length;
            weights_in += t->rows * t->columns;
            ternary_reconstruction_error_add(&all, &result.stats);
            layer_of(t->name ? t->name : "", layer, sizeof(layer));
            slot = named_stats_get(&per_layer, layer);
            if (slot) ternary_reconstruction_error_add(&slot->stats, &result.stats);
            {
                const char *name = t->name ? t->name : "";
                char kind_name[512];
                snprintf(kind_name, sizeof(kind_name), "%s", name);
                if (strcmp(layer, "other") != 0) {
                    char needle[80];
                    const char *found;
                    snprintf(needle, sizeof(needle), ".%s.", layer);
                    found = strstr(name, needle);
                    if (found) {
                        found += strlen(needle);
                        snprintf(kind_name, sizeof(kind_name), "%s", found);
                    }
                }
                slot = named_stats_get(&per_kind, kind_name);
                if (slot) ternary_reconstruction_error_add(&slot->stats, &result.stats);
            }
            append_fmt(&manifest, &manifest_len, &manifest_cap,
                       "%s\t%s\t%s\t%llu\t%llu\t%s\t%.6f\t%.6f\t%.6g\t%.6g\t%.4f\t%llu\t%.1f\t%.1f\t%.1f\t%.1f\t"
                       "%llu\t%llu\t%llu\t%s\t%s\n",
                       idx, t->name ? t->name : "", t->type_name, (unsigned long long)t->rows,
                       (unsigned long long)t->columns, fit_s, ternary_reconstruction_relative_error(&result.stats),
                       ternary_reconstruction_cosine(&result.stats), ternary_reconstruction_rmse(&result.stats),
                       result.stats.max_absolute_error,
                       result.stats.weight_count ? (double)result.stats.zero_codes / (double)result.stats.weight_count
                                                 : 0.0,
                       (unsigned long long)result.stats.zero_groups, result.read_milliseconds, result.fit_milliseconds,
                       result.other_fit_milliseconds, result.stats_milliseconds,
                       (unsigned long long)result.parity_groups_checked,
                       (unsigned long long)result.parity_groups_different, (unsigned long long)blob_length, sha, fname);
            printf("[%zu/%zu] %-56s %-5s %6llux%-6llu rel %.4f cos %.4f zeros %.3f fit %.0f ms%s\n", k + 1,
                   picked_count, t->name ? t->name : "", t->type_name, (unsigned long long)t->rows,
                   (unsigned long long)t->columns, ternary_reconstruction_relative_error(&result.stats),
                   ternary_reconstruction_cosine(&result.stats),
                   result.stats.weight_count ? (double)result.stats.zero_codes / (double)result.stats.weight_count : 0.0,
                   result.fit_milliseconds,
                   options.check_parity ? (result.parity_groups_different ? "  PARITY MISMATCH" : "  parity ok") : "");
            fflush(stdout);
            free(blob);
            ternary_convert_result_clear(&result);
        }
        fclose(model_file);
        total_ms = tool_now_ms() - started;

        append_fmt(&layers, &layers_len, &layers_cap,
                   "layer\tweights\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\n");
        qsort(per_layer.items, per_layer.count, sizeof(*per_layer.items), compare_layer_entries);
        for (i = 0; i < per_layer.count; ++i) {
            const ternary_reconstruction_error *s = &per_layer.items[i].stats;
            append_fmt(&layers, &layers_len, &layers_cap, "%s\t%llu\t%.6f\t%.6f\t%.6g\t%.6g\t%.4f\n",
                       per_layer.items[i].key, (unsigned long long)s->weight_count,
                       ternary_reconstruction_relative_error(s), ternary_reconstruction_cosine(s),
                       ternary_reconstruction_rmse(s), s->max_absolute_error,
                       s->weight_count ? (double)s->zero_codes / (double)s->weight_count : 0.0);
        }
        append_fmt(&kinds, &kinds_len, &kinds_cap,
                   "tensor\tweights\trel_error\tcosine\trmse\tmax_abs_error\tzero_share\n");
        qsort(per_kind.items, per_kind.count, sizeof(*per_kind.items), compare_kind_entries);
        for (i = 0; i < per_kind.count; ++i) {
            const ternary_reconstruction_error *s = &per_kind.items[i].stats;
            append_fmt(&kinds, &kinds_len, &kinds_cap, "%s\t%llu\t%.6f\t%.6f\t%.6g\t%.6g\t%.4f\n",
                       per_kind.items[i].key, (unsigned long long)s->weight_count,
                       ternary_reconstruction_relative_error(s), ternary_reconstruction_cosine(s),
                       ternary_reconstruction_rmse(s), s->max_absolute_error,
                       s->weight_count ? (double)s->zero_codes / (double)s->weight_count : 0.0);
        }

        cc = ternary_query_cuda_counters();
        append_fmt(&summary, &summary_len, &summary_cap, "Ternary conversion summary\n");
        append_fmt(&summary, &summary_len, &summary_cap,
                   "source: %s (%s, %llu bytes, opened read-only)\n", model,
                   source.format == TERNARY_SOURCE_FORMAT_GGUF ? "GGUF" : "safetensors",
                   (unsigned long long)source.file_size);
        {
            char threads_text[16];
            if (options.thread_count) snprintf(threads_text, sizeof(threads_text), "%d", options.thread_count);
            else snprintf(threads_text, sizeof(threads_text), "all");
            append_fmt(&summary, &summary_len, &summary_cap,
                       "fit: %s, backend: %s%s, threads: %s, chunk: %llu Mi weights\n", fit_s, backend_s,
                       options.check_parity ? ", parity checked against the other backend" : "",
                       threads_text, (unsigned long long)(options.chunk_weights >> 20));
        }
        append_fmt(&summary, &summary_len, &summary_cap, "tensors: %zu, weights: %llu, output bytes: %llu\n",
                   picked_count, (unsigned long long)weights_in, (unsigned long long)bytes_out);
        append_fmt(&summary, &summary_len, &summary_cap, "overall: all\t%llu\t%.6f\t%.6f\t%.6g\t%.6g\t%.4f\n",
                   (unsigned long long)all.weight_count, ternary_reconstruction_relative_error(&all),
                   ternary_reconstruction_cosine(&all), ternary_reconstruction_rmse(&all), all.max_absolute_error,
                   all.weight_count ? (double)all.zero_codes / (double)all.weight_count : 0.0);
        append_fmt(&summary, &summary_len, &summary_cap,
                   "time: total %.1f s; read+decode %.1f s; fit %.1f s; other-backend fit %.1f s; error stats %.1f s; "
                   "serialize+write %.1f s\n",
                   total_ms / 1000.0, read_ms / 1000.0, fit_ms / 1000.0, other_ms / 1000.0, stats_ms / 1000.0,
                   write_ms / 1000.0);
        if (options.backend == TERNARY_FIT_BACKEND_CUDA || options.check_parity)
            append_fmt(&summary, &summary_len, &summary_cap,
                       "cuda: kernel %.1f ms, copies %.1f ms, converter's own peak %.1f MiB, whole-device peak seen "
                       "%.1f MiB (includes other programs)\n",
                       cc->kernel_milliseconds, cc->copy_milliseconds, (double)cc->own_bytes_peak / 1048576.0,
                       (double)cc->device_used_peak / 1048576.0);
        if (options.check_parity)
            append_fmt(&summary, &summary_len, &summary_cap, "parity: %llu groups compared, %llu different\n",
                       (unsigned long long)parity_checked, (unsigned long long)parity_diff);
        append_fmt(&summary, &summary_len, &summary_cap,
                   "These numbers compare each ternary matrix with its source matrix only; they are not a measure of "
                   "model quality.\n");

        {
            char path[4600];
            int failed = 0;
            snprintf(path, sizeof(path), "%s/manifest.tsv", out_dir);
            if (!write_text(path, manifest ? manifest : "", error, sizeof(error))) failed = 1;
            snprintf(path, sizeof(path), "%s/layers.tsv", out_dir);
            if (!failed && !write_text(path, layers ? layers : "", error, sizeof(error))) failed = 1;
            snprintf(path, sizeof(path), "%s/tensors_by_kind.tsv", out_dir);
            if (!failed && !write_text(path, kinds ? kinds : "", error, sizeof(error))) failed = 1;
            snprintf(path, sizeof(path), "%s/summary.txt", out_dir);
            if (!failed && !write_text(path, summary ? summary : "", error, sizeof(error))) failed = 1;
            if (failed) {
                fprintf(stderr, "error: %s\n", error);
                ternary_cuda_release();
                return 1;
            }
        }
        ternary_cuda_release();
        printf("%s", summary ? summary : "");
        {
            int rc = (options.check_parity && parity_diff) ? 3 : 0;
            free(manifest);
            free(layers);
            free(kinds);
            free(summary);
            named_stats_free(&per_layer);
            named_stats_free(&per_kind);
            free(picked);
            ternary_source_file_clear(&source);
            string_list_free(&match);
            string_list_free(&skip);
            return rc;
        }
    }
}
