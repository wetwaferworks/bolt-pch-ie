/* Bolt PCHIE - NanoQuant factor tool: reads a safetensors or GGUF model read-only
 * and writes low-rank binary factors under --out. See docs/ternary/nanoquant.md. */
#include "pchie/pchie_nanoquant.h"
#include "pchie/pchie_ternary_convert.h"
extern "C" {
#include "pj_sha256.h"
}

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <fcntl.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static int mkdir_one(const char *path) {
#ifdef _WIN32
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

static int mkdir_p(const char *path) {
    std::string s(path);
    if (s.empty()) return -1;
    for (size_t i = 1; i < s.size(); ++i) {
        if (s[i] == '/' || s[i] == '\\') {
            char c = s[i];
            s[i] = 0;
            mkdir_one(s.c_str());
            s[i] = c;
        }
    }
    mkdir_one(s.c_str());
    return 0;
}

static void hex32(const uint8_t h[32], char out[65]) {
    static const char *d = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out[2 * i] = d[h[i] >> 4];
        out[2 * i + 1] = d[h[i] & 15];
    }
    out[64] = 0;
}

static int sha256_file(const char *path, uint8_t out[32], char *error, size_t cap) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::snprintf(error, cap, "could not open %s to hash", path);
        return 0;
    }
    pj_sha256 ctx;
    pj_sha256_init(&ctx);
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) pj_sha256_update(&ctx, buf, n);
    if (std::ferror(f)) {
        std::fclose(f);
        std::snprintf(error, cap, "short or failed read while hashing %s", path);
        return 0;
    }
    std::fclose(f);
    pj_sha256_final(&ctx, out);
    return 1;
}

static int is_linear_weight(const ternary_source_tensor *t) {
    if (!t || t->dimension_count < 2) return 0;
    if (t->rows <= 0 || t->columns <= 0) return 0;
    if (t->type != TERNARY_SOURCE_WEIGHT_F32 && t->type != TERNARY_SOURCE_WEIGHT_F16 &&
        t->type != TERNARY_SOURCE_WEIGHT_BF16)
        return 0;
    if (!t->name) return 0;
    const char *n = t->name;
    if (!std::strstr(n, "weight")) return 0;
    if (std::strstr(n, "embed") || std::strstr(n, "norm") || std::strstr(n, "Norm") ||
        std::strstr(n, "bias") || std::strstr(n, "conv1d") || std::strstr(n, "rotary") ||
        std::strstr(n, "A_log") || std::strstr(n, "dt_bias"))
        return 0;
    return 1;
}

static std::string file_name_for(const char *tensor_name) {
    std::string s;
    for (const char *p = tensor_name; *p; ++p) {
        char c = *p;
        if (c == '/' || c == '\\' || c == ':' || c == ' ') c = '.';
        s.push_back(c);
    }
    if (s.size() > 180) s = s.substr(s.size() - 180);
    s += ".nq1";
    return s;
}

static int write_nq1(const char *path, const char *name, const pchie_nq_stats *st, const int8_t *u, const int8_t *v,
                     const float *s1, const float *s2) {
    const std::string partial = std::string(path) + ".partial";
    FILE *f = std::fopen(partial.c_str(), "wb");
    if (!f) return 0;
    const uint64_t ubits = (uint64_t)st->rows * (uint64_t)st->rank;
    const uint64_t vbits = (uint64_t)st->cols * (uint64_t)st->rank;
    std::vector<uint8_t> up((ubits + 7) / 8), vp((vbits + 7) / 8);
    pchie_nq_pack_signs(u, ubits, up.data());
    pchie_nq_pack_signs(v, vbits, vp.data());
    const char magic[4] = {'N', 'Q', '1', '\0'};
    const uint32_t version = 1;
    const uint32_t nlen = (uint32_t)std::strlen(name);
    int ok = 1;
    ok = ok && std::fwrite(magic, 1, 4, f) == 4;
    ok = ok && std::fwrite(&version, 4, 1, f) == 1;
    ok = ok && std::fwrite(&st->rows, 4, 1, f) == 1;
    ok = ok && std::fwrite(&st->cols, 4, 1, f) == 1;
    ok = ok && std::fwrite(&st->rank, 4, 1, f) == 1;
    ok = ok && std::fwrite(&nlen, 4, 1, f) == 1;
    ok = ok && std::fwrite(name, 1, nlen, f) == nlen;
    ok = ok && std::fwrite(up.data(), 1, up.size(), f) == up.size();
    ok = ok && std::fwrite(vp.data(), 1, vp.size(), f) == vp.size();
    ok = ok && std::fwrite(s1, 4, (size_t)st->rows, f) == (size_t)st->rows;
    ok = ok && std::fwrite(s2, 4, (size_t)st->cols, f) == (size_t)st->cols;
    ok = ok && std::fwrite(&st->cosine, 8, 1, f) == 1;
    ok = ok && std::fwrite(&st->relative_error, 8, 1, f) == 1;
    std::fclose(f);
    if (!ok) {
        std::remove(partial.c_str());
        return 0;
    }
#ifdef _WIN32
    if (std::rename(partial.c_str(), path) != 0) {
        std::remove(path);
        if (std::rename(partial.c_str(), path) != 0) return 0;
    }
#else
    if (std::rename(partial.c_str(), path) != 0) return 0;
#endif
    return 1;
}

static void usage() {
    std::fprintf(stderr,
                 "usage: pchie_nanoquant factor MODEL --out DIR [--bits 1.0] [--bits 0.8]\n"
                 "                                 [--iters 400] [--seed 0] [--rank N]\n");
}

int main(int argc, char **argv) {
    const char *model = nullptr, *out = nullptr;
    std::vector<float> bits_list;
    int iters = 400, rank_override = 0;
    uint64_t seed = 0;
    if (argc < 2 || std::strcmp(argv[1], "factor") != 0) {
        usage();
        return 2;
    }
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else if (std::strcmp(argv[i], "--bits") == 0 && i + 1 < argc) bits_list.push_back((float)std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "--rank") == 0 && i + 1 < argc) rank_override = std::atoi(argv[++i]);
        else if (!model && argv[i][0] != '-') model = argv[i];
        else {
            usage();
            return 2;
        }
    }
    if (!model || !out) {
        usage();
        return 2;
    }
    if (bits_list.empty()) bits_list.push_back(1.0f);

    char error[TERNARY_CONVERT_ERROR_CAPACITY];
    uint8_t sha[32];
    if (!sha256_file(model, sha, error, sizeof error)) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    char sha_hex[65];
    hex32(sha, sha_hex);
    mkdir_p(out);
    {
        std::string sp = std::string(out) + "/source.sha256";
        FILE *sf = std::fopen(sp.c_str(), "wb");
        if (sf) {
            std::fprintf(sf, "%s  %s\n", sha_hex, model);
            std::fclose(sf);
        }
    }
    std::printf("source sha256 %s (read-only)\n", sha_hex);
    std::fflush(stdout);

    ternary_source_file src;
    std::memset(&src, 0, sizeof src);
    if (!ternary_open_source(model, &src, error, sizeof error)) {
        std::fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    FILE *file = std::fopen(model, "rb");
    if (!file) {
        std::fprintf(stderr, "could not reopen model read-only\n");
        ternary_source_file_clear(&src);
        return 1;
    }

    int rc = 0;
    for (float bits : bits_list) {
        char bitdir[1024];
        std::snprintf(bitdir, sizeof bitdir, "%s/bits-%.2f", out, bits);
        mkdir_p(bitdir);
        std::string tdir = std::string(bitdir) + "/tensors";
        mkdir_p(tdir.c_str());
        std::string tsvp = std::string(bitdir) + "/layers.tsv";
        std::string sump = std::string(bitdir) + "/summary.txt";
        FILE *tsv = std::fopen(tsvp.c_str(), "wb");
        if (tsv) std::fprintf(tsv, "name\trows\tcols\trank\tweights\tbytes\tbpw\tbpw_fp16\tcosine\trel\tsecs\n");

        pchie_nq_config cfg;
        pchie_nq_config_init(&cfg);
        cfg.bits = bits;
        cfg.rank = rank_override;
        cfg.outer_iters = iters;
        cfg.seed = seed;

        uint64_t n_ok = 0, n_skip = 0, n_fail = 0, n_already = 0;
        uint64_t weights = 0, packed = 0;
        double sw2 = 0, se2 = 0, swr = 0, sr2 = 0; /* accumulate via per-tensor stats approximately */
        double cosine_w = 0, rel_w = 0;
        auto t0all = std::chrono::steady_clock::now();
        uint8_t *raw = nullptr;
        size_t raw_cap = 0;

        for (size_t i = 0; i < src.tensor_count; ++i) {
            const ternary_source_tensor *t = &src.tensors[i];
            if (!is_linear_weight(t)) {
                ++n_skip;
                continue;
            }
            if (t->rows > (uint64_t)INT32_MAX || t->columns > (uint64_t)INT32_MAX) {
                std::fprintf(stderr, "skip too large %s\n", t->name);
                ++n_skip;
                continue;
            }
            const int rows = (int)t->rows, cols = (int)t->columns;
            const std::string fn = tdir + "/" + file_name_for(t->name);
            const std::string okf = fn + ".ok";
            FILE *ex = std::fopen(okf.c_str(), "rb");
            if (ex) {
                std::fclose(ex);
                ++n_already;
                continue;
            }
            std::vector<float> W((size_t)rows * (size_t)cols);
            if (!ternary_read_rows(file, &src, t, 0, (uint64_t)rows, &raw, &raw_cap, W.data(), error, sizeof error)) {
                std::fprintf(stderr, "read failed %s: %s\n", t->name, error);
                ++n_fail;
                rc = 1;
                continue;
            }
            const int rank = cfg.rank > 0 ? cfg.rank : pchie_nq_choose_rank(rows, cols, bits);
            std::vector<int8_t> U((size_t)rows * (size_t)rank), V((size_t)cols * (size_t)rank);
            std::vector<float> s1((size_t)rows), s2((size_t)cols);
            pchie_nq_stats st;
            char nerr[PCHIE_NQ_ERROR_CAPACITY];
            auto t0 = std::chrono::steady_clock::now();
            const pchie_nq_status stt =
                pchie_nq_factorize(W.data(), rows, cols, nullptr, nullptr, &cfg, U.data(), V.data(), s1.data(),
                                   s2.data(), &st, nerr, sizeof nerr);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (stt != PCHIE_NQ_OK) {
                std::fprintf(stderr, "factor failed %s: %s\n", t->name, nerr);
                ++n_fail;
                rc = 1;
                continue;
            }
            if (!write_nq1(fn.c_str(), t->name, &st, U.data(), V.data(), s1.data(), s2.data())) {
                std::fprintf(stderr, "write failed %s\n", t->name);
                ++n_fail;
                rc = 1;
                continue;
            }
            FILE *ok = std::fopen(okf.c_str(), "wb");
            if (ok) {
                std::fprintf(ok, "secs=%.3f cosine=%.6f rel=%.6f\n", secs, st.cosine, st.relative_error);
                std::fclose(ok);
            }
            ++n_ok;
            weights += st.weight_count;
            packed += st.packed_bytes;
            cosine_w += st.cosine * (double)st.weight_count;
            rel_w += st.relative_error * (double)st.weight_count;
            (void)sw2;
            (void)se2;
            (void)swr;
            (void)sr2;
            if (tsv)
                std::fprintf(tsv, "%s\t%d\t%d\t%d\t%llu\t%llu\t%.6f\t%.6f\t%.6f\t%.6f\t%.3f\n", t->name, st.rows,
                             st.cols, st.rank, (unsigned long long)st.weight_count, (unsigned long long)st.packed_bytes,
                             st.bits_per_weight, st.bits_per_weight_fp16, st.cosine, st.relative_error, secs);
            std::printf("ok %s  %dx%d r=%d  bpw=%.3f (fp16 %.3f)  cos=%.4f rel=%.4f  %.1fs\n", t->name, st.rows,
                        st.cols, st.rank, st.bits_per_weight, st.bits_per_weight_fp16, st.cosine, st.relative_error,
                        secs);
            std::fflush(stdout);
        }
        std::free(raw);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0all).count();
        const double cos_avg = weights ? cosine_w / (double)weights : 0.0;
        const double rel_avg = weights ? rel_w / (double)weights : 0.0;
        const double bpw = weights ? (8.0 * (double)packed) / (double)weights : 0.0;
        if (tsv) std::fclose(tsv);
        FILE *sum = std::fopen(sump.c_str(), "wb");
        if (sum) {
            std::fprintf(sum, "NanoQuant LB-ADMM initialization\n");
            std::fprintf(sum, "source: %s\n", model);
            std::fprintf(sum, "source sha256: %s (file opened read-only; not written)\n", sha_hex);
            std::fprintf(sum, "bits target: %.2f  iters: %d  seed: %llu\n", bits, iters, (unsigned long long)seed);
            std::fprintf(sum, "tensors factored: %llu  already present: %llu  skipped: %llu  failed: %llu\n",
                         (unsigned long long)n_ok, (unsigned long long)n_already, (unsigned long long)n_skip,
                         (unsigned long long)n_fail);
            std::fprintf(sum, "weights: %llu  packed bytes: %llu  GiB: %.3f  bits/weight (fp32 scales): %.4f\n",
                         (unsigned long long)weights, (unsigned long long)packed, packed / (1024.0 * 1024.0 * 1024.0),
                         bpw);
            std::fprintf(sum, "weight-averaged cosine: %.6f  weight-averaged relative error: %.6f\n", cos_avg, rel_avg);
            std::fprintf(sum, "time: %.1f s\n", elapsed);
            std::fprintf(sum,
                         "These numbers compare each factored matrix with its source matrix only; they are not a "
                         "measure of model quality. Hessian diagonals were taken as 1 (no calibration set).\n");
            std::fclose(sum);
        }
        std::printf("bits %.2f done  factored=%llu already=%llu fail=%llu  weights=%llu  GiB=%.3f  bpw=%.4f  "
                    "cos=%.4f rel=%.4f  %.1fs\n",
                    bits, (unsigned long long)n_ok, (unsigned long long)n_already, (unsigned long long)n_fail,
                    (unsigned long long)weights, packed / (1024.0 * 1024.0 * 1024.0), bpw, cos_avg, rel_avg, elapsed);
        std::fflush(stdout);
    }

    std::fclose(file);
    /* Re-hash the source and refuse if it moved. */
    uint8_t sha2[32];
    if (!sha256_file(model, sha2, error, sizeof error) || std::memcmp(sha, sha2, 32) != 0) {
        std::fprintf(stderr, "source hash changed or could not be re-read; the original must stay untouched\n");
        ternary_source_file_clear(&src);
        return 1;
    }
    std::printf("source sha256 unchanged %s\n", sha_hex);
    ternary_source_file_clear(&src);
    return rc;
}
