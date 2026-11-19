// Bolt PCHIE - NanoQuant LB-ADMM initialization tests.
#include "pchie/pchie_nanoquant.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static void test_rank_and_pack() {
    CHECK(pchie_nq_choose_rank(0, 16, 1.0f) == 0);
    CHECK(pchie_nq_choose_rank(4096, 4096, 1.0f) == 2016); /* (4096/2) - 16 = 2032, floored to 32: 2016 */
    const int r = pchie_nq_choose_rank(2560, 10240, 1.0f);
    /* (2560*10240)/(2560+10240) - 16 = 2048 - 16 = 2032 -> 2016 */
    CHECK(r == 2016);
    const int r08 = pchie_nq_choose_rank(2560, 10240, 0.8f);
    /* 0.8*2048 - 16 = 1638.4 - 16 = 1622.4 -> 1600 */
    CHECK(r08 == 1600);
    CHECK(pchie_nq_choose_rank(16, 16, 1.0f) == 16); /* min(cap, max(32, ...)) capped by 16 */

    std::vector<int8_t> s = {1, -1, 1, 1, -1, -1, 1, -1, 1};
    std::vector<uint8_t> b((s.size() + 7) / 8, 0xFF);
    pchie_nq_pack_signs(s.data(), s.size(), b.data());
    std::vector<int8_t> back(s.size(), 0);
    pchie_nq_unpack_signs(b.data(), s.size(), back.data());
    CHECK(back == s);
    const uint64_t pb = pchie_nq_packed_bytes(32, 64, 16, 4);
    /* U 32*16 bits = 64 bytes, V 64*16 bits = 128 bytes, scales 4*(32+64)=384, total 576 */
    CHECK(pb == 576);
}

static void test_bad_args() {
    pchie_nq_config cfg;
    pchie_nq_config_init(&cfg);
    cfg.use_cuda = 0;
    cfg.rank = 4;
    cfg.outer_iters = 2;
    float w[4] = {1, 2, 3, 4};
    int8_t u[8], v[8];
    float s1[4], s2[4];
    pchie_nq_stats st;
    char err[PCHIE_NQ_ERROR_CAPACITY];
    CHECK(pchie_nq_factorize(nullptr, 2, 2, nullptr, nullptr, &cfg, u, v, s1, s2, &st, err, sizeof err) == PCHIE_NQ_BAD_ARGUMENT);
    CHECK(u[0] == 0 && s1[0] == 0.0f);
    CHECK(pchie_nq_factorize(w, 2, 2, nullptr, nullptr, &cfg, u, v, s1, s2, &st, err, sizeof err) == PCHIE_NQ_OK);
}

static void make_exact(int rows, int cols, int rank, uint64_t seed, std::vector<float> &W, std::vector<int8_t> &U,
                       std::vector<int8_t> &V, std::vector<float> &s1, std::vector<float> &s2) {
    /* Deterministic +/-1 and positive scales. */
    U.assign((size_t)rows * (size_t)rank, 1);
    V.assign((size_t)cols * (size_t)rank, 1);
    s1.assign((size_t)rows, 0.0f);
    s2.assign((size_t)cols, 0.0f);
    uint64_t x = seed ? seed : 1;
    auto nxt = [&x]() {
        x = x * 6364136223846793005ULL + 1;
        return x;
    };
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < rank; ++k)
            U[(size_t)i * (size_t)rank + (size_t)k] = (nxt() & 1) ? (int8_t)1 : (int8_t)-1;
    for (int j = 0; j < cols; ++j)
        for (int k = 0; k < rank; ++k)
            V[(size_t)j * (size_t)rank + (size_t)k] = (nxt() & 1) ? (int8_t)1 : (int8_t)-1;
    for (int i = 0; i < rows; ++i) s1[i] = 0.25f + (float)((nxt() % 1000) / 1000.0);
    for (int j = 0; j < cols; ++j) s2[j] = 0.25f + (float)((nxt() % 1000) / 1000.0);
    W.assign((size_t)rows * (size_t)cols, 0.0f);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int acc = 0;
            for (int k = 0; k < rank; ++k)
                acc += (int)U[(size_t)i * (size_t)rank + (size_t)k] * (int)V[(size_t)j * (size_t)rank + (size_t)k];
            W[(size_t)i * (size_t)cols + (size_t)j] = s1[i] * (float)acc * s2[j];
        }
    }
}

static void test_error_helper() {
    const int rows = 8, cols = 12, rank = 3;
    std::vector<float> W, s1, s2;
    std::vector<int8_t> U, V;
    make_exact(rows, cols, rank, 42, W, U, V, s1, s2);
    pchie_nq_stats st;
    std::memset(&st, 0, sizeof st);
    pchie_nq_accumulate_error(W.data(), U.data(), V.data(), s1.data(), s2.data(), rows, cols, rank, &st);
    CHECK(st.weight_count == (uint64_t)rows * cols);
    CHECK(st.cosine > 0.999999);
    CHECK(st.relative_error < 1e-6);
}

static void test_factorize_random() {
    const int rows = 24, cols = 40, rank = 16;
    std::vector<float> W((size_t)rows * (size_t)cols);
    uint64_t x = 99;
    for (float &v : W) {
        x = x * 6364136223846793005ULL + 1;
        const int s = (int)(x % 2000) - 1000;
        v = (float)s / 200.0f;
    }
    pchie_nq_config cfg;
    pchie_nq_config_init(&cfg);
    cfg.use_cuda = 0;
    cfg.rank = rank;
    cfg.outer_iters = 60;
    cfg.seed = 7;
    cfg.transpose_wide = 1;
    std::vector<int8_t> U((size_t)rows * (size_t)rank), V((size_t)cols * (size_t)rank);
    std::vector<float> s1((size_t)rows), s2((size_t)cols);
    pchie_nq_stats st;
    char err[PCHIE_NQ_ERROR_CAPACITY] = {0};
    const pchie_nq_status ok = pchie_nq_factorize(W.data(), rows, cols, nullptr, nullptr, &cfg, U.data(), V.data(),
                                                  s1.data(), s2.data(), &st, err, sizeof err);
    CHECK(ok == PCHIE_NQ_OK);
    CHECK(st.rank == rank);
    CHECK(st.rows == rows && st.cols == cols);
    CHECK(st.transposed == 1); /* 24 < 40 */
    int bad_sign = 0, nonfinite = 0;
    for (int8_t v : U) if (v != 1 && v != -1) ++bad_sign;
    for (int8_t v : V) if (v != 1 && v != -1) ++bad_sign;
    for (float v : s1) if (!std::isfinite(v)) ++nonfinite;
    for (float v : s2) if (!std::isfinite(v)) ++nonfinite;
    CHECK(bad_sign == 0);
    CHECK(nonfinite == 0);
    CHECK(st.cosine > 0.70); /* init-only ADMM on a small random matrix */
    CHECK(st.bits_per_weight > 0.0 && st.bits_per_weight < 4.0);
    CHECK(st.bits_per_weight_fp16 < st.bits_per_weight);
}

static void test_factorize_lowrank_binary() {
    const int rows = 32, cols = 32, rank = 8;
    std::vector<float> W, s1t, s2t;
    std::vector<int8_t> Ut, Vt;
    make_exact(rows, cols, rank, 12345, W, Ut, Vt, s1t, s2t);
    pchie_nq_config cfg;
    pchie_nq_config_init(&cfg);
    cfg.use_cuda = 0;
    cfg.rank = rank;
    cfg.outer_iters = 120;
    cfg.seed = 3;
    cfg.transpose_wide = 0;
    std::vector<int8_t> U((size_t)rows * (size_t)rank), V((size_t)cols * (size_t)rank);
    std::vector<float> s1((size_t)rows), s2((size_t)cols);
    pchie_nq_stats st;
    char err[PCHIE_NQ_ERROR_CAPACITY] = {0};
    CHECK(pchie_nq_factorize(W.data(), rows, cols, nullptr, nullptr, &cfg, U.data(), V.data(), s1.data(), s2.data(),
                             &st, err, sizeof err) == PCHIE_NQ_OK);
    CHECK(st.cosine > 0.95);
    CHECK(st.relative_error < 0.25);
}

static void test_hessian_ones_match() {
    const int rows = 16, cols = 20, rank = 8;
    std::vector<float> W((size_t)rows * (size_t)cols, 0.0f);
    uint64_t x = 5;
    for (float &v : W) {
        x = x * 6364136223846793005ULL + 1;
        v = ((int)(x % 201) - 100) / 50.0f;
    }
    std::vector<float> din((size_t)cols, 1.0f), dout((size_t)rows, 1.0f);
    pchie_nq_config cfg;
    pchie_nq_config_init(&cfg);
    cfg.use_cuda = 0;
    cfg.rank = rank;
    cfg.outer_iters = 20;
    cfg.seed = 11;
    cfg.transpose_wide = 0;
    std::vector<int8_t> Ua((size_t)rows * rank), Va((size_t)cols * rank), Ub(Ua), Vb(Va);
    std::vector<float> s1a(rows), s2a(cols), s1b(rows), s2b(cols);
    pchie_nq_stats sa, sb;
    CHECK(pchie_nq_factorize(W.data(), rows, cols, nullptr, nullptr, &cfg, Ua.data(), Va.data(), s1a.data(), s2a.data(),
                             &sa, nullptr, 0) == PCHIE_NQ_OK);
    CHECK(pchie_nq_factorize(W.data(), rows, cols, din.data(), dout.data(), &cfg, Ub.data(), Vb.data(), s1b.data(),
                             s2b.data(), &sb, nullptr, 0) == PCHIE_NQ_OK);
    CHECK(Ua == Ub && Va == Vb);
    int same = 1;
    for (int i = 0; i < rows; ++i) if (std::fabs(s1a[i] - s1b[i]) > 1e-5f) same = 0;
    for (int j = 0; j < cols; ++j) if (std::fabs(s2a[j] - s2b[j]) > 1e-5f) same = 0;
    CHECK(same);
}


/* CPU vs CUDA parity, documented tolerances, small matrices only.
 *
 *   |cosine_cpu - cosine_cuda|          <= 1e-4
 *   |rel_cpu    - rel_cuda|             <= 1e-4
 *   ||What_cpu - What_cuda|| / ||What_cpu||  <= 1e-3
 *   scale L2 relative                  <= 1e-3 when every sign agrees
 *
 * Signs may differ where the corresponding latent is under 1e-5 in magnitude;
 * a reconstruction that stays inside the tolerances above still passes. The
 * CUDA section is skipped (not failed) when this build has no device.
 */
static void reconstruct(const int8_t *U, const int8_t *V, const float *s1, const float *s2, int rows, int cols,
                        int rank, std::vector<float> &What) {
    What.assign((size_t)rows * (size_t)cols, 0.0f);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int acc = 0;
            for (int k = 0; k < rank; ++k)
                acc += (int)U[(size_t)i * (size_t)rank + (size_t)k] * (int)V[(size_t)j * (size_t)rank + (size_t)k];
            What[(size_t)i * (size_t)cols + (size_t)j] = s1[i] * (float)acc * s2[j];
        }
    }
}

static double vec_rel(const std::vector<float> &a, const std::vector<float> &b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)a[i] * (double)a[i];
    }
    return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

static void run_one_parity(const char *tag, const std::vector<float> &W, int rows, int cols, const float *din,
                           const float *dout, pchie_nq_config cfg) {
    const int rank = cfg.rank;
    std::vector<int8_t> Uc((size_t)rows * rank), Vc((size_t)cols * rank), Ug(Uc), Vg(Vc);
    std::vector<float> s1c(rows), s2c(cols), s1g(rows), s2g(cols);
    pchie_nq_stats sc, sg;
    char errc[PCHIE_NQ_ERROR_CAPACITY] = {0}, errg[PCHIE_NQ_ERROR_CAPACITY] = {0};
    cfg.use_cuda = 0;
    CHECK(pchie_nq_factorize(W.data(), rows, cols, din, dout, &cfg, Uc.data(), Vc.data(), s1c.data(), s2c.data(), &sc,
                             errc, sizeof errc) == PCHIE_NQ_OK);
    CHECK(sc.backend == 0);
    cfg.use_cuda = 2;
    const pchie_nq_status gst = pchie_nq_factorize(W.data(), rows, cols, din, dout, &cfg, Ug.data(), Vg.data(),
                                                   s1g.data(), s2g.data(), &sg, errg, sizeof errg);
    if (gst != PCHIE_NQ_OK) {
        std::printf("FAIL %s: CUDA factorize: %s\n", tag, errg);
        ++g_fail;
        return;
    }
    CHECK(sg.backend == 1);
    CHECK(std::fabs(sc.cosine - sg.cosine) <= 1e-4);
    CHECK(std::fabs(sc.relative_error - sg.relative_error) <= 1e-4);
    std::vector<float> Wc, Wg;
    reconstruct(Uc.data(), Vc.data(), s1c.data(), s2c.data(), rows, cols, rank, Wc);
    reconstruct(Ug.data(), Vg.data(), s1g.data(), s2g.data(), rows, cols, rank, Wg);
    CHECK(vec_rel(Wc, Wg) <= 1e-3);
    const int signs_agree = (Uc == Ug && Vc == Vg);
    if (signs_agree) {
        CHECK(vec_rel(s1c, s1g) <= 1e-3);
        CHECK(vec_rel(s2c, s2g) <= 1e-3);
    }
    std::printf("parity %s  cos CPU %.6f CUDA %.6f  rel CPU %.6f CUDA %.6f  What rel %.3e  signs %s  dev %llu B\n",
                tag, sc.cosine, sg.cosine, sc.relative_error, sg.relative_error, vec_rel(Wc, Wg),
                signs_agree ? "agree" : "differ", (unsigned long long)sg.device_bytes);
}

static void test_cuda_parity() {
    const pchie_nq_cuda_info info = pchie_nq_query_cuda();
    if (!pchie_nq_cuda_available()) {
        std::printf("CUDA parity skipped (built=%d devices=%d)\n", info.built, info.device_count);
        return;
    }
    std::printf("CUDA parity on %s  (%.1f MiB total, %.1f MiB free)\n", info.device_name,
                info.total_bytes / (1024.0 * 1024.0), info.free_bytes / (1024.0 * 1024.0));

    {
        const int rows = 24, cols = 40, rank = 16;
        std::vector<float> W((size_t)rows * (size_t)cols);
        uint64_t x = 99;
        for (float &v : W) {
            x = x * 6364136223846793005ULL + 1;
            v = ((int)(x % 2000) - 1000) / 200.0f;
        }
        pchie_nq_config cfg;
        pchie_nq_config_init(&cfg);
        cfg.rank = rank;
        cfg.outer_iters = 60;
        cfg.seed = 7;
        cfg.transpose_wide = 1;
        run_one_parity("random 24x40 r16", W, rows, cols, nullptr, nullptr, cfg);
    }
    {
        const int rows = 32, cols = 32, rank = 8;
        std::vector<float> W, s1t, s2t;
        std::vector<int8_t> Ut, Vt;
        make_exact(rows, cols, rank, 12345, W, Ut, Vt, s1t, s2t);
        pchie_nq_config cfg;
        pchie_nq_config_init(&cfg);
        cfg.rank = rank;
        cfg.outer_iters = 120;
        cfg.seed = 3;
        cfg.transpose_wide = 0;
        run_one_parity("exact-binary 32x32 r8", W, rows, cols, nullptr, nullptr, cfg);
    }
    {
        const int rows = 16, cols = 20, rank = 8;
        std::vector<float> W((size_t)rows * (size_t)cols);
        uint64_t x = 5;
        for (float &v : W) {
            x = x * 6364136223846793005ULL + 1;
            v = ((int)(x % 201) - 100) / 50.0f;
        }
        std::vector<float> din((size_t)cols), dout((size_t)rows);
        for (int j = 0; j < cols; ++j) din[j] = 0.25f + 0.05f * (float)j;
        for (int i = 0; i < rows; ++i) dout[i] = 0.5f + 0.03f * (float)i;
        pchie_nq_config cfg;
        pchie_nq_config_init(&cfg);
        cfg.rank = rank;
        cfg.outer_iters = 20;
        cfg.seed = 11;
        cfg.transpose_wide = 0;
        run_one_parity("hessian 16x20 r8", W, rows, cols, din.data(), dout.data(), cfg);
    }
}

int main() {
    test_rank_and_pack();
    test_bad_args();
    test_error_helper();
    test_factorize_random();
    test_factorize_lowrank_binary();
    test_hessian_ones_match();
    test_cuda_parity();
    std::printf("pchie_nanoquant_test: %d pass, %d fail\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
