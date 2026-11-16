// Ternary research path test bench: converter fits against brute force, the B3B and TL3 packings (round trips
// and strict validation), every research kernel against the shipping scalar reference bit for bit (random shapes
// and Archie's adversarial cases), and the integer softmax and RMSNorm bounds.
#include "pchie/pchie_ternary.h"
#include "pchie/pchie_ternary_research.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>
using namespace pchie::ternary;
using namespace pchie::ternary::research;

static int passes = 0, fails = 0;
#define CHECK(c, ...) do { if (c) ++passes; else { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static std::vector<int8_t> rand_trits(std::mt19937& r, size_t n, int zero_pct) {
    std::uniform_int_distribution<int> d(0, 99), s(0, 1);
    std::vector<int8_t> q(n);
    for (auto& v : q) v = d(r) < zero_pct ? 0 : (s(r) ? 1 : -1);
    return q;
}
static Tensor make_i2(const std::vector<int8_t>& q, uint64_t rows, uint64_t cols, ScaleMode m, std::mt19937& r,
                      float lo = 0.001f, float hi = 0.2f) {
    Tensor t; t.packing = Packing::I2; t.scale_mode = m; t.rows = rows; t.cols = cols;
    t.codes.resize(size_t(rows * cols / 4));
    for (uint64_t i = 0; i < rows; ++i) pack_row(q.data() + i * cols, cols, Packing::I2, t.codes.data() + i * cols / 4);
    std::uniform_real_distribution<float> sd(lo, hi);
    t.scales.resize(m == ScaleMode::PerTensor ? 1 : size_t(rows * cols / kBlock));
    for (auto& s : t.scales) s = sd(r);
    return t;
}
static ActRow act_from(const std::vector<float>& x) { ActRow a; quantize_activations(x.data(), x.size(), a); return a; }
static ActRow rand_act(std::mt19937& r, uint64_t n) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(n)); for (auto& v : x) v = nd(r);
    return act_from(x);
}

// Runs every kernel on the same trits and compares bits with the shipping scalar reference.
static void all_kernels_match(const Tensor& t, const ActRow& x, const char* what) {
    View v; Status st = view_of(t, v);
    CHECK(st == Status::Ok, "%s: view_of %s", what, status_name(st));
    std::vector<float> ref(size_t(t.rows)), y(size_t(t.rows));
    Status sref = matvec_i2(v, x, Kernel::I2Scalar, ref.data(), t.rows);
    auto same = [&](Kernel k, Status s) {
        if (!kernel_available(k)) return;
        CHECK(s == sref, "%s: %s status %s vs %s", what, kernel_name(k), status_name(s), status_name(sref));
        CHECK(std::memcmp(y.data(), ref.data(), ref.size() * 4) == 0, "%s: %s bits differ", what, kernel_name(k));
    };
    for (Kernel k : {Kernel::I2Avx2Biased, Kernel::I2Avx2Signed, Kernel::I2Avx2ZeroSkip}) {
        if (!kernel_available(k)) continue;
        same(k, matvec_i2(v, x, k, y.data(), t.rows));
    }
    PackedTensor b3; CHECK(pack_b3b(t, b3) == Status::Ok && validate(b3) == Status::Ok, "%s: b3b pack", what);
    B3BAct xb; CHECK(prepare_b3b(x, xb) == Status::Ok, "%s: b3b act", what);
    for (Kernel k : {Kernel::B3BScalar, Kernel::B3BAvx2}) {
        if (!kernel_available(k)) continue;
        same(k, matvec_b3b(b3, x, xb, k, y.data(), t.rows));
    }
    if (t.rows % kTL3TileRows == 0) {
        PackedTensor tl; CHECK(pack_tl3(t, tl) == Status::Ok && validate(tl) == Status::Ok, "%s: tl3 pack", what);
        TL3Act xt; CHECK(prepare_tl3(x, xt) == Status::Ok, "%s: tl3 act", what);
        for (Kernel k : {Kernel::TL3Scalar, Kernel::TL3Avx2}) {
            if (!kernel_available(k)) continue;
            same(k, matvec_tl3(tl, x, xt, k, y.data(), t.rows));
        }
    }
}

static void test_helpers() {
    CHECK(rshift_rne(5, 1) == 2 && rshift_rne(7, 1) == 4 && rshift_rne(-5, 1) == -2 && rshift_rne(-7, 1) == -4, "rne ties");
    CHECK(rshift_rne(6, 2) == 2 && rshift_rne(-6, 2) == -2 && rshift_rne(9, 2) == 2, "rne quarters");
    bool ok = true;
    for (unsigned v = 0; v < 256; ++v) if (((v * 21846u) >> 16) != v / 3) ok = false;
    CHECK(ok, "mulhi divide-by-3 exact for 0..255");
}

static void test_converter() {
    std::mt19937 r(7);
    // Least squares beats or ties every ternary code with its own best gamma (brute force, n = 8).
    for (int trial = 0; trial < 200; ++trial) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        float w[8], scratch[8]; int8_t q[8]; float g;
        for (auto& v : w) v = nd(r);
        CHECK(ternarize_group(w, 8, Fit::LeastSquares, Round::HalfEven, 0, 0, scratch, q, &g) == Status::Ok, "ls ok");
        double e_ls = 0; for (int i = 0; i < 8; ++i) { const double d = w[i] - double(g) * q[i]; e_ls += d * d; }
        double best = 1e300;
        for (int code = 0; code < 6561; ++code) {
            int c = code; int8_t qq[8]; double sw = 0; int nz = 0;
            for (int i = 0; i < 8; ++i) { qq[i] = int8_t(c % 3 - 1); c /= 3; sw += qq[i] * double(w[i]); nz += qq[i] != 0; }
            const double a = nz ? sw / nz : 0.0;
            double e = 0; for (int i = 0; i < 8; ++i) { const double d = w[i] - a * qq[i]; e += d * d; }
            best = std::min(best, e);
        }
        CHECK(e_ls <= best * (1 + 1e-6) + 1e-12, "least squares optimal (%g vs %g)", e_ls, best);
    }
    // Rounding rules on exact ties: gamma = 1 when |w| = {0.5, 1.5} -> mean 1.
    { float w[2] = {0.5f, -1.5f}; int8_t q[2]; float g;
      ternarize_group(w, 2, Fit::Absmean, Round::HalfEven, 0, 0, nullptr, q, &g);
      CHECK(g == 1.0f && q[0] == 0 && q[1] == -1, "half-even tie to 0");
      ternarize_group(w, 2, Fit::Absmean, Round::HalfAway, 0, 0, nullptr, q, &g);
      CHECK(q[0] == 1 && q[1] == -1, "half-away tie to 1"); }
    // Stochastic: deterministic per seed, unbiased on average.
    { std::vector<float> w(20000, 0.25f); w[0] = 1.75f;   // gamma = (0.25*19999 + 1.75)/20000
      std::vector<int8_t> q1(w.size()), q2(w.size()); float g1, g2;
      ternarize_group(w.data(), w.size(), Fit::Absmean, Round::Stochastic, 42, 0, nullptr, q1.data(), &g1);
      ternarize_group(w.data(), w.size(), Fit::Absmean, Round::Stochastic, 42, 0, nullptr, q2.data(), &g2);
      CHECK(q1 == q2 && g1 == g2, "stochastic deterministic for a seed");
      double m = 0; for (size_t i = 1; i < w.size(); ++i) m += q1[i] * double(g1);
      m /= double(w.size() - 1);
      CHECK(std::fabs(m - 0.25) < 0.01, "stochastic unbiased (%g)", m); }
    // Edge cases: subnormal gamma flushes to an exact zero group; huge weights are fine; NaN refused, q zeroed.
    { float w[4] = {1e-40f, -2e-40f, 0.0f, 3e-41f}; int8_t q[4] = {9, 9, 9, 9}; float g = 1;
      CHECK(ternarize_group(w, 4, Fit::Absmean, Round::HalfEven, 0, 0, nullptr, q, &g) == Status::Ok && g == 0.0f &&
            q[0] == 0 && q[1] == 0 && q[3] == 0, "subnormal gamma flushed"); }
    { float w[4] = {FLT_MAX, -FLT_MAX, 1.0f, 0.0f}; int8_t q[4]; float g;
      CHECK(ternarize_group(w, 4, Fit::Absmean, Round::HalfEven, 0, 0, nullptr, q, &g) == Status::Ok && std::isfinite(g) &&
            q[0] == 1 && q[1] == -1 && q[2] == 0, "huge weights"); }
    { float w[3] = {1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f}; int8_t q[3] = {5, 5, 5}; float g = 3;
      CHECK(ternarize_group(w, 3, Fit::Absmean, Round::HalfEven, 0, 0, nullptr, q, &g) == Status::NonFinite &&
            g == 0.0f && q[0] == 0 && q[2] == 0, "NaN refused, safe state"); }
    { int8_t q[2]; float g; float w[2] = {1, 2};
      CHECK(ternarize_group(w, 2, Fit::LeastSquares, Round::HalfEven, 0, 0, nullptr, q, &g) == Status::BadArgument, "ls needs scratch");
      CHECK(ternarize_group(w, 0, Fit::Absmean, Round::HalfEven, 0, 0, nullptr, q, &g) == Status::BadArgument, "n = 0"); }
}

static void test_packings() {
    std::mt19937 r(11);
    for (uint64_t cols : {128ull, 256ull, 640ull}) {
        const uint64_t rows = 64;
        auto q = rand_trits(r, size_t(rows * cols), 34);
        Tensor t = make_i2(q, rows, cols, ScaleMode::PerBlock, r);
        PackedTensor b3, tl;
        CHECK(pack_b3b(t, b3) == Status::Ok && validate(b3) == Status::Ok, "b3b pack %llu", (unsigned long long)cols);
        CHECK(pack_tl3(t, tl) == Status::Ok && validate(tl) == Status::Ok, "tl3 pack %llu", (unsigned long long)cols);
        CHECK(b3.codes.size() == rows * cols / 128 * 26, "b3b size");
        CHECK(tl.codes.size() == rows / 32 * cols / 128 * 860, "tl3 size");
        std::vector<int8_t> u(static_cast<size_t>(cols)), u2(static_cast<size_t>(cols));
        bool ok_b = true, ok_t = true, ok_sb = true, ok_si = true;
        for (uint64_t row = 0; row < rows; ++row) {
            const int8_t* want = q.data() + row * cols;
            if (unpack_row(b3, row, u.data()) != Status::Ok || std::memcmp(u.data(), want, size_t(cols))) ok_b = false;
            if (unpack_row(tl, row, u.data()) != Status::Ok || std::memcmp(u.data(), want, size_t(cols))) ok_t = false;
            if (kernel_available(Kernel::B3BAvx2)) {
                std::memset(u2.data(), 7, u2.size());
                if (decode_row_b3b_avx2(b3, row, u2.data()) != Status::Ok || std::memcmp(u2.data(), want, size_t(cols))) ok_sb = false;
                View v; view_of(t, v);
                std::memset(u2.data(), 7, u2.size());
                if (decode_row_i2_avx2(v, row, u2.data()) != Status::Ok || std::memcmp(u2.data(), want, size_t(cols))) ok_si = false;
            }
        }
        CHECK(ok_b, "b3b round trip"); CHECK(ok_t, "tl3 round trip");
        CHECK(ok_sb, "b3b AVX2 decode"); CHECK(ok_si, "i2 AVX2 decode");
        // Strict validation: every kind of bad code is refused.
        { PackedTensor bad = b3; bad.codes[3] = 243; CHECK(validate(bad) == Status::BadCode, "b3b byte 243"); }
        { PackedTensor bad = b3; bad.codes[24] = 0; CHECK(validate(bad) == Status::BadCode, "b3b pad weight non-zero"); }
        { PackedTensor bad = b3; bad.codes[25] = uint8_t(bad.codes[25] + 81); CHECK(validate(bad) == Status::BadCode, "b3b pad weight +1"); }
        { PackedTensor bad = b3; bad.codes.pop_back(); CHECK(validate(bad) == Status::BadArgument, "b3b short"); }
        { PackedTensor bad = b3; bad.scales[0] = -1.0f; CHECK(validate(bad) == Status::BadScale, "b3b negative scale"); }
        { PackedTensor bad = tl; bad.codes[0] = uint8_t((bad.codes[0] & 0xF0) | 14); CHECK(validate(bad) == Status::BadCode, "tl3 index 14"); }
        { PackedTensor bad = tl; bad.codes[0] = uint8_t((bad.codes[0] & 0xF0) | 13); bad.codes[16] |= 1;
          CHECK(validate(bad) == Status::BadCode, "tl3 negative zero"); }
        { PackedTensor bad = tl; const size_t off = size_t(42 * 20); bad.codes[off] = uint8_t((bad.codes[off] & 0xF0) | 0);
          CHECK(validate(bad) == Status::BadCode, "tl3 pad weight non-zero"); }
        { PackedTensor bad = tl; bad.rows = 48; CHECK(validate(bad) != Status::Ok, "tl3 rows not tile multiple"); }
    }
    { std::mt19937 r2(3); auto q = rand_trits(r2, 40 * 128, 34); Tensor t = make_i2(q, 40, 128, ScaleMode::PerTensor, r2);
      PackedTensor tl; CHECK(pack_tl3(t, tl) == Status::BadArgument, "tl3 refuses 40 rows"); }
}

static void test_kernels_random() {
    std::mt19937 r(5);
    const uint64_t widths[] = {128, 256, 384, 2560, 4096, 6912, 16384};
    for (uint64_t cols : widths)
        for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock})
            for (int zp : {0, 34, 90}) {
                const uint64_t rows = cols >= 6912 ? 32 : 64;
                auto q = rand_trits(r, size_t(rows * cols), zp);
                Tensor t = make_i2(q, rows, cols, m, r);
                ActRow x = rand_act(r, cols);
                char what[96]; std::snprintf(what, sizeof what, "random %llu mode %d zeros %d%%", (unsigned long long)cols, int(m), zp);
                all_kernels_match(t, x, what);
            }
    // A row count that is not a tile multiple still runs on the I2 and B3B kernels.
    { auto q = rand_trits(r, 5 * 256, 34); Tensor t = make_i2(q, 5, 256, ScaleMode::PerBlock, r);
      all_kernels_match(t, rand_act(r, 256), "5 rows"); }
}

static void test_kernels_adversarial() {
    std::mt19937 r(9);
    const uint64_t rows = 32, cols = 4096;
    // 1. Outlier channels inside blocks: one huge activation per block at the edges and the middle.
    for (int pos : {0, 63, 127}) {
        std::vector<float> xf(static_cast<size_t>(cols), 0.01f);
        std::normal_distribution<float> nd(0.0f, 0.01f);
        for (auto& v : xf) v = nd(r);
        for (uint64_t b = 0; b < cols / 128; ++b) xf[size_t(b * 128 + uint64_t(pos))] = (b % 2 ? -1.0f : 1.0f) * 1000.0f;
        ActRow x = act_from(xf);
        auto q = rand_trits(r, size_t(rows * cols), 34);
        char what[64]; std::snprintf(what, sizeof what, "outlier at %d", pos);
        all_kernels_match(make_i2(q, rows, cols, ScaleMode::PerBlock, r), x, what);
        all_kernels_match(make_i2(q, rows, cols, ScaleMode::PerTensor, r), x, what);
    }
    // 2. Alternating +1 / -1 weights against constant, alternating and maximal activations.
    {
        std::vector<int8_t> q(size_t(rows * cols));
        for (size_t i = 0; i < q.size(); ++i) q[i] = (i % 2) ? -1 : 1;
        std::vector<float> c(static_cast<size_t>(cols), 1.0f), alt(static_cast<size_t>(cols)), mx(static_cast<size_t>(cols));
        for (size_t i = 0; i < alt.size(); ++i) { alt[i] = (i % 2) ? -1.0f : 1.0f; mx[i] = (i % 2) ? -3e38f : 3e38f; }
        for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock}) {
            Tensor t = make_i2(q, rows, cols, m, r);
            all_kernels_match(t, act_from(c), "alternating vs constant");
            all_kernels_match(t, act_from(alt), "alternating vs alternating");
        }
        // Maximal integer sums: every block sums to +-16256 (alternating x alternating = all +127 products).
        Tensor t = make_i2(q, rows, cols, ScaleMode::PerTensor, r);
        View v; view_of(t, v);
        ActRow xa = act_from(alt);
        std::vector<float> y(rows);
        matvec_i2(v, xa, Kernel::I2Scalar, y.data(), rows);
        const float expect = (float(int32_t(127 * cols)) * t.scales[0]) * xa.scale;
        CHECK(y[0] == expect, "maximal sum exact (%g vs %g)", double(y[0]), double(expect));
    }
    // 3. All +1 and all -1 at maximal activations.
    for (int s : {1, -1}) {
        std::vector<int8_t> q(size_t(rows * cols), int8_t(s));
        std::vector<float> xf(static_cast<size_t>(cols), 5.0f);
        all_kernels_match(make_i2(q, rows, cols, ScaleMode::PerBlock, r), act_from(xf), s > 0 ? "all +1" : "all -1");
    }
    // 4. Extreme gammas: smallest normal, mixed magnitudes within a row, and so large the output overflows.
    {
        auto q = rand_trits(r, size_t(rows * cols), 34);
        Tensor t = make_i2(q, rows, cols, ScaleMode::PerBlock, r);
        for (size_t i = 0; i < t.scales.size(); ++i) t.scales[i] = (i % 3 == 0) ? FLT_MIN : (i % 3 == 1 ? 1e30f : 1.0f);
        ActRow x = rand_act(r, cols);
        all_kernels_match(t, x, "mixed extreme gammas");
        for (auto& s : t.scales) s = FLT_MIN;
        all_kernels_match(t, x, "all FLT_MIN gammas");
        for (auto& s : t.scales) s = 3e38f;
        std::vector<float> xf(static_cast<size_t>(cols), 1e30f);
        ActRow xb = act_from(xf);
        all_kernels_match(t, xb, "overflowing gammas");
        View v; view_of(t, v);
        std::vector<float> y(rows, 1.0f);
        const Status st = matvec_i2(v, xb, Kernel::I2Scalar, y.data(), rows);
        bool zeros = true; for (float f : y) if (f != 0.0f) zeros = false;
        CHECK(st == Status::NonFinite && zeros, "overflow reported and zeroed (%s)", status_name(st));
    }
    // 5. Sparse: all-zero blocks (the zero-skip case) mixed with dense ones.
    {
        auto q = rand_trits(r, size_t(rows * cols), 34);
        for (uint64_t row = 0; row < rows; ++row)
            for (uint64_t b = 0; b < cols / 128; b += 2) std::memset(q.data() + row * cols + b * 128, 0, 128);
        all_kernels_match(make_i2(q, rows, cols, ScaleMode::PerBlock, r), rand_act(r, cols), "half the blocks zero");
    }
    // 6. Shape mismatch fails safe.
    {
        auto q = rand_trits(r, size_t(rows * 256), 34);
        Tensor t = make_i2(q, rows, 256, ScaleMode::PerBlock, r);
        View v; view_of(t, v);
        ActRow x = rand_act(r, 512);
        std::vector<float> y(rows, 1.0f);
        CHECK(matvec_i2(v, x, Kernel::I2Scalar, y.data(), rows) == Status::BadArgument && y[0] == 0.0f, "i2 mismatch");
        PackedTensor b3; pack_b3b(t, b3); B3BAct xb; prepare_b3b(x, xb);
        y.assign(rows, 1.0f);
        CHECK(matvec_b3b(b3, x, xb, Kernel::B3BScalar, y.data(), rows) == Status::BadArgument && y[0] == 0.0f, "b3b mismatch");
        PackedTensor tl; pack_tl3(t, tl); TL3Act xt; prepare_tl3(x, xt);
        y.assign(rows, 1.0f);
        CHECK(matvec_tl3(tl, x, xt, Kernel::TL3Scalar, y.data(), rows) == Status::BadArgument && y[0] == 0.0f, "tl3 mismatch");
    }
}

static void test_softmax() {
    // Fraction methods, exhaustive over all 65536 inputs.
    const double limit[3] = {1.4e-3, 3.0e-5, 8.0e-5};
    for (Exp2 m : {Exp2::Table256, Exp2::Table64Lerp, Exp2::Poly3}) {
        double worst = 0; bool mono = true; uint32_t prev = 0;
        for (uint32_t f = 0; f < 65536; ++f) {
            const uint32_t v = exp2_frac_q30(f, m);
            const double e = std::fabs(double(v) / 1073741824.0 / std::exp2(double(f) / 65536.0) - 1.0);
            worst = std::max(worst, e);
            if (m != Exp2::Table256 && v < prev) mono = false;
            prev = v;
        }
        CHECK(worst < limit[int(m)], "%s max rel error %g", exp2_name(m), worst);
        CHECK(mono, "%s monotone", exp2_name(m));
    }
    std::mt19937 r(13);
    for (size_t n : {1ul, 7ul, 128ul, 4096ul}) {
        std::normal_distribution<double> nd(0.0, 4.0);
        std::vector<int32_t> z(n); for (auto& v : z) v = int32_t(std::lround(nd(r) * 65536.0));
        std::vector<uint32_t> p(n);
        for (Exp2 m : {Exp2::Table256, Exp2::Table64Lerp, Exp2::Poly3}) {
            CHECK(softmax_q16(z.data(), n, m, p.data()) == Status::Ok, "softmax ok");
            int64_t s = 0; for (auto v : p) s += v;
            CHECK(std::llabs(s - 65536) <= int64_t(n / 2 + 1), "softmax sum %lld n %zu", (long long)s, n);
        }
    }
    // Extremes: huge spread, equal logits, INT32 range.
    { std::vector<int32_t> z = {INT32_MAX, INT32_MIN, 0, INT32_MAX}; std::vector<uint32_t> p(4);
      CHECK(softmax_q16(z.data(), 4, Exp2::Poly3, p.data()) == Status::Ok && p[0] == 32768 && p[3] == 32768 && p[1] == 0 && p[2] == 0,
            "softmax int32 extremes"); }
    { std::vector<int32_t> z(1000, 12345); std::vector<uint32_t> p(1000);
      softmax_q16(z.data(), z.size(), Exp2::Table64Lerp, p.data());
      bool eq = true; for (auto v : p) if (v != p[0]) eq = false;
      CHECK(eq && (p[0] == 66 || p[0] == 65), "softmax uniform"); }
    { std::vector<int32_t> z(4096); std::mt19937 r2(19); std::normal_distribution<double> nd(0.0, 3.0);
      for (auto& v : z) v = int32_t(std::lround(nd(r2) * 65536.0));
      std::vector<uint32_t> p(z.size());
      for (int ob : {8, 24, 30}) {
          CHECK(softmax_fixed(z.data(), z.size(), Exp2::Table64Lerp, ob, p.data()) == Status::Ok, "softmax Q%d", ob);
          int64_t s = 0; for (auto v : p) s += v;
          CHECK(std::llabs(s - (int64_t(1) << ob)) <= int64_t(z.size() / 2 + 1), "softmax Q%d sum", ob);
      }
      CHECK(softmax_fixed(z.data(), z.size(), Exp2::Poly3, 31, p.data()) == Status::BadArgument && p[0] == 0, "out_bits 31 refused");
      CHECK(softmax_fixed(z.data(), z.size(), Exp2::Poly3, 7, p.data()) == Status::BadArgument, "out_bits 7 refused"); }
    { uint32_t p[2] = {1, 1}; int32_t z[2] = {0, 0};
      CHECK(softmax_q16(nullptr, 2, Exp2::Poly3, p) == Status::BadArgument && p[0] == 0, "softmax null");
      CHECK(softmax_q16(z, 0, Exp2::Poly3, p) == Status::BadArgument, "softmax n = 0"); }
}

static void test_rmsnorm() {
    // rsqrt over the whole mantissa range, sampled densely.
    double worst[4] = {0, 0, 0, 0};
    for (uint64_t u = 1ull << 30; u < (1ull << 32); u += 4099)
        for (int s = 0; s < 3; ++s) {
            const double e = std::fabs(double(rsqrt_q30(uint32_t(u), s)) / 1073741824.0 * std::sqrt(double(u) / 1073741824.0) - 1.0);
            worst[s] = std::max(worst[s], e);
        }
    CHECK(worst[0] < 1.0e-2 && worst[1] < 1.0e-4 && worst[2] < 2.0e-8, "rsqrt steps %g %g %g", worst[0], worst[1], worst[2]);
    // Bounds: maximal row (n = 2^20, all 32767), one spike, all zero, minimal eps.
    {
        std::vector<int16_t> x(kMaxNorm, 32767); std::vector<int32_t> y(kMaxNorm);
        CHECK(rmsnorm_q16(x.data(), x.size(), 1, y.data()) == Status::Ok, "max row ok");
        CHECK(std::abs(y[0] - 65536) <= 2, "max row normalizes to 1.0 (%d)", y[0]);
        std::fill(x.begin(), x.end(), int16_t(0)); x[5] = -32767;
        CHECK(rmsnorm_q16(x.data(), x.size(), 1, y.data()) == Status::Ok, "spike ok");
        const double want = -32767.0 / std::sqrt(32767.0 * 32767.0 / double(kMaxNorm) + 1.0 / 256.0) * 65536.0;
        CHECK(std::fabs(y[5] - want) <= std::fabs(want) * 1e-7 + 1, "spike value %d vs %g", y[5], want);
        std::fill(x.begin(), x.end(), int16_t(0));
        CHECK(rmsnorm_q16(x.data(), x.size(), 1, y.data()) == Status::Ok && y[0] == 0, "zero row");
    }
    { int16_t x[3] = {1, INT16_MIN, 2}; int32_t y[3] = {9, 9, 9};
      CHECK(rmsnorm_q16(x, 3, 256, y) == Status::BadArgument && y[0] == 0 && y[2] == 0, "-32768 refused, zeroed");
      CHECK(rmsnorm_q16(x, 0, 256, y) == Status::BadArgument, "n = 0");
      CHECK(rmsnorm_q16(x, 1, 0, y) == Status::BadArgument, "eps = 0");
      CHECK(rmsnorm_q16(x, kMaxNorm + 1, 256, y) == Status::BadArgument, "n too large"); }
    // Random rows against double.
    std::mt19937 r(17);
    double worst_rel = 0; int bad = 0;
    for (int trial = 0; trial < 50; ++trial) {
        const size_t n = 2560;
        std::normal_distribution<double> nd(0.0, std::exp2(double(trial % 14)));
        std::vector<int16_t> x(n); for (auto& v : x) v = int16_t(std::max(-32767.0, std::min(32767.0, std::round(nd(r)))));
        std::vector<int32_t> y(n);
        rmsnorm_q16(x.data(), n, 1, y.data());
        double ss = 0; for (auto v : x) ss += double(v) * v;
        const double rs = 1.0 / std::sqrt(ss / double(n) + 1.0 / 256.0);
        for (size_t i = 0; i < n; ++i) {
            const double want = double(x[i]) * rs * 65536.0;
            const double err = std::fabs(y[i] - want);
            if (err > 1.0 + 5e-8 * std::fabs(want)) ++bad;
            if (std::fabs(want) > 65536) worst_rel = std::max(worst_rel, err / std::fabs(want));
        }
    }
    CHECK(bad == 0, "rmsnorm vs double: %d outside 1 unit + 5e-8 relative (worst rel above 1.0: %g)", bad, worst_rel);
}

int main() {
    std::printf("AVX2 kernels %s\n", kernel_available(Kernel::I2Avx2Signed) ? "available" : "not available (scalar only)");
    test_helpers();
    test_converter();
    test_packings();
    test_kernels_random();
    test_kernels_adversarial();
    test_softmax();
    test_rmsnorm();
    std::printf("%d passed, %d failed\n", passes, fails);
    return fails ? 1 : 0;
}
