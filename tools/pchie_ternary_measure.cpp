// Bolt PCHIE - Archie's ternary measurements (Nov 16, 2026). Prints markdown tables.
// Usage: pchie_ternary_measure <scales|accum|pack|softmax|norm|all> [threads for the multi-thread GEMV, default 4]
// Toy and synthetic weights only. Random numbers come from mt19937 with our own Gaussian / Laplacian transforms,
// so the inputs are the same on every compiler.
#include "pchie/pchie_ternary.h"
#include "pchie/pchie_ternary_research.h"
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
using namespace pchie::ternary;
using namespace pchie::ternary::research;

// ------------------------------------------------------------------------------------------------ inputs
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    double u() { return (double(g() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }   // (0, 1)
    double gauss() { const double a = u(), b = u(); return std::sqrt(-2.0 * std::log(a)) * std::cos(6.283185307179586 * b); }
    double laplace() { const double a = u() - 0.5; return (a < 0 ? 1.0 : -1.0) * std::log(1.0 - 2.0 * std::fabs(a)) / std::sqrt(2.0); }   // unit variance
    double student4() { double s = 0; for (int i = 0; i < 4; ++i) { const double z = gauss(); s += z * z; } return gauss() / std::sqrt(s / 4.0) / std::sqrt(2.0); }
};
enum class Dist { Gauss, Laplace, Student4, Q4Grid };
static const char* dist_name(Dist d) {
    switch (d) { case Dist::Gauss: return "Gaussian"; case Dist::Laplace: return "Laplacian"; case Dist::Student4: return "Student-t (4 dof)"; case Dist::Q4Grid: return "Q4-style grid"; }
    return "?";
}
// Matrix with optional row-scale and block-scale heterogeneity (log-normal) and outlier columns.
static std::vector<float> make_matrix(Rng& r, uint64_t rows, uint64_t cols, Dist d, double row_sigma, double blk_sigma,
                                      int outlier_cols = 0) {
    std::vector<float> W(static_cast<size_t>(rows * cols));
    std::vector<double> colscale(size_t(cols), 1.0);
    for (int i = 0; i < outlier_cols; ++i) colscale[size_t(r.g() % cols)] = 6.0;
    for (uint64_t i = 0; i < rows; ++i) {
        const double rs = 0.02 * std::exp(row_sigma * r.gauss());
        for (uint64_t b = 0; b < cols / 128; ++b) {
            const double bs = rs * std::exp(blk_sigma * r.gauss());
            if (d == Dist::Q4Grid) {   // like a dequantized Q4 tensor: 32-weight groups on a 16-level grid
                for (int g0 = 0; g0 < 128; g0 += 32) {
                    const double dd = bs * 2.0 / 7.0;
                    for (int k = 0; k < 32; ++k) {
                        long v = std::lround(r.gauss() * 3.5); v = std::max(-8L, std::min(7L, v));
                        W[size_t(i * cols + b * 128 + uint64_t(g0 + k))] = float(dd * double(v) * colscale[size_t(b * 128 + uint64_t(g0 + k))]);
                    }
                }
                continue;
            }
            for (int k = 0; k < 128; ++k) {
                const double z = d == Dist::Gauss ? r.gauss() : d == Dist::Laplace ? r.laplace() : r.student4();
                W[size_t(i * cols + b * 128 + uint64_t(k))] = float(bs * z * colscale[size_t(b * 128 + uint64_t(k))]);
            }
        }
    }
    return W;
}
static std::vector<float> make_act(Rng& r, uint64_t n, int outliers = 0, double outlier_mag = 20.0) {
    std::vector<float> x(static_cast<size_t>(n));
    for (auto& v : x) v = float(r.gauss());
    for (int i = 0; i < outliers; ++i) x[size_t(r.g() % n)] = float(outlier_mag * (r.g() & 1 ? 1 : -1));
    return x;
}
static std::vector<int8_t> unpack_all(const Tensor& t) {
    std::vector<int8_t> q(static_cast<size_t>(t.rows * t.cols));
    for (uint64_t r = 0; r < t.rows; ++r) pchie::ternary::unpack_row(t.codes.data() + r * t.row_bytes(), t.cols, t.packing, q.data() + r * t.cols);
    return q;
}
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class F> static double median_ms(int iters, F&& f) {
    std::vector<double> v; f();   // warm-up
    for (int i = 0; i < iters; ++i) { const double a = now_ms(); f(); v.push_back(now_ms() - a); }
    std::sort(v.begin(), v.end()); return v[v.size() / 2];
}

// ------------------------------------------------------------------------------------------------ M1 scales
struct GroupStats { double err = 0, energy = 0, zeros = 0, n = 0; };
static void fit_groups(const std::vector<float>& W, uint64_t rows, uint64_t cols, uint64_t gcols, Fit fit, Round rnd,
                       GroupStats& st, std::vector<float>* deq = nullptr) {
    std::vector<float> scratch(static_cast<size_t>(rows * gcols)), grp(size_t(rows * gcols));
    std::vector<int8_t> q(static_cast<size_t>(rows * gcols));
    // gcols == cols with rows grouped: per-row. gcols == 0 means per tensor.
    auto do_group = [&](const float* src, size_t n, uint64_t seed, float* out) {
        float g; ternarize_group(src, n, fit, rnd, seed, 8, scratch.data(), q.data(), &g);
        for (size_t i = 0; i < n; ++i) {
            const double wq = double(g) * q[i], d = double(src[i]) - wq;
            st.err += d * d; st.energy += double(src[i]) * src[i]; st.zeros += q[i] == 0; st.n += 1;
            if (out) out[i] = float(wq);
        }
    };
    if (gcols == 0) {
        scratch.resize(W.size()); q.resize(W.size());
        do_group(W.data(), W.size(), 1, deq ? deq->data() : nullptr);
        return;
    }
    for (uint64_t r = 0; r < rows; ++r)
        for (uint64_t c0 = 0; c0 < cols; c0 += gcols)
            do_group(W.data() + r * cols + c0, size_t(gcols), r * 7919 + c0, deq ? deq->data() + r * cols + c0 : nullptr);
}
static double db(double rel) { return -10.0 * std::log10(rel); }

static void section_scales() {
    std::printf("\n## M1. Scales\n\n### M1.1 Absmean vs fitted scale, one group of 128 weights (2,048 groups per row)\n\n");
    std::printf("| Distribution | Fit | Relative MSE | SQNR (dB) | Zeros | Gain over absmean (dB) |\n|---|---|---|---|---|---|\n");
    for (Dist d : {Dist::Gauss, Dist::Laplace, Dist::Student4, Dist::Q4Grid}) {
        Rng r(100 + int(d));
        auto W = make_matrix(r, 2048, 128, d, 0.0, 0.0);
        double base = 0;
        for (Fit f : {Fit::Absmean, Fit::LeastSquares, Fit::Lloyd}) {
            GroupStats st; fit_groups(W, 2048, 128, 128, f, Round::HalfEven, st);
            const double rel = st.err / st.energy;
            if (f == Fit::Absmean) base = db(rel);
            std::printf("| %s | %s | %.4f | %.2f | %.1f%% | %+.2f |\n", dist_name(d),
                        f == Fit::Absmean ? "absmean" : f == Fit::LeastSquares ? "least squares (exact, sort)" : "Lloyd, 8 steps",
                        rel, db(rel), 100.0 * st.zeros / st.n, db(rel) - base);
        }
    }
    // Speed of the fits (converter-side cost).
    {
        Rng r(7); auto W = make_matrix(r, 8192, 128, Dist::Gauss, 0, 0);
        std::printf("\nConverter cost per weight (one thread): ");
        for (Fit f : {Fit::Absmean, Fit::LeastSquares, Fit::Lloyd}) {
            GroupStats st;
            const double ms = median_ms(3, [&] { st = GroupStats(); fit_groups(W, 8192, 128, 128, f, Round::HalfEven, st); });
            std::printf("%s %.1f ns%s", f == Fit::Absmean ? "absmean" : f == Fit::LeastSquares ? "least squares" : "Lloyd",
                        ms * 1e6 / double(W.size()), f == Fit::Lloyd ? ".\n" : ", ");
        }
    }

    std::printf("\n### M1.2 Scale granularity, error per byte (1,024 x 2,560, absmean and least squares)\n\n");
    std::printf("Output error is ||y - W x|| / ||W x|| through the shipping int8-activation path (per-tensor and per-block) "
                "or the same math in double for per-row (not a shipping format).\n\n");
    std::printf("| Weights | Granularity | Fit | Scale cost (bpw) | I2 total bpw | B3 total bpw | Weight SQNR (dB) | Output error |\n|---|---|---|---|---|---|---|---|\n");
    struct Case { const char* name; Dist d; double rs, bs; int oc; };
    const Case cases[] = {{"Gaussian, uniform scale", Dist::Gauss, 0, 0, 0},
                          {"Gaussian, row scales vary (sigma 0.5)", Dist::Gauss, 0.5, 0, 0},
                          {"Gaussian, row + block scales vary, 16 outlier columns", Dist::Gauss, 0.5, 0.4, 16},
                          {"Laplacian, row + block scales vary", Dist::Laplace, 0.5, 0.4, 0}};
    const uint64_t rows = 1024, cols = 2560;
    for (const Case& c : cases) {
        Rng r(200);
        auto W = make_matrix(r, rows, cols, c.d, c.rs, c.bs, c.oc);
        auto xf = make_act(r, cols);
        ActRow x; quantize_activations(xf.data(), cols, x);
        std::vector<double> yref(rows);
        for (uint64_t i = 0; i < rows; ++i) { double s = 0; for (uint64_t k = 0; k < cols; ++k) s += double(W[size_t(i * cols + k)]) * xf[size_t(k)]; yref[size_t(i)] = s; }
        auto out_err = [&](const std::vector<double>& y) {
            double e = 0, n = 0; for (uint64_t i = 0; i < rows; ++i) { e += (y[size_t(i)] - yref[size_t(i)]) * (y[size_t(i)] - yref[size_t(i)]); n += yref[size_t(i)] * yref[size_t(i)]; }
            return std::sqrt(e / n);
        };
        for (int gran = 0; gran < 3; ++gran)
            for (Fit f : {Fit::Absmean, Fit::LeastSquares}) {
                GroupStats st; std::vector<float> deq(W.size());
                fit_groups(W, rows, cols, gran == 0 ? 0 : gran == 1 ? cols : 128, f, Round::HalfEven, st, &deq);
                std::vector<double> y(rows);
                if (gran == 1) {   // per-row: double math on the dequantized weights and the int8 activations
                    for (uint64_t i = 0; i < rows; ++i) { double s = 0; for (uint64_t k = 0; k < cols; ++k) s += double(deq[size_t(i * cols + k)]) * double(x.q[size_t(k)]); y[size_t(i)] = s * x.scale; }
                } else {
                    // Rebuild a shipping tensor from the fitted codes and scales.
                    Tensor t; t.packing = Packing::I2; t.scale_mode = gran == 0 ? ScaleMode::PerTensor : ScaleMode::PerBlock;
                    t.rows = rows; t.cols = cols; t.codes.resize(size_t(rows * cols / 4));
                    t.scales.assign(gran == 0 ? 1 : size_t(rows * cols / 128), 0.0f);
                    std::vector<int8_t> qrow(cols);
                    for (uint64_t i = 0; i < rows; ++i) {
                        for (uint64_t k = 0; k < cols; ++k) {
                            const float v = deq[size_t(i * cols + k)];
                            qrow[size_t(k)] = v > 0 ? 1 : (v < 0 ? -1 : 0);
                            if (v != 0) t.scales[gran == 0 ? 0 : size_t(i * (cols / 128) + k / 128)] = std::fabs(v);
                        }
                        pack_row(qrow.data(), cols, Packing::I2, t.codes.data() + i * cols / 4);
                    }
                    View v; view_of(t, v); std::vector<float> yf(rows);
                    matvec(v, x, yf.data(), rows, Tier::Scalar, nullptr);
                    for (uint64_t i = 0; i < rows; ++i) y[size_t(i)] = yf[size_t(i)];
                }
                const double sc = gran == 0 ? 32.0 / double(rows * cols) : gran == 1 ? 32.0 / double(cols) : 0.25;
                std::printf("| %s | %s | %s | %.4f | %.3f | %.3f | %.2f | %.4f |\n", c.name,
                            gran == 0 ? "per tensor" : gran == 1 ? "per row" : "per 128 block",
                            f == Fit::Absmean ? "absmean" : "least squares", sc, 2.0 + sc, 1.6 + sc, db(st.err / st.energy), out_err(y));
            }
    }

    std::printf("\n### M1.3 Conversion precision: absmean in double, stored as float32, and the W / gamma comparison\n\n");
    {
        // (a) float vs double accumulation of a per-tensor absmean on a large tensor.
        Rng r(300);
        const uint64_t rows = 6912, cols = 2560;
        auto W = make_matrix(r, rows, cols, Dist::Gauss, 0.5, 0.0);
        double sd = 0; float sf = 0; for (float w : W) { sd += std::fabs(double(w)); sf += std::fabs(w); }
        const float gd = float(sd / double(W.size())), gf = sf / float(W.size());
        uint64_t flips = 0;
        for (float w : W) {
            const double td = double(w) / double(gd), tf = double(w) / double(gf);
            const int a = td > 0.5 ? 1 : (td < -0.5 ? -1 : 0), b = tf > 0.5 ? 1 : (tf < -0.5 ? -1 : 0);
            flips += a != b;
        }
        std::printf("Per-tensor absmean over %llu weights: double sum gives gamma %.9g, a float32 running sum gives %.9g "
                    "(relative difference %.2e), which changes %llu codes (%.4f%%).\n\n",
                    (unsigned long long)W.size(), double(gd), double(gf), std::fabs(double(gf) / double(gd) - 1.0),
                    (unsigned long long)flips, 100.0 * double(flips) / double(W.size()));
        // (b) the comparison itself, on random weights and on weights placed at the +-0.5 gamma boundary.
        std::printf("| Comparison | Code changes vs exact, 17.7M random weights | Code changes vs exact, 1M boundary weights |\n|---|---|---|\n");
        std::vector<float> wb; std::vector<float> gb;
        for (int i = 0; i < 1000000; ++i) {
            const float g = float(0.001 + 0.05 * r.u());
            float h = 0.5f * g;
            const int k = int(r.g() % 5) - 2;
            for (int s = 0; s < std::abs(k); ++s) h = k > 0 ? std::nextafter(h, 1.0f) : std::nextafter(h, 0.0f);
            wb.push_back(r.g() & 1 ? h : -h); gb.push_back(g);
        }
        auto exact = [](float w, float g) { const float h = 0.5f * g; return w > h ? 1 : (w < -h ? -1 : 0); };   // 0.5*g exact for normal g
        const char* names[4] = {"double division (shipping)", "float division", "float multiply by 1/gamma", "compare |w| with 0.5f * gamma (exact)"};
        for (int m = 0; m < 4; ++m) {
            auto code = [&](float w, float g) {
                double t;
                if (m == 0) t = double(w) / double(g);
                else if (m == 1) t = double(w / g);
                else if (m == 2) { const float ig = 1.0f / g; t = double(w * ig); }
                else return exact(w, g);
                return t > 0.5 ? 1 : (t < -0.5 ? -1 : 0);
            };
            uint64_t c1 = 0, c2 = 0;
            for (float w : W) c1 += code(w, gd) != exact(w, gd);
            for (size_t i = 0; i < wb.size(); ++i) c2 += code(wb[i], gb[i]) != exact(wb[i], gb[i]);
            std::printf("| %s | %llu | %llu |\n", names[m], (unsigned long long)c1, (unsigned long long)c2);
        }
        // (c) edge cases through the shipping quantizer and matvec.
        std::printf("\nEdge cases through the shipping quantizer and matvec (one 32 x 128 tensor, per-block scales):\n\n");
        std::printf("| Case | Stored gamma | Codes | Shipping matvec, Gaussian activations | Shipping matvec, flat activations (xq = 127) | Research quantizer and matvec_i2 |\n|---|---|---|---|---|---|\n");
        struct Edge { const char* name; float scale; };
        const Edge edges[] = {{"tiny weights (1e-41, subnormal gamma)", 1e-41f}, {"weights near FLT_MIN (1.2e-38)", 1.2e-38f},
                              {"ordinary (0.02)", 0.02f}, {"huge weights (1e36)", 1e36f}, {"weights near FLT_MAX (2e37, peaks about 8e37)", 2e37f}};
        for (const Edge& e : edges) {
            Rng re(5);
            std::vector<float> w(32 * 128); for (auto& v : w) v = float(re.gauss() * double(e.scale));
            Tensor t; Status st = quantize_absmean(w.data(), 32, 128, ScaleMode::PerBlock, Packing::I2, t);
            auto xf = make_act(re, 128, 1, 40.0); ActRow x; quantize_activations(xf.data(), 128, x);
            std::vector<float> xflat(128, 1.0f); ActRow x2; quantize_activations(xflat.data(), 128, x2);
            View v; const Status sv = view_of(t, v);
            auto run = [&](const ActRow& xa) {
                std::vector<float> y(32, 0.0f); Status sm = Status::BadArgument;
                if (st == Status::Ok && sv == Status::Ok) sm = matvec(v, xa, y.data(), 32, Tier::Scalar, nullptr);
                int nonfinite = 0, subn = 0; for (float f : y) { nonfinite += !std::isfinite(f); subn += (f != 0 && std::fabs(f) < FLT_MIN); }
                char buf[128];
                if (st != Status::Ok) std::snprintf(buf, sizeof buf, "quantizer refused (%s)", status_name(st));
                else std::snprintf(buf, sizeof buf, "%s; %d of 32 non-finite, %d subnormal", status_name(sm), nonfinite, subn);
                return std::string(buf);
            };
            int nz = 0; if (st == Status::Ok) { auto q = unpack_all(t); for (auto c : q) nz += c != 0; }
            // Research path: the same block through ternarize_group (subnormal flush), then matvec_i2 (non-finite check).
            float g2; std::vector<int8_t> q2(128); float scr[128];
            ternarize_group(w.data(), 128, Fit::Absmean, Round::HalfEven, 0, 0, scr, q2.data(), &g2);
            Tensor t2; t2.packing = Packing::I2; t2.scale_mode = ScaleMode::PerTensor; t2.rows = 1; t2.cols = 128;
            t2.codes.resize(32); pack_row(q2.data(), 128, Packing::I2, t2.codes.data()); t2.scales = {g2};
            View v2; view_of(t2, v2); float y2 = 1.0f;
            const Status s2 = matvec_i2(v2, x2, Kernel::I2Scalar, &y2, 1);
            int nz2 = 0; for (auto c : q2) nz2 += c != 0;
            std::printf("| %s | %.3g%s | %d of 4096 non-zero | %s | %s | gamma %.3g, %d of 128 non-zero; flat activations: %s, y = %g |\n", e.name,
                        st == Status::Ok ? double(t.scales[0]) : 0.0,
                        (st == Status::Ok && t.scales[0] != 0 && t.scales[0] < FLT_MIN) ? " (subnormal)" : "",
                        nz, run(x).c_str(), run(x2).c_str(), double(g2), nz2, status_name(s2), double(y2));
        }
    }
}

// ------------------------------------------------------------------------------------------------ M2 accumulation
static double neumaier(const std::vector<double>& v) {
    double s = 0, c = 0;
    for (double x : v) { const double t = s + x; c += std::fabs(s) >= std::fabs(x) ? (s - t) + x : (x - t) + s; s = t; }
    return s + c;
}
static double ulp_of(float f) { const float a = std::fabs(f); return double(std::nextafter(a, FLT_MAX) - a); }

static void section_accum() {
    std::printf("\n## M2. Accumulation and rounding\n\n### M2.1 Float tail versus higher precision, per-128-block scales\n\n");
    std::printf("Reference: exact integer block sums, each times its gamma (exact in double), summed with Neumaier "
                "compensation, times the activation scale. Errors in units in the last place of the float32 result "
                "(ulp), 512 rows per width, Gaussian weights with row and block scale variation; 'outliers' adds 8 activation "
                "channels at 20 sigma.\n\n");
    std::printf("| Width | Activations | Shipping: int32 blocks + float tail, max / mean ulp | same, max error / (sum of block magnitudes x 2^-24) | Double tail, max / mean ulp | Float32 dequantized dot, max / mean ulp | same, max normalized | Float32 sum of integer products exact? | Quantization error, relative to ||y|| |\n|---|---|---|---|---|---|---|---|---|\n");
    for (uint64_t cols : {2560ull, 4096ull, 6912ull, 8192ull, 16384ull})
        for (int outl : {0, 8}) {
            Rng r(400 + cols + uint64_t(outl));
            const uint64_t rows = 512;
            auto W = make_matrix(r, rows, cols, Dist::Gauss, 0.5, 0.4);
            Tensor t; quantize_absmean(W.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, t);
            auto xf = make_act(r, cols, outl);
            ActRow x; quantize_activations(xf.data(), cols, x);
            View v; view_of(t, v);
            std::vector<float> y(rows); matvec(v, x, y.data(), rows, Tier::Scalar, nullptr);
            auto q = unpack_all(t);
            const uint64_t nb = cols / 128;
            double mx[3] = {0, 0, 0}, mean[3] = {0, 0, 0}, nrm[3] = {0, 0, 0}; bool int_exact = true; double qerr = 0, qn = 0;
            for (uint64_t i = 0; i < rows; ++i) {
                std::vector<double> parts(static_cast<size_t>(nb)); double dacc = 0; float facc_deq = 0; float fint = 0; int64_t iexact = 0;
                for (uint64_t b = 0; b < nb; ++b) {
                    int32_t s = 0;
                    for (int k = 0; k < 128; ++k) {
                        const uint64_t c = b * 128 + uint64_t(k);
                        const int p = q[size_t(i * cols + c)] * x.q[size_t(c)];
                        s += p; fint += float(p); iexact += p;
                        facc_deq = facc_deq + (t.scales[size_t(i * nb + b)] * float(q[size_t(i * cols + c)])) * (x.scale * float(x.q[size_t(c)]));
                    }
                    parts[size_t(b)] = double(s) * double(t.scales[size_t(i * nb + b)]);
                    dacc += parts[size_t(b)];
                }
                if (double(fint) != double(iexact)) int_exact = false;
                const double ref = neumaier(parts) * double(x.scale);
                double mag = 0; for (double pb : parts) mag += std::fabs(pb);
                mag *= double(x.scale) * std::ldexp(1.0, -24);
                const float cand[3] = {y[size_t(i)], float(dacc * double(x.scale)), facc_deq};
                const double u = ulp_of(float(ref)) > 0 ? ulp_of(float(ref)) : 1e-45;
                for (int m = 0; m < 3; ++m) {
                    const double ae = std::fabs(double(cand[m]) - ref), e = ae / u;
                    mx[m] = std::max(mx[m], e); mean[m] += e / double(rows);
                    if (mag > 0) nrm[m] = std::max(nrm[m], ae / mag);
                }
                double yt = 0; for (uint64_t c = 0; c < cols; ++c) yt += double(W[size_t(i * cols + c)]) * xf[size_t(c)];
                qerr += (ref - yt) * (ref - yt); qn += yt * yt;
            }
            std::printf("| %llu | %s | %.2f / %.3f | %.2f | %.2f / %.3f | %.1f / %.2f | %.1f | %s | %.4f |\n", (unsigned long long)cols,
                        outl ? "8 outliers" : "Gaussian", mx[0], mean[0], nrm[0], mx[1], mean[1], mx[2], mean[2], nrm[2], int_exact ? "yes" : "NO",
                        std::sqrt(qerr / qn));
        }
    std::printf("\nBound check: |q x| <= 127, so a block sums to at most 16,256 and a 16,384-wide row to 2,080,768, under both "
                "2^31 (int32) and 2^24 (exact in float32).\n");

    std::printf("\n### M2.2 Rounding rules for offline conversion (inference stays round-half-to-even)\n\n");
    std::printf("Absmean per 128 block. 'Ties' counts weights with W / gamma exactly +-0.5. 'Slope' is E[W Wq] / E[W^2] (1.0 = no "
                "shrinkage); 'bias' is the mean of (Wq - W) / gamma for positive W (negative W mirrors it).\n\n");
    std::printf("| Weights | Rule | Ties per million | Relative MSE | Slope | Bias for W > 0 (units of gamma) | Output error |\n|---|---|---|---|---|---|---|\n");
    for (Dist d : {Dist::Gauss, Dist::Laplace, Dist::Q4Grid}) {
        Rng r(500 + int(d));
        const uint64_t rows = 1024, cols = 2560;
        auto W = make_matrix(r, rows, cols, d, 0.3, 0.2);
        auto xf = make_act(r, cols);
        std::vector<double> yref(rows);
        for (uint64_t i = 0; i < rows; ++i) { double s = 0; for (uint64_t k = 0; k < cols; ++k) s += double(W[size_t(i * cols + k)]) * xf[size_t(k)]; yref[size_t(i)] = s; }
        for (Round rr : {Round::HalfEven, Round::HalfAway, Round::Stochastic}) {
            double err = 0, en = 0, wwq = 0, bias = 0, npos = 0; uint64_t ties = 0;
            std::vector<double> y(rows, 0.0);
            std::vector<int8_t> q(128); float g;
            for (uint64_t i = 0; i < rows; ++i)
                for (uint64_t b = 0; b < cols / 128; ++b) {
                    const float* w = W.data() + i * cols + b * 128;
                    ternarize_group(w, 128, Fit::Absmean, rr, i * 131 + b, 0, nullptr, q.data(), &g);
                    for (int k = 0; k < 128; ++k) {
                        const double wq = double(g) * q[size_t(k)];
                        if (g > 0 && std::fabs(double(w[k]) / double(g)) == 0.5) ++ties;
                        err += (w[k] - wq) * (w[k] - wq); en += double(w[k]) * w[k]; wwq += double(w[k]) * wq;
                        if (w[k] > 0 && g > 0) { bias += (wq - w[k]) / double(g); npos += 1; }
                        y[size_t(i)] += wq * xf[size_t(b * 128 + uint64_t(k))];
                    }
                }
            double oe = 0, on = 0; for (uint64_t i = 0; i < rows; ++i) { oe += (y[size_t(i)] - yref[size_t(i)]) * (y[size_t(i)] - yref[size_t(i)]); on += yref[size_t(i)] * yref[size_t(i)]; }
            std::printf("| %s | %s | %.1f | %.4f | %.4f | %+.4f | %.4f |\n", dist_name(d),
                        rr == Round::HalfEven ? "half to even (ties to 0)" : rr == Round::HalfAway ? "half away (ties to +-1)" : "stochastic",
                        1e6 * double(ties) / double(rows * cols), err / en, wwq / en, bias / npos, std::sqrt(oe / on));
        }
    }
    // Activation ties (inference rounding): how often x * 127 / absmax lands exactly on .5.
    {
        Rng r(600);
        uint64_t t32 = 0, tbf = 0, n = 0; double b32 = 0, bbf = 0;
        for (int row = 0; row < 4000; ++row) {
            auto x = make_act(r, 2560);
            std::vector<float> xb(x.size());
            for (size_t i = 0; i < x.size(); ++i) { uint32_t u; std::memcpy(&u, &x[i], 4); u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u; std::memcpy(&xb[i], &u, 4); }
            for (int pass = 0; pass < 2; ++pass) {
                const std::vector<float>& v = pass ? xb : x;
                float amax = 0; for (float f : v) amax = std::max(amax, std::fabs(f));
                const float is = 127.0f / amax;
                ActRow a; quantize_activations(v.data(), v.size(), a);
                for (size_t i = 0; i < v.size(); ++i) {
                    const float s = v[i] * is; const float fl = std::floor(s);
                    if (s - fl == 0.5f) (pass ? tbf : t32)++;
                    (pass ? bbf : b32) += double(a.q[i]) - double(s);
                }
            }
            n += x.size();
        }
        std::printf("\nActivations (inference rule, round half to even): exact ties per million values: float32 inputs %.2f, "
                    "bfloat16 inputs %.2f; mean rounding error per value (units of one int8 step): %+.2e / %+.2e.\n",
                    1e6 * double(t32) / double(n), 1e6 * double(tbf) / double(n), b32 / double(n), bbf / double(n));
    }
}

// ------------------------------------------------------------------------------------------------ M3 packing
class AtomicPool {
public:
    explicit AtomicPool(int n) : n_(n) {}
    void run(int count, const std::function<void(int)>& fn) {
        if (n_ <= 1) { for (int i = 0; i < count; ++i) fn(i); return; }
        std::atomic<int> next{0}; std::vector<std::thread> ts;
        for (int t = 1; t < n_; ++t) ts.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < count;) fn(i); });
        for (int i; (i = next.fetch_add(1)) < count;) fn(i);
        for (auto& t : ts) t.join();
    }
private: int n_;
};

static void section_pack(int threads) {
    std::printf("\n## M3. Packing\n\n### M3.1 Density\n\n| Format | Weight bits per weight | Notes |\n|---|---|---|\n");
    std::printf("| I2 (shipping compute) | 2.000 | 32 bytes per 128-weight block |\n");
    std::printf("| B3 (shipping storage, row-contiguous) | 1.600 | ceil(cols / 5) bytes per row; not block aligned |\n");
    std::printf("| B3B (research, block aligned) | 1.625 | 26 bytes per 128-weight block |\n");
    std::printf("| TL3 (research, sign + 4-bit index per 3) | 1.680 | 43 triples per block (1 zero pad), 860 bytes per 32 x 128 tile |\n");
    std::printf("| Entropy limit for uniform trits | 1.585 | log2(3) |\n");
    std::printf("\nPer-128-block float32 scales add 0.25 bpw to any of these; per-tensor adds about 0.\n");

    std::printf("\n### M3.2 Decode throughput (one thread, 2,048 x 8,192 weights, trits to int8)\n\n| Format | Path | Weights per ns | Packed input GB/s |\n|---|---|---|---|\n");
    {
        Rng r(700); const uint64_t rows = 2048, cols = 8192;
        auto W = make_matrix(r, rows, cols, Dist::Gauss, 0.3, 0.2);
        Tensor i2; quantize_absmean(W.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, i2);
        Tensor b3; transcode(i2, Packing::B3, b3);
        PackedTensor bb, tl; pack_b3b(i2, bb); pack_tl3(i2, tl);
        View vi; view_of(i2, vi);
        std::vector<int8_t> out(static_cast<size_t>(cols));
        // 243-entry table for row-contiguous B3 (five int8 trits per byte).
        static int8_t lut[243][5];
        for (int v = 0; v < 243; ++v) { int t = v; for (int k = 0; k < 5; ++k) { lut[v][k] = int8_t(t % 3 - 1); t /= 3; } }
        struct D { const char* f; const char* p; double bytes; std::function<void()> fn; bool ok; };
        std::vector<D> ds = {
            {"I2", "scalar (shipping unpack_row)", double(i2.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) pchie::ternary::unpack_row(i2.codes.data() + i * cols / 4, cols, Packing::I2, out.data()); }, true},
            {"I2", "AVX2", double(i2.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) decode_row_i2_avx2(vi, i, out.data()); }, kernel_available(Kernel::I2Avx2Signed)},
            {"B3", "scalar divide (shipping unpack_row)", double(b3.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) pchie::ternary::unpack_row(b3.codes.data() + i * b3.row_bytes(), cols, Packing::B3, out.data()); }, true},
            {"B3", "scalar 243-entry table", double(b3.codes.size()), [&] {
                for (uint64_t i = 0; i < rows; ++i) { const uint8_t* s = b3.codes.data() + i * b3.row_bytes();
                    uint64_t c = 0; for (; c + 5 <= cols; c += 5) std::memcpy(out.data() + c, lut[s[c / 5]], 5);
                    for (int k = 0; c < cols; ++c, ++k) out[size_t(c)] = lut[s[c / 5]][k]; } }, true},
            {"B3B", "scalar divide", double(bb.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) research::unpack_row(bb, i, out.data()); }, true},
            {"B3B", "AVX2 multiply-high", double(bb.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) decode_row_b3b_avx2(bb, i, out.data()); }, kernel_available(Kernel::B3BAvx2)},
            {"TL3", "scalar", double(tl.codes.size()), [&] { for (uint64_t i = 0; i < rows; ++i) research::unpack_row(tl, i, out.data()); }, true},
        };
        for (auto& d : ds) {
            if (!d.ok) { std::printf("| %s | %s | n/a | n/a |\n", d.f, d.p); continue; }
            const double ms = median_ms(5, d.fn);
            std::printf("| %s | %s | %.2f | %.2f |\n", d.f, d.p, double(rows * cols) / (ms * 1e6), d.bytes / (ms * 1e6));
        }
    }

    std::printf("\n### M3.3 GEMV throughput (per-128-block scales, int8 activations)\n\n");
    std::printf("Every kernel's output is compared with the shipping scalar reference bit for bit on the same run.\n\n");
    std::printf("| Shape | Kernel | Threads | Median ms | Weights per ns | Packed weight GB/s | Bits match reference |\n|---|---|---|---|---|---|---|\n");
    for (uint64_t n : {2048ull, 8192ull}) {
        Rng r(800 + n);
        const uint64_t rows = n, cols = n;
        auto W = make_matrix(r, rows, cols, Dist::Gauss, 0.3, 0.2);
        Tensor i2; quantize_absmean(W.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, i2);
        std::vector<float>().swap(W);
        PackedTensor bb, tl; pack_b3b(i2, bb); pack_tl3(i2, tl);
        View v; view_of(i2, v);
        auto xf = make_act(r, cols); ActRow x; quantize_activations(xf.data(), cols, x);
        B3BAct xb; prepare_b3b(x, xb); TL3Act xt; prepare_tl3(x, xt);
        std::vector<float> ref(rows), y(rows);
        matvec_i2(v, x, Kernel::I2Scalar, ref.data(), rows);
        const Kernel ks[] = {Kernel::I2Scalar, Kernel::I2Avx2Biased, Kernel::I2Avx2Signed, Kernel::I2Avx2ZeroSkip,
                             Kernel::B3BScalar, Kernel::B3BAvx2, Kernel::TL3Scalar, Kernel::TL3Avx2};
        for (Kernel k : ks) {
            if (!kernel_available(k)) continue;
            const bool isb = k == Kernel::B3BScalar || k == Kernel::B3BAvx2, ist = k == Kernel::TL3Scalar || k == Kernel::TL3Avx2;
            const double bytes = double(isb ? bb.codes.size() : ist ? tl.codes.size() : i2.codes.size());
            for (int th : {1, threads}) {
                if (th == threads && threads == 1) continue;
                AtomicPool pool(th);
                double ms;
                if (th == 1) {
                    ms = median_ms(n == 2048 ? 15 : 5, [&] {
                        if (isb) matvec_b3b(bb, x, xb, k, y.data(), rows);
                        else if (ist) matvec_tl3(tl, x, xt, k, y.data(), rows);
                        else matvec_i2(v, x, k, y.data(), rows);
                    });
                } else {
                    // Multi-thread: split rows into slices that each own a sub-tensor (built once, outside timing).
                    static std::vector<PackedTensor> sb, st; static std::vector<Tensor> si; static uint64_t built = 0;
                    if (built != n) {
                        sb.clear(); st.clear(); si.clear();
                        const uint64_t per = ((rows / uint64_t(th) + 31) / 32) * 32;
                        for (uint64_t r0 = 0; r0 < rows; r0 += per) {
                            const uint64_t r1 = std::min(rows, r0 + per);
                            Tensor s; s.packing = Packing::I2; s.scale_mode = ScaleMode::PerBlock; s.rows = r1 - r0; s.cols = cols;
                            s.codes.assign(i2.codes.begin() + long(r0 * cols / 4), i2.codes.begin() + long(r1 * cols / 4));
                            s.scales.assign(i2.scales.begin() + long(r0 * cols / 128), i2.scales.begin() + long(r1 * cols / 128));
                            PackedTensor pb, pt; pack_b3b(s, pb); pack_tl3(s, pt);
                            si.push_back(std::move(s)); sb.push_back(std::move(pb)); st.push_back(std::move(pt));
                        }
                        built = n;
                    }
                    const uint64_t per = si[0].rows;
                    std::vector<View> vs(si.size()); for (size_t i = 0; i < si.size(); ++i) view_of(si[i], vs[i]);
                    ms = median_ms(n == 2048 ? 15 : 7, [&] {
                        pool.run(int(si.size()), [&](int i) {
                            float* yo = y.data() + uint64_t(i) * per;
                            if (isb) matvec_b3b(sb[size_t(i)], x, xb, k, yo, sb[size_t(i)].rows);
                            else if (ist) matvec_tl3(st[size_t(i)], x, xt, k, yo, st[size_t(i)].rows);
                            else matvec_i2(vs[size_t(i)], x, k, yo, si[size_t(i)].rows);
                        });
                    });
                }
                const bool same = std::memcmp(y.data(), ref.data(), rows * 4) == 0;
                std::printf("| %llu x %llu | %s | %d | %.3f | %.2f | %.2f | %s |\n", (unsigned long long)rows, (unsigned long long)cols,
                            kernel_name(k), th, ms, double(rows * cols) / (ms * 1e6), bytes / (ms * 1e6), same ? "yes" : "NO");
            }
        }
        // The shipping entry point (dispatch plus the same AVX2 row kernel), to show what the dispatch costs.
        if (tier_supported(Tier::AVX2)) {
            const double ms = median_ms(n == 2048 ? 15 : 5, [&] { matvec(v, x, y.data(), rows, Tier::AVX2, nullptr); });
            const bool same = std::memcmp(y.data(), ref.data(), rows * 4) == 0;
            std::printf("| %llu x %llu | shipping matvec() entry point, AVX2 tier | 1 | %.3f | %.2f | %.2f | %s |\n",
                        (unsigned long long)rows, (unsigned long long)cols, ms, double(rows * cols) / (ms * 1e6),
                        double(i2.codes.size()) / (ms * 1e6), same ? "yes" : "NO");
        }
        // Activation-side preparation cost (once per activation row, shared by all weight rows).
        const double mq = median_ms(50, [&] { ActRow a; quantize_activations(xf.data(), cols, a); });
        const double mb = median_ms(50, [&] { B3BAct a; prepare_b3b(x, a); });
        const double mt = median_ms(50, [&] { TL3Act a; prepare_tl3(x, a); });
        std::printf("| %llu x %llu | activation prep: int8 quantize / B3B planes / TL3 tables | 1 | %.4f / %.4f / %.4f | | | |\n",
                    (unsigned long long)rows, (unsigned long long)cols, mq, mb, mt);
    }

    std::printf("\n### M3.4 Zero-skip\n\n");
    {
        Rng r(900);
        const uint64_t rows = 2048, cols = 8192, nb = cols / 128;
        auto W = make_matrix(r, rows, cols, Dist::Gauss, 0.3, 0.2);
        Tensor i2; quantize_absmean(W.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, i2);
        auto q = unpack_all(i2);
        uint64_t zeros = 0, zblocks = 0, z32 = 0;
        for (uint64_t i = 0; i < rows; ++i)
            for (uint64_t b = 0; b < nb; ++b) {
                int zb = 0; for (int k = 0; k < 128; ++k) zb += q[size_t(i * cols + b * 128 + uint64_t(k))] == 0;
                zeros += uint64_t(zb); zblocks += zb == 128;
                for (int g = 0; g < 4; ++g) { int z = 0; for (int k = 0; k < 32; ++k) z += q[size_t(i * cols + b * 128 + uint64_t(g * 32 + k))] == 0; z32 += z == 32; }
            }
        std::printf("Absmean on Gaussian weights: %.1f%% of weights are zero, %llu of %llu 128-blocks and %llu of %llu 32-groups "
                    "are all zero (expected about 0.31^128 and 0.31^32, effectively never).\n\n",
                    100.0 * double(zeros) / double(rows * cols), (unsigned long long)zblocks, (unsigned long long)(rows * nb),
                    (unsigned long long)z32, (unsigned long long)(rows * nb * 4));
        std::printf("| All-zero blocks | Biased AVX2, ms | Biased + zero-skip, ms | Speed-up |\n|---|---|---|---|\n");
        auto xf = make_act(r, cols); ActRow x; quantize_activations(xf.data(), cols, x);
        for (int pct : {0, 50, 90, 99}) {
            Tensor t = i2;
            std::vector<int8_t> qq = q;
            Rng rz(901);
            for (uint64_t i = 0; i < rows; ++i) for (uint64_t b = 0; b < nb; ++b)
                if (double(rz.g() % 10000) < pct * 100.0) std::memset(qq.data() + i * cols + b * 128, 0, 128);
            for (uint64_t i = 0; i < rows; ++i) pack_row(qq.data() + i * cols, cols, Packing::I2, t.codes.data() + i * cols / 4);
            View v; view_of(t, v); std::vector<float> y1(rows), y2(rows);
            if (!kernel_available(Kernel::I2Avx2ZeroSkip)) { std::printf("| %d%% | n/a | n/a | |\n", pct); continue; }
            const double a = median_ms(9, [&] { matvec_i2(v, x, Kernel::I2Avx2Biased, y1.data(), rows); });
            const double b = median_ms(9, [&] { matvec_i2(v, x, Kernel::I2Avx2ZeroSkip, y2.data(), rows); });
            std::printf("| %d%% | %.3f | %.3f | %.2fx%s |\n", pct, a, b, a / b, std::memcmp(y1.data(), y2.data(), rows * 4) ? " (BITS DIFFER)" : "");
        }
    }
}

// ------------------------------------------------------------------------------------------------ M4 softmax, norm
static void section_softmax() {
    std::printf("\n## M4. Integer softmax and RMSNorm\n\n### M4.1 2^f for f in [0, 1), exhaustive over all 65,536 Q16 inputs\n\n");
    std::printf("| Method | Table bytes | Max relative error | Mean relative error |\n|---|---|---|---|\n");
    for (Exp2 m : {Exp2::Table256, Exp2::Table64Lerp, Exp2::Poly3}) {
        double mx = 0, mean = 0;
        for (uint32_t f = 0; f < 65536; ++f) {
            const double e = std::fabs(double(exp2_frac_q30(f, m)) / 1073741824.0 / std::exp2(double(f) / 65536.0) - 1.0);
            mx = std::max(mx, e); mean += e / 65536.0;
        }
        std::printf("| %s | %d | %.2e | %.2e |\n", exp2_name(m), m == Exp2::Table256 ? 1024 : m == Exp2::Table64Lerp ? 260 : 16, mx, mean);
    }
    const double l2e = 1.0 / std::log(2.0);
    std::printf("\nlog2(e) in fixed point: Q16 relative error %.2e, Q30 relative error %.2e. At a max-subtracted logit of -30 that "
                "moves the exponent by %.2e (Q16) or %.2e (Q30), a relative change in that weight of the same size.\n",
                std::fabs(std::round(l2e * 65536.0) / 65536.0 / l2e - 1), std::fabs(std::round(l2e * 1073741824.0) / 1073741824.0 / l2e - 1),
                30 * std::fabs(std::round(l2e * 65536.0) / 65536.0 - l2e) * std::log(2.0),
                30 * std::fabs(std::round(l2e * 1073741824.0) / 1073741824.0 - l2e) * std::log(2.0));

    std::printf("\n### M4.2 Softmax against double (probabilities in Q16; 200 rows per case)\n\n");
    std::printf("'Downstream' is max |sum (p - p_ref) v| over 64 random value columns in [-1, 1], the error an attention output would see. "
                "The Q16 output grid alone allows up to n / 2 * 2^-16 of L1 error.\n\n");
    std::printf("| Logits | n | Method | Output | Max abs error in p | Max rel error, p > 1e-3 | Mean L1 error | Max downstream error |\n|---|---|---|---|---|---|---|---|\n");
    struct SC { const char* name; double sigma; bool peaked; };
    const SC cases[] = {{"N(0, 1)", 1.0, false}, {"N(0, 3)", 3.0, false}, {"N(0, 8)", 8.0, false}, {"one logit +10 above N(0, 1)", 1.0, true}};
    for (const SC& c : cases)
        for (size_t n : {128ul, 1024ul, 4096ul})
            for (int ob : {16, 24})
            for (Exp2 m : {Exp2::Table256, Exp2::Table64Lerp, Exp2::Poly3}) {
                if (ob == 24 && c.peaked) continue;
                Rng r(1000 + n);
                const double one = std::ldexp(1.0, ob);
                double mabs = 0, mrel = 0, l1 = 0, down = 0;
                std::vector<int32_t> z(n); std::vector<uint32_t> p(n); std::vector<double> zd(n), pr(n);
                std::vector<double> vcol(n * 64);
                for (auto& v : vcol) v = 2.0 * r.u() - 1.0;
                for (int row = 0; row < 200; ++row) {
                    for (size_t i = 0; i < n; ++i) { zd[i] = c.sigma * r.gauss(); }
                    if (c.peaked) zd[size_t(r.g() % n)] += 10.0;
                    for (size_t i = 0; i < n; ++i) { z[i] = int32_t(std::lround(zd[i] * 65536.0)); zd[i] = double(z[i]) / 65536.0; }
                    softmax_fixed(z.data(), n, m, ob, p.data());
                    const double mx = *std::max_element(zd.begin(), zd.end());
                    double s = 0; for (size_t i = 0; i < n; ++i) { pr[i] = std::exp(zd[i] - mx); s += pr[i]; }
                    double rl1 = 0;
                    for (size_t i = 0; i < n; ++i) {
                        pr[i] /= s; const double e = std::fabs(double(p[i]) / one - pr[i]);
                        mabs = std::max(mabs, e); rl1 += e;
                        if (pr[i] > 1e-3) mrel = std::max(mrel, e / pr[i]);
                    }
                    l1 += rl1 / 200.0;
                    for (int col = 0; col < 64; ++col) {
                        double d = 0; for (size_t i = 0; i < n; ++i) d += (double(p[i]) / one - pr[i]) * vcol[i * 64 + size_t(col)];
                        down = std::max(down, std::fabs(d));
                    }
                }
                std::printf("| %s | %zu | %s | Q%d | %.2e | %.2e | %.2e | %.2e |\n", c.name, n, exp2_name(m), ob, mabs, mrel, l1, down);
            }
}

static void section_norm() {
    std::printf("\n### M4.3 Integer RMSNorm\n\n");
    std::printf("| Newton steps from the 64-entry seed | Max relative error of 1/sqrt, dense sweep of [1, 4) |\n|---|---|\n");
    for (int s = 0; s < 4; ++s) {
        double w = 0;
        for (uint64_t u = 1ull << 30; u < (1ull << 32); u += 1021)
            w = std::max(w, std::fabs(double(rsqrt_q30(uint32_t(u), s)) / 1073741824.0 * std::sqrt(double(u) / 1073741824.0) - 1.0));
        std::printf("| %d | %.2e |\n", s, w);
    }
    std::printf("\nEnd-to-end against double on the float row (2,560 wide, 2,000 rows, Gaussian plus 4 outlier channels at 30 sigma). "
                "The input is quantized per row (absmax) to int8 or int16 first; the norm itself is exact integer math.\n\n");
    std::printf("| Input | Max abs error (units of the normalized value) | RMS error | Max error on the outlier channels |\n|---|---|---|---|\n");
    for (int bits : {8, 16}) {
        Rng r(1100);
        double mx = 0, se = 0, cnt = 0, mo = 0;
        const size_t n = 2560; const double lim = bits == 8 ? 127.0 : 32767.0;
        std::vector<int16_t> xi(n); std::vector<int32_t> y(n);
        for (int row = 0; row < 2000; ++row) {
            auto x = make_act(r, n, 0);
            std::vector<size_t> outl; for (int k = 0; k < 4; ++k) { const size_t i = size_t(r.g() % n); x[i] = float(30.0 * (r.g() & 1 ? 1 : -1)); outl.push_back(i); }
            double amax = 0; for (float f : x) amax = std::max(amax, double(std::fabs(f)));
            for (size_t i = 0; i < n; ++i) xi[i] = int16_t(std::nearbyint(double(x[i]) / amax * lim));
            // eps = 1e-6 in float units -> in quantized units squared, Q8 (at least 1)
            const double scale = lim / amax;
            const uint32_t eps = uint32_t(std::max(1.0, std::round(1e-6 * scale * scale * 256.0)));
            rmsnorm_q16(xi.data(), n, eps, y.data());
            double ss = 0; for (float f : x) ss += double(f) * f;
            const double rs = 1.0 / std::sqrt(ss / double(n) + 1e-6);
            for (size_t i = 0; i < n; ++i) {
                const double e = std::fabs(double(y[i]) / 65536.0 - double(x[i]) * rs);
                mx = std::max(mx, e); se += e * e; cnt += 1;
            }
            for (size_t i : outl) mo = std::max(mo, std::fabs(double(y[i]) / 65536.0 - double(x[i]) * rs));
        }
        std::printf("| int%d | %.2e | %.2e | %.2e |\n", bits, mx, std::sqrt(se / cnt), mo);
    }
    std::printf("\nSafe input bounds (enforced, refused otherwise with zeroed output) and the largest intermediates they allow:\n\n");
    std::printf("| Quantity | Bound | Worst case | Fits |\n|---|---|---|---|\n");
    std::printf("| x_i | -32767..32767 (int16, -32768 refused) | 2^15 | |\n");
    std::printf("| n | 1..2^20 | 2^20 | |\n");
    std::printf("| sum of squares | int64 | 32767^2 x 2^20 < 2^50 | yes, 2^13 headroom |\n");
    std::printf("| mean with S fractional bits | S even, 12..30, chosen so ss << S < 2^62 | < 2^62 | yes |\n");
    std::printf("| eps term | eps_q8 (1..2^32-1) << (S - 8) | < 2^54 | yes, v < 2^63 |\n");
    std::printf("| Newton intermediates | u < 2^32, y <= 2^30 | u * y^2 < 2^62, y * (3 - u y^2) < 2^62 | yes |\n");
    std::printf("| x_i * R | |x| < 2^15, R <= 2^30 | < 2^45 | yes |\n");
    std::printf("| output | |x| / rms <= sqrt(n) | 2^10 in Q16 = 2^26 | yes (int32; clamped at 2^30 anyway) |\n");
    {
        std::vector<int16_t> x(kMaxNorm, 32767); std::vector<int32_t> y(kMaxNorm);
        const Status a = rmsnorm_q16(x.data(), x.size(), 0xFFFFFFFFu, y.data());
        const double ya = double(y[0]) / 65536.0;
        std::fill(x.begin(), x.end(), int16_t(0)); x[0] = 32767;
        const Status b = rmsnorm_q16(x.data(), x.size(), 1, y.data());
        std::printf("\nWorst cases run: all 32767 at n = 2^20 with the largest eps: %s, y[0] = %.6f; one spike at n = 2^20: %s, "
                    "y[0] = %.4f (double: %.4f).\n", status_name(a), ya, status_name(b),
                    double(y[0]) / 65536.0, 32767.0 / std::sqrt(32767.0 * 32767.0 / 1048576.0 + 1.0 / 256.0));
    }
    // Speed (one thread).
    {
        Rng r(1200); const size_t n = 4096;
        std::vector<int16_t> x(n); for (auto& v : x) v = int16_t(std::lround(r.gauss() * 3000));
        std::vector<int32_t> y(n); std::vector<int32_t> z(n); std::vector<uint32_t> p(n);
        for (auto& v : z) v = int32_t(std::lround(r.gauss() * 3 * 65536));
        const double a = median_ms(200, [&] { rmsnorm_q16(x.data(), n, 1, y.data()); });
        std::printf("\nScalar speed, one thread, n = 4,096: RMSNorm %.2f ns per element", a * 1e6 / double(n));
        for (Exp2 m : {Exp2::Table256, Exp2::Table64Lerp, Exp2::Poly3}) {
            const double b = median_ms(200, [&] { softmax_q16(z.data(), n, m, p.data()); });
            std::printf("; softmax (%s) %.2f ns", m == Exp2::Table256 ? "table 256" : m == Exp2::Table64Lerp ? "table 64 + lerp" : "poly 3", b * 1e6 / double(n));
        }
        std::printf(" per element.\n");
    }
}

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "all";
    const int threads = argc > 2 ? std::max(1, std::min(8, std::atoi(argv[2]))) : 4;
    std::printf("# Ternary measurements\n\nTier detected: %s. AVX2 research kernels: %s. Threads for multi-thread rows: %d.\n",
                tier_name(detect_tier()), kernel_available(Kernel::I2Avx2Signed) ? "yes" : "no", threads);
    const double t0 = now_ms();
    if (what == "scales" || what == "all") section_scales();
    if (what == "accum" || what == "all") section_accum();
    if (what == "pack" || what == "all") section_pack(threads);
    if (what == "softmax" || what == "all") section_softmax();
    if (what == "norm" || what == "all") section_norm();
    std::printf("\n(section run time %.1f s)\n", (now_ms() - t0) / 1000.0);
    return 0;
}
