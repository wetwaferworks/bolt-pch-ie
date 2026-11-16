// Ternary path test bench: packing round trips, strict parsing, quantizers, reference math, and every SIMD
// kernel against the scalar reference, bit for bit.
#include "pchie/pchie_ternary.h"
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <vector>
using namespace pchie::ternary;

static int passes = 0, fails = 0;
#define CHECK(c, ...) do { if (c) ++passes; else { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// A test-only dispatcher: workers claim chunk indices from an atomic counter (no locks).
class AtomicPool : public pchie::IJobDispatcher {
public:
    explicit AtomicPool(int n) : n_(n) {}
    void parallel_for(int count, const std::function<void(int)>& fn) override {
        std::atomic<int> next{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < n_; ++t) ts.emplace_back([&] { for (int i; (i = next.fetch_add(1)) < count;) fn(i); });
        for (auto& t : ts) t.join();
    }
    int size() const noexcept override { return n_; }
private:
    int n_;
};

static std::vector<int8_t> rand_trits(std::mt19937& r, size_t n, int zero_pct = 34) {
    std::uniform_int_distribution<int> d(0, 99), s(0, 1);
    std::vector<int8_t> q(n);
    for (auto& v : q) v = d(r) < zero_pct ? 0 : (s(r) ? 1 : -1);
    return q;
}
static Tensor make_tensor(const std::vector<int8_t>& q, uint64_t rows, uint64_t cols, Packing p, ScaleMode m, std::mt19937& r) {
    Tensor t; t.packing = p; t.scale_mode = m; t.rows = rows; t.cols = cols;
    t.codes.resize(static_cast<size_t>(rows * row_bytes_for(p, cols)));
    for (uint64_t i = 0; i < rows; ++i) pack_row(q.data() + i * cols, cols, p, t.codes.data() + i * row_bytes_for(p, cols));
    std::uniform_real_distribution<float> sd(0.001f, 0.2f);
    t.scales.resize(m == ScaleMode::PerTensor ? 1 : static_cast<size_t>(rows * cols / kBlock));
    for (auto& s : t.scales) s = sd(r);
    return t;
}
static ActRow rand_act(std::mt19937& r, uint64_t n, float sigma = 1.0f) {
    std::normal_distribution<float> nd(0.0f, sigma);
    std::vector<float> x(static_cast<size_t>(n)); for (auto& v : x) v = nd(r);
    ActRow a; quantize_activations(x.data(), n, a); return a;
}
// Independent integer dot from the unpacked trits (int64, no shared code with the kernels).
static int64_t idot(const std::vector<int8_t>& q, uint64_t row, uint64_t cols, const ActRow& x) {
    int64_t s = 0; for (uint64_t c = 0; c < cols; ++c) s += int64_t(q[row * cols + c]) * x.q[c]; return s;
}

static std::vector<Tier> tiers_here() {
    std::vector<Tier> v;
    for (Tier t : {Tier::Scalar, Tier::SSSE3, Tier::AVX2, Tier::NEON}) if (tier_supported(t)) v.push_back(t);
    return v;
}

static void test_packing() {
    std::mt19937 r(1);
    for (Packing p : {Packing::I2, Packing::B3})
        for (uint64_t cols : {128ull, 256ull, 640ull, 2560ull, 6912ull}) {
            auto q = rand_trits(r, cols);
            std::vector<uint8_t> buf(static_cast<size_t>(row_bytes_for(p, cols)));
            std::vector<int8_t> back(static_cast<size_t>(cols), 9);
            CHECK(pack_row(q.data(), cols, p, buf.data()) == Status::Ok, "pack");
            CHECK(unpack_row(buf.data(), cols, p, back.data()) == Status::Ok && back == q, "round trip p=%d cols=%llu", int(p), (unsigned long long)cols);
        }
    // every B3 byte value
    int bad = 0;
    for (int b = 0; b < 256; ++b) {
        uint8_t row[26] = {}; std::memset(row, 121, sizeof row);  // 121 = all five trits are zero weights
        row[0] = uint8_t(b);
        std::vector<int8_t> q(128);
        const Status s = unpack_row(row, 128, Packing::B3, q.data());
        if (b <= 242) {
            uint8_t again[26];
            if (s != Status::Ok || pack_row(q.data(), 128, Packing::B3, again) != Status::Ok || again[0] != b) ++bad;
        } else if (s != Status::BadCode) ++bad;
    }
    CHECK(bad == 0, "B3 byte table: %d bad", bad);
    // B3 padding: 128 = 25*5 + 3, so the last byte carries 2 padding weights that must be zero
    {
        std::vector<int8_t> q(128, 0); std::vector<uint8_t> b(26);
        pack_row(q.data(), 128, Packing::B3, b.data());
        CHECK(b[25] == 1 + 3 + 9 + 27 + 81, "padding encodes zeros (got %d)", b[25]);
        b[25] = uint8_t(1 + 3 + 9 + 27 * 2 + 81);  // weight 128 (padding) set to +1
        CHECK(unpack_row(b.data(), 128, Packing::B3, q.data()) == Status::BadCode, "non-zero padding rejected");
    }
    // I2 code 3 anywhere is rejected
    {
        std::vector<int8_t> q(128, 0); std::vector<uint8_t> b(32), back(128);
        pack_row(q.data(), 128, Packing::I2, b.data());
        int missed = 0;
        for (int j = 0; j < 32; ++j) for (int k = 0; k < 4; ++k) {
            auto c = b; c[static_cast<size_t>(j)] = uint8_t(c[static_cast<size_t>(j)] | (3 << (2 * k)));
            std::vector<int8_t> o(128);
            if (unpack_row(c.data(), 128, Packing::I2, o.data()) != Status::BadCode) ++missed;
            Tensor t; t.rows = 1; t.cols = 128; t.codes = c; t.scales = {1.0f}; View v;
            if (view_of(t, v) != Status::BadCode) ++missed;
        }
        CHECK(missed == 0, "code 3 rejected at every position (%d missed)", missed);
    }
    int8_t two[128] = {}; two[5] = 2; uint8_t sink[32];
    CHECK(pack_row(two, 128, Packing::I2, sink) == Status::BadCode, "pack refuses a value outside {-1,0,1}");
    CHECK(row_bytes_for(Packing::I2, 100) == 0 && row_bytes_for(Packing::I2, 0) == 0, "width not a multiple of 128 refused");
}

static void test_quantizers() {
    // absmean: hand example, per tensor. |W| mean = (0.2+0.6+1.0+0.0+...)/128
    std::vector<float> W(128, 0.0f);
    W[0] = 0.2f; W[1] = -0.6f; W[2] = 1.0f; W[3] = -1.0f;
    Tensor t;
    CHECK(quantize_absmean(W.data(), 1, 128, ScaleMode::PerTensor, Packing::I2, t) == Status::Ok, "absmean ok");
    const double g = (0.2 + 0.6 + 1.0 + 1.0) / 128.0;   // computed the same way: double sum of float values
    double gs = 0; for (float w : W) gs += std::fabs(double(w));
    CHECK(bits(t.scales[0]) == bits(float(gs / 128.0)), "gamma = mean|W| (%.9g vs %.9g)", t.scales[0], g);
    std::vector<int8_t> q(128);
    unpack_row(t.codes.data(), 128, Packing::I2, q.data());
    CHECK(q[0] == 1 && q[1] == -1 && q[2] == 1 && q[3] == -1 && q[4] == 0, "absmean codes");
    // ties go to 0: 254, +1, -1 and zeros give gamma = 256 / 128 = 2 exactly, so +-1 sit exactly on +-0.5 gamma
    std::vector<float> T(128, 0.0f); T[0] = 254.0f; T[1] = 1.0f; T[2] = -1.0f;
    CHECK(quantize_absmean(T.data(), 1, 128, ScaleMode::PerTensor, Packing::I2, t) == Status::Ok && t.scales[0] == 2.0f, "tie case gamma 2");
    unpack_row(t.codes.data(), 128, Packing::I2, q.data());
    CHECK(q[0] == 1 && q[1] == 0 && q[2] == 0, "exact +-half rounds to 0 (%d %d %d)", q[0], q[1], q[2]);
    // all-zero tensor
    std::vector<float> Z(256, 0.0f);
    CHECK(quantize_absmean(Z.data(), 2, 128, ScaleMode::PerBlock, Packing::B3, t) == Status::Ok && t.scales.size() == 2 &&
          t.scales[0] == 0.0f, "all zero ok");
    // non-finite refused
    W[7] = std::numeric_limits<float>::quiet_NaN();
    CHECK(quantize_absmean(W.data(), 1, 128, ScaleMode::PerTensor, Packing::I2, t) == Status::NonFinite && t.rows == 0, "NaN weight refused");
    W[7] = std::numeric_limits<float>::infinity();
    CHECK(quantize_absmean(W.data(), 1, 128, ScaleMode::PerTensor, Packing::I2, t) == Status::NonFinite, "inf weight refused");
    CHECK(quantize_absmean(W.data(), 1, 100, ScaleMode::PerTensor, Packing::I2, t) == Status::BadArgument, "bad width refused");
    Limits small; small.max_rows = 4;
    std::vector<float> big(128 * 8, 0.1f);
    CHECK(quantize_absmean(big.data(), 8, 128, ScaleMode::PerTensor, Packing::I2, t, small) == Status::TooLarge, "row limit");
    // per-block scales: block means are separate
    std::vector<float> B(256); for (int i = 0; i < 128; ++i) { B[static_cast<size_t>(i)] = 1.0f; B[static_cast<size_t>(128 + i)] = -3.0f; }
    CHECK(quantize_absmean(B.data(), 1, 256, ScaleMode::PerBlock, Packing::I2, t) == Status::Ok &&
          t.scales[0] == 1.0f && t.scales[1] == 3.0f, "per-block gammas");

    // activations
    ActRow a;
    std::vector<float> x(256, 0.0f); x[0] = 2.0f; x[1] = -1.0f; x[2] = 1.0f / 127.0f * 2.0f * 0.5f; x[200] = -2.0f;
    CHECK(quantize_activations(x.data(), 256, a) == Status::Ok, "act ok");
    CHECK(a.q[0] == 127 && a.q[200] == -127 && a.q[1] == -64, "act values (%d %d %d)", a.q[0], a.q[200], a.q[1]);
    CHECK(bits(a.scale) == bits(2.0f / 127.0f), "act scale");
    int32_t b0 = 0, b1 = 0; for (int i = 0; i < 128; ++i) { b0 += a.q[static_cast<size_t>(i)]; b1 += a.q[static_cast<size_t>(128 + i)]; }
    CHECK(a.bsum[0] == b0 && a.bsum[1] == b1, "block sums");
    x[9] = std::numeric_limits<float>::quiet_NaN();
    CHECK(quantize_activations(x.data(), 256, a) == Status::NonFinite && a.scale == 0.0f, "NaN activation -> safe zero row");
    bool allz = true; for (auto v : a.q) allz = allz && v == 0; CHECK(allz, "row zeroed");
    std::vector<float> zx(128, 0.0f);
    CHECK(quantize_activations(zx.data(), 128, a) == Status::Ok && a.scale == 0.0f, "zero row");
    CHECK(quantize_activations(zx.data(), 127, a) == Status::BadArgument, "bad length");
}

static void test_format() {
    std::mt19937 r(7);
    for (Packing p : {Packing::I2, Packing::B3}) for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock}) {
        const uint64_t rows = 5, cols = 384;
        auto q = rand_trits(r, rows * cols);
        Tensor t = make_tensor(q, rows, cols, p, m, r);
        auto buf = serialize(t);
        View v;
        CHECK(parse(buf.data(), buf.size(), Limits(), v) == Status::Ok, "parse round trip p=%d m=%d", int(p), int(m));
        CHECK(v.rows == rows && v.cols == cols && std::memcmp(v.codes, t.codes.data(), t.codes.size()) == 0, "view matches");
        // every truncation fails cleanly
        int trunc_ok = 0;
        for (size_t n = 0; n < buf.size(); ++n) { View w; if (parse(buf.data(), n, Limits(), w) == Status::Ok) ++trunc_ok; }
        CHECK(trunc_ok == 0, "truncations accepted: %d", trunc_ok);
        // trailing byte
        auto longer = buf; longer.push_back(0); View w;
        CHECK(parse(longer.data(), longer.size(), Limits(), w) != Status::Ok, "trailing byte refused");
        // every single-byte change in the header is refused (the checksum covers the rest)
        int accepted = 0;
        for (size_t i = 0; i < 96; ++i) for (int bit = 0; bit < 8; ++bit) {
            auto c = buf; c[i] ^= uint8_t(1 << bit);
            if (i >= 64 && i < 72) continue;   // flipping the checksum itself is covered below
            View z; if (parse(c.data(), c.size(), Limits(), z) == Status::Ok) ++accepted;
        }
        CHECK(accepted == 0, "header bit flips accepted: %d", accepted);
        // any change to scales or codes is caught by the checksum
        int missed = 0;
        for (size_t i = 96; i < buf.size(); i += 7) { auto c = buf; c[i] ^= 0x10; View z; if (parse(c.data(), c.size(), Limits(), z) == Status::Ok) ++missed; }
        CHECK(missed == 0, "payload changes accepted: %d", missed);
        auto c = buf; c[64] ^= 1; CHECK(parse(c.data(), c.size(), Limits(), w) == Status::BadChecksum, "checksum flip");
        Limits tiny; tiny.max_bytes = buf.size() - 1;
        CHECK(parse(buf.data(), buf.size(), tiny, w) == Status::TooLarge, "byte ceiling");
        Limits fewrows; fewrows.max_rows = 4;
        CHECK(parse(buf.data(), buf.size(), fewrows, w) == Status::TooLarge, "row ceiling");
    }
    // bad scales are refused even with a correct checksum
    std::mt19937 r2(3);
    auto q = rand_trits(r2, 128);
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -1.0f,
                      std::numeric_limits<float>::denorm_min(), 1e-39f}) {
        Tensor t = make_tensor(q, 1, 128, Packing::I2, ScaleMode::PerTensor, r2); t.scales[0] = bad;
        auto buf = serialize(t); View v;
        CHECK(parse(buf.data(), buf.size(), Limits(), v) == Status::BadScale, "bad scale %g refused", double(bad));
    }
    // a code-3 payload with a recomputed checksum is still refused
    Tensor t = make_tensor(q, 1, 128, Packing::I2, ScaleMode::PerTensor, r2); t.codes[3] |= 0xC0;
    auto buf = serialize(t); View v;
    CHECK(parse(buf.data(), buf.size(), Limits(), v) == Status::BadCode && v.rows == 0, "code 3 refused at parse");
    CHECK(parse(nullptr, 100, Limits(), v) == Status::BadArgument, "null buffer");
}

static void test_reference_math() {
    std::mt19937 r(11);
    for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock}) for (Packing p : {Packing::I2, Packing::B3}) {
        const uint64_t rows = 9, cols = 1280;
        auto q = rand_trits(r, rows * cols);
        Tensor t = make_tensor(q, rows, cols, p, m, r);
        View v; view_of(t, v);
        ActRow x = rand_act(r, cols);
        int bad = 0;
        for (uint64_t row = 0; row < rows; ++row) {
            const float got = dot_row_reference(v, row, x);
            float want;
            if (m == ScaleMode::PerTensor) want = (float(int32_t(idot(q, row, cols, x))) * t.scales[0]) * x.scale;
            else {
                float acc = 0.0f;
                for (uint64_t b = 0; b < cols / 128; ++b) {
                    int32_t s = 0; for (uint64_t c = b * 128; c < b * 128 + 128; ++c) s += q[static_cast<size_t>(row * cols + c)] * x.q[static_cast<size_t>(c)];
                    const float pr = float(s) * t.scales[static_cast<size_t>(row * (cols / 128) + b)];
                    acc = acc + pr;
                }
                want = acc * x.scale;
            }
            if (bits(got) != bits(want)) ++bad;
        }
        CHECK(bad == 0, "reference == independent math (m=%d p=%d): %d rows differ", int(m), int(p), bad);
    }
    // end to end against float math: ternarized matvec approximates W x within the quantization error
    const uint64_t rows = 32, cols = 2048;
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> W(rows * cols); for (auto& w : W) w = nd(r);
    Tensor t; quantize_absmean(W.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, t);
    View v; view_of(t, v);
    std::vector<float> xf(cols); std::normal_distribution<float> xd(0.0f, 1.0f); for (auto& e : xf) e = xd(r);
    ActRow x; quantize_activations(xf.data(), cols, x);
    double num = 0, den = 0;
    for (uint64_t row = 0; row < rows; ++row) {
        // exact dot of the dequantized ternary weights with the float activations
        std::vector<int8_t> qq(cols); unpack_row(t.codes.data() + row * cols / 4, cols, Packing::I2, qq.data());
        double exact = 0; for (uint64_t c = 0; c < cols; ++c) exact += double(qq[c]) * t.scales[static_cast<size_t>(row * 16 + c / 128)] * xf[c];
        const double got = dot_row_reference(v, row, x);
        num += (got - exact) * (got - exact); den += exact * exact;
    }
    CHECK(std::sqrt(num / den) < 0.02, "int8 activation error small vs dequantized ternary (rel %.4g)", std::sqrt(num / den));
}

static void test_kernels() {
    std::mt19937 r(2026);
    const auto tiers = tiers_here();
    std::printf("tiers on this machine:"); for (Tier t : tiers) std::printf(" %s", tier_name(t)); std::printf("\n");
    int bad = 0, total = 0;
    for (uint64_t cols : {128ull, 256ull, 384ull, 2560ull, 6912ull, 16384ull})
        for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock})
            for (int pattern = 0; pattern < 6; ++pattern) {
                const uint64_t rows = 7;
                std::vector<int8_t> q;
                if (pattern == 0) q = rand_trits(r, rows * cols);
                else if (pattern == 1) q.assign(rows * cols, 1);
                else if (pattern == 2) q.assign(rows * cols, -1);
                else if (pattern == 3) q.assign(rows * cols, 0);
                else if (pattern == 4) { q.resize(rows * cols); for (size_t i = 0; i < q.size(); ++i) q[i] = (i & 1) ? 1 : -1; }
                else q = rand_trits(r, rows * cols, 90);
                Tensor t = make_tensor(q, rows, cols, Packing::I2, m, r);
                View v; view_of(t, v);
                ActRow x = rand_act(r, cols);
                if (pattern == 1 || pattern == 2) { for (auto& e : x.q) e = 127; for (auto& b : x.bsum) b = 127 * 128; }
                if (pattern == 4) { for (size_t i = 0; i < x.q.size(); ++i) x.q[i] = (i & 1) ? -127 : 127; for (auto& b : x.bsum) b = 0; }
                for (uint64_t row = 0; row < rows; ++row) {
                    const float ref = dot_row_reference(v, row, x);
                    for (Tier tr : tiers) { ++total; if (bits(dot_row(v, row, x, tr)) != bits(ref)) { ++bad; if (bad < 5) std::printf("  mismatch tier %s cols %llu pattern %d\n", tier_name(tr), (unsigned long long)cols, pattern); } }
                }
            }
    CHECK(bad == 0, "every kernel == reference bit for bit (%d of %d differ)", bad, total);

    // widest row: 2^20 columns, all +1 against all 127 -> isum = 127 * 2^20, no overflow anywhere
    {
        const uint64_t cols = kHardMaxCols;
        std::vector<int8_t> q(cols, 1);
        Tensor t; t.rows = 1; t.cols = cols; t.codes.resize(cols / 4); pack_row(q.data(), cols, Packing::I2, t.codes.data()); t.scales = {1.0f};
        View v; CHECK(view_of(t, v) == Status::Ok, "max width view");
        ActRow x; x.n = cols; x.q.assign(cols, 127); x.bsum.assign(cols / 128, 127 * 128); x.scale = 1.0f;
        const float want = float(int32_t(127) * int32_t(cols));
        int okc = 0; for (Tier tr : tiers) okc += bits(dot_row(v, 0, x, tr)) == bits(want);
        CHECK(okc == int(tiers.size()) && bits(dot_row_reference(v, 0, x)) == bits(want), "max width exact on every tier");
    }

    // matvec: serial, 1..12 threads and every tier give identical bits
    {
        const uint64_t rows = 333, cols = 2560;
        auto q = rand_trits(r, rows * cols);
        Tensor t = make_tensor(q, rows, cols, Packing::I2, ScaleMode::PerBlock, r);
        View v; view_of(t, v);
        ActRow x = rand_act(r, cols);
        std::vector<float> y0(rows), y(rows);
        CHECK(matvec(v, x, y0.data(), rows, Tier::Scalar, nullptr) == Status::Ok, "matvec serial");
        int diff = 0;
        for (Tier tr : tiers) for (int th : {1, 2, 4, 12}) {
            AtomicPool pool(th);
            std::fill(y.begin(), y.end(), -1.0f);
            matvec(v, x, y.data(), rows, tr, &pool);
            for (uint64_t i = 0; i < rows; ++i) diff += bits(y[i]) != bits(y0[i]);
        }
        CHECK(diff == 0, "matvec identical across tiers and thread counts (%d diffs)", diff);
        std::vector<float> yb(rows + 1, 5.0f);
        CHECK(matvec(v, x, yb.data(), rows + 1, Tier::Scalar, nullptr) == Status::BadArgument && yb[0] == 0.0f, "shape mismatch -> zeros");
        // B3 storage gives the same result as I2 after transcoding, and directly through the reference
        Tensor b3; CHECK(transcode(t, Packing::B3, b3) == Status::Ok, "transcode to B3");
        Tensor back; CHECK(transcode(b3, Packing::I2, back) == Status::Ok && back.codes == t.codes, "B3 -> I2 round trip");
        View vb; view_of(b3, vb);
        std::vector<float> yb3(rows); matvec(vb, x, yb3.data(), rows, detect_tier(), nullptr);
        int d3 = 0; for (uint64_t i = 0; i < rows; ++i) d3 += bits(yb3[i]) != bits(y0[i]);
        CHECK(d3 == 0, "B3 reference == I2 (%d diffs)", d3);
        std::printf("bytes per weight: I2 %.3f, B3 %.3f\n", double(t.codes.size()) * 8 / double(rows * cols), double(b3.codes.size()) * 8 / double(rows * cols));
    }
}

// Edge scales: subnormal gammas become exact zero groups; outputs that overflow are reported and zeroed.
static void test_edge_scales() {
    std::mt19937 r(21);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> W(32 * 256);
    for (auto& w : W) w = nd(r) * 1e-41f;
    Tensor t;
    CHECK(quantize_absmean(W.data(), 32, 256, ScaleMode::PerBlock, Packing::I2, t) == Status::Ok, "tiny weights quantize");
    bool zero = true; for (float s : t.scales) if (s != 0.0f) zero = false;
    std::vector<int8_t> q(256); bool codes0 = true;
    for (uint64_t i = 0; i < t.rows; ++i) { unpack_row(t.codes.data() + i * t.row_bytes(), 256, Packing::I2, q.data()); for (auto c : q) if (c) codes0 = false; }
    CHECK(zero && codes0, "subnormal gammas flushed to exact zero groups");
    // Mixed: one row tiny, the rest ordinary; only the tiny blocks flush.
    for (size_t i = 0; i < W.size(); ++i) W[i] = nd(r) * (i < 256 ? 1e-41f : 0.02f);
    CHECK(quantize_absmean(W.data(), 32, 256, ScaleMode::PerBlock, Packing::I2, t) == Status::Ok &&
          t.scales[0] == 0.0f && t.scales[1] == 0.0f && t.scales[2] > 0.0f, "only the tiny blocks flush");
    // Overflow: scales near FLT_MAX with flat activations overflow float at every tier.
    auto qq = rand_trits(r, 32 * 256);
    for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock}) {
        Tensor h = make_tensor(qq, 32, 256, Packing::I2, m, r);
        for (auto& s : h.scales) s = 3e38f;
        View v; CHECK(view_of(h, v) == Status::Ok, "huge finite scales are valid");
        std::vector<float> xf(256, 1.0f); ActRow x; quantize_activations(xf.data(), 256, x);
        for (Tier tr : tiers_here()) {
            std::vector<float> y(32, 1.0f);
            const Status st = matvec(v, x, y.data(), 32, tr, nullptr);
            bool zeros = true; for (float f : y) if (f != 0.0f) zeros = false;
            CHECK(st == Status::NonFinite && zeros, "overflow reported and zeroed, tier %s mode %d", tier_name(tr), int(m));
        }
        // Large but safe scales still compute normally.
        for (auto& s : h.scales) s = 1e30f;
        view_of(h, v);
        std::vector<float> y(32);
        CHECK(matvec(v, x, y.data(), 32, Tier::Scalar, nullptr) == Status::Ok && std::isfinite(y[0]), "1e30 scales ok");
    }
}

static void test_dispatch() {
    const Tier best = detect_tier();
    std::printf("detected tier: %s\n", tier_name(best));
    CHECK(tier_supported(best), "detected tier is supported");
#if defined(_WIN32)
    _putenv_s("PCHIE_TERNARY_TIER", "scalar");
#else
    setenv("PCHIE_TERNARY_TIER", "scalar", 1);
#endif
    CHECK(detect_tier() == Tier::Scalar, "override lowers to scalar");
#if defined(_WIN32)
    _putenv_s("PCHIE_TERNARY_TIER", "neon");
#else
    setenv("PCHIE_TERNARY_TIER", "neon", 1);
#endif
    const Tier t2 = detect_tier();
    CHECK(tier_supported(t2), "override to an unsupported tier is ignored (%s)", tier_name(t2));
#if defined(_WIN32)
    _putenv_s("PCHIE_TERNARY_TIER", "");
#else
    unsetenv("PCHIE_TERNARY_TIER");
#endif
}

int main() {
    test_packing();
    test_quantizers();
    test_format();
    test_reference_math();
    test_kernels();
    test_edge_scales();
    test_dispatch();
    std::printf("pchie_ternary_test: %d PASS / %d FAIL\n", passes, fails);
    return fails == 0 ? 0 : 1;
}
