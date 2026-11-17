// Bolt PCHIE - ternary research path: converter fits, B3B and TL3 packings, scalar kernels, integer softmax and
// RMSNorm (clean-room). See include/pchie/pchie_ternary_research.h. Research only: no model uses this file.
#include "pchie/pchie_ternary_research.h"
#include "pchie_ternary_internal.h"
#include "pchie_ternary_research_internal.h"
#include "pchie_ternary_research_tables.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <functional>

namespace pchie {
namespace ternary {
namespace research {

// ------------------------------------------------------------------------------------------------ helpers
int64_t rshift_rne(int64_t v, int s) noexcept {
    if (s <= 0) return v;
    if (s > 62) return 0;                                    // never used with s > 62; documented safe value
    const uint64_t mask = (uint64_t(1) << s) - 1;
    const uint64_t r = uint64_t(v) & mask;                   // two's complement remainder, 0 <= r < 2^s
    int64_t q = v >> s;                                      // arithmetic shift (floor) on every target
    const uint64_t half = uint64_t(1) << (s - 1);
    if (r > half || (r == half && (q & 1))) ++q;
    return q;
}
static uint64_t div_rne_u64(uint64_t num, uint64_t den) noexcept {   // den > 0
    uint64_t q = num / den, r = num % den;
    if (r > den - r || (r == den - r && (q & 1))) ++q;
    return q;
}
static uint64_t splitmix64(uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// ------------------------------------------------------------------------------------------------ converter
Status ternarize_group(const float* w, size_t n, Fit fit, Round rnd, uint64_t seed, int lloyd_iters,
                       float* scratch, int8_t* q, float* gamma) noexcept {
    if (gamma) *gamma = 0.0f;
    if (q && n > 0 && n <= kMaxGroup) std::memset(q, 0, n);
    if (!w || !q || !gamma || n == 0 || n > kMaxGroup) return Status::BadArgument;
    if (uint8_t(fit) > 2 || uint8_t(rnd) > 2 || lloyd_iters < 0 || lloyd_iters > 64) return Status::BadArgument;
    if (fit == Fit::LeastSquares && !scratch) return Status::BadArgument;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(w[i])) return Status::NonFinite;
        s += std::fabs(double(w[i]));
    }
    double g = s / double(n);
    if (fit == Fit::LeastSquares) {
        // For a fixed set of k non-zeros the best gamma is their mean |w| and the error is
        // sum w^2 - (sum of those |w|)^2 / k, so the best set is the k largest |w| and the best k maximizes P_k^2 / k.
        for (size_t i = 0; i < n; ++i) scratch[i] = std::fabs(w[i]);
        std::sort(scratch, scratch + n, std::greater<float>());
        double p = 0.0, best = 0.0; size_t bk = 0; double bp = 0.0;
        for (size_t k = 1; k <= n; ++k) {
            p += double(scratch[k - 1]);
            const double v = p * p / double(k);
            if (v > best) { best = v; bk = k; bp = p; }
        }
        g = bk ? bp / double(bk) : 0.0;
    } else if (fit == Fit::Lloyd) {
        for (int it = 0; it < lloyd_iters; ++it) {
            const double thr = 0.5 * g; double sum = 0.0; size_t cnt = 0;
            for (size_t i = 0; i < n; ++i) { const double a = std::fabs(double(w[i])); if (a > thr) { sum += a; ++cnt; } }
            if (cnt == 0) break;
            const double ng = sum / double(cnt);
            if (ng == g) break;
            g = ng;
        }
    }
    float gf = float(g);
    if (!(gf >= FLT_MIN)) gf = 0.0f;   // subnormal or zero: an exact zero group (no denormal math at inference)
    *gamma = gf;
    if (gf == 0.0f) return Status::Ok;
    const double gd = double(gf);
    for (size_t i = 0; i < n; ++i) {
        const double t = double(w[i]) / gd;   // both operands exact in double; the +-0.5 comparison is exact
        int8_t v = 0;
        if (rnd == Round::HalfEven) v = t > 0.5 ? 1 : (t < -0.5 ? -1 : 0);
        else if (rnd == Round::HalfAway) v = t >= 0.5 ? 1 : (t <= -0.5 ? -1 : 0);
        else {
            const double a = std::min(1.0, std::fabs(t));
            const double u = double(splitmix64(seed ^ (uint64_t(i) * 0xD1B54A32D192ED03ull)) >> 11) * (1.0 / 9007199254740992.0);
            v = u < a ? (t > 0 ? 1 : -1) : 0;
        }
        q[i] = v;
    }
    return Status::Ok;
}

// ------------------------------------------------------------------------------------------------ packings
static bool scales_ok(const std::vector<float>& s) noexcept {
    // Same rule as the main path (57cf857): subnormal scales are refused, so no result depends on FTZ/DAZ.
    for (float v : s) if (!std::isfinite(v) || v < 0.0f || (v != 0.0f && v < FLT_MIN)) return false;
    return true;
}
static uint64_t expect_scales(ScaleMode m, uint64_t rows, uint64_t cols) noexcept {
    return m == ScaleMode::PerTensor ? 1 : rows * (cols / kBlock);
}
static void tl3_encode(int a, int b, int c, uint8_t& idx, uint8_t& sign) noexcept {
    const int v = 9 * (a + 1) + 3 * (b + 1) + (c + 1);
    if (v > 13) { sign = 1; idx = uint8_t(26 - v); } else { sign = 0; idx = uint8_t(v); }
}
static void tl3_decode(uint8_t idx, uint8_t sign, int8_t out[3]) noexcept {
    int a = idx / 9 - 1, b = (idx / 3) % 3 - 1, c = idx % 3 - 1;
    if (sign) { a = -a; b = -b; c = -c; }
    out[0] = int8_t(a); out[1] = int8_t(b); out[2] = int8_t(c);
}
uint64_t tl3_offset(uint64_t tile, uint64_t nb, uint64_t b, int t) noexcept {
    return ((tile * nb + b) * uint64_t(kTL3Triples) + uint64_t(t)) * uint64_t(kTL3TripleBytes);
}

Status pack_b3b(const Tensor& i2, PackedTensor& out) {
    out = PackedTensor();
    View v; Status st = view_of(i2, v);
    if (st != Status::Ok) return st;
    if (i2.packing != Packing::I2) return Status::BadArgument;
    const uint64_t nb = i2.cols / kBlock;
    std::vector<int8_t> q(size_t(i2.cols));
    std::vector<uint8_t> codes(size_t(i2.rows * nb * kB3BBlockBytes));
    for (uint64_t r = 0; r < i2.rows; ++r) {
        st = ternary::unpack_row(i2.codes.data() + r * v.row_bytes, i2.cols, Packing::I2, q.data());
        if (st != Status::Ok) return st;
        for (uint64_t b = 0; b < nb; ++b) {
            uint8_t* dst = codes.data() + (r * nb + b) * kB3BBlockBytes;
            for (int j = 0; j < kB3BBlockBytes; ++j) {
                unsigned val = 0, mul = 1;
                for (int k = 0; k < 5; ++k) {
                    const int idx = 26 * k + j;
                    const unsigned t = idx < kBlock ? unsigned(q[size_t(b * kBlock + uint64_t(idx))] + 1) : 1u;
                    val += t * mul; mul *= 3;
                }
                dst[j] = uint8_t(val);
            }
        }
    }
    out.kind = PackedTensor::Kind::B3B; out.scale_mode = i2.scale_mode; out.rows = i2.rows; out.cols = i2.cols;
    out.codes = std::move(codes); out.scales = i2.scales;
    return Status::Ok;
}

Status pack_tl3(const Tensor& i2, PackedTensor& out) {
    out = PackedTensor();
    View v; Status st = view_of(i2, v);
    if (st != Status::Ok) return st;
    if (i2.packing != Packing::I2 || i2.rows % kTL3TileRows != 0) return Status::BadArgument;
    const uint64_t nb = i2.cols / kBlock, tiles = i2.rows / kTL3TileRows;
    std::vector<int8_t> q(size_t(i2.cols) * kTL3TileRows);
    std::vector<uint8_t> codes(size_t(tiles * nb * kTL3BlockBytes), 0);
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        for (int r = 0; r < kTL3TileRows; ++r) {
            st = ternary::unpack_row(i2.codes.data() + (tile * kTL3TileRows + uint64_t(r)) * v.row_bytes, i2.cols,
                                     Packing::I2, q.data() + size_t(r) * size_t(i2.cols));
            if (st != Status::Ok) return st;
        }
        for (uint64_t b = 0; b < nb; ++b)
            for (int t = 0; t < kTL3Triples; ++t) {
                uint8_t* dst = codes.data() + tl3_offset(tile, nb, b, t);
                for (int r = 0; r < kTL3TileRows; ++r) {
                    const int8_t* qr = q.data() + size_t(r) * size_t(i2.cols) + size_t(b * kBlock);
                    const int a = qr[3 * t], bb = qr[3 * t + 1], c = (3 * t + 2) < kBlock ? qr[3 * t + 2] : 0;
                    uint8_t idx, sign; tl3_encode(a, bb, c, idx, sign);
                    if (r < 16) dst[r] = uint8_t(dst[r] | idx); else dst[r - 16] = uint8_t(dst[r - 16] | (idx << 4));
                    if (sign) dst[16 + r / 8] = uint8_t(dst[16 + r / 8] | (1u << (r % 8)));
                }
            }
    }
    out.kind = PackedTensor::Kind::TL3; out.scale_mode = i2.scale_mode; out.rows = i2.rows; out.cols = i2.cols;
    out.codes = std::move(codes); out.scales = i2.scales;
    return Status::Ok;
}

Status validate(const PackedTensor& t) noexcept {
    if (t.rows == 0 || t.cols == 0 || t.cols % kBlock != 0) return Status::BadArgument;
    if (t.rows > kHardMaxRows || t.cols > kHardMaxCols) return Status::TooLarge;
    if (t.scale_mode != ScaleMode::PerTensor && t.scale_mode != ScaleMode::PerBlock) return Status::BadArgument;
    if (t.scales.size() != expect_scales(t.scale_mode, t.rows, t.cols)) return Status::BadArgument;
    if (!scales_ok(t.scales)) return Status::BadScale;
    const uint64_t nb = t.cols / kBlock;
    if (t.kind == PackedTensor::Kind::B3B) {
        if (t.codes.size() != t.rows * nb * kB3BBlockBytes) return Status::BadArgument;
        for (uint64_t i = 0; i < t.codes.size(); ++i) {
            const uint8_t c = t.codes[size_t(i)];
            if (c > 242) return Status::BadCode;
            const int j = int(i % kB3BBlockBytes);
            if (j >= 24 && c / 81 != 1) return Status::BadCode;   // slots past weight 127 must be the zero weight
        }
        return Status::Ok;
    }
    if (t.kind != PackedTensor::Kind::TL3) return Status::BadArgument;
    if (t.rows % kTL3TileRows != 0) return Status::BadArgument;
    const uint64_t tiles = t.rows / kTL3TileRows;
    if (t.codes.size() != tiles * nb * kTL3BlockBytes) return Status::BadArgument;
    for (uint64_t tile = 0; tile < tiles; ++tile)
        for (uint64_t b = 0; b < nb; ++b)
            for (int tr = 0; tr < kTL3Triples; ++tr) {
                const uint8_t* p = t.codes.data() + tl3_offset(tile, nb, b, tr);
                for (int r = 0; r < kTL3TileRows; ++r) {
                    const uint8_t idx = r < 16 ? (p[r] & 15) : (p[r - 16] >> 4);
                    const uint8_t sign = (p[16 + r / 8] >> (r % 8)) & 1;
                    if (idx > 13) return Status::BadCode;
                    if (idx == 13 && sign) return Status::BadCode;            // non-canonical zero
                    if (tr == kTL3Triples - 1 && idx % 3 != 1) return Status::BadCode;   // padding weight not zero
                }
            }
    return Status::Ok;
}

Status unpack_row(const PackedTensor& t, uint64_t row, int8_t* q) noexcept {
    if (!q || row >= t.rows || t.cols % kBlock != 0 || t.cols == 0) return Status::BadArgument;
    const uint64_t nb = t.cols / kBlock;
    if (t.kind == PackedTensor::Kind::B3B) {
        if (t.codes.size() != t.rows * nb * kB3BBlockBytes) return Status::BadArgument;
        for (uint64_t b = 0; b < nb; ++b) {
            const uint8_t* s = t.codes.data() + (row * nb + b) * kB3BBlockBytes;
            for (int j = 0; j < kB3BBlockBytes; ++j) {
                unsigned v = s[j];
                if (v > 242) return Status::BadCode;
                for (int k = 0; k < 5; ++k) {
                    const int idx = 26 * k + j;
                    if (idx < kBlock) q[b * kBlock + uint64_t(idx)] = int8_t(int(v % 3) - 1);
                    v /= 3;
                }
            }
        }
        return Status::Ok;
    }
    if (t.rows % kTL3TileRows != 0 || t.codes.size() != (t.rows / kTL3TileRows) * nb * kTL3BlockBytes) return Status::BadArgument;
    const uint64_t tile = row / kTL3TileRows; const int r = int(row % kTL3TileRows);
    for (uint64_t b = 0; b < nb; ++b)
        for (int tr = 0; tr < kTL3Triples; ++tr) {
            const uint8_t* p = t.codes.data() + tl3_offset(tile, nb, b, tr);
            const uint8_t idx = r < 16 ? (p[r] & 15) : (p[r - 16] >> 4);
            const uint8_t sign = (p[16 + r / 8] >> (r % 8)) & 1;
            if (idx > 13) return Status::BadCode;
            int8_t d[3]; tl3_decode(idx, sign, d);
            for (int k = 0; k < 3; ++k) { const int w = 3 * tr + k; if (w < kBlock) q[b * kBlock + uint64_t(w)] = d[k]; }
        }
    return Status::Ok;
}

static bool act_ok(const ActRow& x, uint64_t cols) noexcept {
    return x.n == cols && x.q.size() == cols && x.bsum.size() == cols / kBlock;
}

Status prepare_b3b(const ActRow& x, B3BAct& out) {
    out = B3BAct();
    if (x.n == 0 || x.n % kBlock != 0 || !act_ok(x, x.n)) return Status::BadArgument;
    const uint64_t nb = x.n / kBlock;
    out.xp.assign(size_t(nb * kB3BActSlots), 0);
    for (uint64_t b = 0; b < nb; ++b)
        for (int k = 0; k < 5; ++k)
            for (int j = 0; j < kB3BBlockBytes; ++j) {
                const int idx = 26 * k + j;
                if (idx < kBlock) out.xp[size_t(b * kB3BActSlots + uint64_t(k * 32 + j))] = x.q[size_t(b * kBlock + uint64_t(idx))];
            }
    return Status::Ok;
}

Status prepare_tl3(const ActRow& x, TL3Act& out) {
    out = TL3Act();
    if (x.n == 0 || x.n % kBlock != 0 || !act_ok(x, x.n)) return Status::BadArgument;
    const uint64_t nb = x.n / kBlock;
    out.lo.assign(size_t(nb * kTL3Triples * 16), 0);
    out.hi.assign(size_t(nb * kTL3Triples * 16), 0);
    for (uint64_t b = 0; b < nb; ++b)
        for (int t = 0; t < kTL3Triples; ++t) {
            const int8_t* xb = x.q.data() + b * kBlock;
            const int x0 = xb[3 * t], x1 = xb[3 * t + 1], x2 = (3 * t + 2) < kBlock ? xb[3 * t + 2] : 0;
            const size_t base = size_t((b * kTL3Triples + uint64_t(t)) * 16);
            for (int idx = 0; idx < 14; ++idx) {
                int8_t d[3]; tl3_decode(uint8_t(idx), 0, d);
                const int s = d[0] * x0 + d[1] * x1 + d[2] * x2;   // |s| <= 381: fits int16
                const uint16_t u = uint16_t(int16_t(s));
                out.lo[base + size_t(idx)] = uint8_t(u & 0xFF);
                out.hi[base + size_t(idx)] = int8_t(uint8_t(u >> 8));
            }
        }
    return Status::Ok;
}

// ------------------------------------------------------------------------------------------------ kernels
const char* kernel_name(Kernel k) noexcept {
    switch (k) {
        case Kernel::I2Scalar: return "I2 scalar (reference)";
        case Kernel::I2Avx2Biased: return "I2 AVX2 biased (shipping)";
        case Kernel::I2Avx2Signed: return "I2 AVX2 signed";
        case Kernel::I2Avx2ZeroSkip: return "I2 AVX2 biased + zero-skip";
        case Kernel::B3BScalar: return "B3B scalar";
        case Kernel::B3BAvx2: return "B3B AVX2";
        case Kernel::TL3Scalar: return "TL3 scalar";
        case Kernel::TL3Avx2: return "TL3 AVX2";
    }
    return "unknown";
}
bool kernel_available(Kernel k) noexcept {
    switch (k) {
        case Kernel::I2Scalar: case Kernel::B3BScalar: case Kernel::TL3Scalar: return true;
        case Kernel::I2Avx2Biased: case Kernel::I2Avx2Signed: case Kernel::I2Avx2ZeroSkip:
        case Kernel::B3BAvx2: case Kernel::TL3Avx2: return tier_supported(Tier::AVX2);
    }
    return false;
}

static Status finish(float* y, uint64_t n) noexcept {
    for (uint64_t i = 0; i < n; ++i)
        if (!std::isfinite(y[i])) { for (uint64_t j = 0; j < n; ++j) y[j] = 0.0f; return Status::NonFinite; }
    return Status::Ok;
}
static Status fail(float* y, uint64_t n) noexcept {
    if (y) for (uint64_t i = 0; i < n; ++i) y[i] = 0.0f;
    return Status::BadArgument;
}

Status matvec_i2(const View& w, const ActRow& x, Kernel k, float* y, uint64_t y_len) noexcept {
    if (!y || !w.codes || !w.scales_raw || w.packing != Packing::I2 || y_len != w.rows || !act_ok(x, w.cols) ||
        !kernel_available(k))
        return fail(y, y_len);
    for (uint64_t r = 0; r < w.rows; ++r) {
        switch (k) {
            case Kernel::I2Scalar: y[r] = dot_row_reference(w, r, x); break;
            case Kernel::I2Avx2Biased: y[r] = ternary::detail::row_dot_i2_avx2(w, r, x); break;   // the shipping row kernel itself
            case Kernel::I2Avx2Signed: y[r] = detail::row_i2_signed_avx2(w, r, x); break;
            case Kernel::I2Avx2ZeroSkip: y[r] = detail::row_i2_zeroskip_avx2(w, r, x); break;
            default: return fail(y, y_len);
        }
    }
    return finish(y, y_len);
}

static float row_b3b_scalar(const PackedTensor& w, uint64_t row, const ActRow& x, const B3BAct& xa) noexcept {
    const uint64_t nb = w.cols / kBlock;
    auto blk = [&](uint64_t b) {
        const uint8_t* c = w.codes.data() + (row * nb + b) * kB3BBlockBytes;
        const int8_t* xp = xa.xp.data() + b * kB3BActSlots;
        int32_t s = 0;
        for (int j = 0; j < kB3BBlockBytes; ++j) {
            unsigned v = c[j];
            for (int k = 0; k < 5; ++k) { s += (int(v % 3) - 1) * int32_t(xp[k * 32 + j]); v /= 3; }
        }
        return s;
    };
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += blk(b);
        return ternary::detail::tail_tensor(s, w.scales[0], x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b) acc = ternary::detail::tail_block_step(acc, blk(b), w.scales[size_t(row * nb + b)]);
    return acc * x.scale;
}

Status matvec_b3b(const PackedTensor& w, const ActRow& x, const B3BAct& xa, Kernel k, float* y, uint64_t y_len) noexcept {
    const uint64_t nb = w.cols / kBlock;
    if (!y || w.kind != PackedTensor::Kind::B3B || y_len != w.rows || w.cols == 0 || w.cols % kBlock != 0 ||
        !act_ok(x, w.cols) || xa.xp.size() != nb * kB3BActSlots || w.codes.size() != w.rows * nb * kB3BBlockBytes ||
        w.scales.size() != expect_scales(w.scale_mode, w.rows, w.cols) || !kernel_available(k) ||
        (k != Kernel::B3BScalar && k != Kernel::B3BAvx2))
        return fail(y, y_len);
    for (uint64_t r = 0; r < w.rows; ++r)
        y[r] = k == Kernel::B3BScalar ? row_b3b_scalar(w, r, x, xa) : detail::row_b3b_avx2(w, r, x, xa);
    return finish(y, y_len);
}

static void tile_tl3_scalar(const PackedTensor& w, uint64_t tile, const ActRow& x, const TL3Act& xa, float* y) noexcept {
    const uint64_t nb = w.cols / kBlock;
    for (int r = 0; r < kTL3TileRows; ++r) {
        const uint64_t row = tile * kTL3TileRows + uint64_t(r);
        int32_t tot = 0; float acc = 0.0f;
        for (uint64_t b = 0; b < nb; ++b) {
            int32_t s = 0;
            for (int t = 0; t < kTL3Triples; ++t) {
                const uint8_t* p = w.codes.data() + tl3_offset(tile, nb, b, t);
                const uint8_t idx = r < 16 ? (p[r] & 15) : (p[r - 16] >> 4);
                const uint8_t sign = (p[16 + r / 8] >> (r % 8)) & 1;
                const size_t e = size_t((b * kTL3Triples + uint64_t(t)) * 16 + idx);
                const int16_t v = int16_t(uint16_t(xa.lo[e]) | uint16_t(uint16_t(uint8_t(xa.hi[e])) << 8));
                s += sign ? -int32_t(v) : int32_t(v);
            }
            if (w.scale_mode == ScaleMode::PerTensor) tot += s;
            else acc = ternary::detail::tail_block_step(acc, s, w.scales[size_t(row * nb + b)]);
        }
        y[r] = w.scale_mode == ScaleMode::PerTensor ? ternary::detail::tail_tensor(tot, w.scales[0], x.scale) : acc * x.scale;
    }
}

Status matvec_tl3(const PackedTensor& w, const ActRow& x, const TL3Act& xa, Kernel k, float* y, uint64_t y_len) noexcept {
    const uint64_t nb = w.cols / kBlock;
    if (!y || w.kind != PackedTensor::Kind::TL3 || y_len != w.rows || w.rows % kTL3TileRows != 0 || w.cols == 0 ||
        w.cols % kBlock != 0 || !act_ok(x, w.cols) || xa.lo.size() != nb * kTL3Triples * 16 ||
        xa.hi.size() != nb * kTL3Triples * 16 ||
        w.codes.size() != (w.rows / kTL3TileRows) * nb * kTL3BlockBytes ||
        w.scales.size() != expect_scales(w.scale_mode, w.rows, w.cols) || !kernel_available(k) ||
        (k != Kernel::TL3Scalar && k != Kernel::TL3Avx2))
        return fail(y, y_len);
    for (uint64_t tile = 0; tile < w.rows / kTL3TileRows; ++tile) {
        if (k == Kernel::TL3Scalar) tile_tl3_scalar(w, tile, x, xa, y + tile * kTL3TileRows);
        else detail::tile_tl3_avx2(w, tile, x, xa, y + tile * kTL3TileRows);
    }
    return finish(y, y_len);
}

Status decode_row_i2_avx2(const View& w, uint64_t row, int8_t* q) noexcept {
    if (!q || !w.codes || w.packing != Packing::I2 || row >= w.rows || !tier_supported(Tier::AVX2)) return Status::BadArgument;
    detail::decode_i2_avx2(w.codes + row * w.row_bytes, w.cols / kBlock, q);
    return Status::Ok;
}
Status decode_row_b3b_avx2(const PackedTensor& w, uint64_t row, int8_t* q) noexcept {
    const uint64_t nb = w.cols / kBlock;
    if (!q || w.kind != PackedTensor::Kind::B3B || row >= w.rows || w.codes.size() != w.rows * nb * kB3BBlockBytes ||
        !tier_supported(Tier::AVX2))
        return Status::BadArgument;
    detail::decode_b3b_avx2(w, row, q);
    return Status::Ok;
}

// ------------------------------------------------------------------------------------------------ softmax
const char* exp2_name(Exp2 m) noexcept {
    switch (m) {
        case Exp2::Table256: return "256-entry table (bin-centred)";
        case Exp2::Table64Lerp: return "64-entry table + linear interpolation";
        case Exp2::Poly3: return "degree-3 polynomial (minimax relative)";
    }
    return "unknown";
}
// Degree-3 fit of 2^f on [0, 1), minimax in relative error (Lawson iteration, our own fit), Q30.
static const int64_t kPoly3[4] = {1073661380, 747146361, 242737318, 83777698};

uint32_t exp2_frac_q30(uint32_t f, Exp2 m) noexcept {
    if (f > 65535u) f = 65535u;   // out of contract: clamp, never read past a table
    switch (m) {
        case Exp2::Table256: return tables::kExp2T256[f >> 8];
        case Exp2::Table64Lerp: {
            const uint32_t i = f >> 10, r = f & 1023u;
            const int64_t a = tables::kExp2T64[i], b = tables::kExp2T64[i + 1];
            return uint32_t(a + rshift_rne((b - a) * int64_t(r), 10));
        }
        case Exp2::Poly3: {
            const int64_t x = int64_t(f) << 14;                  // Q30, < 2^30
            int64_t p = kPoly3[3];                               // every product below is < 2^62
            p = kPoly3[2] + rshift_rne(p * x, 30);
            p = kPoly3[1] + rshift_rne(p * x, 30);
            p = kPoly3[0] + rshift_rne(p * x, 30);
            return uint32_t(p);
        }
    }
    return 1u << 30;
}

Status softmax_fixed(const int32_t* z, size_t n, Exp2 m, int out_bits, uint32_t* p) noexcept {
    if (p && n > 0 && n <= kMaxSoftmax) std::memset(p, 0, n * sizeof(uint32_t));
    if (!z || !p || n == 0 || n > kMaxSoftmax || uint8_t(m) > 2 || out_bits < 8 || out_bits > 30) return Status::BadArgument;
    int32_t mx = z[0];
    for (size_t i = 1; i < n; ++i) mx = std::max(mx, z[i]);
    constexpr int64_t kFloor = -(int64_t(64) << 16);   // below -64.0 the weight is exactly 0
    uint64_t sum = 0;                                   // each e <= 2^30, n <= 2^20: sum <= 2^50
    for (size_t i = 0; i < n; ++i) {
        const int64_t d = int64_t(z[i]) - int64_t(mx);  // in [-2^32, 0]
        uint32_t e = 0;
        if (d >= kFloor) {
            const int64_t yq = rshift_rne(d * int64_t(tables::kLog2eQ30), 30);   // |d * log2e| < 2^53
            const int64_t a = -yq;                       // >= 0, < 2^23
            const int64_t ni = a >> 16, fr = a & 0xFFFF;
            int64_t v; int64_t sh;
            if (fr == 0) { v = int64_t(1) << 30; sh = ni; }
            else { v = exp2_frac_q30(uint32_t(65536 - fr), m); sh = ni + 1; }
            e = sh >= 40 ? 0u : uint32_t(rshift_rne(v, int(sh)));
        }
        p[i] = e;
        sum += e;
    }
    if (sum == 0) { std::memset(p, 0, n * sizeof(uint32_t)); return Status::BadArgument; }   // cannot happen: max gives 2^30
    // e <= 2^30 and out_bits <= 30, so e << out_bits <= 2^60; sum >= 2^30, so each p <= 2^out_bits.
    for (size_t i = 0; i < n; ++i) p[i] = uint32_t(div_rne_u64(uint64_t(p[i]) << out_bits, sum));
    return Status::Ok;
}

// ------------------------------------------------------------------------------------------------ RMSNorm
uint32_t rsqrt_q30(uint32_t u, int steps) noexcept {
    if (u < (1u << 30)) u = 1u << 30;   // out of contract: clamp into [1, 4)
    if (steps < 0) steps = 0;
    if (steps > 3) steps = 3;
    const uint32_t i = u < (1u << 31) ? ((u - (1u << 30)) >> 25) : 32u + ((u - (1u << 31)) >> 26);
    int64_t y = tables::kRsqrtSeed64[i];                     // in (2^29, 2^30]
    for (int s = 0; s < steps; ++s) {
        // y <- y (3 - u y^2) / 2. y <= 2^30, so y*y <= 2^60; u < 2^32 and y2 <= 2^30, so u*y2 < 2^62;
        // f <= 3 * 2^30, so y*f < 2^62. Every intermediate fits int64.
        const int64_t y2 = rshift_rne(y * y, 30);
        const int64_t t = rshift_rne(int64_t(u) * y2, 30);
        int64_t f = (int64_t(3) << 30) - t;
        if (f < 0) f = 0;
        y = rshift_rne(y * f, 31);
        y = std::min<int64_t>(std::max<int64_t>(y, int64_t(1) << 29), int64_t(1) << 30);
    }
    return uint32_t(y);
}

Status rmsnorm_q16(const int16_t* x, size_t n, uint32_t eps_q8, int32_t* y) noexcept {
    if (y && n > 0 && n <= kMaxNorm) std::memset(y, 0, n * sizeof(int32_t));
    if (!x || !y || n == 0 || n > kMaxNorm || eps_q8 == 0) return Status::BadArgument;
    // Bounds: |x| <= 32767 and n <= 2^20, so ss <= 32767^2 * 2^20 < 2^50.
    int64_t ss = 0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] == INT16_MIN) return Status::BadArgument;   // symmetric range only
        ss += int64_t(x[i]) * int64_t(x[i]);
    }
    // Mean with S fractional bits, S even (so the square root splits exactly) and as large as fits:
    // ss << S < 2^62, S in [12, 30] because ss < 2^50. eps_q8 << (S - 8) <= 2^32 * 2^22 = 2^54, so v < 2^63.
    int lss = 0; for (uint64_t t = uint64_t(ss); t; t >>= 1) ++lss;
    int S = 30;
    while (S > 8 && lss + S > 62) S -= 2;
    const uint64_t v = div_rne_u64(uint64_t(ss) << S, uint64_t(n)) + (uint64_t(eps_q8) << (S - 8));   // >= 2^(S-8)
    int len = 0; for (uint64_t t = v; t; t >>= 1) ++len;                                                // 1..63
    // Pick an even shift 2k so that u = v * 2^-2k lands in [2^30, 2^32).
    const int two_k = ((len - 31) % 2 == 0) ? len - 31 : len - 32;
    const uint64_t u = two_k >= 0 ? (v >> two_k) : (v << (-two_k));   // truncation: relative error < 2^-30
    const int k = two_k / 2;                                            // two_k is even, so this is exact
    const int64_t R = rsqrt_q30(uint32_t(u), 2);                        // 1/sqrt(u) = R * 2^-45
    // mean = v * 2^-S, so y = x * 2^(S/2) / sqrt(v) = x * R * 2^(S/2 - 45 - k); in Q16 the right shift is
    // 29 + k - S/2. v >= 2^(S-8) gives k >= S/2 - 19.5, so the shift is >= 9; k <= 16 and S >= 12 keep it <= 39.
    const int sh = 29 + k - S / 2;
    for (size_t i = 0; i < n; ++i) {
        int64_t r = rshift_rne(int64_t(x[i]) * R, sh);           // |x * R| < 2^45
        // |x| / rms <= sqrt(n) <= 2^10, so |y| <= 2^26 in Q16; clamp anyway (safe state, never reached).
        r = std::min<int64_t>(std::max<int64_t>(r, -(int64_t(1) << 30)), int64_t(1) << 30);
        y[i] = int32_t(r);
    }
    return Status::Ok;
}

}  // namespace research
}  // namespace ternary
}  // namespace pchie
