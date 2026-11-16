// Bolt PCHIE - ternary weights and int8 activations: format, quantizers, scalar reference (clean-room).
// See include/pchie/pchie_ternary.h for the format and docs/ternary/README.md for provenance.
#include "pchie_ternary_internal.h"
#include "pchie/pchie_q4_det.h"   // round_half_even_f32 (the repo's pinned rounding rule)
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if PCHIE_TERNARY_ARM64 && defined(__linux__)
#include <sys/auxv.h>
#endif
#if PCHIE_TERNARY_X86
#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace pchie {
namespace ternary {

using detail::load_f32_le;

const char* status_name(Status s) noexcept {
    switch (s) {
        case Status::Ok: return "ok";
        case Status::BadArgument: return "bad argument";
        case Status::TooLarge: return "too large";
        case Status::Truncated: return "truncated";
        case Status::BadMagic: return "bad magic";
        case Status::BadVersion: return "bad version";
        case Status::BadHeader: return "bad header";
        case Status::BadChecksum: return "bad checksum";
        case Status::BadCode: return "bad code";
        case Status::BadScale: return "bad scale";
        case Status::NonFinite: return "non-finite value";
    }
    return "unknown";
}

const char* tier_name(Tier t) noexcept {
    switch (t) {
        case Tier::Scalar: return "scalar";
        case Tier::SSSE3: return "ssse3";
        case Tier::AVX2: return "avx2";
        case Tier::NEON: return "neon";
    }
    return "unknown";
}

uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h) noexcept {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

uint64_t row_bytes_for(Packing p, uint64_t cols) noexcept {
    if (cols == 0 || cols % kBlock != 0 || cols > kHardMaxCols) return 0;
    if (p == Packing::I2) return cols / 4;
    if (p == Packing::B3) return (cols + 4) / 5;
    return 0;
}
uint64_t Tensor::row_bytes() const noexcept { return row_bytes_for(packing, cols); }

float View::scale_at(uint64_t i) const noexcept { return load_f32_le(scales_raw + 4 * i); }

static bool valid_packing(uint8_t p) { return p == uint8_t(Packing::I2) || p == uint8_t(Packing::B3); }
static bool valid_mode(uint8_t m) { return m == uint8_t(ScaleMode::PerTensor) || m == uint8_t(ScaleMode::PerBlock); }

static Status check_shape(uint64_t rows, uint64_t cols, const Limits& lim) {
    if (rows == 0 || cols == 0 || cols % kBlock != 0) return Status::BadArgument;
    if (rows > kHardMaxRows || cols > kHardMaxCols || rows > lim.max_rows || cols > lim.max_cols) return Status::TooLarge;
    return Status::Ok;
}
static uint64_t scale_count(ScaleMode m, uint64_t rows, uint64_t cols) {
    return m == ScaleMode::PerTensor ? 1 : rows * (cols / kBlock);   // rows, cols <= 2^20: no overflow
}

// ---------------------------------------------------------------------------------------------------------
// Packing

Status pack_row(const int8_t* q, uint64_t cols, Packing p, uint8_t* dst) noexcept {
    if (!q || !dst || row_bytes_for(p, cols) == 0) return Status::BadArgument;
    for (uint64_t i = 0; i < cols; ++i) if (q[i] < -1 || q[i] > 1) return Status::BadCode;
    if (p == Packing::I2) {
        for (uint64_t b = 0; b < cols / kBlock; ++b) {
            const int8_t* qb = q + b * kBlock;
            uint8_t* db = dst + b * kBlockBytes;
            for (int j = 0; j < kBlockBytes; ++j) {
                uint8_t byte = 0;
                for (int k = 0; k < 4; ++k) byte = uint8_t(byte | (uint8_t(qb[k * 32 + j] + 1) << (2 * k)));
                db[j] = byte;
            }
        }
    } else {
        const uint64_t nbytes = (cols + 4) / 5;
        for (uint64_t i = 0; i < nbytes; ++i) {
            unsigned v = 0, mul = 1;
            for (int k = 0; k < 5; ++k) {
                const uint64_t idx = i * 5 + uint64_t(k);
                const unsigned t = idx < cols ? unsigned(q[idx] + 1) : 1u;   // padding weight is 0 (trit 1)
                v += t * mul; mul *= 3;
            }
            dst[i] = uint8_t(v);
        }
    }
    return Status::Ok;
}

static bool decode_b3_byte(uint8_t byte, int8_t out[5]) noexcept {
    if (byte > 242) return false;
    unsigned v = byte;
    for (int k = 0; k < 5; ++k) { out[k] = int8_t(int(v % 3) - 1); v /= 3; }
    return true;
}

Status unpack_row(const uint8_t* src, uint64_t cols, Packing p, int8_t* q) noexcept {
    if (!src || !q || row_bytes_for(p, cols) == 0) return Status::BadArgument;
    if (p == Packing::I2) {
        for (uint64_t b = 0; b < cols / kBlock; ++b) {
            const uint8_t* sb = src + b * kBlockBytes;
            for (int j = 0; j < kBlockBytes; ++j)
                for (int k = 0; k < 4; ++k) {
                    const int code = (sb[j] >> (2 * k)) & 3;
                    if (code == 3) return Status::BadCode;
                    q[b * kBlock + uint64_t(k * 32 + j)] = int8_t(code - 1);
                }
        }
    } else {
        const uint64_t nbytes = (cols + 4) / 5;
        for (uint64_t i = 0; i < nbytes; ++i) {
            int8_t t[5];
            if (!decode_b3_byte(src[i], t)) return Status::BadCode;
            for (int k = 0; k < 5; ++k) {
                const uint64_t idx = i * 5 + uint64_t(k);
                if (idx < cols) q[idx] = t[k];
                else if (t[k] != 0) return Status::BadCode;   // padding must be zero
            }
        }
    }
    return Status::Ok;
}

// Checks every code of a packed region without unpacking into memory.
static Status validate_codes(const uint8_t* codes, uint64_t rows, uint64_t cols, Packing p) noexcept {
    const uint64_t rb = row_bytes_for(p, cols);
    if (p == Packing::I2) {
        const uint64_t n = rows * rb;
        for (uint64_t i = 0; i < n; ++i) {
            const uint8_t b = codes[i];
            // a code of 3 has both bits of its pair set
            if ((b & (b >> 1) & 0x55u) != 0) return Status::BadCode;
        }
        return Status::Ok;
    }
    const uint64_t tail = cols % 5;   // weights used in the last byte of each row (0 = all five)
    for (uint64_t r = 0; r < rows; ++r) {
        const uint8_t* row = codes + r * rb;
        for (uint64_t i = 0; i < rb; ++i) if (row[i] > 242) return Status::BadCode;
        if (tail != 0) {
            int8_t t[5];
            decode_b3_byte(row[rb - 1], t);
            for (uint64_t k = tail; k < 5; ++k) if (t[k] != 0) return Status::BadCode;
        }
    }
    return Status::Ok;
}

// ---------------------------------------------------------------------------------------------------------
// Quantizers

Status quantize_absmean(const float* W, uint64_t rows, uint64_t cols, ScaleMode mode, Packing packing,
                        Tensor& out, const Limits& lim) {
    out = Tensor();
    if (!W || !valid_packing(uint8_t(packing)) || !valid_mode(uint8_t(mode))) return Status::BadArgument;
    Status st = check_shape(rows, cols, lim);
    if (st != Status::Ok) return st;
    const uint64_t rb = row_bytes_for(packing, cols);
    if (rows * rb > lim.max_bytes) return Status::TooLarge;
    const uint64_t n = rows * cols;
    for (uint64_t i = 0; i < n; ++i) if (!std::isfinite(W[i])) return Status::NonFinite;

    const uint64_t nb = cols / kBlock;
    std::vector<float> scales(static_cast<size_t>(scale_count(mode, rows, cols)));
    auto absmean = [&](uint64_t start, uint64_t stride_rows, uint64_t len_cols, uint64_t r0, uint64_t r1) {
        double s = 0.0;   // sequential order: row by row, column by column
        for (uint64_t r = r0; r < r1; ++r)
            for (uint64_t c = 0; c < len_cols; ++c) s += std::fabs(double(W[r * stride_rows + start + c]));
        const double cnt = double(r1 - r0) * double(len_cols);
        const float g = float(s / cnt);
        return g >= FLT_MIN ? g : 0.0f;   // subnormal: an exact zero group, so inference never does denormal math
    };
    if (mode == ScaleMode::PerTensor) scales[0] = absmean(0, cols, cols, 0, rows);
    else for (uint64_t r = 0; r < rows; ++r) for (uint64_t b = 0; b < nb; ++b)
        scales[static_cast<size_t>(r * nb + b)] = absmean(b * kBlock, cols, kBlock, r, r + 1);

    std::vector<int8_t> q(static_cast<size_t>(cols));
    std::vector<uint8_t> codes(static_cast<size_t>(rows * rb));
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t c = 0; c < cols; ++c) {
            const float g = mode == ScaleMode::PerTensor ? scales[0] : scales[static_cast<size_t>(r * nb + c / kBlock)];
            int8_t v = 0;
            if (g > 0.0f) {
                const double t = double(W[r * cols + c]) / double(g);   // exact comparison against +-0.5
                v = t > 0.5 ? 1 : (t < -0.5 ? -1 : 0);                  // ties (+-0.5) go to 0, the even value
            }
            q[static_cast<size_t>(c)] = v;
        }
        st = pack_row(q.data(), cols, packing, codes.data() + r * rb);
        if (st != Status::Ok) return st;
    }
    out.packing = packing; out.scale_mode = mode; out.rows = rows; out.cols = cols;
    out.codes = std::move(codes); out.scales = std::move(scales);
    return Status::Ok;
}

Status quantize_activations(const float* x, uint64_t n, ActRow& out) {
    out = ActRow();
    if (!x || n == 0 || n % kBlock != 0) return Status::BadArgument;
    if (n > kHardMaxCols) return Status::TooLarge;
    out.n = n;
    out.q.assign(static_cast<size_t>(n), 0);
    out.bsum.assign(static_cast<size_t>(n / kBlock), 0);
    float amax = 0.0f;
    for (uint64_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) { out.scale = 0.0f; return Status::NonFinite; }   // row stays all zero
        const float a = std::fabs(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) return Status::Ok;
    const float iscale = 127.0f / amax;
    out.scale = amax / 127.0f;
    for (uint64_t i = 0; i < n; ++i) {
        int32_t v = round_half_even_f32(x[i] * iscale);
        v = std::max(-127, std::min(127, v));
        out.q[static_cast<size_t>(i)] = int8_t(v);
        out.bsum[static_cast<size_t>(i / kBlock)] += v;
    }
    return Status::Ok;
}

// ---------------------------------------------------------------------------------------------------------
// Serialization. Header, 96 bytes, little-endian:
//   0 "PTN1" | 4 u16 version (1) | 6 u8 packing | 7 u8 scale mode | 8 u32 header bytes (96)
//  12 u32 activation mode (1 = per-row absmax int8) | 16 u64 rows | 24 u64 cols | 32 u64 scales offset (96)
//  40 u64 scale count | 48 u64 codes offset (multiple of 32) | 56 u64 codes bytes
//  64 u64 FNV-1a 64 of every byte after the header | 72..95 reserved, zero
// The file ends exactly at codes offset + codes bytes; the gap between the scales and the codes is zero.
namespace {
constexpr uint32_t kHeaderBytes = 96;
constexpr uint16_t kVersion = 1;
constexpr uint32_t kActPerRowAbsmax = 1;
void put_u16(uint8_t* p, uint16_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void put_u32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
void put_u64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }
uint16_t get_u16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t get_u32(const uint8_t* p) { uint32_t v = 0; for (int i = 3; i >= 0; --i) v = (v << 8) | p[i]; return v; }
uint64_t get_u64(const uint8_t* p) { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | p[i]; return v; }
uint64_t round_up32(uint64_t v) { return (v + 31) & ~uint64_t(31); }
}  // namespace

std::vector<uint8_t> serialize(const Tensor& t) {
    const uint64_t ns = t.scales.size();
    const uint64_t codes_off = round_up32(kHeaderBytes + 4 * ns);
    std::vector<uint8_t> buf(static_cast<size_t>(codes_off + t.codes.size()), 0);
    uint8_t* h = buf.data();
    std::memcpy(h, "PTN1", 4);
    put_u16(h + 4, kVersion); h[6] = uint8_t(t.packing); h[7] = uint8_t(t.scale_mode);
    put_u32(h + 8, kHeaderBytes); put_u32(h + 12, kActPerRowAbsmax);
    put_u64(h + 16, t.rows); put_u64(h + 24, t.cols);
    put_u64(h + 32, kHeaderBytes); put_u64(h + 40, ns);
    put_u64(h + 48, codes_off); put_u64(h + 56, t.codes.size());
    for (uint64_t i = 0; i < ns; ++i) { uint32_t u; std::memcpy(&u, &t.scales[static_cast<size_t>(i)], 4); put_u32(h + kHeaderBytes + 4 * i, u); }
    if (!t.codes.empty()) std::memcpy(h + codes_off, t.codes.data(), t.codes.size());
    put_u64(h + 64, fnv1a64(h + kHeaderBytes, buf.size() - kHeaderBytes));
    return buf;
}

static Status check_scales(const uint8_t* raw, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) {
        const float s = load_f32_le(raw + 4 * i);
        if (!std::isfinite(s) || s < 0.0f) return Status::BadScale;   // -0.0 compares equal to 0: allowed
        if (s != 0.0f && s < FLT_MIN) return Status::BadScale;          // subnormal: results would depend on FTZ/DAZ
    }
    return Status::Ok;
}

Status parse(const uint8_t* buf, size_t len, const Limits& lim, View& out) {
    out = View();
    if (!buf) return Status::BadArgument;
    if (uint64_t(len) > lim.max_bytes) return Status::TooLarge;
    if (len < kHeaderBytes) return Status::Truncated;
    if (std::memcmp(buf, "PTN1", 4) != 0) return Status::BadMagic;
    if (get_u16(buf + 4) != kVersion) return Status::BadVersion;
    const uint8_t pk = buf[6], md = buf[7];
    if (!valid_packing(pk) || !valid_mode(md)) return Status::BadHeader;
    if (get_u32(buf + 8) != kHeaderBytes || get_u32(buf + 12) != kActPerRowAbsmax) return Status::BadHeader;
    for (int i = 72; i < 96; ++i) if (buf[i] != 0) return Status::BadHeader;
    const uint64_t rows = get_u64(buf + 16), cols = get_u64(buf + 24);
    const uint64_t s_off = get_u64(buf + 32), s_cnt = get_u64(buf + 40);
    const uint64_t c_off = get_u64(buf + 48), c_bytes = get_u64(buf + 56);
    Status st = check_shape(rows, cols, lim);
    if (st == Status::BadArgument) return Status::BadHeader;
    if (st != Status::Ok) return st;
    const Packing p = Packing(pk); const ScaleMode m = ScaleMode(md);
    const uint64_t rb = row_bytes_for(p, cols);
    // rows, cols <= 2^20, rb <= 2^18: every product below fits in 64 bits.
    if (s_off != kHeaderBytes || s_cnt != scale_count(m, rows, cols)) return Status::BadHeader;
    const uint64_t s_end = s_off + 4 * s_cnt;
    if (c_off != round_up32(s_end) || c_bytes != rows * rb) return Status::BadHeader;
    if (c_off > uint64_t(len) || c_bytes != uint64_t(len) - c_off) return uint64_t(len) < c_off + c_bytes ? Status::Truncated : Status::BadHeader;
    for (uint64_t i = s_end; i < c_off; ++i) if (buf[i] != 0) return Status::BadHeader;
    if (get_u64(buf + 64) != fnv1a64(buf + kHeaderBytes, len - kHeaderBytes)) return Status::BadChecksum;
    st = check_scales(buf + s_off, s_cnt);
    if (st != Status::Ok) return st;
    st = validate_codes(buf + c_off, rows, cols, p);
    if (st != Status::Ok) return st;
    out.packing = p; out.scale_mode = m; out.rows = rows; out.cols = cols;
    out.codes = buf + c_off; out.scales_raw = buf + s_off; out.row_bytes = rb;
    return Status::Ok;
}

Status view_of(const Tensor& t, View& out) {
    out = View();
    if (!valid_packing(uint8_t(t.packing)) || !valid_mode(uint8_t(t.scale_mode))) return Status::BadArgument;
    Limits lim; lim.max_rows = kHardMaxRows; lim.max_cols = kHardMaxCols;
    Status st = check_shape(t.rows, t.cols, lim);
    if (st != Status::Ok) return st;
    const uint64_t rb = row_bytes_for(t.packing, t.cols);
    if (t.codes.size() != t.rows * rb || t.scales.size() != scale_count(t.scale_mode, t.rows, t.cols)) return Status::BadArgument;
    st = check_scales(reinterpret_cast<const uint8_t*>(t.scales.data()), t.scales.size());
    if (st != Status::Ok) return st;
    st = validate_codes(t.codes.data(), t.rows, t.cols, t.packing);
    if (st != Status::Ok) return st;
    out.packing = t.packing; out.scale_mode = t.scale_mode; out.rows = t.rows; out.cols = t.cols;
    out.codes = t.codes.data(); out.scales_raw = reinterpret_cast<const uint8_t*>(t.scales.data()); out.row_bytes = rb;
    return Status::Ok;
}

Status transcode(const Tensor& in, Packing to, Tensor& out) {
    out = Tensor();
    View v;
    Status st = view_of(in, v);
    if (st != Status::Ok) return st;
    if (!valid_packing(uint8_t(to))) return Status::BadArgument;
    const uint64_t rb = row_bytes_for(to, in.cols);
    std::vector<int8_t> q(static_cast<size_t>(in.cols));
    std::vector<uint8_t> codes(static_cast<size_t>(in.rows * rb));
    for (uint64_t r = 0; r < in.rows; ++r) {
        st = unpack_row(in.codes.data() + r * v.row_bytes, in.cols, in.packing, q.data());
        if (st != Status::Ok) return st;
        st = pack_row(q.data(), in.cols, to, codes.data() + r * rb);
        if (st != Status::Ok) return st;
    }
    out.packing = to; out.scale_mode = in.scale_mode; out.rows = in.rows; out.cols = in.cols;
    out.codes = std::move(codes); out.scales = in.scales;
    return Status::Ok;
}

// ---------------------------------------------------------------------------------------------------------
// Reference compute

static bool row_ok(const View& w, uint64_t row, const ActRow& x) noexcept {
    return w.codes && w.scales_raw && row < w.rows && x.n == w.cols && x.q.size() == w.cols &&
           x.bsum.size() == w.cols / kBlock;
}

float dot_row_reference(const View& w, uint64_t row, const ActRow& x) noexcept {
    if (!row_ok(w, row, x)) return 0.0f;
    const uint64_t nb = w.cols / kBlock;
    const uint8_t* rc = w.codes + row * w.row_bytes;
    int8_t blockq[kBlock];
    auto block_isum = [&](uint64_t b) -> int32_t {
        if (w.packing == Packing::I2) return detail::block_dot_i2_reference(rc + b * kBlockBytes, x.q.data() + b * kBlock);
        // B3: decode the 128 weights of this block straight from the row bytes
        for (int i = 0; i < kBlock; ++i) {
            const uint64_t idx = b * kBlock + uint64_t(i);
            unsigned v = rc[idx / 5];
            for (uint64_t k = 0; k < idx % 5; ++k) v /= 3;
            blockq[i] = int8_t(int(v % 3) - 1);
        }
        int32_t s = 0;
        for (int i = 0; i < kBlock; ++i) s += int32_t(blockq[i]) * int32_t(x.q[static_cast<size_t>(b * kBlock + uint64_t(i))]);
        return s;
    };
    if (w.scale_mode == ScaleMode::PerTensor) {
        int32_t s = 0;
        for (uint64_t b = 0; b < nb; ++b) s += block_isum(b);
        return detail::tail_tensor(s, w.scale_at(0), x.scale);
    }
    float acc = 0.0f;
    for (uint64_t b = 0; b < nb; ++b) acc = detail::tail_block_step(acc, block_isum(b), w.scale_at(row * nb + b));
    return acc * x.scale;
}

// ---------------------------------------------------------------------------------------------------------
// Dispatch

#if PCHIE_TERNARY_X86
static void cpuid(uint32_t leaf, uint32_t sub, uint32_t r[4]) {
#if defined(_MSC_VER)
    int t[4]; __cpuidex(t, int(leaf), int(sub)); for (int i = 0; i < 4; ++i) r[i] = uint32_t(t[i]);
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (!__get_cpuid_count(leaf, sub, &a, &b, &c, &d)) { a = b = c = d = 0; }
    r[0] = a; r[1] = b; r[2] = c; r[3] = d;
#endif
}
static uint64_t xgetbv0() {
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    uint32_t lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (uint64_t(hi) << 32) | lo;
#endif
}
static Tier probe_hw_tier() noexcept {
    uint32_t r[4];
    cpuid(0, 0, r);
    const uint32_t max_leaf = r[0];
    if (max_leaf < 1) return Tier::Scalar;
    cpuid(1, 0, r);
    const bool ssse3 = (r[2] >> 9) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!ssse3) return Tier::Scalar;
    bool avx2 = false;
    if (max_leaf >= 7 && osxsave && avx && (xgetbv0() & 6) == 6) {
        cpuid(7, 0, r);
        avx2 = (r[1] >> 5) & 1;
    }
    return avx2 ? Tier::AVX2 : Tier::SSSE3;
}
#elif PCHIE_TERNARY_ARM64
// NEON is part of every AArch64 CPU. A build that uses the dot-product instructions (-mcpu with +dotprod)
// also needs the CPU to have them; on Linux and Android the kernel reports that in AT_HWCAP (ASIMDDP, bit 20).
static Tier probe_hw_tier() noexcept {
#if defined(__ARM_FEATURE_DOTPROD) && defined(__linux__)
    if ((getauxval(AT_HWCAP) & (1ul << 20)) == 0) return Tier::Scalar;
#endif
    return Tier::NEON;
}
#else
static Tier probe_hw_tier() noexcept { return Tier::Scalar; }
#endif

// The probe runs CPUID and XGETBV (or getauxval), which are slow and, under some hypervisors, trap: about a
// microsecond per call on Windows. dot_row asks for every row, so probe once. The answer cannot change while the
// process runs, and a function-local static is initialised exactly once even with several threads.
static Tier hw_tier() noexcept {
    static const Tier cached = probe_hw_tier();
    return cached;
}

bool tier_supported(Tier t) noexcept {
    if (t == Tier::Scalar) return true;
    if (!detail::simd_compiled(t)) return false;
    const Tier hw = hw_tier();
    if (t == Tier::NEON) return hw == Tier::NEON;
    if (hw == Tier::NEON) return false;
    return int(t) <= int(hw);
}

Tier detect_tier() noexcept {
    Tier best = Tier::Scalar;
    for (Tier t : {Tier::NEON, Tier::AVX2, Tier::SSSE3})
        if (tier_supported(t)) { best = t; break; }
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* env = std::getenv("PCHIE_TERNARY_TIER");   // read once at the call; can only lower the tier
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    if (env) {
        Tier want = best;
        if (std::strcmp(env, "scalar") == 0) want = Tier::Scalar;
        else if (std::strcmp(env, "ssse3") == 0) want = Tier::SSSE3;
        else if (std::strcmp(env, "avx2") == 0) want = Tier::AVX2;
        else if (std::strcmp(env, "neon") == 0) want = Tier::NEON;
        if (want == Tier::Scalar || (tier_supported(want) && int(want) <= int(best))) best = want;
    }
    return best;
}

float dot_row(const View& w, uint64_t row, const ActRow& x, Tier tier) noexcept {
    if (!row_ok(w, row, x)) return 0.0f;
    if (w.packing != Packing::I2 || !tier_supported(tier)) return dot_row_reference(w, row, x);
    switch (tier) {
        case Tier::SSSE3: return detail::row_dot_i2_ssse3(w, row, x);
        case Tier::AVX2: return detail::row_dot_i2_avx2(w, row, x);
        case Tier::NEON: return detail::row_dot_i2_neon(w, row, x);
        default: return dot_row_reference(w, row, x);
    }
}

Status matvec(const View& w, const ActRow& x, float* y, uint64_t y_len, Tier tier, IJobDispatcher* pool) {
    if (!y) return Status::BadArgument;
    if (!w.codes || !w.scales_raw || y_len != w.rows || x.n != w.cols || x.q.size() != w.cols ||
        x.bsum.size() != w.cols / kBlock) {
        for (uint64_t i = 0; i < y_len; ++i) y[i] = 0.0f;
        return Status::BadArgument;
    }
    if (!tier_supported(tier)) tier = Tier::Scalar;
    constexpr uint64_t CH = 16;
    const uint64_t rows = w.rows;
    const int chunks = int((rows + CH - 1) / CH);   // rows <= 2^20, so chunks fits in int
    auto work = [&](int c) {
        const uint64_t r0 = uint64_t(c) * CH, r1 = std::min(rows, r0 + CH);
        for (uint64_t r = r0; r < r1; ++r) y[r] = dot_row(w, r, x, tier);
    };
    if (pool) pool->parallel_for(chunks, work);
    else for (int c = 0; c < chunks; ++c) work(c);
    // Finite weights, scales and activations can still overflow float (scales near FLT_MAX). Report it and leave
    // a safe all-zero output rather than infinities. One pass over the outputs, after the parallel phase.
    for (uint64_t i = 0; i < y_len; ++i)
        if (!std::isfinite(y[i])) {
            for (uint64_t j = 0; j < y_len; ++j) y[j] = 0.0f;
            return Status::NonFinite;
        }
    return Status::Ok;
}

}  // namespace ternary
}  // namespace pchie
