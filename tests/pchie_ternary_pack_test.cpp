// Bolt PCHIE - packed ternary storage tests (PTT1 / PTP1, B3, rANS, compact scales, padded fit). CPU only.
#include "pchie/pchie_ternary_pack.h"
#include "pchie/pchie_ternary.h"
#include "pchie/pchie_ternary_research.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static std::vector<int8_t> random_trits(size_t n, uint32_t seed, double p_zero) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int8_t> q(n);
    for (auto& v : q) {
        const double x = u(rng);
        v = x < p_zero ? 0 : (x < p_zero + (1.0 - p_zero) / 2 ? -1 : 1);
    }
    return q;
}

// 1. B3 rows: round trip for many widths, and the same bytes as Packing::B3 in the C++ library.
static void test_b3_rows() {
    int bad = 0, lib_diff = 0, n = 0;
    for (uint64_t cols : {1ull, 2ull, 4ull, 5ull, 6ull, 127ull, 128ull, 129ull, 255ull, 256ull, 640ull, 1000ull, 4096ull, 5376ull}) {
        for (uint32_t seed = 1; seed <= 6; ++seed) {
            const auto q = random_trits(cols, seed * 7 + uint32_t(cols), 0.1 * seed);
            std::vector<uint8_t> b(ternary_b3_row_bytes(cols));
            std::vector<int8_t> back(cols, 9);
            ++n;
            if (b.size() != (cols + 4) / 5 || ternary_pack_row_b3(q.data(), cols, b.data()) != TERNARY_CONVERT_OK ||
                ternary_unpack_row_b3(b.data(), cols, back.data()) != TERNARY_CONVERT_OK || back != q)
                ++bad;
            if (cols % 128 == 0) {
                std::vector<uint8_t> lib(pchie::ternary::row_bytes_for(pchie::ternary::Packing::B3, cols));
                if (pchie::ternary::pack_row(q.data(), cols, pchie::ternary::Packing::B3, lib.data()) != pchie::ternary::Status::Ok ||
                    lib != b)
                    ++lib_diff;
                // direct I2 <-> B3
                std::vector<uint8_t> i2(cols / 4), b2(b.size()), i2b(cols / 4);
                if (ternary_pack_row_i2(q.data(), cols, i2.data()) != TERNARY_CONVERT_OK ||
                    ternary_b3_from_i2_row(i2.data(), cols, b2.data()) != TERNARY_CONVERT_OK || b2 != b ||
                    ternary_i2_from_b3_row(b.data(), cols, i2b.data()) != TERNARY_CONVERT_OK || i2b != i2)
                    ++bad;
            }
        }
    }
    std::printf("b3 rows: %d rows, %d bad, %d differ from Packing::B3\n", n, bad, lib_diff);
    CHECK(bad == 0);
    CHECK(lib_diff == 0);
}

// 2. B3 refusals: bytes over 242, non-zero padding, codes out of range, I2 code 3.
static void test_b3_refusals() {
    std::vector<int8_t> q(7);
    uint8_t b[2] = {243, 0};
    CHECK(ternary_unpack_row_b3(b, 5, q.data()) == TERNARY_CONVERT_BAD_CODE);
    b[0] = 121;  // all zeros (digit 1 everywhere)
    b[1] = 121;  // 5 weights, but only 2 are real: padding digits must be 1 (a zero weight)
    CHECK(ternary_unpack_row_b3(b, 7, q.data()) == TERNARY_CONVERT_OK);
    b[1] = 121 + 27;  // weight 3 of the tail (padding) becomes +1
    CHECK(ternary_unpack_row_b3(b, 7, q.data()) == TERNARY_CONVERT_BAD_CODE);
    bool zeroed = true;
    for (int8_t v : q) zeroed = zeroed && v == 0;
    CHECK(zeroed);
    int8_t c[5] = {0, 2, 0, 0, 0};
    uint8_t o[1];
    CHECK(ternary_pack_row_b3(c, 5, o) == TERNARY_CONVERT_BAD_CODE);
    std::vector<uint8_t> i2(32, 0x55), b3(26);
    i2[3] = 0xFF;  // code 3
    CHECK(ternary_b3_from_i2_row(i2.data(), 128, b3.data()) == TERNARY_CONVERT_BAD_CODE);
    CHECK(ternary_b3_row_bytes(0) == 0);
    CHECK(ternary_pack_row_b3(c, 0, o) == TERNARY_CONVERT_BAD_ARGUMENT);
}

// 3. Float16 conversion: exact for representable values, round-to-nearest-even, overflow and underflow.
static void test_float16() {
    CHECK(ternary_float32_to_float16(1.0f) == 0x3C00);
    CHECK(ternary_float32_to_float16(65504.0f) == 0x7BFF);
    CHECK(ternary_float32_to_float16(65520.0f) == 0x7C00);
    CHECK(ternary_float32_to_float16(std::ldexp(1.0f, -24)) == 0x0001);
    CHECK(ternary_float32_to_float16(std::ldexp(1.0f, -25)) == 0x0000);       // tie to even (0)
    CHECK(ternary_float32_to_float16(std::ldexp(1.5f, -25)) == 0x0001);
    CHECK(ternary_float32_to_float16(1.0f + std::ldexp(1.0f, -11)) == 0x3C00); // tie to even
    CHECK(ternary_float32_to_float16(1.0f + std::ldexp(3.0f, -11)) == 0x3C02);
    CHECK(ternary_float32_to_float16(-2.0f) == 0xC000);
    // exhaustive: every finite half round-trips
    int bad = 0;
    for (uint32_t h = 0; h < 0x7C00; ++h) {
        uint32_t bits;
        float f;
        // decode by hand
        const int e = int((h >> 10) & 31), m = int(h & 1023);
        f = e == 0 ? std::ldexp(float(m), -24) : std::ldexp(float(m | 1024), e - 25);
        std::memcpy(&bits, &f, 4);
        if (ternary_float32_to_float16(f) != h) ++bad;
    }
    CHECK(bad == 0);
}

// 4. Scale encodings: F32 exact; F16 and U8ROW within their step; decode(encode) == decoded; zero groups stay 0.
static void test_scales() {
    std::mt19937 rng(5);
    std::lognormal_distribution<float> ln(-5.0f, 1.0f);
    for (int trial = 0; trial < 20; ++trial) {
        const uint64_t rows = 1 + trial * 3, gpr = 1 + (trial % 7) * 5;
        std::vector<float> s(rows * gpr);
        for (auto& v : s) v = ln(rng);
        if (trial % 3 == 0) s[0] = 0.0f;
        if (trial % 4 == 0) s[s.size() - 1] = FLT_MIN;
        if (trial % 5 == 0) s[s.size() / 2] = 1e30f;
        for (int enc = TERNARY_SCALES_F32; enc <= TERNARY_SCALES_U8ROW; ++enc) {
            const auto e = ternary_scale_encoding(enc);
            std::vector<uint8_t> sec(ternary_scale_section_bytes(e, rows, gpr));
            std::vector<float> dec(s.size()), back(s.size());
            CHECK(!sec.empty());
            CHECK(ternary_encode_scales(e, s.data(), rows, gpr, sec.data(), dec.data()) == TERNARY_CONVERT_OK);
            CHECK(ternary_decode_scales(e, sec.data(), sec.size(), rows, gpr, back.data()) == TERNARY_CONVERT_OK);
            CHECK(std::memcmp(dec.data(), back.data(), dec.size() * 4) == 0);
            int worse = 0;
            float maxs = 0;
            for (float v : s) maxs = std::max(maxs, v);
            for (uint64_t r = 0; r < rows; ++r) {
                float ref = 0;
                for (uint64_t g = 0; g < gpr; ++g) ref = std::max(ref, s[r * gpr + g]);
                for (uint64_t g = 0; g < gpr; ++g) {
                    const float a = s[r * gpr + g], d = back[r * gpr + g];
                    if (!(d == 0.0f || d >= FLT_MIN) || !std::isfinite(d)) ++worse;
                    if (a == 0.0f && d != 0.0f) ++worse;
                    if (enc == TERNARY_SCALES_F32 && std::memcmp(&a, &d, 4) != 0) ++worse;
                    // half precision relative to the value, with an absolute floor set by the tensor's largest scale
                    if (enc == TERNARY_SCALES_F16 && std::fabs(double(d) - double(a)) > double(a) * 4.9e-4 + double(maxs) * std::ldexp(1.0, -37))
                        ++worse;
                    if (enc == TERNARY_SCALES_F16 && a == maxs && std::fabs(double(d) - double(a)) > double(a) * 4.9e-4) ++worse;
                    if (enc == TERNARY_SCALES_U8ROW && std::fabs(double(d) - double(a)) > double(ref) / 510.0 * 1.0001) ++worse;
                    if (enc == TERNARY_SCALES_U8ROW && a == ref && d != a) ++worse;  // the row's largest is exact
                }
            }
            if (worse) std::printf("  scales trial %d encoding %d: %d values out of bounds\n", trial, enc, worse);
            CHECK(worse == 0);
        }
    }
    // refusals
    float bad[2] = {1.0f, -1.0f};
    std::vector<uint8_t> sec(16);
    CHECK(ternary_encode_scales(TERNARY_SCALES_F32, bad, 1, 2, sec.data(), nullptr) == TERNARY_CONVERT_BAD_SCALE);
    bad[1] = NAN;
    CHECK(ternary_encode_scales(TERNARY_SCALES_U8ROW, bad, 1, 2, sec.data(), nullptr) == TERNARY_CONVERT_BAD_SCALE);
    bad[1] = 1e-40f;  // subnormal
    CHECK(ternary_encode_scales(TERNARY_SCALES_F16, bad, 1, 2, sec.data(), nullptr) == TERNARY_CONVERT_BAD_SCALE);
    float out[2];
    uint8_t f16[8] = {0, 0, 0, 0, 0x00, 0x3C, 0x00, 0xBC};  // second is -1
    CHECK(ternary_decode_scales(TERNARY_SCALES_F16, f16, 8, 1, 2, out) == TERNARY_CONVERT_BAD_SCALE);
    f16[7] = 0x7C;  // infinity
    CHECK(ternary_decode_scales(TERNARY_SCALES_F16, f16, 8, 1, 2, out) == TERNARY_CONVERT_BAD_SCALE);
    f16[7] = 0x3C;
    f16[1] = 1;  // reserved
    CHECK(ternary_decode_scales(TERNARY_SCALES_F16, f16, 8, 1, 2, out) == TERNARY_CONVERT_BAD_HEADER);
    CHECK(ternary_decode_scales(TERNARY_SCALES_F16, f16, 7, 1, 2, out) == TERNARY_CONVERT_TRUNCATED);
}

// 5. rANS: lossless for many shapes and distributions; damage is refused.
static void test_rans() {
    int bad = 0, n = 0;
    double bits_sum = 0;
    for (uint64_t cols : {1ull, 3ull, 128ull, 1000ull, 4096ull}) {
        for (uint64_t rows : {1ull, 2ull, 17ull, 300ull}) {
            for (double pz : {0.0, 0.46, 0.9, 1.0}) {
                for (uint32_t rpc : {1u, 7u, 1000u}) {
                    const auto q = random_trits(rows * cols, uint32_t(rows * 31 + cols + rpc), pz);
                    uint8_t* sec = nullptr;
                    size_t len = 0;
                    std::vector<int8_t> back(q.size(), 9);
                    ++n;
                    if (ternary_rans_encode(q.data(), rows, cols, rpc, &sec, &len) != TERNARY_CONVERT_OK ||
                        ternary_rans_check(sec, len, rows, cols) != TERNARY_CONVERT_OK ||
                        ternary_rans_decode(sec, len, rows, cols, back.data()) != TERNARY_CONVERT_OK || back != q)
                        ++bad;
                    if (pz == 0.46 && cols == 4096 && rows == 300 && rpc == 1000) bits_sum = 8.0 * double(len) / double(q.size());
                    // damage: truncation, header fields, and flipped bytes must never decode to wrong trits silently
                    if (sec && len > 24) {
                        std::vector<uint8_t> d(sec, sec + len);
                        if (ternary_rans_decode(d.data(), len - 1, rows, cols, back.data()) == TERNARY_CONVERT_OK) ++bad;
                        d[0] ^= 1;  // frequency sum no longer 4096
                        if (ternary_rans_check(d.data(), len, rows, cols) == TERNARY_CONVERT_OK) ++bad;
                        d[0] ^= 1;
                        d[8] ^= 1;  // rows per chunk
                        if (ternary_rans_check(d.data(), len, rows, cols) == TERNARY_CONVERT_OK &&
                            ternary_rans_decode(d.data(), len, rows, cols, back.data()) == TERNARY_CONVERT_OK && back != q)
                            ++bad;
                    }
                    std::free(sec);
                }
            }
        }
    }
    std::printf("rans: %d streams, %d bad; 46%% zeros -> %.4f bits/trit\n", n, bad, bits_sum);
    CHECK(bad == 0);
    CHECK(bits_sum > 1.50 && bits_sum < 1.56);
    int8_t q[4] = {0, 1, -1, 3};
    uint8_t* sec = nullptr;
    size_t len = 0;
    CHECK(ternary_rans_encode(q, 1, 4, 1, &sec, &len) == TERNARY_CONVERT_BAD_CODE && sec == nullptr);
    CHECK(ternary_rans_encode(q, 1, 3, 0, &sec, &len) == TERNARY_CONVERT_BAD_ARGUMENT);
}

static std::vector<uint8_t> build(const ternary_ptt1_desc& d, const std::vector<uint8_t>& s, const std::vector<uint8_t>& data) {
    uint8_t* b = nullptr;
    size_t len = 0;
    char err[256];
    if (!ternary_build_ptt1(&d, s.empty() ? nullptr : s.data(), s.size(), data.empty() ? nullptr : data.data(), data.size(), &b, &len, err, sizeof err))
        return {};
    std::vector<uint8_t> v(b, b + len);
    std::free(b);
    return v;
}

// 6. PTT1: each kind round-trips; every header field and the checksum are checked.
static void test_ptt1() {
    const uint64_t rows = 9, cols = 300, gpr = 3;
    const auto q = random_trits(rows * cols, 77, 0.45);
    std::vector<float> s(rows * gpr);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.01f * float(i + 1);
    for (int te = TERNARY_TRITS_B3; te <= TERNARY_TRITS_RANS; ++te)
        for (int se = TERNARY_SCALES_F32; se <= TERNARY_SCALES_U8ROW; ++se) {
            ternary_ptt1_desc d{};
            d.kind = TERNARY_KIND_TERNARY;
            d.trit_encoding = te;
            d.scale_encoding = se;
            d.dims[0] = 3; d.dims[1] = 3; d.dims[2] = cols;
            d.dimension_count = 3;
            d.rows = rows;
            d.columns = cols;
            std::vector<uint8_t> sec(ternary_scale_section_bytes(ternary_scale_encoding(se), rows, gpr));
            std::vector<float> dec(s.size()), sback(s.size());
            ternary_encode_scales(ternary_scale_encoding(se), s.data(), rows, gpr, sec.data(), dec.data());
            std::vector<uint8_t> data;
            if (te == TERNARY_TRITS_B3) {
                data.resize(rows * ternary_b3_row_bytes(cols));
                for (uint64_t r = 0; r < rows; ++r) ternary_pack_row_b3(q.data() + r * cols, cols, data.data() + r * ternary_b3_row_bytes(cols));
            } else {
                uint8_t* p = nullptr; size_t n = 0;
                ternary_rans_encode(q.data(), rows, cols, 4, &p, &n);
                data.assign(p, p + n);
                std::free(p);
            }
            const auto buf = build(d, sec, data);
            CHECK(!buf.empty());
            ternary_ptt1_view v;
            CHECK(ternary_parse_ptt1(buf.data(), buf.size(), 1 << 30, &v) == TERNARY_CONVERT_OK);
            std::vector<int8_t> back(q.size());
            CHECK(ternary_ptt1_decode_ternary(&v, back.data(), sback.data()) == TERNARY_CONVERT_OK);
            CHECK(back == q);
            CHECK(std::memcmp(sback.data(), dec.data(), dec.size() * 4) == 0);
            // every single-byte change of the header or body is refused (or, for a body byte, caught by the checksum)
            int accepted = 0;
            for (size_t i = 0; i < buf.size(); i += (i < 160 ? 1 : 13)) {
                auto bad = buf;
                bad[i] ^= 0x40;
                if (ternary_parse_ptt1(bad.data(), bad.size(), 1 << 30, &v) == TERNARY_CONVERT_OK) ++accepted;
            }
            CHECK(accepted == 0);
            auto longer = buf;
            longer.push_back(0);
            CHECK(ternary_parse_ptt1(longer.data(), longer.size(), 1 << 30, &v) != TERNARY_CONVERT_OK);
            CHECK(ternary_parse_ptt1(buf.data(), buf.size() - 1, 1 << 30, &v) != TERNARY_CONVERT_OK);
            CHECK(ternary_parse_ptt1(buf.data(), buf.size(), 100, &v) == TERNARY_CONVERT_TOO_LARGE);
        }
    // raw and q8
    {
        ternary_ptt1_desc d{};
        d.kind = TERNARY_KIND_RAW;
        d.raw_dtype = TERNARY_RAW_BF16;
        d.dims[0] = 5;
        d.dimension_count = 1;
        d.rows = 1;
        d.columns = 5;
        std::vector<uint8_t> data(10, 7);
        auto buf = build(d, {}, data);
        ternary_ptt1_view v;
        CHECK(!buf.empty() && ternary_parse_ptt1(buf.data(), buf.size(), 1 << 20, &v) == TERNARY_CONVERT_OK &&
              v.data_bytes == 10 && v.desc.raw_dtype == TERNARY_RAW_BF16);
        data.push_back(1);
        CHECK(build(d, {}, data).empty());  // size must match the shape
        d.dims[0] = 6;
        CHECK(build(d, {}, data).empty());  // dims and rows x cols must agree
    }
    {
        ternary_ptt1_desc d{};
        d.kind = TERNARY_KIND_Q8;
        d.scale_encoding = TERNARY_SCALES_F32;
        d.dims[0] = 2; d.dims[1] = 130;
        d.dimension_count = 2;
        d.rows = 2;
        d.columns = 130;
        std::vector<uint8_t> sc(2 * 2 * 4, 0), data(260, 1);
        auto buf = build(d, sc, data);
        ternary_ptt1_view v;
        CHECK(!buf.empty() && ternary_parse_ptt1(buf.data(), buf.size(), 1 << 20, &v) == TERNARY_CONVERT_OK);
        d.trit_encoding = TERNARY_TRITS_B3;
        CHECK(build(d, sc, data).empty());
    }
}

// 7. PTP1: a valid pack parses; damage to the index, its hash or the extents is refused.
static void test_ptp1() {
    std::vector<uint8_t> index;
    auto put32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) index.push_back(uint8_t(v >> (8 * i))); };
    auto put64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) index.push_back(uint8_t(v >> (8 * i))); };
    const char* names[2] = {"a.weight", "b.weight"};
    uint64_t off = TERNARY_PTP1_HEADER_BYTES;
    for (int i = 0; i < 2; ++i) {
        put32(uint32_t(std::strlen(names[i])));
        index.insert(index.end(), names[i], names[i] + std::strlen(names[i]));
        put64(off);
        put64(200);
        for (int k = 0; k < 32; ++k) index.push_back(uint8_t(k));
        off += 200;
    }
    uint8_t header[TERNARY_PTP1_HEADER_BYTES];
    ternary_build_ptp1_header(2, off, index.data(), index.size(), header);
    ternary_ptp1_view v;
    const uint64_t size = off + index.size();
    CHECK(ternary_parse_ptp1(header, sizeof header, index.data(), index.size(), size, &v) == TERNARY_CONVERT_OK);
    CHECK(v.tensor_count == 2 && v.entries[1].offset == TERNARY_PTP1_HEADER_BYTES + 200 &&
          std::memcmp(v.entries[1].name, "b.weight", 8) == 0);
    ternary_ptp1_view_clear(&v);
    CHECK(ternary_parse_ptp1(header, sizeof header, index.data(), index.size(), size + 1, &v) != TERNARY_CONVERT_OK);
    auto bad = index;
    bad[5] ^= 1;
    CHECK(ternary_parse_ptp1(header, sizeof header, bad.data(), bad.size(), size, &v) == TERNARY_CONVERT_BAD_CHECKSUM);
    // overlapping extents, rehashed so only the structure check can catch it
    bad = index;
    bad[4 + 8 + 8] = 201 & 0xff;  // first length 201
    uint8_t h2[TERNARY_PTP1_HEADER_BYTES];
    ternary_build_ptp1_header(2, off, bad.data(), bad.size(), h2);
    CHECK(ternary_parse_ptp1(h2, sizeof h2, bad.data(), bad.size(), size, &v) == TERNARY_CONVERT_BAD_HEADER);
    // a name with a NUL in it
    bad = index;
    bad[5] = 0;
    ternary_build_ptp1_header(2, off, bad.data(), bad.size(), h2);
    CHECK(ternary_parse_ptp1(h2, sizeof h2, bad.data(), bad.size(), size, &v) == TERNARY_CONVERT_BAD_HEADER);
    // count says 3
    ternary_build_ptp1_header(3, off, index.data(), index.size(), h2);
    CHECK(ternary_parse_ptp1(h2, sizeof h2, index.data(), index.size(), size, &v) == TERNARY_CONVERT_BAD_HEADER);
}

// 8. Exact parity with the converter's I2: fit -> I2 -> B3 / rANS -> I2 gives the same bytes.
static void test_parity_with_i2() {
    const uint64_t rows = 64, cols = 512;
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> w(rows * cols);
    for (auto& x : w) x = nd(rng);
    std::vector<int8_t> codes(w.size());
    std::vector<float> scales(rows * cols / 128);
    CHECK(ternary_fit_groups_cpu(w.data(), rows * cols / 128, TERNARY_GROUP_FIT_LEAST_SQUARES, 2, codes.data(), scales.data()) == TERNARY_CONVERT_OK);
    std::vector<uint8_t> i2(rows * cols / 4);
    for (uint64_t r = 0; r < rows; ++r) ternary_pack_row_i2(codes.data() + r * cols, cols, i2.data() + r * cols / 4);
    // B3 from I2 directly, then back
    std::vector<uint8_t> b3(rows * ternary_b3_row_bytes(cols)), i2b(i2.size());
    for (uint64_t r = 0; r < rows; ++r) {
        ternary_b3_from_i2_row(i2.data() + r * cols / 4, cols, b3.data() + r * ternary_b3_row_bytes(cols));
        ternary_i2_from_b3_row(b3.data() + r * ternary_b3_row_bytes(cols), cols, i2b.data() + r * cols / 4);
    }
    CHECK(i2b == i2);
    uint8_t* sec = nullptr;
    size_t len = 0;
    std::vector<int8_t> back(codes.size());
    CHECK(ternary_rans_encode(codes.data(), rows, cols, 5, &sec, &len) == TERNARY_CONVERT_OK);
    CHECK(ternary_rans_decode(sec, len, rows, cols, back.data()) == TERNARY_CONVERT_OK && back == codes);
    std::printf("parity: %llu weights, b3 %zu bytes, rans %zu bytes, i2 %zu bytes\n", (unsigned long long)w.size(), b3.size(), len, i2.size());
    std::free(sec);
}

// 9. Padded fit: for rows not a multiple of 128 the fit equals the research least-squares fit of the real weights.
static void test_padded_fit() {
    int mismatches = 0, groups = 0;
    for (uint64_t cols : {1ull, 4ull, 100ull, 128ull, 200ull, 300ull}) {
        const uint64_t rows = 7, gpr = ternary_groups_per_row(cols);
        std::mt19937 rng{uint32_t(cols)};
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> w(rows * cols);
        for (auto& x : w) x = nd(rng);
        std::vector<int8_t> codes(w.size());
        std::vector<float> scales(rows * gpr);
        CHECK(ternary_fit_rows_padded(w.data(), rows, cols, 3, codes.data(), scales.data()) == TERNARY_CONVERT_OK);
        std::vector<float> scratch(128);
        for (uint64_t r = 0; r < rows; ++r)
            for (uint64_t g = 0; g < gpr; ++g) {
                const uint64_t b = g * 128, n = std::min<uint64_t>(128, cols - b);
                int8_t q[128];
                float gamma = -1;
                pchie::ternary::research::ternarize_group(w.data() + r * cols + b, n, pchie::ternary::research::Fit::LeastSquares,
                                                          pchie::ternary::research::Round::HalfEven, 0, 0, scratch.data(), q, &gamma);
                ++groups;
                if (std::memcmp(&gamma, &scales[r * gpr + g], 4) != 0 || std::memcmp(q, codes.data() + r * cols + b, n) != 0)
                    ++mismatches;
            }
    }
    std::printf("padded fit vs research fit: %d groups, %d differ\n", groups, mismatches);
    CHECK(mismatches == 0);
}

int main() {
    test_b3_rows();
    test_b3_refusals();
    test_float16();
    test_scales();
    test_rans();
    test_ptt1();
    test_ptp1();
    test_parity_with_i2();
    test_padded_fit();
    std::printf("pchie_ternary_pack_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
