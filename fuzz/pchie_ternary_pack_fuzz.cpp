// Fuzz driver for the packed ternary storage readers (outside input): PTT1 tensors, PTP1 pack indexes, rANS trit
// streams, B3 rows and scale sections.
//   pchie_ternary_pack_fuzz <iterations> <seed>   (standalone, reproducible; pair with PCHIE_SANITIZE)
//   or build with -DPCHIE_LIBFUZZER and -fsanitize=fuzzer (LLVMFuzzerTestOneInput).
// Half of the mutated PTT1 inputs get their checksum recomputed, so the structural checks behind it are reached.
// Everything that parses is decoded; decoded trits are re-encoded and must decode to the same values, and every
// decoded scale must be finite, >= 0 and 0 or normal.
#include "pchie/pchie_ternary_pack.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static void put64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }

static void check_scales(const float* s, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!(s[i] == 0.0f || (s[i] >= FLT_MIN && s[i] <= FLT_MAX))) std::abort();
}

static void exercise_ptt1(const uint8_t* data, size_t len) {
    ternary_ptt1_view v;
    if (ternary_parse_ptt1(data, len, 1u << 24, &v) != TERNARY_CONVERT_OK) return;
    // the view must lie inside the buffer
    if (v.scales < data || v.data < data || v.scales + v.scales_bytes > data + len || v.data + v.data_bytes > data + len)
        std::abort();
    if (v.desc.kind != TERNARY_KIND_TERNARY) return;
    const uint64_t n = v.desc.rows * v.desc.columns, gpr = ternary_groups_per_row(v.desc.columns);
    if (n > (1u << 22) || v.desc.rows * gpr > (1u << 20)) return;
    std::vector<int8_t> q(n);
    std::vector<float> s(v.desc.rows * gpr);
    if (ternary_ptt1_decode_ternary(&v, q.data(), s.data()) != TERNARY_CONVERT_OK) {
        for (int8_t x : q) if (x) std::abort();  // failures leave zeros
        return;
    }
    check_scales(s.data(), s.size());
    for (int8_t x : q) if (x < -1 || x > 1) std::abort();
    // re-encode with the other trit encoding: must give back the same trits
    if (v.desc.trit_encoding == TERNARY_TRITS_RANS) {
        const uint64_t rb = ternary_b3_row_bytes(v.desc.columns);
        std::vector<uint8_t> b(v.desc.rows * rb);
        std::vector<int8_t> back(n);
        for (uint64_t r = 0; r < v.desc.rows; ++r) {
            if (ternary_pack_row_b3(q.data() + r * v.desc.columns, v.desc.columns, b.data() + r * rb) != TERNARY_CONVERT_OK) std::abort();
            if (ternary_unpack_row_b3(b.data() + r * rb, v.desc.columns, back.data() + r * v.desc.columns) != TERNARY_CONVERT_OK) std::abort();
        }
        if (back != q) std::abort();
    } else {
        uint8_t* sec = nullptr;
        size_t sl = 0;
        std::vector<int8_t> back(n);
        if (ternary_rans_encode(q.data(), v.desc.rows, v.desc.columns, 3, &sec, &sl) != TERNARY_CONVERT_OK) std::abort();
        if (ternary_rans_decode(sec, sl, v.desc.rows, v.desc.columns, back.data()) != TERNARY_CONVERT_OK || back != q) std::abort();
        std::free(sec);
    }
}

static void exercise_ptp1(const uint8_t* data, size_t len) {
    if (len < TERNARY_PTP1_HEADER_BYTES) return;
    uint64_t io = 0;
    for (int i = 7; i >= 0; --i) io = (io << 8) | data[16 + i];
    if (io > len) io = len;
    ternary_ptp1_view v;
    if (ternary_parse_ptp1(data, len, data + io, len - io, len, &v) == TERNARY_CONVERT_OK) {
        uint64_t expect = TERNARY_PTP1_HEADER_BYTES;
        for (uint32_t i = 0; i < v.tensor_count; ++i) {
            const auto& e = v.entries[i];
            if (e.offset != expect || e.offset + e.length > v.index_offset) std::abort();
            if ((const uint8_t*)e.name < data + io || (const uint8_t*)e.name + e.name_length > data + len) std::abort();
            expect += e.length;
        }
        ternary_ptp1_view_clear(&v);
    }
}

static void exercise_raw(const uint8_t* data, size_t len, uint64_t rows, uint64_t cols) {
    std::vector<int8_t> q(rows * cols);
    if (ternary_rans_check(data, len, rows, cols) == TERNARY_CONVERT_OK)
        (void)ternary_rans_decode(data, len, rows, cols, q.data());
    if (len >= ternary_b3_row_bytes(cols) && ternary_unpack_row_b3(data, cols, q.data()) == TERNARY_CONVERT_OK)
        for (uint64_t i = 0; i < cols; ++i) if (q[i] < -1 || q[i] > 1) std::abort();
    std::vector<float> s(rows * ternary_groups_per_row(cols));
    for (int e = TERNARY_SCALES_F32; e <= TERNARY_SCALES_U8ROW; ++e) {
        const uint64_t need = ternary_scale_section_bytes(ternary_scale_encoding(e), rows, ternary_groups_per_row(cols));
        if (need <= len && ternary_decode_scales(ternary_scale_encoding(e), data, need, rows, ternary_groups_per_row(cols), s.data()) == TERNARY_CONVERT_OK)
            check_scales(s.data(), s.size());
    }
}

static std::vector<uint8_t> make_ptt1(int kind, int te, int se, uint64_t rows, uint64_t cols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int8_t> q(rows * cols);
    for (auto& x : q) x = int8_t(int(rng() % 3) - 1);
    const uint64_t gpr = ternary_groups_per_row(cols);
    std::vector<float> s(rows * gpr);
    for (auto& x : s) x = float(rng() % 1000) * 1e-4f;
    ternary_ptt1_desc d{};
    d.kind = ternary_tensor_kind(kind);
    d.dims[0] = rows; d.dims[1] = cols;
    d.dimension_count = 2;
    d.rows = rows;
    d.columns = cols;
    std::vector<uint8_t> sec, data;
    if (kind == TERNARY_KIND_TERNARY) {
        d.trit_encoding = te;
        d.scale_encoding = se;
        sec.resize(ternary_scale_section_bytes(ternary_scale_encoding(se), rows, gpr));
        ternary_encode_scales(ternary_scale_encoding(se), s.data(), rows, gpr, sec.data(), nullptr);
        if (te == TERNARY_TRITS_B3) {
            data.resize(rows * ternary_b3_row_bytes(cols));
            for (uint64_t r = 0; r < rows; ++r) ternary_pack_row_b3(q.data() + r * cols, cols, data.data() + r * ternary_b3_row_bytes(cols));
        } else {
            uint8_t* p = nullptr; size_t n = 0;
            ternary_rans_encode(q.data(), rows, cols, 2, &p, &n);
            data.assign(p, p + n);
            std::free(p);
        }
    } else if (kind == TERNARY_KIND_Q8) {
        d.scale_encoding = TERNARY_SCALES_F32;
        sec.resize(rows * gpr * 4);
        ternary_encode_scales(TERNARY_SCALES_F32, s.data(), rows, gpr, sec.data(), nullptr);
        data.assign(q.begin(), q.end());
    } else {
        d.raw_dtype = TERNARY_RAW_BF16;
        data.resize(rows * cols * 2, 3);
    }
    uint8_t* b = nullptr;
    size_t len = 0;
    char err[256];
    if (!ternary_build_ptt1(&d, sec.data(), sec.size(), data.data(), data.size(), &b, &len, err, sizeof err)) std::abort();
    std::vector<uint8_t> out(b, b + len);
    std::free(b);
    return out;
}

static std::vector<uint8_t> make_ptp1(uint32_t count) {
    std::vector<uint8_t> payload(count * 200, 0xAB), index;
    for (uint32_t i = 0; i < count; ++i) {
        const char name[] = "layer.weight";
        for (int k = 0; k < 4; ++k) index.push_back(uint8_t((sizeof(name) - 1) >> (8 * k)));
        index.insert(index.end(), name, name + sizeof(name) - 1);
        uint8_t le[8];
        put64(le, TERNARY_PTP1_HEADER_BYTES + 200ull * i);
        index.insert(index.end(), le, le + 8);
        put64(le, 200);
        index.insert(index.end(), le, le + 8);
        for (int k = 0; k < 32; ++k) index.push_back(uint8_t(k));
    }
    std::vector<uint8_t> out(TERNARY_PTP1_HEADER_BYTES);
    ternary_build_ptp1_header(count, TERNARY_PTP1_HEADER_BYTES + payload.size(), index.data(), index.size(), out.data());
    out.insert(out.end(), payload.begin(), payload.end());
    out.insert(out.end(), index.begin(), index.end());
    return out;
}

#ifdef PCHIE_LIBFUZZER
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t len) {
    exercise_ptt1(data, len);
    exercise_ptp1(data, len);
    if (len > 2) exercise_raw(data + 2, len - 2, 1 + data[0] % 4, 1 + data[1] % 200);
    return 0;
}
#else
int main(int argc, char** argv) {
    const long iters = argc > 1 ? std::atol(argv[1]) : 20000;
    const unsigned seed = argc > 2 ? unsigned(std::atol(argv[2])) : 1u;
    std::mt19937_64 r(seed);
    std::vector<std::vector<uint8_t>> base;
    for (int te = 1; te <= 2; ++te)
        for (int se = 1; se <= 3; ++se) base.push_back(make_ptt1(TERNARY_KIND_TERNARY, te, se, 1 + te * 3, 5 + se * 70, unsigned(te * 10 + se)));
    base.push_back(make_ptt1(TERNARY_KIND_Q8, 0, 0, 3, 130, 4));
    base.push_back(make_ptt1(TERNARY_KIND_RAW, 0, 0, 2, 7, 5));
    const size_t n_ptt1 = base.size();
    base.push_back(make_ptp1(1));
    base.push_back(make_ptp1(3));
    long parsed = 0;
    for (long it = 0; it < iters; ++it) {
        const size_t which = size_t(r() % base.size());
        std::vector<uint8_t> b = base[which];
        const int n = 1 + int(r() % 6);
        for (int k = 0; k < n; ++k) {
            const size_t at = size_t(r() % b.size());
            switch (r() % 6) {
                case 0: b[at] ^= uint8_t(1u << (r() % 8)); break;
                case 1: b[at] = uint8_t(r()); break;
                case 2: {
                    const uint64_t v[] = {0, 1, 2, 3, 127, 128, 160, b.size(), ~0ull, 1ull << 32, 1ull << 63};
                    if (b.size() >= 8) put64(b.data() + size_t(r() % (b.size() - 7)) / 8 * 8, v[r() % 11]);
                } break;
                case 3:
                    b.resize(size_t(r() % (b.size() + 1)));
                    if (b.empty()) b.push_back(0);
                    break;
                case 4: b.insert(b.end(), size_t(1 + r() % 16), uint8_t(r())); break;
                default: b[at] = uint8_t(r() % 3 == 0 ? 242 + r() % 3 : r() % 243); break;
            }
        }
        if (which < n_ptt1 && b.size() >= TERNARY_PTT1_HEADER_BYTES && (r() & 1))
            put64(b.data() + 128, ternary_fnv1a64(b.data() + TERNARY_PTT1_HEADER_BYTES, b.size() - TERNARY_PTT1_HEADER_BYTES,
                                                  1469598103934665603ull));
        ternary_ptt1_view v;
        if (ternary_parse_ptt1(b.data(), b.size(), 1u << 24, &v) == TERNARY_CONVERT_OK) ++parsed;
        exercise_ptt1(b.data(), b.size());
        exercise_ptp1(b.data(), b.size());
        if (b.size() > 8) exercise_raw(b.data() + 8, b.size() - 8, 1 + r() % 4, 1 + r() % 300);
    }
    std::printf("pchie_ternary_pack_fuzz: %ld inputs, %ld PTT1 parsed, no crash\n", iters, parsed);
    return 0;
}
#endif
