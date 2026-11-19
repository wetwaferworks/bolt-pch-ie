// Bolt PCHIE - ternary converter tests (CPU; no GPU needed). The CUDA parity checks are in
// pchie_ternary_convert_cuda_test.cpp. These tests exercise the C API through extern "C".
#include "pchie/pchie_ternary_convert.h"
#include "pchie/pchie_ternary_research.h"
#include "pchie/pchie_det_math.h"
#include "pchie_ternary_convert_testutil.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace pchie::ternary;
using namespace tconv_test;

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

static void take_source(ternary_source_file *dst, ternary_source_file *src) {
    ternary_source_file_clear(dst);
    *dst = *src;
    std::memset(src, 0, sizeof(*src));
}

// 1. The shared group fit is the research path's ternarize_group (HalfEven), bit for bit.
static void test_fit_matches_research() {
    std::vector<float> scratch(TERNARY_GROUP_WEIGHT_COUNT);
    int8_t q1[TERNARY_GROUP_WEIGHT_COUNT], q2[TERNARY_GROUP_WEIGHT_COUNT];
    float sorted[TERNARY_GROUP_WEIGHT_COUNT];
    int mismatches = 0, groups = 0;
    for (int kind = 0; kind < 6; ++kind)
        for (uint32_t seed = 1; seed <= 40; ++seed) {
            const auto w = weights(TERNARY_GROUP_WEIGHT_COUNT * 8, seed * 31 + uint32_t(kind), kind);
            for (int g = 0; g < 8; ++g) {
                const float *wg = w.data() + g * TERNARY_GROUP_WEIGHT_COUNT;
                for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
                    const research::Fit rf =
                        fk == TERNARY_GROUP_FIT_ABSMEAN ? research::Fit::Absmean : research::Fit::LeastSquares;
                    float g2 = -1.0f;
                    const Status st = research::ternarize_group(wg, TERNARY_GROUP_WEIGHT_COUNT, rf, research::Round::HalfEven,
                                                                0, 0, scratch.data(), q2, &g2);
                    const float g1 = ternary_group_fit(wg, fk, sorted, q1);
                    ++groups;
                    if (st != Status::Ok || !same_bits(g1, g2) ||
                        std::memcmp(q1, q2, TERNARY_GROUP_WEIGHT_COUNT) != 0)
                        ++mismatches;
                }
            }
        }
    std::printf("fit vs research ternarize_group: %d groups, %d differ\n", groups, mismatches);
    CHECK(mismatches == 0);
}

// 1b. The CPU radix sort of |w| gives the same descending values as insertion sort.
static void test_radix_sort_matches_insertion() {
    auto insertion = [](const float *w, float *sorted) {
        for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
            const float a = std::fabs(w[i]);
            int j = i;
            while (j > 0 && sorted[j - 1] < a) {
                sorted[j] = sorted[j - 1];
                --j;
            }
            sorted[j] = a;
        }
    };
    int mismatches = 0, groups = 0;
    float a[TERNARY_GROUP_WEIGHT_COUNT], b[TERNARY_GROUP_WEIGHT_COUNT];
    for (int kind = 0; kind < 6; ++kind)
        for (uint32_t seed = 1; seed <= 80; ++seed) {
            const auto w = weights(TERNARY_GROUP_WEIGHT_COUNT, seed * 17 + uint32_t(kind), kind);
            insertion(w.data(), a);
            ternary_sort_absolute_values_descending(w.data(), b);
            ++groups;
            if (std::memcmp(a, b, sizeof(a)) != 0) ++mismatches;
        }
    std::printf("radix vs insertion sort: %d groups, %d differ\n", groups, mismatches);
    CHECK(mismatches == 0);
}

// 2. Absmean through the converter packs to exactly what quantize_absmean (PerBlock, I2) gives.
static void test_absmean_matches_shipping_quantizer() {
    for (int kind = 0; kind < 6; ++kind) {
        const uint64_t rows = 7, cols = 384;
        const auto w = weights(rows * cols, 900u + uint32_t(kind), kind);
        Tensor ref;
        CHECK(quantize_absmean(w.data(), rows, cols, ScaleMode::PerBlock, Packing::I2, ref) == Status::Ok);
        std::vector<int8_t> q(w.size());
        std::vector<float> g(rows * cols / TERNARY_GROUP_WEIGHT_COUNT);
        CHECK(ternary_fit_groups_cpu(w.data(), g.size(), TERNARY_GROUP_FIT_ABSMEAN, 3, q.data(), g.data()) ==
              TERNARY_CONVERT_OK);
        ternary_packed_tensor packed{};
        packed.rows = rows;
        packed.columns = cols;
        packed.packed_codes_bytes = size_t(rows * ternary_i2_row_bytes(cols));
        packed.packed_codes = (uint8_t *)std::calloc(packed.packed_codes_bytes, 1);
        packed.scale_count = g.size();
        packed.scales = (float *)std::malloc(g.size() * 4);
        std::memcpy(packed.scales, g.data(), g.size() * 4);
        for (uint64_t r = 0; r < rows; ++r)
            CHECK(ternary_pack_row_i2(q.data() + r * cols, cols, packed.packed_codes + r * ternary_i2_row_bytes(cols)) ==
                  TERNARY_CONVERT_OK);
        uint8_t *blob = nullptr;
        size_t blob_len = 0;
        char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
        CHECK(ternary_serialize_ptn1(&packed, &blob, &blob_len, err, sizeof err));
        const auto ref_blob = serialize(ref);
        CHECK(blob && blob_len == ref_blob.size() && std::memcmp(blob, ref_blob.data(), blob_len) == 0);
        std::free(blob);
        ternary_packed_tensor_clear(&packed);
    }
}

// 3. Edge groups.
static void test_edge_groups() {
    float w[TERNARY_GROUP_WEIGHT_COUNT], sorted[TERNARY_GROUP_WEIGHT_COUNT];
    int8_t q[TERNARY_GROUP_WEIGHT_COUNT];
    for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
        for (float &x : w) x = 0.0f;
        CHECK(ternary_group_fit(w, fk, sorted, q) == 0.0f);
        bool allz = true;
        for (int8_t v : q) allz = allz && v == 0;
        CHECK(allz);
        for (float &x : w) x = 1e-40f;
        CHECK(ternary_group_fit(w, fk, sorted, q) == 0.0f);
        allz = true;
        for (int8_t v : q) allz = allz && v == 0;
        CHECK(allz);
        for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) w[i] = (i % 2) ? 3.0e38f : -3.0e38f;
        const float g = ternary_group_fit(w, fk, sorted, q);
        CHECK(std::isfinite(g) && g > 0.0f);
        bool alt = true;
        for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) alt = alt && q[i] == ((i % 2) ? 1 : -1);
        CHECK(alt);
    }
    for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) w[i] = (i < TERNARY_GROUP_WEIGHT_COUNT / 2) ? 3.0f : 1.0f;
    CHECK(ternary_group_fit(w, TERNARY_GROUP_FIT_ABSMEAN, sorted, q) == 2.0f);
    CHECK(q[0] == 1 && q[TERNARY_GROUP_WEIGHT_COUNT - 1] == 0);
    for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) w[i] = 0.001f;
    w[17] = -50.0f;
    CHECK(ternary_group_fit(w, TERNARY_GROUP_FIT_LEAST_SQUARES, sorted, q) == 50.0f);
    CHECK(q[17] == -1 && q[0] == 0);
}

static void test_half_rule() {
    std::mt19937 r(31337);
    std::uniform_int_distribution<uint32_t> bits(0x00800000u, 0x7F7FFFFFu);
    long disagree = 0, cases = 0;
    auto f_of = [](uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; };
    for (int i = 0; i < 200000; ++i) {
        const float g = f_of(bits(r));
        const float half = float(0.5 * double(g));
        float probes[8] = {half, std::nextafter(half, 0.0f), std::nextafter(half, 1e38f), -half,
                           std::nextafter(-half, 0.0f), std::nextafter(-half, -1e38f), f_of(bits(r)), -f_of(bits(r))};
        for (float w : probes) {
            const double t = double(w) / double(g);
            const int a = t > 0.5 ? 1 : (t < -0.5 ? -1 : 0);
            const double h = 0.5 * double(g);
            const int b = double(w) > h ? 1 : (double(w) < -h ? -1 : 0);
            ++cases;
            if (a != b) ++disagree;
        }
    }
    std::printf("half rule: %ld cases, %ld disagree\n", cases, disagree);
    CHECK(disagree == 0);
}

static void test_lsq_not_worse() {
    float sorted[TERNARY_GROUP_WEIGHT_COUNT];
    int8_t qa[TERNARY_GROUP_WEIGHT_COUNT], ql[TERNARY_GROUP_WEIGHT_COUNT];
    int worse = 0;
    for (int kind = 0; kind < 6; ++kind)
        for (uint32_t s = 0; s < 50; ++s) {
            const auto w = weights(TERNARY_GROUP_WEIGHT_COUNT, 7000 + s * 13 + uint32_t(kind), kind);
            const float ga = ternary_group_fit(w.data(), TERNARY_GROUP_FIT_ABSMEAN, sorted, qa);
            const float gl = ternary_group_fit(w.data(), TERNARY_GROUP_FIT_LEAST_SQUARES, sorted, ql);
            double ea = 0, el = 0, n2 = 0;
            for (int i = 0; i < TERNARY_GROUP_WEIGHT_COUNT; ++i) {
                const double wv = double(w[i]);
                n2 += wv * wv;
                const double ra = double(ga) * double(qa[i]), rl = double(gl) * double(ql[i]);
                ea += (wv - ra) * (wv - ra);
                el += (wv - rl) * (wv - rl);
            }
            if (el > ea + 1e-6 * n2) ++worse;
        }
    CHECK(worse == 0);
}

static void test_threads_and_nonfinite() {
    const uint64_t rows = 33, cols = 640;
    const auto w = weights(rows * cols, 4242, 2);
    const uint64_t ng = rows * cols / TERNARY_GROUP_WEIGHT_COUNT;
    std::vector<int8_t> q1(w.size()), q2(w.size());
    std::vector<float> g1(ng), g2(ng);
    for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
        CHECK(ternary_fit_groups_cpu(w.data(), ng, fk, 1, q1.data(), g1.data()) == TERNARY_CONVERT_OK);
        CHECK(ternary_fit_groups_cpu(w.data(), ng, fk, 7, q2.data(), g2.data()) == TERNARY_CONVERT_OK);
        CHECK(q1 == q2 && std::memcmp(g1.data(), g2.data(), ng * 4) == 0);
        ternary_reconstruction_error s1{}, s2{};
        ternary_accumulate_reconstruction_error(w.data(), q1.data(), g1.data(), rows, cols, 1, &s1);
        ternary_accumulate_reconstruction_error(w.data(), q1.data(), g1.data(), rows, cols, 5, &s2);
        CHECK(s1.sum_error_squared == s2.sum_error_squared && s1.sum_weight_squared == s2.sum_weight_squared &&
              s1.sum_weight_times_reconstruction == s2.sum_weight_times_reconstruction &&
              s1.max_absolute_error == s2.max_absolute_error);
        CHECK(s1.weight_count == rows * cols && s1.group_count == ng);
        CHECK(ternary_reconstruction_relative_error(&s1) > 0.0 && ternary_reconstruction_relative_error(&s1) < 1.0 &&
              ternary_reconstruction_cosine(&s1) > 0.0 && ternary_reconstruction_cosine(&s1) <= 1.0);
    }
    auto bad = w;
    bad[1000] = std::numeric_limits<float>::quiet_NaN();
    CHECK(ternary_fit_groups_cpu(bad.data(), ng, TERNARY_GROUP_FIT_LEAST_SQUARES, 4, q1.data(), g1.data()) ==
          TERNARY_CONVERT_NON_FINITE);
    bool zeroed = true;
    for (int8_t v : q1) zeroed = zeroed && v == 0;
    for (float v : g1) zeroed = zeroed && v == 0.0f;
    CHECK(zeroed);
    CHECK(ternary_fit_groups_cpu(nullptr, ng, TERNARY_GROUP_FIT_ABSMEAN, 1, q1.data(), g1.data()) ==
          TERNARY_CONVERT_BAD_ARGUMENT);
}

static bool st_parses(const std::string &json, uint64_t data_bytes, ternary_source_file *keep = nullptr) {
    ternary_source_file f{};
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    const bool ok = ternary_parse_safetensors_header(json.data(), json.size(), 8 + json.size() + data_bytes, "mem", &f,
                                                     err, sizeof err);
    if (!ok && f.tensor_count != 0) {
        ternary_source_file_clear(&f);
        return true;
    }
    if (keep) take_source(keep, &f);
    else ternary_source_file_clear(&f);
    return ok;
}

static void test_safetensors_header() {
    const std::string good =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512]},"
        "\"b.weight\":{\"dtype\":\"F32\",\"shape\":[3,256],\"data_offsets\":[512,3584]},"
        "\"c\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[3584,3592]}}";
    ternary_source_file f{};
    CHECK(st_parses(good, 3592, &f));
    CHECK(f.tensor_count == 3);
    if (f.tensor_count == 3) {
        CHECK(f.tensors[0].rows == 2 && f.tensors[0].columns == 128 && f.tensors[0].type == TERNARY_SOURCE_WEIGHT_BF16);
        CHECK(f.tensors[1].file_offset == 8 + good.size() + 512 && f.tensors[1].bytes_per_row == 1024);
        CHECK(ternary_tensor_is_convertible(&f.tensors[0], nullptr, 0) &&
              ternary_tensor_is_convertible(&f.tensors[1], nullptr, 0) &&
              !ternary_tensor_is_convertible(&f.tensors[2], nullptr, 0));
    }
    ternary_source_file_clear(&f);
    CHECK(!st_parses(good, 3591));

    auto one = [](const std::string &body) { return "{\"t\":" + body + "}"; };
    const char *bad_bodies[] = {
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[512,0]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,511]}",
        "{\"dtype\":\"BF16\",\"shape\":[-2,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":[2.5,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":[1e300,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":[4294967296,4294967296,4294967296],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[-8,504]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512,9]}",
        "{\"dtype\":\"Q4\",\"shape\":[2,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":7,\"shape\":[2,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128]}",
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512],\"x\":1}",
        "{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1,1,1,2,128],\"data_offsets\":[0,512]}",
        "{\"dtype\":\"BF16\",\"shape\":\"2x128\",\"data_offsets\":[0,512]}",
        "[1,2]",
    };
    int refused = 0;
    for (const char *b : bad_bodies) refused += st_parses(one(b), 4096) ? 0 : 1;
    CHECK(refused == int(sizeof(bad_bodies) / sizeof(bad_bodies[0])));
    CHECK(!st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
                     "\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[4,12]}}",
                     64));
    CHECK(!st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
                     "\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[8,16]}}",
                     64));
    CHECK(!st_parses("{\"__metadata__\":{\"k\":5}}", 0));
    CHECK(!st_parses("[]", 0));
    CHECK(!st_parses("{} x", 0));
    CHECK(!st_parses("{\"\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}", 8));
    CHECK(st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]},"
                    "\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}",
                    8));
    ternary_source_file g{};
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    CHECK(!ternary_parse_safetensors_header(good.data(), good.size(), good.size() + 7, "mem", &g, err, sizeof err));
    CHECK(!ternary_parse_safetensors_header(good.data(), 1, 100, "mem", &g, err, sizeof err));
    CHECK(!ternary_parse_safetensors_header(nullptr, 10, 100, "mem", &g, err, sizeof err));
    ternary_source_file_clear(&g);
}

static bool packed_equal(const ternary_packed_tensor &a, const ternary_packed_tensor &b) {
    uint8_t *ab = nullptr, *bb = nullptr;
    size_t al = 0, bl = 0;
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    if (!ternary_serialize_ptn1(&a, &ab, &al, err, sizeof err) ||
        !ternary_serialize_ptn1(&b, &bb, &bl, err, sizeof err)) {
        std::free(ab);
        std::free(bb);
        return false;
    }
    const bool ok = al == bl && ab && bb && std::memcmp(ab, bb, al) == 0;
    std::free(ab);
    std::free(bb);
    return ok;
}

static void test_files_end_to_end() {
    const auto dir = temp_dir("unit");
    const uint64_t rows = 6, cols = 256;
    const auto w = weights(rows * cols, 77, 0);
    std::vector<float> wb(w.size());
    for (size_t i = 0; i < w.size(); ++i) wb[i] = pchie::det_bf16_to_f32(f32_to_bf16_rne(w[i]));

    std::vector<uint8_t> q8;
    std::vector<float> q8f;
    for (uint64_t b = 0; b < rows * cols / 32; ++b) {
        const float d = 0.0078125f;
        const uint16_t h = f32_to_f16_simple(d);
        q8.push_back(uint8_t(h));
        q8.push_back(uint8_t(h >> 8));
        for (int i = 0; i < 32; ++i) {
            const int8_t v = int8_t(int((b * 37 + uint64_t(i) * 11) % 255) - 127);
            q8.push_back(uint8_t(v));
            q8f.push_back(d * float(v));
        }
    }
    std::vector<uint8_t> f16;
    std::vector<float> f16f;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        const float v = float(int(i % 61) - 30) * 0.125f;
        const uint16_t h = f32_to_f16_simple(v);
        f16.push_back(uint8_t(h));
        f16.push_back(uint8_t(h >> 8));
        f16f.push_back(v);
    }
    const std::vector<GgufTensorSpec> specs = {
        {"blk.0.ffn_up.weight", {cols, rows}, 30, bf16_bytes(w), false, 0},
        {"blk.0.attn_q.weight", {cols, rows}, 0, f32_bytes(w), false, 0},
        {"blk.1.ffn_gate.weight", {cols, rows}, 8, q8, false, 0},
        {"blk.1.attn_k.weight", {cols, rows}, 1, f16, false, 0},
        {"blk.0.attn_norm.weight", {cols}, 0, f32_bytes(std::vector<float>(cols, 1.0f)), false, 0},
    };
    const auto gpath = dir / "m.gguf";
    write_file(gpath, make_gguf(specs));
    ternary_source_file f{};
    char err[TERNARY_CONVERT_ERROR_CAPACITY]{};
    CHECK(ternary_open_source(gpath.string().c_str(), &f, err, sizeof err));
    CHECK(f.format == TERNARY_SOURCE_FORMAT_GGUF && f.tensor_count == 5);
    const std::vector<const std::vector<float> *> expect = {&wb, &w, &q8f, &f16f};
    std::FILE *fp = std::fopen(gpath.string().c_str(), "rb");
    for (size_t i = 0; i < 4 && i < f.tensor_count; ++i) {
        CHECK(ternary_tensor_is_convertible(&f.tensors[i], nullptr, 0));
        for (ternary_group_fit_kind fk : {TERNARY_GROUP_FIT_ABSMEAN, TERNARY_GROUP_FIT_LEAST_SQUARES}) {
            ternary_convert_options o;
            ternary_convert_options_init(&o);
            o.fit_kind = fk;
            o.thread_count = 3;
            ternary_convert_result a{}, b{};
            CHECK(ternary_convert_tensor(fp, &f, &f.tensors[i], &o, &a, err, sizeof err));
            o.chunk_weights = 300;
            CHECK(ternary_convert_tensor(fp, &f, &f.tensors[i], &o, &b, err, sizeof err));
            CHECK(packed_equal(a.output, b.output));
            CHECK(a.stats.sum_error_squared == b.stats.sum_error_squared);
            const auto &ex = *expect[i];
            std::vector<int8_t> q(ex.size());
            std::vector<float> g(ex.size() / TERNARY_GROUP_WEIGHT_COUNT);
            CHECK(ternary_fit_groups_cpu(ex.data(), g.size(), fk, 1, q.data(), g.data()) == TERNARY_CONVERT_OK);
            CHECK(std::memcmp(g.data(), a.output.scales, g.size() * 4) == 0);
            std::vector<int8_t> rowq(cols);
            bool codes_ok = true;
            for (uint64_t r = 0; r < rows; ++r) {
                ternary_unpack_row_i2(a.output.packed_codes + r * ternary_i2_row_bytes(cols), cols, rowq.data());
                codes_ok = codes_ok && std::memcmp(rowq.data(), q.data() + r * cols, cols) == 0;
            }
            CHECK(codes_ok);
            uint8_t *blob = nullptr;
            size_t blob_len = 0;
            CHECK(ternary_serialize_ptn1(&a.output, &blob, &blob_len, err, sizeof err));
            ternary_ptn1_view view{};
            CHECK(ternary_parse_ptn1(blob, blob_len, UINT64_C(1) << 30, &view) == TERNARY_CONVERT_OK);
            std::free(blob);
            ternary_convert_result_clear(&a);
            ternary_convert_result_clear(&b);
        }
    }
    CHECK(!ternary_tensor_is_convertible(&f.tensors[4], nullptr, 0));
    if (fp) std::fclose(fp);
    ternary_source_file_clear(&f);

    auto ov = specs;
    ov[1].override_offset = true;
    ov[1].offset = 0;
    const auto opath = dir / "overlap.gguf";
    write_file(opath, make_gguf(ov));
    CHECK(!ternary_open_gguf(opath.string().c_str(), &f, err, sizeof err) && f.tensor_count == 0);
    auto tb = make_gguf(specs);
    tb.resize(tb.size() - 100);
    const auto tpath = dir / "trunc.gguf";
    write_file(tpath, tb);
    CHECK(!ternary_open_gguf(tpath.string().c_str(), &f, err, sizeof err));
    ternary_source_file_clear(&f);

    const std::vector<StSpec> st = {{"model.layers.0.mlp.up_proj.weight", "BF16", {rows, cols}, bf16_bytes(w)},
                                    {"model.layers.0.input_layernorm.weight", "F32", {cols}, f32_bytes(std::vector<float>(cols, 1.0f))}};
    const auto spath = dir / "m.safetensors";
    write_file(spath, make_safetensors(st));
    CHECK(ternary_open_source(spath.string().c_str(), &f, err, sizeof err));
    CHECK(f.format == TERNARY_SOURCE_FORMAT_SAFETENSORS && f.tensor_count == 2);
    fp = std::fopen(spath.string().c_str(), "rb");
    if (fp && f.tensor_count == 2) {
        ternary_convert_options o;
        ternary_convert_options_init(&o);
        ternary_convert_result a{};
        CHECK(ternary_convert_tensor(fp, &f, &f.tensors[0], &o, &a, err, sizeof err));
        std::vector<int8_t> q(wb.size());
        std::vector<float> g(wb.size() / TERNARY_GROUP_WEIGHT_COUNT);
        CHECK(ternary_fit_groups_cpu(wb.data(), g.size(), TERNARY_GROUP_FIT_LEAST_SQUARES, 1, q.data(), g.data()) ==
              TERNARY_CONVERT_OK);
        CHECK(std::memcmp(g.data(), a.output.scales, g.size() * 4) == 0);
        ternary_convert_result_clear(&a);
    }
    if (fp) std::fclose(fp);
    ternary_source_file_clear(&f);
    auto sb = make_safetensors(st);
    sb.resize(sb.size() - 1);
    const auto s2 = dir / "trunc.safetensors";
    write_file(s2, sb);
    CHECK(!ternary_open_safetensors(s2.string().c_str(), &f, err, sizeof err) && f.tensor_count == 0);
    std::vector<uint8_t> huge(16, 0);
    huge[0] = 0xFF;
    huge[7] = 0x7F;
    const auto s3 = dir / "hdrlen.safetensors";
    write_file(s3, huge);
    CHECK(!ternary_open_safetensors(s3.string().c_str(), &f, err, sizeof err));
    ternary_source_file_clear(&f);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

int main() {
    test_fit_matches_research();
    test_radix_sort_matches_insertion();
    test_absmean_matches_shipping_quantizer();
    test_edge_groups();
    test_half_rule();
    test_lsq_not_worse();
    test_threads_and_nonfinite();
    test_safetensors_header();
    test_files_end_to_end();
    std::printf("pchie_ternary_convert_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
