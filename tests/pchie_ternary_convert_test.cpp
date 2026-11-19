// Bolt PCHIE - ternary converter tests (CPU; no GPU needed). The CUDA parity checks are in
// pchie_ternary_convert_cuda_test.cpp.
#include "pchie/pchie_ternary_convert.h"
#include "pchie/pchie_ternary_research.h"
#include "pchie_ternary_convert_testutil.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>

using namespace pchie::ternary;
using namespace pchie::ternary::convert;
using namespace tconv_test;

static int g_fail = 0, g_pass = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } } while (0)

static bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

// 1. The shared group fit is the research path's ternarize_group (HalfEven), bit for bit.
static void test_fit_matches_research() {
    std::vector<float> scratch(kGroup);
    int8_t q1[kGroup], q2[kGroup];
    float sorted[kGroup];
    int mismatches = 0, groups = 0;
    for (int kind = 0; kind < 6; ++kind)
        for (uint32_t seed = 1; seed <= 40; ++seed) {
            const auto w = weights(kGroup * 8, seed * 31 + uint32_t(kind), kind);
            for (int g = 0; g < 8; ++g) {
                const float* wg = w.data() + g * kGroup;
                for (FitKind fk : {FitKind::Absmean, FitKind::LeastSquares}) {
                    const research::Fit rf = fk == FitKind::Absmean ? research::Fit::Absmean : research::Fit::LeastSquares;
                    float g2 = -1.0f;
                    const Status st = research::ternarize_group(wg, kGroup, rf, research::Round::HalfEven, 0, 0,
                                                                scratch.data(), q2, &g2);
                    const float g1 = fit_group(wg, fk, sorted, q1);
                    ++groups;
                    if (st != Status::Ok || !same_bits(g1, g2) || std::memcmp(q1, q2, kGroup) != 0) ++mismatches;
                }
            }
        }
    std::printf("fit vs research ternarize_group: %d groups, %d differ\n", groups, mismatches);
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
        std::vector<float> g(rows * cols / kGroup);
        CHECK(fit_groups_cpu(w.data(), g.size(), FitKind::Absmean, 3, q.data(), g.data()) == Status::Ok);
        Tensor t; t.packing = Packing::I2; t.scale_mode = ScaleMode::PerBlock; t.rows = rows; t.cols = cols;
        t.codes.resize(size_t(rows * row_bytes_for(Packing::I2, cols))); t.scales = g;
        for (uint64_t r = 0; r < rows; ++r)
            CHECK(pack_row(q.data() + r * cols, cols, Packing::I2, t.codes.data() + r * row_bytes_for(Packing::I2, cols)) == Status::Ok);
        CHECK(serialize(t) == serialize(ref));
    }
}

// 3. Edge groups.
static void test_edge_groups() {
    float w[kGroup], sorted[kGroup];
    int8_t q[kGroup];
    for (FitKind fk : {FitKind::Absmean, FitKind::LeastSquares}) {
        for (float& x : w) x = 0.0f;
        CHECK(fit_group(w, fk, sorted, q) == 0.0f);
        bool allz = true; for (int8_t v : q) allz = allz && v == 0; CHECK(allz);
        for (float& x : w) x = 1e-40f;                     // subnormal weights: gamma would be subnormal
        CHECK(fit_group(w, fk, sorted, q) == 0.0f);
        allz = true; for (int8_t v : q) allz = allz && v == 0; CHECK(allz);
        for (int i = 0; i < kGroup; ++i) w[i] = (i % 2) ? 3.0e38f : -3.0e38f;   // near FLT_MAX: no overflow in double
        const float g = fit_group(w, fk, sorted, q);
        CHECK(std::isfinite(g) && g > 0.0f);
        bool alt = true; for (int i = 0; i < kGroup; ++i) alt = alt && q[i] == ((i % 2) ? 1 : -1); CHECK(alt);
    }
    // A tie at exactly gamma / 2 goes to 0: half the group at 2, half at 1, absmean gamma = 1.5 ... use 3 and 1:
    // mean |w| = 2, w = 1 gives w / gamma = 0.5 exactly.
    for (int i = 0; i < kGroup; ++i) w[i] = (i < kGroup / 2) ? 3.0f : 1.0f;
    CHECK(fit_group(w, FitKind::Absmean, sorted, q) == 2.0f);
    CHECK(q[0] == 1 && q[kGroup - 1] == 0);
    // Least squares: one big weight and the rest small picks k = 1 (gamma = the big weight).
    for (int i = 0; i < kGroup; ++i) w[i] = 0.001f;
    w[17] = -50.0f;
    CHECK(fit_group(w, FitKind::LeastSquares, sorted, q) == 50.0f);
    CHECK(q[17] == -1 && q[0] == 0);
}

// 3b. The code rule w > gamma / 2 (what fit_group does) agrees with w / gamma > 0.5 (the research and shipping rule)
// for float pairs, including every neighbour of the exact tie.
static void test_half_rule() {
    std::mt19937 r(31337);
    std::uniform_int_distribution<uint32_t> bits(0x00800000u, 0x7F7FFFFFu);   // positive normal gammas
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
            ++cases; if (a != b) ++disagree;
        }
    }
    std::printf("half rule: %ld cases, %ld disagree\n", cases, disagree);
    CHECK(disagree == 0);
}

// 4. Least squares never has more error than absmean on the same group (allowing for the float scale).
static void test_lsq_not_worse() {
    float sorted[kGroup];
    int8_t qa[kGroup], ql[kGroup];
    int worse = 0;
    for (int kind = 0; kind < 6; ++kind)
        for (uint32_t s = 0; s < 50; ++s) {
            const auto w = weights(kGroup, 7000 + s * 13 + uint32_t(kind), kind);
            const float ga = fit_group(w.data(), FitKind::Absmean, sorted, qa);
            const float gl = fit_group(w.data(), FitKind::LeastSquares, sorted, ql);
            double ea = 0, el = 0, n2 = 0;
            for (int i = 0; i < kGroup; ++i) {
                const double a = double(w[i]) - double(ga) * qa[i], l = double(w[i]) - double(gl) * ql[i];
                ea += a * a; el += l * l; n2 += double(w[i]) * w[i];
            }
            if (el > ea + 1e-6 * n2) ++worse;
        }
    CHECK(worse == 0);
}

// 5. Results do not depend on the thread count; non-finite input fails safe.
static void test_threads_and_nonfinite() {
    const uint64_t rows = 33, cols = 640;
    const auto w = weights(rows * cols, 4242, 2);
    const uint64_t ng = rows * cols / kGroup;
    std::vector<int8_t> q1(w.size()), q2(w.size());
    std::vector<float> g1(ng), g2(ng);
    for (FitKind fk : {FitKind::Absmean, FitKind::LeastSquares}) {
        CHECK(fit_groups_cpu(w.data(), ng, fk, 1, q1.data(), g1.data()) == Status::Ok);
        CHECK(fit_groups_cpu(w.data(), ng, fk, 7, q2.data(), g2.data()) == Status::Ok);
        CHECK(q1 == q2 && std::memcmp(g1.data(), g2.data(), ng * 4) == 0);
        ErrorStats s1, s2;
        accumulate_stats(w.data(), q1.data(), g1.data(), rows, cols, 1, s1);
        accumulate_stats(w.data(), q1.data(), g1.data(), rows, cols, 5, s2);
        CHECK(s1.sum_e2 == s2.sum_e2 && s1.sum_w2 == s2.sum_w2 && s1.sum_wr == s2.sum_wr && s1.max_abs_err == s2.max_abs_err);
        CHECK(s1.n == rows * cols && s1.groups == ng);
        CHECK(s1.rel_error() > 0.0 && s1.rel_error() < 1.0 && s1.cosine() > 0.0 && s1.cosine() <= 1.0);
    }
    auto bad = w;
    bad[1000] = std::numeric_limits<float>::quiet_NaN();
    CHECK(fit_groups_cpu(bad.data(), ng, FitKind::LeastSquares, 4, q1.data(), g1.data()) == Status::NonFinite);
    bool zeroed = true;
    for (int8_t v : q1) zeroed = zeroed && v == 0;
    for (float v : g1) zeroed = zeroed && v == 0.0f;
    CHECK(zeroed);
    CHECK(fit_groups_cpu(nullptr, ng, FitKind::Absmean, 1, q1.data(), g1.data()) == Status::BadArgument);
}

// 6. safetensors header: a good one parses; each malformed one is refused with nothing left behind.
static bool st_parses(const std::string& json, uint64_t data_bytes, SourceFile* keep = nullptr) {
    SourceFile f; std::string err;
    const bool ok = parse_safetensors_header(json.data(), json.size(), 8 + json.size() + data_bytes, "mem", f, err);
    if (!ok && !f.tensors.empty()) return true;   // a refusal must leave nothing behind: report as "parsed" to fail
    if (keep) *keep = f;
    return ok;
}
static void test_safetensors_header() {
    const std::string good =
        "{\"__metadata__\":{\"format\":\"pt\"},"
        "\"a.weight\":{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512]},"
        "\"b.weight\":{\"dtype\":\"F32\",\"shape\":[3,256],\"data_offsets\":[512,3584]},"
        "\"c\":{\"dtype\":\"I64\",\"shape\":[],\"data_offsets\":[3584,3592]}}";
    SourceFile f;
    CHECK(st_parses(good, 3592, &f));
    CHECK(f.tensors.size() == 3);
    if (f.tensors.size() == 3) {
        CHECK(f.tensors[0].rows == 2 && f.tensors[0].cols == 128 && f.tensors[0].type == SrcType::BF16);
        CHECK(f.tensors[1].offset == 8 + good.size() + 512 && f.tensors[1].row_bytes == 1024);
        CHECK(convertible(f.tensors[0], nullptr) && convertible(f.tensors[1], nullptr) && !convertible(f.tensors[2], nullptr));
    }
    CHECK(!st_parses(good, 3591));   // last tensor runs past the end

    auto one = [](const std::string& body) { return "{\"t\":" + body + "}"; };
    const char* bad_bodies[] = {
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[512,0]}",          // begin after end
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,511]}",          // extent != shape * size
        "{\"dtype\":\"BF16\",\"shape\":[-2,128],\"data_offsets\":[0,512]}",         // negative dimension
        "{\"dtype\":\"BF16\",\"shape\":[2.5,128],\"data_offsets\":[0,512]}",        // fractional dimension
        "{\"dtype\":\"BF16\",\"shape\":[1e300,128],\"data_offsets\":[0,512]}",      // huge dimension
        "{\"dtype\":\"BF16\",\"shape\":[4294967296,4294967296,4294967296],\"data_offsets\":[0,512]}",  // count overflow
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[-8,504]}",         // negative offset
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0]}",              // one offset
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512,9]}",        // three offsets
        "{\"dtype\":\"Q4\",\"shape\":[2,128],\"data_offsets\":[0,512]}",            // unknown dtype
        "{\"dtype\":7,\"shape\":[2,128],\"data_offsets\":[0,512]}",                 // dtype not a string
        "{\"dtype\":\"BF16\",\"shape\":[2,128]}",                                   // missing offsets
        "{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,512],\"x\":1}",  // unknown field
        "{\"dtype\":\"BF16\",\"shape\":[1,1,1,1,1,1,1,1,2,128],\"data_offsets\":[0,512]}",  // too many dims
        "{\"dtype\":\"BF16\",\"shape\":\"2x128\",\"data_offsets\":[0,512]}",        // shape not an array
        "[1,2]",                                                                    // entry not an object
    };
    int refused = 0;
    for (const char* b : bad_bodies) refused += st_parses(one(b), 4096) ? 0 : 1;
    CHECK(refused == int(sizeof(bad_bodies) / sizeof(bad_bodies[0])));
    // Overlap, duplicates, metadata type, not an object, trailing junk, empty name.
    CHECK(!st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
                     "\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[4,12]}}", 64));
    CHECK(!st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]},"
                     "\"a\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[8,16]}}", 64));
    CHECK(!st_parses("{\"__metadata__\":{\"k\":5}}", 0));
    CHECK(!st_parses("[]", 0));
    CHECK(!st_parses("{} x", 0));
    CHECK(!st_parses("{\"\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}", 8));
    // Empty tensors may share an offset with a neighbour; they hold no bytes.
    CHECK(st_parses("{\"a\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[0,0]},"
                    "\"b\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}", 8));
    // Header length checks.
    SourceFile g; std::string err;
    CHECK(!parse_safetensors_header(good.data(), good.size(), good.size() + 7, "mem", g, err));   // longer than file
    CHECK(!parse_safetensors_header(good.data(), 1, 100, "mem", g, err));
    CHECK(!parse_safetensors_header(nullptr, 10, 100, "mem", g, err));
}

// 7. GGUF and safetensors files end to end: open, stream, compare with a direct fit; malformed files refused.
static void test_files_end_to_end() {
    const auto dir = temp_dir("unit");
    const uint64_t rows = 6, cols = 256;
    const auto w = weights(rows * cols, 77, 0);
    std::vector<float> wb(w.size());   // the BF16-rounded weights the file will really hold
    for (size_t i = 0; i < w.size(); ++i) wb[i] = pchie::det_bf16_to_f32(f32_to_bf16_rne(w[i]));

    // Q8_0: fp16 scale then 32 int8 values per block.
    std::vector<uint8_t> q8;
    std::vector<float> q8f;
    for (uint64_t b = 0; b < rows * cols / 32; ++b) {
        const float d = 0.0078125f;   // exact in fp16
        const uint16_t h = f32_to_f16_simple(d); q8.push_back(uint8_t(h)); q8.push_back(uint8_t(h >> 8));
        for (int i = 0; i < 32; ++i) { const int8_t v = int8_t(int((b * 37 + uint64_t(i) * 11) % 255) - 127); q8.push_back(uint8_t(v)); q8f.push_back(d * float(v)); }
    }
    std::vector<uint8_t> f16;
    std::vector<float> f16f;
    for (uint64_t i = 0; i < rows * cols; ++i) {
        const float v = float(int(i % 61) - 30) * 0.125f;
        const uint16_t h = f32_to_f16_simple(v); f16.push_back(uint8_t(h)); f16.push_back(uint8_t(h >> 8)); f16f.push_back(v);
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
    SourceFile f; std::string err;
    CHECK(open_source(gpath.string(), f, err));
    CHECK(f.format == SourceFile::Format::Gguf && f.tensors.size() == 5);
    const std::vector<const std::vector<float>*> expect = {&wb, &w, &q8f, &f16f};
    std::FILE* fp = std::fopen(gpath.string().c_str(), "rb");
    for (size_t i = 0; i < 4 && i < f.tensors.size(); ++i) {
        CHECK(convertible(f.tensors[i], nullptr));
        for (FitKind fk : {FitKind::Absmean, FitKind::LeastSquares}) {
            ConvertOptions o; o.fit = fk; o.threads = 3;
            TensorResult a, b;
            CHECK(convert_tensor(fp, f, f.tensors[i], o, a, err));
            o.chunk_weights = 300;   // one row (256) per chunk: streaming must not change anything
            CHECK(convert_tensor(fp, f, f.tensors[i], o, b, err));
            CHECK(serialize(a.out) == serialize(b.out));
            CHECK(a.stats.sum_e2 == b.stats.sum_e2);
            // Direct fit of the decoded weights.
            const auto& ex = *expect[i];
            std::vector<int8_t> q(ex.size()); std::vector<float> g(ex.size() / kGroup);
            CHECK(fit_groups_cpu(ex.data(), g.size(), fk, 1, q.data(), g.data()) == Status::Ok);
            CHECK(std::memcmp(g.data(), a.out.scales.data(), g.size() * 4) == 0);
            std::vector<int8_t> rowq(cols);
            bool codes_ok = true;
            for (uint64_t r = 0; r < rows; ++r) {
                unpack_row(a.out.codes.data() + r * row_bytes_for(Packing::I2, cols), cols, Packing::I2, rowq.data());
                codes_ok = codes_ok && std::memcmp(rowq.data(), q.data() + r * cols, cols) == 0;
            }
            CHECK(codes_ok);
            // And the result survives the untrusted-input parser.
            const auto blob = serialize(a.out);
            View v; CHECK(parse(blob.data(), blob.size(), Limits(), v) == Status::Ok);
        }
    }
    CHECK(!convertible(f.tensors[4], nullptr));   // 1-D norm weights stay as they are
    if (fp) std::fclose(fp);

    // Overlapping GGUF tensors are refused (the repo reader alone would accept this).
    auto ov = specs;
    ov[1].override_offset = true; ov[1].offset = 0;
    const auto opath = dir / "overlap.gguf";
    write_file(opath, make_gguf(ov));
    CHECK(!open_gguf(opath.string(), f, err) && f.tensors.empty());
    // Truncated GGUF.
    auto tb = make_gguf(specs);
    tb.resize(tb.size() - 100);
    const auto tpath = dir / "trunc.gguf";
    write_file(tpath, tb);
    CHECK(!open_gguf(tpath.string(), f, err));

    // safetensors end to end, plus a truncated file.
    const std::vector<StSpec> st = {{"model.layers.0.mlp.up_proj.weight", "BF16", {rows, cols}, bf16_bytes(w)},
                                    {"model.layers.0.input_layernorm.weight", "F32", {cols}, f32_bytes(std::vector<float>(cols, 1.0f))}};
    const auto spath = dir / "m.safetensors";
    write_file(spath, make_safetensors(st));
    CHECK(open_source(spath.string(), f, err));
    CHECK(f.format == SourceFile::Format::Safetensors && f.tensors.size() == 2);
    fp = std::fopen(spath.string().c_str(), "rb");
    if (fp && f.tensors.size() == 2) {
        ConvertOptions o;
        TensorResult a;
        CHECK(convert_tensor(fp, f, f.tensors[0], o, a, err));
        std::vector<int8_t> q(wb.size()); std::vector<float> g(wb.size() / kGroup);
        CHECK(fit_groups_cpu(wb.data(), g.size(), FitKind::LeastSquares, 1, q.data(), g.data()) == Status::Ok);
        CHECK(std::memcmp(g.data(), a.out.scales.data(), g.size() * 4) == 0);
    }
    if (fp) std::fclose(fp);
    auto sb = make_safetensors(st);
    sb.resize(sb.size() - 1);
    const auto s2 = dir / "trunc.safetensors";
    write_file(s2, sb);
    CHECK(!open_safetensors(s2.string(), f, err) && f.tensors.empty());
    std::vector<uint8_t> huge(16, 0); huge[0] = 0xFF; huge[7] = 0x7F;   // header length near 2^63
    const auto s3 = dir / "hdrlen.safetensors";
    write_file(s3, huge);
    CHECK(!open_safetensors(s3.string(), f, err));

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

int main() {
    test_fit_matches_research();
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
