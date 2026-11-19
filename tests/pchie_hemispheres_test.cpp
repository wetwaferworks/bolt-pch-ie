// The hemispheres in PCHIE: verdicts (the same numbers as Mjolnir's test_hemi), the
// Q16 distribution, MJHEMI2 (round trip, bit flips, truncation, resealed bad fields,
// order), the grid-maze world and its answer key, tensor names and BF16 rounding,
// starting Mjolnir without a shell, path checks, and ternary files bound to the manifest's SHA-256.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/Sha256.h"
#include "pchie/hemispheres/GridMazeWorld.h"
#include "pchie/hemispheres/Hemisphere.h"
#include "pchie/hemispheres/SampleSet.h"
#include "pchie/hemispheres/Spawn.h"
#include "pchie/hemispheres/TernaryFolder.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

using namespace pchie;
using namespace pchie::hemispheres;

static int g_pass = 0, g_fail = 0;
#define CHECK(c) do { if (c) ++g_pass; else { ++g_fail; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); } } while (0)

static void probs4(uint16_t* p, uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
    std::memset(p, 0, kMaxActions * sizeof p[0]);
    p[0] = a; p[1] = b; p[2] = c; p[3] = d;
}

static void test_judge() {
    uint16_t p[kMaxActions];
    probs4(p, 40000, 20000, 5535, 0);
    CHECK(pick(p, 4, 0xF) == 0u);
    CHECK(pick(p, 4, 0xE) == 1u);
    CHECK(pick(p, 4, 0x0) == kMaxActions);
    Verdict v = judge(p, 4, 0xF, 0);
    CHECK(v.reason == Reason::Agree && v.choice == 0);
    CHECK(v.score_bp == int16_t((20000 * 10000) / 65535));
    CHECK(v.confidence_bp == uint16_t((20000 * 10000) / 65535));
    v = judge(p, 4, 0xF, 2);
    CHECK(v.reason == Reason::PreferOther && v.choice == 0 && v.score_bp < 0);
    v = judge(p, 4, 0xB, 2);
    CHECK(v.reason == Reason::Illegal && v.score_bp == -10000);
    probs4(p, 33000, 32535, 0, 0);
    v = judge(p, 4, 0xF, 1);
    CHECK(v.reason == Reason::Unsure && v.choice == 0);
    probs4(p, 30000, 30000, 5535, 0);
    CHECK(pick(p, 4, 0xF) == 0u);
}

static void test_q16() {
    const float lg[6] = {1.0f, 2.0f, -3.0f, 0.5f, 7.0f, 2.0f};
    uint16_t p[kMaxActions];
    CHECK(to_q16(lg, 6, 0x2F, p));   // move 4 illegal
    uint32_t sum = 0;
    for (uint32_t a = 0; a < kMaxActions; ++a) sum += p[a];
    CHECK(sum == kProbOne && p[4] == 0 && p[6] == 0);
    CHECK(pick(p, 6, 0x2F) == 1u);   // 1 and 5 tie at 2.0; the remainder goes to the first
    CHECK(p[1] >= p[5]);
    CHECK(!to_q16(lg, 6, 0, p));
    float bad[6] = {0, 0, 0, 0, 0, 0};
    bad[2] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!to_q16(bad, 6, 0x3F, p));
    CHECK(to_q16(bad, 6, 0x3B, p));   // the NaN is on an illegal move: ignored
}

static SampleSet sample(uint32_t n) {
    Header h;
    h.round = 7; h.pass = 2; h.purpose = Purpose::Holdout; h.actor_side = Side::Right; h.world = World::GridMaze;
    h.n_features = 8; h.n_actions = 4; h.seed = 0x0123456789ABCDEFull;
    h.actor.fill(0xA1);
    h.observer.fill(0xB2);
    SampleSet s(h);
    std::string err;
    for (uint32_t i = 0; i < n; ++i) {
        Record r;
        r.episode = i / 3; r.step = uint16_t(i % 3);
        for (uint32_t f = 0; f < 8; ++f) r.features[f] = int8_t(int(i * 7 + f * 13) - 100);
        probs4(r.actor_probs.data(), uint16_t(30000 + i), 20000, uint16_t(15535 - i), 0);
        probs4(r.watcher_probs.data(), 10000, 50000, 5535, 0);
        r.actor_move = 0;
        r.verdict = judge(r.watcher_probs.data(), 4, 0xF, 0);
        r.grade = i % 3 == 2 ? Grade::NotYet : Grade::Cleared;
        CHECK(s.add(r, err));
    }
    return s;
}

static void reseal(std::vector<uint8_t>& b) {
    Sha256 s;
    s.update(b.data(), b.size() - kTrailerBytes);
    s.final(b.data() + b.size() - kTrailerBytes);
}

static void test_format() {
    const SampleSet s = sample(30);
    const std::vector<uint8_t> b = s.encode();
    CHECK(b.size() == kHeaderBytes + 30u * (kRecordFixed + 8 + 16) + kTrailerBytes);
    SampleSet t;
    std::string err;
    CHECK(SampleSet::parse(b.data(), b.size(), t, err));
    CHECK(t.records().size() == 30 && t.header().round == 7 && t.header().purpose == Purpose::Holdout && t.header().seed == 0x0123456789ABCDEFull);
    CHECK(t.encode() == b);
    const Tally ty = t.tally();
    CHECK(ty.steps == 30 && ty.cleared == 20 && ty.not_yet == 10 && ty.agree_right == 0 && ty.object_wrong == 20 &&
          ty.object_right == 10);
    // every single bit flip is caught (by the seal, or by the checks for the seal's own bytes)
    uint32_t caught = 0, flips = 0;
    for (size_t i = 0; i < b.size(); i += 7)
        for (int bit = 0; bit < 8; bit += 3) {
            std::vector<uint8_t> c = b;
            c[i] ^= uint8_t(1u << bit);
            ++flips;
            caught += !SampleSet::parse(c.data(), c.size(), t, err);
        }
    CHECK(caught == flips);
    for (size_t n : {size_t(0), size_t(10), size_t(127), size_t(159), b.size() - 1}) CHECK(!SampleSet::parse(b.data(), n, t, err));
    // resealed but wrong: each field out of range is still refused
    struct { size_t at; uint8_t v; } bad[] = {{16, 3}, {17, 2}, {18, 0}, {18, 4}, {19, 0}, {19, 65}, {20, 1}, {20, 17},
                                              {21, 0}, {21, 3}, {22, 1}, {100, 99}, {110, 1}, {kHeaderBytes + 6, 4}, {kHeaderBytes + 7, 0},
                                              {kHeaderBytes + 7, 4}, {kHeaderBytes + 8, 9}, {kHeaderBytes + 9, 5}};
    for (const auto& x : bad) {
        std::vector<uint8_t> c = b;
        c[x.at] = x.v;
        reseal(c);
        CHECK(!SampleSet::parse(c.data(), c.size(), t, err));
    }
    {   // a probability that breaks the sum
        std::vector<uint8_t> c = b;
        c[kHeaderBytes + kRecordFixed + 8] ^= 0x40;
        reseal(c);
        CHECK(!SampleSet::parse(c.data(), c.size(), t, err));
    }
    {   // records out of order
        std::vector<uint8_t> c = b;
        const size_t rb = kRecordFixed + 8 + 16;
        c[kHeaderBytes + rb + 4] = 0;   // record 1 (episode 0, step 1) -> step 0, same as record 0
        reseal(c);
        CHECK(!SampleSet::parse(c.data(), c.size(), t, err));
    }
    Record r = s.records().back();
    SampleSet u = s;
    CHECK(!u.add(r, err));   // same (episode, step)
}

static void test_world() {
    uint32_t mazes = 0, states = 0, every_key_move_clears = 1, some_fail = 0;
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        const MazeGrid g(seed * 7919, 7, 7);
        const std::vector<MazeGrid> path = GridMazeWorld::guided_states(g);
        if (path.empty()) continue;
        ++mazes;
        const std::vector<MazeAction> key = MazeOracle::shortest_path(g);
        CHECK(key.size() == path.size());
        for (size_t k = 0; k < path.size(); ++k) {
            ++states;
            const uint32_t legal = GridMazeWorld::legal_mask(path[k]);
            const uint32_t m = uint32_t(int32_t(key[k]) - int32_t(MazeAction::North));
            CHECK(m < GridMazeWorld::kActions && (legal & (1u << m)));
            every_key_move_clears &= GridMazeWorld::grade(path[k], m) == Grade::Cleared;
            for (uint32_t a = 0; a < GridMazeWorld::kActions; ++a) some_fail += GridMazeWorld::grade(path[k], a) == Grade::NotYet;
            int8_t f[GridMazeWorld::kFeatures];
            GridMazeWorld::features(path[k], f);
            CHECK(f[2] == 0 || f[2] == 64);
            CHECK(!GridMazeWorld::describe(path[k]).empty());
        }
    }
    std::printf("grid world: %u mazes, %u states\n", mazes, states);
    CHECK(mazes >= 20 && states > 100);
    CHECK(every_key_move_clears == 1);
    CHECK(some_fail > states);
    CHECK(GridMazeWorld::grade(MazeGrid(5, 7, 7), 99) == Grade::NotYet);
}

static void test_names() {
    int l = -1, s = -1;
    CHECK(Gemma4OverlayTarget::parse_name("model.language_model.layers.38.mlp.up_proj.weight", l, s) && l == 38 && s == 5);
    CHECK(Gemma4OverlayTarget::parse_name("model.language_model.layers.0.self_attn.q_proj.weight", l, s) && l == 0 && s == 0);
    CHECK(Gemma4OverlayTarget::parse_name("model.language_model.layers.41.per_layer_projection.weight", l, s) && s == 8);
    CHECK(!Gemma4OverlayTarget::parse_name("model.language_model.layers.038.mlp.up_proj.weight", l, s));
    CHECK(!Gemma4OverlayTarget::parse_name("model.language_model.layers..mlp.up_proj.weight", l, s));
    CHECK(!Gemma4OverlayTarget::parse_name("model.language_model.layers.3.mlp.up_proj.bias", l, s));
    CHECK(!Gemma4OverlayTarget::parse_name("model.language_model.layers.99999.mlp.up_proj.weight", l, s));
    CHECK(!Gemma4OverlayTarget::parse_name("blk.3.ffn_up.weight", l, s));
    CHECK(to_bf16(1.0f) == 0x3F80 && from_bf16(0x3F80) == 1.0f);
    CHECK(to_bf16(1.00390625f) == 0x3F80);   // halfway, rounds to even
    CHECK(to_bf16(1.01171875f) == 0x3F82);   // halfway, rounds to even (up)
    CHECK(std::isnan(from_bf16(to_bf16(std::numeric_limits<float>::quiet_NaN()))));
    CHECK(from_bf16(to_bf16(-2.5f)) == -2.5f);
}


// ---- review of 19 November 2026: no shell, checked paths, ternary files bound by SHA-256 ----

static std::string temp_dir(const char* leaf) {
    std::error_code ec;
    const std::filesystem::path p = std::filesystem::temp_directory_path(ec) / leaf;
    std::filesystem::remove_all(p, ec);
    std::filesystem::create_directories(p, ec);
    return p.string();
}

static bool write_bytes(const std::string& path, const std::string& bytes) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return std::fclose(f) == 0 && ok;
}

static std::string sha_hex(const std::string& bytes) {
    uint8_t d[32];
    Sha256 s;
    s.update(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    s.final(d);
    return hex(d, 32);
}

// Arguments that a shell would act on must arrive in the child exactly as sent.
static void test_spawn(const char* self) {
    bool ok = false;
    CHECK(windows_command_line({"C:\\a b\\mj_hemi.exe", "x\"y", "trail\\", "a\\\"b", ""}, ok) ==
          L"\"C:\\a b\\mj_hemi.exe\" \"x\\\"y\" \"trail\\\\\" \"a\\\\\\\"b\" \"\"");
    CHECK(ok);
    windows_command_line({"bad\xff"}, ok);
    CHECK(!ok);
    CHECK(run_program({}) == 127);
    CHECK(run_program({"/no/such/program-for-the-hemispheres"}) == 127);
    if (!self) { std::printf("spawn: no test program path given, child round trip skipped\n"); return; }
    const std::string dir = temp_dir("pchie-hemi-spawn-test");
    const std::string out = dir + "/args.txt", canary = dir + "/canary";
    const std::vector<std::string> tricky = {
        "plain", "two words", "quote \" inside", "\"", "trailing back\\", "back\\\"quote", "& echo x > " + canary,
        "| echo x > " + canary, "; echo x > " + canary, "$(echo x > " + canary + ")", "`echo x`", "%PATH%", "'single'",
        "", "caf\xc3\xa9 \xf0\x9f\x98\x80"};
    std::vector<std::string> argv = {self, "--echo-args", out};
    argv.insert(argv.end(), tricky.begin(), tricky.end());
    CHECK(run_program(argv) == 0);
    std::FILE* f = std::fopen(out.c_str(), "rb");
    CHECK(f != nullptr);
    std::vector<std::string> got;
    if (f) {
        unsigned len;
        while (std::fscanf(f, "%u:", &len) == 1) {
            std::string s(len, '\0');
            if (len && std::fread(&s[0], 1, len, f) != len) break;
            std::fgetc(f);   // the newline
            got.push_back(s);
        }
        std::fclose(f);
    }
    CHECK(got == tricky);
    std::error_code ec;
    CHECK(!std::filesystem::exists(canary, ec));
    CHECK(run_program({self, "--exit", "3"}) == 3);
    std::filesystem::remove_all(dir, ec);
}

static void test_paths() {
    std::string why;
    CHECK(path_ok("D:\\scratch\\hemi-runs-2026-11-19\\real e4b & more", why));
    CHECK(path_ok("/tmp/caf\xc3\xa9", why));
    CHECK(!path_ok("", why));
    CHECK(!path_ok("a\nb", why));
    CHECK(!path_ok("a\rb", why));
    CHECK(!path_ok(std::string("a\0b", 3), why));
    CHECK(!path_ok("x\" & calc & \"", why));
    CHECK(!path_ok("a|b", why));
    CHECK(!path_ok("a<b", why));
    CHECK(!path_ok("a>b", why));
    CHECK(!path_ok("what?", why));
    CHECK(!path_ok("star*", why));
    CHECK(!path_ok("bad\xff", why));
    CHECK(!path_ok(std::string(1025, 'a'), why));
}

static void test_ternary_folder() {
    const std::string dir = temp_dir("pchie-hemi-ternary-test");
    const std::string a = "PTN1 pretend tensor a", b = "PTN1 pretend tensor b, a little longer";
    CHECK(write_bytes(dir + "/0000_a.ptn", a) && write_bytes(dir + "/0001_b.ptn", b));
    const std::string head = "index\tname\trows\tsha256\tfile\n";
    auto manifest = [&](const std::string& body) { return write_bytes(dir + "/manifest.tsv", head + body); };
    const std::string good = "0000\ta\t1\t" + sha_hex(a) + "\t0000_a.ptn\n0001\tb\t1\t" + sha_hex(b) + "\t0001_b.ptn\r\n";
    TernaryFolder tf;
    std::string err;
    std::vector<uint8_t> bytes;
    CHECK(manifest(good));
    CHECK(tf.open(dir, err));
    CHECK(tf.by_name.size() == 2 && tf.manifest_sha256 == sha_hex(head + good));
    CHECK(tf.read_verified("a", 1 << 20, bytes, err) && std::string(bytes.begin(), bytes.end()) == a);
    CHECK(tf.read_verified("b", 1 << 20, bytes, err));
    CHECK(!tf.read_verified("c", 1 << 20, bytes, err));
    CHECK(!tf.read_verified("b", 8, bytes, err) && bytes.empty());               // larger than allowed
    // a file swapped after the manifest was written: same length, one byte different
    std::string a2 = a;
    a2[5] ^= 1;
    CHECK(write_bytes(dir + "/0000_a.ptn", a2));
    CHECK(!tf.read_verified("a", 1 << 20, bytes, err) && bytes.empty());
    CHECK(err.find("SHA-256 differs") != std::string::npos);
    CHECK(write_bytes(dir + "/0000_a.ptn", a));
    // manifests that must be refused
    const std::string bad[] = {
        "0000\ta\t1\t" + sha_hex(a) + "\t../0000_a.ptn\n",
        "0000\ta\t1\t" + sha_hex(a) + "\tsub/0000_a.ptn\n",
        "0000\ta\t1\t" + sha_hex(a) + "\tC:0000_a.ptn\n",
        "0000\ta\t1\t" + sha_hex(a) + "\t\n",
        "0000\ta\t1\tnot-a-sha\t0000_a.ptn\n",
        "0000\ta\t1\t" + std::string(64, 'A') + "\t0000_a.ptn\n",
        "0000\ta\t1\t" + sha_hex(a) + "\n",
        "0000\ta\t1\t" + sha_hex(a) + "\t0000_a.ptn\n0001\ta\t1\t" + sha_hex(a) + "\t0000_a.ptn\n",
        "",
    };
    for (const std::string& body : bad) {
        CHECK(manifest(body));
        CHECK(!tf.open(dir, err));
    }
    CHECK(write_bytes(dir + "/manifest.tsv", "index\tname\trows\tfile\n0000\ta\t1\t0000_a.ptn\n"));   // no sha256 column
    CHECK(!tf.open(dir, err));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

int main(int argc, char** argv) {
    // child modes for the spawn test
    if (argc >= 3 && std::strcmp(argv[1], "--echo-args") == 0) {
        std::vector<std::string> args(argv, argv + argc);
#ifdef _WIN32
        {   // the narrow argv is in the ANSI code page; read the real (UTF-16) arguments
            int wn = 0;
            LPWSTR* wv = CommandLineToArgvW(GetCommandLineW(), &wn);
            if (!wv) return 1;
            args.clear();
            for (int i = 0; i < wn; ++i) {
                const int n = WideCharToMultiByte(CP_UTF8, 0, wv[i], -1, nullptr, 0, nullptr, nullptr);
                std::string s(size_t(n > 0 ? n - 1 : 0), '\0');
                if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wv[i], -1, &s[0], n, nullptr, nullptr);
                args.push_back(s);
            }
            LocalFree(wv);
        }
#endif
        std::FILE* f = std::fopen(argv[2], "wb");
        if (!f) return 1;
        for (size_t i = 3; i < args.size(); ++i) std::fprintf(f, "%u:%s\n", unsigned(args[i].size()), args[i].c_str());
        return std::fclose(f) == 0 ? 0 : 1;
    }
    if (argc == 3 && std::strcmp(argv[1], "--exit") == 0) return std::atoi(argv[2]);
    test_judge();
    test_q16();
    test_format();
    test_world();
    test_names();
    test_spawn(argc >= 2 ? argv[1] : nullptr);
    test_paths();
    test_ternary_folder();
    std::printf("pchie_hemispheres_test: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
