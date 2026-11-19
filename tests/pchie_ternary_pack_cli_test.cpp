// Bolt PCHIE - pchie_ternary_pack end to end on a tiny safetensors model: convert to PTN1 with the converter, pack
// with parity against it, verify, re-encode with rANS, re-pack the PTN1 files directly, and the refusals.
//   pchie_ternary_pack_cli_test PACK_TOOL CONVERT_TOOL
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
static int g_fail = 0;
static void check(bool ok, const char* what) {
    std::printf("[PACK-CLI] %-80s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_fail;
}
static std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }
static int run(const std::string& line, const fs::path& log) {
    std::string cmd = line + " > " + q(log) + " 2>&1";
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";
    return std::system(cmd.c_str());
#else
    const int rc = std::system(cmd.c_str());
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 255;
#endif
}
static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}
static uint16_t bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return uint16_t((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

struct T { std::string name, dtype; std::vector<uint64_t> shape; };

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: pchie_ternary_pack_cli_test PACK_TOOL CONVERT_TOOL\n"); return 2; }
    const std::string pack = q(argv[1]), conv = q(argv[2]);
    std::error_code ec;
    const fs::path root = fs::temp_directory_path() / "pchie-pack-cli-test";
    fs::remove_all(root, ec);
    const fs::path mdir = root / "model", ref = root / "ptn", out = root / "packed";
    fs::create_directories(mdir);
    fs::create_directories(out);
    const fs::path model = mdir / "m.safetensors", log = root / "log.txt";
    // the model: a ternary matrix, an input embedding, a conv1d kernel, a norm
    const std::vector<T> ts = {{"model.layers.0.mlp.up_proj.weight", "BF16", {6, 256}},
                               {"model.embed_tokens.weight", "BF16", {5, 130}},
                               {"model.layers.0.mixer.conv1d.weight", "BF16", {8, 1, 4}},
                               {"model.layers.0.norm.weight", "F32", {256}}};
    {
        std::string json = "{";
        uint64_t off = 0;
        std::vector<uint8_t> data;
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.0f, 0.05f);
        for (size_t i = 0; i < ts.size(); ++i) {
            uint64_t n = 1;
            for (uint64_t d : ts[i].shape) n *= d;
            const uint64_t bytes = n * (ts[i].dtype == "F32" ? 4 : 2);
            json += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype + "\",\"shape\":[";
            for (size_t k = 0; k < ts[i].shape.size(); ++k) json += (k ? "," : "") + std::to_string(ts[i].shape[k]);
            json += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
            for (uint64_t k = 0; k < n; ++k) {
                const float w = nd(rng);
                if (ts[i].dtype == "F32") {
                    uint8_t b[4];
                    std::memcpy(b, &w, 4);
                    data.insert(data.end(), b, b + 4);
                } else {
                    const uint16_t h = bf16(w);
                    data.push_back(uint8_t(h));
                    data.push_back(uint8_t(h >> 8));
                }
            }
            off += bytes;
        }
        json += "}";
        std::ofstream f(model, std::ios::binary);
        const uint64_t n = json.size();
        for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 0xff));
        f << json;
        f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    }
    check(run(conv + " " + q(model) + " --out " + q(ref) + " --backend cpu", log) == 0 && fs::exists(ref / "manifest.tsv"),
          "the converter made the PTN1 reference");

    const fs::path p1 = out / "a.ptp";
    check(run(pack + " pack " + q(model) + " --out " + q(p1) + " --reference " + q(ref) + " --threads 2", log) == 0 &&
              fs::exists(p1) && fs::exists(fs::path(p1.string() + ".tsv")) && !fs::exists(fs::path(p1.string() + ".partial")),
          "pack with parity against the PTN1 files");
    const std::string s1 = slurp(log);
    check(s1.find("codes differ 0, scales differ 0, missing 0") != std::string::npos, "pack: codes and float32 scales match PTN1");
    check(s1.find("q8: tensors 1") != std::string::npos && s1.find("raw: tensors 2") != std::string::npos,
          "pack: embedding stored as q8, conv1d and norm stored raw");
    check(run(pack + " verify " + q(p1) + " --reference " + q(ref), log) == 0 &&
              slurp(log).find("I2 codes differ in 0, float32 scales differ in 0") != std::string::npos,
          "verify: hashes, decode and exact I2 parity");

    const fs::path p2 = out / "a-rans.ptp";
    check(run(pack + " recode " + q(p1) + " --out " + q(p2) + " --trits rans", log) == 0, "recode to rANS");
    check(run(pack + " verify " + q(p2) + " --reference " + q(ref), log) == 0 &&
              slurp(log).find("I2 codes differ in 0") != std::string::npos,
          "verify the rANS pack: exact I2 parity");
    const fs::path p2b = out / "a-back.ptp";
    check(run(pack + " recode " + q(p2) + " --out " + q(p2b) + " --trits b3", log) == 0 && slurp(p2b) == slurp(p1),
          "rANS back to B3 gives the same file byte for byte");

    const fs::path p3 = out / "from-ptn.ptp";
    check(run(pack + " from-ptn " + q(ref) + " --out " + q(p3), log) == 0, "from-ptn: re-pack the PTN1 files");
    check(run(pack + " verify " + q(p3) + " --reference " + q(ref), log) == 0 &&
              slurp(log).find("I2 codes differ in 0, float32 scales differ in 0") != std::string::npos,
          "verify from-ptn: exact codes and scales");

    const fs::path p4 = out / "conv-ternary.ptp";
    check(run(pack + " pack " + q(model) + " --out " + q(p4) + " --conv1d ternary --embed ternary --scales f32", log) == 0 &&
              slurp(log).find("ternary: tensors 3") != std::string::npos,
          "pack with conv1d and embedding as ternary (rows not a multiple of 128)");
    check(run(pack + " verify " + q(p4) + " --reference " + q(ref), log) == 0 && slurp(log).find("not in PTN1 2") != std::string::npos,
          "verify: the extra ternary tensors decode; the PTN1 ones match");
    check(run(pack + " bench " + q(p2), log) == 0 && slurp(log).find("rans -> int8") != std::string::npos, "bench runs");

    // refusals
    check(run(pack + " pack " + q(model) + " --out " + q(p1), log) != 0 && slurp(log).find("already exists") != std::string::npos,
          "an existing output file is refused");
    check(run(pack + " pack " + q(model) + " --out " + q(mdir / "x.ptp"), log) != 0 && !fs::exists(mdir / "x.ptp") &&
              !fs::exists(mdir / "x.ptp.partial"),
          "an output inside the model's folder is refused");
    check(run(pack + " from-ptn " + q(ref) + " --out " + q(ref / "x.ptp"), log) != 0 && !fs::exists(ref / "x.ptp"),
          "an output inside the reference folder is refused");
    check(run(pack + " recode " + q(p1) + " --out " + q(p1) + " --trits rans", log) != 0, "recode onto its own input is refused");
    {
        std::string b = slurp(p1);
        b[200] = char(b[200] ^ 0x10);  // inside the first payload
        std::ofstream(out / "damaged.ptp", std::ios::binary) << b;
        check(run(pack + " verify " + q(out / "damaged.ptp"), log) != 0 && slurp(log).find("hash failures 1") != std::string::npos,
              "a damaged payload fails verify");
        b = slurp(p1);
        b[b.size() - 3] = char(b[b.size() - 3] ^ 1);  // in the index
        std::ofstream(out / "damaged2.ptp", std::ios::binary) << b;
        check(run(pack + " verify " + q(out / "damaged2.ptp"), log) != 0, "a damaged index fails verify");
    }
    // a damaged reference: parity must catch it
    {
        const fs::path ref2 = root / "ptn2";
        fs::copy(ref, ref2, fs::copy_options::recursive, ec);
        for (const auto& e : fs::directory_iterator(ref2)) {
            if (e.path().extension() != ".ptn") continue;
            std::string b = slurp(e.path());
            const uint64_t codes_off = [&] { uint64_t v = 0; for (int i = 7; i >= 0; --i) v = (v << 8) | uint8_t(b[48 + i]); return v; }();
            b[codes_off] = char(b[codes_off] ^ 0x01);  // a different (still valid) code: 1 <-> 0 in the low pair
            // fix the PTN1 checksum so it still parses
            uint64_t h = 1469598103934665603ull;
            for (size_t i = 96; i < b.size(); ++i) { h ^= uint8_t(b[i]); h *= 1099511628211ull; }
            for (int i = 0; i < 8; ++i) b[64 + i] = char(h >> (8 * i));
            std::ofstream(e.path(), std::ios::binary | std::ios::trunc) << b;
        }
        check(run(pack + " verify " + q(p1) + " --reference " + q(ref2), log) != 0 &&
                  slurp(log).find("I2 codes differ in 1") != std::string::npos,
              "verify notices a reference whose codes differ");
    }
    fs::remove_all(root, ec);
    std::printf("pchie_ternary_pack_cli_test: %s\n", g_fail ? "FAILED" : "all passed");
    return g_fail ? 1 : 0;
}
