// Fuzz driver for the ternary folder's manifest.tsv (pchie::hemispheres::TernaryFolder::parse).
//   - standalone (default): pchie_ternary_manifest_fuzz <iterations> <seed>. Mutates real-looking
//     manifests (bytes, tabs, newlines, path characters, hex digits, truncation, repeats) with a
//     fixed-seed generator, so a run is reproducible.
//   - libFuzzer: build with -DPCHIE_LIBFUZZER and -fsanitize=fuzzer.
// Anything accepted must name only plain files inside the folder, each with a 64-digit
// lowercase SHA-256, and a refusal must leave nothing behind.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/hemispheres/TernaryFolder.h"
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace pchie::hemispheres;

static int exercise(const uint8_t* data, size_t len) {
    TernaryFolder tf;
    std::string err;
    if (!tf.parse(data, len, err)) {
        if (err.empty() || !tf.by_name.empty() || !tf.manifest_sha256.empty()) std::abort();
        return 0;
    }
    if (tf.by_name.empty() || tf.manifest_sha256.size() != 64) std::abort();
    for (const auto& kv : tf.by_name) {
        const std::string& f = kv.second.file;
        if (kv.first.empty() || f.empty() || f.find('/') != std::string::npos || f.find('\\') != std::string::npos ||
            f.find(':') != std::string::npos || f.find("..") != std::string::npos)
            std::abort();
        for (unsigned char c : f)
            if (c < 0x20 || c == 0x7F) std::abort();
        if (kv.second.sha256.size() != 64 || kv.second.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
            std::abort();
    }
    return 1;
}

#if defined(PCHIE_LIBFUZZER)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    exercise(data, size);
    return 0;
}
#else
static std::string seed_manifest(int rows) {
    std::string m = "index\tname\tsource_type\trows\tcols\tsha256\tfile\n";
    for (int i = 0; i < rows; ++i) {
        char idx[16];
        std::snprintf(idx, sizeof idx, "%04d", i);
        const std::string name = "model.language_model.layers." + std::to_string(38 + i) + ".mlp.down_proj.weight";
        std::string sha;
        for (int k = 0; k < 64; ++k) sha += "0123456789abcdef"[(i * 7 + k * 3) % 16];
        m += std::string(idx) + "\t" + name + "\tBF16\t2560\t10240\t" + sha + "\t" + idx + "_" + name + ".ptn\n";
    }
    return m;
}

int main(int argc, char** argv) {
    const long iterations = argc > 1 ? std::atol(argv[1]) : 100000;
    const unsigned seed = argc > 2 ? unsigned(std::atoi(argv[2])) : 1u;
    std::mt19937_64 rng(seed);
    const std::vector<std::string> seeds = {seed_manifest(1), seed_manifest(4), seed_manifest(28)};
    for (const auto& s : seeds)
        if (!exercise(reinterpret_cast<const uint8_t*>(s.data()), s.size())) { std::printf("a seed does not parse\n"); return 1; }
    static const char interesting[] = {'\t', '\n', '\r', '/', '\\', ':', '.', '"', '*', '?', '|', '<', '>', '\0', 'a', 'f', 'g', 'A', '0', '9', ' ', '\x7f', '\xff'};
    long parsed = 0;
    for (long it = 0; it < iterations; ++it) {
        std::string b = seeds[size_t(rng() % seeds.size())];
        const int edits = 1 + int(rng() % 6);
        for (int e = 0; e < edits && !b.empty(); ++e) {
            const size_t at = size_t(rng() % b.size());
            switch (int(rng() % 7)) {
                case 0: b[at] = char(b[at] ^ char(1u << (rng() % 8))); break;
                case 1: b[at] = interesting[rng() % sizeof interesting]; break;
                case 2: b.insert(at, 1, interesting[rng() % sizeof interesting]); break;
                case 3: b.erase(at, size_t(1 + rng() % 8)); break;
                case 4: b.resize(size_t(rng() % (b.size() + 1))); break;
                case 5: { const size_t n = std::min<size_t>(b.size() - at, size_t(1 + rng() % 200)); b.insert(rng() % b.size(), b.substr(at, n)); break; }
                default: b.insert(at, "../"); break;
            }
        }
        parsed += exercise(reinterpret_cast<const uint8_t*>(b.data()), b.size());
    }
    std::printf("pchie_ternary_manifest_fuzz: %ld inputs, %ld accepted, no crash, no invariant broken\n", iterations, parsed);
    return 0;
}
#endif
