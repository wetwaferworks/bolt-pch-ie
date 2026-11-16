// Fuzz driver for the ternary tensor parser (outside input). Two ways to run it:
//   - standalone (default): pchie_ternary_fuzz <iterations> <seed>. Mutates valid seed files (bit flips, byte
//     sets, header fields set to edge values, truncation, extension) with a fixed-seed generator, so a run is
//     reproducible. Pair with PCHIE_SANITIZE=address,undefined.
//   - libFuzzer: build with -DPCHIE_LIBFUZZER and -fsanitize=fuzzer; LLVMFuzzerTestOneInput is the entry.
// Every input that parses is then used: a matvec through every supported tier, compared with the reference.
#include "pchie/pchie_ternary.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace pchie::ternary;

static int exercise(const uint8_t* data, size_t len) {
    Limits lim; lim.max_rows = 64; lim.max_cols = 1024; lim.max_bytes = 1 << 20;   // keep each case small
    View v;
    const Status st = parse(data, len, lim, v);
    if (st != Status::Ok) { if (v.rows != 0 || v.codes) std::abort(); return 0; }   // failure leaves nothing behind
    if (v.rows == 0 || v.rows > lim.max_rows || v.cols == 0 || v.cols > lim.max_cols || v.cols % kBlock) std::abort();
    std::vector<float> xf(static_cast<size_t>(v.cols));
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = float(int(i % 17) - 8) * 0.25f;
    ActRow x; quantize_activations(xf.data(), v.cols, x);
    std::vector<float> ref(static_cast<size_t>(v.rows)), y(static_cast<size_t>(v.rows));
    const Status sref = matvec(v, x, ref.data(), v.rows, Tier::Scalar, nullptr);
    if (sref != Status::Ok && sref != Status::NonFinite) std::abort();
    if (sref == Status::NonFinite) for (float f : ref) if (f != 0.0f) std::abort();   // overflow leaves zeros
    for (Tier t : {Tier::SSSE3, Tier::AVX2, Tier::NEON}) {
        if (!tier_supported(t)) continue;
        if (matvec(v, x, y.data(), v.rows, t, nullptr) != sref) std::abort();
        if (std::memcmp(y.data(), ref.data(), y.size() * sizeof(float)) != 0) std::abort();   // bit for bit
    }
    return 1;
}

#if defined(PCHIE_LIBFUZZER)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { exercise(data, size); return 0; }
#else
static std::vector<std::vector<uint8_t>> seeds() {
    std::vector<std::vector<uint8_t>> out;
    std::mt19937 r(5);
    for (Packing p : {Packing::I2, Packing::B3}) for (ScaleMode m : {ScaleMode::PerTensor, ScaleMode::PerBlock})
        for (uint64_t cols : {128ull, 384ull}) {
            const uint64_t rows = 3;
            std::vector<float> W(static_cast<size_t>(rows * cols)); std::normal_distribution<float> nd(0, 1);
            for (auto& w : W) w = nd(r);
            Tensor t; quantize_absmean(W.data(), rows, cols, m, p, t);
            out.push_back(serialize(t));
        }
    return out;
}

static void refresh_checksum(std::vector<uint8_t>& b) {   // lets mutations reach the checks behind the checksum
    if (b.size() < 96) return;
    const uint64_t h = fnv1a64(b.data() + 96, b.size() - 96);
    for (int i = 0; i < 8; ++i) b[static_cast<size_t>(64 + i)] = uint8_t(h >> (8 * i));
}

int main(int argc, char** argv) {
    const long iters = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 20000;
    const unsigned seed = argc > 2 ? unsigned(std::strtoul(argv[2], nullptr, 10)) : 1u;
    std::mt19937 r(seed);
    const auto base = seeds();
    long parsed = 0;
    for (const auto& s : base) parsed += exercise(s.data(), s.size());
    if (parsed != long(base.size())) { std::printf("seed files do not parse\n"); return 1; }
    static const uint64_t edge[] = {0, 1, 3, 31, 32, 96, 127, 128, 129, 255, 256, 1023, 1024, 1025, 65535,
                                    0xFFFFFFFFull, 0x100000000ull, 0x7FFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull};
    for (long it = 0; it < iters; ++it) {
        std::vector<uint8_t> b = base[r() % base.size()];
        const int nmut = 1 + int(r() % 4);
        for (int k = 0; k < nmut; ++k) {
            switch (r() % 7) {
                case 0: if (!b.empty()) b[r() % b.size()] ^= uint8_t(1u << (r() % 8)); break;
                case 1: if (!b.empty()) b[r() % b.size()] = uint8_t(r()); break;
                case 2: {   // a header u64 field set to an edge value
                    const size_t off = 16 + 8 * (r() % 6);
                    const uint64_t v = edge[r() % (sizeof edge / sizeof edge[0])];
                    if (b.size() >= off + 8) for (int i = 0; i < 8; ++i) b[off + size_t(i)] = uint8_t(v >> (8 * i));
                    break;
                }
                case 3: if (!b.empty()) b.resize(r() % b.size()); break;
                case 4: b.resize(b.size() + 1 + r() % 64, uint8_t(r())); break;
                case 5: if (b.size() > 8) b[4 + r() % 4] = uint8_t(r()); break;   // version, packing, mode
                case 6: if (b.size() > 96) b[96 + r() % (b.size() - 96)] = uint8_t(r() % 2 ? 0xFF : 243); break;
            }
        }
        if (r() % 2) refresh_checksum(b);
        parsed += exercise(b.data(), b.size());
    }
    std::printf("pchie_ternary_fuzz: %ld cases, %ld parsed and matched across tiers, no crash\n", iters, parsed);
    return 0;
}
#endif
