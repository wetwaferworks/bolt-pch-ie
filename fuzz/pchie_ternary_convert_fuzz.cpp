// Fuzz driver for the ternary converter's safetensors header reader (outside input). Two ways to run it:
//   - standalone (default): pchie_ternary_convert_fuzz <iterations> <seed>. Mutates valid seed files (bit flips,
//     digits and JSON punctuation swapped in, the header length field set to edge values, truncation, extension)
//     with a fixed-seed generator, so a run is reproducible. Pair with PCHIE_SANITIZE=address,undefined.
//   - libFuzzer: build with -DPCHIE_LIBFUZZER and -fsanitize=fuzzer; LLVMFuzzerTestOneInput is the entry.
// Every input that parses is re-checked independently (extents inside the data, no overlaps, sizes match the
// shapes), and each small convertible tensor is decoded from the buffer and fitted.
#include "pchie/pchie_ternary_convert.h"
#include "../tests/pchie_ternary_convert_testutil.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace pchie::ternary;
using namespace pchie::ternary::convert;

static int exercise(const uint8_t* data, size_t len) {
    if (len < 8) return 0;
    uint64_t hl = 0;
    for (int i = 7; i >= 0; --i) hl = (hl << 8) | data[i];
    if (hl > len - 8) return 0;   // open_safetensors refuses this before parsing; the parser checks it again
    SourceFile f; std::string err;
    const bool ok = parse_safetensors_header(reinterpret_cast<const char*>(data + 8), hl, len, "fuzz", f, err);
    if (!ok) { if (!f.tensors.empty() || err.empty()) std::abort(); return 0; }
    const uint64_t begin = 8 + hl;
    for (size_t i = 0; i < f.tensors.size(); ++i) {
        const SourceTensor& t = f.tensors[i];
        if (t.offset < begin || t.offset > len || t.nbytes > len - t.offset) std::abort();
        for (size_t j = 0; j < i; ++j) {
            const SourceTensor& u = f.tensors[j];
            if (t.nbytes && u.nbytes && t.offset < u.offset + u.nbytes && u.offset < t.offset + t.nbytes) std::abort();
        }
        if (!convertible(t, nullptr)) continue;
        if (t.rows * t.row_bytes != t.nbytes) std::abort();
        if (t.rows * t.cols > 4096 * 16) continue;   // keep each case small
        std::vector<float> w(static_cast<size_t>(t.rows * t.cols));
        const Status st = decode_rows(t, data + t.offset, t.rows, w.data());
        if (st != Status::Ok) continue;   // NonFinite is a normal refusal
        std::vector<int8_t> q(w.size());
        std::vector<float> g(w.size() / kGroup);
        if (fit_groups_cpu(w.data(), g.size(), FitKind::LeastSquares, 1, q.data(), g.data()) != Status::Ok) std::abort();
        for (float s : g) if (!std::isfinite(s) || s < 0.0f) std::abort();
    }
    return 1;
}

#if defined(PCHIE_LIBFUZZER)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) { exercise(data, size); return 0; }
#else
static std::vector<std::vector<uint8_t>> seeds() {
    using namespace tconv_test;
    std::vector<std::vector<uint8_t>> out;
    out.push_back(make_safetensors({{"a.weight", "BF16", {2, 128}, bf16_bytes(weights(256, 1, 0))},
                                    {"b.weight", "F32", {1, 256}, f32_bytes(weights(256, 2, 1))},
                                    {"n", "F16", {4}, std::vector<uint8_t>(8, 0)}}));
    out.push_back(make_safetensors({{"x", "BF16", {3, 128}, bf16_bytes(weights(384, 3, 3))}}));
    out.push_back(make_safetensors({{"e", "U8", {0}, {}}, {"f", "F32", {2, 128}, f32_bytes(weights(256, 4, 2))}}));
    return out;
}

int main(int argc, char** argv) {
    const long iters = argc > 1 ? std::atol(argv[1]) : 20000;
    const unsigned seed = argc > 2 ? unsigned(std::atol(argv[2])) : 1u;
    std::mt19937_64 r(seed);
    const auto base = seeds();
    const char tokens[] = "{}[],:\"0123456789-.eE ";
    long parsed = 0;
    for (long it = 0; it < iters; ++it) {
        std::vector<uint8_t> b = base[size_t(r() % base.size())];
        const int n = 1 + int(r() % 6);
        for (int k = 0; k < n; ++k) {
            const size_t at = size_t(r() % b.size());
            switch (r() % 7) {
                case 0: b[at] ^= uint8_t(1u << (r() % 8)); break;
                case 1: b[at] = uint8_t(tokens[r() % (sizeof(tokens) - 1)]); break;
                case 2: b[at] = uint8_t(r()); break;
                case 3: { if (b.size() < 8) b.resize(8, 0);
                          const uint64_t v[] = {0, 1, 2, 7, b.size() - 8, b.size(), ~0ull, 1ull << 63};
                          const uint64_t x = v[r() % 8]; for (int i = 0; i < 8; ++i) b[size_t(i)] = uint8_t(x >> (8 * i)); } break;
                case 4: b.resize(size_t(r() % (b.size() + 1))); if (b.empty()) b.push_back(0); break;
                case 5: b.resize(b.size() + size_t(r() % 64), uint8_t(r())); break;
                default: if (b.size() > 16) { const size_t at2 = 8 + size_t(r() % (b.size() - 8));   // swap a digit run in
                            const char* d = "4294967296"; for (size_t i = 0; i < 10 && at2 + i < b.size(); ++i) b[at2 + i] = uint8_t(d[i]); } break;
            }
        }
        parsed += exercise(b.data(), b.size());
    }
    for (const auto& s : base) if (!exercise(s.data(), s.size())) { std::printf("seed file failed to parse\n"); return 1; }
    std::printf("pchie_ternary_convert_fuzz: %ld cases, %ld parsed, no crash\n", iters, parsed);
    return 0;
}
#endif
