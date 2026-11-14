// GGUF reader on untrusted input: a small valid file opens; malformed counts, alignment, shapes, offsets and
// extents are refused with an error (no crash, no huge allocation); random corruption never yields a tensor
// whose bytes fall outside the file.
#include "pchie/pchie_gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace pchie;
static int passes = 0, fails = 0;
static void check(bool ok, const std::string& name) {
    std::printf("[GGUF] %-96s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
    ok ? ++passes : ++fails;
}

struct Tensor { std::string name; std::vector<uint64_t> dims; uint32_t type; uint64_t offset; };
struct Spec {
    uint32_t version = 3;
    bool with_alignment = true;
    uint32_t alignment = 32;
    uint64_t nt_override = 0, nk_override = 0;  // 0: use the real counts
    std::vector<Tensor> tensors{{"tok.weight", {64, 2}, 0, 0}, {"q.weight", {64, 2}, 8, 512}};
    uint64_t data_bytes = 512 + 2 * 34 * 2;
};
using Bytes = std::vector<uint8_t>;
template <class T> static void put(Bytes& b, T v) { uint8_t t[sizeof(T)]; std::memcpy(t, &v, sizeof(T)); b.insert(b.end(), t, t + sizeof(T)); }
static void put_str(Bytes& b, const std::string& s) { put<uint64_t>(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }

static Bytes build(const Spec& s) {
    Bytes b{'G', 'G', 'U', 'F'};
    put<uint32_t>(b, s.version);
    put<uint64_t>(b, s.nt_override ? s.nt_override : s.tensors.size());
    const uint64_t nk = 2 + (s.with_alignment ? 1 : 0);
    put<uint64_t>(b, s.nk_override ? s.nk_override : nk);
    put_str(b, "general.architecture"); put<uint32_t>(b, 8); put_str(b, "test");
    put_str(b, "test.vocab"); put<uint32_t>(b, 9); put<uint32_t>(b, 8); put<uint64_t>(b, 2); put_str(b, "a"); put_str(b, "b");
    if (s.with_alignment) { put_str(b, "general.alignment"); put<uint32_t>(b, 4); put<uint32_t>(b, s.alignment); }
    for (const auto& t : s.tensors) {
        put_str(b, t.name);
        put<uint32_t>(b, uint32_t(t.dims.size()));
        for (uint64_t d : t.dims) put<uint64_t>(b, d);
        put<uint32_t>(b, t.type);
        put<uint64_t>(b, t.offset);
    }
    const uint64_t a = (s.alignment && (s.alignment & (s.alignment - 1)) == 0) ? s.alignment : 32;
    while (b.size() % a) b.push_back(0);
    b.insert(b.end(), size_t(s.data_bytes), uint8_t(0x11));
    return b;
}
static std::string write_tmp(const Bytes& b, const char* name) {
    const std::string p = std::string("gguf_validate_") + name + ".gguf";
    std::ofstream f(p, std::ios::binary); f.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    return p;
}
static bool opens(const Bytes& b, const char* name, std::string* err_out = nullptr, GgufFile* out = nullptr) {
    const std::string p = write_tmp(b, name);
    GgufFile g; std::string err;
    const bool ok = g.open(p, err);
    std::remove(p.c_str());
    if (err_out) *err_out = err;
    if (out) *out = g;
    return ok;
}
static bool rejects(const Spec& s, const char* name, const std::string& label) {
    std::string err;
    const bool ok = opens(build(s), name, &err);
    std::printf("       %s -> %s\n", name, ok ? "(opened)" : err.c_str());
    check(!ok && !err.empty(), label);
    return !ok;
}
// A string that claims more bytes than are left in the file is malformed (refused before it is allocated).
static Bytes long_string_file(uint64_t claimed) {
    Bytes b{'G', 'G', 'U', 'F'};
    put<uint32_t>(b, 3);
    put<uint64_t>(b, 0);
    put<uint64_t>(b, 1);
    put_str(b, "general.architecture"); put<uint32_t>(b, 8);
    put<uint64_t>(b, claimed);
    b.insert(b.end(), 64, uint8_t('x'));
    return b;
}
static bool in_bounds(const GgufFile& g) {
    for (const auto& t : g.tensors) {
        if (t.dims.empty() || t.dims.size() > 4) return false;
        if (t.size_known && (g.data_offset + t.offset < g.data_offset || g.data_offset + t.offset + t.nbytes > g.file_size)) return false;
    }
    return g.alignment && (g.alignment & (g.alignment - 1)) == 0 && g.data_offset <= g.file_size;
}

int main() {
    {
        GgufFile g; std::string err;
        const bool ok = opens(build(Spec{}), "valid", &err, &g);
        check(ok && g.tensors.size() == 2 && g.tensors[0].size_known && g.tensors[0].nbytes == 512 && g.tensors[1].nbytes == 136 &&
                  g.alignment == 32 && in_bounds(g),
              "valid: a small well-formed v3 file opens with sized, in-bounds tensors");
        Spec noal; noal.with_alignment = false;
        GgufFile g2; check(opens(build(noal), "valid_noalign", nullptr, &g2) && g2.alignment == 32, "valid: no alignment key means the default 32");
    }
    { Spec s; s.nt_override = uint64_t(1) << 40; rejects(s, "huge_nt", "counts: a tensor count of 2^40 is refused before any allocation"); }
    { Spec s; s.nk_override = uint64_t(1) << 62; rejects(s, "huge_nk", "counts: a metadata count of 2^62 is refused"); }
    { Spec s; s.alignment = 0; rejects(s, "align0", "alignment: zero is refused"); }
    { Spec s; s.alignment = 3; rejects(s, "align3", "alignment: a value that is not a power of two (3) is refused"); }
    { Spec s; s.alignment = 1u << 30; rejects(s, "alignhuge", "alignment: an oversized power of two (2^30) is refused"); }
    { Spec s; s.tensors[1].offset = 1u << 20; rejects(s, "offset_eof", "offset: a tensor starting past the end of the file is refused"); }
    { Spec s; s.tensors[1].offset = 520; rejects(s, "offset_misaligned", "offset: a tensor offset that is not aligned is refused"); }
    { Spec s; s.tensors[1].dims = {64, 200}; rejects(s, "extent_eof", "extent: a tensor whose bytes run past the end of the file is refused"); }
    { Spec s; s.tensors[0].offset = 0xFFFFFFFFFFFFFFE0ull; rejects(s, "offset_wrap", "extent: an offset that wraps around 2^64 is refused"); }
    { Spec s; s.tensors[0].dims = {64, 0}; rejects(s, "dim_zero", "shape: a zero dimension is refused"); }
    { Spec s; s.tensors[0].dims = {uint64_t(1) << 32, uint64_t(1) << 32, 16}; rejects(s, "dim_overflow", "shape: dimensions whose product overflows 64 bits are refused"); }
    { Spec s; s.tensors[0].dims = {}; rejects(s, "nd0", "shape: a tensor with no dimensions is refused (no dims[0] read)"); }
    { Spec s; s.tensors[0].dims = {2, 2, 2, 2, 2}; rejects(s, "nd5", "shape: more than four dimensions are refused"); }
    { Spec s; s.tensors[1].dims = {33, 2}; rejects(s, "block_partial", "shape: a quantized row that is not a whole number of blocks is refused"); }
    { Spec s; s.version = 1; rejects(s, "version1", "header: an unsupported version is refused"); }
    {
        const Bytes good = build(Spec{});
        bool all = true;
        for (size_t n : {size_t(0), size_t(3), size_t(8), size_t(23), size_t(40), size_t(100), good.size() / 2, good.size() - 1}) {
            Bytes t(good.begin(), good.begin() + std::ptrdiff_t(std::min(n, good.size())));
            GgufFile g; std::string err;
            if (opens(t, "trunc", &err, &g) && !in_bounds(g)) all = false;
            if (n < good.size() - 136 && opens(t, "trunc", &err)) all = false;  // the data section is cut: must refuse
        }
        check(all, "truncation: every truncated copy is refused (or, cut inside the data tail, never points outside)");
    }
    {
        // Random corruption: never a crash, and any file that still opens only describes bytes inside the file.
        const Bytes good = build(Spec{});
        std::mt19937_64 rng(20261113);
        int opened = 0, refused = 0; bool bounded = true;
        for (int it = 0; it < 3000; ++it) {
            Bytes b = good;
            const int flips = 1 + int(rng() % 4);
            for (int k = 0; k < flips; ++k) {
                const size_t at = size_t(rng() % std::min<size_t>(b.size(), 300));
                b[at] = uint8_t(rng());
            }
            if (rng() % 5 == 0) b.resize(size_t(rng() % b.size()));
            GgufFile g; std::string err;
            if (opens(b, "fuzz", &err, &g)) { ++opened; bounded = bounded && in_bounds(g); } else ++refused;
        }
        std::printf("       fuzz: %d opened, %d refused\n", opened, refused);
        check(bounded && refused > 0, "corruption: 3000 random byte edits never crash and never yield an out-of-file tensor");
    }
    {
        std::string err;
        const bool ok_small = opens(long_string_file(64), "str_fits", &err);
        const bool ok_long = opens(long_string_file(65), "str_long", &err);
        std::printf("       str_long -> %s\n", ok_long ? "(opened)" : err.c_str());
        const bool ok_huge = opens(long_string_file(1ull << 23), "str_huge", &err);
        check(ok_small && !ok_long && !ok_huge, "a string longer than the rest of the file is refused before it is allocated");
    }
    std::printf("[GGUF] %d pass, %d fail\n", passes, fails);
    return fails ? 1 : 0;
}
