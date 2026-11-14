#include "pchie/pchie_gguf.h"
#include "pchie/pchie_tree.h"
#include <cstdio>
#include <memory>

namespace pchie {

namespace {
struct Reader {
    std::FILE* f;
    bool ok = true;
    uint64_t size = ~0ull;  // file size; a string may not claim more bytes than are left
    template <class T> T get() { T v{}; if (std::fread(&v, sizeof(T), 1, f) != 1) ok = false; return v; }
    uint64_t left() const {
#if defined(_WIN32)
        const long long p = _ftelli64(f);
#else
        const off_t p = ftello(f);
#endif
        return (p < 0 || uint64_t(p) > size) ? 0 : size - uint64_t(p);
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        // Checked before the allocation: a length past the end of the file is malformed, not a big string.
        if (!ok || n > (1ull << 24) || n > left()) { ok = false; return {}; }
        std::string s(size_t(n), '\0');
        if (n && std::fread(&s[0], 1, size_t(n), f) != n) ok = false;
        return s;
    }
    void skip(uint64_t n) { if (_fseeki64_portable(n) != 0) ok = false; }
    int _fseeki64_portable(uint64_t n) {
#if defined(_WIN32)
        return _fseeki64(f, (long long)n, SEEK_CUR);
#else
        return fseeko(f, (off_t)n, SEEK_CUR);
#endif
    }
};
const uint32_t kScalarSize[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};

double read_num(Reader& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>();  case 1: return r.get<int8_t>();
        case 2: return r.get<uint16_t>(); case 3: return r.get<int16_t>();
        case 4: return r.get<uint32_t>(); case 5: return r.get<int32_t>();
        case 6: return r.get<float>();    case 7: return r.get<uint8_t>();
        case 10: return double(r.get<uint64_t>()); case 11: return double(r.get<int64_t>());
        case 12: return r.get<double>();
        default: r.ok = false; return 0;
    }
}
} // namespace

bool ggml_type_block(uint32_t t, uint64_t& be, uint64_t& bb) noexcept {
    switch (t) {
        case 0: be = 1; bb = 4; return true;     // F32
        case 1: be = 1; bb = 2; return true;     // F16
        case 2: be = 32; bb = 18; return true;   // Q4_0
        case 3: be = 32; bb = 20; return true;   // Q4_1
        case 6: be = 32; bb = 22; return true;   // Q5_0
        case 7: be = 32; bb = 24; return true;   // Q5_1
        case 8: be = 32; bb = 34; return true;   // Q8_0
        case 9: be = 32; bb = 36; return true;   // Q8_1
        case 10: be = 256; bb = 84; return true;  // Q2_K
        case 11: be = 256; bb = 110; return true; // Q3_K
        case 12: be = 256; bb = 144; return true; // Q4_K
        case 13: be = 256; bb = 176; return true; // Q5_K
        case 14: be = 256; bb = 210; return true; // Q6_K
        case 15: be = 256; bb = 292; return true; // Q8_K
        case 24: be = 1; bb = 1; return true;     // I8
        case 25: be = 1; bb = 2; return true;     // I16
        case 26: be = 1; bb = 4; return true;     // I32
        case 27: be = 1; bb = 8; return true;     // I64
        case 28: be = 1; bb = 8; return true;     // F64
        case 30: be = 1; bb = 2; return true;     // BF16
        default: be = 0; bb = 0; return false;
    }
}

namespace {
uint64_t file_length(std::FILE* f) {
#if defined(_WIN32)
    const long long cur = _ftelli64(f);
    if (cur < 0 || _fseeki64(f, 0, SEEK_END) != 0) return 0;
    const long long end = _ftelli64(f);
    if (_fseeki64(f, cur, SEEK_SET) != 0) return 0;
#else
    const off_t cur = ftello(f);
    if (cur < 0 || fseeko(f, 0, SEEK_END) != 0) return 0;
    const off_t end = ftello(f);
    if (fseeko(f, cur, SEEK_SET) != 0) return 0;
#endif
    return end < 0 ? 0 : uint64_t(end);
}
uint64_t tell(std::FILE* f) {
#if defined(_WIN32)
    const long long p = _ftelli64(f);
#else
    const off_t p = ftello(f);
#endif
    return p < 0 ? ~0ull : uint64_t(p);
}
bool mul_ok(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > ~0ull / a) return false;
    out = a * b;
    return true;
}
} // namespace

bool GgufFile::open(const std::string& path, std::string& err) {
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> fp(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!fp) { err = "cannot open " + path; return false; }
    file_size = file_length(fp.get());
    if (file_size < 24) { err = "truncated or malformed GGUF"; return false; }
    Reader r{fp.get()};
    r.size = file_size;
    auto remaining = [&]() -> uint64_t { const uint64_t p = tell(fp.get()); return p > file_size ? 0 : file_size - p; };
    if (r.get<uint32_t>() != 0x46554747u) { err = "bad magic"; return false; }
    version = r.get<uint32_t>();
    if (version != 3) { err = "unsupported GGUF version"; return false; }
    const uint64_t nt = r.get<uint64_t>(), nk = r.get<uint64_t>();
    // Each metadata entry needs at least 12 bytes (key length + type), each tensor entry at least 24.
    if (!r.ok || nk > remaining() / 12 || nt > remaining() / 24 || nt > (1ull << 20) || nk > (1ull << 20)) {
        err = "GGUF header counts exceed the file"; return false;
    }
    for (uint64_t i = 0; i < nk && r.ok; ++i) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (!r.ok) break;
        if (t == 8) meta_str[key] = r.str();
        else if (t == 9) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (!r.ok) break;
            meta_arr_len[key] = n;
            if (n > (1ull << 26)) { r.ok = false; break; }
            if (et == 8) {
                if (n > remaining() / 8) { r.ok = false; break; }  // every string carries an 8-byte length
                auto& v = meta_str_arr[key]; v.reserve(size_t(n)); for (uint64_t j = 0; j < n && r.ok; ++j) v.push_back(r.str());
            } else if (et < 13 && kScalarSize[et]) {
                if (n > remaining() / kScalarSize[et]) { r.ok = false; break; }
                auto& v = meta_num_arr[key]; v.reserve(size_t(n)); for (uint64_t j = 0; j < n && r.ok; ++j) v.push_back(read_num(r, et));
            } else r.ok = false;
        } else meta_num[key] = read_num(r, t);
    }
    if (!r.ok) { err = "truncated or malformed GGUF"; return false; }
    if (meta_num.count("general.alignment")) {
        const double a = meta_num["general.alignment"];
        if (!(a >= 1.0 && a <= double(1u << 20)) || a != double(uint64_t(a))) { err = "GGUF alignment out of range"; return false; }
        alignment = uint64_t(a);
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) { err = "GGUF alignment must be a nonzero power of two"; return false; }
    if (nt > remaining() / 24) { err = "GGUF tensor count exceeds the file"; return false; }
    tensors.resize(size_t(nt));
    for (auto& ti : tensors) {
        if (!r.ok) break;
        ti.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (!r.ok || nd == 0 || nd > 4) { r.ok = false; break; }
        ti.dims.resize(nd);
        for (auto& d : ti.dims) d = r.get<uint64_t>();
        ti.type = r.get<uint32_t>();
        ti.offset = r.get<uint64_t>();
    }
    if (!r.ok) { err = "truncated or malformed GGUF"; return false; }
    const uint64_t pos = tell(fp.get());
    if (pos == ~0ull || pos > file_size) { err = "truncated or malformed GGUF"; return false; }
    data_offset = (pos + alignment - 1) / alignment * alignment;
    if (data_offset < pos || data_offset > file_size) { err = "GGUF data section starts past the end of the file"; return false; }
    const uint64_t data_size = file_size - data_offset;
    for (auto& ti : tensors) {
        uint64_t n = 1;
        for (uint64_t d : ti.dims) {
            if (d == 0 || !mul_ok(n, d, n)) { err = "GGUF tensor " + ti.name + ": bad shape"; return false; }
        }
        ti.n_elements = n;
        if (ti.offset % alignment != 0) { err = "GGUF tensor " + ti.name + ": offset is not aligned"; return false; }
        if (ti.offset > data_size) { err = "GGUF tensor " + ti.name + ": offset past the end of the file"; return false; }
        uint64_t be = 0, bb = 0;
        if (ggml_type_block(ti.type, be, bb)) {
            if (ti.dims[0] % be != 0) { err = "GGUF tensor " + ti.name + ": row is not a whole number of blocks"; return false; }
            uint64_t bytes = 0;
            if (!mul_ok(n / be, bb, bytes)) { err = "GGUF tensor " + ti.name + ": size overflows"; return false; }
            if (bytes > data_size - ti.offset) { err = "GGUF tensor " + ti.name + ": extends past the end of the file"; return false; }
            ti.nbytes = bytes;
            ti.size_known = true;
        }
    }
    return true;
}

const GgufTensorInfo* GgufFile::find(const std::string& n) const {
    for (const auto& t : tensors) if (t.name == n) return &t;
    return nullptr;
}

uint64_t GgufFile::directory_digest() const noexcept {
    uint64_t h = FNV_OFFSET;
    for (const auto& t : tensors) {
        for (unsigned char c : t.name) h = hash_mix64(h, c);
        for (auto d : t.dims) h = hash_mix64(h, d);
        h = hash_mix64(h, t.type);
        h = hash_mix64(h, t.offset);
    }
    return h;
}

const char* ggml_type_name(uint32_t t) noexcept {
    switch (t) {
        case 0: return "F32"; case 1: return "F16"; case 2: return "Q4_0"; case 8: return "Q8_0";
        case 12: return "Q4_K"; case 14: return "Q6_K"; case 30: return "BF16"; default: return "other";
    }
}

} // namespace pchie
