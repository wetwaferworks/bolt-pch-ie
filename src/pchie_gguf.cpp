#include "pchie/pchie_gguf.h"
#include "pchie/pchie_tree.h"
#include <cstdio>
#include <memory>

namespace pchie {

namespace {
struct Reader {
    std::FILE* f;
    bool ok = true;
    template <class T> T get() { T v{}; if (std::fread(&v, sizeof(T), 1, f) != 1) ok = false; return v; }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > (1ull << 24)) { ok = false; return {}; }
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

bool GgufFile::open(const std::string& path, std::string& err) {
    std::unique_ptr<std::FILE, int (*)(std::FILE*)> fp(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!fp) { err = "cannot open " + path; return false; }
    Reader r{fp.get()};
    if (r.get<uint32_t>() != 0x46554747u) { err = "bad magic"; return false; }
    version = r.get<uint32_t>();
    if (version != 3) { err = "unsupported GGUF version"; return false; }
    const uint64_t nt = r.get<uint64_t>(), nk = r.get<uint64_t>();
    for (uint64_t i = 0; i < nk && r.ok; ++i) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (t == 8) meta_str[key] = r.str();
        else if (t == 9) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            meta_arr_len[key] = n;
            if (n > (1ull << 26)) { r.ok = false; break; }
            if (et == 8) { auto& v = meta_str_arr[key]; v.reserve(size_t(n)); for (uint64_t j = 0; j < n && r.ok; ++j) v.push_back(r.str()); }
            else if (et < 13 && kScalarSize[et]) { auto& v = meta_num_arr[key]; v.reserve(size_t(n)); for (uint64_t j = 0; j < n && r.ok; ++j) v.push_back(read_num(r, et)); }
            else r.ok = false;
        } else meta_num[key] = read_num(r, t);
    }
    if (meta_num.count("general.alignment")) alignment = uint64_t(meta_num["general.alignment"]);
    tensors.resize(size_t(nt));
    for (auto& ti : tensors) {
        if (!r.ok) break;
        ti.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (nd > 4) { r.ok = false; break; }
        ti.dims.resize(nd);
        for (auto& d : ti.dims) d = r.get<uint64_t>();
        ti.type = r.get<uint32_t>();
        ti.offset = r.get<uint64_t>();
    }
    if (!r.ok) { err = "truncated or malformed GGUF"; return false; }
#if defined(_WIN32)
    const uint64_t pos = uint64_t(_ftelli64(fp.get()));
#else
    const uint64_t pos = uint64_t(ftello(fp.get()));
#endif
    data_offset = (pos + alignment - 1) / alignment * alignment;
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
