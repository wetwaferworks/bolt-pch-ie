// Bolt PCHIE - helpers for the ternary converter tests and fuzz driver: tiny GGUF and safetensors writers and
// seeded weight generators. Test code only.
#pragma once
#include "pchie/pchie_ternary_convert.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace tconv_test {

inline void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); }
inline void put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i))); }
inline void put_str(std::vector<uint8_t>& b, const std::string& s) { put_u64(b, s.size()); b.insert(b.end(), s.begin(), s.end()); }

inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    return uint16_t((u + 0x7FFFu + lsb) >> 16);
}
// Exact for the small values the tests use (|x| <= 2048 with few mantissa bits, or 0).
inline uint16_t f32_to_f16_simple(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int e = int((u >> 23) & 0xFF) - 127 + 15;
    const uint32_t m = (u >> 13) & 0x3FFu;
    if ((u & 0x7FFFFFFFu) == 0) return uint16_t(sign);
    return uint16_t(sign | (uint32_t(e) << 10) | m);
}

struct GgufTensorSpec {
    std::string name;
    std::vector<uint64_t> dims;   // GGUF order: dims[0] innermost
    uint32_t type = 0;
    std::vector<uint8_t> data;
    bool override_offset = false;
    uint64_t offset = 0;
};
// A GGUF v3 file with no metadata and the given tensors, laid out back to back at 32-byte alignment unless a spec
// overrides its offset.
inline std::vector<uint8_t> make_gguf(const std::vector<GgufTensorSpec>& ts) {
    std::vector<uint8_t> b;
    put_u32(b, 0x46554747u); put_u32(b, 3); put_u64(b, ts.size()); put_u64(b, 0);
    std::vector<uint64_t> offs;
    uint64_t at = 0;
    for (const auto& t : ts) {
        offs.push_back(t.override_offset ? t.offset : at);
        at += (t.data.size() + 31) / 32 * 32;
    }
    for (size_t i = 0; i < ts.size(); ++i) {
        put_str(b, ts[i].name); put_u32(b, uint32_t(ts[i].dims.size()));
        for (uint64_t d : ts[i].dims) put_u64(b, d);
        put_u32(b, ts[i].type); put_u64(b, offs[i]);
    }
    while (b.size() % 32) b.push_back(0);
    const size_t base = b.size();
    uint64_t end = 0;
    for (size_t i = 0; i < ts.size(); ++i) end = std::max<uint64_t>(end, offs[i] + ts[i].data.size());
    b.resize(base + size_t((end + 31) / 32 * 32), 0);
    for (size_t i = 0; i < ts.size(); ++i)
        if (!ts[i].data.empty()) std::memcpy(b.data() + base + offs[i], ts[i].data.data(), ts[i].data.size());
    return b;
}

struct StSpec {
    std::string name, dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;
};
inline std::string st_header_json(const std::vector<StSpec>& ts) {
    std::string j = "{\"__metadata__\":{\"format\":\"pt\"}";
    uint64_t at = 0;
    for (const auto& t : ts) {
        j += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[";
        for (size_t i = 0; i < t.shape.size(); ++i) j += (i ? "," : "") + std::to_string(t.shape[i]);
        j += "],\"data_offsets\":[" + std::to_string(at) + "," + std::to_string(at + t.data.size()) + "]}";
        at += t.data.size();
    }
    return j + "}";
}
inline std::vector<uint8_t> make_safetensors_raw(const std::string& header, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> b;
    std::string h = header;
    while (h.size() % 8) h.push_back(' ');
    put_u64(b, h.size());
    b.insert(b.end(), h.begin(), h.end());
    b.insert(b.end(), data.begin(), data.end());
    return b;
}
inline std::vector<uint8_t> make_safetensors(const std::vector<StSpec>& ts) {
    std::vector<uint8_t> data;
    for (const auto& t : ts) data.insert(data.end(), t.data.begin(), t.data.end());
    return make_safetensors_raw(st_header_json(ts), data);
}

inline std::vector<uint8_t> bf16_bytes(const std::vector<float>& w) {
    std::vector<uint8_t> b;
    for (float f : w) { const uint16_t h = f32_to_bf16_rne(f); b.push_back(uint8_t(h)); b.push_back(uint8_t(h >> 8)); }
    return b;
}
inline std::vector<uint8_t> f32_bytes(const std::vector<float>& w) {
    std::vector<uint8_t> b(w.size() * 4);
    std::memcpy(b.data(), w.data(), b.size());
    return b;
}

// Weights shaped like trained layers: mostly normal, some heavy tails, some exact ties and zero runs.
inline std::vector<float> weights(uint64_t n, uint32_t seed, int kind) {
    std::mt19937 r(seed);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
    std::student_t_distribution<float> td(2.0f);
    std::vector<float> w(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) {
        switch (kind % 6) {
            case 0: w[i] = nd(r); break;
            case 1: w[i] = ud(r); break;
            case 2: w[i] = td(r) * 0.01f; break;
            case 3: w[i] = (i % 7 == 0) ? 0.0f : float(int(r() % 9) - 4) * 0.25f; break;   // many exact ties
            case 4: w[i] = (i % 128 == 5) ? 40.0f : nd(r); break;                           // one outlier per group
            default: w[i] = (r() % 4 == 0) ? nd(r) : 0.0f; break;                           // sparse
        }
    }
    return w;
}

inline std::filesystem::path temp_dir(const char* tag) {
    std::random_device rd;
    const auto p = std::filesystem::temp_directory_path() /
                   (std::string("pchie-tconv-") + tag + "-" + std::to_string(rd()) + std::to_string(rd()));
    std::filesystem::create_directories(p);
    return p;
}
inline void write_file(const std::filesystem::path& p, const std::vector<uint8_t>& b) {
    std::FILE* f = std::fopen(p.string().c_str(), "wb");
    if (!f) return;
    std::fwrite(b.data(), 1, b.size(), f);
    std::fclose(f);
}

} // namespace tconv_test
