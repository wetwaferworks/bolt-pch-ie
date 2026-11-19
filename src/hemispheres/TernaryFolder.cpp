// Bolt PCHIE - reading a ternary converter's output folder. See
// include/pchie/hemispheres/TernaryFolder.h.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#include "pchie/hemispheres/TernaryFolder.h"
#include "pchie/hemispheres/SampleSet.h"
#include "pchie/Sha256.h"

#include <cstdio>

namespace pchie {
namespace hemispheres {

bool read_whole_file(const std::string& path, uint64_t max_bytes, std::vector<uint8_t>& out, std::string& err) {
    out.clear();
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    std::vector<uint8_t> b;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
        if (uint64_t(b.size()) + n > max_bytes) { std::fclose(f); err = path + " is larger than expected"; return false; }
        b.insert(b.end(), buf, buf + n);
    }
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (bad) { err = "cannot read " + path; return false; }
    out.swap(b);
    return true;
}

static bool plain_name(const std::string& f) {
    if (f.empty() || f.size() > 255 || f == "." || f == "..") return false;
    for (unsigned char c : f)
        if (c < 0x20 || c == 0x7F || c == '/' || c == '\\' || c == ':' || c == '"' || c == '<' || c == '>' || c == '|' ||
            c == '?' || c == '*')
            return false;
    return f.find("..") == std::string::npos;
}

static bool hex64(const std::string& s) {
    if (s.size() != 64) return false;
    for (char c : s)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

static std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> col;
    size_t p = 0, q;
    while ((q = line.find('\t', p)) != std::string::npos) { col.push_back(line.substr(p, q - p)); p = q + 1; }
    col.push_back(line.substr(p));
    return col;
}

bool TernaryFolder::open(const std::string& d, std::string& err) {
    dir = d;
    by_name.clear();
    manifest_sha256.clear();
    std::vector<uint8_t> raw;
    if (!read_whole_file(d + "/manifest.tsv", 16u << 20, raw, err)) return false;
    return parse(raw.data(), raw.size(), err);
}

bool TernaryFolder::parse(const uint8_t* data, size_t n, std::string& err) {
    by_name.clear();
    manifest_sha256.clear();
    if (data == nullptr && n != 0) { err = "no manifest"; return false; }
    if (n > (16u << 20)) { err = "manifest.tsv is larger than 16 MB"; return false; }
    uint8_t sha[32];
    Sha256 s;
    s.update(data, n);
    s.final(sha);
    const std::string text(reinterpret_cast<const char*>(data), n);
    std::map<std::string, TernaryEntry> entries;   // moved into by_name only when everything checks out
    size_t name_col = SIZE_MAX, sha_col = SIZE_MAX, file_col = SIZE_MAX, width = 0, line_no = 0, p = 0;
    while (p < text.size()) {
        size_t e = text.find('\n', p);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(p, e - p);
        p = e + 1;
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line.find('\0') != std::string::npos) { err = "manifest.tsv has a NUL byte"; return false; }
        const std::vector<std::string> col = split_tabs(line);
        if (name_col == SIZE_MAX) {   // the header row
            for (size_t k = 0; k < col.size(); ++k) {
                if (col[k] == "name") name_col = k;
                else if (col[k] == "sha256") sha_col = k;
                else if (col[k] == "file") file_col = k;
            }
            if (name_col == SIZE_MAX || sha_col == SIZE_MAX || file_col == SIZE_MAX) {
                err = "manifest.tsv has no name, sha256 and file columns";
                return false;
            }
            width = col.size();
            continue;
        }
        if (col.size() != width) { err = "manifest.tsv line " + std::to_string(line_no) + " has the wrong number of columns"; return false; }
        const std::string& name = col[name_col];
        TernaryEntry en{col[file_col], col[sha_col]};
        if (name.empty()) { err = "manifest.tsv line " + std::to_string(line_no) + " has no tensor name"; return false; }
        if (!plain_name(en.file)) { err = "manifest.tsv line " + std::to_string(line_no) + " names a file outside its folder"; return false; }
        if (!hex64(en.sha256)) { err = "manifest.tsv line " + std::to_string(line_no) + " has no valid sha256"; return false; }
        if (!entries.emplace(name, en).second) { err = "manifest.tsv names a tensor twice"; return false; }
    }
    if (entries.empty()) { err = "empty manifest"; return false; }
    by_name.swap(entries);
    manifest_sha256 = hex(sha, 32);
    return true;
}

bool TernaryFolder::read_verified(const std::string& name, uint64_t max_bytes, std::vector<uint8_t>& out,
                                  std::string& err) const {
    out.clear();
    const auto it = by_name.find(name);
    if (it == by_name.end()) { err = "the ternary folder has no " + name; return false; }
    std::vector<uint8_t> raw;
    if (!read_whole_file(dir + "/" + it->second.file, max_bytes, raw, err)) return false;
    uint8_t got[32];
    Sha256 s;
    s.update(raw.data(), raw.size());
    s.final(got);
    if (hex(got, 32) != it->second.sha256) { err = it->second.file + ": SHA-256 differs from manifest.tsv"; return false; }
    out.swap(raw);
    return true;
}

} // namespace hemispheres
} // namespace pchie
