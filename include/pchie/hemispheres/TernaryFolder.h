// Bolt PCHIE - reading a ternary converter's output folder for the hemispheres.
//
// The converter writes manifest.tsv (a header row, then one row per tensor) and one
// PTN1 file per tensor. A PTN1 file carries only a short FNV-1a checksum, which
// catches accidents but not a deliberate swap. So every file is read whole and its
// SHA-256 must match the manifest's sha256 column before a byte of it is used; the
// manifest itself is identified by its own SHA-256 in every recipe. Columns are found
// by their header names. A file name must be a plain name inside the folder.
// MIT License, (c) 2026 Jaydee Wetwork / Wet Wafer Works.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pchie {
namespace hemispheres {

struct TernaryEntry {
    std::string file;     // a plain file name inside the folder
    std::string sha256;   // 64 lowercase hex digits
};

struct TernaryFolder {
    std::string dir;
    std::string manifest_sha256;                      // of manifest.tsv as read
    std::map<std::string, TernaryEntry> by_name;      // tensor name -> entry

    // Reads and checks DIR/manifest.tsv (at most 16 MB).
    bool open(const std::string& dir, std::string& err);
    // Checks manifest bytes already in memory (what open does after reading; fuzzed).
    bool parse(const uint8_t* data, size_t n, std::string& err);
    // Reads the file for a tensor (at most max_bytes) and checks its SHA-256 against
    // the manifest. On any failure out is left empty.
    bool read_verified(const std::string& name, uint64_t max_bytes, std::vector<uint8_t>& out, std::string& err) const;
};

// Reads a whole file, refusing one larger than max_bytes.
bool read_whole_file(const std::string& path, uint64_t max_bytes, std::vector<uint8_t>& out, std::string& err);

} // namespace hemispheres
} // namespace pchie
