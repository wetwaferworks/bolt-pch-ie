// Bolt PCHIE Step 3a - read-only GGUF v3 reader (clean-room, from the public format description).
// Parses the header, metadata and tensor directory; it never writes to the file.
// The file is untrusted: counts, string and array lengths, the alignment, every tensor's shape, offset and
// byte extent are checked against the file size before anything is allocated or any pointer is formed.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pchie {

struct GgufTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;      // ggml type id (0 F32, 1 F16, 2 Q4_0, 8 Q8_0, 12 Q4_K, 14 Q6_K, 30 BF16, ...)
    uint64_t offset = 0;    // relative to the data section
    uint64_t n_elements = 0; // product of dims (checked for overflow)
    uint64_t nbytes = 0;    // byte extent in the data section; 0 when the type's block size is not known here
    bool size_known = false; // true when nbytes was computed and checked against the file
};

// Block layout of a ggml type: elements per block and bytes per block. Returns false for a type this reader
// does not size (such a tensor can be listed, but a loader must refuse to bind it).
bool ggml_type_block(uint32_t type, uint64_t& block_elems, uint64_t& block_bytes) noexcept;

struct GgufFile {
    uint32_t version = 0;
    uint64_t alignment = 32;
    uint64_t data_offset = 0;
    uint64_t file_size = 0;
    std::map<std::string, std::string> meta_str;   // string scalars
    std::map<std::string, double>      meta_num;   // numeric/bool scalars
    std::map<std::string, uint64_t>    meta_arr_len;
    std::map<std::string, std::vector<std::string>> meta_str_arr; // string arrays (vocab, merges)
    std::map<std::string, std::vector<double>>      meta_num_arr; // numeric/bool arrays
    std::vector<GgufTensorInfo> tensors;

    // Returns false and sets err on any malformed input.
    bool open(const std::string& path, std::string& err);
    const GgufTensorInfo* find(const std::string& name) const;
    // Stable digest of the directory (names, dims, types, offsets) for the hash gate.
    uint64_t directory_digest() const noexcept;
};

const char* ggml_type_name(uint32_t t) noexcept;

} // namespace pchie
