// Bolt PCHIE - ternary converter (clean-room): reads a GGUF or safetensors model one tensor at a time and turns
// its weight matrices into PTN1 ternary tensors (pchie_ternary.h: I2 packing, one scale per 128-weight block).
//
// Inputs are untrusted. Both header readers check every count, shape, offset and extent against the file size
// before anything is allocated or read, refuse overlapping tensors, and fail closed on anything they don't know.
// The model file is opened read-only and never written.
//
// The fit runs on the CPU (threads over disjoint groups, joined at the end) or on CUDA. Both run the same
// fit_group() from pchie_ternary_fit.h, so they give the same codes and scales bit for bit; the tests check it.
// Nothing here is wired into inference. See docs/ternary/README.md, "Converter".
#pragma once
#include "pchie/pchie_ternary.h"
#include "pchie/pchie_ternary_fit.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pchie {
namespace ternary {
namespace convert {

// ------------------------------------------------------------------------------------------------ sources
enum class SrcType : uint8_t { F32 = 0, F16, BF16, Q8_0, Q4_K, Q6_K, Other };
const char* src_type_name(SrcType t) noexcept;

struct SourceTensor {
    std::string name;
    std::string type_name;          // as the file names it ("BF16", "Q4_K", ...)
    SrcType type = SrcType::Other;  // Other: listed and bounds-checked, but this converter can't decode it
    std::vector<uint64_t> shape;    // outermost first (GGUF dims are reversed into this order)
    uint64_t rows = 0, cols = 0;    // row-major view: cols is the innermost (contiguous) dimension
    uint64_t offset = 0;            // absolute file offset of the first byte
    uint64_t nbytes = 0;            // byte extent in the file
    uint64_t row_bytes = 0;         // bytes per row (0 for Other)
};

struct SourceFile {
    enum class Format : uint8_t { Gguf = 1, Safetensors = 2 } format = Format::Gguf;
    std::string path;
    uint64_t file_size = 0;
    uint64_t data_begin = 0, data_end = 0;   // the region every tensor must sit inside
    std::vector<SourceTensor> tensors;       // in the order the file lists them
};

constexpr uint64_t kMaxSafetensorsHeader = uint64_t(16) << 20;   // the MiniJson ceiling
constexpr size_t kMaxTensors = size_t(1) << 20;
constexpr size_t kMaxDims = 8;
constexpr size_t kMaxNameBytes = 4096;

// Parses a safetensors header: json is the header text (header_len bytes, after the 8-byte length) of a file of
// file_size bytes. Checks: header_len <= kMaxSafetensorsHeader and fits the file; the JSON is an object with no
// duplicate keys; "__metadata__", if present, maps strings to strings; every other entry has exactly "dtype"
// (a known dtype), "shape" (at most kMaxDims whole numbers >= 0) and "data_offsets" ([begin, end], whole
// numbers, begin <= end <= data size); end - begin equals the element count times the dtype size (overflow
// checked); no two tensors overlap. On failure out is empty and err says why.
bool parse_safetensors_header(const char* json, uint64_t header_len, uint64_t file_size, const std::string& path,
                              SourceFile& out, std::string& err);
bool open_safetensors(const std::string& path, SourceFile& out, std::string& err);
// GGUF through the repo's reader (GgufFile::open) plus what that reader leaves to its callers: every tensor must
// have a known size (no unsized type can hide an overlap) and no two tensors may overlap.
bool open_gguf(const std::string& path, SourceFile& out, std::string& err);
// Picks the format from the first bytes ("GGUF" magic, else safetensors).
bool open_source(const std::string& path, SourceFile& out, std::string& err);
// The shared layout check: every extent inside [data_begin, data_end), no overlaps. Used by both readers.
bool check_layout(const SourceFile& f, std::string& err);

// Whether a tensor can be converted (2-D or more, cols a multiple of 128, a decodable type, within the hard
// ceilings). why gets the reason when it can't.
bool convertible(const SourceTensor& t, std::string* why);

// Decodes n rows of raw file bytes (n * t.row_bytes) to float, row-major.
Status decode_rows(const SourceTensor& t, const uint8_t* raw, uint64_t n, float* out) noexcept;
// Reads and decodes rows [r0, r0 + n). raw is a reusable buffer. Fails on a short read or a non-finite value.
bool read_rows(std::FILE* fp, const SourceFile& f, const SourceTensor& t, uint64_t r0, uint64_t n,
               std::vector<uint8_t>& raw, float* out, std::string& err);

// ------------------------------------------------------------------------------------------------ fit
// n_groups groups of kGroup weights, contiguous in w. q gets n_groups * kGroup codes, gamma n_groups scales.
// Any non-finite weight gives NonFinite with q and gamma zeroed. threads <= 0 means the hardware count.
Status fit_groups_cpu(const float* w, uint64_t n_groups, FitKind fit, int threads, int8_t* q, float* gamma);

struct CudaInfo {
    bool built = false;          // compiled with the CUDA backend
    int devices = 0;
    std::string device_name;
    uint64_t total_bytes = 0;
};
CudaInfo cuda_info();
struct CudaCounters {
    uint64_t own_bytes_peak = 0;      // device memory this converter held at most
    uint64_t device_used_peak = 0;    // whole-device use (all processes) seen right after our allocations
    double kernel_ms = 0.0, copy_ms = 0.0;
    uint64_t calls = 0;
};
// Same contract and same bits as fit_groups_cpu. Device buffers are kept between calls and grow as needed;
// cuda_release() frees them. Returns Ok, NonFinite (outputs zeroed) or BadArgument; err has CUDA errors.
Status fit_groups_cuda(const float* w, uint64_t n_groups, FitKind fit, int8_t* q, float* gamma, std::string& err);
const CudaCounters& cuda_counters() noexcept;
void cuda_release() noexcept;

// ------------------------------------------------------------------------------------------------ convert
struct ErrorStats {
    uint64_t n = 0, zeros = 0, zero_groups = 0, groups = 0;
    double sum_w2 = 0, sum_e2 = 0, sum_wr = 0, sum_r2 = 0, max_abs_err = 0;
    double rel_error() const noexcept;   // ||W - R|| / ||W|| (Frobenius)
    double cosine() const noexcept;      // <W, R> / (||W|| ||R||)
    double rmse() const noexcept;
    void add(const ErrorStats& o) noexcept;
};
// Adds rows x cols weights and their reconstruction gamma * q to s, row by row in order (any thread count gives
// the same sums).
void accumulate_stats(const float* w, const int8_t* q, const float* gamma, uint64_t rows, uint64_t cols,
                      int threads, ErrorStats& s);

enum class Backend : uint8_t { Cpu = 0, Cuda = 1 };
struct ConvertOptions {
    FitKind fit = FitKind::LeastSquares;
    Backend backend = Backend::Cpu;
    bool check_parity = false;              // also run the other backend and compare every code and scale
    int threads = 0;
    uint64_t chunk_weights = uint64_t(1) << 25;   // weights per streamed chunk (128 MiB of float)
};
struct TensorResult {
    Tensor out;                  // I2, PerBlock
    ErrorStats stats;
    double read_ms = 0, fit_ms = 0, other_fit_ms = 0, stats_ms = 0;
    uint64_t parity_groups_checked = 0, parity_groups_different = 0;
};
// Streams one tensor in row chunks: read and decode, fit, accumulate the error, pack.
bool convert_tensor(std::FILE* fp, const SourceFile& f, const SourceTensor& t, const ConvertOptions& opt,
                    TensorResult& res, std::string& err);

} // namespace convert
} // namespace ternary
} // namespace pchie
