// Bolt PCHIE - ternary weights ({-1, 0, +1} with float scales) and int8 activations (clean-room).
//
// Written from scratch from the published math only (BitNet b1.58 absmean weights, absmax int8 activations;
// see docs/ternary/README.md for every source consulted). No BitNet, bitnet.cpp or llama.cpp code was read.
//
// This is a separate, opt-in path. Nothing here is used by the existing Q4/Q6/Q8/BF16 model paths.
//
// Packings
//   I2 (compute format, 2 bits per weight). A row is cut into blocks of 128 weights; a block is 32 bytes.
//      Byte j of a block holds weights j, j+32, j+64, j+96 in bit pairs 0-1, 2-3, 4-5, 6-7, so one shift and one
//      mask give 32 contiguous weights. Each 2-bit code is q + 1 (0 = -1, 1 = 0, 2 = +1); code 3 is invalid.
//   B3 (storage format, 1.6 bits per weight). Five weights per byte, byte = sum_k (q_k + 1) * 3^k for weights
//      5i .. 5i+4 of the row, in order. Bytes 243..255 are invalid; padding weights past the row end must be 0.
//      Each row takes ceil(cols / 5) bytes. Transcoded to I2 for compute.
// Scales
//   PerTensor: one gamma for the whole matrix (what BitNet b1.58 trains with).
//   PerBlock:  one gamma per 128-weight block of each row (rows * cols/128 scales, row-major).
// Dot product (exact integer core, one fixed float tail)
//   PerTensor: isum = sum over the row of q * xq (int32, exact), y = (float(isum) * gamma) * sx.
//   PerBlock:  acc = 0; for each block in order acc = acc + float(isum_b) * gamma_b; y = acc * sx.
//   Overflow: |q * xq| <= 127 and cols <= 2^20, so |isum| <= 127 * 2^20 < 2^31.
//
// Rounding: round-half-to-even everywhere (round_half_even_f32 from pchie_q4_det.h for activations; an exact
// comparison rule for ternarization). FMA contraction is off for this code.
#pragma once
#include "pchie/IJobDispatcher.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace pchie {
namespace ternary {

constexpr int kBlock = 128;                 // weights per I2 block and per scale block
constexpr int kBlockBytes = 32;             // bytes per I2 block
constexpr uint64_t kHardMaxCols = 1u << 20; // never exceeded, whatever the caller's limits say
constexpr uint64_t kHardMaxRows = 1u << 20;

enum class Packing : uint8_t { I2 = 1, B3 = 2 };
enum class ScaleMode : uint8_t { PerTensor = 1, PerBlock = 2 };

enum class Status : int {
    Ok = 0,
    BadArgument,       // null pointer, zero or unsupported size
    TooLarge,          // over a hard ceiling or a caller limit
    Truncated,         // buffer shorter than the header or a section
    BadMagic,
    BadVersion,
    BadHeader,         // a field out of range, reserved bytes not zero, sections overlapping or misaligned
    BadChecksum,
    BadCode,           // I2 code 3, B3 byte > 242, or non-zero padding weights
    BadScale,          // non-finite, negative or subnormal scale
    NonFinite,         // NaN or infinity in weights or activations
};
const char* status_name(Status s) noexcept;

struct Limits {
    uint64_t max_rows = 1u << 18;
    uint64_t max_cols = 1u << 16;
    uint64_t max_bytes = uint64_t(1) << 30;   // whole serialized tensor
};

// An owned ternary matrix.
struct Tensor {
    Packing packing = Packing::I2;
    ScaleMode scale_mode = ScaleMode::PerTensor;
    uint64_t rows = 0, cols = 0;               // cols is a multiple of 128
    std::vector<uint8_t> codes;                // packed weights, rows * row_bytes()
    std::vector<float> scales;                 // 1 or rows * cols / 128
    uint64_t row_bytes() const noexcept;
    uint64_t blocks_per_row() const noexcept { return cols / kBlock; }
};

// A validated, non-owning view (into a parsed buffer or a Tensor). Only make one through parse() or view_of().
struct View {
    Packing packing = Packing::I2;
    ScaleMode scale_mode = ScaleMode::PerTensor;
    uint64_t rows = 0, cols = 0;
    const uint8_t* codes = nullptr;
    const uint8_t* scales_raw = nullptr;       // little-endian float32, possibly unaligned
    uint64_t row_bytes = 0;
    float scale_at(uint64_t i) const noexcept;
};

// Bytes per row for a packing (0 if cols is not a valid width).
uint64_t row_bytes_for(Packing p, uint64_t cols) noexcept;

// --- quantizers -------------------------------------------------------------------------------------------
// Absmean ternarization (BitNet b1.58): gamma = mean |W| over the scale group (double, sequential order),
// stored as float; q = -1 if W/gamma < -0.5, +1 if W/gamma > 0.5, else 0 (ties go to 0, the even value).
// gamma == 0 gives all zeros, and so does a gamma that would be subnormal in float (below FLT_MIN): the group is
// stored as an exact zero group. W is row-major rows x cols; cols must be a multiple of 128.
Status quantize_absmean(const float* W, uint64_t rows, uint64_t cols, ScaleMode mode, Packing packing,
                        Tensor& out, const Limits& lim = Limits());

// One activation row, quantized: xq = clamp(round_half_even(x * 127 / absmax), -127, 127), scale = absmax / 127.
// bsum[b] = sum of xq over block b (used by the SIMD kernels). n is a multiple of 128.
struct ActRow {
    std::vector<int8_t> q;
    std::vector<int32_t> bsum;
    float scale = 0.0f;
    uint64_t n = 0;
};
// A NaN or infinity anywhere gives NonFinite and an all-zero row (scale 0), never garbage.
Status quantize_activations(const float* x, uint64_t n, ActRow& out);

// --- packing helpers --------------------------------------------------------------------------------------
// Pack / unpack one row of ternary values (int8 in {-1,0,1}, length cols).
Status pack_row(const int8_t* q, uint64_t cols, Packing p, uint8_t* dst) noexcept;
Status unpack_row(const uint8_t* src, uint64_t cols, Packing p, int8_t* q) noexcept;
// B3 -> I2 (and back), whole tensor. Scales are copied.
Status transcode(const Tensor& in, Packing to, Tensor& out);

// --- serialization (untrusted input on the parse side) -----------------------------------------------------
std::vector<uint8_t> serialize(const Tensor& t);
// Validates everything before returning Ok; on any failure `out` is left empty (rows = 0).
Status parse(const uint8_t* buf, size_t len, const Limits& lim, View& out);
Status view_of(const Tensor& t, View& out);

// --- compute ----------------------------------------------------------------------------------------------
enum class Tier : int { Scalar = 0, SSSE3 = 1, AVX2 = 2, NEON = 3 };
const char* tier_name(Tier t) noexcept;
// Best tier this CPU and OS support. PCHIE_TERNARY_TIER=scalar|ssse3|avx2|neon can only lower it.
Tier detect_tier() noexcept;
bool tier_supported(Tier t) noexcept;

// Scalar reference: the oracle. Every other kernel must match it bit for bit.
// Works for both packings (B3 is decoded on the fly).
float dot_row_reference(const View& w, uint64_t row, const ActRow& x) noexcept;
// Fast row dot for I2 at the given tier (falls back to the reference for B3 or an unsupported tier).
float dot_row(const View& w, uint64_t row, const ActRow& x, Tier tier) noexcept;

// y[r] = dot(row r, x) for every row. Rows go to workers in chunks of 16; each output is written by exactly one
// worker in a fixed order, so the result does not depend on the thread count. pool may be null (serial).
// Shapes are checked; on a mismatch y is zero-filled and BadArgument is returned. If any output overflows to
// infinity (finite but huge scales), y is zero-filled and NonFinite is returned.
Status matvec(const View& w, const ActRow& x, float* y, uint64_t y_len, Tier tier, IJobDispatcher* pool);

// FNV-1a 64 over bytes (the format checksum).
uint64_t fnv1a64(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) noexcept;

}  // namespace ternary
}  // namespace pchie
