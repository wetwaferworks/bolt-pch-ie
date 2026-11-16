# Ternary weights in PCHIE

An opt-in path for models whose linear layers use ternary weights ({-1, 0, +1} with float scales) and 8-bit
activations, in the style of BitNet b1.58. It sits next to the existing Q4_K / Q6_K / Q8_0 / BF16 paths and does
not change them: nothing in the Gemma 4 model, the GGUF reader or the existing kernels calls it yet, and every
existing hash stays as it was.

Why: much smaller models for phones and older machines. Two bits per weight is a quarter of BF16 and a little
under half of Q4_K, and the inner product needs only integer adds.

## Status

| Piece | State |
|---|---|
| Tensor format and strict parser | done |
| Absmean ternary quantizer, int8 activation quantizer | done |
| Scalar reference matvec (the oracle) | done |
| B3 (1.6 bits per weight) storage and B3 / I2 transcoding | done |
| SSSE3 kernel (older x86, such as Sandy Bridge without AVX2) | done, matches the reference bit for bit |
| AVX2 kernel | done, matches the reference bit for bit |
| NEON kernel (AArch64, with and without the dot-product instructions) | done; checked under emulation, not yet on a phone |
| Run-time CPU dispatch | done (CPUID and OS check on x86, HWCAP on AArch64 Linux and Android) |
| Converter from existing models, model wiring, CUDA kernel | not started |

## The math

- Weights: `gamma = mean(|W|)` over the scale group (computed in double, row by row, then stored as float).
  `q = +1` if `W / gamma > 0.5`, `-1` if `< -0.5`, otherwise 0; an exact tie goes to 0. A group whose gamma is 0,
  or would be subnormal in float (below FLT_MIN), is stored as an exact zero group.
- Scale groups: one gamma for the whole matrix, or one per 128-weight block of each row.
- Activations, per row: `scale = absmax / 127`, `xq = clamp(round_half_even(x * 127 / absmax), -127, 127)`. The
  rounding is the repo's pinned round-half-to-even routine. A NaN or infinity anywhere gives a "non-finite" error
  and an all-zero row, never garbage.
- Dot product: the integer sum `sum(q * xq)` is exact (int32; with at most 2^20 columns it cannot overflow,
  since 127 * 2^20 < 2^31). Then one fixed float tail:
  - one gamma: `y = (float(isum) * gamma) * scale`;
  - block gammas: `acc = acc + float(isum_b) * gamma_b` for each block in order, then `y = acc * scale`.
  FMA contraction is off for this code, and the tail is one shared function, so every kernel gives the same bits.

## Packings

**I2, for compute (2 bits per weight).** Rows are cut into 128-weight blocks of 32 bytes. Byte j of a block holds
weights j, j+32, j+64 and j+96 in bit pairs 0-1, 2-3, 4-5 and 6-7, so a shift and a mask give 32 contiguous
weights that line up with 32 contiguous activations. Each code is `q + 1`: 0 means -1, 1 means 0, 2 means +1,
and 3 is invalid. Because the codes are unsigned, `sum(q * x) = sum(code * x) - sum(x)`, which maps straight onto
the unsigned-times-signed byte multiply-add of SSSE3 and AVX2 and the byte dot products of NEON. `sum(x)` per
block is computed once per activation row.

**B3, for storage (1.6 bits per weight).** Five weights per byte: `byte = sum_k (q_k + 1) * 3^k` for weights
5i to 5i+4 of the row. Bytes 243 to 255 are invalid, and padding weights after the end of a row must be 0. A row
takes `ceil(cols / 5)` bytes. B3 is 20% smaller than I2; it is transcoded to I2 for fast compute, and the scalar
reference also reads it directly.

## File format (`PTN1`)

Little-endian. A 96-byte header, then the scales, then the packed codes.

| Offset | Field |
|---|---|
| 0 | magic `PTN1` |
| 4 | u16 version, 1 |
| 6 | u8 packing: 1 = I2, 2 = B3 |
| 7 | u8 scale mode: 1 = one gamma, 2 = one per 128-weight block |
| 8 | u32 header size, 96 |
| 12 | u32 activation mode, 1 = per-row absmax int8 |
| 16 | u64 rows |
| 24 | u64 columns (a multiple of 128) |
| 32 | u64 scales offset, 96 |
| 40 | u64 scale count (1, or rows x columns / 128) |
| 48 | u64 codes offset (the end of the scales rounded up to 32) |
| 56 | u64 codes size (rows x row bytes) |
| 64 | u64 FNV-1a 64 of every byte after the header |
| 72 | 24 reserved bytes, all zero |

The file ends exactly at the end of the codes. The gap between the scales and the codes is zero.

## Safety

The parser treats the file as hostile. Before it hands back a view it checks the magic, the version, every field
against its only allowed value or range, the shape against hard ceilings (2^20 rows and columns, whatever the
caller asks for) and the caller's limits (rows, columns, total bytes), the exact size of every section, zero
padding and reserved bytes, the checksum, every scale (finite, not negative, and not subnormal, so no result depends on flush-to-zero settings), and every weight code. On any
failure it returns an error and an empty view. The kernels never allocate and never read outside the validated
sections. The matvec checks the shapes it is given and writes zeros if they do not match. If an output overflows float (finite but huge scales), it writes zeros and reports a non-finite error instead of returning infinities.

Threads: rows go to the existing worker pool in chunks of 16, and each output is written by exactly one worker
in a fixed order, so results do not depend on the number of threads. There are no locks.

## Testing

- `pchie_ternary_test`: packing round trips for both packings, every B3 byte value, invalid codes at every
  position, the quantizers on hand-made cases (including exact ties and non-finite input), the parser against every
  truncation, every single-bit change in the header and changes in the payload, the reference against independent
  integer math, every kernel against the reference bit for bit (random weights, all +1, all -1, all zeros,
  alternating signs, 90% zeros, the widest allowed row), matvec at 1 to 12 threads, and the dispatch override.
- `pchie_ternary_fuzz` (with `PCHIE_FUZZ=ON`): a reproducible mutation fuzzer for the parser. Every input that
  parses is run through every supported kernel and compared with the reference. It also builds as a libFuzzer
  target with `-DPCHIE_LIBFUZZER`.
- Sanitizers: build with `PCHIE_SANITIZE=address,undefined` (or `thread` for the threaded matvec test).
- `PCHIE_TERNARY_TIER=scalar|ssse3|avx2|neon` lowers the kernel tier for testing; it can never raise it above
  what the CPU supports.

## Clean-room provenance

Every line here was written from scratch from published descriptions and plain math. No source code from
microsoft/BitNet, bitnet.cpp, llama.cpp (including its ternary formats), T-MAC or any other ternary
implementation was opened, read or copied.

Sources consulted (papers only, read as prose and math):

- Wang et al., "BitNet: Scaling 1-bit Transformers for Large Language Models", arXiv:2310.11453.
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", arXiv:2402.17764
  (absmean weights, absmax int8 activations).
- Wang, Ma, Wei, "BitNet a4.8: 4-bit Activations for 1-bit LLMs", arXiv:2411.04965.
- Ma et al., "BitNet b1.58 2B4T Technical Report", arXiv:2504.12285 (per-token absmax activations; four ternary
  values per byte on the GPU).
- Wang et al., "1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs", arXiv:2410.16144
  (kernel descriptions only).
- Wang et al., "Bitnet.cpp: Efficient Edge Inference for Ternary LLMs", arXiv:2502.11880 (kernel descriptions
  only).
- "BitNet Distillation", arXiv:2510.13998 (converting existing models; for the converter, later).
- Kim et al., "I-BERT: Integer-only BERT Quantization", arXiv:2101.01321, and Jacob et al., "Quantization and
  Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", arXiv:1712.05877 (integer-only
  arithmetic, for the research notes on normalization and softmax).

Our own choices, not taken from any of these: the lane-interleaved I2 block layout, the biased unsigned codes with
per-block activation sums, the B3 padding rule, the `PTN1` format, the shared float tail and the test bench.

The pinned rounding routine and the thread pool interface come from this repo.
