# Ternary weights in PCHIE

An opt-in path for models whose linear layers use ternary weights ({-1, 0, +1} with float scales) and 8-bit
activations. The method is Microsoft's BitNet research (BitNet b1.58): this implementation was written from the
published papers, not from Microsoft's or anyone else's source code. It sits next to the existing Q4_K / Q6_K /
Q8_0 / BF16 paths and does not change them: nothing in the Gemma 4 model, the GGUF reader or the existing kernels
calls it yet, and every existing hash stays as it was.

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
| Converter from GGUF and safetensors (C17, CPU and CUDA fit, same bits) | done; not wired into a model |
| Model wiring, CUDA matvec kernel | not started |

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

## Research path (opt-in, measurements only)

`include/pchie/pchie_ternary_research.h` holds the alternatives measured for Archie's Nov 16 brief. None of it is
wired into a model, and the shipping path above does not use it.

- Converter fits per group: absmean, the exact least-squares ternary fit (sort |w|, pick the k that maximizes
  (sum of the k largest)^2 / k), and an iterative Lloyd fit; rounding rules half-to-even, half-away and seeded
  stochastic, for offline conversion only.
- Two more packings of the same trits, both bit-identical to the I2 reference through their kernels:
  **B3B** (five trits per byte, block aligned, 26 bytes per 128 weights, 1.625 bpw, decoded with divide-by-3
  multiplies against a plane-ordered activation copy) and **TL3** (sign plus 4-bit index per three weights, tiled
  by 32 rows for byte-shuffle table lookups, 1.68 bpw).
- I2 kernel variants: signed (`sign_epi8`, no activation sums) and zero-skip.
- Integer base-2 softmax with three ways to get 2^f (bin-centred 256-entry table, 64-entry table with linear
  interpolation, degree-3 minimax polynomial), any output precision from Q8 to Q30, and integer RMSNorm (int64 sum
  of squares, adaptive fixed-point mean, 64-entry seed and two Newton steps). Tables are generated once with
  60-digit decimal math and checked by the tests.
- `pchie_ternary_research_test` checks all of it, including Archie's adversarial cases (outlier activation
  channels inside blocks, alternating +1/-1 weights, extreme gammas); `pchie_ternary_measure` prints the tables.

The polynomial coefficients and every table are our own fits; no outside code was consulted for any of it.
## Converter

`pchie_ternary_convert` turns the weight matrices of an existing model into PTN1 tensors (I2 packing, one scale per
128-weight block). It is written in C17, with the GPU fit in CUDA C (no C++ templates, standard library or classes).
It reads GGUF (F32, F16, BF16, Q8_0, Q4_K, Q6_K) and safetensors (F32, F16, BF16), one tensor at a time and in row
chunks (32 Mi weights by default), so memory use stays flat however big the model is. The model file is opened
read-only and never written. Nothing here is wired into a model yet.

```
pchie_ternary_convert MODEL --list
pchie_ternary_convert MODEL --out DIR [--fit lsq|absmean] [--backend cpu|cuda] [--check-parity]
                      [--match TEXT]... [--skip TEXT]... [--no-default-skips] [--limit N] [--threads N]
                      [--chunk-mweights N]
```

- **Fits.** `lsq` (the default) is the exact least-squares ternary fit of each 128-weight group: sort |w| from
  largest to smallest and take gamma = P_k / k for the first k that maximizes P_k^2 / k, where P_k is the sum of
  the k largest. `absmean` is the BitNet b1.58 rule and gives exactly what `quantize_absmean` gives. Both are the
  research path's `ternarize_group` (half-to-even ties), bit for bit; the tests check it. The codes are then
  assigned against the stored float gamma, as in the rest of the ternary path.
- **CPU and CUDA give the same bits.** Both run `ternary_group_fit` from `include/pchie/pchie_ternary_group_fit.h`,
  which uses only IEEE float and double add, multiply, divide and compare, built with FMA contraction off on the
  CPU and `-fmad=false`, IEEE division and no flush-to-zero in CUDA. On the CPU, least squares sorts |w| with an
  8-bit radix sort of the IEEE bit patterns (the same descending values as a comparison sort, and faster than
  insertion sort). The CUDA kernel is one warp per block and fits 32 groups: the warp copies them into shared memory
  with coalesced loads (each row padded so the threads read different banks), sorts each group's |w| with a
  bitonic network when the fit is least squares (any correct sort gives the same sequence), then each thread
  runs the scan and the codes for its own group. I2 packing writes four codes per byte in one store, and unpacking rejects an
  invalid code 3 with a single mask (`byte & (byte >> 1) & 0x55`). One deliberate change from the research code:
  `w / gamma > 0.5` is tested as `w > gamma / 2` in double. For two floats the two tests always agree (halving a
  float is exact in double, and a float w that is not gamma / 2 is much more than half a double ulp away from it),
  and it saves 128 double divisions per group. A test checks the two rules on 1.6 million pairs, including every
  neighbour of the exact tie.
- **Untrusted input.** GGUF goes through the converter's own C reader, which uses the same bounds checks as the
  repo reader: counts, shapes, alignment and every extent against the file. The converter also refuses a file with
  a tensor type of unknown size (it could hide an overlap) or two tensors that overlap. The safetensors reader
  checks the header length against the file and a 16 MiB ceiling, parses the JSON with its own C parser (duplicate
  keys refused, depth capped), allows only `dtype`, `shape` and `data_offsets` per tensor and string values in
  `__metadata__`, requires every dimension and offset to be a whole number from 0 to 2^53, begin <= end <= data
  size, end - begin = element count x dtype size (overflow checked), and no overlaps. A refusal leaves nothing
  behind. Decoded weights must be finite.
- **Output.** `DIR` must be new or empty and may not sit inside the model's folder; no file is ever overwritten
  (each is written as `.partial` and renamed). Per tensor: `NNNN_name.ptn`, re-parsed with the strict parser before
  it is written. Then `manifest.tsv` (shape, source type, error statistics, timings, SHA-256 of each `.ptn`),
  `layers.tsv` and `tensors_by_kind.tsv` (error per layer and per tensor kind) and `summary.txt` (totals, timing,
  CUDA memory). By default names containing `embd`, `embed`, `vision` or `audio` are skipped.
- **Error statistics** compare each weight with gamma * q: relative error ||W - R|| / ||W||, cosine, RMSE, the
  largest absolute error and the share of zero codes, summed in double row by row in order, so the numbers do not
  depend on the thread count. They describe how close each ternary matrix is to its source matrix. They are not a
  measure of model quality; that needs the model wired up and evaluated.
- **Threads.** CPU workers each take a fixed, disjoint range of groups and are joined before the results are read.
  There are no locks.

Tests: `pchie_ternary_convert_test` (the fit against the research fit and the shipping quantizer, edge groups,
least squares never worse than absmean, thread-count independence, non-finite input, malformed safetensors
headers (offsets, sizes, dimensions, dtypes, fields, overlaps, duplicates, header lengths), GGUF and safetensors files end to end with every decodable type, streaming in one-row chunks, overlapping
and truncated files); `pchie_ternary_convert_cuda_test` (label `cuda`; CPU and CUDA codes and scales compared on
every weight kind, around the 32-group block edge and on hand-made edge groups, plus a streamed file with parity
checking on; exits 77, shown as skipped, without CUDA); `pchie_ternary_convert_fuzz` (with `PCHIE_FUZZ=ON`; a
reproducible mutation fuzzer for the safetensors header reader that re-checks every accepted layout and decodes
and fits each small tensor; also a libFuzzer target with `-DPCHIE_LIBFUZZER`).

## Packed storage (`PTT1` tensors in `PTP1` packs)

PTN1 keeps 2 bits per weight (I2, the compute layout) and a float32 scale per 128 weights: 2.25 bits per weight.
`include/pchie/pchie_ternary_pack.h` adds a storage format that holds the very same ternary values in less space.
A reader unpacks it to I2 once, at load time; nothing about compute changes. All of it is C17
(`src/pchie_ternary_pack.c`, `tools/pchie_ternary_pack.c`).

Trits (lossless; unpacking gives back the PTN1 I2 codes bit for bit):

- **B3**: five weights per byte (3^5 = 243 <= 256), `ceil(cols / 5)` bytes per row, so 1.6 bits per weight.
  The same definition as `Packing::B3` in `pchie_ternary.h`; the tests check the bytes match. Unpacking runs
  through a 243-entry table, about 2 G weights a second on one core (4.7 G a second straight to I2).
- **RANS**: the row-major trit stream entropy-coded with a static three-symbol rANS model (frequencies out of
  4096, stored with the tensor), cut into chunks of whole rows (about 1 Mi trits each) that decode
  independently. On these models it comes to about 1.52 bits per weight, about 5% under B3, and decodes at about
  180 M weights a second per core. Optional.

Scales (one per 128-weight group; the last group of a row may be shorter):

- **f32**: as in PTN1, lossless.
- **f16**: IEEE half of `scale * 2^shift`, one int8 shift per tensor so small scales keep their precision.
- **u8row**: per row a float32 reference (the row's largest scale) and one byte per group,
  `scale = (float)(((double)ref * u) / 255.0)`. About 0.07 bits per weight instead of 0.25.

The codes never change, only the scales, and only where the error stays put: `pchie_ternary_pack pack --scales tight`
(the default) measures, for every tensor, the reconstruction error against the source weights with u8row, f16 and
f32 scales and keeps the first whose relative error is within `--tolerance` (default 1e-4, so at most 0.01% worse
than f32). f32 always qualifies. The per-tensor choice and all three errors are in the pack's `.tsv`.

Rows of any length: PTT1 does not need `cols` to be a multiple of 128. `ternary_fit_rows_padded` zero-pads each
group to 128 and runs the same group fit; for least squares that is exactly the fit of the real weights (tested
against the research fit). This is how a Mamba `conv1d` kernel (4 taps per channel) can be stored, if wanted.

A pack can hold a whole model: tensors the ternary conversion skips are kept as **raw** source bytes (norms,
Mamba `A_log` / `D` / `dt_bias`, biases, the vision encoder) or, for the input embedding, as **q8** (int8 with a
float32 scale per 128 weights, about 0.6% relative error). `--embed q8|raw|ternary|skip`,
`--conv1d raw|ternary|skip` and `--extras raw|skip` choose; the summary also reports what ternary would cost on
the embedding and the conv1d kernels when they are kept in another form.

```
pchie_ternary_pack pack MODEL.safetensors --out shard.ptp --reference PTN1_SHARD_DIR   # parity with the PTN1 files
pchie_ternary_pack verify shard.ptp --reference PTN1_SHARD_DIR                         # hashes, decode, exact I2
pchie_ternary_pack recode shard.ptp --out shard-rans.ptp --trits rans                 # lossless
pchie_ternary_pack from-ptn PTN1_SHARD_DIR --out shard.ptp                             # B3 + f32, no source needed
pchie_ternary_pack bench shard.ptp
```

Safety: every reader treats its input as untrusted (sizes and offsets checked before use, overflow-checked
arithmetic, a checksum on each tensor and SHA-256 on each payload and on the index, failures leave zeroed
outputs). The rANS decoder checks every byte it reads and that it ends in its starting state exactly at the
chunk's end. The model, the reference folder and an input pack are opened read-only; output goes to
`FILE.partial`, created new, and is renamed only when complete, never over an existing file. Tests:
`pchie_ternary_pack` (format, refusals, exhaustive half-float rounding, rANS on many shapes and distributions,
parity with I2 and with the C++ B3 packing, the padded fit) and `pchie_ternary_pack_cli` (end to end against the
converter's own PTN1 output). Fuzz target: `pchie_ternary_pack_fuzz`.

## Measured packing of Little Super Nemo

On 19 November 2026 the 63 source shards of Little Super Nemo were packed on TheWorkstation in 1 hour 57 minutes (6:56 AM to 8:54 AM Eastern). The 32.322 GiB PTN1 fallback (`D:\scratch\little-super-nemo-ternary-2026-11-19`, newest file 4:03 AM Eastern) was opened read-only and was not written. Every shard was parity-checked against that fallback.

| Variant | Files | Bytes | GiB | Notes |
| --- | ---: | ---: | ---: | --- |
| PTN1 fallback | (existing) | 34,705,233,389 | 32.322 | 2-bit I2 and float32 scales; left unchanged |
| `b3-f32` | 63 | 28,536,827,040 | 26.577 | five trits per byte, float32 scales; 1.852 bits per ternary weight |
| `b3-tight` | 63 | 27,835,589,658 | 25.924 | compact scales (42,295 byte-per-row, 14 half-float), int8 embedding, 768 raw tensors |
| `rans-tight` | 63 | 26,545,979,701 | 24.723 | same as tight, with rANS on the trit stream (1.518 bits per ternary weight in the stream itself) |

Parity, over all 42,309 ternary tensors and 123,260,108,800 ternary weights, on all three variants: 0 I2 code differences, 0 float32 scale differences, 0 missing tensors, 0 hash, parse or decode failures. The trit packing is 24,673,162,240 bytes (1.601 bits per weight). Compact scales use 1,283,915,832 bytes instead of 3,851,878,400 for float32. The rANS trit stream is 23,383,552,283 bytes. The largest compact-scale change against the fallback was 0.328 relative. Reconstruction relative error of the packed ternary values against the source weights is 0.436, in line with the fallback. Decode: about 4.5-4.9 billion weights a second for B3 to int8, about 210 million for rANS.

## Clean-room provenance

The method this path implements is Microsoft's BitNet research: absmean ternarization and absmax int8
activations, as described in the BitNet and BitNet b1.58 papers listed below. Every line here was written from
scratch from those published papers and from plain math. No source code from microsoft/BitNet, bitnet.cpp,
llama.cpp (including its ternary formats), T-MAC or any other ternary implementation was opened, read or copied.

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

The converter (`pchie_ternary_convert`) has its own C GGUF reader and dequantizers, with the same bounds checks
and the same decode results as the repo's C++ reader. Its safetensors reader was written from the format's public
description (an 8-byte little-endian header length, then a JSON object giving each tensor's dtype, shape and data
offsets). The least-squares fit is the research path's, above.
