# NanoQuant in PCHIE

This path compresses a linear layer to a low-rank pair of binary matrices and two
full-precision scale vectors, following NanoQuant (Chong, Kim, Kim and Choi, Samsung
Research, ICML 2026). The first milestone is the published initialization: a
Hessian-aware precondition, latent binary ADMM (LB-ADMM), magnitude balancing, and
the mean-absolute scale extraction, plus a short least-squares polish of the scales
with the signs held fixed. Block reconstruction, STE refinement and the 128-sample
model-level scale fit are described under "Block and model reconstruction" below.

For a weight matrix `W` of shape `rows x cols`:

```
W[i,j]  ~  s1[i] * (U[i,:] · V[j,:]) * s2[j]
```

with `U` and `V` in `{+1,-1}` and rank chosen from a bits-per-weight target:

```
rank = floor( (bits * rows * cols) / (rows + cols) - 16 )  to a multiple of 32,
       at least 32, at most min(rows, cols)
```

(the 16 is the paper's budget for two 16-bit scale vectors). Stored here, the
signs are packed one bit per entry and the scales are float32, so the on-disk
bits per weight are a little higher than that formula. The tool also reports the
paper's figure with 16-bit scales.

The Hessian diagonals `d_in` and `d_out` precondition `W` as
`W_tilde[i,j] = sqrt(d_out[i]) * W[i,j] * sqrt(d_in[j])`. When they are omitted,
both are taken as 1, which is the exact initialization with no calibration
statistics. Wide matrices (`rows < cols`) are factored as the transpose, as in
the official solver, and written back in the original layout.

Tool:

```
pchie_nanoquant factor MODEL.safetensors --out OUTDIR --bits 1.0 --bits 0.8
pchie_nanoquant factor MODEL.safetensors --out OUTDIR --bits 1.0 --cuda
pchie_nanoquant factor MODEL.safetensors --out OUTDIR --bits 1.0 --cpu
```

The model is opened read-only. Output goes under `OUTDIR` only. A second run
skips tensors that already have a complete file. `--cuda` requires the device;
`--cpu` stays on the host; the default uses the device when one is present.

## CUDA path

The LB-ADMM inner loop (the sign updates, the scale updates, and the matmuls)
runs on the GPU when a device is present. Factors and ADMM state stay in device
memory. The weight matrix is kept in pinned host memory and streamed as
double-buffered tiles with `cudaMemcpyAsync` on a pair of copy streams that
overlap the cuBLAS work. Tile size is chosen from `cudaMemGetInfo` at the start
of each tensor, leaving 256 MiB of device memory in reserve so a 4 GB card is
not asked to page through WDDM shared memory. cuBLAS runs in pedantic / IEEE
mode (no TF32). The r×r system is factored and solved on the device (cuSOLVER) so the many right-hand sides do not sit on the host.

CPU vs CUDA parity is checked in `pchie_nanoquant_test` on small matrices, with
documented tolerances:

- `|cosine_cpu - cosine_cuda| ≤ 1e-4`
- `|rel_cpu - rel_cuda| ≤ 1e-4`
- relative L2 between the two reconstructions `≤ 1e-3`
- scale L2 relative `≤ 1e-3` when every sign agrees

Signs may differ where a latent is under `1e-5` in magnitude; a reconstruction
that stays inside those tolerances still passes. The CUDA section is skipped,
not failed, when the test is built or run without a device.

## Block and model reconstruction (Milestone 2)

The published reconstruction schedule is implemented in C++ for a stack of residual
MLP blocks, `y = x + W2 gelu(W1 x)`. That is enough to exercise the loop and the
resume path without a full decoder forward. A later hook can drive a real model
once that forward exists.

For each linear in a block, in order:

1. The remaining dense (not-yet-factored) linears are tuned with AdamW (weight
   decay 0), cosine learning-rate decay, and a weighted MSE against the teacher
   block's outputs. Importance follows the paper: the down-projection's output
   Hessian diagonal, with Ledoit-Wolf-style shrinkage `H' = (1-α)H + α mean(H)`
   and `α = 0.4`.
2. That linear is factorized with the Milestone 1 LB-ADMM initializer (streamed
   CUDA when `--cuda` is requested).
3. The new factors are refined with a straight-through estimator (signs in the
   forward pass, identity in the backward pass), then the signs are finalized.

After every linear in the block has been replaced, the student's activations
become the next block's inputs, and the teacher's activations are the
full-precision block outputs. After the last block, a scale-only
knowledge-distillation pass trains the scales against teacher logits (KL), with
the head held fixed.

Defaults match the paper's counts where they apply to this path: 128 calibration
samples, 8 epochs for each of the three stages, non-factored batch 4 and
learning rate `1e-4`, factored and model-distillation batch 1 and learning rates
`1e-5`, 400 ADMM iterations with a linear penalty from 0 to 1 and ridge 0.03.
Sequence length defaults to 2048; the tiny test uses a much shorter sequence.

### Checkpoints and resume

Output is only under `--out`. Each finished block is written to
`blocks/block-NNN.nqb` through a temporary file, `fsync`, and an atomic rename
(`MoveFileEx` with `REPLACE_EXISTING` on Windows). A SHA-256 of the payload is
appended and checked on read. A progress journal `journal.nqj` (format `NQJ1`)
records the teacher snapshot hash, the calibration hash, each block hash, and
whether the distillation pass has finished (`kd-final.nqk`). A kill in the
middle of a block leaves the last complete journal entry; the next run restarts
that block from the saved teacher weights and activations with the same seed,
and is bit-identical to a run that was never interrupted. The test
`pchie_nanoquant_recon_test` checks this on a two-block toy model, on the CPU,
so a live GPU initialization job is barely touched.

```
pchie_nanoquant reconstruct --out DIR --tiny --cpu
pchie_nanoquant reconstruct --out DIR --pause-input-ms 2000 --respect-gpu-busy
```

`--tiny` starts a residual-MLP toy run (or resumes it if the journal is already
there). Without `--tiny`, the tool only resumes an existing journal; it will not
invent a job or write an original model.

### Idle runner

`pchie_nq_idle` is meant to be launched by Task Scheduler. It watches a
directory (default `D:\scratch\nanoquant-jobs`). A `job.txt` file in that
directory names a reconstruct output directory that already has a journal. If
the user has used the machine in the last two seconds (`GetLastInputInfo`), or
the GPU is already in use by `pchie_nanoquant`, a Play build (A9–E9,
PlayMutexFree, Play-G1), or any other compute process `nvidia-smi` reports, the
runner exits without starting anything. An empty watch directory is a no-op.

While a job is running, the reconstruct process itself also pauses within
seconds of user input, and the idle runner will stop the child if it has not
exited shortly after that. Checkpoints are atomic, so a mid-block stop is safe
to resume.

A user-level task definition is in `tools/pchie_nq_idle.xml` (an idle trigger
and a nightly 02:00 local start, `RunOnlyIfIdle`, `StopOnIdleEnd`, least
privilege, no administrator). Register it with:

```
schtasks /Create /TN "PCHIE NanoQuant reconstruct (idle)" /XML "D:\scratch\pchie-2026-11-17\tools\pchie_nq_idle.xml" /F
```

If that command is refused, leave the XML and that one-line command as they are;
do not elevate just to register the task.

## Credits

NanoQuant is the work of Hyochan Chong, Dongkyu Kim, Changdong Kim and Minseop Choi
at Samsung Research, Seoul. The paper is "NanoQuant: Efficient Sub-1-Bit
Quantization of Large Language Models", *Proceedings of the 43rd International
Conference on Machine Learning*, PMLR 306:20335-20364, 2026, arXiv:2602.06694
(v3, 14 June 2026). Official code: [github.com/SamsungLabs/NanoQuant](https://github.com/SamsungLabs/NanoQuant),
Apache License 2.0. Every line in this tree was written here from the published
method; the official repository was read only as a reference for solver details
the paper states at a high level (Sign-Value Independent Decomposition, the
penalty schedule, the stabilized Cholesky shift, and the mean-absolute scale
extraction). No source from that tree is copied here.

NanoQuant builds on, and we credit, the methods named in the paper and its
references, including:

- Low-rank binary factorization as used in LittleBit (Lee et al., 2025) and DBF
  (Boža and Macko, 2026), which NanoQuant follows for the `s1 (U V^T) s2` form.
- The alternating direction method of multipliers (Gabay and Mercier, 1976;
  Boyd et al.).
- Sign-Value Independent Decomposition (Pouransari, Cao and Tuzel, 2020; also
  OneBit, Xu et al., 2024).
- Kronecker-factored approximate curvature (K-FAC) for a Hessian-weighted
  reconstruction (Martens and Grosse, 2015), and Ledoit-Wolf shrinkage of the
  diagonal estimates (Ledoit and Wolf, 2004).
- The straight-through estimator (Bengio, Léonard and Courville, 2013), used in
  the later block-reconstruction stage, not in this initialization.
- The broader post-training quantization line the paper sits in, including
  AdaRound (Nagel et al., 2020), GPTQ (Frantar et al., 2022) and related
  Hessian-aware PTQ methods.

Samsung's NOTICE also credits Benjamin Warner's `optimi` (MIT). That code is not
part of this tree.

The Apache 2.0 license text of the official NanoQuant project is at
https://github.com/SamsungLabs/NanoQuant/blob/main/LICENSE.
