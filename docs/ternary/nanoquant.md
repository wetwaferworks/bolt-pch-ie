# NanoQuant in PCHIE

This path compresses a linear layer to a low-rank pair of binary matrices and two
full-precision scale vectors, following NanoQuant (Chong, Kim, Kim and Choi, Samsung
Research, ICML 2026). The first milestone is the published initialization: a
Hessian-aware precondition, latent binary ADMM (LB-ADMM), magnitude balancing, and
the mean-absolute scale extraction, plus a short least-squares polish of the scales
with the signs held fixed. Block reconstruction, STE refinement and the 128-sample
model-level scale fit are later work.

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
```

The model is opened read-only. Output goes under `OUTDIR` only. A second run
skips tensors that already have a complete file.

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
