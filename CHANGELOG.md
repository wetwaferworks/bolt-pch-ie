# Changelog

All notable changes to Bolt PCHIE are listed here, newest first.

## 0.1.1

### Added

- **Research layers (opt-in, off by default).** Go-Explore as L7, an exact constraint peer beside L6 with an optional pinned Z3, a step scorer, Tree of Thoughts, Reflexion lesson receipts (never injected into the prompt) and a MuZero-style frozen dynamics look-ahead, each behind its own `--research-*` flag in a separate `pchie_research` library. `pchie_core`, the `--rl-*` flags and every existing hash are unchanged. Also the 2048 context test path (`--research-ctx`, `experiments/next-runs/run_context_compare.ps1`). See [experiments/research-layers.md](experiments/research-layers.md).
- **Research layers in the maze runs.** `pchie_rl_l1l2_test --maze b1|stagea` takes Go-Explore, the look-ahead (with a pinned frozen model), the step scorer and Reflexion through `--research-*` flags. They can only move or lower prior shares on legal edges, and they write their own side chain. Without research flags the runs and every hash are unchanged. New `pchie_research_tool` (freeze a model from side logs, verify a side chain).
- **Tree of Thoughts and the constraint peer in the maze runs.** `--research-tree-of-thoughts` and `--research-smt` now reach the B1 and Stage A decide loop. Both plan only on Little Nemo's own map and what he saw next to him. The door/key plan can be checked by a pinned Z3 (`--research-smt-sha`), and any disagreement stops the run. Results are the same at every thread count.
- **ASCII views for the context comparison.** `pchie_lk_visnav --obs ascii[=1024|2048]` adds verified Nemo View pictures (`vision.capture_ascii`) to the live view, with their own side chain. Off by default.
- **Harness: tasks, solvers, scorers and transcripts.** A new `pchie_harness` library runs tasks the Inspect AI way, rebuilt in C++: samples, a solver, a scorer, integer scores, and a transcript per sample written onto its own side chain in sample order. Results and the chain head are the same at every thread count. Off the official path. See [experiments/harness.md](experiments/harness.md).
- **Harness: one plug for many games.** Every game now speaks one interface (reset, step, a text observation, fixed actions, integer rewards, a state digest), after the BALROG pattern. Built in: the headless LithoKraft B1 and Stage A mazes, a LithoLattice testbed stub, and a small Craftax-like crafting grid. Agents plug into the other side; this is the socket Mjolnir will use. Grading is read after the episode and never shown to the agent. Game episodes run as harness tasks, so they get transcripts, a side chain and the same results at every thread count.
- **Gymnasium adapter (off-path).** `tools/gym/pchie_gym.py` wraps any harness plug as a Gymnasium environment for outside researchers, talking to the new `pchie_plug_server` over stdin/stdout. It is never in a hashed path and needs no network. It also runs without Gymnasium installed, using small stand-ins.
- **Skill checks (scaffolding).** A small fixed suite (compass, glyph counting, grid reading, maze rules) with exact scoring, to run before and after any model change so a regression shows up as a number. Off the official path.

### Security

These fixes come from our first security review, by Gizmo (ChatGPT Luna 6). Each finding was checked against the code, fixed, and covered by new tests. See [SECURITY.md](SECURITY.md) for the details and the known limits.

- **TESLA seed from the operating system.** TESLA used to seed its key chain from the public maze seed, so anyone could rebuild it. Every run now draws a fresh 32-byte secret from the OS random generator (BCryptGenRandom on Windows, getrandom on Linux). If none is available, TESLA refuses to start and no stream is written; it never falls back to a weaker seed. An installed seed is refused if it is short, one repeated byte, or the public derivation. The old deterministic seed is only available with `--tesla-seed dev-insecure` (or `stream_auth_dev_insecure_seed()` in tests), and results record which kind of seed was used. Seeds are never saved to disk.
- **TESLA watcher hardening.** A step index seen twice is refused as a duplicate. Step and interval indices, stream length and line length are capped, and out-of-range input stops the watch with a flag. Each disclosed key is checked against the last key already trusted, so a full-length stream costs linear hashing instead of quadratic. Fixed a 32-bit wrap in the key-chain check where interval 2^32-1 could be "verified" with the public commitment itself as its key. The watcher tool refuses oversized files and malformed headers with a clear message.
- **Witness log-key pinning.** A witness pins the log's public key, either ahead of time with `pchie_anchor witness-trust <witness_dir> <log_pub_hex>` or on first contact. A packet signed under any other key is refused with `LOG KEY CHANGED`.
- **Known-good registry pins the TESLA commitment.** A registry entry now matches only when the TESLA commitment matches too, so the same binary and weights with a different stream are not accepted.
- **GGUF and Gemma 4 model checks.** The GGUF reader checks header counts, string and array lengths, the alignment, tensor shapes, offsets and byte extents against the file size before it allocates anything or forms a pointer. The Gemma 4 loader checks every tensor it binds against the mapping, checks that model sizes are whole numbers in range, and checks every weight has exactly the shape and type the forward pass expects. The tokenizer's special token ids (BOS and EOS) must be inside the vocabulary, and `embed()` refuses a token id outside it.
- **One-time signing never reuses a leaf.** Signing takes an exclusive lock that every process honors. The next leaf index is reserved durably (unique temp file, flushed, atomic rename) before a signature is released. An exhausted key refuses to sign and leaves its file unchanged. Signing keys come from the OS random generator.
- **Smaller items.** AVX2 is now a CMake option, `-DPCHIE_AVX2=AUTO|ON|OFF`; AUTO (the default) turns it on only when both the compiler and the build machine support it. Witness ids must match `[A-Za-z0-9._-]{1,64}`, with no leading dot or dash and no `..`. Signing seeds and key buffers are wiped after use (best effort).
- **SECURITY.md.** New security notes covering the status of the cryptography, the hardening above, the known limits and how to report a problem.

### Security: hardening review

These fixes come from the hardening review by Helia (Google Hydrogen), cross-checked by Luna (ChatGPT Luna 6), plus a few found while checking their reports. Each fix has its own test, and none of them moves a run hash.

- **CUDA matvec row limit.** The Q4_K, Q6_K and Q8_0 GPU kernels keep one partial sum per block in a fixed shared array. A row with more blocks than that array holds is now refused at backend start, so the run stays on the CPU instead of corrupting the sums.
- **CUDA scratch sizing.** The GPU scratch buffers are sized from the widest head and the widest matvec input, not from the global head and `d_ff`, so an unusual but accepted model cannot write past them.
- **CUDA copy mirror unhooked on shutdown.** Destroying the GPU backend removes its copy-on-write hooks from the KV pools, and a failed GPU page copy stops the next forward pass instead of being ignored.
- **BF16 rows must be a multiple of 8.** The loader refuses other BF16 rows, and the CPU and GPU dot products no longer read past a short last group.
- **Norm, softcap and rope settings must be finite and positive.** The RMS norm epsilon, the logit softcap, both rope bases and every rope frequency factor are checked at load, so a bad file is refused instead of producing NaN logits.
- **NaN-safe deterministic exp and rounding.** NaN no longer reaches an undefined float-to-int cast, and out-of-range values saturate. Results for every finite in-range input are unchanged.
- **Model file size checks and tensor alignment.** The size calls are checked, an empty file is refused, the file handle is closed on every failure, and any tensor not on a 4-byte boundary is refused.
- **KV page bounds.** Every KV read checks the layer, the page table and that the page is live, in release builds too. Reference counts are checked before every release and copy, positions past a 16M-token ceiling are refused, and a failed generation releases its pages.
- **TESLA watcher gap report and step cap.** Missing steps are counted with only the first 1024 listed, and the step space is capped at 2^24, so a short hostile stream can no longer ask for gigabytes.
- **TESLA after-end flag.** A step after the end-of-steps marker, or a second marker, now fails the stream.
- **Anchor emits are serialized.** Emits take a lock on the log directory, re-read the log, check for a duplicate run id and append with a flush to disk before the tree head is signed, so two emits can no longer fork the anchor chain.
- **Anchor argument and cosignature checks.** Every `pchie_anchor` subcommand checks its argument count first, and a cosignature with no signature or tree size is reported as malformed.
- **Merkle ranges bounded.** The Merkle root and proof helpers refuse a tree size larger than the log instead of reading past it.
- **Constant-time compares.** MAC and quote signature checks use a constant-time compare.
- **HMAC and HKDF key wiping.** Key-derived buffers are wiped, also on errors, HKDF refuses more than 8160 bytes, and the anchor signing key wipes its seed when it goes away.
- **Attestation and chain verifier.** A caller can no longer mark a run Attested without a quote appraisal, and the chain verifier no longer names record 0 as the bad link when it cannot locate it.
- **Race-free join errors.** The deterministic join's error paths use atomics (still no mutex), and the job dispatcher refuses a worker count outside 1 to 16.
- **NEML ledger lock owner token.** A stale lock is taken over safely, a holder only removes its own lock, and a snapshot that cannot get the lock reports busy.
- **Stricter MiniJson and MCP handle cleanup.** Raw control characters, lone surrogates and non-JSON numbers are refused, numbers are parsed independent of locale, and the MCP client closes every handle on a failed start.
- **KV free list sorted once per release.** Same page order as before, less work.
- **Checking builds.** `-DPCHIE_SANITIZE=address,undefined,thread` builds everything with sanitizers (MSVC: address only), the three CPU-vs-GPU gates are registered under the `cuda` label when `PCHIE_GATE_MODEL` is set, and `-DPCHIE_FUZZ=ON` builds a reproducible fuzz driver for the MiniJson, TESLA stream and GGUF parsers.

### Changed

- **The TESLA commitment now differs on every run.** This is expected: it comes from a fresh secret each time. It is not an official hash and never fed the seals or the chain. The maze H_world, the B1 actions and the gauntlet H_bolt are unchanged. Only the dev-insecure seed reproduces the old commitments.

### Known limits

- TESLA, the one-time signatures (PCHIE-HBS) and witness cosigning are in-house, research-grade cryptography that has not been independently reviewed.
- The transparency anchor is file-based, not networked. Anyone with access to the machine can rewrite local files; a witness only protects what it has been shown.
- Pinning on first contact trusts the first log key a witness sees. Use `witness-trust` to pin the key ahead of time.
- Wiping secrets from memory is best effort; the compiler, the OS and swap can still keep copies.

## 0.1.0

First public release.