# Changelog

All notable changes to Bolt PCHIE are listed here, newest first.

## Unreleased

### Added

- **Research layers (opt-in, off by default).** Go-Explore as L7, an exact constraint peer beside L6 with an optional pinned Z3, a step scorer, Tree of Thoughts, Reflexion lesson receipts (never injected into the prompt) and a MuZero-style frozen dynamics look-ahead, each behind its own `--research-*` flag in a separate `pchie_research` library. `pchie_core`, the `--rl-*` flags and every existing hash are unchanged. Also the 2048 context test path (`--research-ctx`, `experiments/next-runs/run_context_compare.ps1`). See [experiments/research-layers.md](experiments/research-layers.md).
- **Research layers in the maze runs.** `pchie_rl_l1l2_test --maze b1|stagea` takes Go-Explore, the look-ahead (with a pinned frozen model), the step scorer and Reflexion through `--research-*` flags. They can only move or lower prior shares on legal edges, and they write their own side chain. Without research flags the runs and every hash are unchanged. New `pchie_research_tool` (freeze a model from side logs, verify a side chain).
- **ASCII views for the context comparison.** `pchie_lk_visnav --obs ascii[=1024|2048]` adds verified Nemo View pictures (`vision.capture_ascii`) to the live view, with their own side chain. Off by default.

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

### Changed

- **The TESLA commitment now differs on every run.** This is expected: it comes from a fresh secret each time. It is not an official hash and never fed the seals or the chain. The maze H_world, the B1 actions and the gauntlet H_bolt are unchanged. Only the dev-insecure seed reproduces the old commitments.

### Known limits

- TESLA, the one-time signatures (PCHIE-HBS) and witness cosigning are in-house, research-grade cryptography that has not been independently reviewed.
- The transparency anchor is file-based, not networked. Anyone with access to the machine can rewrite local files; a witness only protects what it has been shown.
- Pinning on first contact trusts the first log key a witness sees. Use `witness-trust` to pin the key ahead of time.
- Wiping secrets from memory is best effort; the compiler, the OS and swap can still keep copies.

## 0.1.0

First public release.