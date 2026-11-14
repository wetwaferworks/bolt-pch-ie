# Security notes

## Status

TESLA stream authentication, the one-time signatures (PCHIE-HBS) and witness cosigning in this tree are research-grade. The cryptography is written in-house from the public descriptions (RFC 4082, RFC 8554, RFC 9162). No one has independently reviewed it. Do not rely on it to protect anything of value.

The transparency anchor is file-based, not networked. The log, its signed tree heads, witness packets and cosignatures are files on disk that people pass around by hand. Nothing here contacts a witness, and there is no gossip between witnesses. A witness only protects what it has actually been shown.

The harness is tamper-evident, never tamper-proof.

## November 2026 hardening

An outside static security review of the tree raised the issues below. Each one was checked against the code and fixed, and each fix has its own tests. The review's point that the anchor is file-based and that local keys can be rewritten by anyone with access to the machine is a design limit, described under Status above, not something a code change removes.

- **TESLA seed.** The maze drivers seeded TESLA from the public maze seed, so anyone could rebuild the key chain. TESLA now draws a fresh 32-byte secret from the operating system (BCryptGenRandom on Windows, getrandom on Linux) on every run. If that is not available, TESLA refuses to start and no stream is written. It never falls back to a weak seed. A caller-installed seed is refused if it is short, one repeated byte, or the public derivation. The old deterministic seed is only available with `--tesla-seed dev-insecure` (or `stream_auth_dev_insecure_seed()` in tests), and results record which seed kind was used. Seeds are never saved to disk; `tesla-seed.bin` stays in `.gitignore`.
- **TESLA watcher.** Duplicate step indices are now refused. Step and interval indices, stream length and line length are capped, and out-of-range input stops the watch with a flag instead of making it do unbounded work. Each disclosed key is checked against the last key already known rather than walked all the way back to the commitment, so a full-length stream costs linear hashing. The review also led to a fix for a 32-bit wrap in the key-chain check, where interval 2^32-1 could be "verified" with the public commitment as its key.
- **Witness key pinning.** A witness pins the log's public key, either ahead of time with `pchie_anchor witness-trust <witness_dir> <log_pub_hex>` or on first contact. A packet signed under any other key is refused with `LOG KEY CHANGED`.
- **GGUF reader.** Header counts, string and array lengths, the alignment (a nonzero power of two), tensor shapes, offsets and byte extents are checked against the file size before anything is allocated or any pointer is formed. The loader also checks every tensor it binds against the mapping, checks the model's sizes and token ids are whole numbers in range, and checks every weight has exactly the shape and a type the forward pass expects, so a malformed model is refused at load time instead of reading past a buffer later. A token id outside the vocabulary is refused.
- **Known-good registry.** An entry now pins the TESLA commitment too, so a matching binary with a different stream is not accepted.
- **One-time signing.** Signing takes an exclusive lock that every process honors. The next leaf index is reserved durably (unique temp file, flushed, atomic rename) before a signature is released, so a leaf is never used twice. An exhausted key refuses to sign without touching its file. Signing keys come from the OS random generator.
- **Smaller items.** AVX2 is a build option (`-DPCHIE_AVX2=AUTO|ON|OFF`; AUTO turns it on only when both the compiler and the build machine support it). Witness ids must match `[A-Za-z0-9._-]{1,64}` (no leading dot or dash, no `..`). Signing seeds and key buffers are wiped after use.

## Reporting

Please report security problems privately to the maintainer before you open a public issue.
