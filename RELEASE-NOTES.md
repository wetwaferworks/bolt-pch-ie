# Release notes

## Next release (unreleased)

This release is a security hardening pass. It comes from our first security review, by Gizmo (ChatGPT Luna 6). Every finding was checked against the code and fixed, and each fix has its own tests. No official or neutrality hash moved: the maze H_world, the B1 actions and the gauntlet H_bolt are the same as before.

### What changed for you

- **TESLA is seeded from the operating system.** Each run draws a fresh 32-byte secret from the OS random generator. If none is available, TESLA refuses to start instead of falling back to a weak seed. For deterministic unit work, pass `--tesla-seed dev-insecure`; that seed is public, so a stream made with it proves nothing.
- **The TESLA commitment changes every run.** That is by design. It is not an official hash.
- **The TESLA watcher is stricter.** It refuses duplicate steps, caps indices, stream length and line length, checks keys in linear time, and no longer accepts the public commitment as the key for interval 2^32-1.
- **Witnesses pin the log key.** Pin it ahead of time with `pchie_anchor witness-trust <witness_dir> <log_pub_hex>`, or it is pinned on first contact. A packet under a different key is refused with `LOG KEY CHANGED`.
- **Known-good registry entries include the TESLA commitment.** An entry matches only the run whose commitment it records.
- **Malformed models are refused at load.** The GGUF reader and the Gemma 4 loader check sizes, shapes, types, offsets and token ids, including the BOS and EOS ids, before trusting the file. `embed()` refuses token ids outside the vocabulary.
- **One-time signing is safe across processes.** An exclusive lock and a durable index reservation mean a leaf is never used twice. An exhausted key refuses to sign.
- **AVX2 is a build option.** `-DPCHIE_AVX2=AUTO|ON|OFF`, AUTO by default. Without AVX2 the portable kernels are used; check the maze hashes before comparing such a build with an AVX2 one.
- **Witness ids are limited** to letters, digits, `.`, `_` and `-`, 1 to 64 characters, with no leading dot or dash and no `..`.
- **Secrets are wiped after use** (best effort).
- **New [SECURITY.md](SECURITY.md)** with the status of the cryptography, the fixes, the known limits and how to report a problem.

### Known limits

- The cryptography here is in-house and research-grade. No one has independently reviewed it.
- The anchor is file-based, not networked. Local files can be rewritten by anyone with access to the machine.
- Pinning on first contact trusts the first key a witness sees.
- Wiping secrets is best effort.

The full list is in [CHANGELOG.md](CHANGELOG.md).

## 0.1.0

First public release.