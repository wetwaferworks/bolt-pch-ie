# Specs

This folder holds the specifications the code follows: the outside standards it implements, and our own design rulings and formats.

## Outside standards

[standards.md](standards.md) lists every RFC and standard the harness implements or relies on. For each one it gives what the standard is, what it does, which files and functions implement it, and what the harness uses it for. Here is the short list:

| Standard | Used for |
|---|---|
| FIPS 180-4 (SHA-256) | Default seal chain, weights hash, and everything tamper-evident |
| RFC 2104 (HMAC), RFC 4231 (test vectors) | Ratchet steps, TESLA MACs, simulated quotes |
| RFC 5869 (HKDF) | Ratchet genesis and epoch re-roots |
| RFC 9162 (CT v2 Merkle tree) | `pchie_anchor` transparency log and proofs |
| RFC 8554 (LMS), as a model only | Hash-based signatures on tree heads and cosignatures |
| RFC 3339 | Tree-head timestamp text |
| RFC 4082 (TESLA), adapted | Stream authentication of sealed steps |
| RFC 9334 (RATS) | Attestation verdict roles and vocabulary |
| RFC 8259 (JSON), JSON-RPC 2.0, MCP 2025-11-25 | Talking to LithoKraft |
| RFC 3629 (UTF-8) | Tokenizer and JSON strings |
| RFC 4648 (base64), PNG with RFC 1950/1951 stored blocks | Visual navigator screenshots |
| GGUF v3 and GGML K-quant layouts | Loading and running Little Nemo's weights |
| IEEE 754 | Bit-exact CPU and CUDA math |
| FNV-1a 64 (Internet-Draft), SplitMix64 | Fast fingerprints and deterministic seeds (not for security) |

## Our own specs

| Spec | Where it lives | What it fixes |
|---|---|---|
| Search rulings (Archie, 2026-10-29) | [search-rulings.md](search-rulings.md) | Leaf value, terminal tokens, FPU, virtual loss, per-node seeds for the PUCT search |
| Transparency Anchoring for the PCHIE Hash Chain (Elfie's brief, 2026-11-12) | Implemented in `anchoring/`; see [anchoring/README.md](../../anchoring/README.md) and [Transparency anchoring](../guide/transparency-anchoring.md) | Anchor records, the log, witness cosigning, the fingerprint, ratchet-file checks. The brief's section numbers (s3.1 to s3.6) are cited in `pchie_anchor.h` |
| Chain record wire format | `include/pchie/ChainRecord.h` | `kind:u32 \| payload_len:u32 \| payload`, little-endian; record kinds |
| Seal chain generations | `include/pchie/ChainGeneration.h`, `include/pchie/RatchetHashChain.h`, [Seal chains](../guide/seal-chains.md) | `pchie.seal.v1-sha256` (default), `pchie.seal.v2-ratchet`, `pchie.seal.v2-ratchet-salted` |
| TESLA adaptation | `include/pchie/TeslaKeyChain.h`, [TESLA stream authentication](../guide/tesla-stream-authentication.md) | `pchie.tesla.v1` tags, chain order as the clock, I = 8, d = 2 |
| Anchoring text formats | `anchoring/src/pchie_anchor.h`, `anchoring/src/pchie_anchor.cpp` | STH line, `pchie.cosig.v1`, `pchie.witness_packet.v1`, fingerprint v1 and v2, witness policy files |
| Known-good registry | [known-good-registry.md](../notes/known-good-registry.md) | Append-only reference values for attestation |
| Stream-auth defaults | [stream-auth-defaults.md](../notes/stream-auth-defaults.md) | TESLA and attestation on by default, and the off flags |
| Loop guard v2 hash moves | [loop-guard-default-hash-moves.md](../notes/loop-guard-default-hash-moves.md) | Which hashes moved, and how `--loop-guard off` reproduces the old ones |
| Little Nemo's episodic memory (NEML) | `include/pchie/neml/`, [quick start, section 6](../guide/quick-start-maze.md) | `NEML0002` ring of 16 x 256-byte receipts, the 384-token memory block, the memory prior, and its `neml.snap.v1` provenance record |
| MCP game seam | `third_party/pch-public/engine/docs/MCP-GAME-SEAM.md` | The engine's tools, and one JSON-RPC message per line over stdio |
