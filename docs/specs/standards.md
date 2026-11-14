# Standards and RFCs the harness uses

This page lists every outside standard the Bolt PCHIE harness code implements or depends on. Each entry was checked against the source on 13 November 2026 by searching for the algorithm and reading the code that implements it. If a standard is not on this page, nothing in the harness uses it.

Scope: first-party code only, meaning `include/`, `src/`, `tools/`, `tests/`, `vision/`, and `anchoring/`. The vendored LithoKraft engine under `third_party/` has its own crypto. It is covered in a short note at the end, because the harness never links it.

All the crypto here is written in-house on one SHA-256. Nothing imports OpenSSL, libsodium, or a system crypto API. "In the manner of" means we follow the standard's construction, but the code is not claimed to interoperate with other implementations of it.

## Hashing and keyed hashing

### FIPS 180-4, Secure Hash Standard (SHA-256)
- **What it is.** The NIST standard for the SHA-2 hash family. We use only SHA-256.
- **What it does.** Turns any byte string into a 32-byte digest. Finding two inputs with the same digest, or running a digest backwards to its input, is computationally infeasible.
- **Where.** `pchie::Sha256` (`include/pchie/Sha256.h`, `src/Sha256.cpp`). It is checked against the FIPS empty-string and "abc" digests in `tests/pchie_hash_chain_test.cpp` and `tests/pchie_nemotron_test.cpp`.
- **What we use it for.** Almost everything that has to be tamper-evident:
  - the default seal chain `pchie.seal.v1-sha256` (`Sha256HashChain`: h_n = SHA-256(h_{n-1} || record_n));
  - the model weights hash (`GgufWeightsHash`) and the maze area hash (`LkMazeAreaHash`);
  - every HMAC, HKDF, TESLA, attestation, and anchoring construction below;
  - the NEML ledger snapshot hash.

### RFC 2104, HMAC: Keyed-Hashing for Message Authentication
- **What it is.** The standard way to build a message authentication code from a hash function: H((K xor opad) || H((K xor ipad) || m)).
- **What it does.** Only someone who holds the key can produce or check the tag, and a tag reveals nothing about the key.
- **Where.** `pchie::hmac_sha256` in `include/pchie/HmacSha256.h`. This is the only HMAC in the tree. The anchoring tool wraps it as `anchor::hmac`.
- **What we use it for.**
  - every step of the ratcheted seal chain (`RatchetHashChain`);
  - TESLA MAC keys and per-step MACs (`tesla_mac_key`, `tesla_step_mac`);
  - the simulated attestation quote signature (`sign_simulated_quote`).

### RFC 4231, Identifiers and Test Vectors for HMAC-SHA-224, HMAC-SHA-256, HMAC-SHA-384, and HMAC-SHA-512
- **What it is.** Published known-answer tests for HMAC-SHA-256.
- **Where.** Test case 1 is checked in `anchoring/tests/anchor_unit.cpp`.
- **What we use it for.** Proof that our HMAC matches the standard byte for byte.

### RFC 5869, HMAC-based Extract-and-Expand Key Derivation Function (HKDF)
- **What it is.** A two-step key derivation:
  - Extract concentrates input key material into a pseudorandom key, using a salt;
  - Expand stretches that key into as many output bytes as needed.
- **What it does.** Gives you keys that are uniformly random even when the input material isn't.
- **Where.** `hkdf_extract_sha256` and `hkdf_expand_sha256` in `include/pchie/HmacSha256.h`. An empty salt becomes 32 zero bytes, as RFC 5869 section 2.2 says. Test case 1 is checked in `anchor_unit`.
- **What we use it for.** The ratcheted seal chain uses Extract to make its first chain key (salt `pchie.ratchet.v2`, or the secret salt in salted mode). It uses Extract again, with the parent head as salt, for every epoch re-root. Expand is implemented and tested, and the anchoring tool exposes it, but no seal chain calls it today.

## Transparency log and signatures

### RFC 9162, Certificate Transparency Version 2.0
- **What it is.** The CT v2 spec. We take only its Merkle tree part, section 2.1.
- **What it does.**
  - It defines an append-only log as a Merkle tree, with leaf hash SHA-256(0x00 || d) and node hash SHA-256(0x01 || left || right).
  - It defines inclusion proofs ("this record is in the tree") and consistency proofs ("this bigger tree extends that smaller one without rewriting it").
- **Where.** In `anchoring/src/pchie_anchor.h`:
  - the hashing and tree functions `leaf_hash`, `node_hash`, and `mth`;
  - proof generation in `inclusion_proof` and `consistency_proof`;
  - checking in `verify_inclusion` (s2.1.3.2) and `verify_consistency` (s2.1.4.2).

  `anchor_unit` checks every inclusion and consistency proof for trees of up to 70 leaves. It also checks the s2.1.5 proof sizes and that the s2.1.2 stack algorithm agrees with MTH.
- **What we use it for.** The `pchie_anchor` side log. It records run chain heads, issues signed tree heads, and lets a witness confirm the log never rewrote history. X.509, SCTs, and CT's TLS and HTTP APIs are not used. See [Transparency anchoring](../guide/transparency-anchoring.md).

### RFC 8554, Leighton-Micali Hash-Based Signatures (LMS / LM-OTS), as a model
- **What it is.** A stateful hash-based signature scheme: one-time Winternitz signatures under a Merkle tree of public keys.
- **What it does.** It signs with nothing but a hash function, so it needs no elliptic curves. Each one-time key must never be used twice.
- **Where.** `Hbs`, `HbsKey`, and `hbs_verify` in `anchoring/src/pchie_anchor.h`, using SHA-256 with n=32, w=4, p=67, ls=4, and h=10, which gives 1024 signatures per key. `sign_persist` in `anchoring/src/pchie_anchor.cpp` saves the key state after every signature.
- **What we use it for.** Signing the log's tree heads and the witness cosignatures. The code models RFC 8554 but is **not** claimed to interoperate with it.

### RFC 3339, Date and Time on the Internet: Timestamps
- **Where.** `now_utc()` in `anchoring/src/pchie_anchor.cpp` writes the `YYYY-MM-DDTHH:MM:SSZ` UTC form.
- **What we use it for.** The `timestamp_utc` field of a signed tree head. The field is signed as text and never parsed. Nothing in the safety checks relies on the clock.

## Stream authentication and attestation

### RFC 4082, Timed Efficient Stream Loss-Tolerant Authentication (TESLA)
- **What it is.** A broadcast authentication scheme.
  - The sender MACs each packet with a key from a one-way hash chain.
  - Each key is disclosed only after a delay.
  - A receiver buffers packets until their key arrives, then checks the key against the chain commitment and the MAC against the key.
- **What it does.** Gives you sender authentication on a one-way stream with only symmetric crypto.
- **Where.**
  - `TeslaKeyChain`, `tesla_chain_step`, and `tesla_mac_key` (`include/pchie/TeslaKeyChain.h`);
  - packets and disclosures (`include/pchie/TeslaStream.h`);
  - the receiver `TeslaWatcher` (`include/pchie/TeslaWatcher.h`);
  - session wiring `StreamAuthSession` (`include/pchie/StreamAuth.h`);
  - the `pchie_tesla_watcher` tool;
  - tests in `tests/pchie_tesla_stream_test.cpp`.
- **What we change.** The chain's own step order is the clock, so the safety test never reads a wall clock. The defaults are 8 sealed steps per interval with a disclosure lag of 2. As RFC 4082 warns, a lag of 1 is refused.
- **What we use it for.** Authenticating the live stream of sealed maze steps. It is on by default in the maze drivers and turned off with `--tesla off`. The official seals are unchanged either way. See [TESLA stream authentication](../guide/tesla-stream-authentication.md).

### RFC 9334, Remote ATtestation procedureS (RATS) Architecture
- **What it is.** Vocabulary and roles for attestation: an attester produces evidence, a verifier appraises it against reference values, and a relying party acts on the result. It is not a wire format.
- **Where.**
  - `include/pchie/AttestationVerdict.h`: `MeasurementSet`, `KnownGoodRegistry`, `SimulatedQuote`, `AttestationAppraisal`, and the three verdicts Attested, Unattested, and Rejected;
  - the reference values are in `docs/notes/known-good-registry.jsonl`.
- **What we use it for.** Recording a verdict beside a run's label without ever changing the label. The appraisal in the code checks a simulated, HMAC-signed quote and never talks to a TPM. Without a quote, the verdict is Unattested. See [Hardware attestation](../guide/hardware-attestation.md).

## Interfaces and data formats

### RFC 8259, The JavaScript Object Notation (JSON) Data Interchange Format
- **Where.** `pchie::MiniJson` (`include/pchie/MiniJson.h`), a small fail-closed parser with a nesting cap of 64 levels. It reads objects, arrays, strings with `\uXXXX` escapes, numbers, `true`, `false`, and `null`. The RL side records under `include/pchie/rl/` write JSON Lines, one object per line.
- **What we use it for.** Reading MCP replies from LithoKraft, plus the RL label, plan-memory, and hash-move records. MiniJson itself is not on the official chain (its own header says so); the scratch RL tuple logs are sealed over their raw lines.

### JSON-RPC 2.0 Specification (jsonrpc.org; not an RFC)
- **Where.** `McpStdioClient` (`include/pchie/McpStdioClient.h`): `request` sends `{"jsonrpc":"2.0","id":...}` and `notify` sends a request with no id. Lines that are not JSON objects are skipped and counted.
- **What we use it for.** The transport under MCP.

### Model Context Protocol (MCP), revision 2025-11-25 (modelcontextprotocol.io; not an RFC)
- **Where.** `LithoKraftEngineHost` (`include/pchie/LithoKraftEngineHost.h`). It sends `initialize` with `protocolVersion` "2025-11-25", then `notifications/initialized`, then `tools/call`. The stdio transport uses one message per line. The game side of the seam is described in `third_party/pch-public/engine/docs/MCP-GAME-SEAM.md`.
- **What we use it for.** Driving LithoKraft from `pchie_lk_pilot`: reading the world and moving Little Nemo in the real game.

### RFC 3629, UTF-8, a transformation format of ISO 10646
- **Where.**
  - `MiniJson` turns `\uXXXX` escapes into UTF-8 bytes.
  - `Gemma4Tokenizer::bpe` (`src/pchie_gemma4.cpp`) splits text into UTF-8 code points by their lead byte before BPE merges, with spaces mapped to U+2581 in the SentencePiece style.
  - `include/pchie/rl/UntrustedTokens.h` treats UTF-8 code units as opaque bytes.
- **What we use it for.** Tokenizing prompts for the model, and handling the game's JSON text.

### RFC 4648, The Base16, Base32, and Base64 Data Encodings (base64, section 4)
- **Where.** `b64decode` in `vision/pchie_lk_visnav.cpp`. It uses the standard alphabet and is lenient: it skips non-alphabet characters and stops at `=`.
- **What we use it for.** Decoding the screenshots that the engine sends over MCP.

### PNG (ISO/IEC 15948:2004, W3C PNG Specification), with RFC 1950 (zlib) and RFC 1951 (DEFLATE)
- **Where.** `png_decode` in `vision/pchie_lk_visnav.cpp`. It is a deliberately narrow reader for the engine's own encoder:
  - it checks the 8-byte signature and the IHDR, IDAT, and IEND chunks;
  - it accepts only 8-bit RGB with filter type 0;
  - it reads only stored DEFLATE blocks (BTYPE 00), after skipping the 2-byte zlib header.

  It does not check chunk CRCs or the zlib Adler-32, and it rejects anything else.
- **What we use it for.** Turning engine screenshots into pixels for the visual navigator.

### GGUF version 3 and the GGML block quantization layouts (ggml project documentation; not an RFC)
- **Where.**
  - A clean-room, read-only reader `GgufFile` (`include/pchie/pchie_gguf.h`, `src/pchie_gguf.cpp`), which refuses any version other than 3.
  - The Q4_K, Q6_K, Q8_0, and Q8_K block structs and deterministic integer kernels in `include/pchie/pchie_kquants_det.h`, matched by the CUDA kernels in `src/pchie_cuda.cu`.
- **What we use it for.** Loading Little Nemo's weights and running exact, bit-reproducible inference.

### IEEE 754-2019, Floating-Point Arithmetic (binary32 and binary16)
- **Where.** `include/pchie/pchie_det_math.h`:
  - bit-exact conversions `det_fp16_to_f32`, `det_bf16_to_f32`, `det_f32_from_bits`, and `det_bits_from_f32`;
  - round-half-even in `det_round_half_even`;
  - math built only from correctly rounded `+ - * /`.

  The build enforces it. `CMakeLists.txt` uses `/fp:precise` or `-ffp-contract=off -fno-fast-math` for C++, and CUDA uses `-fmad=false -prec-div=true -prec-sqrt=true -ftz=false`. bfloat16 is not an IEEE format; it is the top 16 bits of a binary32, and the code treats it that way.
- **What we use it for.** Making CPU and CUDA runs produce identical bits, so run hashes are reproducible.

## Non-cryptographic hashing and seeding

### FNV-1a, 64-bit (Fowler-Noll-Vo; described in the IETF Internet-Draft draft-eastlake-fnv, not an RFC)
- **Where.** `Fnv1a64HashChain` (`include/pchie/Fnv1a64HashChain.h`), with offset basis `FNV_OFFSET` and prime 1099511628211. The same constants appear in `pchie_tree.h` and the maze, PCH-stack, and gate headers.
- **What we use it for.** The fast hot-path chain head and the short `0x...` fingerprints printed beside SHA-256 values, such as the maze H_world `0xD968C7EE5D48C403`. It is not a security hash. Anything that has to resist tampering uses SHA-256.

### SplitMix64 (Steele, Lea and Flood, "Fast Splittable Pseudorandom Number Generators", OOPSLA 2014; not a standards-body document)
- **Where.** `NodeSeed::Splitmix64` (`include/pchie/NodeSeed.h`), with the usual constants `0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, and `0x94D049BB133111EB`. It is also used in the maze generators (`MazeGrid.h`, `ScaledPitMaze.h`, `StageBDoorFactory.h`) and in `RootNoise.h` and `CommitSelector.cpp`.
- **What we use it for.** Per-node seeds tied to the tree coordinate (search ruling 5), deterministic root noise, tie breaks, and maze generation, so results do not depend on thread order.

## Design specifications we follow that are not RFCs

These shaped the design. Only the parts listed are implemented. Details and links are in [anchoring references](../notes/anchoring-references.md).

- **Signal, The Double Ratchet Algorithm (revision 4).** We take its symmetric-key KDF chain: per-step keys from HMAC with constant inputs 0x01 and 0x02, and old keys wiped. That becomes `RatchetHashChain` (`pchie.seal.v2-ratchet`). There is no Diffie-Hellman ratchet and no encryption. It has no RFC number, so never cite one for it.
- **Sigsum witness cosigning, now C2SP tlog-witness and tlog-cosignature.** We take the witness rules: a witness keeps the last head it cosigned, checks consistency before cosigning, and never cosigns a smaller tree. This is `witness_cosign` and `check_policy` in `anchoring/src/pchie_anchor.cpp`. Exchange is files only. We don't use the signed-note format, Ed25519, or ML-DSA.
- **Signal safety numbers.** We take the shape: 60 digits in groups of 5, from two halves of 5200-round iterated SHA-256. This is `fingerprint` and `fingerprint_for` in `anchoring/src/pchie_anchor.h`.

## Mentioned in the docs but not implemented in the harness code

- **TPM 2.0 (Trusted Computing Group TPM 2.0 Library).** No harness code talks to a TPM. The one real measure, extend, and quote on PCR 23 was a separate bench check, described in [Hardware attestation](../guide/hardware-attestation.md). The string `tpm2.0-sim` in the tests is only a label on a simulated quote.
- **Ed25519 (RFC 8032) and SHA-512 (FIPS 180-4).** They exist only in the vendored engine (`third_party/pch-public/engine/src/crypto/`), where LithoKraft uses them for mod signing. The harness does not link them. Our tree heads use the hash-based signature above, and the anchoring notes say so: "There is no elliptic-curve code in our tree".

Not used anywhere in the harness: CBOR, UUIDs, X.509, TLS, AEAD ciphers, PBKDF2, and every other RFC not listed above.
