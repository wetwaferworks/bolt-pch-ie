# Bolt PCHIE transparency anchoring

This is a side log. It reads finished-run artifacts and never feeds a run. Anchoring never upgrades a label. The log, keys, and witness state it creates stay on the machine that runs it and are gitignored; the tool never publishes or sends anything on its own.
Background reading is in `docs/notes/anchoring-references.md` and in the [Transparency anchoring](../docs/guide/transparency-anchoring.md) guide.

- `policy/` holds the witness policies (see below).
- `src/` holds `pchie_anchor.h`, `pchie_anchor.cpp`, and `pchie_anchor_fs.h`/`.cpp` (crash-safe writes and the signing lock): anchor records, the Merkle log (RFC 9162 hashing), signed tree heads, proofs, witness cosigning, the fingerprint, and the verifier. It uses the core `pchie::Sha256` and `include/pchie/HmacSha256.h`. Ratcheted chain files (generation `pchie.seal.v2-ratchet`, from `include/pchie/RatchetHashChain.h`) are verified here too.
- `tests/` holds `anchor_unit.cpp` (87 checks) and `anchor_verifier_test.cpp` (log, witness, policy, and verifier checks end to end with throwaway keys). Both run under CTest as `pchie_anchor_unit` and `pchie_anchor_verifier`.
- Signing takes an exclusive lock on the key and reserves the next one-time leaf durably before a signature is released, so a leaf is never used twice. An exhausted key refuses to sign. See [SECURITY.md](../SECURITY.md).
- `log/` is the anchor log (`anchors.log`, `sth/`, `cosigs/`, `witness_registry.txt`, `keys/`). It is gitignored. `keys/signing.key` is secret; keep it on the machine that holds the log.
- `witness/` holds local witness keys (secret, gitignored).
- `witness_out/` holds packets for a witness: signed tree heads and consistency proofs only, with no records, chains, run ids, or weights. Nothing is sent from here automatically.
- `witness_in/` receives returned cosignatures. Check one with `pchie_anchor intake log <witness-id> witness_in/<file>`.

Build: the root CMake builds `pchie_anchor` and both tests. `build.bat` still builds the tool on its own.
Emit at run close: `pchie_anchor emit log --run-id ID --label Scratch --weights SHA --chain-file CHAIN --expect-head HEAD --witness-dir witness/cortana-producer --witness-id cortana-producer`. A ratcheted chain file is recognized automatically, and its anchor records `chain_generation` and `epoch_id`.
Verify: `pchie_anchor verify log --anchor-index i --chain-file F --fingerprint "....." --witness-held STH --witness-pub HEX --witness-id cortana-producer`. Each check prints PASS or FAIL.
Fingerprint: 60 digits in 12 groups of 5, from two 5200-round iterated SHA-256 halves. v1 records use rule v1 over (version, chain head, run id). Ratcheted records use rule v2, which also covers `chain_generation`. Old fingerprints stay valid.
Secret-salt genesis (supported, off by default; recomputable chains are preferred): pass `--genesis-salt-file S` to emit and verify. The generation is `pchie.seal.v2-ratchet-salted`, and only a salt commitment is stored.
Witness policy: `policy/three-way.txt` is the defined policy (`jd-local`, `cortana-producer`, and `elfie`; 3 of 3). `policy/bench.txt` is a single-machine setup (`jd-local` and `cortana-producer` required; `elfie` listed, not required). Check with `verify ... --policy policy/bench.txt`. Cosign as a local witness with `pchie_anchor cosign-local log <witness_dir> jd-local`, which is file-only. A policy can list any number of witnesses with a threshold.
Epochs: the harness re-roots a ratcheted chain at the B1 freeze retry (resume) and DualPool rollback (fork). It never re-roots a v1 chain.

## Verifying a witness packet

A packet holds `log_pub` (48 bytes hex: a 16-byte key id I and a 32-byte key root), a `new_sth` line, an optional `old_sth` (the last head that witness cosigned), and `consistency_proof` (hex nodes).

1. The STH line minus its `|sig=...` suffix is the signed text. The signature is PCHIE-HBS (SHA-256 hash-based, modeled on RFC 8554 LMS, n=32 w=4 h=10).
2. If `old_sth` is present, check RFC 9162 section 2.1.4.2 consistency from (old size, old root) to (new size, new root) with the listed proof. Refuse if the new size is smaller than the size you last cosigned.
3. Keep your own copy of every head you accept. That copy is what catches a rewrite later.
4. To cosign, sign `pchie.cosig.v1|witness=<witness-id>|log_id=<id>|tree_size=<n>|root=<hex>` and return `<text>|sig=<hex>` to the log holder. The log holder registers the witness public key first. The reference tool does steps 1 to 4: `pchie_anchor witness-keygen <dir>`, then `pchie_anchor witness-cosign <dir> <witness-id> <packet> <out>`. A witness pins the log's public key. Pin it ahead of time with `pchie_anchor witness-trust <dir> <log_pub_hex>`, or it is pinned on first contact. Any later packet under a different key is refused (`LOG KEY CHANGED`). Witness ids are limited to `[A-Za-z0-9._-]`, 1 to 64 characters.

A first packet has no older head, so it carries no proof nodes. Later packets carry a proof from the head that witness last cosigned.
