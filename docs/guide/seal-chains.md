# Seal chains

A seal chain is the official, append-only record of committed steps. PCHIE has two generations. The default is a SHA-256 hash chain. A ratcheted chain sits beside it, off unless a caller asks for it by name. Existing call sites that build the SHA-256 chain directly are untouched, so adding the ratchet cannot move an existing chain or run hash by itself.

## Generation v1: SHA-256

Generation id: `pchie.seal.v1-sha256`. This is the default (`ChainGeneration::Sha256V1`).

```
h_0 = 32 zero bytes
h_n = SHA-256(h_{n-1} || canonical_bytes(record_n))
```

`restore_head` can snapshot or roll back a head and a length without replaying. On this generation, a later resume or fork continues from the restored head as a straight line; there is no epoch re-root. That is existing behavior and is left alone, so every existing v1 run hash stays reproducible.

## Generation v2: ratcheted chain

Generation id: `pchie.seal.v2-ratchet`. It is a new generation, not a re-seal of a v1 chain, and nothing in an existing run uses it unless a caller asks for it through `make_hash_chain`. The default in that switch remains v1.

Derivation is in the manner of the symmetric-key chain in Signal's Double Ratchet specification. There is no RFC number for that construction, and this project does not assign one.

The step is one-way. Each sealing key and each chain key is derived with HMAC-SHA-256 and then wiped. A later chain state cannot be run backward to an earlier one: going back would mean inverting SHA-256.

Let `lp(x)` be a length-prefixed string (`u32be(len(x)) || x`). Genesis and stepping are:

```
G      = "pchie.ratchet.v2.genesis" || lp(generation) || weights_sha256 || lp(run_id) || lp(harness_version)
ck_0   = HKDF-Extract(salt = "pchie.ratchet.v2", ikm = G)
head_0 = HMAC(ck_0, 0x00 || "genesis")

At step n:
  mk_n   = HMAC(ck_{n-1}, 0x01 || "seal")
  seal_n = HMAC(mk_n, "pchie.ratchet.v2.step" || head_{n-1} || u64be(n) || canonical_bytes(record_n))
  ck_n   = HMAC(ck_{n-1}, 0x02 || seal_n)
  head_n = seal_n
```

After each step, `ck_{n-1}` and `mk_n` are wiped. Only the coordinator appends, in canonical step order.

## Genesis options

**Recomputable genesis is the default and the preferred mode.** The genesis inputs (weights hash, run id, harness version, and the generation id) are public, so anyone who holds them and the step records can recompute the whole chain. Stopping a private end-to-end recompute of a published record is the anchor log's job, not the chain's.

**An optional secret genesis salt** is supported and off by default. The generation id becomes `pchie.seal.v2-ratchet-salted`. The initial chain key is:

```
ck_0 = HKDF-Extract(salt = secret, ikm = "pchie.ratchet.v2.secret-genesis" || G)
```

The salt must be at least 16 random bytes. It is never written to a chain file or to the log. Only a holder of the salt can re-derive the chain. A public salt commitment (`SHA-256("pchie.ratchet.v2.salt-commitment" || salt)`) lets a holder check that they have the right salt. The anchor tool accepts a salt file on emit and verify, and stores only the commitment.

The tests pin the default recomputable ratchet head:

`a781fda17ae8a2228d5f03431aa78b933837c02ee206e7ea571fce54a1601c29`

That pin was taken before the salt mode was added, and is required not to move.

## Epochs, resume, and fork

`restore_head` on a ratcheted chain detaches it. The next `append` is refused until `reroot` opens a new epoch, so a resumed or forked line never verifies as the same straight line.

A re-root is:

```
C    = SHA-256("pchie.ratchet.v2.epoch" || u32be(epoch) || parent_head || u64be(parent_length)
               || lp(event) || lp(epoch_run_id) || detail_sha256)
ck   = HKDF-Extract(salt = parent_head, ikm = C)
head = HMAC(ck, 0x03 || C)
```

The harness calls this after `restore_head` at the real resume and fork points it already has (the B1 freeze retry, and DualPool rollback). The helper does nothing when the chain is not a ratcheted chain, so v1 chains and every existing run hash are untouched.

## Interface

Both generations implement `IHashChain`: `reset`, `append`, `head`, `length`, and `restore_head`. Side wrappers such as the TESLA session return the official inner chain through `official_chain()`, so an epoch re-root still sees a ratcheted chain when one is in use. The TESLA key chain is a different chain from the seal ratchet. The two are never derived from each other.
