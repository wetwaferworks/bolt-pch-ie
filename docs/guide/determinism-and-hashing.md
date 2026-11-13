# Determinism and hashing

PCHIE treats a hash as a measurement, not as decoration. The same weights, the same inputs, and the same seed are required to produce the same decisions and the same hashes. If a hash moves, the work stops and the move is explained. It is not averaged away, ignored, or silently replaced.

## Thread-count-independent results

Worker threads may finish in any order. That must not change the result. Only the coordinator appends to the official chain, and it appends in canonical step order, never in worker completion order. The join dispatcher folds work in ascending work-id order. Duplicate or missing ids fail closed; there is no silent partial fold.

This is checked, not merely asserted. Tests and recorded runs compare 1, 2, 4, 8, and 16 workers and require identical actions, chain heads, tree digests, and (where they exist) tuple contents. TESLA streams are included in that agreement check: 1 through 16 workers produce identical streams when steps are emitted in canonical order. Maze and reasoning-layer drivers print that results do not depend on the thread count. The default dispatcher cap is 16 workers.

Floating-point settings are part of the same promise. The core is built with precise floating point (no fused multiply-add contraction on the CPU path; the CUDA backend, when present, is compiled with fused multiply-add off and precise division and square root). Search ties break by lowest token id, so a tie is not a race.

## How a hash is formed

The default official seal chain starts from 32 zero bytes. Each append is:

```
h_0     = 32 zero bytes
h_n     = SHA-256(h_{n-1} || canonical_bytes(record_n))
```

The head is 32 bytes. A shorter FNV-1a 64-bit chain exists as a runtime option for some internal uses; exported logs and official pins use SHA-256. Canonical bytes are a defined encoding of the step record, so two semantically equal records hash the same way and a formatting change that is not in the canonical encoding cannot sneak into the head.

The opt-in ratcheted chain uses HMAC and HKDF instead of this plain hash chain. It is a different generation and is described in [Seal chains](seal-chains.md). Official runners stay on the SHA-256 generation unless a caller asks for the ratchet by name.

Other hashes that appear in logs and reports measure different things, and they are not interchangeable:

- An **actions hash** is SHA-256 of the committed action list.
- **H_world** is a hash of the world or maze state for that fixture.
- A **chain head** is the seal-chain head after the recorded appends.
- A **side-log hash** covers a side log, not the official chain.
- A **tree digest** summarizes search-tree content used in some comparisons.

Re-deriving a chain and comparing it to a stored head is how a run is shown to be chain-valid. Comparing an actions hash or H_world across two runs is how walk identity is shown. A person still has to decide whether the walk is a truthful claim. That second check is not a hash.

## Labels: Official, Scratch, and Provisional

| Label | Meaning |
|---|---|
| **Official** | A frozen record produced by the official runners and accepted by the human truth seat. Official pins in the tests are expected to stay put. |
| **Provisional** | A reproducible candidate that has not been accepted. The hash may still change. |
| **Scratch** | Exploratory work. Useful as evidence, never as a claim. |

Labels move up only by a person's decision. Tooling never upgrades a label. Everything added in November 2026 is Scratch or Provisional unless it already belongs to a frozen Official record. Hashes quoted from that work, including hashes in this guide, are provisional and are not locked.

A few Official pins that the tests require, and that later work was checked not to move:

| What | SHA-256 |
|---|---|
| Official B1 search actions | `0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e` |
| Official B1 search chain head | `de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533` |
| Official B1 H_world | `457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5` |

The official B1 search walk does not loop, so turning the loop guard on does not change those values. `--loop-guard off` reproduces walks that the guard would otherwise change. The ratcheted-chain tests also pin a v1 SHA-256 golden (`f4c122234f3fd97c5c495a028af4c62c593e0b6865c239c155e2643d2c5f2445`) and a default recomputable ratchet head (`a781fda17ae8a2228d5f03431aa78b933837c02ee206e7ea571fce54a1601c29`) so that adding the ratchet, TESLA, or attestation cannot quietly move those goldens.

## Hashes and other verdicts stay on separate axes

An attestation verdict (Attested, Unattested, or Rejected) sits beside the label and never changes it. A TESLA stream that verifies, or fails, does not change a seal or a label either. Chain validity, stream authenticity, attestation, and the human truth label are four different questions. The project does not collapse them into one word.

When the loop guard became the default, some walks that used to loop did change, and those new hashes are provisional. The official records, goldens, and reports were not edited to follow them. The old values remain valid and are reproduced with the guard off. That split is deliberate: a default that changes a looping walk is a new measurement, not a rewrite of the official record.
