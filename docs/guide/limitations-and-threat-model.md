# Limitations and threat model

PCHIE is a safety-first research harness. It is deterministic and tamper-evident. It is not tamper proof, and this guide does not claim that it is. A change to a sealed record, a TESLA packet, an anchor log, or a quoted PCR can be detected when the matching verifier is run again. Detection is not the same as prevention, and a determined adversary who can also rewrite verifiers, keys, or the registry is outside what the current machinery claims to stop.

## What the harness claims

- Same weights, inputs, and seed produce the same decisions and the same hashes, at any thread count that the dispatcher supports (checked at 1, 2, 4, 8, and 16 workers).
- Only the coordinator appends, in canonical step order. Worker completion order cannot change a seal.
- A moved Official pin is a finding, not something to smooth over.
- Gates fail closed. Side logs, TESLA, attestation, and anchoring observe or wrap a run; they do not write back into an official seal, a world hash, or a committed action list, and they do not upgrade a label.
- A later ratcheted-chain state cannot be run backward to an earlier one without inverting SHA-256.
- A TESLA watcher flags late packets, bad MACs, wrong keys, early disclosures, gaps, and bad interval indexes. Late cannot be proven honest. A wrong-key disclosure fails authenticity, full stop.
- Attested requires a valid fresh quote and a known-good registry match. Without a quote the verdict is Unattested, never a faked Attested.
- Untrusted inputs are bounded. The TESLA watcher caps indices, stream length and line length, and the GGUF reader and model loader refuse a malformed file before reading past a buffer.
- Anchoring is a local Merkle log with signed tree heads and witness cosignatures. It does not publish itself. Outgoing witness packets carry tree heads and consistency proofs only, not records, chains, run ids, or weights.

## What it does not claim

- **Tamper proofing.** Delayed key disclosure, hash chains, and TPM quotes are tamper-evident mechanisms. They do not make a log unchangeable.
- **A manufacturer-backed attestation key.** The recorded bench check uses a harness-owned attestation key with no EK certificate chain. It is bench-grade. The quote scheme on that key is RSASSA-SHA1, not SHA-256. Firmware TPM (Intel PTT, in the spike) is less physically resistant than a discrete module.
- **That Attested is required for Official.** Labels and verdicts stay on separate axes.
- **That the default seal chain is the ratchet.** Official runners stay on SHA-256 generation v1 unless a caller asks for v2. Secret-salt genesis is optional and off by default.
- **That a `--tesla-seed dev-insecure` stream proves anything.** That seed is derived from the public maze seed so unit runs stay deterministic; anyone can rebuild its keys. Normal runs use a fresh OS-random seed and refuse to start without one.
- **Model skill the measurements do not show.** At budget 16, L1 did not separate from null. Query-before-plan engaged and did not change the walk. Stage B still stalls honestly at door 1. Those are current limits of the recorded runs, not issues to be described as solved.
- **That every long suite has been re-checked** with the loop guard on. The thirteen long suites listed in [Build and test](build-and-test.md) were excluded from the quick set used when the guard became the default.
- **That the public Personal Control Harness is the 12 November playtest build.** The public tree at https://github.com/JaydeeWetwork/PersonalControlHarness, as of 23 October 2026, is earlier than the playtest described in the component reference. See [LithoKraft and LithoLattice](lithokraft-integration.md).

## Threat sketches that the tests already cover

The TESLA watcher and the anchor verifier are written against a short list of integrity failures, not against a complete attacker.

- **Rewrite, rewind, or recompute of an anchored chain**, including a key-holder rewrite meant to be caught by a witness, and honest-pass fingerprint read-back, are exercised by `pchie_anchor_verifier`.
- **Backward fabrication** of a ratcheted chain (no state or key in the file or log) and **epoch confusion** (a resumed or forked line verifying as the same straight line) are exercised by the ratchet and epoch tests and by the anchor verifier.
- **Generation mixing**: a v2 fingerprint covers `chain_generation` so a record cannot quietly change generation and keep the same fingerprint. v1 fingerprints stay on the v1 rule.
- **Replay of a quote**: a nonce mismatch is Rejected.
- **Stream forgery after disclosure**: a step that arrives after its interval key is public is flagged late and is not released as verified.
- **Missing or reordered steps**: gaps are flagged, never smoothed over. Steps must be emitted in canonical order.
- **Duplicate or flooding streams**: a repeated step index is flagged as a duplicate, and out-of-range indices or oversized streams stop the watch (`pchie_tesla_stream`).
- **A forgeable TESLA seed**: weak, predictable and missing seeds are refused, and TESLA stops rather than fall back (`pchie_tesla_stream`).
- **Log-key substitution**: a packet signed under a key other than the one a witness pinned is refused with `LOG KEY CHANGED` (`pchie_anchor_verifier`).
- **One-time leaf reuse**: concurrent signers never share a leaf, and an exhausted key refuses (`pchie_anchor_verifier`).
- **Malformed models**: bad GGUF headers, shapes, offsets and token ids are refused at load (`pchie_gguf_validate`, `pchie_gemma4_load_validate`).

## Threats that remain open or out of scope

- The TESLA, one-time signature, and witness cosigning code is in-house, research-grade cryptography that no one has independently reviewed. Our first security review, by Gizmo (ChatGPT Luna 6), led to the fixes listed in [SECURITY.md](../../SECURITY.md), but that is not a full audit.
- A witness that pins the log key on first contact trusts whatever key it sees first. Pinning the key ahead of time with `pchie_anchor witness-trust` closes that gap.
- Wiping secrets from memory is best effort. The compiler, the operating system, and swap can still hold copies.
- An adversary who can replace the watcher, the verifier binary, or the known-good registry, or who holds a signing key that was supposed to stay in custody, is not stopped by the log format alone. Key custody is a human decision.
- TESLA does not, by itself, stop an adversary who can both alter packets and disclose keys early; early disclosure is flagged, but the layer is still only tamper evident.
- In-process attestation uses a simulated HMAC quote and never talks to the TPM. A unit-test Attested verdict is not a hardware Attested verdict.
- Anchoring does not touch a network. A witness who never receives a packet cannot cosign it. The bench policy lists Elfie but does not require that role, so a bench log can verify without it.
- The official search still uses a distance oracle where official runners always have. The model-driven path is opt-in and, so far, has not beaten a null control at the recorded budget. Treating an L1 scratch walk as a claim would be a misuse of the labels.
- Some earlier measurement reports note a Stage A first-person Vulkan frame 0 that alternates across runs and is not golden-eligible until that frame is fixed or excluded. That limitation is in the historical reports; it is not re-measured in the 13 November notes.

## Human truth stays human

A chain-valid, stream-authentic, Attested run can still be wrong about the world. A person has to decide whether the walk is a truthful claim, and only a person can move a label up. Tooling that quietly turned Scratch into Official, or Unattested into Attested, would be a break of the design, not a feature.
