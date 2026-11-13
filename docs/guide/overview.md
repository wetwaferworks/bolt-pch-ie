# Overview

Bolt PCHIE (Bolt Personal Control Harness Inference Engine, pronounced "bolt peachy") is a robotics test bed for on-device inference, built around NVIDIA Nemotron models. PCHIE runs a small language model inside a deterministic control harness and asks it to act in a simulated world, with every committed decision sealed into a hash chain so a run can be replayed and checked.

The usual simulated world in this tree is a maze: the headless B1 grid, the Stage B puzzle-maze with doors, and, when a running game is available, LithoKraft mazes driven through a control connection. Model weights are not in the repository. The recorded measurements use a GGUF build of Little Nemo (Nemotron-3.5-Bolt-E4B, Q4_K_M).

## Safety-first

The harness, not the model, decides what the model may touch. Gates, audits, and label rules sit between the model and the world. A rejected action does not leak through: the gate fails closed. Side logs observe a finished or in-progress run and never feed it. Transparency anchoring, TESLA stream authentication, and hardware attestation are the same kind of thing: they wrap or record a run. They do not write back into an official seal, a world hash, or a committed action list.

The project keeps two kinds of validity separate:

- A run is *chain-valid* when a machine re-derives its hash chain and the head matches.
- A run is *truth-valid* only when a person has checked what it claims.

Tooling never upgrades a label. A program can report that a chain re-derives, that a TESLA stream verified, or that an attestation appraisal was Attested, Unattested, or Rejected. None of those findings turns a Scratch or Provisional record into an Official one.

## Deterministic and tamper-evident

PCHIE is deterministic by construction. The same weights, the same inputs, and the same seed are required to produce the same decisions and the same hashes, at any thread count. The join dispatcher appends in canonical step order, not in the order workers happen to finish. Agreement at 1, 2, 4, 8, and 16 workers is checked in the tests. A moved hash is treated as a finding: it is stopped and explained, never smoothed over.

The harness is tamper-evident, not tamper proof. "Tamper-evident" means that a change to a sealed record, a TESLA packet, or an anchor log is detectable when the chain, the watcher, or the verifier is run again. It does not mean a record cannot be altered, and this guide never claims that it cannot. The TESLA watcher says this in so many words: the stream layer is tamper evident, not tamper proof. The same wording applies to the firmware TPM used in the hardware-attestation spike, and to the seal chains themselves.

## Record labels

| Label | Meaning |
|---|---|
| **Official** | A frozen record produced by the official runners and accepted by the human truth seat. |
| **Provisional** | A candidate result that is reproducible, but not yet accepted. Hashes may still change. |
| **Scratch** | Exploratory work. Useful evidence, never a claim. |

Labels only move up by a person's decision. Everything added in November 2026 is Scratch or Provisional unless it belongs to a frozen Official record already in the tree. Hashes quoted in reports and in this guide are provisional when they come from that work, and are not locked.

Attestation verdicts (Attested, Unattested, Rejected) sit beside these labels and never change them. Chain validity and attestation are separate axes and are never merged into one word.

## What this guide covers

The pages that follow describe the architecture; how hashing and labels work; the default SHA-256 seal chain and the opt-in ratcheted chain; the transparency-anchoring side log and its witness policies; TESLA stream authentication; hardware attestation against a TPM 2.0 quote and a known-good registry; the reasoning layers and the loop guard; LithoKraft and LithoLattice integration; how to build and test; results recorded so far, stated honestly; limitations and the threat model; and a glossary.

The source of truth is the tree: headers under `include/pchie/`, tests under `tests/` and `anchoring/tests/`, and the notes under `docs/notes/`. If a sentence here is not backed by those, it does not belong.
