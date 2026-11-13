# PCHIE researcher guide

This guide describes the Bolt PCHIE harness as it exists in this tree: what it is, how its parts fit together, and what the code, tests, and notes actually support. It is written for college-level researchers. Nothing here invents a feature, a hash, or a result that is not in the source, the tests, or the working notes.

Hashes quoted in this guide are provisional unless they belong to a frozen Official record. Tooling never upgrades a label. The system is deterministic and tamper-evident; this guide never calls it tamper proof.

## Pages

- [Quick start: running the maze](quick-start-maze.md) - From a release download to a sealed maze run: prerequisites, the model file, demo runs, and how to read the output.
- [Playing LithoKraft](playing-lithokraft.md) - The playtest build that ships with the release: launch, controls, inventory, components, the Component Test world, and how the harness drives it.
- [Getting started on Windows](getting-started-windows.md) — Prerequisites, a Release build, the quick tests, a first harness run, and common errors.
- [Getting started on Linux](getting-started-linux.md) — The same walk-through for a single-configuration Linux build.
- [Checksums](checksums.md) — How to verify `SHA256SUMS` with `Get-FileHash`, `certutil`, and `sha256sum`.
- [Overview](overview.md) — What PCHIE is, the safety-first framing, and how labels work.
- [Architecture](architecture.md) — Components, data flow, and how a committed step is sealed and observed.
- [Diagrams](diagrams.md) - Component, deployment, decision-step sequence, key-interface class, and run-lifecycle state diagrams.
- [Determinism and hashing](determinism-and-hashing.md) — Thread-count-independent results, provisional hashes, and the Official / Scratch / Provisional labels.
- [Seal chains](seal-chains.md) — The default SHA-256 chain, the opt-in ratcheted chain, and genesis options.
- [Transparency anchoring](transparency-anchoring.md) — The side log, fingerprints, and witness policies, including Elfie's defined role.
- [TESLA stream authentication](tesla-stream-authentication.md) — The adapted TESLA stream, the watcher, and `--tesla off`.
- [Hardware attestation](hardware-attestation.md) — RATS roles, TPM 2.0, PCR 23, the known-good registry, and honest caveats, with `--attest off`.
- [Reasoning layers and the loop guard](reasoning-layers.md) — Opt-in layers, loop guard v2, and what official runners do and do not enable.
- [LithoKraft and LithoLattice](lithokraft-integration.md) — The vendored public snapshot, the MCP control connection, and the pilot.
- [LithoKraft component reference](../components/README.md) — the 12 November playtest build, signal rules, inventory, the Component Test world, and one spec sheet per component.
- [Build and test](build-and-test.md) — CMake, the quick CTest set, and the long suites.
- [Results so far](results.md) — Honest findings from the recorded runs, including where L1 did not separate from null.
- [Limitations and threat model](limitations-and-threat-model.md) — What the harness claims, what it does not, and where the edges are.
- [Glossary](glossary.md) — Terms used throughout the guide.

If you want to build and run first, start with the getting-started pages. Working notes, measurement reports, and design specs live under `docs/notes/`, `docs/reports/`, and `docs/specs/`.
