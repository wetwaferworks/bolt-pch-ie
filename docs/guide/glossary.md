# Glossary

**Actions hash.** SHA-256 of a committed action list. Used to compare walks. Not the same thing as a chain head.

**Anchor log.** A local, append-only Merkle log of finished-run records, with signed tree heads, inclusion and consistency proofs, and optional witness cosignatures. It never feeds a run and never upgrades a label.

**Attested / Unattested / Rejected.** The three hardware-attestation verdicts, in the manner of RFC 9334. They sit beside a run's label and never change it. Attested requires a valid fresh quote and a known-good registry match. Without a quote the verdict is Unattested.

**B1.** The official 16x16 headless grid maze (seed 1 in official search).

**Bench-grade attestation key.** The harness-owned PCHIE-Harness-AK used in the PCR 23 bench check. It is bound to the TPM by creation and has no EK certificate chain.

**Chain-valid.** A machine can re-derive the hash chain and the head matches. Necessary for a serious claim, not sufficient for truth.

**Component Test.** A LithoKraft world: a flat wafer-silicon plain with one named pad per Trace component. See the [component reference](../components/README.md).

**Coordinator.** The only thread that appends to the official seal chain. It appends in canonical step order, never in worker completion order.

**Disclosure lag (`d`).** In TESLA, the number of intervals to wait before disclosing an interval key. The default is 2. A lag of 1 is refused.

**Fail closed.** A bad, unknown, or disallowed input is refused. It is not ignored, defaulted through, or treated as success.

**Fingerprint.** A 60-digit, human-checkable number in 12 groups of 5, from two 5200-round iterated SHA-256 halves. v1 covers version, chain head, and run id. v2 also covers chain generation.

**H_world.** A hash of the world or maze state for a fixture.

**Hard power / soft power.** LithoKraft signal rules. A component output that drives straight into a trace face hard-powers a solid block. Lit flat trace soft-powers touching solid powerable blocks. See the [component reference](../components/signal-rules.md).

**Interval (`I`).** In TESLA, the number of sealed steps that share one interval key. The default is 8.

**Known-good registry.** An append-only list of approved measurement sets. Only the maintainer adds an entry, and Cortana cosigns it. Entries are never edited or removed.

**Label (Official, Provisional, Scratch).** The human truth label on a record. Official is frozen and accepted. Provisional is a reproducible candidate, not yet accepted. Scratch is exploratory evidence, never a claim. Labels move up only by a person's decision.

**LithoKraft.** The game. Spelled with a K. It runs on the LithoLattice engine.

**LithoLattice.** The engine. One word, both Ls capital.

**Little Nemo.** The language model used in the recorded runs (NVIDIA Nemotron 3.5 Bolt E4B, Q4_K_M GGUF). Text in, text out; the vision path feeds it rendered frames encoded to text. Weights are not in the repository.

**Loop guard v2.** On by default. The fourth entry into the same state forces a reasoning check (fresh seed, four times the budget, no 180-degree turn) and turns on cautious mode. `--loop-guard off` reproduces the old walk.

**Multiplexor.** A LithoKraft signal component (engine id `lithokraft:multiplexor`). Spelled with an o, like Dialor, Watchor, Bitor, Pushor, and Swivelor.

**Official pin.** A hash belonging to a frozen Official record, required by tests not to move.

**PCR 23.** The designated TPM 2.0 platform configuration register for the hardware-attestation bench check. PCR 0-7 and 11-14 are not extended.

**Playtest build (12 November 2026).** A packaged LithoKraft playtest. See the [component reference](../components/README.md#the-12-november-playtest-build).

**Provisional hash.** A reproducible hash that has not been locked. November 2026 measurement hashes are provisional unless they are identified as Official pins.

**Query-before-plan (L2).** An opt-in memory of prior situations. Hits are proposals only and must still pass the gate.

**Ratcheted chain.** Seal-chain generation `pchie.seal.v2-ratchet`, off by default. Derivation is in the manner of the symmetric-key chain in Signal's Double Ratchet specification. This project does not assign that construction an RFC number.

**RATS.** Remote ATtestation procedureS, RFC 9334. Roles: attester, verifier, relying party.

**Seal.** The official, append-only record of a committed step, in the hash chain.

**Side log.** A log that observes a run and never feeds it.

**Stage B.** A puzzle-maze fixture with doors that ask a Dividor question.

**Tamper-evident.** A change is detectable when the chain, watcher, or verifier is run again. Not the same as tamper proof.

**TESLA.** Timed Efficient Stream Loss-tolerant Authentication, RFC 4082, adapted here so the chain's own order is the clock. There is a MAC per step. Default schedule: 8 steps per interval, disclosure lag 2. `--tesla off` turns it off.

**Trace.** LithoKraft's red and blue signal wiring. Strengths run 0 through 64.

**Truth-valid.** A person has checked what the run claims. Tooling cannot confer this.

**Via.** A LithoKraft vertical or side-lying insulated tube that carries Trace losslessly.

**Witness policy.** A list of witness ids and a threshold for anchor-log cosigning. The defined three-way policy is the maintainer, Cortana, and Elfie, 3 of 3. The bench policy lists Elfie but does not require that role.
