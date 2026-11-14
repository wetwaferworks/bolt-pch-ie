# Hardware attestation

PCHIE can record a hardware attestation verdict beside a run label, in the manner of RFC 9334 (Remote ATtestation procedureS, RATS). The verdict is one of Attested, Unattested, or Rejected. It sits beside the Official / Scratch / Provisional label and never changes that label. Chain validity and attestation are separate axes and are never merged into one word.

Attestation is on by default in the maze drivers. `--attest off` turns it off for testing; when it is off, no verdict is recorded. `--attest on` is accepted and is the default.

## RATS roles

RFC 9334 is terminology and topology, not a wire format. The roles, as this project uses them, are:

- **Attester** — the runner that produces a quote and a measurement set.
- **Verifier** — the party that appraises the quote against a fresh nonce, the claimed measurements, and the known-good registry. Appraisal of a quote is a distinct step from TESLA stream watching. For bench work the verifier and the relying party may live on the same machine. The roles do not change if the verifier later moves.
- **Relying party** — the person or process that consumes the verdict.

Evidence is the quote, the measurement set, and the information needed to recompute the PCR value. Freshness is a nonce issued for the run, not a wall clock. A replayed quote (a nonce mismatch) is Rejected.

## Two paths, kept distinct

**In-process appraisal** never talks to the TPM. It appraises a simulated quote, signed with a test attestation key by HMAC, against a known-good registry held in memory. Without a quote, the verdict is Unattested, with reason "absent platform". It never fakes Attested. Unit tests never extend a PCR.

**The hardware bench check** is a separate measure, extend, and quote against a real TPM 2.0. The designated PCR is PCR 23. The check does not clear the TPM, take ownership, or change owner auth, and it does not touch BitLocker or the platform's existing Windows attestation keys. PCR 0 through 7, and PCR 11 through 14, are not extended. PCR 23 is an application register, so it does not overlap the registers that platform features such as BitLocker usually depend on.

A firmware TPM (Intel PTT, in the recorded bench check) is treated as tamper evident, never tamper proof. Firmware TPMs are less physically resistant than a discrete module. That is a property of the hardware, not a finding that the quote failed.

## Measurement set and known-good registry

A measurement set is a structured hash of:

- the harness binary SHA-256
- the harness version string
- the model-weights SHA-256
- a config SHA-256
- the TESLA commitment (32 bytes)

The measurement-set hash is SHA-256 of a tagged encoding of those fields (the same layout the in-process code uses).

The known-good registry is append-only. Only the maintainer adds an entry. Cortana cosigns. Existing entries are never edited or removed. Attested requires a valid fresh quote *and* a match in the registry. Without a quote the verdict is Unattested, even if the measurement is already in the list. A valid quote for a measurement that is not in the registry is Rejected. The match covers every field, the TESLA commitment included. Since each run now draws a fresh TESLA secret (see [SECURITY.md](../../SECURITY.md)), an entry matches only the run whose commitment it records.

The first registry entry was added on 13 November 2026 at 9:33 AM ET (America/New_York), after an explicit approval to add that build's measurement set. The measurement-set hash is `832819c2360b587b4f12b2bb72a6a3e9e2f3f2f4cc4de776766f474b1fb40820`. The entry hash is `2ae9e5fb28c3cf4fad0aed8a6f69edb2db99c9f3d2df234923a2fa89b3aea315`. The public record is `docs/notes/known-good-registry.jsonl`, with a short note in `docs/notes/known-good-registry.md`.

## The PCR 23 bench check

On 13 November 2026 a one-time bench check measured, extended, and quoted a real firmware TPM 2.0 on the development workstation, using PCR 23 only. It used an attestation key created only for the harness (PCHIE-Harness-AK). It did not open or change the platform's Windows attestation keys, did not clear the TPM, and did not take ownership. PCR 0 through 7 and 11 through 14 were not extended.

The quote verified under PCHIE-Harness-AK, its nonce matched the watcher nonce, the quoted digest matched the value expected from the measurement set, and the measurement set was in the known-good registry. The bench appraisal was **Attested**.

The check scripts, the raw quote, the register values, the nonce, and the key's public blob are not part of this repository. The approved measurement set is the registry entry above. A later binary is a different measurement and is not that registry entry.

## Honest caveats

These caveats do not change the bench verdict, and they should travel with it:

1. **This is a self-created attestation key. It has no EK certificate chain.** It is bench-grade: the key is bound to this TPM by creation, not by a manufacturer chain.
2. **Windows PCP created the identity key with RSASSA-SHA1, not SHA-256.** The PCR bank remains SHA-256; the quote signature and `pcrDigest` use SHA-1.

Attested is not required for an Official label. An Official run may be Unattested until a person locks a later requirement. A Rejected run keeps the label it was given; it is not quietly moved to Scratch.
