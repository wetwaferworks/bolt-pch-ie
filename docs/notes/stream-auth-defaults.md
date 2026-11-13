# TESLA and attestation on by default (2026-11-13)

Since 13 November 2026, TESLA stream authentication and hardware attestation are ON by default, with flags to turn them off while testing.

## Flags

- --tesla on|off (default on)
- --attest on|off (default on)

These sit beside --loop-guard on|off on the Stage B, RL, and LithoKraft pilot CLIs.

## Neutrality

TESLA is a side stream. The official IHashChain is wrapped so every official append is MAC'd *after* the inner append succeeds. Stage B has no official chain; TESLA is taken over the committed actions after the walk, also as a side stream. Nothing in this layer is written back into a seal, h_world, the actions list, or a side log.

If a TESLA emit fails, the session stops and records an error. It does not throw into the maze and it does not unwind an official append.

## Attestation in-process

When attestation is on, a verdict is always computed and recorded. The in-process path never talks to the TPM. Without a quote the verdict is Unattested (reason: absent platform). It never fakes Attested. --attest off skips the verdict entirely.

The real measure / extend / quote round trip is a separate bench check on PCR 23 only (see the hardware attestation guide). Unit tests never extend a PCR.

## Unit-test seeds

TESLA's secret seed in the maze drivers is derived from the maze seed (SHA-256(pchie.tesla.v1.derived-seed || u64be(seed))) so unit runs stay deterministic. A real run can install a CSPRNG seed via StreamAuthSession::set_secret_seed. The derived seed is a testing convenience, not a production secret.

## First TPM verdicts

Attested requires a valid fresh quote *and* every measurement in the known-good registry. The registry is append-only and stays empty until the maintainer approves an entry, so a first bench verdict of Rejected / unknown-measurement is the honest result when the quote itself is fine.
