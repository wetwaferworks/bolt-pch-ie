# TESLA stream authentication

PCHIE authenticates a live stream of sealed steps in the manner of RFC 4082 (TESLA), adapted so the chain's own order is the clock. There is no wall clock in the safety test. The TESLA interval-key chain is a separate chain from the seal ratchet; one is never derived from the other.

The layer is additive. Maze drivers turn TESLA on by default. `--tesla off` turns it off for testing. Official seals are unchanged. If a TESLA emit fails, the session stops and records an error; it does not throw into the maze and it does not unwind an official append.

## Schedule

The default schedule, used by the code and by the recorded bench check, is:

- `I = 8` sealed steps per interval
- disclosure lag `d = 2` intervals

RFC 4082 notes that a lag of 1 does not work; the code refuses `d < 2`. The last interval may be short. After the last step, the sender discloses the remaining keys so the end of the stream can be verified.

## Interval keys and MACs

For `n` interval keys `K_0 .. K_{n-1}`:

```
K_{n-1}  is the secret seed (or is derived from it; only the first 32 bytes are used)
K_i      = F(K_{i+1}),  F(k) = SHA-256("pchie.tesla.v1.chain" || k)
C        = F(K_0)       public commitment; K_0 stays secret until it is disclosed
```

A disclosed `K_i` checks as `F^{i+1}(K_i) == C`. A later key can be hashed back to an earlier one: `K_j = F^{i-j}(K_i)` for `j < i`.

The MAC key is not the chain key itself:

```
K'_i = HMAC-SHA-256(K_i, "pchie.tesla.v1.mac")
```

There is a **MAC per step**. The MAC is taken over a canonical encoding of the step packet except the MAC field: a tag `pchie.tesla.v1.step`, the step index, the interval index, the seal, and a payload hash. Interval `i` is `floor(step_index / I)`. The interval key for that interval is disclosed only after lag `d`.

A public commitment id, suitable for an anchor record, is a hash of the commitment, not of the secret seed.

The stream is a text format with no wall-clock fields: a header that names `pchie.tesla.v1.stream`, the commitment, `I`, and `d`, then `step|...` and `disclose|...` lines in canonical form. A non-canonical line is rejected.

## Seeds

In maze drivers, the secret seed used for unit runs is derived from the maze seed:

```
SHA-256("pchie.tesla.v1.derived-seed" || u64be(maze_seed))
```

That derived seed is a testing convenience so unit runs stay deterministic. It is not a production secret. A real run can install a CSPRNG seed through the stream session. Interval-key seeds are secrets and are not committed to the repository.

## The watcher

`pchie_tesla_watcher` is a separate program from the runner. It reads a stream file and writes a verdict. It never consults a wall clock and never talks to a TPM. Exit status 0 means the stream is authentic, every step from 0 through the last is released, and there are no flags.

Arrival order is the clock: a step packet is safe only if it arrived before its interval key was disclosed, in canonical stream order. The watcher treats a late packet as forged until proven otherwise, and late cannot be proven. Flags are:

| Flag | Meaning |
|---|---|
| late | The step packet arrived after its interval key was disclosed. |
| bad-mac | The MAC did not verify. |
| wrong-key | A disclosure does not hash to the commitment. Authenticity fails, full stop. |
| early-disclosure | A key arrived before its interval was due. |
| gap | A step index is missing from the released stream. Missing steps are gaps, never smoothed over. |
| bad-interval | The step's interval index is not `floor(step / I)`. |

A bad MAC flags the stream at that step and no later step is released as verified. This layer is tamper evident, not tamper proof: a change is detectable, and a determined adversary who can both alter the stream and disclose keys early is outside what delayed disclosure can catch on its own.

The CTest `pchie_tesla_stream` drives the watcher as a separate process. That test never talks to a TPM.

## Turning it off

```
--tesla off
```

`--tesla on` is accepted and is the default. The flag sits beside `--attest` and `--loop-guard` on the Stage B, reasoning-layer, and LithoKraft pilot command lines. With TESLA off, no stream is emitted and the official chain is unaffected.
