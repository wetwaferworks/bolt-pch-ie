# Peachy (PMCP)

Peachy is a standalone C11 library, `peachy.dll` on Windows and `libpeachy.so` on Linux. It serves **PMCP**, the
PCHIE MCP interface. An outside model connects to PMCP and may only **propose**. Peachy checks every proposal, writes
it to a hash-chained audit log, and decides what happens. In this release, nothing ever runs: sessions are in
`record` or `advisory` mode only.

Peachy is also an MCP *client* of the game's own MCP server. It reads the player's state and nearby blocks through
`player_get_state` and `world_get_block`, and it sends `player_release` whenever it enters the safe state.

Interface status: **alpha** (`PEACHY_ABI_MAJOR 0`). The interface may change between releases, with notes.

## Shape

    MCP client (Grok, Gemini, Muse, ...)
        |  stdio, one JSON-RPC message per line
    peachy_host  ->  peachy.dll  --(stdio or callbacks)-->  game MCP server
                        |
                        +-- planner: pchie.dll directly, or mjolnir.dll which loads pchie.dll

- **One export**: `peachy_get_api(abi_major, &api)` returns a versioned table of functions. Every struct starts with
  `struct_size`, which must be at least the known size and at most 64 KiB. Handles are opaque. Errors come back as
  status codes. Nothing is allocated on one side and freed on the other.
- **No environment variables, no globals, no locks.** The host passes every path, limit and clock. Pump mode runs no
  threads. In own-thread mode, the host talks to the worker only through two single-producer, single-consumer rings
  built on atomics.
- **Hard ceilings.** Each session takes one memory block when it opens (8 MiB by default) and never grows past it.
  Every message, tool call, game round trip and planning call has a deadline.
- **One safe state.** Anything unexpected ends the same way: nothing moves, `player_release` goes to the game, the
  reason is logged, and the session is held. Only the person at the host console can resume. Resume first re-reads
  the audit log from disk and requires it to match the session record for record and hash for hash, refuses a
  client that was cut off, and re-checks the game. It does not re-read the allowlist.

## Tools

| Tool | Granted | What it does |
|---|---|---|
| `peachy_info` | always | Version, interface version, mode, and whether a planner is loaded. |
| `status` | always | Game link, hold state and why, rate headroom, action budget left, audit length and head. |
| `hold` | always | Go to the safe state. Only the console can resume. |
| `audit_head` | grant | Audit log length and head hash, so a client can later prove what it was told. |
| `observe` | grant | `{radius: 0..max}` (max 3 by default, 4 at most). Position, facing, the blocks below, at the feet and at the head around the player, a text map, and the legal moves. Marked `untrusted`. |
| `plan` | grant | Asks the planner for the next step on the latest observation. Planning only. |
| `propose` | grant | `{actions: [1..8 of {kind: move, dir, steps 1..4} \| {kind: look, turn} \| {kind: action, which: use}]}` |
| `proposal_status` | grant | `{id}`: rejected (with the check and the reason), recorded, advisory, or expired. |

`tools/list` shows only the tools a client can call. Every input schema is closed (`additionalProperties: false`),
and unknown fields are refused, not ignored.

**Propose checks, in order:**
1. Client, token and grant.
2. Exact schema.
3. Rate limit and the session's action budget.
4. Legal moves on a fresh observation (no walls, no cell without a floor, nothing outside the observed area).
5. The planner gate. When a planner is loaded, its own choice is recorded next to the proposal.
6. Peachy decides: `record` (nothing runs) or `advisory` (reports a `would_run`; nothing runs).

`actuate` is refused everywhere: at session open, at the console, and in the allowlist.

**Outside text never reaches the planner or the log.** The planner gets Peachy's own observation text, built only
from block ids. The log holds hashes of raw messages plus fixed Peachy text and canonical actions. Tests send
markers through every field to check this.

## Protocol

- MCP protocol versions accepted: `2025-11-25`, `2025-06-18`, `2025-03-26`.
  - An unknown version gets `-32022`, with the list of supported versions, the same as the game.
  - Anything before `initialize` gets `-32600`, except `ping`.
- Error replies:
  - Malformed params: `-32602`.
  - Parse error: `-32700`.
  - Batches: `-32600`.
- A tool that fails returns a result with `isError: true`.
- After `bad_message_limit` malformed messages, the client is cut off and the session is held.
- Game link:
  - Handshake era `2025-11-25`, clientInfo `peachy`.
  - Game replies that say the client is rate limited are retried briefly.
  - Replies naming `world_read` or `player_control` mean the game has not granted access yet. The session is not held.
  - Any other game failure holds the session.

## Limits

| Limit | Default | Hard maximum |
|---|---|---|
| Message size | 1 MiB | 16 MiB |
| JSON depth | 32 | 64 |
| Keys per object | 256 | 4096 |
| Items per array | 4096 | 65536 |
| String bytes (decoded) | 64 KiB | 1 MiB |
| Values per message | 16384 | 1048576 |
| Session memory | 8 MiB | 256 MiB |
| One message, end to end | 5 s | 600 s |
| One tool call | 5 s | 600 s |
| One planning call | 2 s | 600 s |
| One game round trip | 1 s | 600 s |
| Game start and handshake | 120 s | 600 s |
| Tool calls per second / burst | 16 / 32 (and the client's own, if lower) | 256 / 512 |
| Actions per session | 1000 | 1000000 |
| Bad messages before cut-off | 16 | 1000 |
| Observation age for proposals | 2 s | 600 s |
| Game reply line | 256 KiB | fixed |
| Allowlist file | 64 KiB | fixed |
| Audit record | 8 KiB | fixed |

## Allowlist

    {"version": 1, "clients": [
      {"name": "grok", "transports": ["stdio"], "token_sha256": "<64 lowercase hex>",
       "tools": ["observe", "plan", "propose", "proposal_status", "audit_head"],
       "rate_per_sec": 8, "burst": 16, "max_mode": "record"}]}

- Only a token's SHA-256 is stored. The library never copies the token: it hashes the host's buffer, compares the
  hashes in constant time, and wipes its own copy of the hash.
- Grants come only from this file or the host console, never over MCP.
- `max_mode` may be `record` or `advisory`.
- Unknown fields are refused.

## Audit log

- One file per session, created exclusively; an existing file is never overwritten.
- Each line is `<record JSON> <64 hex>`, where the hash is SHA-256(previous hash bytes || record JSON).
- The first previous hash is 32 zero bytes.
- `peachy_auditcheck <log> [--expect-head <hex>] [--expect-length <n>]` re-derives the chain.
  - Records cut from the end leave a whole chain behind, so compare against a head the client got from `audit_head`.
- This is a scratch side chain. It is never the official PCHIE chain, never anchored, and never sent to a witness.

## Planner

- `PEACHY_PLANNER_PCHIE_DIRECT` opens the PCHIE library and calls `pchie_get_api`.
- `PEACHY_PLANNER_MJOLNIR` opens the Mjolnir library and calls `mjolnir_get_api`. Mjolnir attaches the PCHIE library
  from the path the host gives, then hands its table over.
- Both kinds end up with the same PCHIE table, used the same way:
  - a free decide session with no model, and flags 0 (the official defaults)
  - the search budget is counted in nodes, never time
  - a decision outside the legal mask becomes `hold`
- If either library is missing or its interface version is wrong, the session runs without a planner. `plan` says so,
  and `propose` uses the legal-move mask as the gate.
- `include/peachy/planner_abi.h` mirrors the draft `pchie.h` field for field.
- Neither `pchie.dll` nor `mjolnir.dll` exists yet. The tests use stand-in libraries (`tests/doubles/`) that follow
  the same tables.

## Building and testing

    cmake -S peachy -B build && cmake --build build && ctest --test-dir build
    cmake -S peachy -B build-asan -DPEACHY_SANITIZE=address,undefined     # also: thread
    cmake -S peachy -B build-fuzz -DPEACHY_SANITIZE=address,undefined -DPEACHY_LIBFUZZER=ON   (clang)
    cmake -S peachy -B build-fuzz -DPEACHY_SANITIZE=address -DPEACHY_LIBFUZZER=ON             (MSVC)

From the PCHIE tree, Peachy builds with `PCHIE_PEACHY=ON` (the default). The tests cover:
- the JSON reader and writer
- SHA-256 vectors and audit tampering
- the SPSC rings under stress
- the scripted server
- a fake game over the real spawn path and the in-process link
- both planner paths
- the library boundary (1000 load and unload cycles)
- `peachy_host` end to end
- PM-8 stubs
- a check that no lock appears in the source
- a check that the library exports exactly `peachy_get_api`
- a fuzz smoke run

`peachy_fuzz <target> <runs> <seed>` is the long in-house fuzzer.

## Status

| Milestone | State |
|---|---|
| PM-1 design note | this file |
| PM-2 bounded JSON, audit log | done |
| PM-3 server core, tools, limits | done |
| PM-4 library shell, pump and own-thread modes | done |
| PM-5 host tool, allowlist, tokens | done |
| PM-6 game client | done (fake game and callbacks; the real headless game is not yet compared) |
| PM-7 plan and propose | done against stand-in planner libraries |
| PM-8 live frontier-model hookup | client profiles and stub tests only (`configs/`) |

MIT license (see `LICENSE`).
