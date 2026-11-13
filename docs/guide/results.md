# Results so far

This page reports measurements that are already in the notes and tests. It does not invent a success the tree does not show. Every hash below from November 2026 work is provisional unless it is identified as a frozen Official pin. Official records, goldens, and reports were not edited to follow later walks. Times are America/New_York.

## Official pins that have not moved

Official B1 search, with the loop guard on or off (the walk does not loop):

- Actions SHA-256: `0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e`
- Chain head: `de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533`
- Official B1 `H_world`: `457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5`

The default recomputable ratchet-head pin and the v1 SHA-256 golden used in the ratchet and TESLA tests also stayed put. Thread-count checks at 1, 2, 4, 8, and 16 workers produced identical actions, chain heads, tree digests, and (where present) tuple contents.

## Planner budget comparison (13 November 2026)

B1, 60 steps, budget 16, CPU only, loop guard on. L1 (model-driven PUCT) ran against a null-value control and against the official oracle. The comparer verdict was `NO_SEPARATION_SAME_WALK` (exit 10). L1 did not beat null. No higher-budget L1 run was started.

L1 and null committed the same walk. They did not reach the exit. The loop guard broke the two-cell bounce seen in earlier partials; both arms then cycled over three cells and made no progress toward the exit (BFS remaining stayed 50). Each arm triggered the guard seven times. The actions hash, provisional, was `109887938bc412da6dfb46f9655f3281d5c3326fd1fabe670678fd1c6402a2a5` on both arms.

The oracle reached the exit in 50 steps, which is the official B1 BFS length. Its actions hash is the official B1 search hash and did not move. It used 12 workers and finished in 7.4 seconds. L1 and null then ran in parallel at six workers each. Wall times were 4645.5 s for L1 and 3890.2 s for null. Tuple re-hash passed on all three arms. No oracle calls from L1 or null, no gate failures, no leaked KV pages.

The honest finding: at budget 16, even with the loop guard breaking the old bounce, L1 is not separating from null. That result should be reported before any higher-budget L1 run.

## Query-before-plan (13 November 2026)

B1, 25 steps, budget 8, loop guard on. The comparer verdict was `L2_ENGAGED_WALK_UNCHANGED` (exit 0). Memory hits were used as proposals and went through the gate, but they did not change the committed walk.

All three arms — L1 with a seeded store (25/25 hits, 0 rejected), L1 with an empty store (0/25 hits, 25 misses), and null with a seeded store (25/25 hits) — committed the same walk: no exit, path 0, progress 1, BFS remaining 49, three guard triggers, three unique cells. The provisional actions hash was `f45b11638ad054e8e6baaa7b0bfefd88ffd05fb51435403b5f7b6c91c3941475`. Store writes were off. Tuple re-hash passed. Gate fail-closed count was 0.

L1 with the seeded store used 3.7% fewer model forwards than L1 with the empty store (3809 versus 3956). An earlier run with the guard off had a 4.4% saving and the same "engaged, walk unchanged" verdict, on a two-cell walk. With the guard on, the walk is three cells and a different actions hash.

## Stage B standard pair (13 November 2026)

One invocation of the Stage B harness with real Little Nemo: the unit section, then real run 1 and real run 2, checked for run-to-run determinism. Twelve workers, loop guard on. Exit 0, `PASS (0 failures)`.

The door outcome is an honest stall at door 1, the same shape as the official stall record. Door 0 opened on a correct Dividor answer. Door 1 stayed closed: 141 identical Dividor proposals for the wrong component (observed 4, goal 20). No win, path 26, 168 steps (the cap). The official record was not retuned. Loop-guard triggers were 0 and 0: door attempts are not counted, and navigation did not loop.

Run 1 and run 2 match. Hashes, provisional, and unchanged from the 11 November mutex-free Stage B log:

- Side log SHA-256: `0d5242852053eee2f4c037f674745311ac1e51738c4be1f82b3f073be5fb5aa7`
- `H_world` SHA-256: `e553135d3aa51695ce9c4b38db5668c882e60d9e69cad8c6fca2e0f318c0fd16`

Official B1 `H_world` and the draft L1 key checked at the start of the unit section were unchanged. Predicted equalled real for every commit. No KV page leak. Wall times reported by the harness: run 1 1389.166 s; run 2 2876.048 s (cumulative from the start of the real pair).

## Loop-guard walks that did move (provisional)

When the guard became the default, walks that used to loop changed. Two unit-driver examples, both provisional:

- Stage A 16x16, no-model unit driver: the old walk looped until the step budget (seven unique cells). With the guard, one trigger, and the walk reached the exit in 478 moves. Actions `0723de73b63cced9f4e365602298cf97b11eef2cd81eb73275a26e882e655873`.
- B1 16x16, reasoning-layer decide path with null value and no model (40 steps, budget 8): a two-cell bounce became a three-cell cycle with five triggers. Actions `f21fb000b0fb4b070fd546bbf66964a1e268cb35168f2895101e55abab71a197`.

A worker-agreement check of a short real-model null-value walk was identical at 1, 2, 4, 8, and 16 workers once the run label was removed from the tuples.

## Hardware attestation bench check (13 November 2026)

The PCR 23 bench check appraised as Attested, with the two bench-grade caveats in [Hardware attestation](hardware-attestation.md): the harness attestation key has no EK certificate chain, and the quote scheme is RSASSA-SHA1. A morning attempt that produced no quote was honestly Unattested.

## What this does not show

It does not show that L1 is a better planner than null at this budget. It does not show that query-before-plan changes the committed walk. It does not show that Stage B can open door 1 with the current Dividor behavior. It does not show a production-grade attestation key. Those are the honest edges of the recorded work.
