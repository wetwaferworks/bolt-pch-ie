# Loop guard v2 as the default: hashes that moved

Every new value below is provisional. The official records, goldens and reports are frozen and were not edited; their old values stay valid and are reproduced by `--loop-guard off` (or `enable_loop_guard(false)`), which is pinned in `tests/rl/pchie_loop_guard_test.cpp`.

## Why hashes move

On 13 November 2026 loop guard v2 became part of the standard reasoning layer rather than a B1-only add-on. It now lives in one place, `include/pchie/rl/LoopGuard.h`, used by `HeadlessMaze` (B1 and Stage A), `StageBDoorMaze`, and the RL decide path in `include/pchie/rl/MazeL1.h` (`rl::MazeRl`, which the planner-comparison and query-before-plan runs use). It is on by default.

The guard acts only when the walk loops: the 4th entry into the same state forces a reasoning check (fresh seed, 4x budget, a note naming the loop, no 180 turn) and turns on cautious mode. A run that never enters a state 4 times gets exactly the old walk and the old hashes. So a hash moves only where the old run looped.

## Moved

| Run | Old value (and with `--loop-guard off`) | New default value (provisional) | Cause |
|---|---|---|---|
| Stage A 16x16, no-model unit driver (`run_unit`, budget 16), `pchie_loop_guard` | actions `a3f8f6381ab03d1bc508a2c652af8fb0bd68ba2c6221d88ae32f3ddf3c2de0c8`, chain `f89eaf1930e01cdb5233827f7eb7f21157c1b84842fcf1455dccf2c923390d5b`, 7 cells, failed at the step budget | actions `0723de73b63cced9f4e365602298cf97b11eef2cd81eb73275a26e882e655873`, exit reached in 478 moves, 1 trigger | The old walk looped until the step budget. The guard broke the loop once and the walk reached the exit. Same actions hash as the 12 Nov in-engine v2 null-control result. |
| B1 16x16, RL decide path with null value and no model (40 steps, budget 8), `pchie_loop_guard` | actions `dc09b5181f0d62f6a2c814e940c9bb53c0e14099d17f39d108efdf40124995d4`, chain `89332db1e49e965553605abeeef0bdc18a8037221587e1cf54f8d867d74bf005`, 2 cells (a 2-cell bounce) | actions `f21fb000b0fb4b070fd546bbf66964a1e268cb35168f2895101e55abab71a197`, 3 cells, 5 triggers | The decide path used to ignore the guard, which is why the earlier planner-comparison L1/null runs bounced. The guard now breaks each bounce; with a flat value and no model to read the note, the walk then cycles over 3 cells. |
| B1 real model, null value (10 steps, budget 4), worker check in `runs\lg-workers` | not run with the old code (the 12 Nov null s60b16 tuples show the 2-cell bounce) | actions `ffa32a261bee96e3560f76f062a2fb701a3560c23b66afe5561382a89e48907e`, chain `8d75994b426f50d5aa8b5dbeb0ecae20b7fc088fe8dbeb4ac8b1dcc26f4a2e55`, tree digest `c693f89eb8c1a297`, 3 cells, 1 trigger | Same cause. Identical at 1, 2, 4, 8 and 16 workers (tuple contents identical once the run label is removed: `ecbc8c71df5feb59f7c9924f0b6aabd28e6d9f25b0f574040fb6b62a86e31d54`). |
| Planner comparison: L1 and null, B1, 60 steps, budget 16 | 12 Nov partial: identical L1/null moves on a 2-cell bounce; no finished actions hash | actions `109887938bc412da6dfb46f9655f3281d5c3326fd1fabe670678fd1c6402a2a5`, 3 cells, 7 triggers, no exit (provisional). L1 and null committed the same walk. Oracle unchanged: actions `0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e` | The guard broke the 2-cell bounce; both arms then cycled over 3 cells. L1 did not separate from null. Full write-up in `runs-2026-11-13.md`. |
| Query-before-plan: L1 and null, B1, 25 steps, budget 8 | 12 Nov, guard off: actions `199f76a0…`, 2 cells | actions `f45b11638ad054e8e6baaa7b0bfefd88ffd05fb51435403b5f7b6c91c3941475`, 3 cells, 3 triggers (provisional). All three arms committed the same walk | Same cause. Hits went through the gate (25/25 seeded, 0/25 empty) but did not change the walk. |
| Stage B standard pair, real Little Nemo run 1 and run 2 | 11 Nov mutex-free: side log `0d5242852053eee2f4c037f674745311ac1e51738c4be1f82b3f073be5fb5aa7`, `H_world` `e553135d3aa51695ce9c4b38db5668c882e60d9e69cad8c6fca2e0f318c0fd16` | unchanged. Triggers 0 and 0 | Navigation does not loop, and door attempts are not counted. Honest stall at door 1, 141 identical dividor attempts, run 1 matches run 2. |

## Not moved (checked)

- B1 16x16 official search (`run_unit`): actions `0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e`, chain head `de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533`, with the guard on and off. No loop, so no trigger. The epoch test pin is unchanged.
- Stage B unit driver: actions `a1b8944a16e60bdc2f5c8c9f9a1d9858a0569fe672e4017bff2217ff87d2fad6`, on and off. Navigation does not loop.
- The 27 baseline CTest suites from 907d171: identical output to the baseline log apart from timings and how the console encodes one emoji. Quick tests: 34 of 34 pass, so no quick-test pin moved and none needed updating.

## Not checked (long suites, excluded from quick tests)

`stageb_door` (real), `b9_scaled_maze`, `p5_l1_curriculum`, `p3_nemotron`, `b4_gauntlet`, `far_retry`, `b7_budget`, `b1_wander`, `b5_freeze`, `b9_mirror`, `b1_maze`, `b6_injection`, `rl_l1l2`. Most of them use the official search, where nothing moves unless the walk enters a state 4 times. Any that did loop would now show a different, provisional value; compare with `--loop-guard off` before treating it as a regression.