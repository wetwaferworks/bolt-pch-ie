# Next runs: planner comparison, query-before-plan, and the Stage B standard pair

These three launchers were first run on 13 November 2026; the findings are summarized in [Results so far](../guide/results.md). This note is the how-to for running them again.

CPU only, no GPU, no game window. The official build, chain and records are not touched. Hashes from these runs are provisional.

## Defaults since 13 November 2026

- **Loop guard v2 is part of the standard reasoning layer** (`include/pchie/rl/LoopGuard.h`), on by default in every maze driver: B1, Stage A, Stage B, and the RL decide path that the planner-comparison and query-before-plan runs use. `-LoopGuard off` on any launcher (`--loop-guard off` on the harnesses) gives the old walks exactly. Hashes that moved because of this are listed in `loop-guard-default-hash-moves.md`; the new values are provisional.
- **Every run uses all available threads by default.** The launchers detect the thread count (`[Environment]::ProcessorCount`) and the harnesses default to `std::thread::hardware_concurrency()` (capped at the dispatcher limit of 16). Runs that go in parallel share the threads, so the total stays at the machine's count. Results do not depend on the thread count; this was checked with the guard on at 1, 2, 4, 8 and 16 workers (identical actions, chain head, tree digest and tuple contents). `-Workers N` sets a per-run count instead.

Set `PCHIE_MODEL` to the GGUF weights file, or `PCHIE_MODELS` to the directory that contains `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`. See `experiments/README.md`.

Build first, from the clone root, if `build\Release` is older than the source (use the machine's thread count):

```
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
```

Each launcher has `-DryRun`, which checks the exe and model paths and prints the exact harness commands without starting anything. Outputs go under `runs\` in the clone, which is gitignored. A launcher refuses to overwrite a log that already exists.

## Planner budget comparison: L1 model-driven PUCT at budget 16 vs null vs oracle (B1, 60 steps)

```
powershell -ExecutionPolicy Bypass -File experiments\next-runs\run_planner_budget_comparison.ps1
```

- Runs oracle first (about 15 s with model load), then L1 (`--rl-puct-model`) and null (`--rl-puct-prior=model --rl-puct-value=null`) in parallel. The oracle gets all threads; L1 and null get half each. Options: `-Workers`, `-Steps`, `-Budget`, `-Root`, `-LoopGuard on|off`.
- With the loop guard on, the 12 November L1/null 2-cell bounce should now be broken: the 4th entry into a state forces a reasoning check (fresh seed, 4x budget, a note that names the loop, no 180 turn) and turns on cautious mode. A short real-model null run (10 steps, budget 4) on 13 November triggered once and reached 3 cells instead of bouncing between 2. Whether L1 then separates from null is what this comparison measures. Null can still cycle after a break, because its value head is flat; that is a property of the control, not a guard failure.
- Expected duration: about 2 hours on an idle machine; plan for 2.5. The cancelled 12 November L1 partial managed 45 decisions in 96 minutes on one thread while other runs shared the CPU.
- Result: `runs\rl\logs\planner_verdict_s60b16.txt`, written by `pchie_rl_l1l2_test --compare-l1`. The bar is that L1 must beat null: a different walk that gets further toward the goal, or reaches the exit when null does not.
  - `SEPARATED` (exit 0): L1 beat null at budget 16. Report it before picking the next budget.
  - `NO_SEPARATION_SAME_WALK` or `NO_SEPARATION_NOT_BETTER` (exit 10): **stop and report**. Do not start budget 32 or any higher budget until a person has reviewed it.
  - `INVALID` (exit 2): something is wrong with the runs, not the result: missing lines (cancelled run), wrong flags, mismatched steps or budget, L1 calling the oracle, tuples that do not re-hash, or leaked KV pages.
- The oracle line is the ceiling at the same budget (on 12 November it reached the exit in 50 = BFS).

## Query-before-plan (B1, 25 steps, budget 8)

```
powershell -ExecutionPolicy Bypass -File experiments\next-runs\run_query_before_plan.ps1
```

- Seeds the store from the official oracle B1 walk into `runs\rl\scratch\rl\mem\b1_seed_official.jsonl`, then runs L1 with the seeded store, null with the seeded store, and L1 with an empty store, in parallel. The seed walk gets all threads; the three runs get a third each. Loop guard on by default (`-LoopGuard off` for the old walks).
- Memory hits are proposals: a hit raises one root edge's prior only after the gate and the legality check, never a bypass. The store is read-only during a run (`qbp_write=0`). Hit and miss records go on the query-before-plan side chain, and the harness now refuses any `--mem` path outside `<root>/scratch/rl/mem/`.
- Expected duration: about 30 minutes (seed about 1 minute, runs about 26 minutes).
- Result: `runs\rl\logs\qbp_verdict_s25b8.txt`, written by `--compare-l2`. It shows hits per decision, gate rejections, side-chain heads, the change in model forwards, and whether hits changed the committed walk (`L2_ENGAGED_WALK_CHANGED` or `L2_ENGAGED_WALK_UNCHANGED`). It reports `INVALID` if L2 was not really engaged: no hits from the seeded store, hits from the empty one, store writes, or no side-chain record.

## Stage B standard pair

```
powershell -ExecutionPolicy Bypass -File experiments\next-runs\run_stageb_pair.ps1 -Watch
```

- One run of the standard harness `pchie_stageb_door_test` with real Little Nemo: the unit section, then real run1 and real run2, checked against each other for run-to-run determinism. Headless.
- `-Watch` opens the log view (`view_log.vbs`): a console tailing `runs\stageb\stageb_std.log`. It shows without taking focus, is never topmost, and stops tailing when the run ends.
- Expected duration: about an hour.
- The result shows whether Little Nemo opens door 1 or stalls there again (the official record is an honest stall at door 1, 141 identical `dividor` attempts), and whether run1 and run2 match. The unit section alone is in CTest as `pchie_stageb_unit` (0.15 s).
- Loop guard v2 is now on by default in `StageBDoorMaze` too (door attempts are not counted, only navigation commits), so the standard pair now is the "standard + loop guard v2" pairing. The log shows `loop guard: run1 triggers=N run2 triggers=N`. The harness uses all threads by default (`--workers N` to choose). `-LoopGuard off` gives the old standard walk.
- Still a gap: vision-only Stage B (visnav supports b1 and stagea only).

