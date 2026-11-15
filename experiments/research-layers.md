# Research layers

Six layers from the November 2026 research brief, each behind its own `--research-*` flag. All of them are
off by default and live in their own library (`pchie_research`), so `pchie_core`, the reasoning-layer flags
(`--rl-*`) and every existing hash are unchanged. Official runners never pass `--research-*` arguments.

| Layer | Flag | Header | What it does |
|---|---|---|---|
| L7 Go-Explore | `--research-go-explore` | `pchie/research/GoExplore.h` | Archive of every cell Little Nemo has reached (place, key state, view bucket). After a short burst of exploration he picks a promising cell, walks back to it over edges he has already walked, and explores from there. |
| Constraint peer beside L6 | `--research-smt`, `--research-smt-binary=PATH` | `pchie/research/ConstraintSolver.h` | Exact answers for Stage B gate doors and for key/door ordering on his own map. A built-in solver is the reference; a pinned Z3 binary (SHA-256 checked, `smt.random_seed=0`, step-based `rlimit`) can run beside it and must agree. |
| Step scorer | `--research-step-scorer[=record\|enforce]` | `pchie/research/StepScorer.h` | Scores one proposed step from what he knows. Rule weights are the baseline; a frozen table pinned by SHA-256 is the slot for a trained scorer. Enforce mode may lower a prior, never raise one. |
| Tree of Thoughts | `--research-tree-of-thoughts`, `--research-tot-breadth=N`, `--research-tot-depth=N` | `pchie/research/TreeOfThoughts.h` | Beam search over short plans. Scoring runs on many threads into separate slots; ranking uses a fixed tie order. |
| Reflexion | `--research-reflexion` | `pchie/research/Reflexion.h` | Lessons from his own trace (pits, bumps, loops, door stalls, timeouts) written as receipts on a SHA-256 side chain. Putting them into his prompt is not wired and stays off until JD gives his go. |
| Frozen dynamics look-ahead | `--research-lookahead[=N]` | `pchie/research/FrozenDynamics.h` | MuZero-style: a model learned offline from his transitions plus a value table, frozen and hashed; the look-ahead searches inside the model only. |

`--research-ctx=1024|2048` and `pchie/research/ContextBudget.h` set up the 2048 context test path;
`next-runs/run_context_compare.ps1` runs the same B1 walk at both sizes.

## In the maze runs

`pchie_rl_l1l2_test --maze b1|stagea` takes the research flags too. The maze decide loop asks the research
advisor (`pchie/rl/ResearchAdvisor.h`, `pchie/research/MazeAdvisor.h`) after the L2 query:

- Go-Explore and the look-ahead may only move a share of the prior onto a legal, gate-checked edge; an edge
  with no prior stays shut.
- The step scorer in enforce mode may only lower a prior.
- Reflexion writes its lessons to `--research-lessons=PATH` at the end of the run.
- The look-ahead needs `--research-dynamics=PATH --research-dynamics-sha=HEX` and fails closed on a wrong pin.
  `pchie_research_tool freeze OUT LOG...` learns a model from research side logs and prints its SHA-256.
- Every decision is written to `--research-log=PATH`, a SHA-256 side chain (`pchie_research_tool verify LOG`).

The advisor sees only `ResearchView`: his position relative to where he started, key state, open ways and
pits next to him. Stage B has no view yet and refuses research flags. Tree of Thoughts and the constraint
peer are not in the decide loop yet. With no research flag the loop is exactly what it was.

## ASCII views (`--obs ascii`)

`pchie_lk_visnav --nav --obs ascii[=1024|2048]` adds Nemo View pictures from `vision.capture_ascii` to the
live view text (the launcher sets `LL_MCP_NEMO_VIEW=1`). ctx 1024 asks for three 32x12 foveas (forward,
left, right); ctx 2048 asks for the 40x22 forward view and the two foveas. Each capture is decoded, its
shape checked, and its SHA-256 recomputed from the engine's canonical record before it is used
(`pchie/research/AsciiObservation.h`); any mismatch stops the run. Every view goes on its own side chain,
`visnav_ascii_side_chain.jsonl`. The blank control never carries the pictures. Without `--obs ascii`
the prompts are byte for byte what they were.

Every policy sees only `SelfObservation`: his dead-reckoned position, heading, key state and what is at his
feet or next to him. The maze grid is used only to apply his actions and to grade results.

## Tests

`pchie_research_test --unit` (CTest `pchie_research`, no model, well under a second):

- research defaults are clean and the `--rl-*` identity line does not move with research arguments;
- Go-Explore repeats exactly and its 64-maze fold is identical at 1, 2, 4, 8 and 12 workers;
- the gate solver matches the grader on every probe; key/door plans are shortest;
- the frozen step table fails closed on a wrong pin; enforce mode never raises a prior;
- Tree of Thoughts is identical at 1, 2, 4, 8 and 12 workers;
- the lesson ledger verifies and breaks on an edited line;
- look-ahead in the frozen model walks the maze it learned in the shortest number of steps;
- in the maze runs: an idle advisor changes nothing on B1 and Stage A, Go-Explore and the look-ahead repeat
  exactly, side chains verify, Stage B fails closed, and the B1 fold is identical at 1, 2, 4, 8 and 12 workers;
- ASCII views: the engine's fixture capture verifies with the same SHA-256 the engine test prints, the
  run-length code round trips, any changed cell, digits, a wrong shape or another depth table fail closed,
  both context layouts fit, and the view side chain verifies.

Add `--z3=PATH --z3-sha=HEX` to compare with a pinned Z3 binary, and `--report` for a 64-maze table of
Go-Explore against a least-tried baseline.
