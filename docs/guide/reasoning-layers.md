# Reasoning layers and the loop guard

The reasoning layers are opt-in paths that sit on top of the official decide loop. Every layer flag defaults to off. Official runners construct the default flags and must not pass `--rl-*` arguments. When the flags are off, official decide hashes are unchanged.

The layers live under `include/pchie/rl/`. A driver takes the model-driven path only when it is given an engaged reasoning-layer object. Official runners never do that.

## The six layers

| Layer | Flag (default off) | What it is, as implemented |
|---|---|---|
| L1, model-driven PUCT | `--rl-puct-model` (prior and value both model) | Root and leaf priors come from Little Nemo action-word logits on a per-decision branch. Leaf value can be the model's closer / farther / stuck mass, a constant 0.5 (null), or the official distance oracle. Environment terminals (goal reached, dead, no legal move) are game rules and apply the same way to model and null runs. Under model prior and value, oracle calls must stay at zero. Results under the flag are L1 scratch, never official. |
| L2, query-before-plan | `--rl-qbp` | Memory hits are proposals only. A hit may raise one root edge's prior after the gate and legality checks pass. It never commits an action directly. The store is not written while a run is live. A query-before-plan path without a store fails closed. |
| L3, fork-and-score | `--rl-fork-score` | Opt-in fork scoring, with a default horizon of 2. Present in the tree; the recorded November 2026 comparison runs did not turn it on. |
| L4, belief | `--rl-belief`, `--rl-vision-only` | Belief state and a crutch audit. `--rl-vision-only` implies belief. Present; not exercised in the recorded L1/L2 comparisons. |
| L5, options | `--rl-options` | Named option sets (for example corridor, turn, backout, probe). Present; default off. |
| L6, symbolic peer and verifier | `--rl-symbolic-peer`, `--rl-verifier` | A symbolic planner and a verifier that can record or enforce. The verifier starts in record mode. Present; default off. |

Honesty rules on the L1/L2 decide path: legal-move masking and the PCH gate stay in force. A gate-rejected edge gets prior 0 and its leaf is terminal with value 0 (fail closed). Decision tuples are sealed on a scratch side chain. The official hash chain is not written by this path except through the unchanged search calls the official loop already makes.

A post-hoc comparer reads finished logs only. It never feeds a run or touches a chain. For L1 it asks whether the model-driven walk beat a null-value control at the same steps and budget. If it did not — the same walk, or a different walk that gets no further — the rule is to stop and report, and not to start a higher-budget L1 run until that result has been seen. For L2 it checks that query-before-plan was engaged, stayed on the side chain, and whether hits changed the walk.

## Loop guard v2

Loop guard v2 is the reasoning layer's standard loop recovery. It is used by every maze driver: `HeadlessMaze` (B1 and Stage A), `StageBDoorMaze` (Stage B), and the L1/null decide path. It is on by default. `--loop-guard off` (or `set_enabled(false)`) restores the old behavior exactly.

Every committed state is counted by a key the maze supplies (the cell, plus whatever else defines the state). When a state is entered for the fourth time — a loop repeated more than three times — the guard triggers:

- The next decision is a forced new decision point with a reasoning check: a fresh model forward on a prompt that names the loop, a fresh search seed, and four times the search budget.
- That decision may not turn 180 degrees (the state just left is taboo at the root, while any other move exists). If every remaining move would be taboo, the filter leaves the legal set alone.
- Cautious mode stays on for the rest of the run. Drivers with unsafe cells, such as pits, check each step's landing before it is a candidate.

The model still chooses through its own search; no path is forced. Counts restart after the forced decision. This is v2. An earlier v1 banned every loop cell, which also banned the forward crossing and kept re-looping.

A run that never enters a state four times gets exactly the old walk and the old hashes. Official B1 search does not loop, so its actions hash and chain head are the same with the guard on or off. Walks that used to bounce can and do change; those new hashes are provisional. The official records were not edited to follow them.

Door attempts in Stage B are not counted, only navigation commits. The recorded Stage B pair did not loop in navigation, so the guard did not trigger and the honest door-1 stall was unchanged.
