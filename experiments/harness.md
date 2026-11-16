# Harness

The harness runs Little Nemo (or any agent) on tasks and games and grades him the same way every time.
It is scratch research: its own library (`pchie_harness`), off the official path. Nothing in it touches
`pchie_core`, the official chain, goldens, or the defaults of the `--rl-*` and `--research-*` flags.
The engine rule holds here too: no locks. Samples and episodes run in parallel on the join dispatcher, and
`std::thread::join` is the phase boundary; results are folded and written in a fixed order afterwards.

## Tasks, solvers and scorers (`pchie/harness/Task.h`, `Eval.h`)

The structure follows Inspect AI, rebuilt in C++ (no Python runtime):

- A **task** is a list of samples, a solver factory, a scorer and an optional step budget.
- A **solver** turns one sample into an answer and records what happened. Built-ins: `prompt_template`
  (fills `{input}` and `{choices}`), `generate` (one model call), `chain` (runs solvers in order).
- A **scorer** grades the finished sample. Built-ins: `exact_match`, `first_word_match`, `includes`,
  `choice_letter`. Scores are integers in Q16 (65536 is full marks), so results are bit-exact.
- `run_eval` runs every sample (in parallel when given a dispatcher), catches failures per sample, and
  returns per-sample scores, a mean, and a digest over the canonical results.

Every sample keeps a **transcript**: what it was asked, what it saw, what it did, what the model said, and
the score. After the join, the eval writes a side chain in sample order:
`pchie.harness.eval.v1` (header), `pchie.harness.event.v1` (one line per transcript event),
`pchie.harness.score.v1` (per sample, with the transcript's SHA-256) and `pchie.harness.summary.v1`.
Each line carries the previous line's hash (the research `SideLog` format), so
`pchie_research_tool verify FILE` checks it. The digest and the side-chain head are identical at
1/2/4/8/12 workers.

Tests: `pchie_harness_test --unit` (CTest `pchie_harness`).

## One plug for many games (`pchie/harness/GamePlug.h`)

After the BALROG pattern, every game speaks the same interface: `reset(seed)`, `step(action)` with a text
observation (plus an optional glyph picture and, for the mazes, Little Nemo's own senses), a fixed action
list, integer rewards, and a SHA-256 state digest for replay checks. Agents plug into the other side through
`IAgentPlug`. **This is the socket Mjolnir plugs into later:** an adapter is just another agent.

Grading stays outside the agent. `grade()` reads the world after the episode (success, distance closed or
achievements), and nothing from it is passed to `act()`; a test checks that no observation carries it.

| Plug | What it is |
|---|---|
| `lithokraft.maze.b1` | The headless B1 maze, any seed. He is told his odometry, facing, what is at his feet and next to him, and which ways are open. Graded by BFS distance closed. |
| `lithokraft.maze.stagea` | The 64x64 Stage A pit maze, any seed (default 64). Falling ends the episode. Graded by the pit-avoiding distance. |
| `litholattice.testbed.stub` | **Stub.** A seeded corridor so the socket is exercised end to end. The real LithoLattice testbeds will answer through the engine's MCP seam. |
| `craftax_like.grid` | A small Craftax-style crafting grid (9x9, 5x5 view): collect wood, place a table, make a wooden pickaxe, mine stone. |

Agents: `go_explore_agent()` (research L7 on his own senses) and `cycle_agent(...)` for smoke runs.
`game_task(plug, seeds, budget, agent)` turns episodes into a harness task, so games get the same transcripts,
side chain and thread-count identity as any other task. The scorer gives full marks for success and otherwise
half marks scaled by graded progress.

Note on B1 seeds: most seeds fail the generator's key-and-door check and fall back to the same corner layout
(seeds 2 and 3 are the same maze). The maze plug grades with a layout hash, and the maze runner prints one, so
a comparison can prove its seeds are different mazes.
