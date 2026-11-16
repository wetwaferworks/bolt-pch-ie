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
