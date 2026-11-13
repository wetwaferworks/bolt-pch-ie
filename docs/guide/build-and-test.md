# Build and test

For a first walk-through, including prerequisites, a first harness run, and common errors, see [Getting started on Windows](getting-started-windows.md) or [Getting started on Linux](getting-started-linux.md). This page is the shorter reference.

Windows, Visual Studio 2022 Build Tools, CMake 3.24 or newer, and optional CUDA 12. Model weights are not in the repository. The reasoning layer is part of the root build.

From the repository root:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
```

CUDA is controlled by `PCHIE_CUDA` (on by default when a CUDA compiler is found). Without a CUDA compiler, the CUDA backend is skipped and the CPU path still builds. Sub-projects can also be built on their own: `anchoring/` and `vision/`. The root build already produces `pchie_anchor`, `pchie_tesla_watcher`, and `pchie_lk_pilot`.

Launchers under `experiments/next-runs/`, and any test or tool that loads Little Nemo, take the model path from the environment: set `PCHIE_MODEL` to the GGUF weights file, or `PCHIE_MODELS` to the directory that contains `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`. There is no machine-specific default; a missing setting is an error. The Stage B unit section (`pchie_stageb_unit`) does not load the model and does not require these variables. Launchers write under `runs/` (gitignored) and do not touch official records.

Older one-shot lab scripts under `experiments/` use the same model variables, plus roots such as `PCHIE_LAB`, `PCHIE_ENGINE_ROOT`, `PCHIE_WORKBENCH`, and `PCHIE_GAME` (or `LITHOKRAFT`) when they need a working tree or a LithoKraft executable. They fail clearly if a needed variable is unset. See `experiments/LAB_ENV.md`.

## Quick tests and long suites

The full CTest list includes both short checks and long suites that load Little Nemo or run large mazes. The quick set is what day-to-day work uses. A typical run is:

```
ctest -C Release -j %NUMBER_OF_PROCESSORS% -E "pchie_stageb_door$|pchie_scaled_pit_maze|pchie_trace_curriculum|pchie_nemotron|pchie_determinism_gauntlet|pchie_far_retry|pchie_budget_timeout|pchie_wander_regression|pchie_freeze_recovery|pchie_mirror_sim_gate|pchie_maze|pchie_prompt_injection|pchie_rl_l1l2"
```

The thirteen names in that exclude list are the long suites recorded as "not checked" in the quick-test notes when the loop guard became the default. `pchie_stageb_unit` is *not* in the list: it is the Stage B harness's unit section only (no model, well under a second) and belongs in the quick set. `pchie_stageb_door` without `--unit` is the full Stage B pair with real Little Nemo, and is a long suite.

Most of those long suites use the official search, which does not move unless a walk enters a state four times. Any suite that did loop would now show a different, provisional value with the guard on. Compare with `--loop-guard off` before treating a moved hash as a regression.

The TESLA stream test drives `pchie_tesla_watcher` as a separate process and never talks to a TPM. The anchor verifier test makes throwaway keys and logs under the build folder and removes them on a clean pass. In-process attestation tests use a simulated quote only.

## What to compare

After the tests, compare reported chain heads with the frozen Official pins and with the measurements in [Results so far](results.md). Official B1 search actions and chain head, and official B1 `H_world`, are listed in [Determinism and hashing](determinism-and-hashing.md). A moved Official pin is a finding. A new provisional hash from a looping walk, with the guard on, is a new measurement, not a rewrite of the official record.

To check an anchored run:

```
pchie_anchor verify <log> --anchor-index i --chain-file <side log> --fingerprint "<digits>"
```

Each check prints PASS or FAIL.
