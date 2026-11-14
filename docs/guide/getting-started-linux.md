# Getting started on Linux

This page is a first walk through a Release build of Bolt PCHIE on Linux. It stays with what the tree actually builds and tests. For the longer design notes, start with the [researcher guide](README.md). To check a downloaded file, see [Checksums](checksums.md).

## Prerequisites

- A 64-bit Linux system with Git, CMake 3.24 or newer, and a C++17 compiler (GCC or Clang)
- Make or Ninja
- A CPU that supports AVX2 (the harness is built with `-mavx2`)
- The usual C++ build tools (`g++`, `make`, and a pthread-capable C library). The root CMake only requires the Threads package beyond the compiler
- Optional: CUDA 12, only if you want the CUDA backend. Without a CUDA compiler the CPU path still builds
- Optional, for a first run that loads the model: the Little Nemo GGUF weights file `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`. The weights are not in this repository

You do not need LithoKraft to build the harness or to run the quick tests. LithoKraft playtest packages are not part of this repository.

The next-runs launchers under `experiments/next-runs/` are PowerShell scripts written for Windows. On Linux, run the same Release binaries directly as shown below.

## Configure and build Release

From the repository root:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"
```

Ninja works too, if you prefer it: add `-G Ninja` to the configure line. This is a single-configuration build, so you do not pass `--config Release` to the build or to CTest.

The reasoning layer is part of this root build. The tools you will use first land in `build/`, including `pchie_anchor`, `pchie_tesla_watcher`, `pchie_neml`, `pchie_stageb_door_test`, and `pchie_rl_l1l2_test`. The LithoKraft pilot and the vision driver talk to the game through Windows pipes, so they build on Windows only. CUDA is controlled by `PCHIE_CUDA` and is on by default when a CUDA compiler is found.

## Run the quick tests

The full CTest list includes long suites that load Little Nemo or run large mazes. Day-to-day work uses the quick set, which excludes those thirteen long suites. `pchie_stageb_unit` stays in the quick set: it is the Stage B unit section only, it does not load the model, and it should finish in well under a second.

```
cd build
ctest -j "$(nproc)" -E "pchie_stageb_door$|pchie_scaled_pit_maze|pchie_trace_curriculum|pchie_nemotron|pchie_determinism_gauntlet|pchie_far_retry|pchie_budget_timeout|pchie_wander_regression|pchie_freeze_recovery|pchie_mirror_sim_gate|pchie_maze|pchie_prompt_injection|pchie_rl_l1l2"
```

Several tests in that quick set do load the weights (`pchie_obs_contract`, and several of the planner, search, and HTN tests). Set the model path before you run them:

```
export PCHIE_MODEL=/path/to/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
```

Or set `PCHIE_MODELS` to the directory that holds that file. There is no machine-specific default; a missing setting is an error for any test or tool that needs the model.

After the tests, compare reported Official pins with [Determinism and hashing](determinism-and-hashing.md) and [Results so far](results.md). A moved Official pin is a finding: stop and report it. A new provisional hash from a looping walk, with the loop guard on, is a new measurement, not a rewrite of an official record.

## A first harness run

You can confirm the Stage B unit section without weights:

```
./build/pchie_stageb_door_test --unit
```

The reasoning-layer unit section is the same idea: `./build/pchie_rl_l1l2_test --unit`.

A first run that loads Little Nemo should leave official records alone and write under a fresh directory, for example `runs/stageb/` (that `runs/` folder is gitignored). The Stage B standard pair with real Little Nemo is a long run (on the order of an hour):

```
export PCHIE_MODEL=/path/to/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
mkdir -p runs/stageb
./build/pchie_stageb_door_test --loop-guard on --workers "$(nproc)" | tee runs/stageb/stageb_std.log
```

Do not point that command at an official record directory. Loop guard v2 is on by default; `--loop-guard off` reproduces an older walk. TESLA stream authentication and attestation are on by default; `--tesla off` and `--attest off` turn each one off.

If you want to walk a LithoKraft maze with the pilot, you also need a game build and `--game` (or `PCHIE_GAME` / `LITHOKRAFT`). That is a later step; it is not required to build or to run the quick tests. See [LithoKraft and LithoLattice](lithokraft-integration.md).

## Common errors

- **`cmake: command not found`, or the version is older than 3.24.** Install a current CMake and make sure the `cmake` on your PATH is that one (`cmake --version`).
- **The compiler does not accept `-mavx2`, or the process dies with an illegal instruction.** The harness requires AVX2. Install a compiler that can target it, and run on a machine that supports it.
- **A quick test fails with a missing model.** Set `PCHIE_MODEL` or `PCHIE_MODELS` as above. `pchie_stageb_unit` must still pass without those variables; if it asks for a model, stop and report it.
- **CTest looks in the wrong directory, or you passed `-C Release` and nothing ran.** This is a single-configuration build. Run `ctest` from the `build/` directory without `-C`.
- **CUDA is missing.** That is fine. The CUDA backend is skipped and the CPU path still builds. You only need CUDA 12 if you asked for that backend.
- **You tried to run the `experiments/next-runs/*.ps1` launchers.** Those scripts are for Windows PowerShell. On Linux, invoke the Release binaries directly, as in the first-run example above.
- **An Official pin moved.** Stop and report it. Do not retune official records. If you are looking at a walk that used to loop, compare with `--loop-guard off` before treating the new hash as a regression.
