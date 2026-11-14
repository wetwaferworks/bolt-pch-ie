# Getting started on Windows

This page is a first walk through a Release build of Bolt PCHIE on Windows. It stays with what the tree actually builds and tests. For the longer design notes, start with the [researcher guide](README.md). To check a downloaded file, see [Checksums](checksums.md).

## Prerequisites

- 64-bit Windows 10 or 11
- Git
- CMake 3.24 or newer
- Visual Studio 2022 Build Tools or later, with the Desktop development with C++ workload. The Visual Studio 17 2022 and Visual Studio 18 2026 generators both work. CMake may arrive as part of those tools; if `cmake` is not on your PATH, use the CMake that ships with Visual Studio, or open an "x64 Native Tools" developer command prompt
- A CPU that supports AVX2 (by default `-DPCHIE_AVX2=AUTO` builds the AVX2 kernels with `/arch:AVX2` when the compiler and this machine support it; `-DPCHIE_AVX2=OFF` builds the portable kernels)
- Optional: CUDA 12, only if you want the CUDA backend. Without a CUDA compiler the CPU path still builds
- Optional, for a first run that loads the model: the Little Nemo GGUF weights file `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`. The weights are not in this repository

You do not need LithoKraft to build the harness or to run the quick tests. LithoKraft playtest packages are not part of this repository.

## Configure and build Release

From the repository root, in a developer command prompt (in PowerShell, write `$env:NUMBER_OF_PROCESSORS` instead of `%NUMBER_OF_PROCESSORS%`), which uses every hardware thread:

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS%
```

If your Build Tools are Visual Studio 2026, use `-G "Visual Studio 18 2026"` instead. `cmake --help` lists the generators you have.

The reasoning layer is part of this root build. The tools you will use first land in `build\Release\`, including `pchie_anchor.exe`, `pchie_tesla_watcher.exe`, `pchie_lk_pilot.exe`, `pchie_lk_visnav.exe`, `pchie_neml.exe`, `pchie_stageb_door_test.exe`, and `pchie_rl_l1l2_test.exe`. CUDA is controlled by `PCHIE_CUDA` and is on by default when a CUDA compiler is found.

## Run the quick tests

The full CTest list includes long suites that load Little Nemo or run large mazes. Day-to-day work uses the quick set, which excludes those thirteen long suites. `pchie_stageb_unit` stays in the quick set: it is the Stage B unit section only, it does not load the model, and it should finish in well under a second.

```
cd build
ctest -C Release -j %NUMBER_OF_PROCESSORS% -E "pchie_stageb_door$|pchie_scaled_pit_maze|pchie_trace_curriculum|pchie_nemotron|pchie_determinism_gauntlet|pchie_far_retry|pchie_budget_timeout|pchie_wander_regression|pchie_freeze_recovery|pchie_mirror_sim_gate|pchie_maze|pchie_prompt_injection|pchie_rl_l1l2"
```

Several tests in that quick set do load the weights (`pchie_obs_contract`, and several of the planner, search, and HTN tests). Set the model path before you run them:

```
set PCHIE_MODEL=C:\path\to\Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
```

Or set `PCHIE_MODELS` to the directory that holds that file. There is no machine-specific default; a missing setting is an error for any test or tool that needs the model.

After the tests, compare reported Official pins with [Determinism and hashing](determinism-and-hashing.md) and [Results so far](results.md). A moved Official pin is a finding: stop and report it. A new provisional hash from a looping walk, with the loop guard on, is a new measurement, not a rewrite of an official record.

## A first harness run

You can confirm the Stage B unit section without weights:

```
build\Release\pchie_stageb_door_test.exe --unit
```

The reasoning-layer unit section is the same idea: `build\Release\pchie_rl_l1l2_test.exe --unit`.

A first run that loads Little Nemo should leave official records alone and write under a fresh folder. The launchers in `experiments\next-runs\` do that. They take `PCHIE_MODEL` or `PCHIE_MODELS`, use this clone's Release build, and write under `runs\` (gitignored).

```
powershell -File experiments\next-runs\run_stageb_pair.ps1 -DryRun
```

The Stage B standard pair with real Little Nemo is a long run (on the order of an hour). The planner budget comparison and query-before-plan launchers in the same folder are also long measurements, not a smoke test. Add `-Watch` if you want a log tail that does not steal focus. Loop guard v2 is on by default; pass `-LoopGuard off` only when you mean to reproduce an older walk. TESLA stream authentication and attestation are on by default in the harness; `--tesla off` and `--attest off` turn each one off.

If you want to walk a LithoKraft maze with the pilot, you also need a game build and `--game` (or `PCHIE_GAME` / `LITHOKRAFT`). That is a later step; it is not required to build or to run the quick tests. See [LithoKraft and LithoLattice](lithokraft-integration.md).

## Common errors

- **`cmake` is not recognized.** Use a Visual Studio developer command prompt, add the Build Tools CMake directory to `PATH`, or invoke CMake by its full path. The build will not start until `cmake -S . -B build` works.
- **The generator does not match the installed Visual Studio.** If `"Visual Studio 17 2022"` is rejected, try `"Visual Studio 18 2026"`, or run `cmake --help` and pick a listed Visual Studio generator. Keep `-A x64`.
- **A quick test fails with a missing model.** Set `PCHIE_MODEL` or `PCHIE_MODELS` as above. `pchie_stageb_unit` must still pass without those variables; if it asks for a model, stop and report it.
- **`ctest` cannot find the tests, or it runs a Debug binary.** Configure and build Release, then pass `-C Release`. Do not mix a Debug build with the Release test configuration.
- **The compiler rejects AVX2, or the process crashes on an older CPU.** The harness requires AVX2. Build and run on a machine that supports it.
- **CUDA is missing.** That is fine. The CUDA backend is skipped and the CPU path still builds. You only need CUDA 12 if you asked for that backend.
- **A launcher says the Release executable is missing.** Build Release first. The next-runs scripts look for `build\Release\pchie_rl_l1l2_test.exe` and `build\Release\pchie_stageb_door_test.exe`.
- **A launcher refuses to start because a log already exists.** It will not overwrite a previous label. Use a fresh output folder, or move the old log aside.
- **An Official pin moved.** Stop and report it. Do not retune official records. If you are looking at a walk that used to loop, compare with `--loop-guard off` before treating the new hash as a regression.
