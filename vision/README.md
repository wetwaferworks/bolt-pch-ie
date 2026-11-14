# vision

Vision-only navigation ("visnav"). The model sees live rendered frames from the game, encoded to text, and must find the maze exit without reading the world map. A crutch audit flags any forbidden read. Status: Scratch.

One driver, `pchie_lk_visnav.cpp`. The older way-shut, reach, seeded-maze, and CUDA copies were the same file with flags added. Way-shut, reach, act-blank, and memory injection stay opt-in and default off. Memory receipts are on by default. Seed, size, and game resolution keep the original defaults (seed 1, size 16, 640x360). A run with no extra flags keeps the reach-off decision path.

| Flag or switch | What it does |
|---|---|
| `--way-shut` | A timed-out walk that recentres marks that directed edge shut. |
| `--reach N` | Interaction range in world blocks, 0..4. Default 0. |
| `--b1-seed`, `--b1-size` | Stratify generated B1 mazes. Defaults are seed 1 and size 16. |
| `--act-blank` | Holdout: act on the blank-view choice. |
| `--game-res WxH` | Game window resolution. Capture size stays `cap_w` x `cap_h`. Default `640x360`. |
| `--neml on\|off\|DIR`, `--neml-inject` | Episodic memory ledger. A receipt is written after every nav run unless `--neml off`; DIR picks another ledger folder. Inject (the memory block in the prompt) stays off. |
| `PCHIE_SCRATCH_CUDA_VISNAV` | Compile-time. The decide step uses the CUDA adapter. The CPU build does not define it. |

The ledger lives in `include/pchie/neml/` (its checks are `tests/pchie_neml_test.cpp`). `cuda/CudaNemotronModel.h` is the labeled CUDA adapter. `scripts/extract.py` is the pure-region extractor. `visnav_pure_test.cpp` needs `pure.inc`, which is generated and not checked in: from `vision/`, run `python scripts/extract.py pchie_lk_visnav.cpp pure.inc`.

The root CMake Release build on Windows builds the CPU driver as `build\Release\pchie_lk_visnav.exe`, with everything else. The two scripts here build it on its own, from `vision/`, after the root Release `pchie_core.lib` exists. They find Visual Studio 2022 through `vswhere` (any edition with the C++ tools), or use the compiler already on `PATH` in a Developer Command Prompt.

- `build_visnav.bat` builds the CPU driver into `vision\build\`.
- `build_cuda.bat` builds the CUDA driver. It needs `CUDA_PATH`.
