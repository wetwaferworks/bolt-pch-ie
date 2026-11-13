# vision

Vision-only navigation ("visnav"). The model sees live rendered frames from the game, encoded to text, and must find the maze exit without reading the world map. A crutch audit flags any forbidden read. Status: Scratch.

One driver, `pchie_lk_visnav.cpp`. The older way-shut, reach, seeded-maze, and CUDA copies were the same file with flags added. Way-shut, reach, act-blank, and NEML stay opt-in and default off. Seed, size, and game resolution keep the original defaults (seed 1, size 16, 640x360). A run with no extra flags keeps the reach-off decision path.

| Flag or switch | What it does |
|---|---|
| `--way-shut` | A timed-out walk that recentres marks that directed edge shut. |
| `--reach N` | Interaction range in world blocks, 0..4. Default 0. |
| `--b1-seed`, `--b1-size` | Stratify generated B1 mazes. Defaults are seed 1 and size 16. |
| `--act-blank` | Holdout: act on the blank-view choice. |
| `--game-res WxH` | Game window resolution. Capture size stays `cap_w` x `cap_h`. Default `640x360`. |
| `--neml DIR`, `--neml-inject` | NEML episodic ledger. Off unless a directory is given. Inject stays off. |
| `PCHIE_SCRATCH_CUDA_VISNAV` | Compile-time. The decide step uses the CUDA adapter. The CPU build does not define it. |

`neml/NemlLedger.h` is the ledger. `neml/neml_tool.cpp` is its unit tool. `cuda/CudaNemotronModel.h` is the labeled CUDA adapter. `scripts/` holds the pure-region extractor and the old reach launch notes. Those notes still name the scratch paths they were written for. `visnav_pure_test.cpp` needs `pure.inc`, which is generated and not checked in.

Build from `vision/`, after the root Release `pchie_core.lib` exists:

- `build_visnav.bat` builds the CPU driver.
- `build_cuda.bat` builds the CUDA driver. It needs `CUDA_PATH`.
- `build_neml_tool.bat` builds the ledger tool.
