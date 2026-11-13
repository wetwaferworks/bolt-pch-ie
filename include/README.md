# Core harness (include/ and src/)

Public headers live in `include/pchie/`, implementations in `src/`. Reasoning-layer headers live in `include/pchie/rl/`, implementations in `src/rl/`. Highlights: GGUF reader and weights hash, deterministic Q4/K-quant math, CPU host and CUDA backends, copy-on-write KV pools, search (MCTS/PUCT, virtual loss, first-play urgency), the PCH stack and gates, maze fixtures (`HeadlessMaze.h` with opt-in loop guard v2, tiptoe, and `set_rl()`, `StageBDoorMaze.h`) and SHA-256 hash chains.
