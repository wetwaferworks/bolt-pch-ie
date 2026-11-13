# Reasoning scratch data

The reasoning layer builds with the root CMake project.

- Headers: `include/pchie/rl/`
- Sources: `src/rl/`
- Tests: `tests/rl/` (`pchie_rl_contract`, `pchie_rl_flags`, `pchie_rl_l1l2`)

`set_rl()` on `HeadlessMaze` and `StageBDoorMaze` is the decide path. It stays null unless a test sets it. `PchStack` stores `RlFlags` and fail-closes when they are not the official defaults.

This folder holds scratch notes and fixtures only: `data/` (label stubs, hash-move notes) and `scripts/`. Those scripts are not the build.
