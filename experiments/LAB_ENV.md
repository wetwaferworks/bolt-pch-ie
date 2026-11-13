# Lab script environment

Scripts under this tree do not hard-code machine paths.
Set the variables you need before running them. There is no default path; a
missing variable is an error.

- `PCHIE_MODEL` — path to the Little Nemo GGUF, or `PCHIE_MODELS` — directory
  that contains `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`
- `PCHIE_LAB` — work root for measurement-chain and vision-batch helpers
  (status files, stop files, visnav ledgers)
- `PCHIE_ENGINE_ROOT` — LithoLattice / LithoKraft source tree used by the
  in-engine build and patch helpers
- `PCHIE_WORKBENCH` — workbench tree (baselines, visnav builds, encoder configs)
- `PCHIE_DEPS` — pre-fetched CMake dependency sources for those engine builds
- `PCHIE_RL_ROOT`, `PCHIE_WK_ROOT`, `PCHIE_BACKUP_ROOT`, `PCHIE_XFER` — extra
  roots only some older helpers need
- `PCHIE_GAME` or `LITHOKRAFT` — LithoKraft executable, for tools that launch
  the game
- `PCHIE_OUT` — optional output directory (defaults to the current directory)
- `CMAKE_DIR` — optional directory to prepend to `PATH` so `cmake` can be found

Preferred day-to-day launchers are under `next-runs/` and only need
`PCHIE_MODEL` or `PCHIE_MODELS`.
