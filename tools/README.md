# tools

| Path | Purpose |
|---|---|
| `maze-preview/` | Prints offline statistics for generated B1 mazes (shortest path, open cells, dead ends) to help stratify seeds. |
| `pchie_lk_pilot.cpp` | LithoKraft pilot that drives the game through the harness. Takes `--tesla on|off`, `--tesla-seed os|dev-insecure` and `--attest on|off`. |
| `pchie_neml.cpp` | Reads Little Nemo's episodic memory ledger (`--status`, `--block`, `--priors`). It never writes the ledger. |
| `pchie_tesla_watcher.cpp` | Reads a TESLA stream file and prints a verdict. Exit 0 verified, 1 not verified, 2 unreadable or oversized input. |
| `run-search-gate.ps1` | Runs the search-consistency gate (bring-up tests plus CPU/CUDA hash agreement across thread counts). |
