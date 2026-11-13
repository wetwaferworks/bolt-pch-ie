# experiments

How the November 2026 measurements were run.

The launchers under `next-runs/` take the model path from the environment: set `PCHIE_MODEL` to the GGUF weights file, or `PCHIE_MODELS` to the directory that contains `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`. They use this clone's Release build and write under `runs\` (gitignored). Official records are not touched.

| Folder | Contents |
|---|---|
| `next-runs` | Launchers for the planner budget comparison (`run_planner_budget_comparison.ps1`), query-before-plan (`run_query_before_plan.ps1`), and the Stage B standard pair (`run_stageb_pair.ps1`). Commands, durations and what each result shows: `docs/notes/next-runs.md`. |

Findings are summarized in `docs/reports/` and in the researcher guide. The older one-off lab scripts are not included.
