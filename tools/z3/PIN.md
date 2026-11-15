# Pinned Z3 for the constraint peer

The constraint peer (`--research-smt-binary=PATH`, see `include/pchie/research/ConstraintSolver.h`) runs only
a Z3 whose SHA-256 matches this pin. The binary is not in the repository; download the release archive,
check its SHA-256, and unpack it anywhere outside the source tree.

Z3 is MIT licensed (Microsoft Corporation). Release: `z3-5.3.3`
(https://github.com/Z3Prover/z3/releases/tag/z3-5.3.3).

| File | SHA-256 |
|---|---|
| `z3-5.3.3-x64-win.zip` | `4e48624a6895b49606b3cf91ca4fbcc069fb2973b9706532170a144baa3fd798` |
| `bin/z3.exe` (Windows x64) | `2bffeee10ba02a510d07be9c3fdf9c22f10ccc9724dc84201c5a40196f1a1e8f` |
| `bin/libz3.dll` (Windows x64) | `b6a30c2922293c2a2cab4299637935302d8fb0427762b5a3d2dc16f392f27b8c` |
| `z3-5.3.3-x64-glibc-2.39.zip` | `d56f60b06f432f33de5810f7f1a384a339131f361e2d54d08ea83eea14326880` |
| `bin/z3` (Linux x64, glibc 2.39) | `fcd0ca765d1c0b4eac776f442ec0f10294b9a6416be91fad4e52fa19012c8d01` |

Every problem is sent with `smt.random_seed=0`, `sat.random_seed=0` and a step-based `rlimit` (never a wall
clock timeout), so the same problem gives the same answer on every machine. The built-in solver is the
reference: a Z3 answer that disagrees with it is a test failure, not a result.

Check: `pchie_research_test --unit --z3=<path to z3> --z3-sha=<pin above>`; on Windows also
`--z3-lib=<path to libz3.dll> --z3-lib-sha=<pin above>`.
