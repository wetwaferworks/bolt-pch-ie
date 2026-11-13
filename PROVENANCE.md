# Provenance

`third_party/pch-public/` is a verbatim `git archive` of the original public
PersonalControlHarness repository:

- Source: https://github.com/JaydeeWetwork/PersonalControlHarness
- Commit: `754d37f1f45e5f11c2b5f0796dd2d12db27bb709` (2026-10-23)
- Paths imported: `engine/` (LithoLattice, older public engine), `game/LithoKraft/` (visualization environment), `LICENSE`
- Not imported: `third_party/DeepSeek-V3`, `pch/`, `testbeds/`, `blackbox/`, docs.
- No code was taken from the private engine, the Threading Handshake, the LithoRoot rewrites, or from later private engine trees. The import has no `host/`, `_private`, or key directories.

The Bolt host seam (`include/pchie/pchie_host.h`) is new clean-room code. It uses its own
Timing Handshake clock (K ticks per token) and a flat world. It does not use procedural
world generation, and it does not link the vendored procgen module.

The vendored `engine/README.md` and `engine/tools/build.ps1` still mention a
user-profile `agent-tools/logs` directory as the default for engine build logs.
That is leftover packaging text in the 23 Oct 2026 snapshot; this project does
not use that path. Prefer a `build/logs` folder in your own build tree. The
vendored snapshot is otherwise left unchanged; only the copyright line of its `LICENSE` was
aligned with the root `LICENSE` (same owner, same MIT terms).
