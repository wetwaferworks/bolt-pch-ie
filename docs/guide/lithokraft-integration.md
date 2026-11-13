# LithoKraft and LithoLattice

LithoLattice is the engine. LithoKraft is the game that runs on it. The names are one word each, with both Ls capital in LithoLattice and a K in LithoKraft. PCHIE talks to a running LithoKraft game as a pilot host; it also vendors a public snapshot of an older engine-and-game tree for reference. Those two relationships are not the same copy of the source, and this page keeps them distinct.

A component-level description of LithoKraft as of the 12 November 2026 playtest build, including signal rules, inventory, the Component Test world, and one spec sheet per named component, is in the [LithoKraft component reference](../components/README.md).

## What PCHIE vendors

`third_party/pch-public/` is a verbatim archive of the public PersonalControlHarness repository at commit `754d37f1f45e5f11c2b5f0796dd2d12db27bb709` (23 October 2026). The imported paths are `engine/` (the older public LithoLattice engine), `game/LithoKraft/` (the visualization environment), and `LICENSE` (MIT). Docs, tests, private engine work, the Threading Handshake, LithoRoot rewrites, and key material were not imported. The root CMake build does not add that tree as a subdirectory and does not compile the game out of this snapshot.

That snapshot is earlier than the 12 November playtest build described in the component reference. Do not assume that every block in that build, item, or inventory rule is present in the vendored copy. The PCHIE host seam (`include/pchie/pchie_host.h`) is clean-room code in this repository. It uses its own timing handshake and a flat world, and it does not link the vendored procedural-generation module.

## The LithoKraft pilot

`pchie_lk_pilot` decides a maze in the PCHIE harness and can then actuate the committed walk inside a running LithoKraft game. The control connection is the engine's own MCP server over stdio (`LithoKraftEngineHost`). The server is deny-by-default. The player grants access in the game console:

```
mcp grant world_read player_control
```

The pilot starts with no permissions and polls until those two grants exist. Rate-limited tool results are retried after a short sleep. Any other tool error fails the call (fail closed). The protocol version the host announces is `2025-11-25`; the client name is `pchie_lk_pilot`.

The decide step uses the official B1 grid (seed 1, 16x16, one key) by default, with the same driver, budget, and gate settings as the official B1 test, on a fresh side chain — never the official log. `--unit` uses the null-model unit driver. `--maze stagea` uses the labeled Stage A 64x64 pit-maze fixture, decided the same way as the scaled-maze test. Hashes are compared only; nothing is locked, and this path is not a golden and not the official chain.

Before it moves, the pilot reads the maze back from the world, cell by cell, and refuses to walk unless that world equals the grid the model decided on. Each committed action is carried out with the player's normal input (`player_move`, `player_look`, `player_action`), and the real position is read back after every step. An honest miss is still actuated; the walk is not retuned to make the game succeed.

## The 12 November playtest build, in brief

The 12 November playtest build is a packaged LithoKraft playtest (America/New_York). The executable's SHA-256 is `C0D3A18BE5FAD0A722121E8AA9859DB0B5BEBD71A67B59541E3824A925056A1F` (20,596,736 bytes, built at 11:02:08 AM ET). The game's `mod.json` names LithoKraft version `0.1.0` and depends on LithoLattice `>=0.1.0`.

That package includes playtest fixes for creative-inventory flicker, the selected-hotbar outline, Shift-click quick move, and the Agent equipment frame. Headless tests at that build, and the items that remain open, are listed in the [component reference](../components/README.md#the-12-november-playtest-build). The PCHIE pilot does not require this exact binary; it requires a running LithoKraft game whose MCP server offers the tools above and a world that matches the decided grid.
