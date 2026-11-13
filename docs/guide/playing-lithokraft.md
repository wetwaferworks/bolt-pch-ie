# Playing LithoKraft

LithoKraft is the voxel game that the harness walks its mazes in. It runs on the LithoLattice engine. The release ships the **12 November 2026 playtest build** as `lithokraft-d9-windows-x64.zip`: the same executable that was playtested, unchanged. This page describes that build, from its source and the [LithoKraft component reference](../components/README.md). It does not describe features the build does not have.

The repository does not build this game. `third_party/pch-public/` holds an older public snapshot of the engine and game for reference only; see [LithoKraft and LithoLattice](lithokraft-integration.md).

## What is in the download

| File | What it is |
|---|---|
| `LithoKraft.exe` | The game, LithoKraft 0.1.0. SHA-256 `c0d3a18be5fad0a722121e8aa9859db0b5bebd71a67b59541e3824a925056a1f`, 20,596,736 bytes. |
| `mod.json` | The game manifest (LithoKraft 0.1.0, needs LithoLattice 0.1.0 or later). |
| `assets/` | The engine assets: shaders, materials, UI, and branding. |
| `LithoKraft/assets/` | The game assets: block and item models, icons, texture recipes, packs, and world generation. |
| `scripts/` | The sample script folder. |
| `THIRD_PARTY_LICENSES.md`, `LICENSE` | Third-party notices, and the MIT license. |
| `SHA256SUMS.txt` | The checksum file from the playtest package. It also lists `LithoKraft.pdb`, the debug symbols, which are not in this download. |

The executable is a DebugOpt build: optimized, with checks and profiling logs still on.

## Requirements

- Windows 10 or 11, x64.
- A GPU and driver with Vulkan. Vulkan is the default renderer. If Vulkan cannot start, the game falls back to the software renderer. You can also choose one with `--renderer vulkan|gl33|software`.
- No separate runtime install. The game uses the static C++ runtime.

## Launch

Unpack the zip to a folder you can write to and run `LithoKraft.exe`. The game finds its assets next to the executable (`assets/` and `LithoKraft/assets/`). It writes `config/` (`settings.json`, `input.json`) and `logs/lithokraft.log` in the same folder. To keep config and logs somewhere else, set `LL_DATA_ROOT` to another folder before you launch.

The game opens borderless full screen. Useful options:

| Option | Effect |
|---|---|
| `--display windowed\|maximized\|borderless\|fullscreen`, `--resolution WxH` | Window mode and size. |
| `--renderer vulkan\|gl33\|software` | Choose the renderer. |
| `--component-test` | Start in the Component Test world. |
| `--test-world`, `--test-world-examples`, `--flat-lab`, `--bolt-lab` | Start in another catalog world. |
| `--seed n`, `--dim wafer\|organic` | Seed and dimension for the default world. |
| `--no-save` | Do not save the world on exit. |
| `--version` | Print the version and exit. |

Any unknown option prints the full option list.

## Controls

The defaults follow the Minecraft Java creative layout. You can rebind them in `config/input.json`.

| Action | Keyboard and mouse | Gamepad |
|---|---|---|
| Move | W A S D | Left stick |
| Look | Mouse | Right stick |
| Jump (fly up) | Space | A |
| Sneak (fly down) | Left Shift | B |
| Sprint | Left Ctrl | Left stick press |
| Crawl | C | D-pad right |
| Break | Left click | Right trigger |
| Place, or use a usable block | Right click | Left trigger, X |
| Pick block | Middle click | D-pad up |
| Secondary setting (with right click or the wheel) | Hold Left Alt | |
| Inventory | E | Y |
| Drop | Q | D-pad down |
| Hotbar slots 1 to 10 | 1 to 9, then 0 | |
| Next or previous hotbar slot | Mouse wheel | Bumpers |
| Pause menu | Esc | Start |
| Console | ~ (tilde) | |
| Hide the HUD / hide the controls panel | F1 / F4 | |
| Screenshot | F2 | |
| Debug overlay (shows exact signal strengths) | F3 | Back |
| Save the world | F5 | |
| Switch dimension | F6 | |
| Cycle the tick rate | F7 | |
| Reload resources | F9 | |
| Script editor | F10 | |

Double-tap Jump, or type `fly` in the console, to toggle flight.

The pause menu has Resume, Settings, Save world, Load world, Exit to menu, and Quit to desktop. **Load world** opens the world picker: Default world, Test World, Test World + Examples, Wire Torture, Flat Lab (Trace), Bolt PCHIE Lab, and Component Test. Each world lists its named spawns.

## Inventory and the creative palette

The game is always in creative mode. Press E for the inventory. The left panel is the creative palette, grouped in families such as Trace and Lamps/LEDs. Click an item to pick it up and click a slot to drop it there.

The player inventory has 58 slots: a 16-slot hotbar, two hand slots, a main grid of four rows of eight, and eight worn gear slots (head, chest, legs, feet, arms, face, heart, and mind) inside the magenta equipment frame. Stacks hold up to 512. Shift-click moves a stack between the hotbar and the main grid, topping up matching stacks first and then filling the first empty slot. The selected hotbar slot has a steady purple outline.

The console command `give <itemId> [count]` also adds items.

## Trace and components

Trace is the game's wiring, in red and blue. The two colours never connect, so they can run side by side. Signal strength runs from 0 to 64 and drops by 1 per block along flat Trace. Lamps light from the strongest signal next to them. Exact numbers show in the F3 overlay. Full rules, including soft and hard power, are in [Signal rules, inventory, and the Component Test world](../components/signal-rules.md).

The components in this build, each with its own spec sheet:

- [Multiplexor](../components/multiplexor.md): a 2:1 mux. It can carry a signal from one colour to the other.
- [Dialor](../components/dialor.md): a 0 to 64 signal source. Set the dial with Alt and the mouse wheel.
- [Via](../components/via.md): red and blue insulated tubes that carry Trace up, down, and sideways with no loss.
- [Watchor](../components/watchor.md): an eye that pulses 64 out of its back for one tick when the block in front of it changes.
- [Bitor](../components/bitor.md): bitwise AND, OR, XOR, or NOR of two strengths, picked by a control strength.
- [Pushor](../components/pushor.md): senses power. Pushing is not implemented yet.
- [Swivelor](../components/swivelor.md): a rotary base that holds a detent. Turning is not implemented yet.
- [Resin](../components/resin.md): an item only, for now. It places nothing.

The build also has Trace itself, lamps and LEDs, Pulsor, Repeator, and Dividor, the halving block used in the Stage B doors.

## The Component Test world

Choose **Component Test** in the world picker, or launch with `--component-test`. It is a flat plain of wafer silicon with no generated terrain. Each component stands at the centre of its own 48 by 48 block pad, with its name spelled out in the floor around it. Each pad has a named spawn, such as `dialor` or `red-via`, so you can jump straight to it from the world picker. Resin's pad shows only its name, because Resin is an item.

## Useful console commands

Open the console with ~. `help` lists every command, and `help <command>` describes one.

| Command | What it does |
|---|---|
| `fly [on\|off]` | Toggle flight. |
| `give <itemId> [count]` | Add items. |
| `world_load <id>` | Load a catalog world, such as `default`, `testworld`, `flatlab`, or `component-test`. |
| `map [dimension]` | Switch dimension, or list them. |
| `tick_freeze`, `tick_step <n>`, `tick_unfreeze` | Freeze the simulation and step it tick by tick. |
| `save`, `quit` | Save now, or quit to the desktop. |
| `screenshot [path]` | Save a screenshot. |
| `mcp grant <perm...>`, `mcp revoke <perm...>` | Grant or revoke permissions for an MCP agent (see below). |

## The harness in the game

The harness controls the game through the engine's own MCP server. The game starts that server when it is launched with `--mcp-stdio`. You do not launch the game yourself for this. `pchie_lk_pilot` from the harness release does it:

```
.\pchie_lk_pilot.exe --out run1 --game <folder>\LithoKraft.exe
```

The pilot then does four things:

1. It decides the maze with Little Nemo (or the null-model driver with `--unit`) on a fresh side chain.
2. It launches the game with `--mcp-stdio --pchie-maze <maze file> --spawn maze`, windowed at 1280 by 720, with its own data folder under `run1/lk_data`.
3. It reads the maze back from the world, block by block, and refuses to move unless that world matches the grid it decided on.
4. It walks each committed move with the player's normal controls, reading the real position back after every step. A wrong move is still carried out; the walk is never retuned to make the game succeed.

The server is deny-by-default, so the agent starts with no permissions. By default the pilot writes the grant line into the game's own startup script, which counts as the launcher's approval. With `--grant manual`, you approve it yourself in the console:

```
mcp grant world_read player_control
```

The pilot waits up to `--grant-timeout` seconds (default 300) for the grant. `--maze stagea` uses the 64 by 64 Stage A fixture. `--renderer` and `--resolution` pass through to the game. The pilot writes `pilot_report.txt`, the engine trace, and the game's log into the `--out` folder.

The vision driver, `pchie_lk_visnav`, launches the game the same way. It also grants `pilot_senses` so Little Nemo can see the live framebuffer; see [Quick start: running the maze](quick-start-maze.md#5-what-you-can-do-with-it).

## Known limits of this build

- Pushor extension, Swivelor turning, and Resin bonding are not implemented. Their spec sheets list what is still open.
- At this build, the game's self-test showed 49 known baseline failures and the harness suite one known async-save failure. None was new at that build.
- Saves from older playtest layouts load. The wider old main inventory is remapped and its extra items spill into free slots rather than being dropped.