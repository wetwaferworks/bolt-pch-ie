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

### Every component in this build

This list comes from the Play D9 source: the block registry, the behaviour catalog, and the creative palette's Trace and Lamps/LEDs families, in palette order. Each entry says what the block does, which faces it reads and drives, and how you set it. "Front" is the way the block faces when you place it, "back" is the opposite face, and left and right are as seen from behind the block. Where the code has a placeholder instead of a behaviour, the entry says **not implemented yet**.

How settings work: a right-click uses a block only if the block is marked usable; on anything else, a right-click places the block in your hand. Holding Left Alt with a right-click (or Shift with a right-click, on blocks that have a second setting) changes the second setting of any component. Holding Left Alt and turning the wheel steps a 1 to 64 setting up or down; the Routor ignores the wheel. The look-at readout and the F3 overlay show each block's current setting.

**Trace family**

| Component | What it does | Faces | Controls |
|---|---|---|---|
| Red Trace, Blue Trace | The wiring. Strength drops by 1 per block. Red and blue never connect. Trace needs a solid block under it or above it. | Links to neighbouring Trace and to component faces. | None. |
| Red Via, Blue Via | Insulated tubes that carry Trace up, down, and sideways with no loss. Side-lying forms are built automatically at junctions. See [Via](../components/via.md). | Ends link to same-colour vias and Trace. | None. |
| Pulsor | Fires short pulses. With a clock on a side, a rising clock edge fires a pulse at the back's strength. With no clock, a rising edge on the back comes out one tick later as a pulse. A pulse travels a connected Trace run without losing strength (up to 64 blocks) and stays high for 4 ticks. | Back: data. Left, right: clock. Front: out. | None. |
| Memor | Stores a value. A new non-zero value on the back is stored; the front puts it out steadily. A rising edge on either side clears it. The readout shows the value and its character. 8-tick delay. | Back: write. Left, right: clear. Front: stored value. | None. |
| Randor | Rolls a random number. Each rising clock edge rolls 1 to N, where N is the back's strength, and the front holds the roll until the next one. 8-tick delay. | Back: range N. Left, right: clock. Front: the roll. | None. |
| Fusor | Adds: front = back + left + right, capped at 64. | Back, left, right: inputs. Front: sum. | None. |
| Dividor | Halves: front = back ÷ 2, rounded down. 1-tick delay. | Back: input. Front: out. Sides ignored. | None. |
| Repeator | Boosts any input to a full 64 after its delay. A new non-zero value dips the output for one tick, so a clock downstream sees a fresh edge. A powered Repeator pointing into its side locks it. | Back: in. Front: out. Left, right: lock (Repeators only). | Right-click cycles the delay 2, 4, 8, 16, 32, 64 ticks. |
| Thresholdor | Compares. With one reference, it passes the back's value when it is at least the reference; with both, when it is between them; with none, always. Subtract mode puts out back minus the stronger side, never below 0. Read mode also reads the level of the block behind it (a Timer Buffer's count or a Catchor's caught value). 1-tick delay. | Back: input. Left: low reference. Right: high reference. Front: out. | Right-click: compare or subtract. Alt + right-click: read mode on or off. |
| Multiplexor | A 2:1 switch. Control off picks input A, control on picks input B. The output has no colour, so it can move a signal from red to blue. Split mode routes red and blue on separate lanes. See [Multiplexor](../components/multiplexor.md). | Back: A. Right: B. Left: control. Front: out. | None by click; it is switched only by what feeds its control face. |
| Routor | A crossover with two paths that never mix, one tick each. Mode 1: back to left, right to front. Mode 2: back to right, left to front. Mode 3: back to front, and left and right pass across each other. | Back, left, right: per mode. | Alt + right-click cycles the mode. |
| Latchor | Matches a value. On any tick when the back is exactly its set value (1 to 64), the front puts out that value; otherwise 0. | Back: in. Front: out. | Alt + right-click, or Alt + wheel, sets the value. |
| Catchor | Catches one address on a line. When placed, it binds to the first Trace touching a side, and that side and colour become its input. When the input equals its match value, it sends that value out as a one-tick pulse. The line itself carries on. | Input: the bound side. Output: straight on, left, or right. | Right-click cycles the output side. Alt + right-click or Alt + wheel sets the match value (1 to 64). |
| Switch | A lever: a steady 64 while on. On its own it powers every side; mounted on a block, it powers through that block. | All sides, or its support. | Right-click toggles it. |
| Button | A 64 pulse for 10 ticks. | All sides, or its support. | Right-click presses it. |
| Keyboard | A typing terminal. Each key you type leaves as one pulse: the character's code + 1 out the front, Enter (1) and Backspace (2) out the right side. | Front: characters. Right: Enter and Backspace. | Right-click starts and stops typing. |
| Tone Block | Plays a note. The back's strength is the pitch, 1 to 64 semitones up from C2; 0 is silent. It drives nothing. | Back: pitch. | None. |
| Signal Transmittor | A radio sender. When placed, it binds to the colour of the Trace beside it, then broadcasts the back's strength on its channel. It never drives Trace. Its class also has a steady beacon setting on a plain right-click, but the block is not marked usable, so a right-click does not reach it in this build. | Back: in. Out: radio. | Alt + right-click cycles the channel. |
| Signal Receivor | A radio receiver. It puts out what it hears on its channel, on the colour it received, one tick later. | Front: out, on the received colour. | Alt + right-click cycles the channel. |
| Dish | A long-range receiver: it hears much farther than a Signal Receivor. It is a single block; a larger multi-block dish is **not implemented yet**. | Front: out. | Alt + right-click cycles the channel. |
| Overclockor | A radio jammer: with jamming on, it blocks every channel but its own within 64 blocks. Its class turns jamming on with a plain right-click, which this build does not route to it (it is not marked usable). Speeding up ticks is **not implemented yet**. It drives no signal. | None. | Alt + right-click cycles the jam channel. |
| Portor | Marks one face of a chip as an input or an output, in the Trace colour beside it. In the build area it is only a marker and drives nothing; printing a chip is **not implemented yet**. | Reads adjacent Trace (input mode). | Right-click toggles input or output. |
| Breakor | A circuit breaker. It passes the back's signal to the front until the signal goes over its limit, then trips and puts out 0 until it is reset. A Repeator turning on at a side also resets it. | Back: in. Front: out. Left, right: reset (Repeators only). | Right-click resets it. Alt + right-click or Alt + wheel sets the limit (1 to 64, 64 by default). |
| Chipor, Chipor Viewport | Printed-chip blocks. The Viewport shows a still miniature of a circuit through its glass top. Neither one simulates: the tiny circuit does not run (**not implemented yet**). | None. | None. |
| Lithographor | The chip printer's model. It has **no behaviour yet**. | None. | None. |
| Etchor | Holds a name for an etched chip. The etch itself runs from the console: `etch <name>` turns a blank wafer into a Chipor with that name. The Selector, Connector, and Laysor machine around it is **not implemented yet**. | None. | None in the world. |
| Dialor | A 0 to 64 source. The front puts out the dial value unless the back is powered. 1-tick delay. See [Dialor](../components/dialor.md). | Back: inhibit. Front: out. Sides: none. | Alt + wheel sets the dial (wraps 0 and 64). A plain right-click does nothing. |
| Watchor | An eye. When the block in front of it changes, it pulses 64 out of its back for one tick, at most every two ticks. Appear, remove, and sweep cases are **not decided yet**. See [Watchor](../components/watchor.md). | Front: watched cell. Back: pulse. Sides: none. | None. |
| Bitor | Bitwise AND, OR, XOR, or NOR of two strengths. The back's strength picks the operation: 0 to 15 AND, 16 to 31 OR, 32 to 47 XOR, 48 to 64 NOR. Updates two ticks after a change. See [Bitor](../components/bitor.md). | West, east: operands. Back: control. Front: out. | None. |
| Pushor | Senses soft and hard power and lights when powered. Extending, spitting, and retracting are **not implemented yet**. See [Pushor](../components/pushor.md). | Back, sides, top, bottom: power. Front: the pushing face. | None. |
| Swivelor | A rotary base that holds a detent (0, 16, 32, 64). Turning, and the face that sets the detent, are **not implemented yet**; it reads and drives nothing. See [Swivelor](../components/swivelor.md). | None yet. | None. |

**Lamps/LEDs family**

| Component | What it does | Faces | Controls |
|---|---|---|---|
| Red LED, Blue LED | Inverters. An LED is lit unless the block it is mounted on is powered, with a one-tick delay. A lit LED powers its own colour of Trace and the solid block above it. | Reads its support. Drives its colour and the block above. | None. |
| Lamp | Lights from any powered neighbour. Lamps end a signal; they pass nothing on. | All sides. | None. |
| lamp16, lamp32, lamp64 | Colour lamps. The strongest neighbouring signal picks a colour from a 16, 32, or 64 colour set, wrapping around. | All sides. | None. |
| Emitor | The light source of the chip printer. It lights like a lamp when powered; its beam is drawn, not simulated. | All sides. | None. |

**Outside the palette**

- **Timer Buffer** is registered and works (a count from 0 to 64 that ticks down, in loop or once mode), but it is hidden from the palette and the Component Test world until its art is done. You can still place it by id.
- **Resin** is an item only. It places nothing, and nothing in the simulation reads it yet. See [Resin](../components/resin.md).
- **Selector, Connector, Laysor, Input Buffer, Fabor, AutoFabor, and Vior** are registered blocks with no behaviour yet. They are parts of machines that are still to come.

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