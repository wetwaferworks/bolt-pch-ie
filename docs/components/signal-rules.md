# Signal rules, inventory, and the Component Test world

## Signal levels

Trace is LithoKraft's wiring. There are two colours, red and blue. They never connect to each other and can run side by side. Strengths are integers from 0 through 64. The constant in the code is `kMaxSignal = 64`. Dialor's 65 positions (0 through 64) are why some ranges, such as Bitor's last op band, are one step wider than the others.

On flat Trace, strength drops by 1 per block along a run, from every source, in a bucket BFS. A Pulsor pulse travels the connected run without decay, up to 64 blocks. Lamps light from the strongest adjacent signal and terminate it.

Exact numbers show in the look-at debug overlay. On the wire art, four center dots each light at every 16 levels (16, 32, 48, 64). Red dots are round; blue dots are diamonds.

Trace needs support: it lies flat on top of a block or hangs under a ceiling. It does not float, and it pops off when its support is removed. Staircasing up and down is allowed. A solid, opaque, or insulating block over the lower step cuts the stair; transparent and non-solid blocks, including glass and frosted glass, do not. Glass and frosted glass can carry Trace on top but are never powered. Ceramic is an opaque insulator and is never powered.

## Soft power and hard power

**Hard power** is a component output driving straight into a Trace face, or a lit torch under a block (which hard-powers at full strength 64). By default, any component's steady output that faces a solid powerable block hard-powers that block, not only a Repeator. `hardPower` returns 0 when the block is not hard-powered, otherwise the strength. Hard power carries that strength.

**Soft power** is the weaker neighbour case. By default, a lit flat Trace soft-powers every solid powerable block it touches. Soft power wakes components around those blocks. A Pushor that is hard powered also soft-powers adjacent Pushors on all six world axes; a Pushor powered only that way does not pass it on (quasi-connectivity).

Both rules can be turned off with environment switches for comparison (`LL_HARD_ANY=0`, `LL_SOFT_POWER=0`). In the 12 November playtest build both are on.

Vias are lossless: when a signal enters or leaves a via, strength does not drop. Vertical via links are same-colour, straight up and down. Horizontal links use a side-lying via whose axis matches the step. See [Via](via.md).

Component outputs are hard power and, unless a block says otherwise, colourless: they can drive red or blue Trace.

## Inventory

The live layout is 58 flat slots:

| Region | Flat indexes | Count |
|---|---|---|
| Hotbar | 0-15 | 16 |
| Hands (left and right off-hand) | 16-17 | 2 |
| Main inventory | 18-49 | 32 (four rows of eight) |
| Gear (worn, not counted as carry slots) | 50-57 | 8: head, chest, legs, feet, arms, face, heart, mind |

That is 50 carry slots plus 8 gear slots. The maximum stack is 512, equal to the tick-rate number used as the global cap. Per-item caps are `min(ItemDef::maxStack, 512)`. Named or certificate-bearing stacks only merge with stacks whose data is identical.

The 12 November playtest build keeps this layout; there was no save-format change. Older saves are remapped section by section. An 8-slot-hotbar save (50 entries) lands on the current gear and main slots. An older 90-slot save (16 hotbar, 64 main, 8 gear) keeps hotbar, hands, and main 0 through 31; main 32 through 63 spill into free slots and are not silently dropped; gear lands on 50 through 57.

Shift-click quick move, as of the 12 November playtest build, runs between the hotbar and the main inventory only, and only while not carrying: merge first, then the first empty slot. Hands and gear ignore Shift and take a normal press.

## Component Test world

`--component-test`, or "Component Test" in the world picker, loads a dead-flat wafer-silicon plain (dimension `componenttest`) with no procedural generation, including in far chunks. Each Trace component gets one 3-by-3-chunk pad (48 by 48 blocks) on a chunk-aligned grid, with a one-chunk (16-block) aisle between pads. The component stands on the silicone at the pad centre. Its name is set flush in the floor in Ceramic, in a 3-by-5 block font, along a square spiral around the component, readable from the aisle.

The pad list starts from the creative palette's Trace and Lamps/LEDs families. Palette-hidden placement variants (the side Vias) are skipped. Registry display names are uppercased, so a new Trace block joins the world when it is registered. Approved names that were added as newer components — Watchor, Bitor, Pushor, Swivelor, and Resin — get pads and spirals even when the centre would otherwise be empty. Resin's pad is name-only: Resin is an item, never a block. There is no Mortor pad (no concept was picked) and no Via elbow pad (elbows are auto-built at a junction, not placed).

Named spawns exist for the world spawn and for each pad (for example `dialor`, `red-via`). The world spawn is on the north aisle, mid-grid, looking at the first pad row.
