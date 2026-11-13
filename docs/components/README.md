# LithoKraft component reference

This reference describes LithoKraft, the game, and LithoLattice, the engine it runs on, as of the 12 November 2026 playtest build (America/New_York). It is written from the game and engine source and tests of that tree, and from the notes that record that package. It does not invent a feature, a motion, or a number that is not there.

LithoLattice is the engine (one word, both Ls capital). LithoKraft is the game (spelled with a K). Component names in this family keep the -or spelling: Multiplexor, Dialor, Watchor, Bitor, Pushor, Swivelor.

A PCHIE-side account of the vendored public snapshot and the MCP pilot, which is an earlier and separate copy of the engine and game, is in the [LithoKraft and LithoLattice](../guide/lithokraft-integration.md) guide page. Do not assume that every block described here is present in that snapshot.

Artwork on the spec sheets is by CloudAtlas. Each still shows one approved state, named in the caption. An image that shows an extended, turned, or bonded assembly is not a claim that that motion is implemented.

## Pages

- [Signal rules, inventory, and the Component Test world](signal-rules.md) — Strength 0 through 64, soft and hard power, the 58-slot inventory, and the Component Test pads.
- [Multiplexor](multiplexor.md) — A 2:1 colourless mux, with an optional split mode for independent red and blue lanes.
- [Dialor](dialor.md) — An independent 0 through 64 signal source, set with Alt and the scroll wheel.
- [Via](via.md) — Red and blue insulated tubes that carry Trace losslessly.
- [Watchor](watchor.md) — An eye that pulses 64 out the back for one tick when the block ahead changes.
- [Bitor](bitor.md) — Bitwise AND, OR, XOR, or NOR of two strengths, selected by a control range.
- [Pushor](pushor.md) — Reads soft and hard power; extension, spit, and retract are still open.
- [Swivelor](swivelor.md) — A rotary base that holds a detent; turning is not implemented.
- [Resin](resin.md) — An item only (the pot). It places nothing, and nothing in the sim reads it yet.

## The 12 November playtest build

The 12 November playtest build is a packaged LithoKraft playtest. The executable's SHA-256 is `C0D3A18BE5FAD0A722121E8AA9859DB0B5BEBD71A67B59541E3824A925056A1F` (20,596,736 bytes, built at 11:02:08 AM ET, America/New_York). The game's `mod.json` names LithoKraft version `0.1.0` and depends on LithoLattice `>=0.1.0`.

That package carries four playtest fixes on top of the previous playtest build:

1. **Creative-inventory flicker.** The cause was pinned in code: in the default three-way render layout, HUD streams were borrowed pointers, and a rebuild on the main thread could race the render thread, so the static GUI could drop for a frame. The fix snapshots those streams into storage the job owns. A headless harness test covers the snapshot and the in-place rebuild hazard. The live Vulkan timing was not proven headless; it still wants an in-game check that carrying a stack over the left panel no longer flashes.
2. **Selected hotbar outline.** Colour `#8A2EFF` (138, 46, 255), 3 screen pixels, opaque and steady, on the selected slot only. Unselected slots keep the thin dark border. The component readout colour `#A970FF` and the Bitor lamps were not changed.
3. **Shift-click quick move.** Shift plus a press, while not carrying, moves between the hotbar (slots 0-15) and the main inventory (18-49): first topping up same-identity stacks in the destination, then the remainder into the first empty slot. Hands and gear are unchanged by Shift. A full destination moves nothing. The creative palette's Shift-click is unchanged.
4. **Agent equipment frame.** The magenta frame encloses the worn slots and the figure, with equal padding on all four sides. The 4-by-8 main grid and the hotbar are not wrapped in that frame.

Tests recorded for that build, America/New_York:

- Self-test: 1789 pass, 49 fail. The 49 failures are the same baseline set as before; none were new, and none went away.
- Harness: 175 pass, 1 fail. The failure is a known async-save case (the player restores at x=200).
- Items self-test: 0 failures.

It keeps the previous build's 58-slot inventory layout; there was no save-format change. A warning about loading earlier package layouts still applies. An older build's wider 90-slot mains are remapped on load: the first 32 main slots keep their place, and the rest spill into free slots rather than being dropped.
