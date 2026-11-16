# Provenance

The original Personal Control Harness is public at
https://github.com/JaydeeWetwork/PersonalControlHarness
and is no longer copied into this repository.

That public tree holds the older LithoLattice engine and the LithoKraft visualization environment. This repository does not include a copy of that source, and the root CMake build does not compile it.

The Bolt host seam (`include/pchie/pchie_host.h`) is new clean-room code. It uses its own
Timing Handshake clock (K ticks per token) and a flat world. It does not use procedural
world generation, and it does not link the engine's procedural-generation module.

No code was taken from the private engine, the Threading Handshake, the LithoRoot rewrites, or from later private engine trees.
