# Bolt PCHIE Roadmap

1. Spine: variation tree, CoW KV, deterministic Q4 dot, thread-invariant rollouts. Done.
2. Engine host: Timing Handshake (generation mode with K ticks per token, and scrub mode that replays the log) on a flat world. Done.
3. C++ inference server (vLLM-style): GGUF loader, paged KV, scheduler, canonical batching.
4. GPU: CUDA (fixed-order math, `--fmad=false`), plus a compute-shader fallback.
5. Robot sim and visuals: a variation-tree scrubber.
6. Ship: leak check, credits, public release.

## Later

- LithoTrain: character animation and IK bone rigging (later step; source TBD to be decided; nothing is imported from the private engine)
