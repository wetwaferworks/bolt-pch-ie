# Bolt PCHIE: CPU spine

Build, CPU only:

    cmake -S . -B build && cmake --build build --config Release -j 4
    ctest --test-dir build -C Release --output-on-failure

The gate covers several things:

- The variation tree. Each record carries `parent_node_id` and `step_seed`. `step_back` keeps the active fork.
- The copy-on-write paged KV cache. `release_block_table` returns pages to a sorted free list, so the lowest page id is always allocated first.
- The pinned rounding rule. Rounding is round-half-to-even through `round_half_even_f32` (see `pchie_q4_det.h`).
- Q4_0 x Q8_0 dot products. Each block uses an exact int32 sum, and blocks are added in a fixed sequential fp32 order with FMA contraction off.
- Thread invariance. 64 independent fork rollouts are statically partitioned, and the final digest must be identical at 1, 2 and 16 threads across 3 repeated runs.
