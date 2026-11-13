# Reports

Dated lab write-ups from November 2026 are not part of this repository. Scientific findings that should travel with the
repository are summarized here and in the researcher guide, especially
[Results so far](../guide/results.md). Official pins live in the tests; this
page does not change their hashes or labels. Hashes quoted from November 2026
are provisional unless a test or the results page identifies them as a frozen
Official pin.

## Official pins

Official B1 search, with the loop guard on or off (the walk does not loop):

- Actions SHA-256: `0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e`
- Chain head: `de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533`
- Official B1 `H_world`: `457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5`

The default recomputable ratchet-head pin and the v1 SHA-256 golden used in
the ratchet and TESLA tests also stay put. Thread-count checks at 1, 2, 4, 8,
and 16 workers produced identical actions, chain heads, tree digests, and
(where present) tuple contents.

## Findings that held up

- At budget 16 on B1, L1 did not beat a null-value control
  (`NO_SEPARATION_SAME_WALK`). The loop guard broke an earlier two-cell bounce;
  both arms then cycled over three cells and did not reach the exit.
- Query-before-plan engaged the store and sent hits through the gate, with a
  small saving in model forwards, but the committed walk was unchanged
  (`L2_ENGAGED_WALK_UNCHANGED`).
- The Stage B standard pair with real Little Nemo is an honest stall at door 1,
  the same shape as the official stall record. Run 1 matches run 2. The
  official record was not retuned.
- A flags-off official-token compare on the mutex-free tree reported 98/98
  identity with the post-gate-split baseline.
- When the loop guard became the default, walks that used to loop changed
  (provisional new measurements, not a rewrite of official records).
- Visnav lab work recorded stalls, a way-shut loop-breaker, reach experiments,
  a NEML ledger, and a labeled CUDA baseline. None of those are official goldens.
- Hardware attestation and transparency-anchoring design are documented in the
  researcher guide. No live witness data is in this tree.

