# Bolt PCHIE search rulings (Archie, 2026-10-29)

Status: locked. These rulings answer five design questions about the PUCT variation search.

## 1. Leaf value function V(s_L)
Use a bounded, length-normalized log-probability plus the harness gate, in [0, 1], behind a fixed evaluation interface so a learned value head can replace it later.

V(s_L) = sigmoid( (1 / L^alpha) * sum_{t=1..L} log P(y_t | y_<t) - beta * H(s_L) ) * G(s_L)

- L = rollout depth from the search root; alpha in [0.6, 0.8] (length normalization, Google NMT style).
- H(s_L) = the leaf's next-token predictive entropy (penalizes high-uncertainty cliff-edges).
- G(s_L) in {0, 1} = the deterministic PCH structural/syntax gate.
- Keep a clean `eval_leaf(node, hidden_state)` hook so a linear value projection on the final layer's hidden state can drop in without touching the backup logic.

## 2. End-of-turn tokens
Hard-stop expansion at `<end_of_turn>`, EOS and PCH step delimiters. Any node that emits an end-of-turn or harness fence token is marked `is_terminal = true`. Later hits on that node don't expand; they backpropagate the terminal leaf value right away. Reason: crossing a turn boundary lets the model hallucinate the next user or environment turn, which pollutes Q values and wastes sliding-window KV pages.

## 3. Single-line drilling at batch = 1
Fix first-play urgency (FPU) first, then tune c_puct and the prior temperature.
- Cause: LLM priors are sharply peaked (top-1 often > 0.85, top-2 < 0.05). With FPU = 0 and Q(a1) around 0.7, a sibling's PUCT bonus never catches up.
- Parent-relative FPU reduction: FPU(s) = V(s) - c_fpu * sqrt( sum over visited b of P(s, b) ), with c_fpu around 0.20 to 0.25.
- Prior temperature tau_prior = 1.25, applied before storing P(s, a) on edges. c_puct = 1.5 to 2.0.

## 4. Virtual loss weight
v_loss = 1.0 in normalized Q space, applied to both the visit count and the value sum:

Q_eff(s, a) = ( W(s, a) - v_loss * N_virt(s, a) ) / ( N(s, a) + N_virt(s, a) )

At batch = 1 it clears before the next selection and has no effect, which is why #3 is fixed through FPU. At batch 4, 8 and 16 it deflects co-batched workers away from duplicate KV leaves without pushing lanes into garbage tokens (P < 0.001).

## 5. Per-node seed without sampling
Intended, so keep it. Expansion is deterministic top-K by prior, with no random sampling. The seed, `splitmix64(parent_seed ^ token_id ^ depth)`, ties any pseudo-randomness to the tree coordinate rather than to thread order. It is used for:
- deterministic Gumbel or Dirichlet root exploration noise;
- breaking exact ties in quantized (Q4_K / Q6_K) integer logits;
- commit-time action selection from N(s, a)^(1/T) when T > 0, keeping run hashes bit-identical across 1 to 16 workers.

## Impact on the code
- These change search behavior, so the batch 1/2/4/8 run hashes (including the golden candidate 0x34B0108DFDA94183) will change. Re-baseline with the 100-run gate after they're implemented.
- Order: eval_leaf hook + terminal stop, then FPU/prior temperature/c_puct, then virtual-loss formula, then seed uses.

Executive direction, vision and verification: Jaydee Wetwork. Systems architecture and optimization counsel: Archie (co-architect). Lead engine implementation and empirical profiling: Cortana.