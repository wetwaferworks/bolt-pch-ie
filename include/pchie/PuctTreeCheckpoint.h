// Bolt PCHIE - one joined checkpoint record: PUCT node table + tree digest.
// Plain record (one per file). The matching KV pages and scratch KV chain head live in
// MctsJoinedStepBack's checkpoint DualPool store under the same label. There is no
// official chain head here: the joined step-back step-back is scratch-chain only.
#pragma once

#include "pchie/pchie_mcts.h"

#include <cstdint>
#include <vector>

namespace pchie {

struct PuctTreeCheckpoint {
  bool valid = false;
  uint32_t task_id = 0;
  std::vector<PuctNode> nodes;       // full node table (PUCT nodes are small; budget-bounded)
  uint64_t digest = 0;               // PuctSearch::digest() at checkpoint (verified on restore)
};

}  // namespace pchie
