// Bolt PCHIE - model-independent branch/KV view (CoW page tables + cursor).
#pragma once

#include <cstdint>
#include "pchie/pchie_kv_pools.h"

namespace pchie {

struct ModelBranch {
  KVPagePool::Table swa;
  KVPagePool::Table glob;
  uint32_t n_past = 0;
};

}  // namespace pchie