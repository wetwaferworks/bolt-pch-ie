// Bolt PCHIE - epoch re-root at real fork and resume points.
// Harness code calls this right after IHashChain::restore_head. For a ratcheted chain it opens a new epoch
// whose commitment names the event and where it happened, so a forked or resumed line never verifies as the
// same straight line. For every other chain (v1 SHA-256, FNV, trace wrappers) it does nothing, so v1 chains
// and every existing run hash are untouched. The ratchet itself stays opt-in.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include "pchie/IHashChain.h"
#include "pchie/RatchetHashChain.h"
#include "pchie/Sha256.h"

namespace pchie {

// detail = SHA256("pchie.epoch.detail" | lp(site) | u64be(where...)). Returns true when a re-root happened.
inline bool reroot_if_ratcheted(IHashChain* chain, const std::string& event, const std::string& site, std::initializer_list<uint64_t> where) {
  auto* r = dynamic_cast<RatchetHashChain*>(chain ? chain->official_chain() : nullptr);
  if (!r) return false;
  std::vector<uint8_t> d; ratchet_detail::put_str(d, "pchie.epoch.detail"); ratchet_detail::put_lp(d, site);
  for (uint64_t w : where) ratchet_detail::put_u64(d, w);
  r->reroot(event, r->epoch_run_id(), Sha256::hash(d.data(), d.size()));
  return true;
}

}  // namespace pchie
