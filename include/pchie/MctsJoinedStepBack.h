// Bolt PCHIE - MctsJoinedStepBack: DualPool KV step-back and PUCT tree
// snapshot/step-back, explicit and joined.
// Concrete final. Coordinator-thread only: no mutex, no atomic (phase-boundary use).
//
// Two operations, named so they cannot be confused:
//  * kv_snapshot / kv_reposition (per PUCT node id): KV page refs + n_past + scratch KV
//    chain head ONLY. The PUCT tree is NOT touched. This is the in-search move "go to the
//    parent's KV so the leaf token can be forwarded" (what MctsDualPoolStepBack did).
//  * checkpoint / step_back (per caller label): the PUCT node table, the KV branch and the
//    scratch KV chain head, captured and restored together. step_back also invalidates
//    every per-node KV snapshot whose node id is at or above the restored tree size, so a
//    reused node id can never roll back to the KV of a discarded node.
// Scratch chain only: this class never
// takes, reads, or rewinds the official chain, and never calls IHashChain::restore_head on
// it. Official records stay append-only: records appended to the official chain by work
// that is later stepped back remain on it. No step-back marker record is written.
// Fail closed: checkpoint refuses while any node carries virtual loss (not a phase
// boundary); step_back refuses on an unknown label or a task-id mismatch, and in those
// cases changes nothing.
#pragma once

#include "pchie/IHashChain.h"
#include "pchie/MctsDualPoolStepBack.h"
#include "pchie/MctsSearch.h"
#include "pchie/ModelBranch.h"
#include "pchie/PuctTreeCheckpoint.h"
#include "pchie/pchie_kv_pools.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

class MctsJoinedStepBack final {
 public:
  MctsJoinedStepBack(KVPagePool& swa, KVPagePool& glob) : node_kv_(swa, glob), ckpt_kv_(swa, glob) {}

  MctsJoinedStepBack(const MctsJoinedStepBack&) = delete;
  MctsJoinedStepBack& operator=(const MctsJoinedStepBack&) = delete;

  // ---- KV only (tree untouched) ----
  void kv_snapshot(uint32_t node_id, const ModelBranch& b, const IHashChain& kv_chain) {
    node_kv_.snapshot(node_id, b, kv_chain);
  }
  bool kv_reposition(uint32_t node_id, ModelBranch& out, IHashChain& kv_chain) {
    return node_kv_.rollback(node_id, out, kv_chain);
  }

  // ---- joined: tree + KV + scratch KV chain head (never the official chain) ----
  bool checkpoint(uint32_t label, const MctsSearch& s, const ModelBranch& b, const IHashChain& kv_chain,
                  std::string& err) {
    err.clear();
    const auto& nodes = s.puct().nodes;
    if (nodes.empty()) { err = "empty tree"; return false; }
    for (const PuctNode& n : nodes)
      if (n.vl != 0) { err = "virtual loss in flight (not a phase boundary)"; return false; }
    if (label >= ckpts_.size()) ckpts_.resize(size_t(label) + 1);
    PuctTreeCheckpoint& c = ckpts_[label];
    c = PuctTreeCheckpoint{};
    c.task_id = s.task_id();
    c.nodes = nodes;
    c.digest = s.digest();
    ckpt_kv_.snapshot(label, b, kv_chain);
    c.valid = true;
    return true;
  }

  bool step_back(uint32_t label, MctsSearch& s, ModelBranch& out, IHashChain& kv_chain, std::string& err) {
    err.clear();
    if (label >= ckpts_.size() || !ckpts_[label].valid) { err = "unknown checkpoint"; return false; }
    const PuctTreeCheckpoint& c = ckpts_[label];
    if (s.task_id() != c.task_id) { err = "task id mismatch"; return false; }
    if (!ckpt_kv_.store().find(label)) { err = "checkpoint KV missing"; return false; }
    // Digest-before-mutate (item 3, 2026-11-11): verify the checkpointed node table against its
    // recorded digest BEFORE the KV rollback or tree restore, so a corrupt checkpoint fails closed
    // and changes nothing (the class contract). PuctSearch::digest() covers nodes only, so this is
    // exactly the post-restore digest; hash-neutral.
    { PuctSearch probe; probe.nodes = c.nodes; if (probe.digest() != c.digest) { err = "checkpoint tree digest mismatch (refused before restore)"; return false; } }
    // Validation done; restore tree + KV + scratch KV chain head. The official chain is not
    // a parameter and is never touched here.
    if (!ckpt_kv_.rollback(label, out, kv_chain)) { err = "kv rollback"; return false; }  // not reachable
    s.puct().nodes = c.nodes;
    node_kv_.store().invalidate_from(uint32_t(c.nodes.size()));
    if (s.digest() != c.digest) { err = "tree digest mismatch after restore"; return false; }
    ++step_backs_;
    return true;
  }

  void drop_checkpoint(uint32_t label) {
    if (label < ckpts_.size() && ckpts_[label].valid) {
      ckpts_[label] = PuctTreeCheckpoint{};
      ckpt_kv_.store().invalidate(label);
    }
  }
  bool has_checkpoint(uint32_t label) const { return label < ckpts_.size() && ckpts_[label].valid; }
  size_t step_backs() const noexcept { return step_backs_; }

  DualPoolForkSnapshotStore& node_kv() { return node_kv_.store(); }
  const DualPoolForkSnapshotStore& node_kv() const { return node_kv_.store(); }

 private:
  MctsDualPoolStepBack node_kv_;  // per PUCT node id
  MctsDualPoolStepBack ckpt_kv_;  // per checkpoint label
  std::vector<PuctTreeCheckpoint> ckpts_;
  size_t step_backs_ = 0;
};

}  // namespace pchie
