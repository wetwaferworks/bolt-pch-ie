// Bolt PCHIE - MctsSearch: thin wrapper around existing PuctSearch.
// Concrete final. Coordinator-thread only: no mutex, no atomic.
// Does not rewrite PUCT math. Does not own the model or KV pools.
// Chain records are coordinator-only; payloads are opaque (an open review question).
#pragma once

#include "pchie/ChainRecord.h"
#include "pchie/CommitSelector.h"
#include "pchie/IHashChain.h"
#include "pchie/pchie_mcts.h"

#include <cstdint>
#include <vector>

namespace pchie {

class MctsSearch final {
 public:
  MctsSearch() { reset(); }

  void reset(uint64_t root_seed = 0, uint32_t task_id = 0) {
    puct_.reset();
    puct_.nodes[0].seed = root_seed;
    task_id_ = task_id;
  }

  uint32_t task_id() const noexcept { return task_id_; }
  PuctSearch& puct() { return puct_; }
  const PuctSearch& puct() const { return puct_; }

  void expand(uint32_t node, std::vector<PuctEdge> edges, IHashChain* chain) {
    if (node >= puct_.nodes.size()) return;
    puct_.expand(node, std::move(edges));
    if (!chain) return;
    std::vector<uint8_t> p;
    put_u32(p, task_id_);
    put_u32(p, node);
    put_u32(p, uint32_t(puct_.nodes[node].edges.size()));
    for (const PuctEdge& e : puct_.nodes[node].edges) {
      put_i32(p, e.token);
      put_u32(p, e.prior_q20);
    }
    append(chain, ChainRecordKind::Expansion, std::move(p));
  }

  bool select(PuctPick& out, IHashChain* chain) {
    // PuctPick's leaf/parent/token have no default; PuctSearch::select does not
    // clear out.terminal on a new leaf. Always value-initialize first.
    out = PuctPick{};
    if (puct_.nodes.empty() || !puct_.nodes[0].expanded) return false;
    if (!puct_.select(out)) {
      out = PuctPick{};
      return false;
    }
    if (chain) {
      std::vector<uint8_t> p;
      put_u32(p, task_id_);
      put_u32(p, out.parent);
      put_u32(p, out.leaf);
      put_i32(p, out.token);
      put_u32(p, out.terminal ? 1u : 0u);
      append(chain, ChainRecordKind::Selection, std::move(p));
    }
    return true;
  }

  void record_leaf(uint32_t leaf, uint32_t value_q20, bool gate_pass,
                   int32_t last_reject, IHashChain* chain) {
    if (!chain) return;
    std::vector<uint8_t> v;
    put_u32(v, task_id_);
    put_u32(v, leaf);
    put_u32(v, value_q20);
    append(chain, ChainRecordKind::LeafValue, std::move(v));
    std::vector<uint8_t> g;
    put_u32(g, task_id_);
    put_u32(g, gate_pass ? 1u : 0u);
    put_i32(g, last_reject);
    append(chain, ChainRecordKind::GateVerdict, std::move(g));
  }

  void backup(const PuctPick& p, uint32_t value_q20, IHashChain* chain) {
    for (uint32_t a : p.path) {
      if (a >= puct_.nodes.size()) return;
    }
    puct_.backup(p, value_q20);
    if (chain) {
      std::vector<uint8_t> b;
      put_u32(b, task_id_);
      put_u32(b, p.leaf);
      put_u32(b, value_q20);
      append(chain, ChainRecordKind::Backup, std::move(b));
    }
  }

  void mark_terminal(uint32_t node, uint32_t value_q20) {
    if (node >= puct_.nodes.size()) return;
    puct_.mark_terminal(node, value_q20);
  }

  std::vector<int32_t> commit(IHashChain* chain) const {
    std::vector<int32_t> path;
    if (puct_.nodes.empty()) return path;
    const CommitSelector sel(puct_.cfg.commit_temp);
    uint32_t cur = 0;
    const size_t cap = puct_.nodes.size();
    for (size_t step = 0; step < cap; ++step) {
      if (cur >= puct_.nodes.size() || !puct_.nodes[cur].expanded) break;
      const int ei = sel.Select(puct_, cur);
      if (ei < 0) break;
      const std::vector<PuctEdge>& edges = puct_.nodes[cur].edges;
      if (size_t(ei) >= edges.size()) break;
      path.push_back(edges[size_t(ei)].token);
      const uint32_t nxt = edges[size_t(ei)].child;
      if (nxt == 0xFFFFFFFFu) break;
      cur = nxt;
    }
    if (chain) {
      std::vector<uint8_t> p;
      put_u32(p, task_id_);
      put_u32(p, uint32_t(path.size()));
      for (int32_t t : path) put_i32(p, t);
      append(chain, ChainRecordKind::Commit, std::move(p));
    }
    return path;
  }

  uint64_t digest() const noexcept { return puct_.digest(); }

 private:
  static void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
    b.push_back(uint8_t(v >> 16));
    b.push_back(uint8_t(v >> 24));
  }
  static void put_i32(std::vector<uint8_t>& b, int32_t v) {
    put_u32(b, static_cast<uint32_t>(v));
  }
  static void append(IHashChain* chain, ChainRecordKind kind, std::vector<uint8_t> payload) {
    if (!chain) return;
    chain->append(ChainRecord{kind, std::move(payload)});
  }

  PuctSearch puct_;
  uint32_t task_id_ = 0;
};

}  // namespace pchie