#include "pchie/research/FrozenDynamics.h"

#include "pchie/research/Digest.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace pchie {
namespace research {

void DynamicsTrainer::add(const Transition& t) {
  Tally& s = seen_[{t.state, t.action}];
  ++s.next[t.next];
  s.reward_sum += t.reward_q20;
  ++s.count;
}

FrozenDynamics DynamicsTrainer::freeze(const LookaheadConfig& vc) const {
  FrozenDynamics f;
  for (const auto& kv : seen_) {
    DynamicsEntry e;
    uint32_t best = 0;
    for (const auto& n : kv.second.next)
      if (n.second > best) {  // strictly greater keeps the lower state on ties
        best = n.second;
        e.next = n.first;
      }
    e.count = kv.second.count;
    e.reward_q20 = int32_t(kv.second.reward_sum / int64_t(kv.second.count));
    f.table_[kv.first] = e;
  }
  // Offline value iteration (Gauss-Seidel in key order, integer, so it is the same everywhere).
  for (const auto& kv : f.table_) {
    f.values_[kv.first.first] = 0;
    f.values_[kv.second.next] = 0;
  }
  for (uint32_t sweep = 0; sweep < 4096; ++sweep) {
    bool changed = false;
    for (auto& v : f.values_) {
      int64_t best = 0;  // an unknown or terminal state is worth nothing more
      bool any = false;
      for (uint32_t a = 0; a < vc.actions; ++a) {
        auto it = f.table_.find({v.first, uint8_t(a)});
        if (it == f.table_.end() || it->second.next == v.first) continue;
        const DynamicsEntry& e = it->second;
        // A rewarding step ends the episode (the exit), so nothing is added after it.
        const int64_t tail = e.reward_q20 >= (1 << 20) ? 0 : f.values_[e.next] * vc.discount_num / vc.discount_den;
        const int64_t q = int64_t(e.reward_q20) - vc.step_cost_q20 + tail;
        if (!any || q > best) best = q;
        any = true;
      }
      if (best < 0) best = 0;
      if (best != v.second) {
        v.second = best;
        changed = true;
      }
    }
    f.value_sweeps_ = sweep + 1;
    if (!changed) break;
  }
  f.sha_ = sha256_hex(f.canonical());
  return f;
}

std::string FrozenDynamics::canonical() const {
  std::ostringstream o;
  o << "frozen_dynamics.v1|" << table_.size();
  for (const auto& kv : table_)
    o << '|' << kv.first.first << ',' << int(kv.first.second) << ',' << kv.second.next << ',' << kv.second.count << ','
      << kv.second.reward_q20;
  o << "|values|" << values_.size();
  for (const auto& kv : values_) o << '|' << kv.first << ',' << kv.second;
  return o.str();
}

bool FrozenDynamics::knows(uint64_t state) const {
  auto it = table_.lower_bound({state, uint8_t(0)});
  return it != table_.end() && it->first.first == state;
}

bool FrozenDynamics::from_canonical(const std::string& text, const std::string& pin, FrozenDynamics& out,
                                    std::string& err) {
  std::string body = text;
  while (!body.empty() && (body.back() == '\n' || body.back() == '\r')) body.pop_back();
  if (sha256_hex(body) != pin) {
    err = "frozen dynamics do not match their pin";
    return false;
  }
  std::vector<std::string> f;
  size_t s = 0;
  for (size_t i = 0; i <= body.size(); ++i)
    if (i == body.size() || body[i] == '|') {
      f.push_back(body.substr(s, i - s));
      s = i + 1;
    }
  FrozenDynamics d;
  size_t i = 0;
  auto num = [&](const std::string& t, unsigned long long& v) {
    char* e = nullptr;
    v = std::strtoull(t.c_str(), &e, 10);
    return !t.empty() && e && *e == 0;
  };
  unsigned long long n = 0;
  if (f.size() < 2 || f[0] != "frozen_dynamics.v1" || !num(f[1], n)) {
    err = "bad frozen dynamics header";
    return false;
  }
  i = 2;
  for (unsigned long long k = 0; k < n; ++k, ++i) {
    if (i >= f.size()) {
      err = "truncated frozen dynamics";
      return false;
    }
    unsigned long long st = 0, a = 0, nx = 0, c = 0;
    long long r = 0;
    char tail = 0;
    if (std::sscanf(f[i].c_str(), "%llu,%llu,%llu,%llu,%lld%c", &st, &a, &nx, &c, &r, &tail) != 5) {
      err = "bad frozen dynamics entry";
      return false;
    }
    DynamicsEntry e;
    e.next = nx;
    e.count = uint32_t(c);
    e.reward_q20 = int32_t(r);
    d.table_[{st, uint8_t(a)}] = e;
  }
  if (i + 1 >= f.size() || f[i] != "values" || !num(f[i + 1], n)) {
    err = "bad frozen dynamics values";
    return false;
  }
  i += 2;
  for (unsigned long long k = 0; k < n; ++k, ++i) {
    unsigned long long st = 0;
    long long v = 0;
    char tail = 0;
    if (i >= f.size() || std::sscanf(f[i].c_str(), "%llu,%lld%c", &st, &v, &tail) != 2) {
      err = "bad frozen dynamics value";
      return false;
    }
    d.values_[st] = v;
  }
  d.sha_ = sha256_hex(d.canonical());
  if (d.sha_ != pin) {
    err = "frozen dynamics did not round-trip";
    return false;
  }
  out = std::move(d);
  return true;
}

bool FrozenDynamics::predict(uint64_t state, uint8_t action, DynamicsEntry& out) const {
  auto it = table_.find({state, action});
  if (it == table_.end()) return false;
  out = it->second;
  return true;
}

int64_t FrozenDynamics::leaf_value(uint64_t state) const {
  auto it = values_.find(state);
  return it == values_.end() ? 0 : it->second;
}

int64_t FrozenDynamics::step_value(uint64_t state, uint32_t a, uint32_t depth, const LookaheadConfig& cfg,
                                   uint64_t& nodes) const {
  ++nodes;
  DynamicsEntry e;
  if (!predict(state, uint8_t(a), e)) return cfg.untried_value_q20;
  if (e.next == state && e.reward_q20 <= 0) return int64_t(e.reward_q20) - cfg.step_cost_q20 * 2;  // a bump
  if (e.reward_q20 >= (1 << 20)) return int64_t(e.reward_q20) - cfg.step_cost_q20;                 // the exit
  const int64_t tail = depth == 0 ? leaf_value(e.next) : value(e.next, depth, cfg, nodes);
  return int64_t(e.reward_q20) - cfg.step_cost_q20 + tail * cfg.discount_num / cfg.discount_den;
}

int64_t FrozenDynamics::value(uint64_t state, uint32_t depth, const LookaheadConfig& cfg, uint64_t& nodes) const {
  if (depth == 0) return leaf_value(state);
  int64_t best = INT64_MIN;
  for (uint32_t a = 0; a < cfg.actions; ++a) {
    const int64_t v = step_value(state, a, depth - 1, cfg, nodes);
    if (v > best) best = v;
  }
  return best;
}

LookaheadChoice FrozenDynamics::best_action(uint64_t state, const LookaheadConfig& cfg) const {
  LookaheadChoice c;
  const uint32_t rest = cfg.depth > 0 ? cfg.depth - 1 : 0;
  for (uint32_t a = 0; a < cfg.actions; ++a) {
    const int64_t v = step_value(state, a, rest, cfg, c.nodes);
    if (c.action < 0 || v > c.value_q20) {  // strictly greater: lower action id wins ties
      c.action = int(a);
      c.value_q20 = v;
    }
  }
  return c;
}

}  // namespace research
}  // namespace pchie
