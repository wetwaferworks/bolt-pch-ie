// L1. Model-driven PUCT — leaf value / prior source contracts.
// Default (flags off) is the current oracle. Official decide hashes must never
// report a model-value result as official. Results under the flag are "L1 scratch".
#pragma once

#include "pchie/rl/RlCommon.h"
#include "pchie/rl/RlFlags.h"

#include <functional>
#include <string>
#include <vector>

namespace pchie {
namespace rl {

constexpr uint64_t kLeafSrcOracle = 1;
constexpr uint64_t kLeafSrcModel = 2;
constexpr uint64_t kLeafSrcNull = 3;

struct LeafValue {
  int32_t q20 = 0;
  uint64_t source_tag = kLeafSrcOracle;
};

// Standalone leaf context for L1. This does not replace official LeafContext
// (PchCpuLeafPath). Hook wiring into PchStack is deferred (see PchStackRlHooks.h).
struct RlLeafContext {
  uint32_t task_id = 0;
  std::vector<int32_t> obs_tokens;  // untrusted; already tokenize_untrusted
  int32_t bfs_distance = -1;        // oracle only; must be ignored by Model/Null
  void* kv_branch = nullptr;        // opaque; unused in contract tests
};

class ILeafValueSource {
 public:
  virtual ~ILeafValueSource() = default;
  virtual LeafValue value(const RlLeafContext& ctx) = 0;
  virtual const char* name() const = 0;
};

// Current behaviour (default). Uses 1/(1+BFS distance) quantized to Q20.
class OracleLeafValue final : public ILeafValueSource {
 public:
  LeafValue value(const RlLeafContext& ctx) override {
    called_ = true;
    LeafValue v;
    v.source_tag = kLeafSrcOracle;
    if (ctx.bfs_distance < 0) {
      v.q20 = 0;
      return v;
    }
    const double d = 1.0 / (1.0 + double(ctx.bfs_distance));
    int32_t q = int32_t(d * double(kQ20) + 0.5);
    if (q < 0) q = 0;
    if (q > kQ20) q = kQ20;
    v.q20 = q;
    return v;
  }
  const char* name() const override { return "oracle"; }
  bool called() const { return called_; }
  void clear_called() { called_ = false; }

 private:
  bool called_ = false;
};

// L1: Little Nemo verdict-token mass (closer / farther / stuck), already Q20 and
// normalized by the caller (or normalized here). No BFS, get_block, or world map.
class ModelLeafValue final : public ILeafValueSource {
 public:
  // closer + farther + stuck should sum to kQ20; leftover is ignored.
  void set_verdict_mass(int32_t closer_q20, int32_t farther_q20, int32_t stuck_q20) {
    closer_ = closer_q20;
    farther_ = farther_q20;
    stuck_ = stuck_q20;
  }
  LeafValue value(const RlLeafContext& ctx) override {
    (void)ctx;  // observation tokens are the model's input; mass is injected here
                // because the real tokenizer/forward is not on this box pass.
    LeafValue v;
    v.source_tag = kLeafSrcModel;
    int64_t sum = int64_t(closer_) + int64_t(farther_) + int64_t(stuck_);
    if (sum <= 0) {
      v.q20 = 0;
      return v;
    }
    // Scalar value = mass on "closer" after normalize. Farther/stuck contribute 0.
    v.q20 = int32_t((int64_t(closer_) * int64_t(kQ20)) / sum);
    return v;
  }
  const char* name() const override { return "model"; }

 private:
  int32_t closer_ = 0, farther_ = 0, stuck_ = 0;
};

// Uniform constant control. An L1 "solve" counts only if the model run beats this.
class NullLeafValue final : public ILeafValueSource {
 public:
  explicit NullLeafValue(int32_t q20 = kQ20 / 2) : q20_(q20) {}
  LeafValue value(const RlLeafContext& ctx) override {
    (void)ctx;
    return LeafValue{q20_, kLeafSrcNull};
  }
  const char* name() const override { return "null"; }

 private:
  int32_t q20_;
};

// Poison oracle used by the L1 honesty test. If this is called under --rl-puct-model,
// the test fails: the model path must not reach the BFS oracle.
class PoisonOracleLeafValue final : public ILeafValueSource {
 public:
  std::function<void()> on_call;
  LeafValue value(const RlLeafContext& ctx) override {
    (void)ctx;
    called_ = true;
    if (on_call) on_call();
    return LeafValue{0, kLeafSrcOracle};
  }
  const char* name() const override { return "poison-oracle"; }
  bool called() const { return called_; }

 private:
  bool called_ = false;
};

enum class PriorSource : uint8_t { CurrentMix = 0, ModelLogits = 1 };

inline PriorSource prior_source_from_flags(const RlFlags& f) {
  return (f.puct_model || f.puct_prior == PriorSrc::Model) ? PriorSource::ModelLogits
                                                           : PriorSource::CurrentMix;
}

// Factory: flags off => Oracle. Does not take ownership of a caller-supplied model
// mass; the caller sets ModelLeafValue::set_verdict_mass after construction.
inline ILeafValueSource* make_leaf_value_source(const RlFlags& f, OracleLeafValue* oracle,
                                                ModelLeafValue* model, NullLeafValue* null_src) {
  if (f.puct_value == ValueSrc::Null && !f.puct_model) return null_src;
  if (f.puct_model || f.puct_value == ValueSrc::Model) return model;
  if (f.puct_value == ValueSrc::Null) return null_src;
  return oracle;
}

// Gate fail-closed helper (L1 test: illegal-move edge keeps prior 0).
// A local stand-in until ProposalTokens from the later gate-split header is
// available on this box (snapshot of PchGate is pre-ProposalTokens).
struct PriorEdge {
  int32_t token = 0;
  uint32_t prior_q20 = 0;
  bool fail_closed = false;
};

inline void apply_gate_fail_closed(const std::vector<int32_t>& reject_ids, std::vector<PriorEdge>& edges) {
  for (PriorEdge& e : edges) {
    for (int32_t r : reject_ids) {
      if (e.token == r) {
        e.prior_q20 = 0;
        e.fail_closed = true;
        break;
      }
    }
  }
}

}  // namespace rl
}  // namespace pchie
