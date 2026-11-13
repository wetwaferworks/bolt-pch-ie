// Bolt PCHIE B5 - headless freeze/recovery over the B1 maze.
// Concrete final. Serial coordinator; a deterministic hook stalls one rollout
// slot mid-search. The in-flight PUCT/chain mutation is discarded, the
// iteration is retried, and post-recovery H_bolt must equal the clean run.
// Watch mode / UI and the mutex/CV LithoLattice handshake authority are
// deferred and fail cleanly, not faked. No mutex, no atomic.
#pragma once

#include "pchie/HeadlessMaze.h"
#include "pchie/GauntletChainTrace.h"
#include "pchie/DeterminismGauntlet.h"
#include "pchie/FreezeWatchdog.h"
#include "pchie/ChainRecord.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

struct FreezeTrial {
  std::vector<uint8_t> head_sha;
  std::vector<uint8_t> world_sha;
  std::vector<uint8_t> actions_sha;
  uint64_t head_fnv = 0;
  size_t chain_length = 0;
  bool ok = false;
  int discarded_slot = -1;
  std::vector<uint32_t> committed_slots;
  GauntletDiff first_diff;
};

struct FreezeResult {
  std::vector<FreezeTrial> trials;
  bool pass = false;
  std::string err;
  std::vector<uint8_t> clean_head_sha;
  std::vector<uint8_t> clean_world_sha;
  std::vector<uint8_t> clean_actions_sha;
  size_t clean_chain_length = 0;
  int matching = 0;
};

class FreezeRecovery final {
 public:
  FreezeRecovery() = default;
  void set_seed(uint64_t s) { seed_ = s; }
  void set_trials(int n) { trials_ = n; }
  void set_size(int width, int height) {
    width_ = width;
    height_ = height;
  }
  void set_freeze_point(uint32_t task_id, int step, int iter) {
    task_id_ = task_id;
    step_ = step;
    iter_ = iter;
  }

  bool run_unit(FreezeResult& out) { return run_common(nullptr, out); }
  bool run(NemotronModel& m, FreezeResult& out) { return run_common(&m, out); }

  bool run_watch_unsupported(FreezeResult& out) {
    out = {};
    out.err = "watch mode / UI is not built (deferred to the in-engine pilot). Not faked.";
    return false;
  }
  bool run_threaded_handshake_unsupported(FreezeResult& out) {
    out = {};
    out.err =
        "threaded LithoLattice TimingHandshakeAuthority is not used "
        "(it needs mutex/condition_variable/atomic). Not faked.";
    return false;
  }

 private:
  struct HookCtx {
    FreezeWatchdog* wd = nullptr;
    uint32_t target_task = 0;
    int target_step = -1;
    int target_iter = -1;
    bool fired = false;
    uint32_t current_slot = 0;
    uint32_t discarded_slot = 0;
    bool discarded_valid = false;
    std::vector<uint32_t> committed_slots;
  };

  static uint64_t u64le(const std::vector<uint8_t>& b) {
    uint64_t h = 0;
    for (size_t i = 0; i < b.size() && i < 8; ++i) h |= uint64_t(b[i]) << (8 * i);
    return h;
  }

  static std::vector<uint8_t> hash_actions(const std::vector<int32_t>& acts) {
    std::vector<uint8_t> bytes(acts.size() * 4);
    for (size_t i = 0; i < acts.size(); ++i) {
      const uint32_t v = uint32_t(acts[i]);
      bytes[i * 4 + 0] = uint8_t(v);
      bytes[i * 4 + 1] = uint8_t(v >> 8);
      bytes[i * 4 + 2] = uint8_t(v >> 16);
      bytes[i * 4 + 3] = uint8_t(v >> 24);
    }
    if (bytes.empty()) return Sha256::hash(nullptr, 0);
    return Sha256::hash(bytes.data(), bytes.size());
  }

  static bool should_discard(void* c, uint32_t task_id, int step, int iter) {
    auto* ctx = static_cast<HookCtx*>(c);
    if (!ctx || !ctx->wd) return false;
    // The B1 hook already discarded and is retrying: drop the latch so later
    // successful rollouts can try_commit (the stalled slot stays discarded).
    if (ctx->fired && ctx->wd->frozen()) ctx->wd->resetAfterFreeze();
    const uint32_t slot = ctx->wd->allocate_slot();
    if (!ctx->wd->begin_rollout(slot)) return false;
    ctx->current_slot = slot;
    if (!ctx->fired && task_id == ctx->target_task && step == ctx->target_step &&
        iter == ctx->target_iter) {
      if (!ctx->wd->force_freeze(slot, task_id, step, iter)) return false;
      ctx->fired = true;
      ctx->discarded_slot = slot;
      ctx->discarded_valid = true;
      return true;
    }
    return false;
  }

  static void on_committed(void* c, uint32_t /*task_id*/, int /*step*/, int /*iter*/) {
    auto* ctx = static_cast<HookCtx*>(c);
    if (!ctx || !ctx->wd) return;
    const uint32_t slot = ctx->current_slot;
    const uint32_t gen = ctx->wd->generation();
    if (ctx->wd->try_commit(slot, gen)) {
      ctx->committed_slots.push_back(slot);
      ctx->wd->end_rollout(slot);
    }
  }

  bool run_common(NemotronModel* model, FreezeResult& out) {
    out = {};
    if (trials_ < 3) {
      out.err = "trials must be >= 3";
      return false;
    }

    GauntletChainTrace clean_ch;
    clean_ch.reset();
    HeadlessMaze clean_maze(seed_, width_, height_);
    MazeResult cr;
    const bool cok = model ? clean_maze.run(*model, "", &clean_ch, cr)
                           : clean_maze.run_unit("", &clean_ch, cr);
    if (!cok || !cr.ok) {
      out.err = cr.err.empty() ? "clean maze failed" : cr.err;
      return false;
    }
    out.clean_head_sha = clean_ch.head();
    out.clean_world_sha = cr.h_world_sha;
    out.clean_actions_sha = hash_actions(cr.actions);
    out.clean_chain_length = clean_ch.length();

    bool all_ok = true;
    for (int t = 0; t < trials_; ++t) {
      FreezeWatchdog wd;
      wd.reset();
      HookCtx ctx;
      ctx.wd = &wd;
      ctx.target_task = task_id_;
      ctx.target_step = step_;
      ctx.target_iter = iter_;

      GauntletChainTrace ch;
      ch.reset();
      HeadlessMaze maze(seed_, width_, height_);
      maze.set_freeze_check(&FreezeRecovery::should_discard, &ctx);
      maze.set_iter_done(&FreezeRecovery::on_committed, &ctx);
      MazeResult res;
      const bool rok = model ? maze.run(*model, "", &ch, res) : maze.run_unit("", &ch, res);

      if (wd.frozen()) wd.resetAfterFreeze();

      FreezeTrial row;
      row.head_sha = ch.head();
      row.world_sha = res.h_world_sha;
      row.actions_sha = hash_actions(res.actions);
      row.head_fnv = u64le(ch.fnv_head());
      row.chain_length = ch.length();
      row.committed_slots = ctx.committed_slots;
      if (ctx.discarded_valid) row.discarded_slot = int(ctx.discarded_slot);
      row.first_diff = DeterminismGauntlet::first_diff(clean_ch, ch);

      const bool slot_ok = ctx.discarded_valid && row.discarded_slot >= 0 &&
                           !contains(row.committed_slots, uint32_t(row.discarded_slot)) &&
                           !wd.has_duplicate_slot();
      row.ok = rok && res.ok && row.head_sha == out.clean_head_sha &&
               row.world_sha == out.clean_world_sha &&
               row.actions_sha == out.clean_actions_sha &&
               row.chain_length == out.clean_chain_length && row.first_diff.index < 0 &&
               slot_ok;
      if (row.ok) ++out.matching;
      else all_ok = false;
      out.trials.push_back(std::move(row));
    }

    out.pass = all_ok && out.matching == trials_ && int(out.trials.size()) == trials_;
    if (!out.pass && out.err.empty()) out.err = "freeze recovery mismatch";
    return out.pass;
  }

  static bool contains(const std::vector<uint32_t>& v, uint32_t s) {
    for (uint32_t x : v)
      if (x == s) return true;
    return false;
  }

  uint64_t seed_ = 1;
  int trials_ = 3;
  int width_ = 16;
  int height_ = 16;
  uint32_t task_id_ = 5;
  int step_ = 0;
  int iter_ = 0;
};

}  // namespace pchie
