// Bolt PCHIE B7 - headless node / tick / token budget harness over B1.
// Concrete final. Serial, deterministic. No mutex, no condition_variable, no atomic.
// Node budget is a per-search PUCT node cap (including the root). Tick budget is
// a global deterministic stand-in for wall-clock time (one tick per MCTS
// iteration started). Token budget caps committed maze actions. Exhaustion
// appends ChainRecordKind::BudgetExhausted and still returns a best-so-far
// action. The board budget slider (UI) and a real wall-clock timeout are
// deferred and fail cleanly.
#pragma once

#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/DeterminismGauntlet.h"
#include "pchie/InjectionChainAudit.h"
#include "pchie/ChainRecord.h"
#include "pchie/NemotronModel.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace pchie {

struct BudgetRow {
  int node_budget = 0;
  int runs = 0;
  int unique_hashes = 0;
  int max_nodes = 0;
  int exhaust_records = 0;
  int off_legal = 0;
  bool reached_exit = false;
  bool pass = false;
  std::vector<uint8_t> head_sha;
  uint64_t head_fnv = 0;
  size_t chain_length = 0;
  GauntletDiff first_diff;
};

struct BudgetResult {
  std::vector<BudgetRow> rows;
  bool pass = false;
  std::string err;
  std::vector<uint8_t> open_head_sha;
  size_t open_chain_length = 0;
  int tick_limit_used = 0;
  int token_limit_used = 0;
  int tick_exhaust_records = 0;
  int token_exhaust_records = 0;
};

class BudgetHarness final {
 public:
  BudgetHarness() = default;
  void set_seed(uint64_t s) { seed_ = s; }
  void set_size(int width, int height) {
    width_ = width;
    height_ = height;
  }
  void set_node_budgets(std::vector<int> b) { node_budgets_ = std::move(b); }
  void set_runs(int n) { runs_ = n; }

  bool run_unit(BudgetResult& out) { return run_nodes(nullptr, out); }
  bool run(NemotronModel& m, BudgetResult& out) { return run_nodes(&m, out); }

  bool run_tick_budget_unit(int ticks, int runs, BudgetResult& out) {
    return run_tick_or_token(nullptr, /*ticks*/ticks, /*tokens*/0, runs, out);
  }
  bool run_token_budget_unit(int tokens, int runs, BudgetResult& out) {
    return run_tick_or_token(nullptr, /*ticks*/0, /*tokens*/tokens, runs, out);
  }

  bool run_watch_unsupported(BudgetResult& out) {
    out = {};
    out.err = "watch mode / budget-slider UI is not built (deferred to the in-engine pilot). Not faked.";
    return false;
  }
  bool run_wallclock_unsupported(BudgetResult& out) {
    out = {};
    out.err =
        "real wall-clock timeout is not used (it would make H_bolt non-deterministic). "
        "The honest path is a tick budget. Not faked.";
    return false;
  }
  bool run_cuda_unsupported(BudgetResult& out) {
    out = {};
    out.err = "CUDA budget run is not built (PCHIE_CUDA=OFF; no GPU path). Not faked.";
    return false;
  }

 private:
  static uint64_t u64le(const std::vector<uint8_t>& b) {
    uint64_t h = 0;
    for (size_t i = 0; i < b.size() && i < 8; ++i) h |= uint64_t(b[i]) << (8 * i);
    return h;
  }

  static bool legal_action(int32_t a) {
    return a == int32_t(MazeAction::North) || a == int32_t(MazeAction::South) ||
           a == int32_t(MazeAction::East) || a == int32_t(MazeAction::West) ||
           a == int32_t(MazeAction::Pickup) || a == int32_t(MazeAction::Use);
  }

  static int off_legal_count(const std::vector<int32_t>& acts) {
    int n = 0;
    for (int32_t a : acts)
      if (!legal_action(a)) ++n;
    return n;
  }

  static int count_kind(const InjectionChainAudit& ch, ChainRecordKind k) {
    int n = 0;
    for (ChainRecordKind x : ch.trace().kinds())
      if (x == k) ++n;
    return n;
  }

  static int unique_count(const std::vector<std::vector<uint8_t>>& hs) {
    int u = 0;
    for (size_t i = 0; i < hs.size(); ++i) {
      bool seen = false;
      for (size_t j = 0; j < i; ++j)
        if (hs[j] == hs[i]) {
          seen = true;
          break;
        }
      if (!seen) ++u;
    }
    return u;
  }

  bool run_one(NemotronModel* model, int node_budget, int tick_budget, int token_budget,
               InjectionChainAudit& ch, MazeResult& res) {
    ch.reset();
    HeadlessMaze maze(seed_, width_, height_);
    if (node_budget > 0) {
      maze.set_node_budget(node_budget);
      maze.set_search_budget(node_budget < 16 ? 16 : node_budget);
    }
    if (tick_budget > 0) maze.set_tick_budget(tick_budget);
    if (token_budget > 0) maze.set_token_budget(token_budget);
    const bool ok =
        model ? maze.run(*model, "", &ch, res) : maze.run_unit("", &ch, res);
    (void)ok;  // a tight cap may not reach the exit; B7 still reads the result
    return true;
  }

  bool run_nodes(NemotronModel* model, BudgetResult& out) {
    out = {};
    if (runs_ <= 0) {
      out.err = "runs must be positive";
      return false;
    }
    for (int b : node_budgets_) {
      if (b <= 0) {
        out.err = "node budget must be positive";
        return false;
      }
    }

    InjectionChainAudit open_ch;
    MazeResult open_res;
    HeadlessMaze open(seed_, width_, height_);
    const bool ook =
        model ? open.run(*model, "", &open_ch, open_res) : open.run_unit("", &open_ch, open_res);
    if (!ook || !open_res.ok) {
      out.err = open_res.err.empty() ? "open maze failed" : open_res.err;
      return false;
    }
    out.open_head_sha = open_ch.head();
    out.open_chain_length = open_ch.length();

    bool all_ok = true;
    for (int cap : node_budgets_) {
      BudgetRow row;
      row.node_budget = cap;
      row.runs = runs_;
      std::vector<std::vector<uint8_t>> hashes;
      InjectionChainAudit first_ch;
      for (int r = 0; r < runs_; ++r) {
        InjectionChainAudit ch;
        MazeResult res;
        run_one(model, cap, /*tick*/0, /*token*/0, ch, res);
        hashes.push_back(ch.head());
        const int exhaust = count_kind(ch, ChainRecordKind::BudgetExhausted);
        const int off = off_legal_count(res.actions);
        if (res.max_nodes > row.max_nodes) row.max_nodes = res.max_nodes;
        if (r == 0) {
          first_ch = ch;
          row.head_sha = ch.head();
          row.head_fnv = u64le(ch.fnv_head());
          row.chain_length = ch.length();
          row.exhaust_records = exhaust;
          row.reached_exit = res.reached_exit;
        } else if (row.first_diff.index < 0) {
          row.first_diff = DeterminismGauntlet::first_diff(first_ch.trace(), ch.trace());
        }
        row.off_legal += off;
        if (res.max_nodes > cap) all_ok = false;
      }
      row.unique_hashes = unique_count(hashes);
      row.pass = row.unique_hashes == 1 && row.max_nodes > 0 && row.max_nodes <= cap &&
                 row.exhaust_records >= 1 && row.off_legal == 0 && row.first_diff.index < 0 &&
                 !row.head_sha.empty();
      if (!row.pass) all_ok = false;
      out.rows.push_back(std::move(row));
    }
    out.pass = all_ok && int(out.rows.size()) == int(node_budgets_.size());
    if (!out.pass && out.err.empty()) out.err = "node budget mismatch";
    return out.pass;
  }

  bool run_tick_or_token(NemotronModel* model, int ticks, int tokens, int runs, BudgetResult& out) {
    out = {};
    if (runs <= 0) {
      out.err = "runs must be positive";
      return false;
    }
    if (ticks <= 0 && tokens <= 0) {
      out.err = "budget must be positive";
      return false;
    }
    std::vector<std::vector<uint8_t>> hashes;
    int max_used = 0;
    int exhaust = 0;
    int off = 0;
    bool cap_ok = true;
    for (int r = 0; r < runs; ++r) {
      InjectionChainAudit ch;
      MazeResult res;
      run_one(model, /*node*/0, ticks, tokens, ch, res);
      hashes.push_back(ch.head());
      const int ex = count_kind(ch, ChainRecordKind::BudgetExhausted);
      if (r == 0) exhaust = ex;
      off += off_legal_count(res.actions);
      if (ticks > 0) {
        if (res.ticks_used > max_used) max_used = res.ticks_used;
        if (res.ticks_used > ticks) cap_ok = false;
      }
      if (tokens > 0) {
        if (res.path_length > max_used) max_used = res.path_length;
        if (int(res.actions.size()) > tokens) cap_ok = false;
      }
    }
    if (ticks > 0) {
      out.tick_limit_used = max_used;
      out.tick_exhaust_records = exhaust;
    }
    if (tokens > 0) {
      out.token_limit_used = max_used;
      out.token_exhaust_records = exhaust;
    }
    out.pass = cap_ok && unique_count(hashes) == 1 && exhaust >= 1 && off == 0;
    if (!out.pass && out.err.empty()) out.err = "tick/token budget mismatch";
    return out.pass;
  }

  uint64_t seed_ = 1;
  int width_ = 16;
  int height_ = 16;
  std::vector<int> node_budgets_{1, 10, 100, 1000};
  int runs_ = 3;
};

}  // namespace pchie
