// Bolt PCHIE B4 - CPU determinism gauntlet over the B1 headless maze.
// Concrete final. Serial coordinator; workers only inside the model forward
// (JoinJobDispatcher, no mutex / CV / atomic). Official chain is append-only.
// Honest scope: CPU, Little Nemo or the labeled unit B1 fixture, worker counts
// 1,2,4,8,16. CUDA, B3, the Gauntlet-board UI, and Windows-vs-Linux are deferred
// and fail cleanly rather than being faked.
#pragma once

#include "pchie/HeadlessMaze.h"
#include "pchie/GauntletChainTrace.h"
#include "pchie/ChainRecord.h"
#include "pchie/JoinJobDispatcher.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pchie {

struct GauntletDiff {
  long long index = -1;  // -1 = identical
  ChainRecordKind kind_a = ChainRecordKind::Custom;
  ChainRecordKind kind_b = ChainRecordKind::Custom;
  bool kind_a_present = false;
  bool kind_b_present = false;
};

struct GauntletRow {
  int workers = 0;
  int runs = 0;
  int unique_hashes = 0;
  int b1_failures = 0;
  std::vector<uint8_t> head_sha;
  uint64_t head_fnv = 0;
  bool pass = false;
  GauntletDiff first_diff;
  int first_diff_run = -1;
};

struct GauntletResult {
  std::vector<GauntletRow> rows;
  bool pass = false;
  std::string err;
  std::vector<uint8_t> ref_head_sha;
  std::vector<uint8_t> ref_actions_sha;
  std::vector<uint8_t> ref_world_sha;
  size_t ref_chain_length = 0;
  int total_runs = 0;
};

class DeterminismGauntlet final {
 public:
  DeterminismGauntlet() = default;
  void set_workers(std::vector<int> w) { workers_ = std::move(w); }
  void set_runs(int r) { runs_ = r; }
  void set_seed(uint64_t s) { seed_ = s; }
  void set_size(int width, int height) {
    width_ = width;
    height_ = height;
  }
  // Unit-test hook only: run (config_index, run_index) uses seed+1.
  void set_perturb(int config_index, int run_index) {
    pc_ = config_index;
    pr_ = run_index;
  }

  static GauntletDiff first_diff(const GauntletChainTrace& a, const GauntletChainTrace& b) {
    GauntletDiff d;
    const size_t na = a.kinds().size();
    const size_t nb = b.kinds().size();
    const size_t n = na < nb ? na : nb;
    for (size_t i = 0; i < n; ++i) {
      const bool kind_diff = a.kinds()[i] != b.kinds()[i];
      const bool head_diff =
          i >= a.heads().size() || i >= b.heads().size() || a.heads()[i] != b.heads()[i];
      if (kind_diff || head_diff) {
        d.index = static_cast<long long>(i);
        d.kind_a = a.kinds()[i];
        d.kind_b = b.kinds()[i];
        d.kind_a_present = true;
        d.kind_b_present = true;
        return d;
      }
    }
    if (na != nb) {
      d.index = static_cast<long long>(n);
      if (n < na) {
        d.kind_a = a.kinds()[n];
        d.kind_a_present = true;
      }
      if (n < nb) {
        d.kind_b = b.kinds()[n];
        d.kind_b_present = true;
      }
    }
    return d;
  }

  bool run_unit(GauntletResult& out) { return run_common(nullptr, out); }
  bool run(NemotronModel& m, GauntletResult& out) { return run_common(&m, out); }

  bool run_cuda_unsupported(GauntletResult& out) {
    out = {};
    out.err = "CUDA gauntlet is not built (PCHIE_CUDA=OFF; no GPU path). Not faked.";
    return false;
  }
  bool run_b3_unsupported(GauntletResult& out) {
    out = {};
    out.err = "Trace fixture is not built (needs the Trace grader). Not faked.";
    return false;
  }

 private:
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

  bool run_common(NemotronModel* model, GauntletResult& out) {
    out = {};
    if (workers_.empty()) {
      out.err = "no workers";
      return false;
    }
    if (runs_ <= 0) {
      out.err = "no runs";
      return false;
    }

    GauntletChainTrace ref;
    ref.reset();
    HeadlessMaze ref_maze(seed_, width_, height_);
    MazeResult rr;
    const bool rok =
        model ? ref_maze.run(*model, "", &ref, rr) : ref_maze.run_unit("", &ref, rr);
    if (!rok || !rr.ok) {
      out.err = rr.err.empty() ? "reference maze failed" : rr.err;
      return false;
    }
    out.ref_head_sha = ref.head();
    out.ref_world_sha = rr.h_world_sha;
    out.ref_actions_sha = hash_actions(rr.actions);
    out.ref_chain_length = ref.length();

    bool all_pass = true;
    for (int ci = 0; ci < int(workers_.size()); ++ci) {
      const int w = workers_[size_t(ci)];
      GauntletRow row;
      row.workers = w;
      row.runs = runs_;
      JoinJobDispatcher disp(w);
      if (disp.size() != w) {
        row.unique_hashes = 0;
        row.pass = false;
        all_pass = false;
        out.err = "invalid workers";
        out.rows.push_back(row);
        continue;
      }
      // Always clear the hook, including if a run throws, so ext_ cannot dangle
      // after this JoinJobDispatcher leaves scope.
      struct DispatcherGuard {
        NemotronModel* m = nullptr;
        DispatcherGuard(NemotronModel* model, IJobDispatcher* d) : m(model) {
          if (m) m->set_dispatcher(d);
        }
        ~DispatcherGuard() {
          if (m) m->set_dispatcher(nullptr);
        }
        DispatcherGuard(const DispatcherGuard&) = delete;
        DispatcherGuard& operator=(const DispatcherGuard&) = delete;
      };
      DispatcherGuard hook(model, &disp);

      std::vector<std::vector<uint8_t>> hashes;
      hashes.reserve(size_t(runs_));
      for (int r = 0; r < runs_; ++r) {
        uint64_t seed = seed_;
        if (ci == pc_ && r == pr_) seed = seed_ + 1;  // labeled unit-test hook only
        GauntletChainTrace ch;
        ch.reset();
        HeadlessMaze maze(seed, width_, height_);
        MazeResult res;
        const bool ok =
            model ? maze.run(*model, "", &ch, res) : maze.run_unit("", &ch, res);
        ++out.total_runs;
        if (!ok || !res.ok) ++row.b1_failures;
        hashes.push_back(ch.head());
        if (row.head_sha.empty()) {
          row.head_sha = ch.head();
          row.head_fnv = u64le(ch.fnv_head());
        }
        if (row.first_diff.index < 0) {
          const GauntletDiff d = first_diff(ref, ch);
          if (d.index >= 0) {
            row.first_diff = d;
            row.first_diff_run = r;
          }
        }
      }

      row.unique_hashes = unique_count(hashes);
      row.pass = row.b1_failures == 0 && row.unique_hashes == 1 &&
                 row.head_sha == out.ref_head_sha && row.first_diff.index < 0;
      if (!row.pass) all_pass = false;
      out.rows.push_back(std::move(row));
    }

    out.pass = all_pass;
    if (!all_pass && out.err.empty()) out.err = "gauntlet mismatch";
    return all_pass;
  }

  std::vector<int> workers_{1, 2, 4, 8, 16};
  int runs_ = 100;
  uint64_t seed_ = 1;
  int width_ = 16;
  int height_ = 16;
  int pc_ = -1, pr_ = -1;
};

}  // namespace pchie
