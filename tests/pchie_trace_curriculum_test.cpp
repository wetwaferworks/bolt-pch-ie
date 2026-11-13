// Bolt PCHIE - L1 curriculum (Multiplexor + Dividor) and Little Nemo harness.
// Draft answer key is provisional, unit-only, never a golden, pending review.
// Proposal outputs stay off the official chain.
#include "pchie/Dividor.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/Multiplexor.h"
#include "pchie/NemotronModel.h"
#include "pchie/PchGate.h"
#include "pchie/Sha256.h"
#include "pchie/SimTraceGrader.h"
#include "pchie/TraceCircuit.h"
#include "pchie/TraceFacing.h"
#include "pchie/TraceL1AnswerKey.h"
#include "pchie/TraceL1Curriculum.h"
#include "pchie/TraceL1Harness.h"
#include "pchie/TraceStrength.h"
#include "pchie/pchie_tree.h"
#include "pchie/ModelPath.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace pchie;

static const std::string& model_path() {
  static const std::string p = [] {
    std::string path, err;
    if (!pchie::resolve_model_path(path, err)) {
      std::fprintf(stderr, "%s\n", err.c_str());
      return std::string();
    }
    return path;
  }();
  return p;
}

static std::string draft_key_path() {
  std::error_code ec;
  const auto dir = std::filesystem::temp_directory_path() / "pchie_p5";
  std::filesystem::create_directories(dir, ec);
  return (dir / "P5_L1_DRAFT_ANSWER_KEY.txt").string();
}

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[TRACE-CURRICULUM] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static bool collide8(const std::vector<uint8_t>& sha, uint64_t fnv) {
  return sha.size() >= 8 && std::memcmp(&fnv, sha.data(), 8) == 0;
}

static void test_strength_reject() {
  check(TraceStrength::ok(0) && TraceStrength::ok(64) && !TraceStrength::ok(-1) &&
            !TraceStrength::ok(65),
        "strengths 0..64 are valid; negatives and 65 are rejected");
  int out = 99;
  check(!Dividor::evaluate(-1, out) && !Dividor::evaluate(65, out),
        "Dividor rejects out-of-range input instead of clamping");
  check(Dividor::evaluate(0, out) && out == 0 && Dividor::evaluate(5, out) && out == 2,
        "Dividor still evaluates 0 -> 0 and 5 -> 2 inside the domain");
  int m = 99;
  check(!Multiplexor::evaluate(-1, 1, 1, m) && !Multiplexor::evaluate(1, 1, 65, m),
        "Multiplexor rejects any out-of-range port");
  check(Multiplexor::evaluate(7, 11, 1, m) && m == 11,
        "east strength > 0 (any source) selects west");
}

static void test_dividor_facing_north() {
  int front = 99;
  check(Dividor::evaluate_facing(TraceFacing::North, /*south*/5, /*west*/0, /*east*/0,
                                 /*north unused*/0, front) &&
            front == 2,
        "Dividor facing north halves BACK (south) to FRONT (north)");
  check(Dividor::back_face(TraceFacing::North) == TraceFacing::South &&
            Dividor::front_face(TraceFacing::North) == TraceFacing::North,
        "facing north => in=south (back), out=north (front)");
}

static void test_l1_curriculum_and_key() {
  const auto tasks = TraceL1Curriculum::tasks();
  int n_div = 0, n_mux = 0;
  bool goals_ok = !tasks.empty();
  for (const auto& t : tasks) {
    if (t.facing != TraceFacing::North) goals_ok = false;
    if (t.goal < 0 || t.goal > 64) goals_ok = false;
    const auto* ans = TraceL1AnswerKey::for_task(t.id);
    if (!ans) {
      goals_ok = false;
      continue;
    }
    if (std::strcmp(ans->component, Dividor::kName) == 0) ++n_div;
    if (std::strcmp(ans->component, Multiplexor::kName) == 0) ++n_mux;
    TraceCircuit right;
    right.component = ans->component;
    right.facing = ans->facing;
    right.south = t.south;
    right.west = t.west;
    right.east = t.east;
    const TraceGrade gok = SimTraceGrader().grade_against_key(right, t);
    if (!gok.pass || gok.observed != t.goal) {
      std::printf("[TRACE-CURRICULUM] key circuit failed task=%s observed=%d goal=%d reason=%s\n",
                  t.id, gok.observed, t.goal, gok.reason ? gok.reason : "");
      goals_ok = false;
    }
    TraceCircuit wrong = right;
    wrong.component = (std::strcmp(ans->component, Dividor::kName) == 0) ? Multiplexor::kName
                                                                         : Dividor::kName;
    const TraceGrade gbad_out = SimTraceGrader().grade_circuit(wrong, t.goal);
    const TraceGrade gbad_key = SimTraceGrader().grade_against_key(wrong, t);
    if (gbad_out.pass || gbad_key.pass) {
      std::printf("[TRACE-CURRICULUM] wrong component unexpectedly passed task=%s out=%d key=%d\n",
                  t.id, int(gbad_out.pass), int(gbad_key.pass));
      goals_ok = false;
    }
  }
  check(n_div >= 1 && n_mux >= 1, "L1 curriculum has Multiplexor and Dividor tasks");
  check(goals_ok, "each L1 task has a goal; key circuit PASSES, wrong component FAILS");

  std::string err;
  check(TraceL1AnswerKey::write_readable(draft_key_path().c_str(), err) && err.empty(),
        "draft answer key is emitted as a readable file for review");
  const auto bytes = TraceL1AnswerKey::canonical_bytes();
  const auto sha = Sha256::hash(bytes.data(), bytes.size());
  const uint64_t fnv = TraceL1AnswerKey::fnv();
  check(!bytes.empty() && sha.size() == 32, "draft answer key has canonical bytes and SHA-256");
  if (collide8(sha, fnv)) {
    std::printf("[TRACE-CURRICULUM] ALERT: FNV vs SHA disagreement FAILED on draft answer key -- report it immediately\n");
  }
  check(!collide8(sha, fnv), "draft answer key: FNV != first 8 of SHA-256");
  std::printf("[TRACE-CURRICULUM] draft answer key path=%s\n", draft_key_path().c_str());
  std::printf("[TRACE-CURRICULUM] draft key sha256=%s fnv=0x%016llX (provisional, unit-only, never a golden)\n",
              GgufWeightsHash::to_hex(sha).c_str(), (unsigned long long)fnv);
}

static void test_real_little_nemo() {
  const auto tasks = TraceL1Curriculum::tasks();
  NemotronModel m;
  std::string err;
  if (!m.load(model_path().c_str(), err)) {
    check(false, "real-weight L1: NemotronModel::load(Little Nemo) succeeds");
    std::printf("[TRACE-CURRICULUM] load err: %s\n", err.c_str());
    return;
  }
  m.init_cache(/*max_ctx*/128, /*branches*/4);
  const uint32_t live0 = m.pool_swa().live_pages() + m.pool_glob().live_pages();

  std::vector<TraceL1Harness::Result> a, b;
  bool oka = true, okb = true;
  int pass_n = 0, fail_n = 0, gate_fail = 0;
  for (const auto& t : tasks) {
    TraceL1Harness::Result r1, r2;
    if (!TraceL1Harness::run(m, t, r1, err)) oka = false;
    if (!TraceL1Harness::run(m, t, r2, err)) okb = false;
    a.push_back(r1);
    b.push_back(r2);
    if (r1.grader.pass) ++pass_n;
    else ++fail_n;
    if (!r1.gate_pass) ++gate_fail;
    std::printf("[TRACE-CURRICULUM] Little Nemo task=%s proposed=%s gate=%d grade=%s observed=%d goal=%d key=%s\n",
                t.id, r1.circuit.component ? r1.circuit.component : "?", int(r1.gate_pass),
                r1.grader.pass ? "PASS" : "FAIL", r1.grader.observed, t.goal,
                TraceL1AnswerKey::for_task(t.id) ? TraceL1AnswerKey::for_task(t.id)->component : "?");
  }
  check(oka && okb, "real Little Nemo: every L1 task produced a proposal twice");
  bool det = a.size() == b.size();
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (std::strcmp(a[i].circuit.component ? a[i].circuit.component : "",
                    b[i].circuit.component ? b[i].circuit.component : "") != 0)
      det = false;
    if (a[i].gate_pass != b[i].gate_pass || a[i].grader.pass != b[i].grader.pass) det = false;
  }
  check(det, "real Little Nemo: L1 proposals and grades are run-to-run deterministic");
  std::printf("[TRACE-CURRICULUM] Little Nemo L1 honest score: pass=%d fail=%d gate_reject=%d of %zu (not retuned)\n",
              pass_n, fail_n, gate_fail, tasks.size());
  const uint32_t live_end = m.pool_swa().live_pages() + m.pool_glob().live_pages();
  check(live_end == live0, "real Little Nemo: no KV page leak after L1 harness");
  std::printf("[TRACE-CURRICULUM] live_pages start=%u end=%u\n", live0, live_end);
}

int main() {
  std::printf("[TRACE-CURRICULUM] curriculum + draft answer key + Little Nemo harness (official chain untouched)\n");
  test_strength_reject();
  test_dividor_facing_north();
  test_l1_curriculum_and_key();
  test_real_little_nemo();
  std::printf("[TRACE-CURRICULUM] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
