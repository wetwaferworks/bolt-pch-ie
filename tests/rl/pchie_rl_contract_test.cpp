// Bolt PCHIE reasoning-layer contracts and flags-off identity.
// LABELED UNIT-TEST FIXTURE ONLY where L6 stubs are loaded.
// The flags-off official-token regression is deferred (see reasoning/scripts/flags-off-hash-regression.md).
#include "pchie/rl/BeliefState.h"
#include "pchie/rl/CrutchAudit.h"
#include "pchie/rl/ForkScore.h"
#include "pchie/rl/HashMoveRecord.h"
#include "pchie/rl/LabelSchema.h"
#include "pchie/rl/LeafValueSource.h"
#include "pchie/rl/Options.h"
#include "pchie/rl/PchStackRlHooks.h"
#include "pchie/rl/PlanMemory.h"
#include "pchie/rl/RlFlags.h"
#include "pchie/rl/RlSideChain.h"
#include "pchie/rl/SymbolicPlanner.h"
#include "pchie/rl/UntrustedTokens.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace pchie::rl;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[RL] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}

static std::string test_root() {
  // append_hash_move only accepts paths that contain "scratch/rl".
  std::error_code ec;
  const auto p = std::filesystem::temp_directory_path() / "pchie-rl-contract" / "scratch" / "rl" / "tmp-test";
  std::filesystem::create_directories(p, ec);
  return p.generic_string();
}

static std::string labels_dir() {
  std::error_code ec;
  std::vector<std::filesystem::path> starts;
#ifdef PCHIE_REPO_ROOT
  starts.emplace_back(PCHIE_REPO_ROOT);
#endif
  starts.emplace_back(std::filesystem::path(__FILE__).parent_path());
  starts.emplace_back(std::filesystem::current_path());
  for (auto s : starts) {
    for (int i = 0; i < 10 && !s.empty(); ++i) {
      const auto cand = s / "reasoning" / "data" / "labels" / "v0";
      if (std::filesystem::is_directory(cand, ec)) return cand.generic_string() + "/";
      s = s.parent_path();
    }
  }
  return {};
}

static void clean_tree(const std::string& p) {
  std::error_code ec;
  std::filesystem::remove_all(p, ec);
  std::filesystem::create_directories(p, ec);
}

static UntrustedSymbolGrid grid3(const char* r0, const char* r1, const char* r2) {
  UntrustedSymbolGrid g;
  g.radius = 1;
  g.cells = {r0, r1, r2};
  return g;
}

static void test_flags_off_identity() {
  const RlFlags f{};
  PchStackRlHooks hooks;
  check(flags_off_hooks_are_idle(f, hooks), "flags-off identity: hooks idle, official_clean");

  const std::string log = test_root() + "/hash_moves.jsonl";
  check(hash_move_count(log) == 0, "flags-off identity: no hash_moves.jsonl records exist");

  // Official path refused.
  HashMoveRecord rec;
  rec.run_id = "rl-flags-off";
  rec.token = "H_world";
  rec.cause = "should not be written to an official path";
  rec.attribution = "unknown";
  std::string err;
  check(!append_hash_move("/tmp/official-chain.jsonl", rec, err) && !err.empty(),
        "hash-move append refuses a path outside scratch/rl/");
  check(hash_move_count(log) == 0, "refused append wrote 0 records");

  RlSideChain side;
  check(side.length() == 0 && side.head_hex() == std::string(64, '0'),
        "scratch side chain starts at 32 zero bytes");
  check(!side.append_jsonl_and_chain("/tmp/official-side.jsonl", "{}", err),
        "side chain refuses a path outside scratch/rl/");
  check(side.length() == 0, "refused side-chain write did not advance the head");
}

static void test_hash_move_scratch() {
  const std::string log = test_root() + "/hash_moves.jsonl";
  HashMoveRecord rec;
  rec.run_id = "rl-L1-fixture";
  rec.log = "pchie_rl_contract_test";
  rec.token = "tree_digest";
  rec.variable = "tree digest";
  rec.old_value = "aaa";
  rec.new_value = "bbb";
  rec.baseline = "post-gate-split baseline (98 tokens)";
  rec.change_id = "L1:--rl-puct-model";
  rec.cause = "L1 value/prior source = model";
  rec.attribution = "single-flag ablation";
  rec.official_token = false;
  std::string err;
  check(append_hash_move(log, rec, err) && err.empty(), "hash-move append under scratch/rl/ succeeds");
  check(hash_move_count(log) == 1, "one hash-move record written");
  // This fixture record is not a real move against the 98/98 baseline.
  // Remove it so the tree's default log stays empty of claimed official moves.
  std::filesystem::remove(log);
  check(hash_move_count(log) == 0, "fixture record cleaned; default log empty again");
}

static void test_l1() {
  OracleLeafValue oracle;
  ModelLeafValue model;
  NullLeafValue null_src;
  RlFlags off{};
  auto* src = make_leaf_value_source(off, &oracle, &model, &null_src);
  check(src == &oracle && std::string(src->name()) == "oracle", "flags off: leaf source is oracle");

  RlLeafContext ctx;
  ctx.bfs_distance = 0;
  const LeafValue v0 = src->value(ctx);
  check(v0.source_tag == kLeafSrcOracle && v0.q20 == kQ20, "oracle at distance 0 is Q20");
  ctx.bfs_distance = 1;
  const LeafValue v1 = oracle.value(ctx);
  check(v1.q20 == kQ20 / 2, "oracle at distance 1 is Q20/2 (1/(1+1))");

  // Determinism
  const LeafValue v1b = oracle.value(ctx);
  check(v1.q20 == v1b.q20 && v1.source_tag == v1b.source_tag, "oracle is deterministic");

  // Model path must not call a poison oracle.
  PoisonOracleLeafValue poison;
  bool poison_fired = false;
  poison.on_call = [&] { poison_fired = true; };
  model.set_verdict_mass(kQ20, 0, 0);
  RlFlags on = parse_rl_flags(std::vector<std::string>{"--rl-puct-model"}).flags;
  auto* msrc = make_leaf_value_source(on, &oracle, &model, &null_src);
  check(msrc == &model, "--rl-puct-model selects ModelLeafValue");
  oracle.clear_called();
  const LeafValue mv = msrc->value(ctx);
  check(mv.source_tag == kLeafSrcModel && mv.q20 == kQ20, "model closer-mass = 1 is Q20");
  check(!oracle.called() && !poison_fired, "oracle-poison: model path does not call the oracle");
  (void)poison.value(ctx);
  check(poison.called(), "poison oracle would fire if selected (control)");

  // Null control
  RlFlags nf = parse_rl_flags(std::vector<std::string>{"--rl-puct-value=null"}).flags;
  auto* nsrc = make_leaf_value_source(nf, &oracle, &model, &null_src);
  check(nsrc == &null_src && nsrc->value(ctx).source_tag == kLeafSrcNull, "null control selected");
  check(null_src.value(ctx).q20 == kQ20 / 2, "null control is the uniform Q20/2 constant");

  // Gate fail-closed
  std::vector<PriorEdge> edges = {{10, 1000, false}, {99, 2000, false}, {11, 3000, false}};
  apply_gate_fail_closed({99}, edges);
  check(edges[0].prior_q20 == 1000 && !edges[0].fail_closed, "legal edge prior unchanged");
  check(edges[1].prior_q20 == 0 && edges[1].fail_closed, "illegal-move edge prior 0 and fail-closed");

  check(prior_source_from_flags(off) == PriorSource::CurrentMix, "flags off: prior source is current mix");
  check(prior_source_from_flags(on) == PriorSource::ModelLogits, "L1 flag: prior source is model logits");
}

static void test_l2() {
  const std::string path = test_root() + "/mem/verified.jsonl";
  std::filesystem::create_directories(test_root() + "/mem");
  {
    FilePlanMemory mem(path, MinTruth::Sim);
    check(mem.load_ok() && mem.size() == 0, "empty store loads as empty");
    SituationKey key;
    key.task_id = 1;
    key.obs_tokens = {1, 2, 3};
    const auto miss = mem.query_verified(key);
    check(!miss.has_value() && mem.miss_count() == 1, "empty store: miss recorded");

    mem.set_run_live(true);
    MemoryRow row;
    row.situation_hash = situation_hash(key);
    row.plan_fragment = {1, 1, 1};  // E E E
    row.truth = TruthState::SimVerified;
    check(!mem.append_between_runs(row) && mem.last_append_refused(), "mid-run write is refused");

    mem.set_run_live(false);
    check(mem.append_between_runs(row), "between-run write of a verified row succeeds");
    check(mem.size() == 1, "store has one row");
  }
  {
    FilePlanMemory mem(path, MinTruth::Sim);
    check(mem.load_ok() && mem.size() == 1, "store reloads one chained row");
    SituationKey key;
    key.task_id = 1;
    key.obs_tokens = {1, 2, 3};
    const auto hit = mem.query_verified(key);
    check(hit && hit->plan_fragment.size() == 3 && mem.hit_count() == 1, "seeded verified fragment is a hit");
    check(hit->chain_valid && hit->truth == TruthState::SimVerified, "hit is chain_valid and SimVerified");
  }
  // Unverified / refuted never returned.
  {
    const std::string p2 = test_root() + "/mem/mixed.jsonl";
    FilePlanMemory mem(p2, MinTruth::Sim);
    SituationKey ku, kr, kh;
    ku.task_id = 2;
    kr.task_id = 3;
    kh.task_id = 4;
    MemoryRow u, r, h;
    u.situation_hash = situation_hash(ku);
    u.truth = TruthState::Unverified;
    u.plan_fragment = {0};
    r.situation_hash = situation_hash(kr);
    r.truth = TruthState::Refuted;
    r.plan_fragment = {0};
    h.situation_hash = situation_hash(kh);
    h.truth = TruthState::HumanVerified;
    h.plan_fragment = {2};
    check(mem.append_between_runs(u) && mem.append_between_runs(r) && mem.append_between_runs(h),
          "mixed-truth rows append");
    check(!mem.query_verified(ku) && !mem.query_verified(kr), "unverified and refuted are never returned");
    check(mem.query_verified(kh).has_value(), "HumanVerified is returned at min-truth=sim");
    FilePlanMemory human(p2, MinTruth::Human);
    // human store reloads the file; Human row should hit, Sim-only would miss
    check(human.query_verified(kh).has_value(), "min-truth=human returns HumanVerified");
  }
  // Tampered row_hash => chain_valid=false, never returned, broken link reported.
  {
    const std::string p3 = test_root() + "/mem/tamper.jsonl";
    SituationKey k;
    k.task_id = 9;
    k.obs_tokens = {7};
    {
      FilePlanMemory mem(p3, MinTruth::Sim);
      MemoryRow row;
      row.situation_hash = situation_hash(k);
      row.plan_fragment = {1};
      row.truth = TruthState::SimVerified;
      check(mem.append_between_runs(row), "tamper fixture wrote one honest row");
    }
    // Flip one hex nibble in row_hash.
    {
      std::ifstream in(p3);
      std::string line;
      std::getline(in, line);
      auto pos = line.find("\"row_hash\":\"");
      check(pos != std::string::npos, "tamper fixture has row_hash");
      pos += 12;  // after "row_hash":"
      if (pos < line.size()) line[pos] = (line[pos] == 'a' ? 'b' : 'a');
      std::ofstream out(p3, std::ios::trunc);
      out << line << '\n';
    }
    FilePlanMemory mem(p3, MinTruth::Sim);
    check(!mem.load_ok() && mem.broken_link(), "tampered row_hash reports a broken link");
    check(!mem.query_verified(k).has_value(), "tampered row is never returned");
  }
  // Poisoned (illegal) fragment is a proposal: the gate fail-closes; plan is not executed.
  {
    std::vector<PriorEdge> edges = {{99, 4096, false}};
    apply_gate_fail_closed({99}, edges);
    check(edges[0].fail_closed && edges[0].prior_q20 == 0,
          "poisoned fragment (illegal move) is rejected by the gate and fails closed");
  }
}

static void test_l3() {
  check(step_back_allowed_v<ChainKind::Scratch>, "compile-time: scratch step_back allowed");
  check(!step_back_allowed_v<ChainKind::Official>, "compile-time: official step_back forbidden");
  std::string err;
  check(!may_step_back(ChainKind::Official, err) && err.find("official") != std::string::npos,
        "runtime: official-chain step_back is refused");
  check(may_step_back(ChainKind::Scratch, err) && err.empty(), "runtime: scratch step_back allowed");

  DecisionTuple t;
  t.state_hash.fill(0x11);
  t.options = {0, 1, 2, 3};
  t.decision = 1;
  t.distribution_q20 = {100, 200, 300, 400};
  t.confidence_q20 = 200;
  t.weights_hash.fill(0x22);
  t.blank_control_decision = 0;
  t.verifier_verdict = 0xFF;
  const auto seal = seal_decision_tuple(t);
  check(tuple_rehashes_to(t, seal), "decision tuple re-hashes to its sealed value");
  t.decision = 2;
  check(!tuple_rehashes_to(t, seal), "a changed decision breaks the seal");

  // Join-order independence.
  std::vector<ForkWorkItem> a = {{3, 10}, {1, 20}, {2, 30}};
  std::vector<ForkWorkItem> b = {{2, 30}, {3, 10}, {1, 20}};
  std::vector<ForkWorkItem> c = {{1, 20}, {3, 10}, {2, 30}};
  check(fold_digest(a) == fold_digest(b) && fold_digest(b) == fold_digest(c),
        "fork-join fold is independent of arrival order");
}

static void test_l4() {
  BeliefState b;
  Pose at{1, 1, Heading::E};
  b.observe(grid3("UWU", "FVF", "UWU"), at);
  check(b.cell(1, 1) == Cell::Visited, "agent cell is Visited");
  check(b.cell(1, 0) == Cell::Wall && b.cell(1, 2) == Cell::Wall, "north/south walls observed");
  check(b.cell(0, 1) == Cell::Floor && b.cell(2, 1) == Cell::Floor, "west/east floor observed");
  check(b.cell(0, 0) == Cell::Unknown, "unknown stays unknown");
  const ActionMask m0 = b.mask();
  check(m0.e && m0.w && !m0.n && !m0.s, "mask excludes observed walls, allows floor");

  const auto d1 = b.digest();
  BeliefState b2;
  b2.observe(grid3("UWU", "FVF", "UWU"), at);
  check(b2.digest() == d1, "belief digest is deterministic for a fixed observation sequence");

  // Blocked move marks Bumped and is then masked out.
  b.move_result(1 /*E*/, MoveOutcome::Blocked);
  check(b.cell(2, 1) == Cell::Bumped, "blocked move marks Bumped");
  check(!b.mask().e, "mask then excludes the bumped direction");
  check(b.pose().x == 1 && b.pose().y == 1, "blocked move does not change pose");

  // Revisit counts grow.
  BeliefState loop;
  loop.observe(grid3("UFU", "FVF", "UFU"), Pose{0, 0, Heading::N});
  loop.move_result(0, MoveOutcome::Moved);  // to (0,-1)
  check(loop.revisit_count(0, -1) == 1, "first visit count is 1");
  loop.move_result(2, MoveOutcome::Moved);  // back to (0,0)
  loop.move_result(0, MoveOutcome::Moved);  // to (0,-1) again
  check(loop.revisit_count(0, -1) >= 2, "revisit counts grow on a loop path");

  // tokenize_untrusted policy
  UntrustedTokenizeRequest req;
  req.text = std::string(kImStartLiteral) + " " + kBosLiteral;
  auto tok = tokenize_untrusted(req);
  check(tok.ok && !tok.tokens.empty(), "untrusted tokenize accepts special-token *strings*");
  // The literal characters are preserved as ordinary bytes, not stripped as specials.
  std::string round;
  for (int32_t t : tok.tokens) round.push_back(char(t));
  check(round == req.text, "special-token strings stay plain tokens (parse_special=false)");
  req.parse_special = true;
  check(!tokenize_untrusted(req).ok, "parse_special=true is refused");
  req.parse_special = false;
  req.add_bos = true;
  check(!tokenize_untrusted(req).ok, "BOS is refused");

  // Crutch audit
  CrutchAudit audit;
  check(audit.note_world_read("get_block"), "world read allowed when audit is off");
  audit.enable(true);
  check(!audit.note_world_read("get_block") && audit.violated(),
        "deliberate world read under --rl-vision-only aborts with CRUTCH_VIOLATION");
  check(audit.message().find("CRUTCH_VIOLATION") != std::string::npos, "message names CRUTCH_VIOLATION");
}

static void test_l5() {
  OptionRegistry reg;
  check(reg.add_option(spec_follow_corridor()) && reg.add_option(spec_turn()) &&
            reg.add_option(spec_backout()) && reg.add_option(spec_probe()),
        "built-in options register in the reserved range");
  OptionSpec bad = spec_probe();
  bad.id = 7;  // primitive-looking id, not reserved
  check(!reg.add_option(bad) && reg.err.find("reserved") != std::string::npos,
        "id outside the reserved range is refused at registration");

  BeliefState b;
  b.observe(grid3("UFU", "WVW", "UFU"), Pose{1, 1, Heading::N});
  check(can_start_follow_corridor(b), "N-S corridor can start FollowCorridor");

  // max_steps is honored: cap=1 stops after one accepted primitive.
  OptionSpec cap = spec_follow_corridor();
  cap.max_steps = 1;
  auto rcap = run_option(cap, b, /*reject*/ {});
  check(rcap.ok && rcap.stop == OptionStop::Cap && rcap.committed.size() == 1, "max_steps is honored");

  // Gate-rejected middle (here, the first) step fails closed, no partial commit.
  // Use a reject on the north primitive (0).
  auto rrej = run_option(spec_follow_corridor(), b, /*reject*/ {0});
  check(!rrej.ok && rrej.stop == OptionStop::GateReject && rrej.committed.empty(),
        "gate-rejected step fails closed and leaves no partial action committed");
  check(rrej.end.x == rrej.start.x && rrej.end.y == rrej.start.y, "fail-closed leaves pose uncommitted");

  // Deterministic boundaries on a fixed belief: two runs agree.
  auto a = run_option(spec_turn(), b, {});
  auto c = run_option(spec_turn(), b, {});
  check(a.ok && a.committed == c.committed && a.stop == c.stop,
        "option boundaries are deterministic on a fixed belief");
}

static void test_l6() {
  std::printf("[RL] %s\n", kFixtureBanner);

  // Verifier: 2-cycle loop is Incoherent.
  BeliefState b;
  b.observe(grid3("UFU", "FVF", "UFU"), Pose{1, 1, Heading::E});
  ThinPlanVerifier ver;
  DecisionTuple tup;
  Proposal loop;
  loop.src = ProposalSource::Model;
  loop.steps = {1, 3};  // E then W
  auto vr = ver.check(tup, loop, b);
  check(vr.v == Verdict::Incoherent && (vr.reason_bits & kReasonImmediateLoop),
        "verifier flags a 2-cycle loop as Incoherent");

  // Move into ObservedWall is Incoherent.
  BeliefState w;
  w.observe(grid3("UWU", "FVF", "UWU"), Pose{1, 1, Heading::N});
  Proposal into_wall;
  into_wall.src = ProposalSource::Model;
  into_wall.steps = {0};  // N into wall
  auto vw = ver.check(tup, into_wall, w);
  check(vw.v == Verdict::Incoherent && (vw.reason_bits & kReasonMovesIntoKnownWall),
        "verifier flags a move into ObservedWall as Incoherent");

  // Legal step is Coherent (or at least not wall/loop).
  Proposal legal;
  legal.src = ProposalSource::Symbolic;
  legal.steps = {1};  // E into floor
  legal.prior_q20 = 1000;
  auto vl = ver.check(tup, legal, w);
  check(vl.v == Verdict::Coherent, "legal step under belief is Coherent");

  // Verifier cannot raise a prior.
  const int32_t before = 1000;
  const int32_t rec = apply_verdict(before, vl, VerifierMode::Record);
  const int32_t enf_ok = apply_verdict(before, vl, VerifierMode::Enforce);
  const int32_t enf_bad = apply_verdict(before, vw, VerifierMode::Enforce);
  check(rec == before, "record mode leaves the prior unchanged");
  check(enf_ok == before && !verdict_raised_prior(before, enf_ok), "coherent enforce does not raise the prior");
  check(enf_bad == 0 && !verdict_raised_prior(before, enf_bad), "incoherent enforce zeros the prior, never raises");
  VerifierResult bogus = vl;
  bogus.score_q20 = 999999;
  check(apply_verdict(before, bogus, VerifierMode::Enforce) == before,
        "even a huge verifier score cannot raise the prior");

  // Symbolic proposals are tagged.
  StubSymbolicPeer peer;
  auto props = peer.propose(w, Subgoal::ReachExit);
  check(!props.empty() && props[0].src == ProposalSource::Symbolic, "symbolic-only proposals are tagged");
  check(!props[0].steps.empty(), "no candidate appears with a missing source or empty untagged plan");

  // Stubs: load for schema tests, refuse for runs.
  const std::string dir = labels_dir();
  check(!dir.empty(), "in-repo L6 label fixtures are at reasoning/data/labels/v0");
  const char* ids[5] = {"L6-v0-001", "L6-v0-002", "L6-v0-003", "L6-v0-004", "L6-v0-005"};
  int stub_ok = 0;
  for (const char* id : ids) {
    auto loaded = load_label_file(dir + std::string(id) + ".json");
    check(loaded.ok, (std::string("schema-test load ") + id).c_str());
    if (!loaded.ok) {
      std::printf("    err: %s\n", loaded.err.c_str());
      continue;
    }
    check(loaded.example.status == kStubStatus, (std::string(id) + " status is STUB_AWAITING_JD_REVIEW").c_str());
    check(loaded.example.schema_version == kLabelSchemaVersion, (std::string(id) + " schema l6-label-v0").c_str());
    check(loaded.example.label_hash == label_hash_hex(loaded.example),
          (std::string(id) + " label_hash matches canonical SHA-256").c_str());
    auto run = load_label_for_run(dir + std::string(id) + ".json");
    check(!run.ok && run.err.find("not for any run") != std::string::npos,
          (std::string(id) + " is refused for runs").c_str());
    if (loaded.ok) ++stub_ok;
  }
  check(stub_ok == 5, "all 5 stubs load for schema tests and are refused for runs");

  // 004: USES_HIDDEN_INFO is the crutch rule as a label.
  auto s4 = load_label_file(dir + std::string("L6-v0-004.json"));
  bool has_hidden = false;
  if (s4.ok) {
    for (const auto& c : s4.example.reason_codes)
      if (c == "USES_HIDDEN_INFO") has_hidden = true;
  }
  check(s4.ok && has_hidden && s4.example.coherence_label == "incoherent",
        "USES_HIDDEN_INFO on the vision-path stub is labeled incoherent");
}

int main() {
  clean_tree(test_root());
  test_flags_off_identity();
  test_hash_move_scratch();
  test_l1();
  test_l2();
  test_l3();
  test_l4();
  test_l5();
  test_l6();

  if (g_fail) {
    std::printf("[RL] %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("[RL] ALL PASS (contract / flags-off identity). 98/98 official-token regression is DEFERRED.\n");
  return 0;
}
