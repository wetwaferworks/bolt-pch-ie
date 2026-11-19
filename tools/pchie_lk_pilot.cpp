// Bolt PCHIE - pchie_lk_pilot: the PCHIE reasoning model used directly inside LithoKraft.
//
// What it does (2026-11-10):
//   1. Builds the official B1 grid (seed 1, 16x16, 1-key) and has the PCHIE stack decide the
//      run on it: real Little Nemo (NemotronModel, CPU) by default, or --unit for the null-model
//      unit driver. Same driver, budget and gate settings as the official B1 test; the chain is a
//      fresh side chain (never the official log). Nothing is locked; hashes are compared only.
//   2. Launches LithoKraft with --mcp-stdio --pchie-maze (the engine's own MCP server; the agent
//      starts with NO permissions). The player approves it in the console:
//         mcp grant world_read player_control
//      (--grant auto writes that line into a scratch autoexec.lth under LL_DATA_ROOT; it is the
//      launcher's own approval, never the agent granting itself.)
//   3. Reads the maze back from the world, cell by cell (world_get_block), and refuses to move unless
//      it equals the grid the model decided on.
//   4. Carries out each committed action with the player's normal input (player_move / player_look /
//      player_action attack), reading the real position back after every step.
// View-only example tooling. Not a golden, not the official chain, not the renderer bar.
//
// LithoKraft testbeds (2026-11-10; backup *.pre-lkbed.bak):
//   --maze b1|stagea  b1 = official B1 seed 1 16x16 (default, unchanged); stagea = the labeled Stage A
//                     64x64 fixture (ScaledPitMaze seed 64, 2x2 rooms, pits with a safe lane), decided
//                     exactly as pchie_scaled_pit_maze_test does (default budgets, Mirror-Sim side log on).
//                     An honest Stage A miss is still actuated in-engine (the committed actions are the
//                     model's); nothing is retuned.
//   --no-vsync        passes --no-vsync to the game.
//   --settings FILE   copies FILE to <out>/lk_data/config/settings.json (scratch data root only).
//   The read-back block ids are hashed as LithoKraft.maze_area.v1 (LithoKraft namespace, never a PCHIE hash).
#include "pchie/HeadlessMaze.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/MazeGrid.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/MirrorSimSideLog.h"
#include "pchie/LkMazeAreaHash.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/LithoKraftEngineHost.h"
#include "pchie/LkActuationPlan.h"
#include "pchie/LkMazeText.h"
#include "pchie/LkMazeVoxelMap.h"
#include "pchie/LkPilotController.h"
#include "pchie/McpStdioClient.h"
#include "pchie/NemotronModel.h"
#include "pchie/Sha256.h"
#include "pchie/Sha256HashChain.h"
#include "pchie/ModelPath.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
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
namespace fs = std::filesystem;

// Official recorded (provisional, unlocked) B1 values, compared only (PROGRESS.txt wander-fix table).
static const char* kB1HWorldSha = "457668e71e112ab1fee54d08220ad5dc445245f53a1a91390c95d8c324be8ec5";
static const char* kB1RealActionsSha = "0c21184625c3c03b8cfb4259213604f731098618cbbb578e0473eaa44333aa4e";
// Current post-BOS baseline real H_bolt (B4/B5/B6) and the pre-BOS pilot value; unit H_bolt.
static const char* kRealHBoltPostBos = "05f42ccc5b7ae53d06e99c3c3eb42585ee4016873fa86fb345c2553fbe8b9ae0";
static const char* kRealHBoltPreBosP7 = "cf1e57ca5fc50cc5f1c7a639c542f8180862daf2a6a1e793744a310160d72ba8";
static const char* kUnitHBolt = "de479b0a2dd6c4ad69dc379c34139a3a64a7a27d6c9074d59e22a6b632e79533";
// Stage A real side log (pchie_scaled_pit_maze_test, FAR baseline; labeled fixture, never a golden).
static const char* kStageASideLogSha = "8c4d61e37a62a95fceda95f67b1d923cf0dc58ca5f8b534d8e19b330b56a65a0";
static const unsigned long long kStageASideLogFnv = 0xE51DE1CF28C9ED86ULL;

static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }

static std::string hex(const std::vector<uint8_t>& v) { return GgufWeightsHash::to_hex(v); }

static double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Args {
  std::string input_template;
  std::string game, out, model = model_path().c_str(), renderer = "vulkan", resolution = "1280x720", frame_dump, ffmpeg;
  std::string maze = "b1", settings;
  bool unit = false, auto_grant = true, decide_only = false, no_vsync = false;
  bool tiptoe = false;  // scratch tiptoe crossing check (opt-in, default off)
  bool loop_guard = true;   // loop guard v2, on by default; --loop-guard off restores the old default
  StreamAuthFlags stream_auth;  // TESLA + attestation on by default
  int hold_ms = 4000, grant_timeout_s = 300;
};

static bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&](std::string& v) { if (i + 1 < argc) { v = argv[++i]; return true; } return false; };
    if (const int n = neml::parse_flag(argc, argv, i, neml::process_config()); n != 0) {  // --neml, --neml-reason
      if (n < 0) return false;
      i += n - 1;
      continue;
    }
    if (k == "--game") { if (!next(a.game)) return false; }
    else if (k == "--out") { if (!next(a.out)) return false; }
    else if (k == "--model") { if (!next(a.model)) return false; }
    else if (k == "--renderer") { if (!next(a.renderer)) return false; }
    else if (k == "--resolution") { if (!next(a.resolution)) return false; }
    else if (k == "--frame-dump") { if (!next(a.frame_dump)) return false; }
    else if (k == "--ffmpeg") { if (!next(a.ffmpeg)) return false; }
    else if (k == "--input-template") { if (!next(a.input_template)) return false; }
    else if (k == "--unit") a.unit = true;
    else if (k == "--maze") { if (!next(a.maze)) return false; if (a.maze != "b1" && a.maze != "stagea") return false; }
    else if (k == "--settings") { if (!next(a.settings)) return false; }
    else if (k == "--no-vsync") a.no_vsync = true;
    else if (k == "--decide-only") a.decide_only = true;
    else if (k == "--tiptoe") a.tiptoe = true;
    else if (k == "--loop-guard") {  // --loop-guard [on|off]; a bare --loop-guard (the old opt-in) means on
      a.loop_guard = true;
      if (i + 1 < argc && (std::string(argv[i + 1]) == "on" || std::string(argv[i + 1]) == "off")) a.loop_guard = std::string(argv[++i]) == "on";
    }
    else if (k == "--tesla") {
      std::string v; if (!next(v) || (v != "on" && v != "off")) return false;
      a.stream_auth.tesla = v == "on";
    }
    else if (k == "--tesla-seed") {  // os (default) or dev-insecure (forgeable maze-seed key chain, unit work only)
      std::string v; if (!next(v) || (v != "os" && v != "dev-insecure")) return false;
      a.stream_auth.tesla_dev_insecure_seed = v == "dev-insecure";
    }
    else if (k == "--attest") {
      std::string v; if (!next(v) || (v != "on" && v != "off")) return false;
      a.stream_auth.attest = v == "on";
    }
    else if (k == "--grant") { std::string v; if (!next(v)) return false; a.auto_grant = (v == "auto"); }
    else if (k == "--hold-ms") { std::string v; if (!next(v)) return false; a.hold_ms = std::atoi(v.c_str()); }
    else if (k == "--grant-timeout") { std::string v; if (!next(v)) return false; a.grant_timeout_s = std::atoi(v.c_str()); }
    else return false;
  }
  return !a.out.empty() && (a.decide_only || !a.game.empty());
}

static bool parse_legend(const std::string& log, LkLegend& L, int& ox, int& fy, int& oz) {
  const size_t p = log.find("pchie maze: legend ");
  if (p == std::string::npos) return false;
  unsigned f = 0, w = 0, s = 0, e = 0, k = 0, d = 0;
  if (std::sscanf(log.c_str() + p, "pchie maze: legend floor=%u wall=%u start=%u exit=%u key=%u door=%u origin=(%d,%d,%d)",
                  &f, &w, &s, &e, &k, &d, &ox, &fy, &oz) != 9)
    return false;
  L.floor = uint16_t(f); L.wall = uint16_t(w); L.start = uint16_t(s);
  L.exit = uint16_t(e); L.key = uint16_t(k); L.door = uint16_t(d);
  return true;
}

static std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    std::fprintf(stderr,
                 "usage: pchie_lk_pilot --out DIR (--game LithoKraft.exe | --decide-only) [--unit] [--model gguf]\n"
                 "       [--renderer vulkan|gl33|software] [--resolution WxH] [--frame-dump DIR] [--ffmpeg exe]\n"
                 "       [--grant auto|manual] [--grant-timeout s] [--hold-ms ms] [--input-template input.json]\n"
                 "       [--maze b1|stagea] [--no-vsync] [--settings settings.json]\n"
                 "       [--neml on|off|DIR] [--neml-reason on|off]\n"
                 "       [--tesla on|off] [--tesla-seed os|dev-insecure] [--attest on|off]\n");
    return 2;
  }
  fs::create_directories(a.out);
  std::ofstream rep(fs::path(a.out) / "pilot_report.txt", std::ios::binary);
  auto say = [&](const char* fmt, auto... xs) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf), fmt, xs...);
    std::printf("[PILOT] %s\n", buf);
    std::fflush(stdout);
    rep << buf << "\n";
    rep.flush();
  };
  say("pchie_lk_pilot: view-only example run, never a golden, side chain only (not the official log)");
  const double t_start = now_s();

  // 1. Decide on the grid: official B1 (seed 1 16x16) or the labeled Stage A 64x64 fixture.
  const bool stagea = (a.maze == "stagea");
  const MazeGrid grid = stagea ? ScaledPitMaze::make() : MazeGrid(1, 16, 16);
  const std::string maze_text = LkMazeText::encode(grid);
  const fs::path maze_path = fs::path(a.out) / (stagea ? "pchie_stagea_seed64_64x64.txt" : "pchie_b1_seed1_16x16.txt");
  say("maze %s: %dx%d seed %llu", stagea ? "Stage A (labeled fixture, never a golden)" : "official maze", grid.width(),
      grid.height(), (unsigned long long)grid.seed());
  { std::ofstream m(maze_path, std::ios::binary); m << maze_text; }
  say("maze file %s", maze_path.string().c_str());

  NemotronModel model;
  if (!a.unit) {
    std::string err;
    const double t0 = now_s();
    if (!model.load(a.model, err)) { say("FAIL model load: %s", err.c_str()); return 1; }
    model.init_cache(32, 8);
    say("model Little Nemo loaded in %.1f s (CPU): %s", now_s() - t0, a.model.c_str());
  }
  HeadlessMaze driver = stagea ? HeadlessMaze(grid) : HeadlessMaze(1, 16, 16);
  MirrorSimSideLog mlog;
  if (stagea) driver.enable_mirror_sim(&mlog);  // as pchie_scaled_pit_maze_test (default budgets)
  else driver.set_search_budget(16);            // as the official B1 test
  if (a.tiptoe) {
    driver.enable_tiptoe(true);
    say("TIPTOE (scratch, opt-in): crossing check ON -- at a pit-adjacent cell each step is checked, taken one block, and footing verified; never an official run");
  }
  driver.enable_loop_guard(a.loop_guard);
  driver.set_stream_auth(a.stream_auth);
  say(a.loop_guard ? "LOOP GUARD v2 (default): a state entered a 4th time forces a new decision point + reasoning check"
                   : "LOOP GUARD OFF (--loop-guard off): old default behavior");
  say("TESLA %s, attestation %s (verdict Unattested without a TPM quote)", a.stream_auth.tesla ? "ON (default)" : "OFF (--tesla off)",
      a.stream_auth.attest ? "ON (default)" : "OFF (--attest off)");
  // Episodic memory: recording on by default; the memory prior reads a snapshot taken now (empty ledger = not used,
  // so the run is the same as with memory off).
  const neml::Config& ncfg = neml::process_config();
  neml::Snapshot mem_snap = neml::Snapshot::empty();
  if (ncfg.memory_prior) mem_snap = neml::Snapshot::take(ncfg);
  if (!mem_snap.ok) {
    // A damaged or busy ledger is never used as an empty memory: stop, so the run is not quietly a different run.
    say("FAIL NEML: the memory ledger was refused: %s", mem_snap.why.c_str());
    say("     fix or move the ledger folder (pchie_neml --verify says what is wrong), or run with --neml-reason off");
    return 1;
  }
  if (ncfg.memory_prior && !mem_snap.runs.empty()) {
    driver.set_memory(&mem_snap);
    say("NEML memory prior ON: snapshot of %zu past runs sha256=%s (hashed into the side chain; hashes will differ from the official run)",
        mem_snap.runs.size(), hex(mem_snap.sha256).c_str());
  } else {
    say("NEML memory prior %s", ncfg.memory_prior ? "on, ledger empty: not used" : "OFF (--neml-reason off)");
  }
  say("NEML recording %s", ncfg.record ? "ON (default): one receipt at teardown" : "OFF");
  Sha256HashChain side_chain;
  side_chain.reset();
  MazeResult res;
  const double t_dec = now_s();
  const bool dec_ok = a.unit ? driver.run_unit("", &side_chain, res) : driver.run(model, "", &side_chain, res);
  const double dec_s = now_s() - t_dec;
  if (res.neml_receipt_id) say("NEML receipt run_id=%u events=%zu forks=%d dead_ends=%d memory_nudges=%d", res.neml_receipt_id,
                               res.neml_events, res.neml_forks, res.neml_dead_ends, res.neml_memory_nudges);
  else if (ncfg.record && !a.unit) say("NEML receipt not written: %s", res.neml_err.c_str());
  const auto act_bytes = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(res.actions.data()),
                                              reinterpret_cast<const uint8_t*>(res.actions.data()) + res.actions.size() * sizeof(int32_t));
  const std::string act_sha = hex(Sha256::hash(act_bytes.data(), act_bytes.size()));
  const std::string hw_sha = hex(res.h_world_sha);
  say("decide (%s): ok=%d exit=%d path=%d bfs=%d in %.1f s; side-chain len=%zu head=%s",
      a.unit ? "unit null-model driver" : "real Little Nemo", int(dec_ok), int(res.reached_exit), res.path_length,
      res.bfs_length, dec_s, side_chain.length(), hex(side_chain.head()).c_str());
  const std::string head_sha = hex(side_chain.head());
  if (!stagea) {
    say("decide H_world sha256=%s (%s official recorded maze run)", hw_sha.c_str(), hw_sha == kB1HWorldSha ? "matches" : "DIFFERS from");
    say("decide actions sha256=%s%s", act_sha.c_str(),
        a.unit ? " (unit driver; official real-actions value not applicable)"
               : (act_sha == kB1RealActionsSha ? " (matches official recorded maze real actions)" : " (DIFFERS from official recorded maze real actions)"));
    if (a.unit)
      say("decide H_bolt side chain %s unit baseline de479b0a... (len %zu)", head_sha == kUnitHBolt ? "MATCHES" : "DIFFERS from",
          side_chain.length());
    else
      say("decide H_bolt side chain %s (current post-BOS baseline 05f42ccc...: %s; pre-BOS pilot cf1e57ca...: %s) len %zu",
          head_sha.c_str(), head_sha == kRealHBoltPostBos ? "MATCH" : "differs", head_sha == kRealHBoltPreBosP7 ? "match" : "differs",
          side_chain.length());
    if (!dec_ok || !res.ok) { say("FAIL decide: %s", res.err.c_str()); return 1; }
  } else {
    const std::string ml_sha = hex(mlog.sha256());
    say("decide H_world sha256=%s (Stage A; no official record)", hw_sha.c_str());
    say("decide actions sha256=%s count=%zu", act_sha.c_str(), res.actions.size());
    say("decide Stage A side log sha256=%s fnv=0x%016llX (%s FAR-baseline record 8c4d61e3.../0xE51DE1CF28C9ED86%s)",
        ml_sha.c_str(), (unsigned long long)mlog.fnv(),
        (ml_sha == kStageASideLogSha && mlog.fnv() == kStageASideLogFnv) ? "MATCHES" : "DIFFERS from",
        a.unit ? "; the record line is printed by the real-weight run of pchie_scaled_pit_maze_test" : "");
    say("decide Stage A: allowed=%d refused=%d exhaust=%d err='%s'", mlog.allowed_count(), mlog.refused_count(),
        int(res.budget_exhausted), res.err.c_str());
    if (res.actions.empty()) { say("FAIL decide: no committed actions (%s)", res.err.c_str()); return 1; }
    if (!res.reached_exit)
      say("HONEST MISS (decide): the model did not reach the exit within its budget; the committed actions are actuated "
          "in-engine as decided; no retune");
  }
  if (a.loop_guard) {
    say("loop guard: %d trigger(s)", res.loop_triggers);
    for (const std::string& l : res.loop_log) say("%s", l.c_str());
  }
  std::vector<LkStep> steps;
  std::string perr;
  if (!LkActuationPlan::build(grid, res.actions, steps, perr)) { say("FAIL plan: %s", perr.c_str()); return 1; }
  say("actuation plan: %zu steps for %zu committed actions", steps.size(), res.actions.size());
  MazeGrid decided_end = grid;
  for (int32_t act : res.actions) (void)decided_end.apply(static_cast<MazeAction>(act));
  {  // retreat diagnostics (both modes): immediate 180 reversals, overall and from pit-adjacent cells
    MazeGrid w = grid;
    int rev = 0, rev_cross = 0, first_cross_step = -1, far_x = 0, far_d = -1;
    int32_t prev = 0;
    auto opp = [](int32_t a, int32_t b) {
      return (a == int32_t(MazeAction::North) && b == int32_t(MazeAction::South)) || (a == int32_t(MazeAction::South) && b == int32_t(MazeAction::North)) ||
             (a == int32_t(MazeAction::East) && b == int32_t(MazeAction::West)) || (a == int32_t(MazeAction::West) && b == int32_t(MazeAction::East));
    };
    for (size_t i = 0; i < res.actions.size(); ++i) {
      const bool cross = HeadlessMaze::pit_adjacent(w, w.x(), w.y());
      if (opp(prev, res.actions[i])) { ++rev; if (cross) ++rev_cross; }
      prev = res.actions[i];
      (void)w.apply(static_cast<MazeAction>(res.actions[i]));
      if (first_cross_step < 0 && HeadlessMaze::pit_adjacent(w, w.x(), w.y())) first_cross_step = int(i);
      const int d = w.x() + w.y();
      if (d > far_d) { far_d = d; far_x = int(i); }
    }
    say("retreats: %d immediate 180s (%d from pit-adjacent cells); first pit-adjacent cell entered at step %d; tiptoe checks=%d crossings=%d footing_ok=%d footing_fail=%d",
        rev, rev_cross, first_cross_step, res.tiptoe_checks, res.tiptoe_crossings, res.tiptoe_footing_ok, res.tiptoe_footing_fail);
    (void)far_x;
  }
  say("decided end cell (%d,%d)%s", decided_end.x(), decided_end.y(), decided_end.at_exit() ? " = Exit" : " (not the exit)");
  if (a.decide_only) { say("decide-only: done"); return 0; }

  // 2. Launch LithoKraft with its MCP server (deny-by-default).
  const fs::path data_root = fs::absolute(fs::path(a.out) / "lk_data");
  fs::create_directories(data_root / "config");
  fs::create_directories(data_root / "logs");
  if (a.auto_grant) {
    std::ofstream ax(data_root / "config" / "autoexec.lth", std::ios::binary);
    ax << "// pchie_lk_pilot --grant auto: the launcher's own approval for this scratch run only.\n"
          "wait 256\nmcp grant world_read player_control\n";
  } else {
    fs::remove(data_root / "config" / "autoexec.lth");
  }
  // The MCP pilot turns player_action into synthetic KEY events using the current bindings; with the
  // default map Break is mouse-only, so an MCP "attack" is dropped as unbound. This scratch data root
  // (never a player's own config) adds key:J to Break so the model breaks blocks through the normal input path.
  if (!a.input_template.empty()) {
    std::string in = slurp(a.input_template);
    const std::string from = "\"Break\": [\"mouse:Left\"", to = "\"Break\": [\"mouse:Left\", \"key:J\"";
    const size_t at = in.find(from);
    if (at == std::string::npos || in.find("key:J") != std::string::npos) {
      say("FAIL: input template has no default Break binding (or key:J is taken): %s", a.input_template.c_str());
      return 1;
    }
    in.replace(at, from.size(), to);
    std::ofstream f(data_root / "config" / "input.json", std::ios::binary);
    f << in;
    say("scratch input.json: Break += key:J (MCP attack needs a key binding)");
  }
  if (!a.settings.empty()) {
    std::error_code ec;
    fs::copy_file(a.settings, data_root / "config" / "settings.json", fs::copy_options::overwrite_existing, ec);
    if (ec) { say("FAIL: could not copy settings profile %s", a.settings.c_str()); return 1; }
    say("scratch settings.json profile: %s", a.settings.c_str());
  }
  SetEnvironmentVariableW(L"LL_DATA_ROOT", data_root.wstring().c_str());
  if (!a.ffmpeg.empty()) SetEnvironmentVariableW(L"LL_FFMPEG", widen(a.ffmpeg).c_str());
  std::string cl = "\"" + a.game + "\" --mcp-stdio --pchie-maze \"" + fs::absolute(maze_path).string() +
                   "\" --spawn maze --renderer " + a.renderer + " --display windowed --resolution " + a.resolution +
                   " --seed 1 --no-save --mute --capture";
  if (a.no_vsync) cl += " --no-vsync";
  if (!a.frame_dump.empty()) cl += " --frame-dump \"" + a.frame_dump + "\"";
  say("launch: %s", cl.c_str());
  const fs::path game_log = fs::path(a.out) / "game_stderr.log";
  McpStdioClient client;
  std::string err;
  if (!client.start(widen(cl), widen(fs::path(a.game).parent_path().string()), game_log.wstring(), err)) {
    say("FAIL launch: %s", err.c_str());
    return 1;
  }
  LithoKraftEngineHost host(client);
  int rc = 1;
  do {
    if (!host.initialize(err)) { say("FAIL initialize: %s", err.c_str()); break; }
    say("MCP initialized (server '%s'; %d non-JSON boot log lines skipped on stdout); waiting for the player's grant (%s)", host.server_name().c_str(),
        client.skipped_lines(), a.auto_grant ? "autoexec" : "type `mcp grant world_read player_control` in the console");
    if (!host.wait_for_grant(a.grant_timeout_s * 1000, err)) { say("FAIL grant: %s", err.c_str()); break; }
    say("granted: world_read player_control");

    // 3. Read the maze back from the world.
    LkLegend L;
    int ox = 0, fy = 0, oz = 0;
    bool have = false;
    for (int i = 0; i < 40 && !have; ++i) {
      have = parse_legend(slurp(game_log), L, ox, fy, oz) ||
             parse_legend(slurp(data_root / "logs" / "lithokraft.log"), L, ox, fy, oz);
      if (!have) Sleep(250);
    }
    if (!have) { say("FAIL: no 'pchie maze: legend' line in the game log (scene not built?)"); break; }
    const LkMazeVoxelMap map(ox, fy, oz);
    std::string first;
    const double t_v = now_s();
    std::vector<uint16_t> area_ids;
    area_ids.reserve(size_t(grid.width()) * size_t(grid.height()) * 3u);
    const int bad = map.verify(grid, L, [&](int x, int y, int z, uint16_t& id) {
      const bool okr = host.get_block(x, y, z, id, err);
      if (okr) area_ids.push_back(id);
      return okr;
    }, first);
    say("read-back: %d of %d cells mismatch (%s) in %.1f s; origin (%d,%d,%d)", bad, grid.width() * grid.height(),
        bad == 0 ? "engine maze == decided grid" : first.c_str(), now_s() - t_v, ox, fy, oz);
    if (bad != 0) { say("FAIL: engine maze does not match the grid; no input sent"); break; }
    say("LithoKraft.maze_area.v1 sha256=%s (LithoKraft namespace; %zu block ids; legend floor=%u wall=%u start=%u exit=%u key=%u door=%u; provisional, unlocked)",
        LkMazeAreaHash::sha256_hex(grid.width(), grid.height(), ox, fy, oz, area_ids).c_str(), area_ids.size(), unsigned(L.floor),
        unsigned(L.wall), unsigned(L.start), unsigned(L.exit), unsigned(L.key), unsigned(L.door));
    LkPlayerState s0;
    if (!host.player_state(s0, err)) { say("FAIL state: %s", err.c_str()); break; }
    int scx = 0, scy = 0;
    map.world_to_cell(s0.x, s0.z, scx, scy);
    if (scx != grid.x() || scy != grid.y()) { say("FAIL: player not on Start (%d,%d)", scx, scy); break; }

    // 4. Act, with the player's own input.
    client.focus_window();
    Sleep(500);
    LkPilotController pilot(host, map);
    if (!pilot.calibrate(err)) { say("FAIL calibrate: %s", err.c_str()); break; }
    say("look gain yaw=%.6f pitch=%.6f rad/px", pilot.yaw_gain(), pilot.pitch_gain());
    const double t_act = now_s();
    bool ok = true;
    int tiptoe_engine_ok = 0;
    for (size_t i = 0; i < steps.size() && ok; ++i) {
      ok = pilot.run_step(steps[i], err);
      if (!ok) say("FAIL step %zu of %zu (action %d): %s", i, steps.size(), steps[i].action_index, err.c_str());
      if (ok && a.tiptoe && HeadlessMaze::pit_adjacent(grid, steps[i].to_x, steps[i].to_y)) {
        // tiptoe: the pilot already walks sneaking (slow, edge-safe); verify footing in the engine after the step
        LkPlayerState sf;
        int fx = 0, fyc = 0;
        if (!host.player_state(sf, err)) { ok = false; say("FAIL tiptoe footing read at step %zu: %s", i, err.c_str()); }
        else {
          map.world_to_cell(sf.x, sf.z, fx, fyc);
          const bool footing = std::fabs(sf.y - s0.y) < 0.25;
          if (fx != steps[i].to_x || fyc != steps[i].to_y || !footing) {
            ok = false;
            say("FAIL tiptoe footing at step %zu: cell (%d,%d) want (%d,%d), y %.3f vs start %.3f", i, fx, fyc, steps[i].to_x, steps[i].to_y, sf.y, s0.y);
          } else {
            ++tiptoe_engine_ok;
          }
        }
      }
      else if (stagea && (i + 1) % 100 == 0)
        say("progress: %zu of %zu steps, %d walks, %.1f s", i + 1, steps.size(), pilot.walks(), now_s() - t_act);
    }
    if (!ok) {
      const std::string tr = pilot.trace();
      say("partial engine trace sha256=%s (%d walks before the failure; side log only)",
          hex(Sha256::hash(reinterpret_cast<const uint8_t*>(tr.data()), tr.size())).c_str(), pilot.walks());
      { std::ofstream t(fs::path(a.out) / "engine_trace_partial.txt", std::ios::binary); t << tr << "\n"; }
    }
    if (!ok) break;
    LkPlayerState s1;
    if (!host.player_state(s1, err)) { say("FAIL state: %s", err.c_str()); break; }
    int ecx = 0, ecy = 0;
    map.world_to_cell(s1.x, s1.z, ecx, ecy);
    int ex = 0, ey = 0;
    grid.find(MazeCell::Exit, ex, ey);
    const std::string trace = pilot.trace();
    const std::string trace_sha = hex(Sha256::hash(reinterpret_cast<const uint8_t*>(trace.data()), trace.size()));
    say("act: %d walks, %d blocks broken, %.1f s; player ended in cell (%d,%d), exit is (%d,%d): %s", pilot.walks(),
        pilot.breaks(), now_s() - t_act, ecx, ecy, ex, ey, (ecx == ex && ecy == ey) ? "EXIT REACHED IN ENGINE" : "NOT at exit");
    say("engine trace sha256=%s (side log only, view-only, never a golden; %d MCP tool calls)", trace_sha.c_str(), host.tool_calls());
    if (a.tiptoe) say("tiptoe engine footing verified on %d crossing steps", tiptoe_engine_ok);
    { std::ofstream t(fs::path(a.out) / "engine_trace.txt", std::ios::binary); t << trace << "\n"; }
    const bool at_decided = (ecx == decided_end.x() && ecy == decided_end.y());
    say("player end cell %s the decided end cell (%d,%d)", at_decided ? "EQUALS" : "DIFFERS from", decided_end.x(), decided_end.y());
    // B1: success = exit reached. Stage A: success = the engine carried out every committed action and the
    // player stands where the model's plan ends (an honest decide miss stays a miss; rc 3 marks it).
    rc = stagea ? (at_decided ? ((ecx == ex && ecy == ey) ? 0 : 3) : 1) : ((ecx == ex && ecy == ey) ? 0 : 1);
    (void)host.move(0, 0, false, err);
    Sleep(DWORD(a.hold_ms));
  } while (false);
  client.stop(8000);
  say("game closed; total %.1f s; exit code %d", now_s() - t_start, rc);
  return rc;
}
