// Bolt PCHIE - in-engine pilot pieces (tests first; labeled unit fixtures, never a golden).
// EXPECTED (defined before the pilot ran in LithoKraft):
//   - MiniJson parses the MCP envelope shapes (nested object/array, escapes, numbers) and rejects junk.
//   - LkMazeText encodes the official B1 seed=1 16x16 grid as the --pchie-maze text: header + 16 rows of
//     16 chars, exactly one S/K/D/E, and decodes back to the same cells.
//   - LkMazeVoxelMap read-back: a world built from expected() verifies with 0 mismatches; one tampered cell
//     is exactly 1 mismatch; a read error is -1 (fail closed).
//   - LkActuationPlan on the unit driver's B1 actions: one BreakKey right before the Walk into the key cell,
//     one PickUp, one BreakDoor, Walks == move actions, last Walk lands on Exit; an illegal action fails closed.
//   - LkPilotController on a kinematic fake LithoKraft: reaches the exit cell, breaks 3 blocks (key + 2 door),
//     identical trace on two runs; calibrate fails closed when look input is ignored.
//   - No official chain, B1/B4-B7/B9/Stage A/B hash is touched (this test links nothing new into pchie_core).
// LithoKraft testbeds (2026-11-10; EXPECTED written before the pilot changes were built):
//   - LkMazeAreaHash (LithoKraft namespace): deterministic; one changed block id, or a moved origin, changes it;
//     a wrong id count fails closed (empty); the canonical bytes start with the domain tag.
//   - Stage A 64x64 (ScaledPitMaze seed 64): LkMazeText encodes 64 rows of 64 chars, one S, one E, two D,
//     P count == pit_count; read-back of a fake world built from it is 0 mismatches.
//   - Stage A plan on the unit driver's actions (honest miss expected): builds, Walks == move actions, no Walk
//     ever lands on a Pit cell, the last Walk ends on the grid-replayed end cell.
//   - Stage A plan on the pits-as-walls oracle path (labeled fixture): Use -> BreakDoor for BOTH door cells,
//     the last Walk lands on Exit; the kinematic fake reaches the Exit and breaks 5 blocks (key + 2x2 door).
//   - B1 plan is unchanged by the multi-cell door rule (still exactly one BreakDoor; checked above).
#include "pchie/HeadlessMaze.h"
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/ILkPilotHost.h"
#include "pchie/LkActuationPlan.h"
#include "pchie/LkMazeAreaHash.h"
#include "pchie/LkMazeText.h"
#include "pchie/LkMazeVoxelMap.h"
#include "pchie/LkPilotController.h"
#include "pchie/MiniJson.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[PILOT] %-84s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

static LkLegend legend() {
  LkLegend L;
  L.floor = 101; L.wall = 102; L.start = 103; L.exit = 104; L.key = 104; L.door = 105;
  return L;
}

// Voxel world for the fakes: (x, y, z) -> id, air = 0 when absent.
class FakeVoxels final {
 public:
  void set(int x, int y, int z, uint16_t id) { m_[std::make_tuple(x, y, z)] = id; }
  uint16_t get(int x, int y, int z) const {
    auto it = m_.find(std::make_tuple(x, y, z));
    return it == m_.end() ? uint16_t(0) : it->second;
  }
  void build(const MazeGrid& g, const LkMazeVoxelMap& map, const LkLegend& L) {
    for (int y = 0; y < g.height(); ++y)
      for (int x = 0; x < g.width(); ++x) {
        uint16_t e[LkMazeVoxelMap::kLayers];
        LkMazeVoxelMap::expected(g.cell(x, y), L, e);
        for (int k = 0; k < LkMazeVoxelMap::kLayers; ++k)
          if (e[k]) set(map.block_x(x), map.floor_y() + k, map.block_z(y), e[k]);
      }
  }

 private:
  std::map<std::tuple<int, int, int>, uint16_t> m_;
};

// Kinematic fake of the LithoKraft pilot seam: sneak walk at 1.3 blocks/s along yaw, solid blocks at
// feet/head height stop the move, mouse look at 0.0025 rad/px, attack breaks the first solid block on
// the eye ray within 5 blocks.
class FakeLithoKraft final : public ILkPilotHost {
 public:
  FakeLithoKraft(FakeVoxels& w, int fy, double x, double z, bool look_works)
      : w_(w), fy_(fy), x_(x), z_(z), look_works_(look_works) {}
  bool player_state(LkPlayerState& s, std::string&) override {
    s.x = x_; s.y = fy_ + 1.0; s.z = z_; s.yaw = yaw_; s.pitch = pitch_;
    return true;
  }
  bool get_block(int x, int y, int z, uint16_t& id, std::string&) override { id = w_.get(x, y, z); return true; }
  bool move(double f, double, bool, std::string&) override { fwd_ = f; return true; }
  bool look(double dx, double dy, std::string&) override {
    if (!look_works_) return true;
    yaw_ += dx * 0.0025;
    pitch_ -= dy * 0.0025;
    if (pitch_ > 1.55) pitch_ = 1.55;
    if (pitch_ < -1.55) pitch_ = -1.55;
    return true;
  }
  bool action(const std::string& which, std::string&) override {
    if (which != "attack") return false;
    const double ex = x_, ey = fy_ + 1.0 + LkPilotController::kEyeHeight, ez = z_;
    const double dx = std::cos(pitch_) * std::cos(yaw_), dy = std::sin(pitch_), dz = std::cos(pitch_) * std::sin(yaw_);
    for (double t = 0.0; t < 5.0; t += 0.02) {
      const int bx = int(std::floor(ex + dx * t)), by = int(std::floor(ey + dy * t)), bz = int(std::floor(ez + dz * t));
      if (w_.get(bx, by, bz)) {
        w_.set(bx, by, bz, 0);
        return true;
      }
    }
    return true;
  }
  void sleep_ms(int ms) override {
    if (fwd_ == 0.0) return;
    const double step = 1.3 * fwd_ * ms / 1000.0;
    const double nx = x_ + std::cos(yaw_) * step, nz = z_ + std::sin(yaw_) * step;
    if (!solid(nx, nz)) { x_ = nx; z_ = nz; }
  }

 private:
  bool solid(double x, double z) const {
    const double r = 0.3;
    for (double ox : {-r, r})
      for (double oz : {-r, r}) {
        const int bx = int(std::floor(x + ox)), bz = int(std::floor(z + oz));
        if (w_.get(bx, fy_ + 1, bz) || w_.get(bx, fy_ + 2, bz)) return true;
      }
    return false;
  }
  FakeVoxels& w_;
  int fy_;
  double x_, z_, yaw_ = 0.0, pitch_ = -0.32, fwd_ = 0.0;
  bool look_works_;
};

static void test_minijson() {
  MiniJson j;
  std::string err;
  const bool ok = MiniJson::parse(
      "{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"{\\\"x\\\":-12.5,\\\"blockId\\\":3}\"}],"
      "\"isError\":false},\"u\":\"\\u00e9\\n\",\"n\":null,\"e\":1.5e2}",
      j, err);
  check(ok, "MiniJson: parses an MCP tools/call response envelope");
  const MiniJson* r = j.get("result");
  check(r && r->get("content") && r->get("content")->arr.size() == 1 && !r->is_true("isError"),
        "MiniJson: nested result/content/isError are readable");
  MiniJson inner;
  const bool ok2 = r && MiniJson::parse(r->get("content")->arr[0].str_or("text", ""), inner, err);
  check(ok2 && inner.num_or("x", 0) == -12.5 && inner.num_or("blockId", 0) == 3, "MiniJson: text payload parses as JSON");
  check(j.num_or("id", 0) == 7 && j.num_or("e", 0) == 150.0 && j.str_or("u", "") == "\xc3\xa9\n",
        "MiniJson: numbers and \\u escapes decode");
  MiniJson bad;
  check(!MiniJson::parse("{\"a\":}", bad, err) && !MiniJson::parse("[1,2", bad, err) && !MiniJson::parse("{} x", bad, err),
        "MiniJson: malformed input is rejected");
  std::string deep(200, '[');
  check(!MiniJson::parse(deep, bad, err), "MiniJson: depth cap fails closed");
  check(MiniJson::quote("a\"b\\c\n") == "\"a\\\"b\\\\c\\n\"", "MiniJson: quote escapes");
}

static void test_maze_text() {
  const MazeGrid g(1, 16, 16);
  const std::string t = LkMazeText::encode(g);
  std::istringstream in(t);
  std::string magic;
  int w = 0, h = 0;
  long long seed = 0;
  in >> magic >> w >> h >> seed;
  std::string line;
  std::getline(in, line);
  std::vector<std::string> rows;
  while (std::getline(in, line)) rows.push_back(line);
  check(magic == "PCHIE-B1" && w == 16 && h == 16 && seed == 1 && rows.size() == 16, "LkMazeText: header and 16 rows");
  int S = 0, K = 0, D = 0, E = 0;
  bool widths = true, same = true;
  for (int y = 0; y < int(rows.size()); ++y) {
    if (rows[size_t(y)].size() != 16) widths = false;
    for (int x = 0; x < 16 && x < int(rows[size_t(y)].size()); ++x) {
      const char c = rows[size_t(y)][size_t(x)];
      S += c == 'S'; K += c == 'K'; D += c == 'D'; E += c == 'E';
      if (c != LkMazeText::glyph(g.cell(x, y))) same = false;
    }
  }
  check(widths && S == 1 && K == 1 && D == 1 && E == 1, "LkMazeText: 16 chars per row, exactly one S K D E");
  check(same, "LkMazeText: every char decodes back to the grid cell");
  check(LkMazeText::encode(MazeGrid(1, 16, 16)) == t, "LkMazeText: deterministic");
}

static void test_voxel_readback() {
  const MazeGrid g(1, 16, 16);
  const LkLegend L = legend();
  const LkMazeVoxelMap map(-7, 40, 300);
  FakeVoxels w;
  w.build(g, map, L);
  std::string first;
  auto rd = [&](int x, int y, int z, uint16_t& id) { id = w.get(x, y, z); return true; };
  check(map.verify(g, L, rd, first) == 0, "read-back: built world verifies with 0 mismatches");
  int dx = 0, dy = 0;
  g.find(MazeCell::DoorA, dx, dy);
  w.set(map.block_x(dx), map.floor_y() + 1, map.block_z(dy), 0);  // half-open door
  check(map.verify(g, L, rd, first) == 1 && !first.empty(), "read-back: one tampered cell is exactly 1 mismatch");
  auto fail = [&](int, int, int, uint16_t&) { return false; };
  check(map.verify(g, L, fail, first) == -1, "read-back: a read error fails closed (-1)");
  int cx = 0, cy = 0;
  map.world_to_cell(map.centre_x(3), map.centre_z(5), cx, cy);
  check(cx == 3 && cy == 5, "read-back: cell centre maps back to its cell");
}

static std::vector<int32_t> unit_actions(MazeResult& r) {
  HeadlessMaze m(1, 16, 16);
  m.set_search_budget(16);
  (void)m.run_unit("", nullptr, r);
  return r.actions;
}

static void test_plan() {
  const MazeGrid g(1, 16, 16);
  MazeResult r;
  const auto acts = unit_actions(r);
  check(r.ok && !acts.empty(), "plan: unit driver solved the maze (labeled fixture input)");
  std::vector<LkStep> steps;
  std::string err;
  const bool ok = LkActuationPlan::build(g, acts, steps, err);
  check(ok, "plan: builds for the committed actions");
  int walks = 0, bk = 0, pu = 0, bd = 0, moves = 0;
  bool bk_before_key = false;
  int kx = 0, ky = 0, ex = 0, ey = 0;
  g.find(MazeCell::KeyA, kx, ky);
  g.find(MazeCell::Exit, ex, ey);
  for (size_t i = 0; i < steps.size(); ++i) {
    switch (steps[i].kind) {
      case LkStep::Kind::Walk: ++walks; break;
      case LkStep::Kind::BreakKey:
        ++bk;
        bk_before_key = i + 1 < steps.size() && steps[i + 1].kind == LkStep::Kind::Walk && steps[i + 1].to_x == kx &&
                        steps[i + 1].to_y == ky && steps[i].to_x == kx && steps[i].to_y == ky;
        break;
      case LkStep::Kind::PickUp: ++pu; break;
      case LkStep::Kind::BreakDoor: ++bd; break;
    }
  }
  for (int32_t a : acts) moves += (a >= int32_t(MazeAction::North) && a <= int32_t(MazeAction::West));
  check(bk == 1 && bk_before_key, "plan: one BreakKey, right before the Walk into the key cell");
  check(pu == 1 && bd == 1, "plan: one PickUp and one BreakDoor");
  check(walks == moves, "plan: one Walk per committed move action");
  check(!steps.empty() && steps.back().kind == LkStep::Kind::Walk && steps.back().to_x == ex && steps.back().to_y == ey,
        "plan: last Walk lands on the Exit cell");
  // Fail closed: walking into the first wall next to Start.
  std::vector<int32_t> wall;
  wall.push_back(int32_t(MazeAction::North));  // Start (1,1): y=0 is the border wall
  check(!LkActuationPlan::build(g, wall, steps, err) && steps.empty(), "plan: an illegal action fails closed (no steps)");
  std::vector<int32_t> early_use{int32_t(MazeAction::Use)};
  check(!LkActuationPlan::build(g, early_use, steps, err), "plan: Use without the key fails closed");
}

static bool run_fake(const MazeGrid& g, const std::vector<LkStep>& steps, std::string& trace, int& breaks,
                     int& cx, int& cy) {
  const LkLegend L = legend();
  const LkMazeVoxelMap map(-7, 40, 300);
  FakeVoxels w;
  w.build(g, map, L);
  FakeLithoKraft host(w, map.floor_y(), map.centre_x(g.x()), map.centre_z(g.y()), true);
  LkPilotController pilot(host, map);
  std::string err;
  if (!pilot.calibrate(err)) return false;
  for (const LkStep& s : steps)
    if (!pilot.run_step(s, err)) {
      std::printf("[PILOT] fake run error: %s\n", err.c_str());
      return false;
    }
  LkPlayerState st;
  host.player_state(st, err);
  map.world_to_cell(st.x, st.z, cx, cy);
  trace = pilot.trace();
  breaks = pilot.breaks();
  return true;
}

static void test_controller() {
  const MazeGrid g(1, 16, 16);
  MazeResult r;
  const auto acts = unit_actions(r);
  std::vector<LkStep> steps;
  std::string err;
  (void)LkActuationPlan::build(g, acts, steps, err);
  std::string t1, t2;
  int b1 = 0, b2 = 0, x1 = 0, y1 = 0, x2 = 0, y2 = 0;
  const bool ok1 = run_fake(g, steps, t1, b1, x1, y1);
  const bool ok2 = run_fake(g, steps, t2, b2, x2, y2);
  int ex = 0, ey = 0;
  g.find(MazeCell::Exit, ex, ey);
  check(ok1 && x1 == ex && y1 == ey, "controller: kinematic fake reaches the Exit cell");
  check(b1 == 3, "controller: breaks exactly 3 blocks (key + 2 door)");
  check(ok2 && t1 == t2 && !t1.empty(), "controller: identical trace on two runs");
  // Fail closed when the game ignores look (unfocused window / cursor not captured).
  const LkLegend L = legend();
  const LkMazeVoxelMap map(-7, 40, 300);
  FakeVoxels w;
  w.build(g, map, L);
  FakeLithoKraft dead(w, map.floor_y(), map.centre_x(g.x()), map.centre_z(g.y()), false);
  LkPilotController pilot(dead, map);
  check(!pilot.calibrate(err) && !err.empty(), "controller: calibrate fails closed when look input is ignored");
}

// ---- LithoKraft testbeds (2026-11-10) -------------------------------------------------------------
static void test_area_hash() {
  const MazeGrid g(1, 16, 16);
  const LkLegend L = legend();
  const LkMazeVoxelMap map(-7, 40, 300);
  FakeVoxels w;
  w.build(g, map, L);
  std::vector<uint16_t> ids;
  std::string first;
  auto rd = [&](int x, int y, int z, uint16_t& id) { id = w.get(x, y, z); ids.push_back(id); return true; };
  const int bad = map.verify(g, L, rd, first);
  const std::string h1 = LkMazeAreaHash::sha256_hex(16, 16, map.origin_x(), map.floor_y(), map.origin_z(), ids);
  check(bad == 0 && ids.size() == 16u * 16u * 3u && h1.size() == 64, "area hash: read-back ids (768) hash to a 64-hex digest");
  check(LkMazeAreaHash::sha256_hex(16, 16, map.origin_x(), map.floor_y(), map.origin_z(), ids) == h1,
        "area hash: deterministic");
  std::vector<uint16_t> t = ids;
  t[100] = uint16_t(t[100] + 1);
  check(LkMazeAreaHash::sha256_hex(16, 16, map.origin_x(), map.floor_y(), map.origin_z(), t) != h1,
        "area hash: one changed block id changes it");
  check(LkMazeAreaHash::sha256_hex(16, 16, map.origin_x() + 1, map.floor_y(), map.origin_z(), ids) != h1,
        "area hash: a moved origin changes it");
  std::vector<uint16_t> short_ids(ids.begin(), ids.end() - 1);
  check(LkMazeAreaHash::sha256_hex(16, 16, 0, 0, 0, short_ids).empty() &&
            LkMazeAreaHash::canonical(16, 16, 0, 0, 0, short_ids).empty(),
        "area hash: wrong id count fails closed (empty)");
  const std::vector<uint8_t> c = LkMazeAreaHash::canonical(16, 16, 0, 0, 0, ids);
  const std::string tag = std::string(LkMazeAreaHash::kDomain) + "\n";
  check(c.size() > tag.size() && std::string(c.begin(), c.begin() + long(tag.size())) == tag,
        "area hash: canonical bytes start with the LithoKraft.maze_area.v1 domain tag");
}

static void test_stagea_text_and_readback() {
  const MazeGrid g = ScaledPitMaze::make();
  const std::string t = LkMazeText::encode(g);
  std::istringstream in(t);
  std::string magic;
  int w = 0, h = 0;
  long long seed = 0;
  in >> magic >> w >> h >> seed;
  std::string line;
  std::getline(in, line);
  std::vector<std::string> rows;
  while (std::getline(in, line)) rows.push_back(line);
  int S = 0, E = 0, D = 0, P = 0;
  bool widths = rows.size() == 64;
  for (const auto& r : rows) {
    if (r.size() != 64) widths = false;
    for (char ch : r) { S += ch == 'S'; E += ch == 'E'; D += ch == 'D'; P += ch == 'P'; }
  }
  check(magic == "PCHIE-B1" && w == 64 && h == 64 && seed == 64 && widths,
        "Stage A text: header 64x64 seed 64 and 64 rows of 64 chars");
  check(S == 1 && E == 1 && D == 2 && P == ScaledPitMaze::pit_count(g) && P > 0,
        "Stage A text: one S, one E, two D (2-wide door), P count == pit_count");
  const LkLegend L = legend();
  const LkMazeVoxelMap map(-7, 40, 300);
  FakeVoxels fw;
  fw.build(g, map, L);
  std::string first;
  auto rd = [&](int x, int y, int z, uint16_t& id) { id = fw.get(x, y, z); return true; };
  check(map.verify(g, L, rd, first) == 0, "Stage A read-back: fake world built from the grid verifies with 0 mismatches");
}

static void test_stagea_plan_unit() {
  const MazeGrid g = ScaledPitMaze::make();
  HeadlessMaze m(g);
  MazeResult r;
  (void)m.run_unit("", nullptr, r);
  check(!r.actions.empty(), "Stage A plan: unit driver committed actions (labeled fixture; honest miss allowed)");
  std::vector<LkStep> steps;
  std::string err;
  const bool ok = LkActuationPlan::build(g, r.actions, steps, err);
  check(ok, "Stage A plan: builds for the unit driver's committed actions");
  int walks = 0, moves = 0;
  bool pit = false;
  for (const auto& s : steps)
    if (s.kind == LkStep::Kind::Walk) {
      ++walks;
      if (g.cell(s.to_x, s.to_y) == MazeCell::Pit) pit = true;
    }
  for (int32_t a : r.actions) moves += (a >= int32_t(MazeAction::North) && a <= int32_t(MazeAction::West));
  MazeGrid rep = g;
  for (int32_t a : r.actions) (void)rep.apply(static_cast<MazeAction>(a));
  int lx = g.x(), ly = g.y();
  for (const auto& s : steps)
    if (s.kind == LkStep::Kind::Walk) { lx = s.to_x; ly = s.to_y; }
  check(walks == moves && !pit, "Stage A plan: one Walk per move action and no Walk lands on a Pit");
  check(lx == rep.x() && ly == rep.y(), "Stage A plan: last Walk ends on the grid-replayed end cell");
}

static void test_stagea_oracle_two_cell_door() {
  const MazeGrid g = ScaledPitMaze::make();
  const std::vector<MazeAction> path = MazeOracle::shortest_path_avoiding_pits(g);
  std::vector<int32_t> acts;
  for (MazeAction a : path) acts.push_back(int32_t(a));
  std::vector<LkStep> steps;
  std::string err;
  const bool ok = !acts.empty() && LkActuationPlan::build(g, acts, steps, err);
  int bd = 0, bk = 0;
  for (const auto& s : steps) { bd += s.kind == LkStep::Kind::BreakDoor; bk += s.kind == LkStep::Kind::BreakKey; }
  int ex = 0, ey = 0;
  g.find(MazeCell::Exit, ex, ey);
  check(ok && bd == 2 && bk == 1, "Stage A oracle plan: Use breaks BOTH door cells (2 BreakDoor) and one BreakKey");
  check(ok && steps.back().kind == LkStep::Kind::Walk && steps.back().to_x == ex && steps.back().to_y == ey,
        "Stage A oracle plan: last Walk lands on the Exit cell");
  std::string tr;
  int breaks = 0, cx = 0, cy = 0;
  const bool ran = ok && run_fake(g, steps, tr, breaks, cx, cy);
  check(ran && cx == ex && cy == ey && breaks == 5,
        "Stage A oracle plan: kinematic fake reaches the Exit and breaks 5 blocks (key + 2x2 door)");
}

int main() {
  std::printf("[PILOT] in-engine pilot unit tests (labeled fixtures; never a golden)\n");
  test_minijson();
  test_maze_text();
  test_voxel_readback();
  test_plan();
  test_controller();
  test_area_hash();
  test_stagea_text_and_readback();
  test_stagea_plan_unit();
  test_stagea_oracle_two_cell_door();
  std::printf("[PILOT] %s (%d failures)\n", g_fail ? "FAIL" : "ALL PASS", g_fail);
  return g_fail ? 1 : 0;
}
