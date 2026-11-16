// Built-in game plugs: the LithoKraft maze (B1 and Stage A, headless), a LithoLattice testbed stub, and a
// small Craftax-like crafting grid. All are seeded and integer-only, so an episode replays exactly.
#include "pchie/harness/GamePlug.h"

#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/NodeSeed.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/research/Digest.h"

#include <string>
#include <vector>

namespace pchie {
namespace harness {

namespace {

const char* kMoveNames[research::kMoves] = {"north", "east", "south", "west"};

// ---------------- LithoKraft maze (headless B1 / Stage A) ----------------
// He is told only what he senses himself: odometry from his start, facing, what is at his feet and next
// to him, and which ways are open. The grid is used to apply his actions and, after the episode, to grade.
class MazePlug final : public IGamePlug {
 public:
  explicit MazePlug(bool stagea) : stagea_(stagea) {}

  PlugInfo info() const override {
    PlugInfo i;
    i.name = stagea_ ? "lithokraft.maze.stagea" : "lithokraft.maze.b1";
    i.version = "1";
    i.family = "lithokraft";
    i.actions = {"north", "east", "south", "west", "pickup", "use"};
    i.instructions = stagea_ ? "Reach the exit without falling into a pit. You only know what you have seen yourself."
                             : "Find the key, open the door with it, and reach the exit. You only know what you have seen yourself.";
    return i;
  }

  PlugObservation reset(uint64_t seed) override {
    if (stagea_)
      grid_ = ScaledPitMaze::make(seed ? seed : ScaledPitMaze::kLabeledSeed);
    else
      grid_ = MazeGrid(seed ? seed : 1, MazeGrid::kDefaultWidth, MazeGrid::kDefaultHeight);
    sx_ = grid_.x();
    sy_ = grid_.y();
    heading_ = research::kNorth;
    start_ = distance();
    std::vector<uint8_t> cells;
    for (int y = 0; y < grid_.height(); ++y)
      for (int x = 0; x < grid_.width(); ++x) cells.push_back(uint8_t(grid_.cell(x, y)));
    layout_ = research::sha256_hex(cells);
    return observe();
  }

  PlugStep step(int a) override {
    static const MazeAction kMap[research::kActs] = {MazeAction::North, MazeAction::East, MazeAction::South,
                                                     MazeAction::West,  MazeAction::Pickup, MazeAction::Use};
    PlugStep s;
    if (a < 0 || a >= research::kActs) {
      s.obs = observe();
      return s;
    }
    const bool had_key = grid_.have_key();
    s.action_ok = grid_.apply(kMap[a]);
    if (s.action_ok && a < research::kMoves) heading_ = uint8_t(a);
    if (!had_key && grid_.have_key()) s.reward += 1 << 18;
    if (grid_.at_exit()) {
      s.reward += 1 << 20;
      s.done = true;
      s.success = true;
    } else if (grid_.in_pit()) {
      s.done = true;
    }
    s.obs = observe();
    return s;
  }

  PlugGrade grade() const override {
    PlugGrade g;
    g.success = grid_.at_exit();
    g.target = start_ > 0 ? start_ : 0;
    const int now = distance();
    g.progress = g.success ? g.target : (now >= 0 && start_ >= 0 ? start_ - now : 0);
    if (g.progress < 0) g.progress = 0;
    g.note = "layout=" + layout_.substr(0, 16) + (grid_.in_pit() ? " fell" : "");
    return g;
  }

  std::string state_digest() const override {
    std::vector<uint8_t> b;
    for (int y = 0; y < grid_.height(); ++y)
      for (int x = 0; x < grid_.width(); ++x) b.push_back(uint8_t(grid_.cell(x, y)));
    b.push_back(uint8_t(grid_.x()));
    b.push_back(uint8_t(grid_.y()));
    b.push_back(uint8_t(grid_.have_key()));
    b.push_back(uint8_t(grid_.door_open()));
    b.push_back(heading_);
    return research::sha256_hex(b);
  }

 private:
  int distance() const { return stagea_ ? MazeOracle::shortest_length_avoiding_pits(grid_) : MazeOracle::shortest_length(grid_); }

  PlugObservation observe() const {
    PlugObservation p;
    research::SelfObservation& o = p.self;
    o.x = grid_.x() - sx_;
    o.y = grid_.y() - sy_;
    o.heading = heading_;
    o.have_key = grid_.have_key();
    o.key_here = grid_.on_key();
    o.door_near = !grid_.door_open() && grid_.adjacent_to_door();
    o.at_exit = grid_.at_exit();
    for (int m = 0; m < research::kMoves; ++m) {
      const int nx = grid_.x() + research::move_dx(m), ny = grid_.y() + research::move_dy(m);
      if (grid_.cell(nx, ny) == MazeCell::Pit) o.danger_mask |= uint8_t(1u << m);
      if (grid_.walkable(nx, ny)) p.open_mask |= uint8_t(1u << m);
    }
    p.has_self = true;
    std::string t = "You are at (" + std::to_string(o.x) + "," + std::to_string(o.y) + ") from your start, facing " +
                    kMoveNames[heading_ & 3] + ". Open:";
    bool any = false;
    for (int m = 0; m < research::kMoves; ++m)
      if (p.open_mask & (1u << m)) { t += std::string(" ") + kMoveNames[m]; any = true; }
    if (!any) t += " none";
    t += o.have_key ? ". You hold the key." : ". You hold no key.";
    if (o.key_here) t += " A key is at your feet.";
    if (o.door_near) t += " A shut door is next to you.";
    for (int m = 0; m < research::kMoves; ++m)
      if (o.danger_mask & (1u << m)) t += std::string(" A pit is to the ") + kMoveNames[m] + ".";
    if (o.at_exit) t += " You are at the exit.";
    p.text = t;
    return p;
  }

  bool stagea_;
  MazeGrid grid_;
  int sx_ = 0, sy_ = 0;
  uint8_t heading_ = research::kNorth;
  int start_ = 0;
  std::string layout_;
};

// ---------------- LithoLattice testbed (stub) ----------------
// A placeholder so the socket is exercised end to end. The real testbeds will answer through the engine's
// MCP seam; until then this is a seeded corridor and info().stub is true.
class TestbedStub final : public IGamePlug {
 public:
  PlugInfo info() const override {
    PlugInfo i;
    i.name = "litholattice.testbed.stub";
    i.version = "0";
    i.family = "litholattice";
    i.stub = true;
    i.actions = {"forward", "back", "wait"};
    i.instructions = "Testbed stub: walk to the end of the corridor.";
    return i;
  }
  PlugObservation reset(uint64_t seed) override {
    len_ = 6 + int(seed % 5);
    at_ = 0;
    return observe();
  }
  PlugStep step(int a) override {
    PlugStep s;
    if (a == 0 && at_ < len_ - 1) { ++at_; s.action_ok = true; }
    else if (a == 1 && at_ > 0) { --at_; s.action_ok = true; }
    else if (a == 2) s.action_ok = true;
    if (at_ == len_ - 1) { s.done = true; s.success = true; s.reward = 1 << 20; }
    s.obs = observe();
    return s;
  }
  PlugGrade grade() const override {
    PlugGrade g;
    g.success = at_ == len_ - 1;
    g.progress = at_;
    g.target = len_ - 1;
    g.note = "stub";
    return g;
  }
  std::string state_digest() const override { return research::sha256_hex("testbed.stub|" + std::to_string(len_) + "|" + std::to_string(at_)); }

 private:
  PlugObservation observe() const {
    PlugObservation p;
    p.text = "Corridor cell " + std::to_string(at_ + 1) + " of " + std::to_string(len_) + ".";
    std::string g(size_t(len_), '.');
    g[size_t(len_ - 1)] = 'E';
    g[size_t(at_)] = '@';
    p.glyphs = g;
    p.glyph_cols = len_;
    p.glyph_rows = 1;
    return p;
  }
  int len_ = 6, at_ = 0;
};

// ---------------- Craftax-like crafting grid ----------------
// 9x9 world: '.' grass, 'T' tree, 'S' stone, '~' water, 'B' crafting table, '@' him. He sees a 5x5 window.
// Achievements: collect wood, place a table (2 wood), make a wooden pickaxe (1 wood, next to a table),
// collect stone (needs the pickaxe). Collecting stone ends the episode as a success.
class CraftaxLike final : public IGamePlug {
 public:
  static constexpr int kW = 9, kH = 9, kView = 5, kAchievements = 4;

  PlugInfo info() const override {
    PlugInfo i;
    i.name = "craftax_like.grid";
    i.version = "1";
    i.family = "craftax_like";
    i.actions = {"noop", "north", "east", "south", "west", "do", "place_table", "make_wood_pickaxe"};
    i.instructions = "Collect wood, place a crafting table, make a wooden pickaxe, then mine stone.";
    return i;
  }

  PlugObservation reset(uint64_t seed) override {
    uint64_t rng = NodeSeed::Splitmix64(seed ^ 0xC4AF7A11ULL);
    auto next = [&]() { rng = NodeSeed::Splitmix64(rng); return rng; };
    cells_.assign(size_t(kW * kH), '.');
    bool tree = false, stone = false;
    for (int y = 0; y < kH; ++y)
      for (int x = 0; x < kW; ++x) {
        char c = '.';
        if (x == 0 || y == 0 || x == kW - 1 || y == kH - 1) {
          c = '~';
        } else {
          const uint64_t r = next() % 100;
          if (r < 15) c = 'T';
          else if (r < 25) c = 'S';
          else if (r < 30) c = '~';
        }
        cells_[size_t(y * kW + x)] = c;
      }
    px_ = kW / 2;
    py_ = kH / 2;
    at(px_, py_) = '.';
    for (char c : cells_) { tree = tree || c == 'T'; stone = stone || c == 'S'; }
    if (!tree) at(2, 2) = 'T';
    if (!stone) at(6, 6) = 'S';
    face_ = 1;
    wood_ = stone_ = 0;
    pickaxe_ = false;
    ach_ = 0;
    return observe();
  }

  PlugStep step(int a) override {
    PlugStep s;
    static const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
    const uint8_t before = ach_;
    if (a >= 1 && a <= 4) {
      face_ = a - 1;
      const int nx = px_ + dx[face_], ny = py_ + dy[face_];
      if (at(nx, ny) == '.') { px_ = nx; py_ = ny; s.action_ok = true; }
    } else if (a == 5) {
      char& f = at(px_ + dx[face_], py_ + dy[face_]);
      if (f == 'T') { ++wood_; ach_ |= 1; s.action_ok = true; }
      else if (f == 'S' && pickaxe_) { ++stone_; f = '.'; ach_ |= 8; s.action_ok = true; }
    } else if (a == 6) {
      char& f = at(px_ + dx[face_], py_ + dy[face_]);
      if (wood_ >= 2 && f == '.') { f = 'B'; wood_ -= 2; ach_ |= 2; s.action_ok = true; }
    } else if (a == 7) {
      if (wood_ >= 1 && !pickaxe_ && near_table()) { --wood_; pickaxe_ = true; ach_ |= 4; s.action_ok = true; }
    } else if (a == 0) {
      s.action_ok = true;
    }
    for (int b = 0; b < kAchievements; ++b)
      if ((ach_ & (1u << b)) && !(before & (1u << b))) s.reward += 1 << 16;
    if (ach_ & 8) { s.done = true; s.success = true; }
    s.obs = observe();
    return s;
  }

  PlugGrade grade() const override {
    PlugGrade g;
    g.success = (ach_ & 8) != 0;
    for (int b = 0; b < kAchievements; ++b) g.progress += (ach_ >> b) & 1;
    g.target = kAchievements;
    return g;
  }

  std::string state_digest() const override {
    std::string s(cells_.begin(), cells_.end());
    s += "|" + std::to_string(px_) + "," + std::to_string(py_) + "|" + std::to_string(face_) + "|" + std::to_string(wood_) + "|" +
         std::to_string(stone_) + "|" + std::to_string(int(pickaxe_)) + "|" + std::to_string(int(ach_));
    return research::sha256_hex(s);
  }

 private:
  char& at(int x, int y) {
    if (x < 0 || y < 0 || x >= kW || y >= kH) { oob_ = '~'; return oob_; }  // per game, never shared
    return cells_[size_t(y * kW + x)];
  }
  char get(int x, int y) const { return (x < 0 || y < 0 || x >= kW || y >= kH) ? '~' : cells_[size_t(y * kW + x)]; }
  bool near_table() const {
    for (int y = -1; y <= 1; ++y)
      for (int x = -1; x <= 1; ++x)
        if (get(px_ + x, py_ + y) == 'B') return true;
    return false;
  }
  PlugObservation observe() const {
    PlugObservation p;
    static const char* kFace[4] = {"north", "east", "south", "west"};
    int n = 0;
    for (int b = 0; b < kAchievements; ++b) n += (ach_ >> b) & 1;
    p.text = "Wood " + std::to_string(wood_) + ", stone " + std::to_string(stone_) + ", pickaxe " + (pickaxe_ ? "yes" : "no") +
             ". Facing " + kFace[face_] + ". Achievements " + std::to_string(n) + "/" + std::to_string(kAchievements) + ".";
    std::string g;
    for (int y = -kView / 2; y <= kView / 2; ++y) {
      for (int x = -kView / 2; x <= kView / 2; ++x) g.push_back((x == 0 && y == 0) ? '@' : get(px_ + x, py_ + y));
      if (y < kView / 2) g.push_back('\n');
    }
    p.glyphs = g;
    p.glyph_cols = kView;
    p.glyph_rows = kView;
    return p;
  }

  std::vector<char> cells_;
  int px_ = 0, py_ = 0, face_ = 1, wood_ = 0, stone_ = 0;
  bool pickaxe_ = false;
  uint8_t ach_ = 0;
  char oob_ = '~';
};

}  // namespace

std::unique_ptr<IGamePlug> make_maze_plug(bool stagea) { return std::make_unique<MazePlug>(stagea); }
std::unique_ptr<IGamePlug> make_testbed_stub() { return std::make_unique<TestbedStub>(); }
std::unique_ptr<IGamePlug> make_craftax_like() { return std::make_unique<CraftaxLike>(); }

}  // namespace harness
}  // namespace pchie
