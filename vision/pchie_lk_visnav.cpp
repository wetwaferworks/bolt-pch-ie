// Vision-only navigation (2026-11-11): pchie_lk_visnav.
// "Planning and the vision boundary are one reasoning path." Little Nemo decides every mid-run action
// from the LIVE Vulkan framebuffer of the player's view (vision.capture_display, LL_MCP_DISPLAY_VISION=1).
// There is no precomputed step list, no xpath, no BFS distance, and NO world get_block map in the decision:
//   * the observation is built from pixels only (visnav encoder: nearest-prototype pixel classes, then a
//     per-direction depth profile read from fixed camera-geometry bands of the centre strip);
//   * the candidate filter is the encoder's own verdict (wall / pit seen in the next cell), never the map;
//   * the choice is a short search on KV forks of one primed branch: an observation fork carries the look-around
//     (all four live views); from it, a prior fork scores the move words and one option fork per vision-allowed
//     action reads that action and the view it leads into and scores " yes"/" no" (the model's own value).
//     choice = argmax(value + prior_w * log prior). Pointwise option scoring is order-free. Not PUCT/MCTS.
//   * memory is the agent's own: recent moves and a dead-reckoned visit count integrated from its OWN
//     committed moves (never the engine position, never the map).
// World reads remain ONLY for: the pre-run build check (game built the intended maze), actuation
// (walk_to closed loop / attack confirm), the anti-cheat audit before an attack (abort if a wall would be
// broken), the result check (exit / fall) and the side log. None of them feed the observation or the choice.
// Untrusted observation text goes through IModel::tokenize_untrusted (parse_special=false, no BOS).
// Never official, never a golden. Official pilot, build-cpu, official headers are NOT modified.
// SCRATCH FIX (2026-11-11, after the B1 cap-end at the (11,13) glass door; both opt-in, defaults reproduce the run):
//   * encoder cfg key `lamp_beats_glass 1`: a band is L only if lamp >= lamp_pct AND lamp > glass (a glass-plurality band
//     is no longer preempted by lamp-coloured light on the frosted glass). Default 0 = old lamp-first rule.
//   * --way-shut: a timed-out walk that recentres marks that directed edge shut (own memory, one failure is enough);
//     a shut direction is dropped from the walk leaves and tagged (shut) instead of (new). With the key carried, open
//     becomes a leaf on a shut direction whose first non-floor occluder is a door in cell 1 or cell 2 (cell 2 = the floor
//     at a raised door's feet). A door in cell 2 WITHOUT a failed walk never makes open a leaf (no hallway false opens).
// Scratch reach (engine reach is 8 blocks, Little Nemo's interaction range is half of that, 4):
//   * --reach N (opt-in, 0..4, default 0 = off = the way-shut binary's behaviour, byte for byte on the decision path).
//     LithoKraft's own Break reach is 8 blocks in survival / 16 in creative (game.cpp Game::handleWorldActions:
//     `const float reach = creativeMode() ? 16.f : 8.f;`, hit distance <= reach + 0.5 along the camera ray). That
//     engine constant is NOT changed; only Little Nemo's decision policy uses N.
//   * One maze cell is one world block (LkMazeVoxelMap::block_x = origin + cx), and encoder band k is the floor of
//     the k-th cell ahead, so band k == k world blocks EXCEPT one measured case: the lit glass door one block ahead
//     reads "floor, door" (glass + lamp light at its feet in band 1). occluder_world() maps that case back to 1 block
//     (band-1 signature L>=3 and G>=12 and F<70: 55/55 B1 door-at-1 frames, 0/485 other frames), or a shut walk proves 1.
//   * take / open become leaves when the first occluder that way is the right object (key / door; carrying rules
//     unchanged) at 1..N world blocks with only floor in between. A ranged key needs glass-free key bands (G<=10).
//     A shut walk toward a door read 3+ blocks away never ranges (the bump contradicts the clear floor).
//   * a ranged interaction (2+ blocks) is audited before any attack: a wall target stops the run INVALID (as before);
//     a target whose blocks are not the chosen object (not a key / not a door) is REFUSED (no attack sent), counted, and
//     that ranged leaf is dropped from this cell/direction (own memory, like way-shut) so it can walk closer instead.
// One driver. --way-shut, --reach, --act-blank, and --neml-inject default off; NEML recording is on (--neml off). --b1-seed 1, --b1-size 16,
// and --game-res 640x360 are the nav2 defaults. The CUDA decide path is a separate compile
// (PCHIE_SCRATCH_CUDA_VISNAV). With the opt-in flags off, the decision path is the reach-off path.
#define NOMINMAX
#include "pchie/MazeGrid.h"
#include "pchie/ScaledPitMaze.h"
#include "pchie/GgufWeightsHash.h"
#include "pchie/JoinJobDispatcher.h"
#include "pchie/LithoKraftEngineHost.h"
#include "pchie/LkMazeText.h"
#include "pchie/LkMazeVoxelMap.h"
#include "pchie/LkPilotController.h"
#include "pchie/McpStdioClient.h"
#include "pchie/NemotronModel.h"
#ifdef PCHIE_SCRATCH_CUDA_VISNAV
#include "cuda/CudaNemotronModel.h"  // Scratch labeled CUDA IModel adapter
#endif
#include "pchie/neml/NemlStore.h"  // episodic memory: a receipt per nav run (on by default; --neml off)
#include <ctime>
#include "pchie/Sha256.h"
#include "pchie/UntrustedText.h"
#include "pchie/ModelPath.h"
#include "pchie/research/AsciiObservation.h"  // --obs ascii (opt-in)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
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

static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }
static std::string hex(const std::vector<uint8_t>& v) { return GgufWeightsHash::to_hex(v); }
static std::string sha_hex(const std::string& s) { return hex(Sha256::hash(reinterpret_cast<const uint8_t*>(s.data()), s.size())); }
static double now_s() { using namespace std::chrono; return duration<double>(steady_clock::now().time_since_epoch()).count(); }
static std::string slurp(const fs::path& p) { std::ifstream f(p, std::ios::binary); std::ostringstream ss; ss << f.rdbuf(); return ss.str(); }
static bool parse_legend(const std::string& log, LkLegend& L, int& ox, int& fy, int& oz) {
  const size_t p = log.find("pchie maze: legend ");
  if (p == std::string::npos) return false;
  unsigned f = 0, w = 0, s = 0, e = 0, k = 0, d = 0;
  if (std::sscanf(log.c_str() + p, "pchie maze: legend floor=%u wall=%u start=%u exit=%u key=%u door=%u origin=(%d,%d,%d)", &f, &w, &s, &e, &k, &d, &ox, &fy, &oz) != 9) return false;
  L.floor = uint16_t(f); L.wall = uint16_t(w); L.start = uint16_t(s); L.exit = uint16_t(e); L.key = uint16_t(k); L.door = uint16_t(d);
  return true;
}

// VISNAV_PURE_BEGIN png
// ---------------- base64 + stored-deflate PNG (engine encoder format) ----------------
static bool b64decode(const std::string& in, std::vector<uint8_t>& out) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
  };
  out.clear(); uint32_t acc = 0; int bits = 0;
  for (char c : in) { if (c == '=') break; const int v = val(c); if (v < 0) continue; acc = (acc << 6) | uint32_t(v); bits += 6; if (bits >= 8) { bits -= 8; out.push_back(uint8_t((acc >> bits) & 0xFF)); } }
  return !out.empty();
}
static uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }
static bool png_decode(const std::vector<uint8_t>& png, int& w, int& h, std::vector<uint8_t>& rgb, std::string& err) {
  static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  if (png.size() < 8 || std::memcmp(png.data(), sig, 8) != 0) { err = "png signature"; return false; }
  size_t i = 8; std::vector<uint8_t> z; w = h = 0;
  while (i + 12 <= png.size()) {
    const uint32_t len = be32(&png[i]);
    const std::string type(reinterpret_cast<const char*>(&png[i + 4]), 4);
    if (i + 12 + len > png.size()) { err = "png truncated"; return false; }
    const uint8_t* d = &png[i + 8];
    if (type == "IHDR") { w = int(be32(d)); h = int(be32(d + 4)); if (d[8] != 8 || d[9] != 2) { err = "png not RGB8"; return false; } }
    else if (type == "IDAT") z.insert(z.end(), d, d + len);
    else if (type == "IEND") break;
    i += 12 + len;
  }
  if (w <= 0 || h <= 0 || z.size() < 2) { err = "png no image"; return false; }
  size_t p = 2; std::vector<uint8_t> raw;
  for (;;) {
    if (p + 5 > z.size()) { err = "zlib truncated"; return false; }
    const uint8_t hd = z[p];
    if (((hd >> 1) & 3) != 0) { err = "not a stored deflate block"; return false; }
    const uint16_t L = uint16_t(z[p + 1] | (z[p + 2] << 8));
    p += 5;
    if (p + L > z.size()) { err = "stored block truncated"; return false; }
    raw.insert(raw.end(), z.begin() + p, z.begin() + p + L);
    p += L;
    if (hd & 1) break;
  }
  const size_t stride = size_t(w) * 3;
  if (raw.size() < size_t(h) * (stride + 1)) { err = "png short raw"; return false; }
  rgb.resize(size_t(h) * stride);
  for (int y = 0; y < h; ++y) { if (raw[size_t(y) * (stride + 1)] != 0) { err = "png filter != 0"; return false; } std::memcpy(&rgb[size_t(y) * stride], &raw[size_t(y) * (stride + 1) + 1], stride); }
  return true;
}
// VISNAV_PURE_END png

// VISNAV_PURE_BEGIN encoder
// ---------------- visnav encoder (pixels only) ----------------
// Pixel classes: F floor, W wall, P pit, L lamp, G glass, S sky. Nearest prototype (integer RGB distance),
// prototypes measured on the labeled probe fixture (visnav_enc.cfg). Bands = rows of the 320x180 frame where
// the floor of the k-th cell ahead projects for the fixed look pitch (camera geometry, measured on the
// probe fixture; independent of the maze).
struct Proto { int r, g, b; char c; };
struct EncCfg {
  std::vector<Proto> protos;
  int band_top[5] = {0, 99, 67, 43, 30};   // band k covers rows [band_top[k], band_bot[k])
  int band_bot[5] = {0, 127, 99, 67, 43};
  int strip_half = 18;                     // centre strip half width (px, 320-wide frame)
  int pit_pct = 25, lamp_pct = 20, glass_pct = 30;
  int lamp_beats_glass = 0;                // 1: L needs lamp > glass as well as lamp >= lamp_pct (scratch fix 2026-11-11)
  void defaults() {
    protos = {{235, 238, 242, 'F'}, {160, 165, 172, 'F'}, {120, 124, 130, 'F'}, {200, 120, 80, 'F'},
              {36, 46, 75, 'W'}, {48, 59, 94, 'W'}, {27, 34, 56, 'W'},
              {56, 58, 64, 'P'}, {74, 76, 80, 'P'}, {40, 42, 48, 'P'},
              {255, 230, 140, 'L'}, {210, 230, 240, 'G'}, {4, 4, 6, 'S'}};
  }
  bool load(const std::string& path, std::string& err) {
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return false; }
    protos.clear();
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#' || line[0] == ';') continue;
      char key[32] = {0};
      if (std::sscanf(line.c_str(), "%31s", key) != 1) continue;
      const std::string k(key);
      if (k == "proto") { int r, g, b; char c; if (std::sscanf(line.c_str(), "proto %d %d %d %c", &r, &g, &b, &c) == 4) protos.push_back({r, g, b, c}); }
      else if (k == "band") { int i, t, bt; if (std::sscanf(line.c_str(), "band %d %d %d", &i, &t, &bt) == 3 && i >= 1 && i <= 4) { band_top[i] = t; band_bot[i] = bt; } }
      else if (k == "strip_half") std::sscanf(line.c_str(), "strip_half %d", &strip_half);
      else if (k == "pit_pct") std::sscanf(line.c_str(), "pit_pct %d", &pit_pct);
      else if (k == "lamp_pct") std::sscanf(line.c_str(), "lamp_pct %d", &lamp_pct);
      else if (k == "glass_pct") std::sscanf(line.c_str(), "glass_pct %d", &glass_pct);
      else if (k == "lamp_beats_glass") std::sscanf(line.c_str(), "lamp_beats_glass %d", &lamp_beats_glass);
    }
    if (protos.empty()) { err = "no prototypes in " + path; return false; }
    return true;
  }
};
static char px_class(const EncCfg& c, int r, int g, int b) {
  long best = -1; char s = '?';
  for (const auto& p : c.protos) { const long d = long(r - p.r) * (r - p.r) + long(g - p.g) * (g - p.g) + long(b - p.b) * (b - p.b); if (best < 0 || d < best) { best = d; s = p.c; } }
  return s;
}
enum CellSeen { kFloor = 0, kWall, kPit, kKey, kDoor, kExit, kLamp, kUnknown };
static const char* kSeenWord[8] = {"floor", "wall", "pit", "key", "door", "exit", "glow", "unknown"};
struct DirView {
  int cells[4] = {kUnknown, kUnknown, kUnknown, kUnknown};  // cell 1..4 ahead as seen (after occlusion walk)
  int n = 0;                                                  // number of cells described
  bool open_beyond = false;
  char band_cls[4] = {'?', '?', '?', '?'};                    // raw band verdicts
  std::string band_pct;                                       // per-band class percentages (log)
  int bpct[4][6] = {};                                        // per-band class percentages F W P L G S (ints; --reach ranging)
};
static DirView encode_dir(const EncCfg& c, int w, int h, const std::vector<uint8_t>& rgb) {
  DirView v;
  const double sx = double(w) / 320.0, sy = double(h) / 180.0;
  char raw[4];
  for (int k = 1; k <= 4; ++k) {
    int cnt[7] = {0, 0, 0, 0, 0, 0, 0};  // F W P L G S ?
    static const char C[7] = {'F', 'W', 'P', 'L', 'G', 'S', '?'};
    const int y0 = int(c.band_top[k] * sy), y1 = int(c.band_bot[k] * sy);
    const int x0 = int((160 - c.strip_half) * sx), x1 = int((160 + c.strip_half) * sx);
    int tot = 0;
    for (int y = std::max(0, y0); y < std::min(h, y1); ++y)
      for (int x = std::max(0, x0); x < std::min(w, x1); ++x) {
        const uint8_t* q = &rgb[(size_t(y) * w + x) * 3];
        const char pc = px_class(c, q[0], q[1], q[2]);
        int i = 6; for (int j = 0; j < 6; ++j) if (C[j] == pc) i = j;
        ++cnt[i]; ++tot;
      }
    tot = std::max(1, tot);
    char vb = 'F';
    if (cnt[3] * 100 >= c.lamp_pct * tot && (!c.lamp_beats_glass || cnt[3] > cnt[4])) vb = 'L';
    else if (cnt[4] * 100 >= c.glass_pct * tot) vb = 'G';
    else if (cnt[2] * 100 >= c.pit_pct * tot) vb = 'P';
    else { int b = 0; for (int j = 1; j < 7; ++j) if (cnt[j] > cnt[b]) b = j; vb = C[b]; if (vb == 'S' || vb == '?') vb = 'W'; }
    raw[k - 1] = vb; v.band_cls[k - 1] = vb;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%sb%d[F%d W%d P%d L%d G%d S%d]", k > 1 ? " " : "", k, cnt[0] * 100 / tot, cnt[1] * 100 / tot, cnt[2] * 100 / tot,
                  cnt[3] * 100 / tot, cnt[4] * 100 / tot, cnt[5] * 100 / tot);
    v.band_pct += buf;
    for (int j = 0; j < 6; ++j) v.bpct[k - 1][j] = cnt[j] * 100 / tot;
  }
  // Occlusion walk: floor continues; anything else is the thing in that cell and hides what lies beyond.
  for (int k = 0; k < 4; ++k) {
    const char b = raw[k];
    int cs = kFloor;
    if (b == 'F') cs = kFloor;
    else if (b == 'W') cs = kWall;
    else if (b == 'P') cs = kPit;
    else if (b == 'G') cs = kDoor;
    else if (b == 'L') cs = (k + 1 < 4) ? (raw[k + 1] == 'L' ? kKey : kExit) : kLamp;  // raised block covers the next band too
    v.cells[k] = cs; v.n = k + 1;
    if (cs != kFloor) return v;
  }
  v.open_beyond = true;
  return v;
}
static std::string dir_phrase(const DirView& v) {
  std::string s;
  for (int k = 0; k < v.n; ++k) s += std::string(k ? ", " : "") + kSeenWord[v.cells[k]];
  if (v.open_beyond) s += ", more";
  return s;
}
// VISNAV_PURE_END encoder
// Legacy 12x6 grid (vulkfb rule classifier) kept for the log only, to compare encodings.
static std::string legacy_grid(int w, int h, const std::vector<uint8_t>& rgb) {
  const int cols = 12, rows = 6, hc = std::max(rows, int(h * 0.70));
  std::string out;
  for (int gy = 0; gy < rows; ++gy) {
    for (int gx = 0; gx < cols; ++gx) {
      int cnt[4] = {0, 0, 0, 0}; static const char S[4] = {'~', '.', '#', 'O'};
      const int x0 = gx * w / cols, x1 = (gx + 1) * w / cols, y0 = gy * hc / rows, y1 = (gy + 1) * hc / rows;
      for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
        const uint8_t* q = &rgb[(size_t(y) * w + x) * 3]; const int l = (q[0] + q[1] + q[2]) / 3;
        const int k = l < 12 ? 0 : (l >= 110 ? 1 : (q[2] - q[0] >= 18 ? 2 : 3)); ++cnt[k];
      }
      int b = 0; for (int k = 1; k < 4; ++k) if (cnt[k] > cnt[b]) b = k;
      if (cnt[3] * 4 >= std::max(1, (x1 - x0) * (y1 - y0))) b = 3;
      out += S[b];
    }
    out += '\n';
  }
  return out;
}

// ---------------- Little Nemo: one primed branch, policy fork + leaf forks ----------------
static const char kRules[] =
    "Little Nemo walks through a maze in first person, one cell per move. Before every move it looks forward, right, "
    "back and left. For each direction it reports the cells it sees in that direction, nearest first: floor (walkable), "
    "wall (blocked), pit (a hole; falling in ends the run), key (a glowing block), door (a glass door), exit (the glowing "
    "goal floor), more (the floor goes on). Little Nemo wants to reach the exit. The way to the exit may be shut by a "
    "glass door that opens only with the key. When the key is right next to Little Nemo, taking it is always the best "
    "move. When Little Nemo carries the key and a door is right next to it, opening the door is always the best move. "
    "It never walks into a wall or a pit. It prefers ways it has not walked before, it turns around at dead ends, and it "
    "does not walk back and forth.\n";
static const char kSees[] = "\nLittle Nemo sees:\n";
static const char kAsk[] = "\nLittle Nemo's next move:";
static const char kOptPre[] = "\nOption: ";
static const char kOptAsk[] = "\nIs this the best move for Little Nemo right now? Answer:";
// VISNAV_PURE_BEGIN acts
static const char* kRel[4] = {"forward", "right", "back", "left"};
enum Act { kFwd = 0, kRight, kBack, kLeft, kTake, kOpen, kNAct };
static const char* kActWord[kNAct] = {"forward", "right", "back", "left", "take", "open"};
// VISNAV_PURE_END acts

struct Brain {
  ModelBranch base;
  std::string neml_prefix;  // SCRATCH NEML digest (empty unless --neml-inject)
  int32_t ids[kNAct];
  int32_t yes = -1, no = -1;
  size_t base_len = 0, toks = 0;
  template <size_t N> static std::vector<int32_t> lit(IModel& m, const char (&s)[N]) { return m.tokenize_no_bos(std::string(s, N - 1)); }
  int reach = 0;  // --reach N: one trusted sentence is added to the rules (only when N > 0)
  static std::string rules_text(int reach) {
    std::string r(kRules, sizeof(kRules) - 1);
    if (reach > 0) {
      const std::string n = std::to_string(reach);
      r.insert(r.size() - 1, " Little Nemo's arm reaches " + n + " blocks: a key or a door up to " + n +
                                 " blocks ahead, with only floor in between, is within reach and counts as right next to it.");
    }
    return r;
  }
  bool prime(IModel& m, std::string& err) {
    const auto t = m.tokenize(neml_prefix + rules_text(reach));  // trusted prefix, one leading BOS (existing contract)
    base = m.new_branch();
    for (int32_t x : t) m.forward(base, x);
    base_len = t.size();
    for (int k = 0; k < kNAct; ++k) { const auto w = m.tokenize_no_bos(std::string(" ") + kActWord[k]); if (w.empty()) { err = "empty act word"; return false; } ids[k] = w[0]; }
    for (int a = 0; a < kNAct; ++a) for (int b = a + 1; b < kNAct; ++b) if (ids[a] == ids[b]) { err = "act words share a first token"; return false; }
    yes = m.tokenize_no_bos(" yes")[0]; no = m.tokenize_no_bos(" no")[0];
    if (yes == no) { err = "yes/no share a token"; return false; }
    return true;
  }
  void feed(IModel& m, ModelBranch& f, const std::vector<int32_t>& v) { for (int32_t x : v) m.forward(f, x); toks += v.size(); }
  // KV fork of the primed rules carrying the look-around (untrusted observation tokens, no BOS)
  ModelBranch observe(IModel& m, const UntrustedText& obs) {
    ModelBranch f = m.fork(base);
    feed(m, f, lit(m, kSees)); feed(m, f, m.tokenize_untrusted(obs.str()));
    return f;
  }
  // policy prior: logits of the six action words, on a fork of the observation branch
  void prior(IModel& m, const ModelBranch& obsb, float out[kNAct]) {
    ModelBranch f = m.fork(obsb);
    feed(m, f, lit(m, kAsk));
    const auto& lg = m.logits();
    for (int k = 0; k < kNAct; ++k) out[k] = lg[size_t(ids[k])];
    m.release(f);
  }
  // option value: yes - no for one candidate action, on a fork of the observation branch (order-free, pointwise)
  float option(IModel& m, const ModelBranch& obsb, const UntrustedText& opt) {
    ModelBranch f = m.fork(obsb);
    feed(m, f, lit(m, kOptPre)); feed(m, f, m.tokenize_untrusted(opt.str())); feed(m, f, lit(m, kOptAsk));
    const auto& lg = m.logits();
    const float v = lg[size_t(yes)] - lg[size_t(no)];
    m.release(f);
    return v;
  }
};
static double log_sigmoid(double x) { return x >= 0 ? -std::log1p(std::exp(-x)) : x - std::log1p(std::exp(x)); }


// VISNAV_PURE_BEGIN gate
// ---------------- one decision: shared by --nav and --policy-test ----------------
struct DecideIn {
  std::string phrase[4];   // pixel-derived description per relative direction
  int cell1[4] = {7, 7, 7, 7};  // encoder verdict for the next cell per direction
  int nvis[4] = {0, 0, 0, 0};   // agent's own dead-reckoned visit count of that next cell
  bool shut[4] = {false, false, false, false};  // --way-shut: own memory of a timed-out walk from this pose in that direction
  int occ[4] = {7, 7, 7, 7};    // first non-floor cell kind per direction (kUnknown if the view is all floor)
  int occ_at[4] = {0, 0, 0, 0}; // its 1-based cell index (0 = none)
  int reach = 0;                // --reach N: interaction range in world blocks (0 = off)
  int occ_world[4] = {0, 0, 0, 0};  // world blocks to that occluder (occluder_world; 0 = none)
  bool key_clean[4] = {true, true, true, true};  // key bands glass-free (ranged take guard)
  bool reach_miss[4] = {false, false, false, false};  // own memory: a ranged interaction from this cell that way was refused / failed
  bool carrying = false;
  std::string recent;      // last moves (agent's own)
  std::string ascii;       // --obs ascii: checked Nemo View rows for the live view (empty = pixels only, prompt unchanged)
};
struct DecideOut {
  std::string obs, blank, leaf_log;
  float P[kNAct], PB[kNAct];
  bool allowed[kNAct];
  int take_w = 0, open_w = 0;  // world blocks to the take / open target (1 = right next to Little Nemo)
  int take_r = -1, open_r = -1, n_allowed = 0, raw_top = 0, pol_top = -1, blank_top = -1, choice = -1;
  size_t ntok = 0;
  double secs = 0;
};
static std::string mem_word(int n) { return n == 0 ? std::string("new") : (n == 1 ? std::string("walked once") : "walked " + std::to_string(n) + " times"); }
// visit tag per direction: a shut direction is never "(new)"
static std::string dir_tag(const DecideIn& in, int r) { return in.shut[r] ? std::string("shut") : mem_word(in.nvis[r]); }
// first non-floor occluder of an encoded view (kUnknown/0 if the view is all floor)
static void first_occluder(const DirView& v, int& kind, int& at) {
  kind = kUnknown; at = 0;
  for (int k = 0; k < v.n; ++k) if (v.cells[k] != kFloor) { kind = v.cells[k]; at = k + 1; return; }
}
// --reach ranging (pure). Band 1 of the lit glass door one block ahead: lamp light + glass over floor (measured, B1).
static bool glass_at_feet(const DirView& v) { return v.bpct[0][3] >= 3 && v.bpct[0][4] >= 12 && v.bpct[0][0] < 70; }
// world blocks to the first occluder (1 maze cell = 1 world block); 0 if none
static int occluder_world(const DirView& v, int kind, int at) {
  if (at <= 0) return 0;
  if (kind == kDoor && at == 2 && glass_at_feet(v)) return 1;
  return at;
}
// a key read at band `at` covers bands at and at+1; ranged take only if those bands carry no glass (d030 door read as key)
static bool key_bands_clean(const DirView& v, int at) {
  if (at <= 0) return false;
  for (int k = at - 1; k <= std::min(at, 3); ++k) if (v.bpct[k][4] > 10) return false;
  return true;
}
static void gate(const DecideIn& in, DecideOut& o);
// observation text (live view and blank control); pure. --reach: a ranged target (2+ blocks, or the band-2 door at 1)
// is tagged in the LIVE view only ("; the door is within reach"); the blank control never carries it.
static void build_obs(const DecideIn& in, DecideOut& o) {
  o.obs.clear(); o.blank.clear(); o.leaf_log.clear();
  DecideOut g; if (in.reach > 0) gate(in, g);
  for (int r = 0; r < 4; ++r) {
    const bool walk = in.cell1[r] == kFloor || in.cell1[r] == kExit;
    std::string rt;
    if (in.reach > 0 && walk) {
      if (g.take_r == r && in.occ[r] == kKey) rt = "; the key is within reach";
      else if (g.open_r == r && in.occ[r] == kDoor) rt = "; the door is within reach";
    }
    o.obs += std::string(kRel[r]) + ": " + in.phrase[r] + ((walk || in.shut[r]) ? " (" + dir_tag(in, r) + rt + ")" : "") + "\n";
    o.blank += std::string(kRel[r]) + ": unknown (" + dir_tag(in, r) + ")\n";
  }
  o.obs += in.ascii;  // live view only; the blank control never carries it
  const std::string tail = std::string("key carried: ") + (in.carrying ? "yes" : "no") + "\nlast moves: " + in.recent;
  o.obs += tail; o.blank += tail;
}
// vision-only allowed set (the hard gate); pure. Without any shut direction this is exactly the pre-fix rule.
static void gate(const DecideIn& in, DecideOut& o) {
  for (int k = 0; k < kNAct; ++k) o.allowed[k] = false;
  o.take_r = o.open_r = -1; o.take_w = o.open_w = 0;
  for (int r = 0; r < 4; ++r) {
    const int c1 = in.cell1[r];
    if (!in.shut[r]) {  // the way is shut: one timed-out walk drops this direction from the walk leaves
      if (c1 == kFloor || c1 == kExit) o.allowed[r] = true;
      if (c1 == kKey && in.carrying) o.allowed[r] = true;  // only one key exists: once carried, a raised glow may be the exit (a wrong guess just bumps)
    }
    if (c1 == kKey && !in.carrying && o.take_r < 0) o.take_r = r;
    if (c1 == kDoor && in.carrying && o.open_r < 0) o.open_r = r;
    // a raised door one cell ahead is seen as "floor, door" (band 1 = the floor at its feet). Promote it to "next to
    // Little Nemo" ONLY after a walk that way was blocked; a far door down a hallway never makes open a leaf.
    if (in.shut[r] && in.carrying && o.open_r < 0 && c1 == kFloor && in.occ[r] == kDoor && in.occ_at[r] == 2) o.open_r = r;
    if (o.take_r == r && o.take_w == 0) o.take_w = 1;
    if (o.open_r == r && o.open_w == 0) o.open_w = 1;
    // --reach N: the first occluder that way is the right object within N world blocks, only floor in between
    if (in.reach > 0 && !in.reach_miss[r] && !(in.shut[r] && in.occ_at[r] >= 3)) {
      const int w = in.occ_world[r];
      if (w >= 1 && w <= in.reach) {
        if (in.occ[r] == kKey && !in.carrying && in.key_clean[r] && o.take_r < 0) { o.take_r = r; o.take_w = w; }
        if (in.occ[r] == kDoor && in.carrying && o.open_r < 0) { o.open_r = r; o.open_w = w; }
      }
    }
  }
  if (o.take_r >= 0) o.allowed[kTake] = true;
  if (o.open_r >= 0) o.allowed[kOpen] = true;
  o.n_allowed = 0; for (int k = 0; k < kNAct; ++k) o.n_allowed += o.allowed[k];
  if (o.n_allowed == 0) { for (int k = 0; k < 4; ++k) if (!in.shut[k]) { o.allowed[k] = true; ++o.n_allowed; } }
  if (o.n_allowed == 0) { for (int k = 0; k < 4; ++k) o.allowed[k] = true; o.n_allowed = 4; }
}
// own memory of shut directed edges (odometry cell + absolute direction); pure
struct ShutMemory {
  std::set<std::tuple<int, int, int>> e;
  void mark(int x, int y, int d) { e.insert(std::make_tuple(x, y, d)); }
  void clear(int x, int y, int d) { e.erase(std::make_tuple(x, y, d)); }
  bool is_shut(int x, int y, int d) const { return e.count(std::make_tuple(x, y, d)) != 0; }
};
// fill the --reach fields of one direction from its encoded view (pure)
static void reach_fields(const DirView& v, DecideIn& in, int r) {
  in.occ_world[r] = occluder_world(v, in.occ[r], in.occ_at[r]);
  in.key_clean[r] = in.occ[r] == kKey ? key_bands_clean(v, in.occ_at[r]) : true;
}
// VISNAV_PURE_END gate
static void decide(IModel& model, Brain& brain, const DecideIn& in, double prior_w, bool use_blank, DecideOut& o) {
  const double t0 = now_s();
  build_obs(in, o);
  gate(in, o);
  const size_t tok0 = brain.toks;
  auto opt_text = [&](int k, bool with_view) -> std::string {
    if (k < 4) {
      std::string t = std::string("walk ") + kRel[k];
      if (with_view) t += ", toward: " + in.phrase[k];
      return t + " (" + dir_tag(in, k) + ")";
    }
    if (!with_view) return k == kTake ? std::string("take (what is ahead is unknown)") : std::string("open (what is ahead is unknown)");
    if (k == kTake && o.take_w >= 2) return std::string("take the key, which is within Little Nemo's reach, ") + std::to_string(o.take_w) + " blocks ahead (" + kRel[o.take_r] + ")";
    if (k == kOpen && o.open_w >= 2) return std::string("open the glass door with the key; the door is within Little Nemo's reach, ") + std::to_string(o.open_w) + " blocks ahead (" + kRel[o.open_r] + ")";
    if (k == kTake) return std::string("take the key, which is right next to Little Nemo (") + kRel[o.take_r] + ")";
    return std::string("open the glass door with the key; the door is right next to Little Nemo (") + kRel[o.open_r] + ")";
  };
  auto search = [&](const std::string& obs_text, bool with_view, float P[kNAct], float V[kNAct], int& best, int& ptop) {
    ModelBranch ob = brain.observe(model, UntrustedText(obs_text));
    brain.prior(model, ob, P);
    double mx = -1e30; for (int k = 0; k < kNAct; ++k) if (o.allowed[k]) mx = std::max(mx, double(P[k]));
    double z = 0; for (int k = 0; k < kNAct; ++k) if (o.allowed[k]) z += std::exp(double(P[k]) - mx);
    double lp[kNAct]; for (int k = 0; k < kNAct; ++k) lp[k] = o.allowed[k] ? (double(P[k]) - mx - std::log(z)) : -1e30;
    ptop = -1; for (int k = 0; k < kNAct; ++k) if (o.allowed[k] && (ptop < 0 || lp[k] > lp[ptop])) ptop = k;
    best = -1; double bs = -1e30;
    for (int k = 0; k < kNAct; ++k) {
      V[k] = 0;
      if (!o.allowed[k]) continue;
      if (o.n_allowed > 1) V[k] = brain.option(model, ob, UntrustedText(opt_text(k, with_view)));
      const double sc = double(V[k]) + prior_w * lp[k];
      if (with_view) { char b[160]; std::snprintf(b, sizeof(b), "%s%s v=%.2f lp=%.2f s=%.2f", o.leaf_log.empty() ? "" : "; ", kActWord[k], V[k], lp[k], sc); o.leaf_log += b; }
      if (sc > bs) { bs = sc; best = k; }
    }
    model.release(ob);
  };
  float V[kNAct], VB[kNAct]; int bb = -1, pb = -1;
  search(o.obs, true, o.P, V, o.choice, o.pol_top);
  if (use_blank) search(o.blank, false, o.PB, VB, bb, pb); else { for (int k = 0; k < kNAct; ++k) o.PB[k] = o.P[k]; bb = o.choice; }
  o.blank_top = bb;
  o.raw_top = 0; for (int k = 1; k < kNAct; ++k) if (o.P[k] > o.P[o.raw_top]) o.raw_top = k;
  o.ntok = brain.toks - tok0;
  o.secs = now_s() - t0;
}
static int seen_of_word(const std::string& w) { for (int k = 0; k < 8; ++k) if (w == kSeenWord[k]) return k; return kUnknown; }

struct Args {
  std::string mode, out, game, model = model_path().c_str();
  std::string enc, offline_dir, settings, input_template, maze = "b1";
  int max_dec = 120, ctx = 1024, cap_w = 320, cap_h = 180, workers = 8, max_minutes = 100;
  std::string game_res = "640x360";  // Scratch: game window resolution (capture size stays cap_w x cap_h)
  bool neml_inject = false;  // NEML: prepend the memory block to the trusted rules prime (off by default; not approved for runs)
  double look_pitch = -0.75, prior_w = 0.2;
  bool blank = true;
  bool way_shut = false;   // scratch loop-breaker (opt-in)
  int reach = 0;           // scratch interaction range in world blocks (opt-in, 0..4; half the engine's 8)
  int b1_seed = 1;         // Scratch: B1 maze seed (default 1)
  int b1_size = 16;        // Scratch: square B1 size (default 16)
  bool act_blank = false;  // Scratch: holdout, act on the blank-view choice
  int obs_ascii = 0;       // --obs ascii[=1024|2048]: add checked Nemo View rows to the live view (0 = off, pixels only)
};
static bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&](std::string& v) { if (i + 1 < argc) { v = argv[++i]; return true; } return false; };
    auto nexti = [&](int& v) { std::string s; if (!next(s)) return false; v = std::atoi(s.c_str()); return true; };
    auto nextd = [&](double& v) { std::string s; if (!next(s)) return false; v = std::atof(s.c_str()); return true; };
    if (k == "--probe" || k == "--nav" || k == "--speed") a.mode = k.substr(2);
    else if (k == "--policy-test") { a.mode = "policytest"; if (!next(a.offline_dir)) return false; }
    else if (k == "--offline") { a.mode = "offline"; if (!next(a.offline_dir)) return false; }
    else if (k == "--out") { if (!next(a.out)) return false; }
    else if (k == "--game") { if (!next(a.game)) return false; }
    else if (k == "--model") { if (!next(a.model)) return false; }
    else if (k == "--enc") { if (!next(a.enc)) return false; }
    else if (k == "--settings") { if (!next(a.settings)) return false; }
    else if (k == "--input-template") { if (!next(a.input_template)) return false; }
    else if (k == "--maze") { if (!next(a.maze)) return false; if (a.maze != "b1" && a.maze != "stagea") return false; }
    else if (k == "--max-dec") { if (!nexti(a.max_dec)) return false; }
    else if (k == "--max-minutes") { if (!nexti(a.max_minutes)) return false; }
    else if (k == "--ctx") { if (!nexti(a.ctx)) return false; }
    else if (k == "--game-res") { if (!next(a.game_res)) return false; }
    else if (const int n = neml::parse_flag(argc, argv, i, neml::process_config()); n != 0) { if (n < 0) return false; i += n - 1; }
    else if (k == "--neml-inject") a.neml_inject = true;
    else if (k == "--workers") { if (!nexti(a.workers)) return false; }
    else if (k == "--prior-w") { if (!nextd(a.prior_w)) return false; }
    else if (k == "--look-pitch") { if (!nextd(a.look_pitch)) return false; }
    else if (k == "--no-blank") a.blank = false;
    else if (k == "--way-shut") a.way_shut = true;
    else if (k == "--reach") { if (!nexti(a.reach) || a.reach < 0 || a.reach > 4) return false; }
    else if (k == "--b1-seed") { if (!nexti(a.b1_seed) || a.b1_seed < 1) return false; }
    else if (k == "--act-blank") a.act_blank = true;
    else if (k == "--obs") {
      std::string v; if (!next(v)) return false;
      if (v == "pixels") a.obs_ascii = 0;
      else if (v == "ascii") a.obs_ascii = -1;  // follows --ctx
      else if (v == "ascii=1024") a.obs_ascii = 1024;
      else if (v == "ascii=2048") a.obs_ascii = 2048;
      else return false;
    }
    else if (k == "--b1-size") { if (!nexti(a.b1_size) || a.b1_size < 8 || a.b1_size > 32) return false; }
    else return false;
  }
  if (a.obs_ascii == -1) a.obs_ascii = a.ctx;
  if (a.obs_ascii != 0 && a.obs_ascii != 1024 && a.obs_ascii != 2048) return false;  // ascii views need ctx 1024 or 2048
  return !a.out.empty() && !a.mode.empty();
}

// Labeled PROBE fixture (palette + band calibration only; never navigated, never a golden).
static const char kProbeFixture[] =
    "PCHIE-B1 11 9 7\n"
    "###########\n"
    "#S...P...##\n"
    "#.#########\n"
    "#..K......#\n"
    "#.#########\n"
    "#..D......#\n"
    "#.#########\n"
    "#E........#\n"
    "###########\n";

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) { std::fprintf(stderr, "usage: pchie_lk_visnav (--speed|--probe|--nav|--offline DIR) --out DIR [--game LithoKraft.exe] [--enc cfg] [--maze b1|stagea] [--obs pixels|ascii[=1024|2048]] ...\n"); return 2; }
  if (a.game.empty()) {
    if (const char* e = std::getenv("PCHIE_GAME")) a.game = e;
    else if (const char* e = std::getenv("LITHOKRAFT")) a.game = e;
  }
  const bool needs_game = (a.mode == "speed" || a.mode == "probe" || a.mode == "nav");
  if (needs_game && a.game.empty()) {
    std::fprintf(stderr, "Set --game to your LithoKraft executable, or set PCHIE_GAME or LITHOKRAFT.\n");
    return 2;
  }
  fs::create_directories(a.out);
  std::ofstream rep(fs::path(a.out) / "visnav_report.txt", std::ios::binary);
  auto say = [&](const char* fmt, auto... xs) {
    static char buf[32768];
    std::snprintf(buf, sizeof(buf), fmt, xs...);
    std::printf("[VISNAV] %s\n", buf); std::fflush(stdout);
    rep << buf << "\n"; rep.flush();
  };
  say("pchie_lk_visnav: SCRATCH RESEARCH (visnav, opt-in), never official, never a golden; mode=%s maze=%s pitch=%.2f workers=%d prior_w=%.2f blank=%d way_shut=%d reach=%d b1_seed=%d b1_size=%d act_blank=%d",
      a.mode.c_str(), a.maze.c_str(), a.look_pitch, a.workers, a.prior_w, int(a.blank), int(a.way_shut), a.reach, a.b1_seed, a.b1_size, int(a.act_blank));
  const double t_start = now_s();
  EncCfg enc; enc.defaults();
  if (!a.enc.empty()) { std::string e; if (!enc.load(a.enc, e)) { say("FAIL enc: %s", e.c_str()); return 1; } say("encoder cfg %s: %zu prototypes; lamp_beats_glass=%d; sha256 %s", a.enc.c_str(), enc.protos.size(), enc.lamp_beats_glass, sha_hex(slurp(a.enc)).substr(0, 16).c_str()); }
#ifdef PCHIE_SCRATCH_CUDA_VISNAV
  CudaNemotronModel model; model.set_threads(a.workers);
#else
  NemotronModel model;
#endif
  JoinJobDispatcher disp(a.workers);
  Brain brain; brain.reach = a.reach;
  auto load_model = [&]() -> bool {
    std::string err; const double t0 = now_s();
    if (!model.load(a.model, err)) { say("FAIL model load: %s", err.c_str()); return false; }
    model.init_cache(uint32_t(a.ctx), 6);
#ifdef PCHIE_SCRATCH_CUDA_VISNAV
    say("SCRATCH CUDA text model: %s ok=%d err=%s gpu_layers=%d/%d vram_alloc=%.1f MiB free %zu -> %zu MiB of %zu (threads=%d for host layers/head)", model.gpu_name_.c_str(), int(model.cuda_ok()), model.cuda_err().c_str(), model.gpu_layers(), model.n_layer(), model.vram_mb(), model.free_before_mb_, model.free_after_mb_, model.total_mb_, a.workers);
    if (!model.cuda_ok()) { say("FAIL cuda init: %s", model.cuda_err().c_str()); return false; }
#endif
    if (a.workers > 1) model.set_dispatcher(&disp);
    say("model Little Nemo loaded in %.1f s (%s, dispatcher=%d): %s weights sha256 %s", now_s() - t0,
#ifdef PCHIE_SCRATCH_CUDA_VISNAV
        "SCRATCH-CUDA",
#else
        "CPU",
#endif
        model.dispatcher_size(), a.model.c_str(), hex(model.weights_sha256()).substr(0, 16).c_str());
    if (neml::process_config().record || a.neml_inject) {
      size_t nt = 0;
      const std::string dg = neml::memory_block([&](const std::string& x) { return model.tokenize_no_bos(x).size(); }, 384, &nt);
      say("NEML: memory block %zu tokens (cap 384, real tokenizer); recording=%d inject=%d", nt, int(neml::process_config().record), int(a.neml_inject));
      if (a.neml_inject) brain.neml_prefix = dg;
    }
    const double t1 = now_s();
    if (!brain.prime(model, err)) { say("FAIL prime: %s", err.c_str()); return false; }
    say("rules primed: %zu trusted tokens in %.1f s", brain.base_len, now_s() - t1);
    return true;
  };

  if (a.mode == "speed") {
    if (!load_model()) return 1;
    float p[kNAct];
    const double t0 = now_s();
    ModelBranch ob = brain.observe(model, UntrustedText("forward: floor, floor, wall (new)\nright: wall\nback: floor, more (walked once)\nleft: wall\nkey carried: no\nlast moves: none"));
    brain.prior(model, ob, p);
    const float v = brain.option(model, ob, UntrustedText("walk forward, toward: floor, floor, wall (new)"));
    model.release(ob);
    say("speed: %zu tok in %.1f s: F=%.3f R=%.3f B=%.3f L=%.3f take=%.3f open=%.3f; option v=%.3f", brain.toks, now_s() - t0, p[0], p[1], p[2], p[3], p[4], p[5], v);
    return 0;
  }

  if (a.mode == "policytest") {
    // Scenario file: blocks separated by "----". Lines: "forward: <phrase>|<visits>" (x4, relative), "carry: 0|1", "recent: ...", "note: ..."
    if (!load_model()) return 1;
    std::istringstream is(slurp(a.offline_dir));
    std::string line; DecideIn in; in.reach = a.reach; std::string note; int filled = 0, n = 0;
    auto run = [&]() {
      if (filled < 4) return;
      DecideOut o; decide(model, brain, in, a.prior_w, a.blank, o);
      { std::string al; for (int k = 0; k < kNAct; ++k) if (o.allowed[k]) al += std::string(al.empty() ? "" : ",") + kActWord[k]; say("SCENARIO %d allowed={%s} take_w=%d open_w=%d", n, al.c_str(), o.take_w, o.open_w); }
      say("SCENARIO %d [%s]\n%s\n  policy F=%.2f R=%.2f B=%.2f L=%.2f take=%.2f open=%.2f raw_top=%s pol_top=%s blank_top=%s | leaves: %s | CHOICE %s (%.1f s)",
          n++, note.c_str(), o.obs.c_str(), o.P[0], o.P[1], o.P[2], o.P[3], o.P[4], o.P[5], kActWord[o.raw_top], kActWord[o.pol_top], kActWord[o.blank_top],
          o.leaf_log.c_str(), kActWord[o.choice], o.secs);
      in = DecideIn(); in.reach = a.reach; note.clear(); filled = 0;
    };
    while (std::getline(is, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.rfind("----", 0) == 0) { run(); continue; }
      const size_t c = line.find(':');
      if (c == std::string::npos) continue;
      const std::string key = line.substr(0, c); std::string val = line.substr(c + 1); while (!val.empty() && val[0] == ' ') val.erase(0, 1);
      int r = -1; for (int k = 0; k < 4; ++k) if (key == kRel[k]) r = k;
      if (r >= 0) {
        const size_t bar = val.find('|');
        in.phrase[r] = val.substr(0, bar); in.nvis[r] = bar == std::string::npos ? 0 : std::atoi(val.c_str() + bar + 1);
        in.shut[r] = bar != std::string::npos && val.find("|shut", bar) != std::string::npos;
        { DirView pv; std::istringstream ws(in.phrase[r]); std::string wd;  // phrase words -> cells (for occluder / cell1)
          while (pv.n < 4 && std::getline(ws, wd, ',')) { while (!wd.empty() && wd[0] == ' ') wd.erase(0, 1); if (wd == "more") break; pv.cells[pv.n++] = seen_of_word(wd); }
          in.cell1[r] = pv.n ? pv.cells[0] : int(kUnknown); first_occluder(pv, in.occ[r], in.occ_at[r]); }
        in.occ_world[r] = in.occ_at[r];  // phrase-only scenario: band k == k blocks unless overridden with |w=N
        if (bar != std::string::npos) { const size_t wq = val.find("|w=", bar); if (wq != std::string::npos) in.occ_world[r] = std::atoi(val.c_str() + wq + 3);
          in.key_clean[r] = val.find("|dirty", bar) == std::string::npos; in.reach_miss[r] = val.find("|miss", bar) != std::string::npos; }
        ++filled;
      } else if (key == "carry") in.carrying = std::atoi(val.c_str()) != 0;
      else if (key == "recent") in.recent = val;
      else if (key == "note") note = val;
    }
    run();
    return 0;
  }
  if (a.mode == "offline") {
    std::vector<fs::path> pngs;
    for (auto& e : fs::directory_iterator(a.offline_dir)) if (e.path().extension() == ".png") pngs.push_back(e.path());
    std::sort(pngs.begin(), pngs.end());
    for (const auto& p : pngs) {
      const std::string s = slurp(p); std::vector<uint8_t> png(s.begin(), s.end()), rgb; int w = 0, h = 0; std::string err;
      if (!png_decode(png, w, h, rgb, err)) { say("%s: %s", p.filename().string().c_str(), err.c_str()); continue; }
      const DirView v = encode_dir(enc, w, h, rgb);
      say("%s -> %s   {%s}", p.filename().string().c_str(), dir_phrase(v).c_str(), v.band_pct.c_str());
    }
    return 0;
  }

  // ---------- maze ----------
  const bool probe = (a.mode == "probe");
  const bool stagea = (a.maze == "stagea");
  std::string maze_text;
  MazeGrid grid;
  if (probe) maze_text = kProbeFixture;
  else { grid = stagea ? ScaledPitMaze::make() : MazeGrid(uint64_t(a.b1_seed), a.b1_size, a.b1_size); maze_text = LkMazeText::encode(grid); }
  const fs::path maze_path = fs::path(a.out) / (probe ? "visnav_probe_fixture_11x9.txt" : (stagea ? "pchie_stagea_seed64_64x64.txt" : ("pchie_b1_seed" + std::to_string(a.b1_seed) + "_" + std::to_string(a.b1_size) + "x" + std::to_string(a.b1_size) + ".txt")));
  { std::ofstream m(maze_path, std::ios::binary); m << maze_text; }
  // parse it back (for the build check, result check and probe truth only)
  std::vector<std::string> rows;
  { std::istringstream is(maze_text); std::string line; std::getline(is, line); while (std::getline(is, line)) if (!line.empty()) rows.push_back(line); }
  auto glyph_at = [&](int x, int y) -> char { if (y < 0 || y >= int(rows.size()) || x < 0 || x >= int(rows[size_t(y)].size())) return '#'; return rows[size_t(y)][size_t(x)]; };
  int exit_x = -1, exit_y = -1;
  for (int y = 0; y < int(rows.size()); ++y) for (int x = 0; x < int(rows[size_t(y)].size()); ++x) if (rows[size_t(y)][size_t(x)] == 'E') { exit_x = x; exit_y = y; }
  say("maze %s (%zu rows) sha256 %s; exit cell (%d,%d) [result check only]", maze_path.filename().string().c_str(), rows.size(), sha_hex(maze_text).substr(0, 16).c_str(), exit_x, exit_y);
  if (a.mode == "nav" && !load_model()) return 1;

  // ---------- launch LithoKraft on VULKAN with the opt-in display-vision tool ----------
  const fs::path data_root = fs::absolute(fs::path(a.out) / "lk_data");
  fs::create_directories(data_root / "config"); fs::create_directories(data_root / "logs");
  { std::ofstream ax(data_root / "config" / "autoexec.lth", std::ios::binary); ax << "// pchie_lk_visnav (SCRATCH visnav): launcher's own approval for this scratch run only.\nwait 256\nmcp grant world_read player_control pilot_senses\n"; }
  if (!a.input_template.empty()) {
    std::string in = slurp(a.input_template);
    const std::string from = "\"Break\": [\"mouse:Left\"", to = "\"Break\": [\"mouse:Left\", \"key:J\"";
    const size_t at = in.find(from);
    if (at == std::string::npos || in.find("key:J") != std::string::npos) { say("FAIL: input template has no default Break binding"); return 1; }
    in.replace(at, from.size(), to);
    std::ofstream f(data_root / "config" / "input.json", std::ios::binary); f << in;
    say("scratch input.json: Break += key:J (MCP attack needs a key binding)");
  }
  if (!a.settings.empty()) { std::error_code ec; fs::copy_file(a.settings, data_root / "config" / "settings.json", fs::copy_options::overwrite_existing, ec); }
  SetEnvironmentVariableW(L"LL_DATA_ROOT", data_root.wstring().c_str());
  SetEnvironmentVariableW(L"LL_MCP_DISPLAY_VISION", L"1");
  SetEnvironmentVariableW(L"LL_MCP_NO_FOCUS_PAUSE", L"1");
  if (a.obs_ascii) SetEnvironmentVariableW(L"LL_MCP_NEMO_VIEW", L"1");  // --obs ascii only
  const std::string cl = "\"" + a.game + "\" --mcp-stdio --pchie-maze \"" + fs::absolute(maze_path).string() +
                         "\" --spawn maze --renderer vulkan --display windowed --resolution " + a.game_res + " --seed 1 --no-save --mute --capture --no-vsync";
  say("launch: %s  [env LL_MCP_DISPLAY_VISION=1 LL_MCP_NO_FOCUS_PAUSE=1]", cl.c_str());
  const fs::path game_log = fs::path(a.out) / "game_stderr.log";
  McpStdioClient client; std::string err;
  if (!client.start(widen(cl), widen(fs::path(a.game).parent_path().string()), game_log.wstring(), err)) { say("FAIL launch: %s", err.c_str()); return 1; }
  LithoKraftEngineHost host(client);
  int rc = 1;
  const fs::path view_dir = fs::path(a.out) / "view"; fs::create_directories(view_dir);
  int n_caps = 0;
  struct Cap { std::string sha, grid; DirView v; double cam_yaw = 0, cam_pitch = 0; };
  auto capture = [&](const std::string& tag, Cap& c, std::string& e) -> bool {
    MiniJson p; char args[128];
    std::snprintf(args, sizeof(args), "{\"pilot\":\"player\",\"width\":%d,\"height\":%d}", a.cap_w, a.cap_h);
    if (!host.call_ok("vision.capture_display", args, &p, e)) return false;
    const std::string rn = p.str_or("renderer", "");
    if (rn != "vulkan") { e = "display renderer is '" + rn + "', not vulkan - refusing"; return false; }
    std::vector<uint8_t> png, rgb;
    if (!b64decode(p.str_or("data", ""), png)) { e = "no png data"; return false; }
    { std::ofstream f(view_dir / (tag + ".png"), std::ios::binary); f.write(reinterpret_cast<const char*>(png.data()), png.size()); }
    int w = 0, h = 0;
    if (!png_decode(png, w, h, rgb, e)) return false;
    c.sha = hex(Sha256::hash(png.data(), png.size()));
    c.grid = legacy_grid(w, h, rgb);
    c.v = encode_dir(enc, w, h, rgb);
    ++n_caps;
    MiniJson cam; if (const MiniJson* cc = p.get("camera")) cam = *cc;
    c.cam_yaw = cam.num_or("yaw", 0); c.cam_pitch = cam.num_or("pitch", 0);
    say("view %s: renderer=%s %dx%d tick=%.0f yaw=%.1f pitch=%.1f png=%s -> \"%s\" {%s}", tag.c_str(), rn.c_str(), w, h, p.num_or("tick", 0), cam.num_or("yaw", 0),
        cam.num_or("pitch", 0), c.sha.substr(0, 16).c_str(), dir_phrase(c.v).c_str(), c.v.band_pct.c_str());
    return true;
  };
  // --obs ascii: one checked Nemo View per slot of the context plan, logged on its own side chain
  const std::vector<research::AsciiRequest> ascii_reqs = a.obs_ascii ? research::ascii_requests(uint32_t(a.obs_ascii)) : std::vector<research::AsciiRequest>{};
  const std::string ascii_path = a.obs_ascii ? (fs::path(a.out) / "visnav_ascii_side_chain.jsonl").string() : std::string();
  research::SideLog ascii_log(ascii_path);
  int n_ascii = 0;
  if (a.obs_ascii) say("obs ascii: ctx %d, %zu views per decision [env LL_MCP_NEMO_VIEW=1]; side chain visnav_ascii_side_chain.jsonl", a.obs_ascii, ascii_reqs.size());
  auto capture_ascii = [&](const std::string& tag, const research::AsciiRequest& q, std::string& text, std::string& e) -> bool {
    MiniJson p; char args[160];
    std::snprintf(args, sizeof(args), "{\"pilot\":\"player\",\"cols\":%d,\"region\":\"%s\",\"view\":\"objects\",\"rle\":true}", q.cols, q.fovea ? "fovea" : "full");
    if (!host.call_ok("vision.capture_ascii", args, &p, e)) return false;
    research::AsciiCapture c; research::AsciiView v; std::string g;
    if (!research::capture_from_json(p, c, e) || !research::verify_capture(c, v, e)) return false;  // fail closed
    if (!research::fit_glyph(v, q.fit_cols, q.fit_rows, g, e)) return false;
    if (!research::log_view(ascii_log, tag, q.slot, v, e)) return false;
    { std::ofstream f(view_dir / (tag + "_" + q.slot + ".txt"), std::ios::binary); f << v.glyph << "\n"; }
    text += research::observation_text(q.slot, g, q.fit_cols, q.fit_rows);
    ++n_ascii;
    say("ascii %s %s: %dx%d %s sha=%s", tag.c_str(), q.slot.c_str(), v.cols, v.rows, v.view.c_str(), v.sha256.substr(0, 16).c_str());
    return true;
  };

  do {
    if (!host.initialize(err)) { say("FAIL initialize: %s", err.c_str()); break; }
    if (!host.wait_for_grant(120000, err)) { say("FAIL grant: %s", err.c_str()); break; }
    { std::string t; bool ie = true; for (int i = 0; i < 40; ++i) { if (!host.call_tool("vision.track_player", "{\"pilot\":\"player\"}", t, ie, err)) break; if (!ie) break; Sleep(250); } if (ie) { say("FAIL: pilot_senses not granted: %s", t.c_str()); break; } }
    say("granted: world_read player_control pilot_senses (scratch autoexec)");
    LkLegend L; int ox = 0, fy = 0, oz = 0; bool have = false;
    for (int i = 0; i < 40 && !have; ++i) { have = parse_legend(slurp(game_log), L, ox, fy, oz) || parse_legend(slurp(data_root / "logs" / "lithokraft.log"), L, ox, fy, oz); if (!have) Sleep(250); }
    if (!have) { say("FAIL: no legend line"); break; }
    const LkMazeVoxelMap map(ox, fy, oz);
    say("legend floor=%u wall=%u start=%u exit=%u key=%u door=%u origin (%d,%d,%d)", L.floor, L.wall, L.start, L.exit, L.key, L.door, ox, fy, oz);
    // PRE-RUN BUILD CHECK ONLY (never used by any decision): the game built the intended maze.
    if (!probe) {
      std::string first;
      const int bad = map.verify(grid, L, [&](int x, int y, int z, uint16_t& id) { return host.get_block(x, y, z, id, err); }, first);
      say("pre-run build check (setup only, not a decision input): %d mismatched cells %s", bad, first.c_str());
      if (bad != 0) { say("FAIL: maze build mismatch"); break; }
    }
    Sleep(800);
    LkPilotController pilot(host, map);
    if (!pilot.calibrate(err)) { say("FAIL calibrate: %s", err.c_str()); break; }
    say("calibrated look gains yaw %.5f pitch %.5f (focus_window not used)", pilot.yaw_gain(), pilot.pitch_gain());
    static const int DX[4] = {0, 1, 0, -1}, DY[4] = {-1, 0, 1, 0};
    static const char* DN[4] = {"N", "E", "S", "W"};
    auto yaw_of = [](int d) { return std::atan2(double(DY[d]), double(DX[d])); };
    LkPlayerState s0;
    if (!host.player_state(s0, err)) { say("FAIL state: %s", err.c_str()); break; }
    int cx = 0, cy = 0; map.world_to_cell(s0.x, s0.z, cx, cy);
    int heading = 0;
    { double best = 1e9; for (int d = 0; d < 4; ++d) { const double e = std::fabs(LkPilotController::wrap(s0.yaw - yaw_of(d))); if (e < best) { best = e; heading = d; } } }
    say("start cell (%d,%d) heading %s", cx, cy, DN[heading]);
    // face a cardinal direction at the look pitch, wait for the camera to settle, capture; re-face if the frame's camera yaw lags
    auto look_capture = [&](int d, const std::string& tag, Cap& c, std::string& e) -> bool {
      for (int attempt = 0; attempt < 3; ++attempt) {
        if (!pilot.face(yaw_of(d), a.look_pitch, e)) return false;
        Sleep(attempt == 0 ? 300 : 600);
        if (!capture(tag, c, e)) return false;
        double dyaw = std::fabs(LkPilotController::wrap(c.cam_yaw * LkPilotController::kPi / 180.0 - yaw_of(d))) * 180.0 / LkPilotController::kPi;
        double dp = std::fabs(c.cam_pitch * LkPilotController::kPi / 180.0 - a.look_pitch) * 180.0 / LkPilotController::kPi;
        if (dyaw < 4.0 && dp < 4.0) return true;
        say("  view %s: camera not settled (yaw off %.1f deg, pitch off %.1f deg) - re-facing", tag.c_str(), dyaw, dp);
      }
      return true;  // keep the last frame; logged above
    };

    if (probe) {
      // fixed labeled probe route: (cell, look dirs)
      struct Spot { int x, y; const char* dirs; };
      static const Spot spots[] = {{1, 1, "NESW"}, {2, 1, "EW"}, {3, 1, "EW"}, {4, 1, "EW"}, {3, 1, ""}, {2, 1, ""}, {1, 1, ""}, {1, 2, "E"},
                                   {1, 3, "E"}, {2, 3, "E"}, {1, 3, ""}, {1, 4, "E"}, {1, 5, "ES"}, {2, 5, "E"}, {1, 5, ""}, {1, 6, "SN"}};
      bool ok = true;
      for (const auto& sp : spots) {
        if (sp.x != cx || sp.y != cy) { if (!pilot.walk_to(sp.x, sp.y, err)) { say("walk: %s", err.c_str()); ok = false; break; } cx = sp.x; cy = sp.y; }
        for (const char* q = sp.dirs; *q; ++q) {
          const int d = (*q == 'N') ? 0 : (*q == 'E') ? 1 : (*q == 'S') ? 2 : 3;
          std::string truth;
          for (int k = 1; k <= 6; ++k) truth += glyph_at(cx + DX[d] * k, cy + DY[d] * k);
          char tag[64]; std::snprintf(tag, sizeof(tag), "probe_c%d_%d_%s_T%s", cx, cy, DN[d], truth.c_str());
          Cap c; if (!look_capture(d, tag, c, err)) { say("FAIL look: %s", err.c_str()); ok = false; break; }
          say("  truth ahead 1..6 (fixture glyphs, PROBE ONLY): %s", truth.c_str());
        }
        if (!ok) break;
      }
      rc = ok ? 0 : 1;
      break;
    }

    // ---------------- nav: vision-only, model-chosen, no step list ----------------
    std::vector<uint8_t> chain(32, 0); size_t chain_len = 0;
    std::ofstream olog(fs::path(a.out) / "visnav_obs_side_log.txt", std::ios::binary);
    std::map<std::pair<int, int>, int> visits;  // dead-reckoned from the agent's OWN committed moves
    ShutMemory shut;                            // --way-shut: own record of timed-out walks (never the map)
    ShutMemory reach_miss;                      // --reach: own record of refused / failed ranged interactions (cell + abs dir)
    int reach_acts = 0, reach_refused = 0, reach_max = 0;
    int ox_ = cx, oy_ = cy;                     // odometry cell (agent frame == maze frame at start; integrated from own moves)
    visits[{ox_, oy_}] = 1;
    std::vector<std::string> recent;
    std::vector<std::pair<int, int>> path{{cx, cy}};
    bool carrying = false, fell = false, at_exit = false, invalid = false;
    int decisions = 0, view_caused = 0, view_caused_n = 0, raw_blocked = 0, bumps = 0, breaks = 0, odo_mismatch = 0, leaf_flips = 0;
    const double t_nav = now_s();
    neml::Recorder nrec;  // NEML per-event triggers (a receipt is written at the end unless --neml off)
    bool neml_infra_fail = false;  // game/MCP failure: no receipt (not an honest miss)
    while (decisions < a.max_dec) {
      if ((now_s() - t_nav) > 60.0 * a.max_minutes) { say("wall-clock cap reached (%d min)", a.max_minutes); break; }
      // 1) look around: forward, right, back, left (relative to heading), live Vulkan frames
      Cap caps[4];
      std::vector<std::string> ascii_part(ascii_reqs.size());
      for (int r = 0; r < 4; ++r) {
        const int d = (heading + r) % 4;
        char tag[64]; std::snprintf(tag, sizeof(tag), "d%03d_look_%s", decisions, kRel[r]);
        if (!look_capture(d, tag, caps[r], err)) { say("FAIL look: %s", err.c_str()); neml_infra_fail = true; goto nav_end; }
        for (size_t q = 0; q < ascii_reqs.size(); ++q)
          if (ascii_reqs[q].rel == r && !capture_ascii(tag, ascii_reqs[q], ascii_part[q], err)) { say("FAIL ascii view: %s", err.c_str()); neml_infra_fail = true; goto nav_end; }
      }
      {
        // 2)-5) observation (pixels + own memory only), vision-only filter, policy fork, blank fork, leaf forks
        auto nv = [&](int r) { const int d = (heading + r) % 4; auto it = visits.find({ox_ + DX[d], oy_ + DY[d]}); return it == visits.end() ? 0 : it->second; };
        std::string rec_s; for (size_t k = recent.size() > 4 ? recent.size() - 4 : 0; k < recent.size(); ++k) rec_s += (rec_s.empty() ? "" : ", ") + recent[k];
        if (rec_s.empty()) rec_s = "none";
        DecideIn in;
        for (int r = 0; r < 4; ++r) {
          in.phrase[r] = dir_phrase(caps[r].v); in.cell1[r] = caps[r].v.cells[0]; in.nvis[r] = nv(r);
          first_occluder(caps[r].v, in.occ[r], in.occ_at[r]);
          in.shut[r] = a.way_shut && shut.is_shut(ox_, oy_, (heading + r) % 4);
          if (a.reach > 0) { reach_fields(caps[r].v, in, r); in.reach_miss[r] = reach_miss.is_shut(ox_, oy_, (heading + r) % 4); }
        }
        in.carrying = carrying; in.recent = rec_s; in.reach = a.reach;
        if (!ascii_reqs.empty()) { in.ascii = std::string(research::ascii_legend()) + "\n"; for (const auto& t : ascii_part) in.ascii += t; }
        DecideOut o; decide(model, brain, in, a.prior_w, a.blank, o);
        if (!o.allowed[o.raw_top]) ++raw_blocked;
        if (o.choice != o.pol_top) ++leaf_flips;
        const int choice = (a.act_blank && o.blank_top >= 0) ? o.blank_top : o.choice;  // SCRATCH holdout: act on blank-view choice
        const int take_r = o.take_r, open_r = o.open_r, n_allowed = o.n_allowed, take_w = o.take_w, open_w = o.open_w;
        if (a.act_blank) say("act-blank holdout: acting on blank_top=%s (live-view choice was %s)", kActWord[choice], kActWord[o.choice]);
        const bool caused = n_allowed > 1 && choice != o.blank_top;
        if (n_allowed > 1) { ++view_caused_n; if (caused) ++view_caused; }
        if (caused) nrec.event(neml::kVisualRecovery, decisions, 255, ox_, oy_, "view_caused");
        else if (o.choice != o.pol_top && !o.allowed[o.raw_top]) nrec.event(neml::kVisualRecovery, decisions, 160, ox_, oy_, "leaf_flip_raw_blocked");
        else if (o.choice != o.pol_top && n_allowed > 1) nrec.event(neml::kForkPivot, decisions, 128, ox_, oy_, "leaf_flip_redecide");
        if (choice == 2) nrec.event(neml::kDeadEnd, decisions, 192, ox_, oy_, "reversal_back");
        say("d%03d odo=(%d,%d) heading=%s carrying=%d | obs:\n%s\n  policy F=%.2f R=%.2f B=%.2f L=%.2f take=%.2f open=%.2f raw_top=%s pol_top=%s | blank_top=%s | leaves: %s | CHOICE %s | %zu tok %.1f s",
            decisions, ox_, oy_, DN[heading], int(carrying), o.obs.c_str(), o.P[0], o.P[1], o.P[2], o.P[3], o.P[4], o.P[5], kActWord[o.raw_top], kActWord[o.pol_top],
            kActWord[o.blank_top], o.leaf_log.c_str(), kActWord[choice], o.ntok, o.secs);
        const std::string& obs = o.obs;
        // side log
        LkPlayerState es; int ecx = -99, ecy = -99;
        if (host.player_state(es, err)) map.world_to_cell(es.x, es.z, ecx, ecy);  // LOG ONLY
        if (ecx != ox_ || ecy != oy_) ++odo_mismatch;
        char rbuf[2048];
        std::snprintf(rbuf, sizeof(rbuf), "visnav.obs.v1|dec=%d|odo=(%d,%d)|engine_cell_logonly=(%d,%d)|heading=%s|obs_sha=%s|views=%s,%s,%s,%s|allowed=%d|choice=%s|pol_top=%s|blank_top=%s|raw_top=%s|view_caused=%d",
                      decisions, ox_, oy_, ecx, ecy, DN[heading], sha_hex(obs).substr(0, 16).c_str(), caps[0].sha.substr(0, 12).c_str(), caps[1].sha.substr(0, 12).c_str(),
                      caps[2].sha.substr(0, 12).c_str(), caps[3].sha.substr(0, 12).c_str(), n_allowed, kActWord[choice], kActWord[o.pol_top], kActWord[o.blank_top], kActWord[o.raw_top], int(caused));
        std::string rec(rbuf);
        if (a.reach > 0) { char rb[96]; std::snprintf(rb, sizeof(rb), "|reach=%d|take_w=%d|open_w=%d", a.reach, take_w, open_w); rec += rb; }
        { std::vector<uint8_t> buf(chain); const auto rh = Sha256::hash(reinterpret_cast<const uint8_t*>(rec.data()), rec.size()); buf.insert(buf.end(), rh.begin(), rh.end()); chain = Sha256::hash(buf.data(), buf.size()); ++chain_len; }
        olog << rec << "\n"; olog.flush();
        ++decisions;
        // 6) act
        if (choice == kTake || choice == kOpen) {
          const int r = (choice == kTake) ? take_r : open_r;
          const int d = (heading + r) % 4;
          const int w = std::max(1, (choice == kTake) ? take_w : open_w);  // world blocks to the target (1 unless --reach ranged it)
          const int tx = ox_ + DX[d] * w, ty = oy_ + DY[d] * w;
          // anti-cheat audit (world read; can only ABORT the run, never steer): refuse to break maze walls
          uint16_t id1 = 0, id2 = 0;
          host.get_block(map.block_x(tx), fy + 1, map.block_z(ty), id1, err);
          host.get_block(map.block_x(tx), fy + 2, map.block_z(ty), id2, err);
          if (id1 == L.wall || id2 == L.wall) { say("AUDIT: the chosen %s targets a WALL block -> run marked INVALID and stopped (no attack sent)", kActWord[choice]); invalid = true; break; }
          bool ok = true;
          if (w >= 2) {  // --reach ranged interaction: the same audit read may only REFUSE (never steer, never pick a target)
            const bool right_obj = (choice == kTake) ? (id1 == L.key) : (id1 == L.door || id2 == L.door);
            if (!right_obj) {
              ++reach_refused; reach_miss.mark(ox_, oy_, d);
              say("reach-audit: ranged %s %s @%d blocks -> cell (%d,%d) is not a %s; REFUSED (no attack sent), ranged leaf dropped from (%d,%d)->%s",
                  kActWord[choice], DN[d], w, tx, ty, choice == kTake ? "key" : "door", ox_, oy_, DN[d]);
              recent.push_back(std::string(kActWord[choice]) + " " + kRel[r]);
              heading = d;
              continue;
            }
            ++reach_acts; reach_max = std::max(reach_max, w);
          }
          if (choice == kTake) ok = pilot.break_block(tx, ty, 1, err);
          else ok = pilot.break_block(tx, ty, 2, err) && pilot.break_block(tx, ty, 1, err);
          if (!ok && w >= 2) reach_miss.mark(ox_, oy_, d);
          if (ok) { ++breaks; if (choice == kTake) carrying = true; }  // carrying = the agent's own record of a successful take (no world read)
          if (ok && a.way_shut && shut.is_shut(ox_, oy_, d)) { shut.clear(ox_, oy_, d); say("way-shut: (%d,%d)->%s cleared after a successful %s", ox_, oy_, DN[d], kActWord[choice]); }
          if (w >= 2) say("%s %s @%d blocks (reach %d, target cell (%d,%d)): %s", kActWord[choice], DN[d], w, a.reach, tx, ty, ok ? "done" : err.c_str());
          else say("%s %s: %s", kActWord[choice], DN[d], ok ? "done" : err.c_str());
          recent.push_back(std::string(kActWord[choice]) + " " + kRel[r]);
          heading = d;
          continue;
        }
        const int d = (heading + choice) % 4;
        const int tx = ox_ + DX[d], ty = oy_ + DY[d];
        if (!pilot.walk_to(tx, ty, err)) {
          ++bumps;
          nrec.event(neml::kDeadEnd, decisions - 1, 255, ox_, oy_, "confirmed_block");
          LkPlayerState s;
          if (host.player_state(s, err) && s.y < double(fy) + 0.5) { fell = true; map.world_to_cell(s.x, s.z, cx, cy); say("FELL at (%d,%d) y=%.2f", cx, cy, s.y); break; }
          say("move %s blocked: %s; returning to own cell", kActWord[choice], err.c_str());
          std::string e2; const bool recentred = pilot.walk_to(ox_, oy_, e2);
          if (!recentred) say("recentre failed: %s", e2.c_str());
          if (a.way_shut && recentred) { shut.mark(ox_, oy_, d); say("way-shut: (%d,%d)->%s marked shut (timed-out walk, recentred); dropped from walk leaves", ox_, oy_, DN[d]); }
          recent.push_back(std::string(kActWord[choice]) + " (blocked)");
          continue;
        }
        ox_ = tx; oy_ = ty; heading = d;
        ++visits[{ox_, oy_}];
        recent.push_back(kActWord[choice]);
        path.push_back({ox_, oy_});
        Sleep(150);
        LkPlayerState s;
        if (host.player_state(s, err)) {
          map.world_to_cell(s.x, s.z, cx, cy);
          if (s.y < double(fy) + 0.5) { fell = true; say("FELL into a pit at (%d,%d) y=%.2f", cx, cy, s.y); break; }
          if (cx == exit_x && cy == exit_y) { at_exit = true; nrec.event(neml::kExitFound, decisions - 1, 255, ox_, oy_, "exit"); say("EXIT REACHED IN ENGINE at (%d,%d) (result check)", cx, cy); break; }
        }
      }
    }
  nav_end:
    {
      std::string ps; for (auto& c : path) { char b[24]; std::snprintf(b, sizeof(b), "(%d,%d)", c.first, c.second); ps += b; }
      std::set<std::pair<int, int>> uniq(path.begin(), path.end());
      say("walk path (%zu cells): %s", path.size(), ps.c_str());
      say("summary: maze=%s decisions=%d moves=%zu distinct_cells=%zu end=(%d,%d) fell=%d exit=%d invalid=%d key=%d breaks=%d bumps=%d view_caused=%d/%d (decisions with >1 allowed action) leaf_flips=%d raw_top_blocked=%d odo_mismatch=%d vulkan_captures=%d model_tokens=%zu nav_minutes=%.1f",
          a.maze.c_str(), decisions, path.size() - 1, uniq.size(), ox_, oy_, int(fell), int(at_exit), int(invalid), int(carrying), breaks, bumps, view_caused, view_caused_n, leaf_flips,
          raw_blocked, odo_mismatch, n_caps, brain.toks, (now_s() - t_nav) / 60.0);
      const bool neml_on = neml::process_config().record;
      if (neml_on && neml_infra_fail) say("NEML: infrastructure failure (game/MCP), receipt NOT written (not an honest miss)");
      if (neml_on && !neml_infra_fail) {
        neml::ReceiptInfo info;
        info.maze_seed = uint32_t(a.b1_seed);
        info.maze_w = info.maze_h = a.b1_size;
        info.path_kind = neml::kPathVision;
        info.flags = neml::kFlagModel;
        std::string nerr;
        const uint32_t id = neml::write_receipt(neml::process_config(), nrec, decisions, at_exit, "vision " + a.maze, info, &nerr);
        if (id) say("NEML: receipt run_id=%u exit=%d steps=%d forks=%u dead_ends=%u events_fired=%zu", id, int(at_exit), decisions, nrec.forks, nrec.dead_ends, nrec.all.size());
        else say("NEML: receipt NOT written: %s", nerr.c_str());
      }
      if (a.obs_ascii) {
        std::string ah, ae; size_t an = 0;
        const bool aok = research::SideLog::verify(ascii_path, ah, an, ae);
        say("ascii side chain (SCRATCH, never official): views=%d verify=%s head=%s lines=%zu", n_ascii, aok ? "ok" : ae.c_str(), ah.c_str(), an);
      }
      if (a.reach > 0) say("reach: N=%d ranged_acts=%d refused=%d max_blocks_used=%d", a.reach, reach_acts, reach_refused, reach_max);
      say("visnav obs side chain (SCRATCH, never official, never a golden): sha256 head=%s len=%zu; trace W-cells sha256=%s", hex(chain).c_str(), chain_len, sha_hex(pilot.trace()).c_str());
      (void)host.move(0, 0, false, err);
      rc = at_exit ? 0 : 3;
    }
  } while (false);
  client.stop(8000);
  say("game closed; total %.1f s; rc %d", now_s() - t_start, rc);
  return rc;
}
