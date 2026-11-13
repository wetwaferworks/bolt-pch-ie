// Box-side harness (reach4 extension 2026-11-11): compiles the VISNAV_PURE regions of pchie_lk_visnav.cpp verbatim (no model, no game).
// pure.inc is not in the tree. From vision/: python scripts/extract.py pchie_lk_visnav.cpp pure.inc
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>
#include "pure.inc"
static int fails = 0, passes = 0;
#define CHECK(c, msg) do { if (c) { ++passes; std::printf("  PASS %s\n", msg); } else { ++fails; std::printf("  FAIL %s\n", msg); } } while (0)
static std::string allowed_str(const DecideOut& o) { std::string s; for (int k = 0; k < kNAct; ++k) if (o.allowed[k]) s += std::string(s.empty() ? "" : ",") + kActWord[k]; return s; }
static DirView view_of(const std::string& phrase) {  // phrase words -> DirView (as the encoder would emit)
  DirView v; std::istringstream ws(phrase); std::string w;
  while (v.n < 4 && std::getline(ws, w, ',')) { while (!w.empty() && w[0] == ' ') w.erase(0, 1); if (w == "more") { v.open_beyond = true; break; }
    int k = kUnknown; for (int j = 0; j < 8; ++j) if (w == kSeenWord[j]) k = j; v.cells[v.n++] = k; }
  return v;
}
static DecideIn mk(const char* f, const char* r, const char* b, const char* l, bool carry, bool sf = false, bool sr = false, bool sb = false, bool sl = false, int nf = 0, int nr = 0, int nb = 1, int nl = 1) {
  DecideIn in; const char* ph[4] = {f, r, b, l}; bool sh[4] = {sf, sr, sb, sl}; int nv[4] = {nf, nr, nb, nl};
  for (int k = 0; k < 4; ++k) { DirView v = view_of(ph[k]); in.phrase[k] = dir_phrase(v); in.cell1[k] = v.cells[0]; first_occluder(v, in.occ[k], in.occ_at[k]); in.shut[k] = sh[k]; in.nvis[k] = nv[k]; }
  in.carrying = carry; in.recent = "none"; return in;
}
static void show(const char* name, const DecideIn& in, DecideOut& o) { build_obs(in, o); gate(in, o); std::printf("[%s] allowed={%s} open_r=%d take_r=%d\n%s\n", name, allowed_str(o).c_str(), o.open_r, o.take_r, o.obs.c_str()); }
int main(int argc, char** argv) {
  if (argc >= 3 && std::string(argv[1]) == "--encode") {  // --encode cfg png... : C++ encode_dir over saved frames
    EncCfg c; c.defaults(); std::string err; if (!c.load(argv[2], err)) { std::printf("cfg: %s\n", err.c_str()); return 1; }
    for (int i = 3; i < argc; ++i) {
      std::ifstream f(argv[i], std::ios::binary); std::ostringstream ss; ss << f.rdbuf(); const std::string s = ss.str();
      std::vector<uint8_t> png(s.begin(), s.end()), rgb; int w = 0, h = 0;
      if (!png_decode(png, w, h, rgb, err)) { std::printf("%s\tERR %s\n", argv[i], err.c_str()); continue; }
      const DirView v = encode_dir(c, w, h, rgb); std::printf("%s\t%s\t%s\n", argv[i], dir_phrase(v).c_str(), v.band_pct.c_str());
    }
    return 0;
  }
  if (argc >= 4 && std::string(argv[1]) == "--replay") {  // --replay cfg list : gate with --reach N over saved 4-view decisions
    EncCfg c; c.defaults(); std::string err; if (!c.load(argv[2], err)) { std::printf("cfg: %s\n", err.c_str()); return 1; }
    const int reach = argc >= 5 ? std::atoi(argv[4]) : 4;
    std::ifstream L(argv[3]); std::string line;
    while (std::getline(L, line)) {
      std::istringstream is(line); std::string tag; int carry = 0, shutf = 0; std::string png[4];
      is >> tag >> carry >> shutf >> png[0] >> png[1] >> png[2] >> png[3];
      DecideIn in; in.reach = reach; in.carrying = carry; in.recent = "none";
      for (int r = 0; r < 4; ++r) {
        std::ifstream f(png[r], std::ios::binary); std::ostringstream ss; ss << f.rdbuf(); const std::string sb = ss.str();
        std::vector<uint8_t> pb(sb.begin(), sb.end()), rgb; int w = 0, h = 0;
        if (!png_decode(pb, w, h, rgb, err)) { std::printf("%s ERR %s\n", png[r].c_str(), err.c_str()); return 1; }
        const DirView v = encode_dir(c, w, h, rgb);
        in.phrase[r] = dir_phrase(v); in.cell1[r] = v.cells[0]; first_occluder(v, in.occ[r], in.occ_at[r]); reach_fields(v, in, r);
      }
      in.shut[0] = shutf != 0;
      DecideOut o; build_obs(in, o); gate(in, o);
      DecideIn in0 = in; in0.reach = 0; DecideOut o0; build_obs(in0, o0); gate(in0, o0);
      std::printf("%s\ttake_r=%d take_w=%d open_r=%d open_w=%d\toff_take_r=%d off_open_r=%d\tallowed={%s}\toff={%s}\t%s|%s|%s|%s\n", tag.c_str(), o.take_r, o.take_w, o.open_r, o.open_w,
                  o0.take_r, o0.open_r, allowed_str(o).c_str(), allowed_str(o0).c_str(), in.phrase[0].c_str(), in.phrase[1].c_str(), in.phrase[2].c_str(), in.phrase[3].c_str());
    }
    return 0;
  }
  std::printf("== gate / loop-breaker unit tests (pure regions of the scratch source) ==\n");
  DecideOut o;
  // T1: d031 first look at (11,13) W with the FIXED encoder phrase, no failed walk yet: forward is a walk leaf, open is NOT (conservative)
  { DecideIn in = mk("floor, door", "wall", "floor, floor, more", "floor, wall", true); show("T1 d031 first look, not shut", in, o);
    CHECK(o.allowed[kFwd] && o.allowed[kBack] && o.allowed[kLeft] && !o.allowed[kRight], "T1 forward/back/left are walk leaves, right (wall) is not");
    CHECK(!o.allowed[kOpen], "T1 open is NOT a leaf before any failed walk (door in cell 2 is not promoted)");
    CHECK(o.obs.find("forward: floor, door (new)") != std::string::npos, "T1 forward still tagged (new) before a failure"); }
  // T2: same pose after ONE timed-out walk forward (shut): forward dropped, tagged (shut), open becomes a leaf
  { DecideIn in = mk("floor, door", "wall", "floor, floor, more", "floor, wall", true, true); show("T2 after one timeout, shut", in, o);
    CHECK(!o.allowed[kFwd], "T2 shut forward is dropped from allowed");
    CHECK(o.obs.find("forward: floor, door (shut)") != std::string::npos && o.obs.find("forward: floor, door (new)") == std::string::npos, "T2 shut forward tagged (shut), not (new)");
    CHECK(o.blank.find("forward: unknown (shut)") != std::string::npos, "T2 blank control also says (shut)");
    CHECK(o.allowed[kOpen] && o.open_r == kFwd, "T2 open is a leaf on forward (carrying, first non-floor occluder = door in cell 2)");
    CHECK(o.allowed[kBack] && o.allowed[kLeft], "T2 back and left remain leaves");
    CHECK(o.n_allowed == 3, "T2 leaves are exactly back, left, open"); }
  // T3: shut but NOT carrying -> no open; only back/left
  { DecideIn in = mk("floor, door", "wall", "floor, floor, more", "floor, wall", false, true); show("T3 shut, not carrying", in, o);
    CHECK(!o.allowed[kOpen] && !o.allowed[kFwd] && o.allowed[kBack] && o.allowed[kLeft], "T3 no open without the key; forward dropped"); }
  // T4: OLD encoder phrase "floor, key" with shut + carrying: forward dropped (loop broken) but no open (needs the encoder fix)
  { DecideIn in = mk("floor, key", "wall", "floor, floor, more", "floor, wall", true, true); show("T4 old phrase floor,key shut", in, o);
    CHECK(!o.allowed[kFwd] && !o.allowed[kOpen] && o.allowed[kBack] && o.allowed[kLeft], "T4 loop-breaker alone drops forward; open needs the door label"); }
  // T5: d029-like: door in cell 3, shut, carrying -> open NOT a leaf (only cell 1/2 promoted)
  { DecideIn in = mk("floor, floor, door", "wall", "floor, more", "wall", true, true); show("T5 door in cell 3 shut", in, o);
    CHECK(!o.allowed[kOpen], "T5 door in cell 3 never makes open a leaf, even shut"); }
  // T6: d029 not shut: unchanged pre-fix behaviour (forward walk, no open)
  { DecideIn in = mk("floor, floor, door", "wall", "floor, more", "wall", true); show("T6 d029 not shut", in, o);
    CHECK(o.allowed[kFwd] && !o.allowed[kOpen], "T6 d029 forward walk, no open (as in the run)"); }
  // T7: probe-style adjacent door (cell 1) carrying, not shut -> open (pre-existing rule kept)
  { DecideIn in = mk("door", "wall", "floor, more", "wall", true); show("T7 adjacent door cell1", in, o);
    CHECK(o.allowed[kOpen] && o.open_r == kFwd && !o.allowed[kFwd], "T7 door in cell 1 + key -> open leaf (unchanged)"); }
  // T8: d025 real key, not carrying -> take (unchanged), and shut does not remove take
  { DecideIn in = mk("key", "wall", "floor, floor, more", "floor, wall", false); show("T8 d025 key", in, o);
    CHECK(o.allowed[kTake] && o.take_r == kFwd && !o.allowed[kOpen], "T8 real key -> take leaf (unchanged)");
    in.shut[kFwd] = true; build_obs(in, o); gate(in, o); CHECK(o.allowed[kTake], "T8b take survives a shut flag"); }
  // T9: every walk direction shut and no open/take -> fallback never re-opens a shut way unless ALL are shut
  { DecideIn in = mk("floor, more", "floor, more", "floor, more", "floor, more", false, true, true, true, false); show("T9 three shut", in, o);
    CHECK(o.allowed[kLeft] && !o.allowed[kFwd] && !o.allowed[kRight] && !o.allowed[kBack], "T9 only the unshut way is a leaf");
    in.shut[kLeft] = true; build_obs(in, o); gate(in, o); CHECK(o.n_allowed == 4, "T9b all four shut -> last-resort fallback (all walks)"); }
  // T10: no shut flags anywhere -> gate identical to pre-fix rule over a grid of cell1 x carrying
  { int bad = 0; for (int c = 0; c < 8; ++c) for (int carry = 0; carry < 2; ++carry) {
      DecideIn in; for (int r = 0; r < 4; ++r) { in.cell1[r] = (c + r) % 8; in.occ[r] = in.cell1[r] == kFloor ? kDoor : in.cell1[r]; in.occ_at[r] = 2; } in.carrying = carry;
      DecideOut n; gate(in, n);
      bool al[kNAct] = {}; int tr = -1, orr = -1;  // pre-fix rule, transcribed
      for (int r = 0; r < 4; ++r) { const int c1 = in.cell1[r]; if (c1 == kFloor || c1 == kExit) al[r] = true; if (c1 == kKey && carry) al[r] = true; if (c1 == kKey && !carry && tr < 0) tr = r; if (c1 == kDoor && carry && orr < 0) orr = r; }
      if (tr >= 0) al[kTake] = true; if (orr >= 0) al[kOpen] = true; int na = 0; for (int k = 0; k < kNAct; ++k) na += al[k]; if (!na) for (int k = 0; k < 4; ++k) al[k] = true;
      for (int k = 0; k < kNAct; ++k) if (al[k] != n.allowed[k]) ++bad; if (tr != n.take_r || orr != n.open_r) ++bad; }
    CHECK(bad == 0, "T10 with no shut direction the gate equals the pre-fix rule (16 combos, incl. floor-then-door views)"); }
  // T11: ShutMemory bookkeeping replay of the stall: (11,13) heading W (d=3). first timeout marks, open clears.
  { ShutMemory m; CHECK(!m.is_shut(11, 13, 3), "T11 edge (11,13)->W not shut before d031");
    m.mark(11, 13, 3); CHECK(m.is_shut(11, 13, 3), "T11 one timed-out walk marks (11,13)->W shut");
    CHECK(!m.is_shut(11, 13, 1) && !m.is_shut(12, 13, 3), "T11 other edges unaffected");
    m.clear(11, 13, 3); CHECK(!m.is_shut(11, 13, 3), "T11 successful open clears the edge so the door cell can be walked"); }
  // T12: replay d031..: a "policy" that always picks forward when allowed else the first allowed (worst case = the frozen forward preference)
  { ShutMemory m; int x = 11, y = 13, heading = 3; bool carry = true; int bumps = 0; std::string trace;
    for (int dec = 31; dec < 36; ++dec) {
      DecideIn in = mk("floor, door", "wall", "floor, floor, more", "floor, wall", carry, m.is_shut(x, y, heading));
      DecideOut q; build_obs(in, q); gate(in, q);
      int ch = q.allowed[kFwd] ? kFwd : (q.allowed[kOpen] ? kOpen : -1);
      if (ch < 0) for (int k = 0; k < kNAct; ++k) if (q.allowed[k]) { ch = k; break; }
      trace += "d" + std::to_string(dec) + ":" + kActWord[ch] + " ";
      if (ch == kFwd) { ++bumps; m.mark(x, y, heading); continue; }  // the door blocks the walk; recentre ok
      if (ch == kOpen) { m.clear(x, y, heading); break; }
      break;
    }
    std::printf("[T12] replay: %s bumps=%d\n", trace.c_str(), bumps);
    CHECK(bumps == 1, "T12 exactly one blocked walk before the forward leaf is gone (was 53)");
    CHECK(trace.find("d32:open") != std::string::npos, "T12 with the fixed phrase and key, open is reachable at d032 (if the model picks it)"); }

  // ================= --reach N =================
  std::printf("== --reach tests ==\n");
  auto mkr = [&](const char* f, bool carry, int reach, bool shut = false) { DecideIn in = mk(f, "wall", "floor, floor, more", "wall", carry, shut);
    for (int r = 0; r < 4; ++r) in.occ_world[r] = in.occ_at[r]; in.reach = reach; return in; };
  // R1: key 1..3 blocks ahead (encoder key needs bands k and k+1, so 3 is the farthest key band), not carrying -> take leaf at that range
  for (int k = 1; k <= 3; ++k) { std::string ph; for (int j = 1; j < k; ++j) ph += "floor, "; ph += "key";
    DecideIn in = mkr(ph.c_str(), false, 4); DecideOut q; build_obs(in, q); gate(in, q);
    char m[128]; std::snprintf(m, sizeof(m), "R1 key %d block(s) ahead + reach 4 -> take leaf, take_w=%d", k, k);
    CHECK(q.allowed[kTake] && q.take_r == kFwd && q.take_w == k, m); }
  // R1b: a key FOUR blocks ahead (world distance 4, e.g. a future encoder band) -> take leaf without walking onto it
  { DecideIn in = mkr("floor, floor, floor, key", false, 4); DecideOut q; build_obs(in, q); gate(in, q);
    CHECK(q.allowed[kTake] && q.take_w == 4, "R1b key 4 blocks ahead -> take leaf at 4 (no walk onto it)");
    CHECK(q.allowed[kFwd], "R1b forward walk stays a leaf too (the model chooses)"); }
  // R2: door 1..4 blocks ahead, carrying -> open leaf at that range, NO bump needed
  for (int k = 1; k <= 4; ++k) { std::string ph; for (int j = 1; j < k; ++j) ph += "floor, "; ph += "door";
    DecideIn in = mkr(ph.c_str(), true, 4); DecideOut q; build_obs(in, q); gate(in, q);
    char m[128]; std::snprintf(m, sizeof(m), "R2 door %d block(s) ahead + key + reach 4 -> open leaf, open_w=%d, not shut", k, k);
    CHECK(q.allowed[kOpen] && q.open_r == kFwd && q.open_w == k && !in.shut[kFwd], m); }
  // R3: 5+ blocks -> never a leaf (world distance override 5..8)
  for (int w = 5; w <= 8; ++w) { DecideIn in = mkr("floor, floor, floor, door", true, 4); in.occ_world[kFwd] = w; DecideOut q; build_obs(in, q); gate(in, q);
    DecideIn ik = mkr("floor, floor, key", false, 4); ik.occ_world[kFwd] = w; DecideOut qk; build_obs(ik, qk); gate(ik, qk);
    char m[128]; std::snprintf(m, sizeof(m), "R3 door/key %d blocks ahead -> neither open nor take is a leaf", w);
    CHECK(!q.allowed[kOpen] && !qk.allowed[kTake], m); }
  // R3b: reach 2 limits to 2
  { DecideIn in = mkr("floor, floor, door", true, 2); DecideOut q; build_obs(in, q); gate(in, q); CHECK(!q.allowed[kOpen], "R3b reach 2: a door 3 blocks ahead is not a leaf"); }
  // R4: carrying rules unchanged: no take while carrying, no open without the key, take never targets a door, open never targets a key
  { DecideIn a1 = mkr("floor, key", true, 4); DecideOut q; build_obs(a1, q); gate(a1, q); CHECK(!q.allowed[kTake] && !q.allowed[kOpen], "R4 carrying + key ahead -> no take, no open");
    DecideIn a2 = mkr("floor, door", false, 4); build_obs(a2, q); gate(a2, q); CHECK(!q.allowed[kOpen] && !q.allowed[kTake], "R4 not carrying + door ahead -> no open, no take");
    DecideIn a3 = mkr("floor, floor, exit", true, 4); build_obs(a3, q); gate(a3, q); CHECK(!q.allowed[kOpen] && !q.allowed[kTake], "R4 exit ahead is never take/open");
    DecideIn a4 = mkr("floor, wall", true, 4); build_obs(a4, q); gate(a4, q); CHECK(!q.allowed[kOpen] && !q.allowed[kTake], "R4 wall ahead is never take/open"); }
  // R5: only the FIRST occluder counts (a door behind a wall / pit is not in reach)
  { DecideIn in = mkr("floor, pit", true, 4); DecideOut q; build_obs(in, q); gate(in, q); CHECK(!q.allowed[kOpen], "R5 pit before anything -> no open"); }
  // R6: flush / cell-0 behaviour unchanged with reach on: adjacent key -> take_w 1, adjacent door -> open_w 1, same option text as before
  { DecideIn in = mkr("key", false, 4); DecideOut q; build_obs(in, q); gate(in, q); CHECK(q.allowed[kTake] && q.take_w == 1, "R6 adjacent key -> take, take_w=1");
    DecideIn i2 = mkr("door", true, 4); build_obs(i2, q); gate(i2, q); CHECK(q.allowed[kOpen] && q.open_w == 1 && !q.allowed[kFwd], "R6 adjacent door -> open, open_w=1, no forward walk"); }
  // R7: reach 0 == way-shut gate exactly (all R-scenarios + T-scenarios, every field)
  { int bad = 0; const char* phs[] = {"key", "floor, key", "floor, floor, key", "door", "floor, door", "floor, floor, door", "floor, floor, floor, door", "floor, more", "wall", "floor, exit", "pit"};
    for (const char* ph : phs) for (int carry = 0; carry < 2; ++carry) for (int sh = 0; sh < 2; ++sh) {
      DecideIn in = mkr(ph, carry, 0, sh); DecideOut q; build_obs(in, q); gate(in, q);
      DecideIn ref = mk(ph, "wall", "floor, floor, more", "wall", carry, sh); DecideOut qr; build_obs(ref, qr); gate(ref, qr);
      for (int k = 0; k < kNAct; ++k) if (q.allowed[k] != qr.allowed[k]) ++bad;
      if (q.take_r != qr.take_r || q.open_r != qr.open_r || q.obs != qr.obs || q.blank != qr.blank) ++bad;
      if ((q.take_r >= 0 && q.take_w != 1) || (q.open_r >= 0 && q.open_w != 1)) ++bad; }
    CHECK(bad == 0, "R7 --reach 0 gate/obs identical to the way-shut gate (44 combos); every flag-off target is 1 block"); }
  // R8: way-shut still drops a blocked walk with reach on; shut door at band 2 -> open at 1 block (bump proves it)
  { DecideIn in = mkr("floor, door", true, 4, true); in.occ_world[kFwd] = 2; DecideOut q; build_obs(in, q); gate(in, q);
    CHECK(!q.allowed[kFwd], "R8 shut forward still dropped from the walk leaves with reach 4");
    CHECK(q.allowed[kOpen] && q.open_w == 1, "R8 shut + door band 2 -> open at 1 block (odometry: the walk into block 1 was blocked)");
    DecideIn s3 = mkr("floor, floor, door", true, 4, true); build_obs(s3, q); gate(s3, q);
    CHECK(!q.allowed[kOpen] && !q.allowed[kFwd], "R8b shut + door read 3 blocks ahead -> no ranged open (bump contradicts clear floor), no walk"); }
  // R9: B1 lit-door close-up: band-2 door with the glass-at-feet signature in band 1 -> 1 block, open WITHOUT a bump
  { DirView v = view_of("floor, door"); v.bpct[0][0] = 60; v.bpct[0][3] = 18; v.bpct[0][4] = 20;  // d031..d083 band 1 (F60 L18 G20)
    int kind, at; first_occluder(v, kind, at); CHECK(occluder_world(v, kind, at) == 1, "R9 glass-at-feet signature maps band-2 door to 1 block");
    DirView p = view_of("floor, door"); p.bpct[0][0] = 92; p.bpct[0][4] = 3;  // probe T.D band 1 (F92 G3): real door 2 blocks away
    first_occluder(p, kind, at); CHECK(occluder_world(p, kind, at) == 2, "R9 probe door two cells away (clean floor band 1) stays 2 blocks");
    DecideIn in = mk("floor, door", "wall", "floor, floor, more", "floor, wall", true); in.reach = 4; reach_fields(v, in, kFwd);
    DecideOut q; build_obs(in, q); gate(in, q);
    CHECK(q.allowed[kOpen] && q.open_w == 1 && !in.shut[kFwd], "R9 d031 first look + key + reach 4 -> open leaf at 1 block, no bump first"); }
  // R10: d030 door read as key (glass in the key bands) -> no ranged take even when not carrying
  { DirView v = view_of("floor, floor, key"); v.bpct[2][3] = 48; v.bpct[2][4] = 44; v.bpct[3][3] = 29; v.bpct[3][4] = 27;
    DecideIn in = mk("floor, floor, key", "wall", "floor, more", "wall", false); in.reach = 4; int kind, at; first_occluder(v, kind, at); in.occ[0] = kind; in.occ_at[0] = at; reach_fields(v, in, kFwd);
    DecideOut q; build_obs(in, q); gate(in, q); CHECK(!q.allowed[kTake], "R10 glassy key bands (d030) -> no ranged take");
    DirView k = view_of("key"); k.bpct[0][3] = 44; k.bpct[1][3] = 98;  // d025 real key (G=0)
    first_occluder(k, kind, at); CHECK(key_bands_clean(k, at), "R10 d025 real key bands are clean"); }
  // R11: own memory of a refused ranged interaction drops only the ranged leaf; walking closer stays
  { DecideIn in = mkr("floor, floor, door", true, 4); in.reach_miss[kFwd] = true; DecideOut q; build_obs(in, q); gate(in, q);
    CHECK(!q.allowed[kOpen] && q.allowed[kFwd], "R11 reach_miss -> ranged open dropped, forward walk kept");
    DecideIn i2 = mkr("door", true, 4); i2.reach_miss[kFwd] = true; build_obs(i2, q); gate(i2, q);
    CHECK(q.allowed[kOpen] && q.open_w == 1, "R11b reach_miss never removes the flush (old-rule) open"); }
  // R12: B1 nav2 approach replay with reach 4: d028 left (door 4 ahead), d029 forward (door 3 ahead) -> open at range, 0 bumps
  { DecideIn in = mk("floor, floor, floor, more", "wall", "floor, more", "floor, floor, floor, door", true); in.reach = 4; for (int r = 0; r < 4; ++r) in.occ_world[r] = in.occ_at[r];
    DecideOut q; build_obs(in, q); gate(in, q); CHECK(q.allowed[kOpen] && q.open_r == kLeft && q.open_w == 4, "R12 d028: door 4 blocks to the left -> open leaf (left) at 4");
    DecideIn i2 = mk("floor, floor, door", "wall", "floor, more", "wall", true); i2.reach = 4; for (int r = 0; r < 4; ++r) i2.occ_world[r] = i2.occ_at[r];
    build_obs(i2, q); gate(i2, q); CHECK(q.allowed[kOpen] && q.open_w == 3 && q.allowed[kFwd], "R12 d029: door 3 ahead -> open leaf at 3 and the forward walk"); }
  // R13: the live view tags a ranged target; flush views and the blank control are unchanged
  { DecideIn in = mkr("floor, floor, door", true, 4); DecideOut q; build_obs(in, q); gate(in, q);
    CHECK(q.obs.find("forward: floor, floor, door (new; the door is within reach)") != std::string::npos, "R13 ranged door tagged in the live view");
    CHECK(q.blank.find("within reach") == std::string::npos, "R13 blank control carries no reach tag");
    DecideIn k = mkr("floor, key", false, 4); build_obs(k, q); CHECK(q.obs.find("forward: floor, key (new; the key is within reach)") != std::string::npos, "R13 ranged key tagged");
    DecideIn f = mkr("door", true, 4); build_obs(f, q); CHECK(q.obs.find("forward: door\n") != std::string::npos, "R13 flush door view text unchanged (no tag)");
    DecideIn n = mkr("floor, floor, door", false, 4); build_obs(n, q); CHECK(q.obs.find("within reach") == std::string::npos, "R13 no key carried -> no door tag"); }
  std::printf("== %d passed, %d failed ==\n", passes, fails);
  return fails ? 1 : 0;
}
