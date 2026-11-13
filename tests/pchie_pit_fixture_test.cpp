// Bolt PCHIE Mirror-Sim gate - pit fixture (labeled unit-test only, never a golden or baseline).
// Rule: every pit must have a safe route around it. BFS with all pit
// cells treated as walls must still reach the exit. Reaching the exit is a
// win; falling into a pit is fail/stuck, never a win.
// EXPECTED OUTPUTS (defined before implementation; not a golden lock):
//   - pit_safe_bypass() is a hand-built 7x6 grid with one pit on the apparent
//     shortcut and a safe route around it. shortest_length_avoiding_pits == 10.
//   - That BFS path never steps onto a pit. at_exit is the only win.
//   - Stepping into the pit yields in_pit, not at_exit, and no further legal
//     actions (stuck / fail).
//   - Default seed=1 16x16 B1 maze has no pits, so avoiding-pits BFS equals
//     the ordinary BFS (50). Default hashes are not this fixture.
// The Mirror-Sim CoW action gate is not built here.
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"

#include <cstdio>
#include <vector>

using namespace pchie;

static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[PIT-FIXTURE] %-76s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
  std::fflush(stdout);
}

int main() {
  std::printf("[PIT-FIXTURE] pit-safe-bypass fixture (unit only; Mirror-Sim gate not built)\n");

  const MazeGrid g = MazeGrid::pit_safe_bypass();
  check(g.width() == 7 && g.height() == 6 && g.x() == 1 && g.y() == 1,
        "fixture: 7x6, start at (1,1)");
  check(g.cell(2, 2) == MazeCell::Pit && g.cell(5, 1) == MazeCell::Exit && !g.in_pit() && !g.at_exit(),
        "fixture: one pit at (2,2), exit at (5,1), start is not in a pit");

  const int avoid = MazeOracle::shortest_length_avoiding_pits(g);
  check(avoid == 10, "fixture: BFS with pits treated as walls reaches the exit in 10 steps");
  std::printf("[PIT-FIXTURE] pit-avoiding BFS path length=%d (expected 10)\n", avoid);

  const auto path = MazeOracle::shortest_path_avoiding_pits(g);
  MazeGrid walk = g;
  bool path_ok = (int(path.size()) == avoid) && avoid > 0;
  for (MazeAction a : path) {
    if (!walk.apply(a) || walk.in_pit()) {
      path_ok = false;
      break;
    }
  }
  check(path_ok && walk.at_exit() && !walk.in_pit(),
        "fixture: the pit-avoiding path never steps on a pit and ends at the exit");

  // Falling in is fail/stuck, never a win: from (1,2) East is the pit.
  MazeGrid trap = g;
  check(trap.apply(MazeAction::South) && trap.x() == 1 && trap.y() == 2 && !trap.in_pit(),
        "fixture: opening South reaches the cell west of the pit");
  check(trap.apply(MazeAction::East) && trap.in_pit() && trap.x() == 2 && trap.y() == 2,
        "fixture: East from (1,2) falls into the pit");
  check(!trap.at_exit() && trap.legal_actions().empty(),
        "fixture: in a pit is fail/stuck (not at_exit, no legal actions)");
  check(!trap.apply(MazeAction::East) && !trap.apply(MazeAction::North) && trap.in_pit() && !trap.at_exit(),
        "fixture: a pit stay is never a win");

  // Default B1 maze is unchanged: no pits, same 50-step BFS.
  MazeGrid def(1, 16, 16);
  int pit_count = 0;
  for (int y = 0; y < def.height(); ++y)
    for (int x = 0; x < def.width(); ++x)
      if (def.cell(x, y) == MazeCell::Pit) ++pit_count;
  const int bfs = MazeOracle::shortest_length(def);
  const int bfs_safe = MazeOracle::shortest_length_avoiding_pits(def);
  check(pit_count == 0 && bfs == 50 && bfs_safe == 50,
        "default seed=1 16x16: zero pits, avoiding-pits BFS == BFS == 50");

  std::printf("[PIT-FIXTURE] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
