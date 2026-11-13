// Scratch: read-only maze preview for stratifying the vision volume (setup only; never a decision input).
#include "pchie/MazeGrid.h"
#include "pchie/MazeOracle.h"
#include "pchie/LkMazeText.h"
#include "pchie/Sha256.h"
#include <cstdio>
#include <cstdlib>
#include <string>
using namespace pchie;
int main(int argc, char** argv) {
  const int size = argc > 1 ? std::atoi(argv[1]) : 16, s0 = argc > 2 ? std::atoi(argv[2]) : 1, s1 = argc > 3 ? std::atoi(argv[3]) : 40;
  for (int s = s0; s <= s1; ++s) {
    MazeGrid g(uint64_t(s), size, size);
    const std::string t = LkMazeText::encode(g);
    int floor = 0, dead = 0;
    for (int y = 0; y < size; ++y) for (int x = 0; x < size; ++x) {
      const char c = t[t.find('\n') + 1 + size_t(y) * (size + 1) + size_t(x)];
      if (c == '#') continue; ++floor; int n = 0;
      const int dx[4] = {1,-1,0,0}, dy[4] = {0,0,1,-1};
      for (int k = 0; k < 4; ++k) { int X = x + dx[k], Y = y + dy[k]; if (X < 0 || Y < 0 || X >= size || Y >= size) continue; if (t[t.find('\n') + 1 + size_t(Y) * (size + 1) + size_t(X)] != '#') ++n; }
      if (n == 1 && c == '.') ++dead;
    }
    const auto h = Sha256::hash(reinterpret_cast<const uint8_t*>(t.data()), t.size());
    char hx[17]; for (int i = 0; i < 8; ++i) std::snprintf(hx + 2 * i, 3, "%02x", h[i]);
    std::printf("size=%d seed=%d bfs=%d open_cells=%d dead_end_cells=%d sha16=%s\n", size, s, MazeOracle::shortest_length(g), floor, dead, hx);
  }
  return 0;
}
