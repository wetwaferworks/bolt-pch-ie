// Bolt PCHIE - pchie_neml: look at Little Nemo's episodic memory ledger.
//   pchie_neml --status            where the ledger is, the switches, and the receipts in it
//   pchie_neml --block [--cap N]   the [SYSTEM: EPISODIC MEMORY LEDGER] block the chat interface will use
//                                  (token count from Little Nemo's real tokenizer; PCHIE_MODEL or the default model)
//   pchie_neml --priors            the earned priors text
// --neml DIR reads another ledger folder. The tool only reads; it never writes the ledger.
#include "pchie/ModelPath.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/neml/NemlStore.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_gguf.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace pchie;

static std::string hexv(const std::vector<uint8_t>& v) {
  std::string s; char b[3];
  for (uint8_t x : v) { std::snprintf(b, sizeof b, "%02x", x); s += b; }
  return s;
}

int main(int argc, char** argv) {
  std::string mode;
  size_t cap = 384;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (const int n = neml::parse_flag(argc, argv, i, neml::process_config()); n != 0) {
      if (n < 0) { std::fprintf(stderr, "bad value for %s\n", k.c_str()); return 2; }
      i += n - 1;
    } else if (k == "--status" || k == "--block" || k == "--priors") mode = k;
    else if (k == "--cap" && i + 1 < argc) cap = size_t(std::atoi(argv[++i]));
    else { mode.clear(); break; }
  }
  if (mode.empty()) {
    std::fprintf(stderr, "usage: pchie_neml --status | --block [--cap N] | --priors   [--neml DIR]\n");
    return 2;
  }
  const neml::Config& c = neml::process_config();
  const neml::Snapshot s = neml::Snapshot::take(c);
  if (!s.ok) { std::printf("ledger %s is unreadable\n", neml::ring_path(c).c_str()); return 1; }
  if (mode == "--status") {
    std::printf("ledger: %s\n", neml::ring_path(c).c_str());
    std::printf("recording: %s   memory prior: %s   chat write-back: %s\n", c.record ? "on" : "off",
                c.memory_prior ? "on" : "off", c.chat_writeback ? "on" : "off");
    std::printf("receipts: %zu in the ring, %u written in all; snapshot sha256 %s\n", s.runs.size(), s.lifetime, hexv(s.sha256).c_str());
    for (const auto& r : s.runs)
      std::printf("  #%u %s %s %ux%u seed %u steps=%u forks=%u dead_ends=%u %s\n", r.run_id, r.exit_reached ? "EXIT" : "MISS",
                  neml::path_name(r.path_kind), r.maze_w, r.maze_h, r.maze_seed, r.total_steps, r.mcts_forks_triggered,
                  r.dead_ends_hit, neml::get_str(r.note, sizeof r.note).c_str());
    return 0;
  }
  if (mode == "--priors") { std::printf("%s", neml::priors_text(neml::priors(s.runs)).c_str()); return 0; }
  GgufFile gf;
  Gemma4Tokenizer tk;
  std::string path, err;
  if (!resolve_model_path(path, err) || !gf.open(path, err) || !tk.load(gf, err)) {
    std::fprintf(stderr, "need Little Nemo's tokenizer for the token cap: %s\n", err.c_str());
    return 1;
  }
  size_t nt = 0;
  const std::string blk = neml::memory_block(c, [&](const std::string& x) { return tk.encode(x).size(); }, cap, &nt);
  std::printf("%s", blk.c_str());
  std::fprintf(stderr, "%zu tokens (cap %zu)\n", nt, cap);
  return 0;
}