// Bolt PCHIE - pchie_neml: look at Little Nemo's episodic memory ledger.
//   pchie_neml --status            where the ledger is, the switches, and the receipts in it
//   pchie_neml --block [--cap N]   the [SYSTEM: EPISODIC MEMORY LEDGER] block the chat interface will use
//                                  (token count from Little Nemo's real tokenizer; PCHIE_MODEL or the default model)
//   pchie_neml --priors            the earned priors text
//   pchie_neml --verify [--expect SEQ:SHA]...
//                                  check the ledger and the chat notes against their journals; --expect pins the
//                                  journal record SEQ of the ledger to the SHA-256 you noted earlier (exit 0 good,
//                                  1 refused, 2 usage)
//   pchie_neml --repair-tail       cut a torn last line (a save that was cut short) off either journal, after
//                                  checking every record before it; the whole journal is copied to a .bak first
// --neml DIR reads another ledger folder. Only --repair-tail writes, and only a torn last line.
#include "pchie/ModelPath.h"
#include "pchie/neml/NemlQuery.h"
#include "pchie/neml/NemlStore.h"
#include "pchie/pchie_gemma4.h"
#include "pchie/pchie_gguf.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using namespace pchie;

static std::string hexv(const std::vector<uint8_t>& v) {
  std::string s; char b[3];
  for (uint8_t x : v) { std::snprintf(b, sizeof b, "%02x", x); s += b; }
  return s;
}

// --verify: both save files against their journals, plus any pinned records of the ledger's journal.
static int verify(const neml::Config& c, const std::vector<std::string>& expects) {
  std::vector<std::pair<uint64_t, std::string>> pins;
  for (const auto& e : expects) {
    const size_t colon = e.find(':');
    char* end = nullptr;
    const unsigned long long seq = colon == std::string::npos ? 0 : std::strtoull(e.c_str(), &end, 10);
    if (colon == std::string::npos || colon == 0 || end != e.c_str() + colon || e[0] == '-' ||
        !neml::is_hex64(e.c_str() + colon + 1, e.size() - colon - 1)) {
      std::fprintf(stderr, "--expect takes SEQ:SHA (64 lowercase hex), got %s\n", e.c_str());
      return 2;
    }
    pins.emplace_back(uint64_t(seq), e.substr(colon + 1));
  }
  int bad = 0;
  const neml::SaveFile files[2] = {neml::ring_save(c), neml::chat_save(c)};
  const char* lock_names[2] = {"nemo_episodic.lock", "nemo_chat_notes.lock"};
  for (int i = 0; i < 2; ++i) {
    std::error_code ec;
    if (!std::filesystem::exists(files[i].file, ec) && !std::filesystem::exists(files[i].journal, ec)) {
      std::printf("%s: none yet\n", files[i].file.c_str());
      continue;
    }
    neml::SaveLock lock(neml::resolved_dir(c), lock_names[i]);
    if (!lock.held()) { std::printf("%s: %s\n", files[i].file.c_str(), neml::busy_text()); ++bad; continue; }
    neml::Loaded ld;
    std::string why;
    if (!neml::load_checked(files[i], ld, why)) { std::printf("REFUSED %s\n", why.c_str()); ++bad; continue; }
    if (!ld.has_journal) std::printf("%s: OK, no journal yet (it starts with the next save)\n", files[i].file.c_str());
    else std::printf("%s: OK, %llu journal records, last %llu:%s%s\n", files[i].file.c_str(), (unsigned long long)ld.records,
                     (unsigned long long)(ld.records - 1), ld.head.c_str(), ld.pending ? " (one record still to be written to the file; the next save does it)" : "");
    if (i == 0)
      for (const auto& p : pins) {
        if (p.first >= ld.line_sha.size()) { std::printf("REFUSED journal record %llu is missing\n", (unsigned long long)p.first); ++bad; }
        else if (ld.line_sha[size_t(p.first)] != p.second) { std::printf("REFUSED journal record %llu is %s, expected %s\n", (unsigned long long)p.first, ld.line_sha[size_t(p.first)].c_str(), p.second.c_str()); ++bad; }
        else std::printf("journal record %llu matches\n", (unsigned long long)p.first);
      }
  }
  return bad ? 1 : 0;
}

// --repair-tail: cut a torn last line, only after the rest checks out.
static int repair_tail(const neml::Config& c) {
  int bad = 0, done = 0;
  const neml::SaveFile files[2] = {neml::ring_save(c), neml::chat_save(c)};
  const char* lock_names[2] = {"nemo_episodic.lock", "nemo_chat_notes.lock"};
  for (int i = 0; i < 2; ++i) {
    std::error_code ec;
    if (!std::filesystem::exists(files[i].journal, ec)) continue;
    neml::SaveLock lock(neml::resolved_dir(c), lock_names[i]);
    if (!lock.held()) { std::printf("%s: %s\n", files[i].journal.c_str(), neml::busy_text()); ++bad; continue; }
    neml::Loaded ld;
    std::string why;
    if (neml::load_checked(files[i], ld, why)) { std::printf("%s: OK, nothing to repair\n", files[i].journal.c_str()); continue; }
    size_t removed = 0;
    std::string backup;
    if (!neml::journal_repair_tail(files[i], removed, backup, why)) { std::printf("NOT REPAIRED %s\n", why.c_str()); ++bad; continue; }
    std::printf("%s: cut %zu bytes of a torn last line; the whole journal was copied to %s first\n", files[i].journal.c_str(), removed, backup.c_str());
    ++done;
    if (!neml::load_checked(files[i], ld, why)) { std::printf("STILL REFUSED %s\n", why.c_str()); ++bad; }
  }
  if (!bad && !done) std::printf("nothing to repair\n");
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  std::string mode;
  size_t cap = 384;
  std::vector<std::string> expects;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    if (const int n = neml::parse_flag(argc, argv, i, neml::process_config()); n != 0) {
      if (n < 0) { std::fprintf(stderr, "bad value for %s\n", k.c_str()); return 2; }
      i += n - 1;
    } else if (k == "--status" || k == "--block" || k == "--priors" || k == "--verify" || k == "--repair-tail") mode = k;
    else if (k == "--cap" && i + 1 < argc) cap = size_t(std::atoi(argv[++i]));
    else if (k == "--expect" && i + 1 < argc) expects.push_back(argv[++i]);
    else { mode.clear(); break; }
  }
  if (mode.empty()) {
    std::fprintf(stderr, "usage: pchie_neml --status | --block [--cap N] | --priors | --verify [--expect SEQ:SHA]... | --repair-tail   [--neml DIR]\n");
    return 2;
  }
  if (!expects.empty() && mode != "--verify") { std::fprintf(stderr, "--expect goes with --verify\n"); return 2; }
  const neml::Config& c = neml::process_config();
  if (mode == "--verify") return verify(c, expects);
  if (mode == "--repair-tail") return repair_tail(c);
  const neml::Snapshot s = neml::Snapshot::take(c);
  if (!s.ok) { std::fprintf(stderr, "ledger %s refused: %s\n", neml::ring_path(c).c_str(), s.why.c_str()); return 1; }
  if (mode == "--status") {
    std::printf("ledger: %s\n", neml::ring_path(c).c_str());
    std::printf("recording: %s   memory prior: %s   chat write-back: %s\n", c.record ? "on" : "off",
                c.memory_prior ? "on" : "off", c.chat_writeback ? "on" : "off");
    std::printf("receipts: %zu in the ring, %llu written in all; snapshot sha256 %s\n", s.runs.size(), (unsigned long long)s.lifetime, hexv(s.sha256).c_str());
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
  std::string berr;
  const std::string blk = neml::memory_block(c, [&](const std::string& x) { return tk.encode(x).size(); }, cap, &nt, &berr);
  if (!berr.empty()) { std::fprintf(stderr, "memory refused: %s\n", berr.c_str()); return 1; }
  std::printf("%s", blk.c_str());
  std::fprintf(stderr, "%zu tokens (cap %zu)\n", nt, cap);
  return 0;
}