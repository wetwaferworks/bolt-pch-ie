#include "pchie/research/TreeOfThoughts.h"

#include "pchie/research/Digest.h"

#include <algorithm>
#include <sstream>

namespace pchie {
namespace research {

TotResult TreeOfThoughts::run(const Thought& root, const ProposeFn& propose, const EvaluateFn& evaluate,
                              IJobDispatcher* jobs) const {
  TotResult res;
  res.best = root;
  std::vector<Thought> frontier{root};
  frontier.front().id = 0;
  uint32_t next_id = 1;
  bool have_best = false;
  for (uint32_t level = 0; level < cfg_.depth && !frontier.empty(); ++level) {
    // Phase 1 (serial): propose in frontier order, ids in that order.
    std::vector<Thought> kids;
    for (const Thought& p : frontier) {
      std::vector<Thought> got = propose(p, cfg_.breadth);
      if (got.size() > cfg_.breadth) got.resize(cfg_.breadth);
      for (Thought& t : got) {
        t.id = next_id++;
        t.parent = p.id;
        t.depth = p.depth + 1;
        kids.push_back(std::move(t));
      }
    }
    if (kids.empty()) break;
    // Phase 2 (parallel): every job scores only its own slot.
    std::vector<int32_t> slot(kids.size(), 0);
    auto job = [&](int i) { slot[size_t(i)] = evaluate(kids[size_t(i)]); };
    if (jobs && jobs->size() > 0) jobs->parallel_for(int(kids.size()), job);
    else
      for (int i = 0; i < int(kids.size()); ++i) job(i);
    // Phase boundary: the join above. Phase 3 (serial): fold and rank.
    for (size_t i = 0; i < kids.size(); ++i) kids[i].score_q20 = slot[i];
    res.evaluated.insert(res.evaluated.end(), kids.begin(), kids.end());
    std::stable_sort(kids.begin(), kids.end(), [](const Thought& a, const Thought& b) {
      return a.score_q20 != b.score_q20 ? a.score_q20 > b.score_q20 : a.id < b.id;
    });
    if (kids.size() > cfg_.beam) kids.resize(cfg_.beam);
    const Thought& top = kids.front();
    if (!have_best || top.score_q20 > res.best.score_q20 ||
        (top.score_q20 == res.best.score_q20 && top.depth > res.best.depth)) {
      res.best = top;
      have_best = true;
    }
    frontier = std::move(kids);
    ++res.levels;
  }
  std::ostringstream o;
  o << "tree_of_thoughts.v1|" << cfg_.breadth << '|' << cfg_.depth << '|' << cfg_.beam;
  for (const Thought& t : res.evaluated) o << '|' << t.id << ',' << t.parent << ',' << t.text << ',' << t.score_q20;
  o << "|best=" << res.best.id;
  res.digest = sha256_hex(o.str());
  return res;
}

}  // namespace research
}  // namespace pchie
