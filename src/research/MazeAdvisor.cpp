#include "pchie/research/MazeAdvisor.h"

#include "pchie/research/Digest.h"

#include <fstream>
#include <iterator>
#include <sstream>

namespace pchie {
namespace research {

namespace {
const int32_t kTokenOfAct[kActs] = {7, 9, 8, 10, 11, 12};  // North, East, South, West, Pickup, Use
}

int MazeResearchAdvisor::act_of_token(int32_t token) {
  for (int a = 0; a < kActs; ++a)
    if (kTokenOfAct[a] == token) return a;
  return -1;
}

int32_t MazeResearchAdvisor::token_of_act(int act) { return (act >= 0 && act < kActs) ? kTokenOfAct[act] : -1; }

uint64_t MazeResearchAdvisor::model_state(int32_t x, int32_t y, bool have_key, bool opened) {
  return (place_key(x, y) << 2) | (uint64_t(have_key) << 1) | uint64_t(opened);
}

MazeResearchAdvisor::MazeResearchAdvisor(ResearchFlags flags) : f_(std::move(flags)), log_(f_.log) {
  trace_.run_id = "maze";
}

bool MazeResearchAdvisor::engaged() const { return f_.go_explore || f_.lookahead || f_.step_scorer || f_.reflexion; }

bool MazeResearchAdvisor::prepare(std::string& err) {
  if (f_.lookahead) {
    std::ifstream in(f_.dynamics, std::ios::binary);
    if (!in) {
      err = "research: cannot read the frozen dynamics file (fail closed)";
      return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string why;
    if (!FrozenDynamics::from_canonical(text, f_.dynamics_sha, model_, why)) {
      err = "research: " + why + " (fail closed)";
      return false;
    }
    model_ready_ = true;
  }
  std::ostringstream o;
  o << "{\"format\":\"pchie.research.v1\",\"event\":\"start\",\"flags\":\"" << f_.identity_line() << "\",\"dynamics\":\""
    << (model_ready_ ? model_.sha256() : std::string("-")) << "\"";
  return log_.append(o.str(), err);
}

SelfObservation MazeResearchAdvisor::observe(const rl::ResearchView& v) const {
  SelfObservation o;
  o.x = v.x - ox_;
  o.y = v.y - oy_;
  o.heading = heading_;
  o.have_key = v.have_key;
  o.key_here = v.key_here;
  o.door_near = v.door_near;
  o.at_exit = v.at_exit;
  o.danger_mask = v.danger_mask;
  return o;
}

void MazeResearchAdvisor::close_transition(const SelfObservation& now) {
  if (!pending_) return;
  const int a = act_of_token(pending_token_);
  const SelfObservation& from = pending_from_;
  bool ok = true;
  if (a >= 0 && a < kMoves) ok = (now.x != from.x || now.y != from.y);
  else if (a == kPickup) ok = now.have_key && !from.have_key;
  else if (a == kUse) ok = from.door_near && !now.door_near;
  if (a == kUse && ok) opened_ = true;
  if (a >= 0 && a < kMoves && ok) heading_ = uint8_t(a);
  if (f_.go_explore) explore_.outcome(a, ok);
  if (a >= 0 && a < kMoves) map_.tried(from.x, from.y, a, ok);
  int32_t reward = 0;
  if (now.at_exit) reward = 1 << 20;
  else if (now.have_key && !from.have_key) reward = 1 << 18;
  Transition t{model_state(from.x, from.y, from.have_key, pending_from_opened_), uint8_t(a < 0 ? 0 : a),
               model_state(now.x, now.y, now.have_key, opened_), reward};
  transitions_.push_back(t);
  ++st_.transitions;
  StepRecord r;
  r.x = from.x;
  r.y = from.y;
  r.action = a;
  r.succeeded = ok;
  trace_.steps.push_back(r);
  std::ostringstream o;
  o << "{\"format\":\"pchie.research.v1\",\"event\":\"transition\",\"state\":" << t.state << ",\"action\":"
    << int(t.action) << ",\"next\":" << t.next << ",\"reward_q20\":" << t.reward_q20 << ",\"ok\":" << int(ok);
  std::string e;
  if (!log_.append(o.str(), e)) err_ = e;
  pending_ = false;
}

bool MazeResearchAdvisor::advise(uint32_t task, const rl::ResearchView& view, const std::vector<int32_t>& legal,
                                 std::vector<rl::ResearchProposal>& proposals, std::vector<rl::ResearchCap>& caps,
                                 std::string& err) {
  if (!err_.empty()) {
    err = "research: " + err_ + " (fail closed)";
    return false;
  }
  if (!have_origin_) {
    ox_ = view.x;
    oy_ = view.y;
    have_origin_ = true;
  }
  const SelfObservation o = observe(view);
  close_transition(o);
  map_.stand(o);
  map_.sense(o.x, o.y, view.open_mask);
  ++st_.decisions;
  last_task_ = task;
  last_obs_ = o;
  last_props_.clear();
  auto is_legal = [&](int32_t tok) {
    for (int32_t l : legal)
      if (l == tok) return true;
    return false;
  };
  if (f_.go_explore) {
    explore_.sense(o, view.open_mask);
    const int a = explore_.decide(o);
    const int32_t tok = token_of_act(a);
    if (tok >= 0 && is_legal(tok)) {
      proposals.push_back(rl::ResearchProposal{tok, go_explore_share_q20, kFromGoExplore});
      ++st_.go_explore_proposals;
    }
  }
  if (f_.lookahead && model_ready_) {
    const uint64_t s = model_state(o.x, o.y, o.have_key, opened_);
    if (!model_.knows(s)) {
      ++st_.lookahead_unknown_state;
    } else {
      LookaheadConfig cfg;
      cfg.depth = f_.lookahead_depth;
      cfg.untried_value_q20 = -(1 << 16);
      const LookaheadChoice c = model_.best_action(s, cfg);
      const int32_t tok = token_of_act(c.action);
      if (tok >= 0 && is_legal(tok)) {
        proposals.push_back(rl::ResearchProposal{tok, lookahead_share_q20, kFromLookahead});
        ++st_.lookahead_proposals;
      }
    }
  }
  std::ostringstream sc;
  if (f_.step_scorer) {
    for (int32_t tok : legal) {
      const int a = act_of_token(tok);
      if (a < 0 || a >= kMoves) continue;
      const int32_t score = scorer_.score_q20(step_features(map_, o, a, 0));
      ++st_.scored_moves;
      sc << (sc.tellp() > 0 ? "," : "") << '[' << tok << ',' << score << ']';
      if (f_.step_scorer_mode == ScorerMode::Enforce) caps.push_back(rl::ResearchCap{tok, score});
    }
  }
  last_props_ = proposals;
  std::ostringstream lg;
  lg << "{\"format\":\"pchie.research.v1\",\"event\":\"advise\",\"decision\":" << st_.decisions << ",\"task\":" << task
     << ",\"x\":" << o.x << ",\"y\":" << o.y << ",\"key\":" << int(o.have_key) << ",\"open\":" << int(view.open_mask)
     << ",\"proposals\":[";
  for (size_t i = 0; i < proposals.size(); ++i)
    lg << (i ? "," : "") << '[' << proposals[i].token << ',' << proposals[i].share_q20 << ',' << int(proposals[i].source)
       << ']';
  lg << "],\"scores\":[" << sc.str() << "]";
  if (!log_.append(lg.str(), err)) return false;
  return true;
}

void MazeResearchAdvisor::decided(int32_t token, const rl::ResearchView& after) {
  pending_ = true;
  pending_token_ = token;
  pending_from_ = last_obs_;
  pending_from_opened_ = opened_;
  pending_after_ = after;
  std::ostringstream o;
  o << "{\"format\":\"pchie.research.v1\",\"event\":\"decided\",\"decision\":" << st_.decisions << ",\"token\":" << token;
  std::string e;
  if (!log_.append(o.str(), e)) err_ = e;
}

void MazeResearchAdvisor::end_run() {
  if (pending_) close_transition(observe(pending_after_));
  trace_.reached_exit = !transitions_.empty() && transitions_.back().reward_q20 >= (1 << 20);
  if (f_.reflexion) {
    const auto lessons = Reflexion::reflect(trace_);
    st_.lessons = lessons.size();
    LessonLedger ledger(f_.lessons);
    std::string e;
    if (!ledger.append(trace_.run_id + ":" + log_.head().substr(0, 16), lessons, e)) err_ = e;
  }
  std::ostringstream o;
  o << "{\"format\":\"pchie.research.v1\",\"event\":\"end\",\"decisions\":" << st_.decisions
    << ",\"transitions\":" << st_.transitions << ",\"lessons\":" << st_.lessons << ",\"go_explore\":\""
    << explore_.digest() << "\"";
  std::string e;
  if (!log_.append(o.str(), e)) err_ = e;
}

std::string MazeResearchAdvisor::digest() const { return sha256_hex(log_.head() + "|" + explore_.digest()); }

}  // namespace research
}  // namespace pchie
