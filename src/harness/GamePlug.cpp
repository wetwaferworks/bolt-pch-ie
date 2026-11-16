#include "pchie/harness/GamePlug.h"

#include "pchie/research/Digest.h"
#include "pchie/research/GoExplore.h"

#include <vector>

namespace pchie {
namespace harness {

// Defined in Plugs.cpp.
std::unique_ptr<IGamePlug> make_maze_plug(bool stagea);
std::unique_ptr<IGamePlug> make_testbed_stub();
std::unique_ptr<IGamePlug> make_craftax_like();

std::vector<std::string> plug_names() {
  return {"lithokraft.maze.b1", "lithokraft.maze.stagea", "litholattice.testbed.stub", "craftax_like.grid"};
}

std::unique_ptr<IGamePlug> make_plug(const std::string& name, std::string& err) {
  if (name == "lithokraft.maze.b1") return make_maze_plug(false);
  if (name == "lithokraft.maze.stagea") return make_maze_plug(true);
  if (name == "litholattice.testbed.stub") return make_testbed_stub();
  if (name == "craftax_like.grid") return make_craftax_like();
  err = "unknown plug: " + name;
  return nullptr;
}

EpisodeResult run_episode(IGamePlug& game, IAgentPlug& agent, uint64_t seed, uint32_t budget, Transcript* tr) {
  EpisodeResult r;
  const PlugInfo info = game.info();
  agent.reset(info);
  PlugObservation obs = game.reset(seed);
  if (tr) tr->add(EventKind::Input, 0, info.name + " seed=" + std::to_string(seed) + " agent=" + agent.name() + "\n" + info.instructions);
  std::vector<uint8_t> acts;
  for (uint32_t s = 0; s < budget; ++s) {
    if (tr) tr->add(EventKind::Observation, s, obs.glyphs.empty() ? obs.text : obs.text + "\n" + obs.glyphs);
    std::string why;
    const int a = agent.act(obs, why);
    if (a < 0 || a >= int(info.actions.size())) {
      r.err = "agent chose an action outside the list";
      if (tr) tr->add(EventKind::Error, s, r.err);
      break;
    }
    const PlugStep st = game.step(a);
    agent.outcome(a, st);
    acts.push_back(uint8_t(a));
    ++r.steps;
    r.reward += st.reward;
    if (!st.action_ok) ++r.failed_actions;
    if (tr) tr->add(EventKind::Action, s + 1, info.actions[size_t(a)] + (st.action_ok ? "" : " (nothing happened)") + (why.empty() ? "" : " | " + why));
    obs = st.obs;
    if (st.done) {
      r.done = true;
      r.success = st.success;
      break;
    }
  }
  r.grade = game.grade();
  r.actions_sha256 = research::sha256_hex(acts);
  r.final_state = game.state_digest();
  if (tr)
    tr->add(EventKind::Note, r.steps,
            "graded success=" + std::to_string(int(r.grade.success)) + " progress=" + std::to_string(r.grade.progress) + "/" +
                std::to_string(r.grade.target) + " reward=" + std::to_string(r.reward) + " actions_sha256=" + r.actions_sha256 +
                " state=" + r.final_state + (r.grade.note.empty() ? "" : " " + r.grade.note));
  return r;
}

namespace {

int action_index(const PlugInfo& info, const std::string& n) {
  for (size_t i = 0; i < info.actions.size(); ++i)
    if (info.actions[i] == n) return int(i);
  return -1;
}

class GoExploreAgent final : public IAgentPlug {
 public:
  explicit GoExploreAgent(bool sense_open) : sense_open_(sense_open) {}
  std::string name() const override { return sense_open_ ? "go_explore" : "go_explore(no_sense)"; }
  void reset(const PlugInfo& info) override {
    p_ = research::GoExplorePolicy();
    map_.clear();
    static const char* kNames[research::kActs] = {"north", "east", "south", "west", "pickup", "use"};
    for (int i = 0; i < research::kActs; ++i) map_.push_back(action_index(info, kNames[i]));
    noop_ = action_index(info, "noop");
  }
  int act(const PlugObservation& obs, std::string& why) override {
    if (!obs.has_self) { why = "no senses on this plug"; return noop_ >= 0 ? noop_ : 0; }
    if (sense_open_) p_.sense(obs.self, obs.open_mask);
    const int a = p_.decide(obs.self);
    last_ = a;
    if (a < 0 || a >= int(map_.size()) || map_[size_t(a)] < 0) { why = "at the exit"; return noop_ >= 0 ? noop_ : 0; }
    why = p_.phase() == research::GoExplorePhase::Return ? "return" : "explore";
    return map_[size_t(a)];
  }
  void outcome(int action, const PlugStep& step) override {
    int ra = -1;
    for (size_t i = 0; i < map_.size(); ++i)
      if (map_[i] == action) ra = int(i);
    if (ra >= 0) p_.outcome(ra, step.action_ok);
  }

 private:
  research::GoExplorePolicy p_;
  std::vector<int> map_;
  int noop_ = -1;
  int last_ = -1;
  bool sense_open_ = true;
};

class CycleAgent final : public IAgentPlug {
 public:
  explicit CycleAgent(std::vector<std::string> n) : names_(std::move(n)) {}
  std::string name() const override { return "cycle"; }
  void reset(const PlugInfo& info) override {
    idx_.clear();
    for (const auto& n : names_) idx_.push_back(action_index(info, n));
    k_ = 0;
  }
  int act(const PlugObservation&, std::string&) override {
    if (idx_.empty()) return -1;
    const int a = idx_[k_ % idx_.size()];
    ++k_;
    return a;
  }

 private:
  std::vector<std::string> names_;
  std::vector<int> idx_;
  size_t k_ = 0;
};

class AgentSolver final : public Solver {
 public:
  AgentSolver(std::string plug, AgentFactory f) : plug_(std::move(plug)), f_(std::move(f)) {}
  std::string name() const override { return "play(" + plug_ + ")"; }
  bool solve(TaskState& st, const Generate&, std::string& err) override {
    std::unique_ptr<IGamePlug> g = make_plug(plug_, err);
    if (!g) return false;
    std::unique_ptr<IAgentPlug> a = f_ ? f_() : nullptr;
    if (!a) { err = "no agent"; return false; }
    const EpisodeResult r = run_episode(*g, *a, st.sample->seed, st.step_budget, &st.transcript);
    if (!r.err.empty()) { err = r.err; return false; }
    st.steps = r.steps;
    st.completed = true;
    st.success = r.grade.success;
    st.progress = r.grade.progress;
    st.progress_target = r.grade.target;
    st.output = r.actions_sha256;
    st.facts.push_back({"reward", std::to_string(r.reward)});
    st.facts.push_back({"failed_actions", std::to_string(r.failed_actions)});
    st.facts.push_back({"final_state", r.final_state});
    return true;
  }

 private:
  std::string plug_;
  AgentFactory f_;
};

class GameScorer final : public Scorer {
 public:
  std::string name() const override { return "game_progress"; }
  Score score(const TaskState& st) const override {
    Score s;
    s.answer = st.success ? "success" : "unfinished";
    if (st.success) {
      s.value_q16 = kFullScore;
    } else if (st.progress_target > 0 && st.progress > 0) {
      const int32_t p = st.progress < st.progress_target ? st.progress : st.progress_target;
      s.value_q16 = int32_t((int64_t(p) * (kFullScore / 2)) / st.progress_target);
    }
    s.explanation = "progress " + std::to_string(st.progress) + "/" + std::to_string(st.progress_target);
    return s;
  }
};

}  // namespace

std::unique_ptr<IAgentPlug> go_explore_agent(bool sense_open) { return std::make_unique<GoExploreAgent>(sense_open); }
std::unique_ptr<IAgentPlug> cycle_agent(std::vector<std::string> names) { return std::make_unique<CycleAgent>(std::move(names)); }
std::shared_ptr<const Scorer> game_scorer() { return std::make_shared<GameScorer>(); }

Task game_task(const std::string& plug_name, const std::vector<uint64_t>& seeds, uint32_t step_budget, AgentFactory agent) {
  Task t;
  t.name = "play:" + plug_name;
  t.version = "1";
  t.step_budget = step_budget;
  for (uint64_t s : seeds) {
    Sample x;
    x.id = plug_name + "#" + std::to_string(s);
    x.seed = s;
    t.samples.push_back(x);
  }
  t.solver = [plug_name, agent] { return std::unique_ptr<Solver>(new AgentSolver(plug_name, agent)); };
  t.scorer = game_scorer();
  return t;
}

}  // namespace harness
}  // namespace pchie
