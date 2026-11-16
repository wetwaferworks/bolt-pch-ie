// One plug interface for many games, after the BALROG pattern: every game exposes the same reset/step
// surface with a text observation (plus an optional glyph picture and his own senses), a fixed action
// list, integer rewards and a state digest. Agents plug into the other side through IAgentPlug.
// This is the socket Mjolnir plugs into later: an adapter (or a dreamed LoRA) is just another agent.
//
// Grading stays outside the agent: grade() reads the world after the episode, and nothing from it is
// ever passed to act(). Scratch research only, off the official path.
#pragma once

#include "pchie/harness/Task.h"
#include "pchie/research/SelfObservation.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pchie {
namespace harness {

struct PlugInfo {
  std::string name;                  // registry name, e.g. "lithokraft.maze.b1"
  std::string version;
  std::string family;                // "lithokraft", "litholattice", "craftax_like"
  bool stub = false;                 // a placeholder adapter, not the real game yet
  std::vector<std::string> actions;  // action i is actions[i]
  std::string instructions;
};

struct PlugObservation {
  std::string text;                  // what he is told this step
  std::string glyphs;                // optional picture, rows joined with '\n'
  int glyph_cols = 0;
  int glyph_rows = 0;
  bool has_self = false;             // his own senses (maze games)
  research::SelfObservation self;
  uint8_t open_mask = 0;             // moves he can see are open (bit m = research move m)
};

struct PlugStep {
  PlugObservation obs;
  int32_t reward = 0;
  bool action_ok = false;            // the action did something
  bool done = false;
  bool success = false;
};

/// After the episode only; never shown to the agent.
struct PlugGrade {
  bool success = false;
  int32_t progress = 0;
  int32_t target = 0;
  std::string note;
};

class IGamePlug {
 public:
  virtual ~IGamePlug() = default;
  virtual PlugInfo info() const = 0;
  virtual PlugObservation reset(uint64_t seed) = 0;
  virtual PlugStep step(int action) = 0;
  virtual PlugGrade grade() const = 0;
  /// SHA-256 of the full game state (for replay checks).
  virtual std::string state_digest() const = 0;
};

class IAgentPlug {
 public:
  virtual ~IAgentPlug() = default;
  virtual std::string name() const = 0;
  virtual void reset(const PlugInfo& info) = 0;
  /// Index into info.actions. why may hold a short reason for the transcript.
  virtual int act(const PlugObservation& obs, std::string& why) = 0;
  virtual void outcome(int action, const PlugStep& step) { (void)action; (void)step; }
};

using PlugFactory = std::function<std::unique_ptr<IGamePlug>()>;
using AgentFactory = std::function<std::unique_ptr<IAgentPlug>()>;

/// Built-in plugs: lithokraft.maze.b1, lithokraft.maze.stagea, litholattice.testbed.stub, craftax_like.grid.
std::vector<std::string> plug_names();
std::unique_ptr<IGamePlug> make_plug(const std::string& name, std::string& err);

struct EpisodeResult {
  bool done = false;
  bool success = false;
  uint32_t steps = 0;
  uint32_t failed_actions = 0;
  int64_t reward = 0;
  PlugGrade grade;
  std::string actions_sha256;
  std::string final_state;
  std::string err;
};

/// Runs one episode. With a transcript, every observation and action is recorded.
EpisodeResult run_episode(IGamePlug& game, IAgentPlug& agent, uint64_t seed, uint32_t budget, Transcript* transcript);

// ---- agents ----
/// Go-Explore (research L7) on his own senses; needs a plug with has_self (the maze games).
/// sense_open feeds the open moves he sees around him into his map (off = the research test's walk).
std::unique_ptr<IAgentPlug> go_explore_agent(bool sense_open = true);
/// Repeats a fixed list of action names (tests and smoke runs).
std::unique_ptr<IAgentPlug> cycle_agent(std::vector<std::string> action_names);

// ---- bridge to tasks ----
/// One sample per seed; the solver plays one episode, the scorer gives full marks for success and
/// otherwise half marks scaled by graded progress.
Task game_task(const std::string& plug_name, const std::vector<uint64_t>& seeds, uint32_t step_budget, AgentFactory agent);
std::shared_ptr<const Scorer> game_scorer();

}  // namespace harness
}  // namespace pchie
