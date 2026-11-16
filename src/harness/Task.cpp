#include "pchie/harness/Task.h"

#include <cctype>

namespace pchie {
namespace harness {

std::string trim_lower(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  std::string o = s.substr(a, b - a);
  for (char& c : o) c = char(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

namespace {

std::string choices_text(const std::vector<std::string>& c) {
  std::string s;
  for (size_t i = 0; i < c.size() && i < 26; ++i) {
    s += char('A' + i);
    s += ") ";
    s += c[i];
    s += '\n';
  }
  return s;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
  if (from.empty()) return;
  size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
}

class PromptTemplate final : public Solver {
 public:
  explicit PromptTemplate(std::string t) : t_(std::move(t)) {}
  std::string name() const override { return "prompt_template"; }
  bool solve(TaskState& st, const Generate&, std::string& err) override {
    if (!st.sample) { err = "no sample"; return false; }
    std::string p = t_;
    replace_all(p, "{choices}", choices_text(st.sample->choices));
    replace_all(p, "{input}", st.sample->input);
    st.prompt = p;
    return true;
  }

 private:
  std::string t_;
};

class GenerateSolver final : public Solver {
 public:
  std::string name() const override { return "generate"; }
  bool solve(TaskState& st, const Generate& gen, std::string& err) override {
    if (!gen) { err = "no model to call"; return false; }
    st.transcript.add(EventKind::Input, st.steps, st.prompt);
    st.output = gen(st.prompt);
    ++st.steps;
    st.transcript.add(EventKind::Model, st.steps, st.output);
    st.completed = true;
    return true;
  }
};

class Chain final : public Solver {
 public:
  explicit Chain(std::vector<std::unique_ptr<Solver>> p) : parts_(std::move(p)) {}
  std::string name() const override {
    std::string n = "chain(";
    for (size_t i = 0; i < parts_.size(); ++i) n += (i ? "," : "") + parts_[i]->name();
    return n + ")";
  }
  bool solve(TaskState& st, const Generate& gen, std::string& err) override {
    for (auto& s : parts_) {
      if (!s) { err = "empty solver in chain"; return false; }
      if (!s->solve(st, gen, err)) return false;
      if (st.completed) break;
    }
    return true;
  }

 private:
  std::vector<std::unique_ptr<Solver>> parts_;
};

Score verdict(bool ok, const std::string& answer, const std::string& why) {
  Score s;
  s.value_q16 = ok ? kFullScore : 0;
  s.answer = answer;
  s.explanation = why;
  return s;
}

class ExactMatch final : public Scorer {
 public:
  std::string name() const override { return "exact_match"; }
  Score score(const TaskState& st) const override {
    const std::string& t = st.sample ? st.sample->target : std::string();
    return verdict(st.output == t, st.output, st.output == t ? "matches" : "differs");
  }
};

class FirstWordMatch final : public Scorer {
 public:
  std::string name() const override { return "first_word_match"; }
  Score score(const TaskState& st) const override {
    std::string a = trim_lower(st.output);
    const size_t sp = a.find_first_of(" \t\r\n.,;:!?");
    if (sp != std::string::npos) a = a.substr(0, sp);
    const std::string t = trim_lower(st.sample ? st.sample->target : std::string());
    return verdict(!t.empty() && a == t, a, a == t ? "first word matches" : "first word differs");
  }
};

class Includes final : public Scorer {
 public:
  std::string name() const override { return "includes"; }
  Score score(const TaskState& st) const override {
    const std::string a = trim_lower(st.output);
    const std::string t = trim_lower(st.sample ? st.sample->target : std::string());
    const bool ok = !t.empty() && a.find(t) != std::string::npos;
    return verdict(ok, st.output, ok ? "target found" : "target not found");
  }
};

class ChoiceLetter final : public Scorer {
 public:
  std::string name() const override { return "choice_letter"; }
  Score score(const TaskState& st) const override {
    std::string got;
    for (const char c : st.output) {
      const char u = char(std::toupper(static_cast<unsigned char>(c)));
      if (u >= 'A' && u <= 'Z') { got = std::string(1, u); break; }
    }
    std::string t = st.sample ? st.sample->target : std::string();
    for (char& c : t) c = char(std::toupper(static_cast<unsigned char>(c)));
    return verdict(!got.empty() && got == t, got, got == t ? "right choice" : "wrong choice");
  }
};

}  // namespace

std::unique_ptr<Solver> prompt_template(std::string tmpl) { return std::make_unique<PromptTemplate>(std::move(tmpl)); }
std::unique_ptr<Solver> generate() { return std::make_unique<GenerateSolver>(); }
std::unique_ptr<Solver> chain(std::vector<std::unique_ptr<Solver>> parts) { return std::make_unique<Chain>(std::move(parts)); }
std::shared_ptr<const Scorer> exact_match() { return std::make_shared<ExactMatch>(); }
std::shared_ptr<const Scorer> first_word_match() { return std::make_shared<FirstWordMatch>(); }
std::shared_ptr<const Scorer> includes() { return std::make_shared<Includes>(); }
std::shared_ptr<const Scorer> choice_letter() { return std::make_shared<ChoiceLetter>(); }

}  // namespace harness
}  // namespace pchie
