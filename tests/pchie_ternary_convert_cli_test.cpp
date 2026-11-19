// Bolt PCHIE - pchie_ternary_convert refuses an output folder inside the model's folder, also when the way in
// goes through a symbolic link or junction, or through ".." in a part that does not exist yet.
//   pchie_ternary_convert_cli_test TOOL
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;
static int g_fail = 0;
static void check(bool ok, const char* what) {
  std::printf("[CONVERT-CLI] %-74s %s\n", what, ok ? "PASS" : "FAIL");
  if (!ok) ++g_fail;
}
static std::string q(const fs::path& p) { return "\"" + p.string() + "\""; }
static int run(const std::string& tool, const fs::path& model, const fs::path& out, const fs::path& log) {
  std::string cmd = q(tool) + " " + q(model) + " --out " + q(out) + " --no-default-skips --backend cpu > " + q(log) + " 2>&1";
#ifdef _WIN32
  cmd = "\"" + cmd + "\"";  // cmd /c strips one pair of outer quotes
#endif
  const int rc = std::system(cmd.c_str());
#ifdef _WIN32
  return rc;
#else
  return WIFEXITED(rc) ? WEXITSTATUS(rc) : 255;
#endif
}
static std::string slurp(const fs::path& p) { std::ifstream f(p, std::ios::binary); return std::string(std::istreambuf_iterator<char>(f), {}); }

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: pchie_ternary_convert_cli_test TOOL\n"); return 2; }
  const std::string tool = argv[1];
  std::error_code ec;
  const fs::path root = fs::temp_directory_path() / "pchie-convert-cli-test";
  fs::remove_all(root, ec);
  const fs::path mdir = root / "model", other = root / "other";
  fs::create_directories(mdir);
  fs::create_directories(other);
  // a tiny safetensors model: one 2 x 128 F32 matrix
  const std::string json = "{\"blk.0.ffn_up.weight\":{\"dtype\":\"F32\",\"shape\":[2,128],\"data_offsets\":[0,1024]}}";
  {
    std::ofstream f(mdir / "m.safetensors", std::ios::binary);
    const uint64_t n = json.size();
    for (int i = 0; i < 8; ++i) f.put(char((n >> (8 * i)) & 0xff));
    f << json;
    std::vector<float> w(256);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 7) - 3) * 0.25f;
    f.write(reinterpret_cast<const char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
  }
  const fs::path model = mdir / "m.safetensors", log = root / "log.txt";

  check(run(tool, model, mdir / "out", log) != 0 && slurp(log).find("inside the model's folder") != std::string::npos && !fs::exists(mdir / "out"),
        "an output folder inside the model's folder is refused");
  check(run(tool, model, other / "nope" / ".." / ".." / "model" / "out2", log) != 0 && !fs::exists(mdir / "out2"),
        "'..' in a part that does not exist yet is refused");
  // a link outside that points at the model's folder
  const fs::path link = other / "link-to-model";
  bool linked = false;
#ifdef _WIN32
  {
    const std::string mk = "cmd /c mklink /J " + q(link) + " " + q(mdir) + " >NUL 2>&1";
    linked = std::system(mk.c_str()) == 0 && fs::exists(link / "m.safetensors");
  }
#else
  fs::create_directory_symlink(mdir, link, ec);
  linked = !ec;
#endif
  check(linked, "made a link (junction on Windows) to the model's folder");
  if (linked) {
    check(run(tool, model, link / "out3", log) != 0 && slurp(log).find("inside the model's folder") != std::string::npos && !fs::exists(mdir / "out3"),
          "an output folder reached through a link into the model's folder is refused");
    // and the model reached through the link, output in the real model folder
    check(run(tool, link / "m.safetensors", mdir / "out4", log) != 0 && !fs::exists(mdir / "out4"),
          "a model opened through a link is checked against its real folder");
  }
  check(run(tool, model, other / "fresh", log) == 0 && fs::exists(other / "fresh"), "an output folder elsewhere is accepted (positive control)");
  if (linked) {
#ifdef _WIN32
    fs::remove(link, ec);  // removes the junction, not the folder it points at
#else
    fs::remove(link, ec);
#endif
  }
  fs::remove_all(root, ec);
  std::printf("[CONVERT-CLI] %s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
