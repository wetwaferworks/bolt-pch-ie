// Bolt PCHIE - MiniJson size and depth limits (labeled unit checks, never a golden).
// EXPECTED (written before the limits were built):
//   - Every limit passes at its edge and refuses one past it: whole text, depth, one string, items in one
//     array, keys in one object, values in the document.
//   - Tighter caller limits work; looser ones are clamped to the defaults (never widened).
//   - Duplicate keys parse by default (old behavior) and are refused when asked.
//   - A refused parse leaves a short error and never crashes; the MCP shapes that parsed before still parse.
//   - Nothing here touches any chain or hash.
#include "pchie/MiniJson.h"

#include <cstdio>
#include <string>

using namespace pchie;

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char* what) {
  std::printf("[MJLIM] %-84s %s\n", what, ok ? "PASS" : "FAIL");
  if (ok) ++g_pass; else ++g_fail;
  std::fflush(stdout);
}

static bool ok(const std::string& t, const MiniJson::Limits& l) {
  MiniJson j;
  std::string e;
  return MiniJson::parse(t, j, e, l);
}
static std::string err_of(const std::string& t, const MiniJson::Limits& l) {
  MiniJson j;
  std::string e;
  MiniJson::parse(t, j, e, l);
  return e;
}
static std::string nest(int depth) { return std::string(size_t(depth), '[') + std::string(size_t(depth), ']'); }
static std::string arr(size_t n) {
  std::string s = "[";
  for (size_t k = 0; k < n; ++k) s += k ? ",0" : "0";
  return s + "]";
}
static std::string obj(size_t n) {
  std::string s = "{";
  for (size_t k = 0; k < n; ++k) s += (k ? ",\"k" : "\"k") + std::to_string(k) + "\":0";
  return s + "}";
}

int main() {
  const MiniJson::Limits def{};
  check(def.max_bytes == MiniJson::kMaxBytes && def.max_depth == 64, "defaults: 16 MiB text, depth 64");

  // Old behavior kept.
  check(ok("{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"{\\\"x\\\":1}\"}]}}", def),
        "MCP reply shape still parses with default limits");
  check(ok(nest(65), def) && !ok(nest(66), def), "depth: root + 64 levels ok, one more refused (unchanged)");

  // Whole text.
  MiniJson::Limits l{};
  l.max_bytes = 16;
  check(ok("[1,2,3,4,5,6,7]", l), "text: 15 bytes under a 16-byte cap ok");
  check(ok("[1,2,3,4,5,6,77]", l), "text: exactly 16 bytes ok");
  check(!ok("[1,2,3,4,5,6,777]", l) && err_of("[1,2,3,4,5,6,777]", l) == "json: input too large",
        "text: 17 bytes refused with 'input too large'");
  check(!ok(std::string(MiniJson::kMaxBytes + 1, ' '), def), "text: one byte over 16 MiB refused before reading");

  // Depth.
  l = MiniJson::Limits{};
  l.max_depth = 3;
  check(ok(nest(4), l), "depth: cap 3 allows root + 3 levels");
  check(!ok(nest(5), l) && err_of(nest(5), l) == "json: too deep", "depth: one more refused 'too deep'");

  // One string.
  l = MiniJson::Limits{};
  l.max_string_bytes = 8;
  check(ok("\"12345678\"", l), "string: 8 bytes at an 8-byte cap ok");
  check(!ok("\"123456789\"", l) && err_of("\"123456789\"", l) == "json: string too long", "string: 9 bytes refused");
  check(!ok("{\"123456789\":1}", l), "string: key over the cap refused too");
  check(ok("\"\\u00e9\\u00e9\\u00e9\\u00e9\"", l), "string: cap counts decoded bytes (4 x 2-byte chars = 8) ok");
  check(!ok("\"\\u00e9\\u00e9\\u00e9\\u00e9x\"", l), "string: decoded 9 bytes refused");
  check(!ok("\"12345678", l), "string: unterminated still refused");

  // Items in one array, keys in one object.
  l = MiniJson::Limits{};
  l.max_items = 4;
  check(ok(arr(4), l), "array: 4 items at a cap of 4 ok");
  check(!ok(arr(5), l) && err_of(arr(5), l) == "json: too many items in one array", "array: 5 items refused");
  check(ok(obj(4), l), "object: 4 keys at a cap of 4 ok");
  check(!ok(obj(5), l) && err_of(obj(5), l) == "json: too many keys in one object", "object: 5 keys refused");
  check(ok("[[0,0,0,0],[0,0,0,0]]", l), "array: the cap is per container, not per document");

  // Values in the document.
  l = MiniJson::Limits{};
  l.max_values = 5;
  check(ok("[1,2,3,4]", l), "values: root + 4 = 5 at a cap of 5 ok");
  check(!ok("[1,2,3,4,5]", l) && err_of("[1,2,3,4,5]", l) == "json: too many values", "values: 6 refused");

  // Looser limits are clamped, never widened.
  l = MiniJson::Limits{};
  l.max_depth = 100000;
  check(!ok(nest(66), l), "clamp: depth 100000 asked, still capped at 64");
  l.max_depth = 0;
  check(ok(nest(65), l) && !ok(nest(66), l), "clamp: zero means the default");
  l = MiniJson::Limits{};
  l.max_bytes = size_t(1) << 40;
  check(!ok(std::string(MiniJson::kMaxBytes + 1, ' '), l), "clamp: text cap cannot be raised past 16 MiB");

  // Duplicate keys.
  check(ok("{\"a\":1,\"a\":2}", def), "duplicates: allowed by default (old behavior)");
  l = MiniJson::Limits{};
  l.reject_duplicate_keys = true;
  check(!ok("{\"a\":1,\"a\":2}", l) && err_of("{\"a\":1,\"a\":2}", l) == "json: duplicate key",
        "duplicates: refused when asked");
  check(ok("{\"a\":1,\"b\":{\"a\":2}}", l), "duplicates: same key in a nested object is fine");

  // A refused parse leaves out empty and usable.
  {
    MiniJson j;
    std::string e;
    l = MiniJson::Limits{};
    l.max_items = 2;
    const bool r = MiniJson::parse("[1,2,3]", j, e, l);
    check(!r && !e.empty(), "refusal: returns false with a message");
    const bool r2 = MiniJson::parse("[1,2]", j, e, l);
    check(r2 && j.type == MiniJson::Type::Array && j.arr.size() == 2, "refusal: the same object parses again");
  }

  // Big but legal: a 1 MiB string inside the default caps.
  check(ok("\"" + std::string(size_t(1) << 20, 'a') + "\"", def), "default: a 1 MiB string still parses");

  std::printf("[MJLIM] %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
