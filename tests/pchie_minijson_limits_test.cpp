// Bolt PCHIE - MiniJson size and depth limits (labeled unit checks, never a golden).
// EXPECTED (written before the limits were built):
//   - Every limit passes at its edge and refuses one past it: whole text, depth, one string, items in one
//     array, keys in one object, values in the document.
//   - Tighter caller limits work; looser ones are clamped to the defaults (never widened).
//   - Duplicate keys parse by default (old behavior) and are refused when asked.
//   - A refused parse leaves a short error and never crashes; the MCP shapes that parsed before still parse.
//   - Nothing here touches any chain or hash.
//   - (2026-11-17) Memory: every buffer the parse allocates is charged before it is allocated. A 16 MiB line of
//     empty arrays is refused at the 64 MiB ceiling, not after 427 MiB. Tighter budgets work at their edge, a
//     looser one is clamped, and long number tokens cost no copy. A big object with duplicate refusal on is
//     checked in n log n.
#include "pchie/MiniJson.h"

#include <chrono>
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

  // Memory ceiling (review 2026-11-16, M4).
  check(def.max_memory_bytes == MiniJson::kMaxMemoryBytes && MiniJson::kMaxMemoryBytes == (size_t(64) << 20),
        "memory: default ceiling 64 MiB");
  {
    // The review's case: a 16 MiB line of nested empty arrays, under every other cap until the 4M-value one.
    const size_t target = MiniJson::kMaxBytes - 8;
    std::string b = "[[";
    b.reserve(target + 16);
    size_t n = 0;
    while (b.size() + 6 < target) {
      b += "[],";
      if (++n % 1000000 == 0) {
        b.back() = ']';
        b += ",[";
      }
    }
    b += "[]]]";
    check(b.size() <= MiniJson::kMaxBytes && err_of(b, def) == "json: memory ceiling",
          "memory: 16 MiB of empty arrays refused at the ceiling, before the value cap");
  }
  l = MiniJson::Limits{};
  l.max_memory_bytes = 64 * 1024;
  check(ok(arr(100), l), "memory: 100 numbers fit a 64 KiB budget");
  check(!ok(arr(20000), l) && err_of(arr(20000), l) == "json: memory ceiling", "memory: 20000 numbers do not");
  check(ok("\"" + std::string(20000, 'a') + "\"", l), "memory: a 20 KB string fits a 64 KiB budget");
  check(!ok("\"" + std::string(70000, 'a') + "\"", l), "memory: a 70 KB string does not");
  check(!ok(obj(5000), l), "memory: an object's members are charged too");
  {
    // Find the exact edge for one shape, then check one byte less refuses it.
    const std::string t = arr(50);
    size_t lo = 1, hi = size_t(1) << 20;
    MiniJson::Limits e{};
    while (lo < hi) {
      const size_t mid = lo + (hi - lo) / 2;
      e.max_memory_bytes = mid;
      if (ok(t, e)) hi = mid; else lo = mid + 1;
    }
    e.max_memory_bytes = lo;
    const bool at = ok(t, e);
    e.max_memory_bytes = lo - 1;
    check(at && !ok(t, e), "memory: passes at its edge, refused one byte under");
  }
  {
    // A long number token is read in place: a 1 MB run of digits needs no heap at all.
    MiniJson::Limits n1{};
    n1.max_memory_bytes = 1;
    check(ok("1." + std::string(1000000, '0'), n1), "memory: a long number token is read in place, not copied");
  }
  l = MiniJson::Limits{};
  l.max_memory_bytes = size_t(1) << 40;
  check(err_of(arr(800000), l) == "json: memory ceiling", "clamp: memory ceiling cannot be raised past 64 MiB");
  check(ok(arr(100000), def), "memory: 100000 numbers still parse with the default ceiling");

  // Duplicate refusal on a big object costs n log n, and still finds the duplicate.
  {
    l = MiniJson::Limits{};
    l.reject_duplicate_keys = true;
    std::string big = obj(100000);
    const auto t0 = std::chrono::steady_clock::now();
    const bool clean = ok(big, l);
    big.back() = ',';
    big += "\"k12345\":1}";
    const bool dup = !ok(big, l) && err_of(big, l) == "json: duplicate key";
    const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    check(clean, "duplicates: 100000 distinct keys pass with refusal on");
    check(dup, "duplicates: one repeated key at the end of 100000 is found");
    check(el < 20.0, "duplicates: the big-object check finishes in seconds, not hours");
    std::printf("[MJLIM] 100000-key duplicate checks: %.2f s\n", el);
  }

  std::printf("[MJLIM] %d PASS / %d FAIL\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
