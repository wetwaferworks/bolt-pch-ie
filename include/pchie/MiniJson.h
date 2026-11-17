// Bolt PCHIE - minimal JSON value + parser for the LithoKraft MCP client (pchie_lk_pilot).
// Concrete final. Serial, no mutex, no atomic. Reads only what the MCP wire needs: objects,
// arrays, strings (with \uXXXX -> UTF-8), numbers, true/false/null. Depth capped (fail closed).
// Size capped too (2026-11-16): the whole text, each string, the items in one array or object, and the
// number of values in the document all have ceilings, checked while reading. The defaults match the game's
// own 16 MiB line cap, so every reply that parsed before still parses. A caller may pass tighter Limits;
// values looser than the defaults are clamped to them. Duplicate keys can be refused on request.
// Memory capped too (2026-11-17): every string and container buffer the parse allocates is charged against a
// byte budget (64 MiB by default) before it is allocated, and nothing is refunded, so the charge is an upper bound
// on what the parse ever holds. Before this a 16 MiB line of empty arrays peaked at 427 MiB before the value cap
// refused it. Buffers grow by doubling under our own reserve calls, so the charge doesn't depend on the standard
// library's growth policy. The duplicate-key check sorts once per object (n log n), not pairwise.
// Not on the official chain; nothing here is hashed.
#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace pchie {

class MiniJson final {
 public:
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string str;
  std::vector<MiniJson> arr;
  std::vector<std::pair<std::string, MiniJson>> obj;

  static constexpr int kMaxDepth = 64;
  static constexpr size_t kMaxBytes = size_t(16) << 20;        // the game's MCP line cap
  static constexpr size_t kMaxStringBytes = size_t(16) << 20;  // one decoded string
  static constexpr size_t kMaxItems = size_t(1) << 20;         // entries in one array or object
  static constexpr size_t kMaxValues = size_t(4) << 20;        // values in the whole document
  static constexpr size_t kMaxMemoryBytes = size_t(64) << 20;  // heap the parse may allocate, all buffers

  // Ceilings for one parse. Zero, or anything above the matching kMax*, means the kMax* default.
  struct Limits {
    size_t max_bytes = kMaxBytes;
    int max_depth = kMaxDepth;
    size_t max_string_bytes = kMaxStringBytes;
    size_t max_items = kMaxItems;
    size_t max_values = kMaxValues;
    size_t max_memory_bytes = kMaxMemoryBytes;
    bool reject_duplicate_keys = false;
  };

  static bool parse(const std::string& text, MiniJson& out, std::string& err) {
    return parse(text, out, err, Limits{});
  }

  static bool parse(const std::string& text, MiniJson& out, std::string& err, const Limits& requested) {
    Ctx cx;
    cx.lim = clamp(requested);
    out = MiniJson{};
    if (text.size() > cx.lim.max_bytes) {
      err = "json: input too large";
      return false;
    }
    size_t i = 0;
    if (!value(text, i, out, 0, err, cx)) return false;
    skip_ws(text, i);
    if (i != text.size()) {
      err = "json: trailing characters";
      return false;
    }
    return true;
  }

  const MiniJson* get(const std::string& key) const {
    if (type != Type::Object) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  bool is_true(const std::string& key) const {
    const MiniJson* v = get(key);
    return v && v->type == Type::Bool && v->boolean;
  }
  double num_or(const std::string& key, double dflt) const {
    const MiniJson* v = get(key);
    return (v && v->type == Type::Number) ? v->number : dflt;
  }
  std::string str_or(const std::string& key, const std::string& dflt) const {
    const MiniJson* v = get(key);
    return (v && v->type == Type::String) ? v->str : dflt;
  }

  // JSON string literal for request building (quotes included).
  static std::string quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
          if (c < 0x20) {
            static const char* hex = "0123456789abcdef";
            o += "\\u00";
            o += hex[c >> 4];
            o += hex[c & 15];
          } else {
            o += char(c);
          }
      }
    }
    o += "\"";
    return o;
  }

 private:
  struct Ctx {
    Limits lim;
    size_t values = 0;
    size_t mem = 0;  // bytes charged so far; never above lim.max_memory_bytes
  };

  // Charges `bytes` that are about to be allocated. False (with err set) if that would cross the ceiling.
  static bool charge(Ctx& cx, size_t bytes, std::string& err) {
    if (bytes > cx.lim.max_memory_bytes - cx.mem) {
      err = "json: memory ceiling";
      return false;
    }
    cx.mem += bytes;
    return true;
  }

  // Makes room for one more element, doubling under our own reserve so the charge is exact. The item cap is
  // checked before this, so size() < max_items and the new capacity is always larger than the old one.
  template <class V>
  static bool room_for_one(V& v, Ctx& cx, std::string& err) {
    if (v.size() < v.capacity()) return true;
    size_t ncap = v.capacity() ? v.capacity() * 2 : 4;
    if (ncap > cx.lim.max_items) ncap = cx.lim.max_items;
    if (ncap <= v.size()) ncap = v.size() + 1;
    const size_t es = sizeof(typename V::value_type);
    if (ncap > (cx.lim.max_memory_bytes - cx.mem) / es) {
      err = "json: memory ceiling";
      return false;
    }
    cx.mem += ncap * es;
    v.reserve(ncap);
    return true;
  }

  // Makes room for up to 4 more bytes in a string being decoded (one UTF-8 sequence).
  static bool room_for_bytes(std::string& o, Ctx& cx, std::string& err) {
    if (o.size() + 4 <= o.capacity()) return true;
    size_t ncap = o.capacity() * 2;
    if (ncap < 32) ncap = 32;
    if (ncap > cx.lim.max_string_bytes + 8) ncap = cx.lim.max_string_bytes + 8;
    if (ncap < o.size() + 4) ncap = o.size() + 4;
    if (!charge(cx, ncap + 1, err)) return false;
    o.reserve(ncap);
    return true;
  }

  // Duplicate keys in one finished object. Small objects compare pairwise; larger ones sort pointers to the
  // keys once, so the whole document costs n log n, never n^2. The pointer array is freed before anything else
  // allocates, so it is checked against the ceiling but not kept in the charge.
  static bool has_duplicate_key(const MiniJson& v, Ctx& cx, std::string& err) {
    const size_t n = v.obj.size();
    if (n <= 16) {
      for (size_t a = 0; a < n; ++a)
        for (size_t b = a + 1; b < n; ++b)
          if (v.obj[a].first == v.obj[b].first) {
            err = "json: duplicate key";
            return true;
          }
      return false;
    }
    if (n > (cx.lim.max_memory_bytes - cx.mem) / sizeof(const std::string*)) {
      err = "json: memory ceiling";
      return true;
    }
    std::vector<const std::string*> keys;
    keys.reserve(n);
    for (const auto& kv : v.obj) keys.push_back(&kv.first);
    std::sort(keys.begin(), keys.end(), [](const std::string* a, const std::string* b) { return *a < *b; });
    for (size_t k = 1; k < n; ++k)
      if (*keys[k - 1] == *keys[k]) {
        err = "json: duplicate key";
        return true;
      }
    return false;
  }

  static Limits clamp(const Limits& r) {
    Limits l = r;
    if (l.max_bytes == 0 || l.max_bytes > kMaxBytes) l.max_bytes = kMaxBytes;
    if (l.max_depth <= 0 || l.max_depth > kMaxDepth) l.max_depth = kMaxDepth;
    if (l.max_string_bytes == 0 || l.max_string_bytes > kMaxStringBytes) l.max_string_bytes = kMaxStringBytes;
    if (l.max_items == 0 || l.max_items > kMaxItems) l.max_items = kMaxItems;
    if (l.max_values == 0 || l.max_values > kMaxValues) l.max_values = kMaxValues;
    if (l.max_memory_bytes == 0 || l.max_memory_bytes > kMaxMemoryBytes) l.max_memory_bytes = kMaxMemoryBytes;
    return l;
  }

  static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }

  static bool lit(const std::string& s, size_t& i, const char* w) {
    size_t k = 0;
    while (w[k]) {
      if (i + k >= s.size() || s[i + k] != w[k]) return false;
      ++k;
    }
    i += k;
    return true;
  }

  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) {
      o += char(cp);
    } else if (cp < 0x800) {
      o += char(0xC0 | (cp >> 6));
      o += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      o += char(0xE0 | (cp >> 12));
      o += char(0x80 | ((cp >> 6) & 0x3F));
      o += char(0x80 | (cp & 0x3F));
    } else {
      o += char(0xF0 | (cp >> 18));
      o += char(0x80 | ((cp >> 12) & 0x3F));
      o += char(0x80 | ((cp >> 6) & 0x3F));
      o += char(0x80 | (cp & 0x3F));
    }
  }

  static bool hex4(const std::string& s, size_t& i, uint32_t& v) {
    if (i + 4 > s.size()) return false;
    v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s[i + size_t(k)];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else return false;
    }
    i += 4;
    return true;
  }

  static bool string_lit(const std::string& s, size_t& i, std::string& o, std::string& err, size_t max_len, Ctx& cx) {
    if (i >= s.size() || s[i] != '"') {
      err = "json: expected string";
      return false;
    }
    ++i;
    o.clear();
    while (i < s.size()) {
      if (o.size() > max_len) {
        err = "json: string too long";
        return false;
      }
      const char c = s[i++];
      if (c == '"') {
        if (o.size() > max_len) {
          err = "json: string too long";
          return false;
        }
        return true;
      }
      if (static_cast<unsigned char>(c) < 0x20) {  // RFC 8259: control characters must be escaped
        err = "json: raw control character in string";
        return false;
      }
      if (!room_for_bytes(o, cx, err)) return false;
      if (c != '\\') {
        o += c;
        continue;
      }
      if (i >= s.size()) break;
      const char e = s[i++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = 0;
          if (!hex4(s, i, cp)) {
            err = "json: bad \\u escape";
            return false;
          }
          if (cp >= 0xDC00 && cp < 0xE000) {
            err = "json: unpaired low surrogate";
            return false;
          }
          if (cp >= 0xD800 && cp < 0xDC00) {
            bool paired = false;
            if (i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
              size_t j = i + 2;
              uint32_t lo = 0;
              if (hex4(s, j, lo) && lo >= 0xDC00 && lo < 0xE000) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                i = j;
                paired = true;
              }
            }
            if (!paired) {
              err = "json: unpaired high surrogate";
              return false;
            }
          }
          put_utf8(o, cp);
          break;
        }
        default:
          err = "json: bad escape";
          return false;
      }
    }
    err = "json: unterminated string";
    return false;
  }

  static bool value(const std::string& s, size_t& i, MiniJson& v, int depth, std::string& err, Ctx& cx) {
    if (depth > cx.lim.max_depth) {
      err = "json: too deep";
      return false;
    }
    if (++cx.values > cx.lim.max_values) {
      err = "json: too many values";
      return false;
    }
    skip_ws(s, i);
    if (i >= s.size()) {
      err = "json: unexpected end";
      return false;
    }
    const char c = s[i];
    if (c == '{') {
      v.type = Type::Object;
      ++i;
      skip_ws(s, i);
      if (i < s.size() && s[i] == '}') {
        ++i;
        return true;
      }
      for (;;) {
        skip_ws(s, i);
        if (v.obj.size() >= cx.lim.max_items) {
          err = "json: too many keys in one object";
          return false;
        }
        std::string key;
        if (!string_lit(s, i, key, err, cx.lim.max_string_bytes, cx)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') {
          err = "json: expected ':'";
          return false;
        }
        ++i;
        MiniJson child;
        if (!value(s, i, child, depth + 1, err, cx)) return false;
        if (!room_for_one(v.obj, cx, err)) return false;
        v.obj.emplace_back(std::move(key), std::move(child));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') {
          ++i;
          return !(cx.lim.reject_duplicate_keys && has_duplicate_key(v, cx, err));
        }
        err = "json: expected ',' or '}'";
        return false;
      }
    }
    if (c == '[') {
      v.type = Type::Array;
      ++i;
      skip_ws(s, i);
      if (i < s.size() && s[i] == ']') {
        ++i;
        return true;
      }
      for (;;) {
        if (v.arr.size() >= cx.lim.max_items) {
          err = "json: too many items in one array";
          return false;
        }
        MiniJson child;
        if (!value(s, i, child, depth + 1, err, cx)) return false;
        if (!room_for_one(v.arr, cx, err)) return false;
        v.arr.push_back(std::move(child));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') {
          ++i;
          return true;
        }
        err = "json: expected ',' or ']'";
        return false;
      }
    }
    if (c == '"') {
      v.type = Type::String;
      return string_lit(s, i, v.str, err, cx.lim.max_string_bytes, cx);
    }
    if (lit(s, i, "true")) {
      v.type = Type::Bool;
      v.boolean = true;
      return true;
    }
    if (lit(s, i, "false")) {
      v.type = Type::Bool;
      v.boolean = false;
      return true;
    }
    if (lit(s, i, "null")) {
      v.type = Type::Null;
      return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      size_t j = i;
      if (s[j] == '-') ++j;
      while (j < s.size() && ((s[j] >= '0' && s[j] <= '9') || s[j] == '.' || s[j] == 'e' ||
                              s[j] == 'E' || s[j] == '+' || s[j] == '-'))
        ++j;
      const char* num = s.data() + i;  // read in place: a long number token must not cost a heap copy
      const size_t num_len = j - i;
      v.type = Type::Number;
      // RFC 8259 number grammar, parsed without the C locale: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?
      size_t k = 0;
      auto digits = [&]() { const size_t s0 = k; while (k < num_len && num[k] >= '0' && num[k] <= '9') ++k; return k - s0; };
      bool good = true;
      if (k < num_len && num[k] == '-') ++k;
      if (k < num_len && num[k] == '0') ++k;
      else good = digits() > 0;
      if (good && k < num_len && num[k] == '.') { ++k; good = digits() > 0; }
      if (good && k < num_len && (num[k] == 'e' || num[k] == 'E')) {
        ++k;
        if (k < num_len && (num[k] == '+' || num[k] == '-')) ++k;
        good = digits() > 0;
      }
      good = good && k == num_len;
      if (good) {
        const auto r = std::from_chars(num, num + num_len, v.number);
        good = r.ec == std::errc() && r.ptr == num + num_len;
      }
      if (!good) {
        err = "json: bad number";
        return false;
      }
      i = j;
      return true;
    }
    err = "json: unexpected character";
    return false;
  }
};

}  // namespace pchie
