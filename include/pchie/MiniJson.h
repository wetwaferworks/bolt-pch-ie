// Bolt PCHIE - minimal JSON value + parser for the LithoKraft MCP client (pchie_lk_pilot).
// Concrete final. Serial, no mutex, no atomic. Reads only what the MCP wire needs: objects,
// arrays, strings (with \uXXXX -> UTF-8), numbers, true/false/null. Depth capped (fail closed).
// Not on the official chain; nothing here is hashed.
#pragma once

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

  static bool parse(const std::string& text, MiniJson& out, std::string& err) {
    size_t i = 0;
    out = MiniJson{};
    if (!value(text, i, out, 0, err)) return false;
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

  static bool string_lit(const std::string& s, size_t& i, std::string& o, std::string& err) {
    if (i >= s.size() || s[i] != '"') {
      err = "json: expected string";
      return false;
    }
    ++i;
    o.clear();
    while (i < s.size()) {
      const char c = s[i++];
      if (c == '"') return true;
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
          if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= s.size() && s[i] == '\\' && s[i + 1] == 'u') {
            size_t j = i + 2;
            uint32_t lo = 0;
            if (hex4(s, j, lo) && lo >= 0xDC00 && lo < 0xE000) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              i = j;
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

  static bool value(const std::string& s, size_t& i, MiniJson& v, int depth, std::string& err) {
    if (depth > kMaxDepth) {
      err = "json: too deep";
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
        std::string key;
        if (!string_lit(s, i, key, err)) return false;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != ':') {
          err = "json: expected ':'";
          return false;
        }
        ++i;
        MiniJson child;
        if (!value(s, i, child, depth + 1, err)) return false;
        v.obj.emplace_back(std::move(key), std::move(child));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') {
          ++i;
          return true;
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
        MiniJson child;
        if (!value(s, i, child, depth + 1, err)) return false;
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
      return string_lit(s, i, v.str, err);
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
      const std::string num = s.substr(i, j - i);
      char* end = nullptr;
      v.type = Type::Number;
      v.number = std::strtod(num.c_str(), &end);
      if (!end || *end != '\0' || num.empty() || num == "-") {
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
