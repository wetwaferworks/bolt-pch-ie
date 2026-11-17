/* Bounded JSON reader and writer: conformance, every limit at its edge, and writer round trips. MIT license. */
#include "pt.h"
#include "pj_base.h"
#include "pj_json.h"

static pj_limits L = {4096, 32, 64, 256, 1024, 2048};
static unsigned char scratch_mem[1 << 20];

static int parse_lim(const char* s, size_t n, const pj_limits* lim, pj_doc* d) {
  size_t need = pj_json_scratch_bytes(lim);
  if (need > sizeof(scratch_mem)) return -2;
  return pj_json_parse(s, n, lim, scratch_mem, need, d);
}
static int parse(const char* s, pj_doc* d) { return parse_lim(s, strlen(s), &L, d); }

static const char* const k_valid[] = {
    "{}", "[]", "0", "-0", "1.5e3", "1E-2", "-12.5E+2", "\"a\"", "true", "false", "null",
    "{\"a\":[1,2,{\"b\":null}]}", " \t\n\r[1] ", "\"\\u00e9\"", "\"\\ud83d\\ude00\"", "\"\xc3\xa9\"",
    "\"\xf0\x9f\x98\x80\"", "\"\\\"\\\\\\/\\b\\f\\n\\r\\t\"", "[[],[[]],{}]", "{\"\":0}", "\"\xef\xbf\xbf\"",
    "123456789012345678", "-123456789012345678", "0.000001", "1e308", "[0,-1,2.25,3e0]", "\"\xe2\x82\xac\"",
    "{\"a\":{\"b\":{\"c\":{}}}}", NULL};

static const char* const k_invalid[] = {
    "", " ", "{", "}", "[", "]", "[1,]", "{\"a\":1,}", "01", "-01", "1.", ".5", "+1", "1e", "1e+", "-", "NaN",
    "Infinity", "-Infinity", "tru", "nul", "fals", "'a'", "\"\\x\"", "\"\x01\"", "\"\\u0000\"", "\"\\ud800\"",
    "\"\\udc00\"", "\"\\ud800\\u0041\"", "\"\xc0\xaf\"", "\"\xed\xa0\x80\"", "\"\xf4\x90\x80\x80\"", "\"\xe2\x82\"",
    "\"\x80\"", "\"\xff\"", "[1] x", "[1][2]", "{\"a\":1,\"a\":2}", "{1:2}", "[1 2]", "{\"a\" 1}", "{\"a\":}",
    "\xef\xbb\xbf[]", "\"abc", "[\"a\",", "{\"a\":1 \"b\":2}", "1e400", "-1e400", "1e-400", "\"\\u12\"",
    "\"\\uZZZZ\"", "[,1]", "{,}", "\"\t\"", "TRUE", "nulll", "[1,,2]", "{\"a\":1}}", NULL};

static void conformance(void) {
  pj_doc d;
  int i;
  for (i = 0; k_valid[i]; ++i) {
    char what[128];
    snprintf(what, sizeof(what), "valid #%d accepted", i);
    CHECK(parse(k_valid[i], &d) == 0, what);
  }
  for (i = 0; k_invalid[i]; ++i) {
    char what[128];
    snprintf(what, sizeof(what), "invalid #%d refused", i);
    CHECK(parse(k_invalid[i], &d) != 0, what);
  }
  /* an embedded NUL byte in the raw text is refused, inside or outside a string */
  CHECK(parse_lim("[1]\0", 4, &L, &d) != 0, "raw NUL after a value refused");
  CHECK(parse_lim("\"a\0b\"", 5, &L, &d) != 0, "raw NUL inside a string refused");
}

static void values(void) {
  pj_doc d;
  int64_t v;
  CHECK(parse("123456789012345678", &d) == 0 && d.t[0].is_int && d.t[0].i == 123456789012345678LL, "18 digits exact");
  CHECK(parse("-5", &d) == 0 && d.t[0].is_int && d.t[0].i == -5, "negative int");
  CHECK(parse("9223372036854775807", &d) == 0 && !d.t[0].is_int && d.t[0].d > 9.2e18, "19 digits become a double");
  CHECK(parse("0.1", &d) == 0 && !d.t[0].is_int && d.t[0].d > 0.0999999 && d.t[0].d < 0.1000001, "0.1");
  CHECK(parse("123.456e-2", &d) == 0 && d.t[0].d > 1.23455 && d.t[0].d < 1.23457, "123.456e-2");
  CHECK(parse("-0.5", &d) == 0 && d.t[0].d == -0.5, "-0.5");
  CHECK(parse("0e-400", &d) == 0 && d.t[0].d == 0.0, "zero with a big negative exponent");
  CHECK(parse("{\"a\":7,\"b\":\"x\",\"c\":true}", &d) == 0, "object");
  CHECK(pj_get_int(&d, pj_get(&d, 0, "a"), 0, 10, &v) == 0 && v == 7, "pj_get_int in range");
  CHECK(pj_get_int(&d, pj_get(&d, 0, "a"), 8, 10, &v) != 0, "pj_get_int out of range");
  CHECK(pj_get_int(&d, pj_get(&d, 0, "b"), 0, 10, &v) != 0, "pj_get_int on a string");
  CHECK(pj_get(&d, 0, "zz") == -1, "missing key");
  CHECK(pj_str_eq(&d, pj_get(&d, 0, "b"), "x") && !pj_str_eq(&d, pj_get(&d, 0, "b"), "xy"), "pj_str_eq");
  CHECK(pj_is(&d, pj_get(&d, 0, "c"), PJ_BOOL) && d.t[pj_get(&d, 0, "c")].b == 1, "bool");
  CHECK(pj_get(&d, -1, "a") == -1 && !pj_is(&d, -1, PJ_OBJ), "negative index is safe");
  CHECK(parse("\"\\ud83d\\ude00\"", &d) == 0 && d.t[0].len == 4 && (unsigned char)d.t[0].s[0] == 0xF0, "surrogate pair to UTF-8");
  CHECK(parse("1.5", &d) == 0 && pj_get_int(&d, 0, 0, 10, &v) != 0, "pj_get_int refuses a fraction");
}

static void build_nested(char* b, int depth, size_t* n) {
  int i;
  size_t k = 0;
  for (i = 0; i < depth; ++i) b[k++] = '[';
  b[k++] = '1';
  for (i = 0; i < depth; ++i) b[k++] = ']';
  b[k] = 0;
  *n = k;
}

static void limits(void) {
  pj_doc d;
  static char b[300000];
  size_t n, i;
  pj_limits l;
  /* depth */
  l = L;
  l.max_depth = 4;
  build_nested(b, 4, &n);
  CHECK(parse_lim(b, n, &l, &d) == 0, "depth at the limit");
  build_nested(b, 5, &n);
  CHECK(parse_lim(b, n, &l, &d) != 0, "depth one over the limit");
  l.max_depth = 64;
  build_nested(b, 64, &n);
  CHECK(parse_lim(b, n, &l, &d) == 0, "depth 64");
  l.max_depth = 65;
  CHECK(parse_lim("1", 1, &l, &d) != 0, "max_depth above the hard maximum refused");
  /* depth bomb */
  l = L;
  l.max_bytes = 200000;
  l.max_values = 200000;
  for (i = 0; i < 100000; ++i) b[i] = '[';
  CHECK(parse_lim(b, 100000, &l, &d) != 0, "deep nesting bomb refused quickly");
  /* bytes */
  l = L;
  l.max_bytes = 10;
  CHECK(parse_lim("[1,2,3,44]", 10, &l, &d) == 0, "bytes at the limit");
  CHECK(parse_lim("[1,2,3,444]", 11, &l, &d) != 0, "bytes one over");
  /* items */
  l = L;
  l.max_items = 3;
  CHECK(parse_lim("[1,2,3]", 7, &l, &d) == 0, "items at the limit");
  CHECK(parse_lim("[1,2,3,4]", 9, &l, &d) != 0, "items one over");
  /* keys */
  l = L;
  l.max_keys = 2;
  CHECK(parse_lim("{\"a\":1,\"b\":2}", 13, &l, &d) == 0, "keys at the limit");
  CHECK(parse_lim("{\"a\":1,\"b\":2,\"c\":3}", 19, &l, &d) != 0, "keys one over");
  /* string bytes (decoded) */
  l = L;
  l.max_string = 4;
  CHECK(parse_lim("\"abcd\"", 6, &l, &d) == 0, "string at the limit");
  CHECK(parse_lim("\"abcde\"", 7, &l, &d) != 0, "string one over");
  CHECK(parse_lim("{\"abcde\":1}", 11, &l, &d) != 0, "key over the string limit");
  CHECK(parse_lim("\"\\u00e9\\u00e9\"", 14, &l, &d) == 0, "decoded length counts (2x2 bytes)");
  CHECK(parse_lim("\"\\u00e9\\u00e9a\"", 15, &l, &d) != 0, "decoded length one over");
  /* values: [1,2] is 3 values */
  l = L;
  l.max_values = 3;
  CHECK(parse_lim("[1,2]", 5, &l, &d) == 0, "values at the limit");
  CHECK(parse_lim("[1,2,3]", 7, &l, &d) != 0, "values one over");
  /* scratch too small */
  {
    unsigned char small[64];
    CHECK(pj_json_parse("[]", 2, &L, small, sizeof(small), &d) != 0, "scratch too small refused");
  }
  /* wide array bomb within bytes */
  l = L;
  l.max_bytes = 200000;
  l.max_values = 1000;
  l.max_items = 100000;
  b[0] = '[';
  n = 1;
  for (i = 0; i < 50000; ++i) {
    b[n++] = '0';
    b[n++] = ',';
  }
  b[n - 1] = ']';
  CHECK(parse_lim(b, n, &l, &d) != 0, "wide array over max_values refused");
}

static void writer(void) {
  char buf[256];
  pj_w w;
  pj_doc d;
  unsigned i, round;
  pj_w_init(&w, buf, sizeof(buf));
  pj_w_milli(&w, -1500);
  pj_w_raw(&w, " ", 1);
  pj_w_milli(&w, 0);
  pj_w_raw(&w, " ", 1);
  pj_w_milli(&w, -1);
  pj_w_raw(&w, " ", 1);
  pj_w_milli(&w, INT64_MIN);
  CHECK(!w.overflow && w.len == strlen("-1.500 0.000 -0.001 -9223372036854775.808") &&
            memcmp(buf, "-1.500 0.000 -0.001 -9223372036854775.808", w.len) == 0,
        "pj_w_milli is exact and locale free");
  pj_w_init(&w, buf, sizeof(buf));
  pj_w_i64(&w, INT64_MIN);
  CHECK(w.len == 20 && memcmp(buf, "-9223372036854775808", 20) == 0, "pj_w_i64 INT64_MIN");
  /* overflow never writes past cap */
  memset(buf, 'Z', sizeof(buf));
  pj_w_init(&w, buf, 8);
  pj_w_cstr(&w, "0123456789");
  CHECK(w.overflow && buf[8] == 'Z', "overflow stops at cap");
  /* escapes */
  pj_w_init(&w, buf, sizeof(buf));
  pj_w_str(&w, "a\"b\\c\n\x01\x1f", 8);
  CHECK(!w.overflow && parse_lim(buf, w.len, &L, &d) == 0 && d.t[0].len == 8 && memcmp(d.t[0].s, "a\"b\\c\n\x01\x1f", 8) == 0,
        "escapes round trip");
  /* random valid UTF-8 strings round trip */
  {
    uint32_t seed = 12345;
    int ok = 1;
    for (round = 0; round < 20000; ++round) {
      char src[128];
      size_t n = 0;
      unsigned len;
      seed = seed * 1103515245u + 12345u;
      len = (seed >> 16) % 20;
      for (i = 0; i < len; ++i) {
        uint32_t cp;
        seed = seed * 1103515245u + 12345u;
        cp = (seed >> 8) % 0x110000u;
        if (cp == 0 || (cp >= 0xD800 && cp < 0xE000)) cp = 'x';
        if (cp < 0x80) src[n++] = (char)cp;
        else if (cp < 0x800) {
          src[n++] = (char)(0xC0 | (cp >> 6));
          src[n++] = (char)(0x80 | (cp & 63));
        } else if (cp < 0x10000) {
          src[n++] = (char)(0xE0 | (cp >> 12));
          src[n++] = (char)(0x80 | ((cp >> 6) & 63));
          src[n++] = (char)(0x80 | (cp & 63));
        } else {
          src[n++] = (char)(0xF0 | (cp >> 18));
          src[n++] = (char)(0x80 | ((cp >> 12) & 63));
          src[n++] = (char)(0x80 | ((cp >> 6) & 63));
          src[n++] = (char)(0x80 | (cp & 63));
        }
      }
      pj_w_init(&w, buf, sizeof(buf));
      pj_w_str(&w, src, n);
      if (w.overflow || parse_lim(buf, w.len, &L, &d) != 0 || d.t[0].len != n || memcmp(d.t[0].s, src, n) != 0) ok = 0;
    }
    CHECK(ok, "20000 random UTF-8 strings round trip");
  }
}

static void random_bytes(void) {
  pj_doc d;
  uint32_t seed = 777;
  char b[64];
  unsigned k, i, accepted = 0;
  for (k = 0; k < 200000; ++k) {
    unsigned n;
    seed = seed * 1664525u + 1013904223u;
    n = (seed >> 24) % sizeof(b);
    for (i = 0; i < n; ++i) {
      seed = seed * 1664525u + 1013904223u;
      b[i] = "[]{}\",:01-.e tfnrul\\a"[(seed >> 16) % 22];
    }
    if (parse_lim(b, n, &L, &d) == 0) ++accepted;
  }
  CHECK(1, "200000 random inputs parsed without a crash");
  printf("random inputs accepted: %u of 200000\n", accepted);
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  conformance();
  values();
  limits();
  writer();
  random_bytes();
  return pt_done("peachy_json");
}
