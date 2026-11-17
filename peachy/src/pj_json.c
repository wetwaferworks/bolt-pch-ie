/* Peachy bounded JSON reader. MIT license. */
#include "pj_json.h"

#include <string.h>

#define PJ_HARD_DEPTH 64

typedef struct frame {
  int32_t tok;
  int32_t last;
  int32_t pending_key; /* objects: the key waiting for its value */
} frame;

size_t pj_json_scratch_bytes(const pj_limits* lim) {
  return (size_t)lim->max_values * sizeof(pj_tok) + (size_t)lim->max_bytes + (size_t)lim->max_values + 64u;
}

static int fail(pj_doc* d, const char* why) {
  size_t n = strlen(why);
  if (n >= sizeof(d->err)) n = sizeof(d->err) - 1;
  memcpy(d->err, why, n);
  d->err[n] = 0;
  return -1;
}

static int hexv(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

static int hex4(const char* s, size_t len, size_t i, uint32_t* v) {
  int k;
  if (i + 4 > len) return -1;
  *v = 0;
  for (k = 0; k < 4; ++k) {
    int h = hexv(s[i + (size_t)k]);
    if (h < 0) return -1;
    *v = (*v << 4) | (uint32_t)h;
  }
  return 0;
}

/* Length of one valid UTF-8 sequence at s[i], or 0. */
static size_t utf8_len(const unsigned char* s, size_t len, size_t i) {
  unsigned char c = s[i];
  if (c < 0x80) return 1;
  if (c >= 0xC2 && c <= 0xDF) return (i + 1 < len && (s[i + 1] & 0xC0) == 0x80) ? 2 : 0;
  if (c >= 0xE0 && c <= 0xEF) {
    unsigned char c1;
    if (i + 2 >= len) return 0;
    c1 = s[i + 1];
    if ((c1 & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80) return 0;
    if (c == 0xE0 && c1 < 0xA0) return 0; /* overlong */
    if (c == 0xED && c1 >= 0xA0) return 0; /* surrogate */
    return 3;
  }
  if (c >= 0xF0 && c <= 0xF4) {
    unsigned char c1;
    if (i + 3 >= len) return 0;
    c1 = s[i + 1];
    if ((c1 & 0xC0) != 0x80 || (s[i + 2] & 0xC0) != 0x80 || (s[i + 3] & 0xC0) != 0x80) return 0;
    if (c == 0xF0 && c1 < 0x90) return 0;
    if (c == 0xF4 && c1 >= 0x90) return 0;
    return 4;
  }
  return 0;
}

static size_t put_utf8(char* o, uint32_t cp) {
  if (cp < 0x80) {
    o[0] = (char)cp;
    return 1;
  }
  if (cp < 0x800) {
    o[0] = (char)(0xC0 | (cp >> 6));
    o[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  }
  if (cp < 0x10000) {
    o[0] = (char)(0xE0 | (cp >> 12));
    o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    o[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  }
  o[0] = (char)(0xF0 | (cp >> 18));
  o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
  o[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
  o[3] = (char)(0x80 | (cp & 0x3F));
  return 4;
}

typedef struct pstate {
  const char* s;
  size_t len;
  size_t i;
  const pj_limits* lim;
  pj_doc* d;
  uint32_t cap_tok;
  char* sbuf;
  size_t sbuf_cap;
  size_t sbuf_used;
} pstate;

static int new_tok(pstate* p, int type, int32_t* out) {
  pj_tok* t;
  if (p->d->n >= p->cap_tok) return fail(p->d, "too many values");
  t = &p->d->t[p->d->n];
  memset(t, 0, sizeof(*t));
  t->type = (uint8_t)type;
  t->first = -1;
  t->next = -1;
  *out = (int32_t)p->d->n++;
  return 0;
}

static int read_string(pstate* p, const char** out, uint32_t* out_len) {
  const unsigned char* u = (const unsigned char*)p->s;
  char* o = p->sbuf + p->sbuf_used;
  size_t start = p->sbuf_used, n = 0;
  if (p->i >= p->len || p->s[p->i] != '"') return fail(p->d, "expected string");
  ++p->i;
  for (;;) {
    unsigned char c;
    if (p->i >= p->len) return fail(p->d, "unterminated string");
    if (n > p->lim->max_string) return fail(p->d, "string too long");
    /* worst case one escape decodes to 4 bytes; keep room for it and the NUL */
    if (start + n + 5 > p->sbuf_cap) return fail(p->d, "string space exhausted");
    c = u[p->i];
    if (c == '"') {
      ++p->i;
      break;
    }
    if (c < 0x20) return fail(p->d, "raw control character in string");
    if (c == '\\') {
      char e;
      if (p->i + 1 >= p->len) return fail(p->d, "unterminated string");
      e = p->s[p->i + 1];
      p->i += 2;
      switch (e) {
        case '"': o[n++] = '"'; break;
        case '\\': o[n++] = '\\'; break;
        case '/': o[n++] = '/'; break;
        case 'b': o[n++] = '\b'; break;
        case 'f': o[n++] = '\f'; break;
        case 'n': o[n++] = '\n'; break;
        case 'r': o[n++] = '\r'; break;
        case 't': o[n++] = '\t'; break;
        case 'u': {
          uint32_t cp, lo;
          if (hex4(p->s, p->len, p->i, &cp)) return fail(p->d, "bad \\u escape");
          p->i += 4;
          if (cp == 0) return fail(p->d, "NUL in string");
          if (cp >= 0xDC00 && cp < 0xE000) return fail(p->d, "lone low surrogate");
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (p->i + 6 > p->len || p->s[p->i] != '\\' || p->s[p->i + 1] != 'u' || hex4(p->s, p->len, p->i + 2, &lo) ||
                lo < 0xDC00 || lo >= 0xE000)
              return fail(p->d, "lone high surrogate");
            p->i += 6;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          n += put_utf8(o + n, cp);
          break;
        }
        default:
          return fail(p->d, "bad escape");
      }
      continue;
    }
    {
      size_t k = utf8_len(u, p->len, p->i), j;
      if (!k) return fail(p->d, "invalid UTF-8");
      for (j = 0; j < k; ++j) o[n++] = p->s[p->i + j];
      p->i += k;
    }
  }
  if (n > p->lim->max_string) return fail(p->d, "string too long");
  o[n] = 0;
  p->sbuf_used = start + n + 1;
  *out = o;
  *out_len = (uint32_t)n;
  return 0;
}

static int read_number(pstate* p, pj_tok* t) {
  size_t i = p->i, int_start, int_digits = 0, frac_digits = 0;
  int neg = 0, exp_neg = 0, has_frac = 0, has_exp = 0;
  uint64_t mant = 0;
  int sig = 0, drop = 0;
  long exp10 = 0, ev = 0;
  if (p->s[i] == '-') {
    neg = 1;
    ++i;
  }
  int_start = i;
  if (i < p->len && p->s[i] == '0') {
    ++i;
    int_digits = 1;
  } else {
    while (i < p->len && p->s[i] >= '0' && p->s[i] <= '9') {
      ++i;
      ++int_digits;
    }
  }
  if (!int_digits) return fail(p->d, "bad number");
  {
    size_t k;
    for (k = int_start; k < i; ++k) {
      if (sig < 19) {
        mant = mant * 10u + (uint64_t)(p->s[k] - '0');
        if (mant) ++sig;
      } else {
        ++drop;
      }
    }
  }
  if (i < p->len && p->s[i] == '.') {
    ++i;
    has_frac = 1;
    while (i < p->len && p->s[i] >= '0' && p->s[i] <= '9') {
      if (sig < 19) {
        mant = mant * 10u + (uint64_t)(p->s[i] - '0');
        if (mant) ++sig;
        --exp10;
      }
      ++i;
      ++frac_digits;
    }
    if (!frac_digits) return fail(p->d, "bad number");
  }
  if (i < p->len && (p->s[i] == 'e' || p->s[i] == 'E')) {
    size_t ed = 0;
    ++i;
    has_exp = 1;
    if (i < p->len && (p->s[i] == '+' || p->s[i] == '-')) {
      exp_neg = p->s[i] == '-';
      ++i;
    }
    while (i < p->len && p->s[i] >= '0' && p->s[i] <= '9') {
      if (ev < 100000) ev = ev * 10 + (p->s[i] - '0');
      ++i;
      ++ed;
    }
    if (!ed) return fail(p->d, "bad number");
  }
  exp10 += drop + (exp_neg ? -ev : ev);
  t->type = PJ_NUM;
  if (!has_frac && !has_exp && int_digits <= 18) {
    t->is_int = 1;
    t->i = neg ? -(int64_t)mant : (int64_t)mant;
    t->d = (double)t->i;
  } else {
    double v = (double)mant;
    long e = exp10;
    /* A zero mantissa is zero whatever the exponent: skip the scaling loops, which would otherwise run up to
     * ~10^5 steps per number ("0e99999") and let one message cost seconds of CPU. */
    if (mant == 0) e = 0;
    if (e > 308 + 19 || e < -340) return fail(p->d, "number out of range");
    while (e > 0) {
      v *= 10.0;
      --e;
    }
    while (e < 0) {
      v /= 10.0;
      ++e;
    }
    if (v > 1.7976931348623157e308) return fail(p->d, "number out of range");
    t->d = neg ? -v : v;
    t->is_int = 0;
  }
  p->i = i;
  return 0;
}

static void skip_ws(pstate* p) {
  while (p->i < p->len) {
    char c = p->s[p->i];
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
    ++p->i;
  }
}

/* Link value tok into the open container on top of the stack. */
static int attach(pstate* p, frame* st, int depth, int32_t tok) {
  frame* f;
  pj_tok* c;
  if (depth == 0) return 0; /* root */
  f = &st[depth - 1];
  c = &p->d->t[f->tok];
  if (c->type == PJ_OBJ) {
    p->d->t[f->pending_key].first = tok;
    f->pending_key = -1;
    return 0;
  }
  if (c->len >= p->lim->max_items) return fail(p->d, "too many items in one array");
  if (f->last < 0) c->first = tok;
  else p->d->t[f->last].next = tok;
  f->last = tok;
  ++c->len;
  return 0;
}

int pj_json_parse(const char* text, size_t len, const pj_limits* lim, void* scratch, size_t scratch_cap, pj_doc* doc) {
  pstate p;
  frame st[PJ_HARD_DEPTH + 1];
  int depth = 0;
  size_t tok_bytes;
  enum { S_VALUE, S_KEY_OR_END, S_KEY, S_COLON, S_COMMA_OR_END, S_VALUE_OR_END, S_DONE } state = S_VALUE;
  memset(doc, 0, sizeof(*doc));
  if (!text || !lim || !scratch) return fail(doc, "bad arguments");
  if (lim->max_depth == 0 || lim->max_depth > PJ_HARD_DEPTH || lim->max_values == 0)
    return fail(doc, "bad limits");
  if (scratch_cap < pj_json_scratch_bytes(lim)) return fail(doc, "scratch too small");
  if (len > lim->max_bytes) return fail(doc, "message too large");
  tok_bytes = (size_t)lim->max_values * sizeof(pj_tok);
  memset(&p, 0, sizeof(p));
  p.s = text;
  p.len = len;
  p.lim = lim;
  p.d = doc;
  p.cap_tok = lim->max_values;
  doc->t = (pj_tok*)scratch;
  p.sbuf = (char*)scratch + tok_bytes;
  p.sbuf_cap = scratch_cap - tok_bytes;

  for (;;) {
    char c;
    skip_ws(&p);
    if (state == S_DONE) {
      if (p.i != p.len) return fail(doc, "trailing characters");
      return 0;
    }
    if (p.i >= p.len) return fail(doc, "unexpected end");
    c = p.s[p.i];
    switch (state) {
      case S_KEY_OR_END:
      case S_KEY: {
        frame* f = &st[depth - 1];
        pj_tok* o = &doc->t[f->tok];
        int32_t k, j;
        const char* ks;
        uint32_t kl;
        if (state == S_KEY_OR_END && c == '}') {
          ++p.i;
          --depth;
          state = depth ? S_COMMA_OR_END : S_DONE;
          break;
        }
        if (o->len >= lim->max_keys) return fail(doc, "too many keys in one object");
        if (read_string(&p, &ks, &kl)) return -1;
        for (j = o->first; j >= 0; j = doc->t[j].next)
          if (doc->t[j].len == kl && memcmp(doc->t[j].s, ks, kl) == 0) return fail(doc, "duplicate key");
        if (new_tok(&p, PJ_KEY, &k)) return -1;
        o = &doc->t[f->tok];
        doc->t[k].s = ks;
        doc->t[k].len = kl;
        if (f->last < 0) o->first = k;
        else doc->t[f->last].next = k;
        f->last = k;
        ++o->len;
        f->pending_key = k;
        state = S_COLON;
        break;
      }
      case S_COLON:
        if (c != ':') return fail(doc, "expected ':'");
        ++p.i;
        state = S_VALUE;
        break;
      case S_COMMA_OR_END: {
        pj_tok* top = &doc->t[st[depth - 1].tok];
        if (c == ',') {
          ++p.i;
          state = top->type == PJ_OBJ ? S_KEY : S_VALUE;
        } else if ((c == '}' && top->type == PJ_OBJ) || (c == ']' && top->type == PJ_ARR)) {
          ++p.i;
          --depth;
          state = depth ? S_COMMA_OR_END : S_DONE;
        } else {
          return fail(doc, "expected ',' or end");
        }
        break;
      }
      case S_VALUE_OR_END:
        if (c == ']') {
          ++p.i;
          --depth;
          state = depth ? S_COMMA_OR_END : S_DONE;
          break;
        }
        /* fall through */
      case S_VALUE: {
        int32_t t;
        if (c == '{' || c == '[') {
          if ((uint32_t)depth >= lim->max_depth) return fail(doc, "too deep");
          if (new_tok(&p, c == '{' ? PJ_OBJ : PJ_ARR, &t)) return -1;
          if (attach(&p, st, depth, t)) return -1;
          st[depth].tok = t;
          st[depth].last = -1;
          st[depth].pending_key = -1;
          ++depth;
          ++p.i;
          state = c == '{' ? S_KEY_OR_END : S_VALUE_OR_END;
          break;
        }
        if (new_tok(&p, PJ_NULL, &t)) return -1;
        if (c == '"') {
          const char* sv;
          uint32_t sl;
          if (read_string(&p, &sv, &sl)) return -1;
          doc->t[t].type = PJ_STR;
          doc->t[t].s = sv;
          doc->t[t].len = sl;
        } else if (c == '-' || (c >= '0' && c <= '9')) {
          if (read_number(&p, &doc->t[t])) return -1;
        } else if (p.len - p.i >= 4 && memcmp(p.s + p.i, "true", 4) == 0) {
          doc->t[t].type = PJ_BOOL;
          doc->t[t].b = 1;
          p.i += 4;
        } else if (p.len - p.i >= 5 && memcmp(p.s + p.i, "false", 5) == 0) {
          doc->t[t].type = PJ_BOOL;
          p.i += 5;
        } else if (p.len - p.i >= 4 && memcmp(p.s + p.i, "null", 4) == 0) {
          p.i += 4;
        } else {
          return fail(doc, "unexpected character");
        }
        if (attach(&p, st, depth, t)) return -1;
        state = depth ? S_COMMA_OR_END : S_DONE;
        break;
      }
      default:
        return fail(doc, "internal state");
    }
  }
}

int32_t pj_get(const pj_doc* d, int32_t obj, const char* key) {
  int32_t j;
  size_t kl;
  if (obj < 0 || (uint32_t)obj >= d->n || d->t[obj].type != PJ_OBJ) return -1;
  kl = strlen(key);
  for (j = d->t[obj].first; j >= 0; j = d->t[j].next)
    if (d->t[j].len == kl && memcmp(d->t[j].s, key, kl) == 0) return d->t[j].first;
  return -1;
}

int pj_is(const pj_doc* d, int32_t i, int type) { return i >= 0 && (uint32_t)i < d->n && d->t[i].type == type; }

int pj_str_eq(const pj_doc* d, int32_t i, const char* s) {
  size_t n = strlen(s);
  return pj_is(d, i, PJ_STR) && d->t[i].len == n && memcmp(d->t[i].s, s, n) == 0;
}

int pj_get_int(const pj_doc* d, int32_t i, int64_t lo, int64_t hi, int64_t* out) {
  if (!pj_is(d, i, PJ_NUM) || !d->t[i].is_int) return -1;
  if (d->t[i].i < lo || d->t[i].i > hi) return -1;
  *out = d->t[i].i;
  return 0;
}
