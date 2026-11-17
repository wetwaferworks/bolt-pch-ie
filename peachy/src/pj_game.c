/* Peachy client of the game's MCP server (handshake era 2025-11-25). MIT license.
 * Serial, no reader thread, one JSON-RPC message per line. Every wait has a deadline. Lines longer than the
 * limit, a dead pipe, a protocol error or a timeout all fail the call; the session then holds. */
#include <string.h>

#include "pj_internal.h"
#include "pj_sha256.h"

#define PJ_GAME_MAX_SKIPPED 20000u

static uint64_t now(pj_game* g) { return g->clock ? g->clock(g->clock_ctx) : pj_mono_ns(); }

int pj_game_setup(pj_game* g, pj_arena* a, const pj_limits* jl, uint32_t max_line) {
  memset(g, 0, sizeof(*g));
  g->jl = *jl;
  g->jl.max_bytes = max_line;
  g->rcap = (size_t)max_line + 65536u;
  g->line_cap = (size_t)max_line + 1u;
  g->text_cap = (size_t)max_line + 1u;
  g->scratch_cap = pj_json_scratch_bytes(&g->jl);
  g->rbuf = (char*)pj_arena_alloc(a, g->rcap);
  g->line = (char*)pj_arena_alloc(a, g->line_cap);
  g->text = (char*)pj_arena_alloc(a, g->text_cap);
  g->scratch = pj_arena_alloc(a, g->scratch_cap);
  g->next_id = 1;
  return (g->rbuf && g->line && g->text && g->scratch) ? 0 : -1;
}

static int send_line(pj_game* g, const char* msg, size_t len, const char* tool) {
  uint8_t h[32];
  int rc;
  pj_sha256_once(msg, len, h);
  if (g->kind == PEACHY_GAME_SPAWN_STDIO) {
    rc = pj_proc_write(&g->proc, msg, len);
    if (rc == 0) rc = pj_proc_write(&g->proc, "\n", 1);
  } else {
    rc = g->io.send_line(g->io.ctx, msg, len) == 0 ? 0 : -1;
  }
  if (g->audit) g->audit(g->audit_ctx, 1, tool, h, rc == 0);
  return rc;
}

/* Next non-blank line into g->line. 0 ok, -1 fail, -2 timeout. */
static int read_line(pj_game* g, uint64_t deadline, size_t* out_len) {
  for (;;) {
    uint64_t t = now(g);
    uint32_t wait_ms;
    if (g->kind == PEACHY_GAME_SPAWN_STDIO) {
      char* nl = g->rlen ? (char*)memchr(g->rbuf, '\n', g->rlen) : NULL;
      if (nl) {
        size_t n = (size_t)(nl - g->rbuf);
        size_t keep = g->rlen - n - 1;
        if (n && g->rbuf[n - 1] == '\r') --n;
        if (n >= g->line_cap) return -1;
        memcpy(g->line, g->rbuf, n);
        g->line[n] = 0;
        memmove(g->rbuf, nl + 1, keep);
        g->rlen = keep;
        if (n == 0) continue;
        *out_len = n;
        return 0;
      }
      if (g->rlen >= g->line_cap) return -1; /* longer than the limit: protocol failure */
      if (t >= deadline) return -2;
      wait_ms = (uint32_t)((deadline - t) / 1000000u);
      if (wait_ms > 50) wait_ms = 50;
      {
        size_t got = 0;
        if (pj_proc_read(&g->proc, g->rbuf + g->rlen, g->rcap - g->rlen, &got, wait_ms) != 0) return -1;
        g->rlen += got;
      }
    } else {
      size_t got = 0;
      int32_t r;
      if (t >= deadline) return -2;
      wait_ms = (uint32_t)((deadline - t) / 1000000u);
      if (wait_ms > 50) wait_ms = 50;
      r = g->io.read_line(g->io.ctx, g->line, g->line_cap - 1, &got, wait_ms);
      if (r == 1) continue;
      if (r != 0 || got >= g->line_cap) return -1;
      g->line[got] = 0;
      if (got && g->line[got - 1] == '\r') g->line[--got] = 0;
      if (!got) continue;
      *out_len = got;
      return 0;
    }
  }
}

/* Sends a request and waits for the reply with the same id. On success *result indexes g->doc. */
static int request(pj_game* g, const char* method, const char* params, const char* tool, uint64_t deadline,
                   int32_t* result) {
  char msg[2048];
  pj_w w;
  int64_t id = g->next_id++;
  uint32_t skips = 0;
  pj_w_init(&w, msg, sizeof(msg));
  pj_w_cstr(&w, "{\"jsonrpc\":\"2.0\",\"id\":");
  pj_w_i64(&w, id);
  pj_w_cstr(&w, ",\"method\":");
  pj_w_cjstr(&w, method);
  pj_w_cstr(&w, ",\"params\":");
  pj_w_cstr(&w, params);
  pj_w_cstr(&w, "}");
  if (w.overflow || send_line(g, msg, w.len, tool)) return PJ_G_FAIL;
  for (;;) {
    size_t n = 0;
    int r = read_line(g, deadline, &n);
    int32_t mid, err, res;
    uint8_t h[32];
    if (r) return r == -2 ? PJ_G_TIMEOUT : PJ_G_FAIL;
    if (g->line[0] != '{') { /* boot log lines before the game moves its logs to stderr */
      if (++g->skipped_lines > PJ_GAME_MAX_SKIPPED) return PJ_G_FAIL;
      continue;
    }
    pj_sha256_once(g->line, n, h);
    if (pj_json_parse(g->line, n, &g->jl, g->scratch, g->scratch_cap, &g->doc) != 0) {
      if (g->audit) g->audit(g->audit_ctx, 0, tool, h, 0);
      return PJ_G_FAIL;
    }
    mid = pj_get(&g->doc, 0, "id");
    if (!pj_is(&g->doc, mid, PJ_NUM) || !g->doc.t[mid].is_int || g->doc.t[mid].i != id) {
      if (++skips > 1000) return PJ_G_FAIL; /* notifications or stale replies */
      continue;
    }
    if (g->audit) g->audit(g->audit_ctx, 0, tool, h, 1);
    err = pj_get(&g->doc, 0, "error");
    if (err >= 0) return PJ_G_FAIL;
    res = pj_get(&g->doc, 0, "result");
    if (!pj_is(&g->doc, res, PJ_OBJ)) return PJ_G_FAIL;
    *result = res;
    return PJ_G_OK;
  }
}

int pj_game_open(pj_game* g, const peachy_session_config* cfg, uint64_t deadline) {
  int32_t res, si, nm;
  g->kind = (int)cfg->game_kind;
  g->connected = 0;
  if (g->kind == PEACHY_GAME_SPAWN_STDIO) {
    if (pj_proc_spawn(&g->proc, cfg->game_argv, cfg->game_argc, cfg->game_workdir, cfg->game_stderr_log)) return PJ_G_FAIL;
  } else if (g->kind == PEACHY_GAME_CALLBACKS) {
    if (!cfg->game_io || cfg->game_io->struct_size < sizeof(peachy_game_io) || !cfg->game_io->send_line ||
        !cfg->game_io->read_line)
      return PJ_G_FAIL;
    g->io = *cfg->game_io;
  } else {
    return PJ_G_FAIL;
  }
  if (request(g, "initialize",
              "{\"protocolVersion\":\"2025-11-25\",\"capabilities\":{},\"clientInfo\":{\"name\":\"peachy\",\"version\":\"" PEACHY_VERSION_STR "\"}}",
              "initialize", deadline, &res) != PJ_G_OK) {
    pj_game_close(g);
    return PJ_G_FAIL;
  }
  si = pj_get(&g->doc, res, "serverInfo");
  nm = pj_get(&g->doc, si, "name");
  if (pj_is(&g->doc, nm, PJ_STR)) pj_copy_cstr(g->server_name, sizeof(g->server_name), g->doc.t[nm].s);
  {
    static const char note[] = "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}";
    if (send_line(g, note, sizeof(note) - 1, "initialized")) {
      pj_game_close(g);
      return PJ_G_FAIL;
    }
  }
  g->connected = 1;
  return PJ_G_OK;
}

static int contains(const char* s, size_t n, const char* needle) {
  size_t k = strlen(needle), i;
  if (k > n) return 0;
  for (i = 0; i + k <= n; ++i)
    if (memcmp(s + i, needle, k) == 0) return 1;
  return 0;
}

int pj_game_call(pj_game* g, const char* tool, const char* args_json, uint64_t deadline, int32_t* payload) {
  char params[1024];
  pj_w w;
  int attempt;
  *payload = -1;
  if (!g->connected) return PJ_G_FAIL;
  pj_w_init(&w, params, sizeof(params));
  pj_w_cstr(&w, "{\"name\":");
  pj_w_cjstr(&w, tool);
  pj_w_cstr(&w, ",\"arguments\":");
  pj_w_cstr(&w, args_json);
  pj_w_cstr(&w, "}");
  if (w.overflow) return PJ_G_FAIL;
  for (attempt = 0; attempt < 4; ++attempt) {
    int32_t res, content, part, ie;
    size_t tl = 0;
    int is_error, r = request(g, "tools/call", params, tool, deadline, &res);
    if (r != PJ_G_OK) {
      if (r == PJ_G_FAIL) g->connected = 0;
      return r;
    }
    ie = pj_get(&g->doc, res, "isError");
    is_error = pj_is(&g->doc, ie, PJ_BOOL) && g->doc.t[ie].b;
    content = pj_get(&g->doc, res, "content");
    if (!pj_is(&g->doc, content, PJ_ARR)) return PJ_G_FAIL;
    for (part = g->doc.t[content].first; part >= 0; part = g->doc.t[part].next) {
      int32_t tx = pj_get(&g->doc, part, "text");
      if (pj_is(&g->doc, tx, PJ_STR)) {
        if (tl + g->doc.t[tx].len >= g->text_cap) return PJ_G_FAIL;
        memcpy(g->text + tl, g->doc.t[tx].s, g->doc.t[tx].len);
        tl += g->doc.t[tx].len;
      }
    }
    g->text[tl] = 0;
    if (is_error) {
      if (contains(g->text, tl, "rate") && contains(g->text, tl, "limit")) {
        uint64_t t = now(g);
        if (t + 60000000ull >= deadline) return PJ_G_TIMEOUT;
        pj_sleep_ms(60); /* short bounded retry, like the pilot */
        continue;
      }
      if (contains(g->text, tl, "world_read") || contains(g->text, tl, "player_control")) return PJ_G_NOT_GRANTED;
      return PJ_G_TOOL_ERROR;
    }
    if (tl == 0) return PJ_G_OK;
    if (pj_json_parse(g->text, tl, &g->jl, g->scratch, g->scratch_cap, &g->doc) != 0) return PJ_G_FAIL;
    *payload = 0;
    return PJ_G_OK;
  }
  return PJ_G_TIMEOUT;
}

void pj_game_close(pj_game* g) {
  if (g->kind == PEACHY_GAME_SPAWN_STDIO) pj_proc_stop(&g->proc, 3000);
  else if (g->kind == PEACHY_GAME_CALLBACKS && g->io.close) g->io.close(g->io.ctx);
  g->connected = 0;
  g->kind = PEACHY_GAME_NONE;
}
