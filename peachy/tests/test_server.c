/* PMCP server, scripted: handshake, refusals, grants, rate limits, cut-off, holds, audit, own-thread mode.
 * No game and no planner here (see test_game.c and test_plan.c). MIT license. */
#include "pt.h"
#include "pj_audit.h"
#include "pj_json.h"
#include "pj_platform.h"

void pj_test_fail_audit_after(PeachySession* s, uint64_t record);
const char* pj_test_audit_path(PeachySession* s);

static const peachy_api* A;
static char g_dir[600], g_allow[700];
static const char* T_GROK = "grok-test-token-0123456789abcdef";
static const char* T_GEM = "gemini-test-token-0123456789abcd";
static const char* T_IN = "inproc-test-token-0123456789abcd";

static void write_allowlist(void) {
  char a[65], b[65], c[65], doc[2048];
  int n;
  pt_token_sha_hex(T_GROK, a);
  pt_token_sha_hex(T_GEM, b);
  pt_token_sha_hex(T_IN, c);
  n = snprintf(doc, sizeof(doc),
               "{\"version\":1,\"clients\":["
               "{\"name\":\"grok\",\"transports\":[\"stdio\"],\"token_sha256\":\"%s\",\"tools\":[\"observe\",\"plan\","
               "\"propose\",\"proposal_status\",\"audit_head\"],\"rate_per_sec\":256,\"burst\":512,\"max_mode\":\"advisory\"},"
               "{\"name\":\"gemini\",\"transports\":[\"stdio\"],\"token_sha256\":\"%s\",\"tools\":[],\"rate_per_sec\":2,"
               "\"burst\":3},"
               "{\"name\":\"inproc\",\"transports\":[\"in_process\"],\"token_sha256\":\"%s\"},"
               "{\"name\":\"notoken\",\"transports\":[\"stdio\"]}]}",
               a, b, c);
  snprintf(g_allow, sizeof(g_allow), "%s/allow.json", g_dir);
  pt_write_file(g_allow, doc, (size_t)n);
}

static void base_cfg(peachy_session_config* c, pt_client* cl, const char* client, const char* token) {
  memset(c, 0, sizeof(*c));
  c->struct_size = sizeof(*c);
  c->write = pt_write;
  c->write_ctx = cl;
  c->allowlist_path = g_allow;
  c->audit_dir = g_dir;
  c->session_name = "test";
  c->client_name = client;
  c->client_token = (const uint8_t*)token;
  c->client_token_len = token ? strlen(token) : 0;
  c->client_transport = PEACHY_TRANSPORT_STDIO;
}

static peachy_status open_s(peachy_session_config* c, PeachySession** s, char* err) {
  return A->session_open(c, s, err, 128);
}

/* Feeds one line and pumps. Returns the newest output (or "" if none was added). */
static const char* rpc(PeachySession* s, pt_client* cl, const char* line) {
  int before = cl->n;
  uint32_t done = 0;
  size_t acc = 0;
  A->feed(s, line, strlen(line), &acc);
  A->feed(s, "\n", 1, &acc);
  A->pump(s, 1000000000ull, &done);
  return cl->n > before || (cl->n == PT_MAX_OUT && before == PT_MAX_OUT) ? pt_last(cl) : "";
}

static int has(const char* hay, const char* needle) { return strstr(hay, needle) != NULL; }

static void all_outputs_are_json(pt_client* cl, const char* what) {
  static unsigned char scratch[4 << 20];
  pj_limits l = {1u << 20, 32, 256, 4096, 1u << 20, 65536};
  pj_doc d;
  int k, ok = 1;
  for (k = 0; k < cl->n; ++k)
    if (pj_json_parse(cl->msgs[k], cl->lens[k], &l, scratch, pj_json_scratch_bytes(&l), &d) != 0) ok = 0;
  CHECK(ok, what);
}

static void open_refusals(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128];
  peachy_limits L;
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "nobody", T_GROK);
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && !s && has(err, "allowlist"), "unknown client refused");
  base_cfg(&c, &cl, "grok", "grok-test-token-0123456789abcdeX");
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && has(err, "token") && !has(err, "grok-test"), "wrong token refused");
  base_cfg(&c, &cl, "grok", "short");
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED, "short token refused");
  base_cfg(&c, &cl, "grok", NULL);
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED, "no token refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.client_transport = PEACHY_TRANSPORT_IN_PROCESS;
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && has(err, "transport"), "wrong transport refused");
  base_cfg(&c, &cl, "notoken", "anything-at-all-0123456789");
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED, "client without a token hash refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.requested_mode = PEACHY_MODE_ACTUATE;
  CHECK(open_s(&c, &s, err) == PEACHY_E_UNSUPPORTED, "actuate refused");
  base_cfg(&c, &cl, "gemini", T_GEM);
  c.requested_mode = PEACHY_MODE_ADVISORY;
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && has(err, "max_mode"), "mode above max_mode refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.session_name = "Bad Name";
  CHECK(open_s(&c, &s, err) == PEACHY_E_INVALID_ARG, "bad session name refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  memset(&L, 0, sizeof(L));
  L.struct_size = sizeof(L);
  L.max_depth = 65;
  c.limits = &L;
  CHECK(open_s(&c, &s, err) == PEACHY_E_INVALID_ARG, "limit above the hard maximum refused");
  L.max_depth = 0;
  L.struct_size = 8;
  CHECK(open_s(&c, &s, err) == PEACHY_E_INVALID_ARG, "short limits struct refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.struct_size = 12;
  CHECK(open_s(&c, &s, err) == PEACHY_E_INVALID_ARG, "short config struct refused");
  CHECK(A->session_open(NULL, &s, err, sizeof(err)) == PEACHY_E_INVALID_ARG, "NULL config refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  CHECK(A->session_open(&c, NULL, err, sizeof(err)) == PEACHY_E_INVALID_ARG, "NULL out refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.allowlist_path = "/no/such/allowlist.json";
  CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED, "missing allowlist refused");
  {
    char bad[700];
    snprintf(bad, sizeof(bad), "%s/bad-allow.json", g_dir);
    {
      const char* t1 = "{\"version\":1,\"clients\":[],\"extra\":1}";
      pt_write_file(bad, t1, strlen(t1));
    }
    base_cfg(&c, &cl, "grok", T_GROK);
    c.allowlist_path = bad;
    CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && has(err, "unknown field"), "allowlist unknown field refused");
    {
      const char* t2 = "{\"version\":1,\"clients\":[{\"name\":\"x\",\"transports\":[\"stdio\"],\"tools\":[\"hold\"]}]}";
      pt_write_file(bad, t2, strlen(t2));
    }
    CHECK(open_s(&c, &s, err) == PEACHY_E_DENIED && has(err, "cannot be granted"), "allowlist granting an always tool refused");
  }
  base_cfg(&c, &cl, "grok", T_GROK);
  c.audit_dir = "/no/such/dir/for/peachy";
  CHECK(open_s(&c, &s, err) == PEACHY_E_IO, "missing audit dir refused");
  base_cfg(&c, &cl, "grok", T_GROK);
  c.planner_kind = 9;
  CHECK(open_s(&c, &s, err) == PEACHY_E_INVALID_ARG, "unknown planner kind refused");
  base_cfg(&c, &cl, "inproc", T_IN);
  c.client_transport = PEACHY_TRANSPORT_IN_PROCESS;
  CHECK(open_s(&c, &s, err) == PEACHY_OK && s, "in-process client accepted");
  A->session_close(s);
  pt_client_clear(&cl);
}

static void protocol(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128];
  const char* r;
  peachy_status_info st;
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "grok", T_GROK);
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "grok opens");
  if (!s) return;
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
  CHECK(has(r, "\"id\":1,\"result\":{}"), "ping before initialize");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}");
  CHECK(has(r, "-32600") && has(r, "not initialized"), "tools/list before initialize refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"1999-01-01\"}}");
  CHECK(has(r, "-32022") && has(r, "2025-11-25"), "unknown protocol version refused with the supported list");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"initialize\",\"params\":{}}");
  CHECK(has(r, "-32602"), "initialize without protocolVersion refused");
  {
    int before = cl.n;
    rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
    CHECK(cl.n == before, "initialize as a notification gets no reply");
  }
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\","
                  "\"capabilities\":{},\"clientInfo\":{\"name\":\"x\",\"version\":\"1\"}}}");
  CHECK(has(r, "\"protocolVersion\":\"2025-06-18\"") && has(r, "\"name\":\"peachy\"") && has(r, "PMCP"), "initialize ok");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
  CHECK(has(r, "-32600") && has(r, "already"), "second initialize refused");
  {
    int before = cl.n;
    rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}");
    CHECK(cl.n == before, "initialized notification gets no reply");
  }
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/list\"}");
  CHECK(has(r, "\"peachy_info\"") && has(r, "\"status\"") && has(r, "\"hold\"") && has(r, "\"observe\"") &&
            has(r, "\"plan\"") && has(r, "\"propose\"") && has(r, "\"proposal_status\"") && has(r, "\"audit_head\""),
        "tools/list shows granted tools");
  CHECK(has(r, "\"additionalProperties\":false"), "schemas are closed");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":8,\"method\":\"resources/list\"}");
  CHECK(has(r, "-32601"), "unknown method");
  r = rpc(s, &cl, "{");
  CHECK(has(r, "-32700") && has(r, "\"id\":null"), "parse error");
  r = rpc(s, &cl, "[{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}]");
  CHECK(has(r, "-32600"), "batch refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1.5,\"method\":\"ping\"}");
  CHECK(has(r, "-32600") && has(r, "id"), "float id refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"method\":\"ping\"}");
  CHECK(has(r, "\"id\":\"abc\",\"result\":{}"), "string id echoed");
  r = rpc(s, &cl, "{\"jsonrpc\":\"1.0\",\"id\":10,\"method\":\"ping\"}");
  CHECK(has(r, "-32600"), "wrong jsonrpc refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/list\",\"params\":[]}");
  CHECK(has(r, "-32602"), "array params refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{}}");
  CHECK(has(r, "-32602"), "tools/call without a name refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"tools/call\",\"params\":{\"name\":\"player_action\"}}");
  CHECK(has(r, "\"isError\":true") && has(r, "unknown tool"), "unknown tool is a tool error");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":14,\"method\":\"tools/call\",\"params\":{\"name\":\"peachy_info\",\"arguments\":{\"x\":1}}}");
  CHECK(has(r, "-32602"), "arguments to a no-argument tool refused");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":15,\"method\":\"tools/call\",\"params\":{\"name\":\"peachy_info\",\"arguments\":{}}}");
  CHECK(has(r, "\"isError\":false") && has(r, "structuredContent") && has(r, "\"actuate\":false") &&
            has(r, "\"proposals_only\":true"),
        "peachy_info");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":16,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
  CHECK(has(r, "\"held\":false") && has(r, "\"granted\":["), "status");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":17,\"method\":\"tools/call\",\"params\":{\"name\":\"audit_head\"}}");
  CHECK(has(r, "\"length\":") && has(r, "\"head\":\""), "audit_head");
  {
    int before = cl.n;
    rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":99,\"result\":{}}");
    CHECK(cl.n == before, "a response from the client is ignored");
    rpc(s, &cl, "");
    CHECK(cl.n == before, "blank line ignored");
  }
  {
    size_t acc;
    uint32_t done;
    const char* part1 = "{\"jsonrpc\":\"2.0\",\"id\":20,";
    const char* part2 = "\"method\":\"ping\"}\r\n{\"jsonrpc\":\"2.0\",\"id\":21,\"method\":\"ping\"}\n";
    int before = cl.n;
    A->feed(s, part1, strlen(part1), &acc);
    A->pump(s, 1000000000ull, &done);
    CHECK(cl.n == before && done == 0, "half a message waits");
    A->feed(s, part2, strlen(part2), &acc);
    A->pump(s, 1000000000ull, &done);
    CHECK(done == 2 && cl.n == before + 2 && has(cl.msgs[cl.n - 2], "\"id\":20") && has(cl.msgs[cl.n - 1], "\"id\":21"),
          "split message with CRLF, then a second message");
  }
  all_outputs_are_json(&cl, "every reply is valid JSON");
  st.struct_size = sizeof(st);
  CHECK(A->status(s, &st) == PEACHY_OK && !st.held && st.messages_in > 20 && st.audit_length > 20, "status counters");
  {
    char path[600];
    pj_audit_report rep;
    snprintf(path, sizeof(path), "%s", pj_test_audit_path(s));
    A->session_close(s);
    CHECK(pj_audit_verify(path, &rep) == 0 && rep.records > st.audit_length, "audit log verifies after close");
  }
  pt_client_clear(&cl);
}

static void grants_and_rate(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128], rep[256];
  const char* r;
  pt_clock ck = {1000000000ull, 0};
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "gemini", T_GEM);
  c.clock = pt_clock_fn;
  c.clock_ctx = &ck;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "gemini opens");
  if (!s) return;
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/list\"}");
  CHECK(has(r, "\"peachy_info\"") && has(r, "\"status\"") && has(r, "\"hold\"") && !has(r, "\"observe\"") &&
            !has(r, "\"propose\""),
        "ungranted client sees only the always tools");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":{}}}");
  CHECK(has(r, "\"isError\":true") && has(r, "not granted"), "ungranted tool refused");
  CHECK(A->console(s, "grant observe propose", rep, sizeof(rep)) == PEACHY_OK, "console grant");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/list\"}");
  CHECK(has(r, "\"observe\"") && has(r, "\"propose\""), "granted tools appear");
  CHECK(A->console(s, "revoke propose", rep, sizeof(rep)) == PEACHY_OK, "console revoke");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/list\"}");
  CHECK(has(r, "\"observe\"") && !has(r, "\"propose\""), "revoked tool gone");
  CHECK(A->console(s, "grant peachy_info", rep, sizeof(rep)) == PEACHY_E_INVALID_ARG, "always tools are not grantable");
  CHECK(A->console(s, "grant player_action", rep, sizeof(rep)) == PEACHY_E_INVALID_ARG, "game tools are not grantable");
  CHECK(A->console(s, "mode advisory", rep, sizeof(rep)) == PEACHY_E_DENIED, "advisory above max_mode refused");
  CHECK(A->console(s, "mode actuate", rep, sizeof(rep)) == PEACHY_E_UNSUPPORTED, "actuate refused at the console");
  CHECK(A->console(s, "status", rep, sizeof(rep)) == PEACHY_OK && has(rep, "running"), "console status");
  CHECK(A->console(s, "bogus", rep, sizeof(rep)) == PEACHY_E_INVALID_ARG, "unknown console command");
  CHECK(A->console(s, "grant\tobserve", rep, sizeof(rep)) == PEACHY_E_INVALID_ARG, "control characters refused");
  {
    char longline[400];
    memset(longline, 'a', sizeof(longline) - 1);
    longline[sizeof(longline) - 1] = 0;
    CHECK(A->console(s, longline, rep, sizeof(rep)) == PEACHY_E_LIMIT, "long console line refused");
  }
  /* rate: the bucket started at burst 3; the three calls above (3,4 is list: not counted, 5 list) used 1 */
  {
    int ok_calls = 0, limited = 0, k;
    for (k = 0; k < 6; ++k) {
      r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":30,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
      if (has(r, "rate limited")) ++limited;
      else if (has(r, "\"isError\":false")) ++ok_calls;
    }
    CHECK(ok_calls == 3 && limited == 3, "burst of 3 then rate limited");
    ck.t += 1000000000ull; /* one second: 2 more tokens */
    ok_calls = limited = 0;
    for (k = 0; k < 3; ++k) {
      r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":31,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
      if (has(r, "rate limited")) ++limited;
      else ++ok_calls;
    }
    CHECK(ok_calls == 2 && limited == 1, "refill at rate_per_sec");
  }
  /* hold is the safe state, so it is never rate limited and never spends the budget (review 2026-11-16) */
  {
    peachy_status_info st;
    int ok_calls = 0, k;
    r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":32,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
    CHECK(has(r, "rate limited"), "budget is spent before the hold");
    for (k = 0; k < 4; ++k) {
      r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":33,\"method\":\"tools/call\",\"params\":{\"name\":\"hold\"}}");
      CHECK(!has(r, "rate limited") && has(r, "\"isError\":false"), "hold works with the budget spent");
    }
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    A->status(s, &st);
    CHECK(st.held, "the client's hold took effect");
    ck.t += 1000000000ull;
    for (k = 0; k < 3; ++k) {
      r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":34,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
      if (!has(r, "rate limited")) ++ok_calls;
    }
    CHECK(ok_calls == 2, "the holds cost nothing from the bucket");
  }
  A->session_close(s);
  pt_client_clear(&cl);
}

static void cut_off_and_oversize(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128], rep[256];
  const char* r;
  peachy_limits L;
  peachy_status_info st;
  static char big[5000];
  size_t acc;
  memset(&cl, 0, sizeof(cl));
  memset(&L, 0, sizeof(L));
  L.struct_size = sizeof(L);
  L.bad_message_limit = 3;
  L.max_message_bytes = 1024;
  base_cfg(&c, &cl, "grok", T_GROK);
  c.limits = &L;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open with small limits");
  if (!s) return;
  memset(big, 'x', sizeof(big));
  A->feed(s, big, sizeof(big), &acc);
  CHECK(acc == sizeof(big), "an oversized message is swallowed");
  A->feed(s, "\n", 1, &acc);
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
  CHECK(has(r, "\"id\":1,\"result\""), "the next message after an oversized one still works");
  st.struct_size = sizeof(st);
  A->status(s, &st);
  CHECK(st.bad_messages == 1, "oversized message counted as bad");
  rpc(s, &cl, "nope");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":[],\"method\":\"ping\"}");
  A->status(s, &st);
  CHECK(st.client_cut_off == 1 && st.held == 1 && has(st.hold_reason, "too many bad"), "third bad message cuts off");
  CHECK(A->feed(s, "x\n", 2, &acc) == PEACHY_E_DENIED, "feed after cut-off denied");
  CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_E_HELD && has(rep, "cut off"), "resume after cut-off refused");
  A->session_close(s);
  pt_client_clear(&cl);
  /* backpressure: the input buffer holds at most one message limit */
  L.bad_message_limit = 0;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open again");
  if (!s) return;
  {
    static char many[4096];
    size_t n = 0, k;
    uint32_t done;
    int got;
    const char* ping = "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n";
    for (k = 0; k < 60; ++k) {
      memcpy(many + n, ping, strlen(ping));
      n += strlen(ping);
    }
    CHECK(A->feed(s, many, n, &acc) == PEACHY_E_LIMIT && acc < n && acc > 0, "full input buffer pushes back");
    got = cl.n;
    A->pump(s, 1000000000ull, &done);
    CHECK(done > 0 && cl.n > got, "pump drains");
    {
      size_t off = acc;
      int rounds = 0;
      while (off < n && rounds++ < 100) {
        A->feed(s, many + off, n - off, &acc);
        off += acc;
        A->pump(s, 1000000000ull, &done);
      }
      CHECK(off == n, "the rest goes in after pumps");
    }
    CHECK(cl.n == 60, "all 60 answered");
  }
  A->session_close(s);
  pt_client_clear(&cl);
}

/* Every inbound message pays a generous bucket (512/s, burst 8192), so pings, notifications and stray responses
 * can't flood the audit log; going over counts as a bad message and ends in a cut-off (review 2026-11-16). */
static void message_flood(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128];
  const char* r;
  peachy_limits L;
  peachy_status_info st;
  pt_clock ck = {1000000000ull, 0};
  int k, ok = 0, refused = 0;
  memset(&cl, 0, sizeof(cl));
  memset(&L, 0, sizeof(L));
  L.struct_size = sizeof(L);
  L.bad_message_limit = 16;
  base_cfg(&c, &cl, "grok", T_GROK);
  c.limits = &L;
  c.clock = pt_clock_fn;
  c.clock_ctx = &ck;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "flood session opens");
  if (!s) return;
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
  /* the burst: 8192 messages in all, a mix of pings, notifications and stray responses, with the clock stopped */
  for (k = 1; k < 8192; ++k) {
    if (k % 3 == 0) r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}");
    else if (k % 3 == 1) r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":9,\"result\":{}}");
    else r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}");
    if (has(r, "message rate exceeded")) ++refused;
  }
  memset(&st, 0, sizeof(st));
  st.struct_size = sizeof(st);
  A->status(s, &st);
  CHECK(refused == 0 && st.bad_messages == 0 && !st.held, "the whole burst is accepted");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}");
  CHECK(has(r, "\"id\":3,\"error\"") && has(r, "message rate exceeded"), "one past the burst is refused");
  /* one second refills 512 */
  ck.t += 1000000000ull;
  for (k = 0; k < 513; ++k) {
    r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"ping\"}");
    if (has(r, "\"id\":4,\"result\"")) ++ok;
  }
  CHECK(ok == 512 && has(r, "message rate exceeded"), "refill at 512 per second, then refused again");
  /* two refusals so far; notifications over the rate count too, and the 16th bad message cuts the client off */
  for (k = 0; k < 13; ++k) rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\"}");
  A->status(s, &st);
  CHECK(st.bad_messages == 15 && !st.client_cut_off, "15 bad messages, not yet cut off");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{}}");
  A->status(s, &st);
  CHECK(st.client_cut_off == 1 && st.held == 1, "the flood ends in a cut-off and a hold");
  /* at most one record in and one out per message, plus the open, hold and cut-off records */
  CHECK(st.audit_length <= 2u * (8192 + 1 + 513 + 14) + 64, "audit records stay bounded by the message budget");
  printf("flood: %llu audit records for %d messages\n", (unsigned long long)st.audit_length, 8192 + 1 + 513 + 14);
  A->session_close(s);
  pt_client_clear(&cl);
}

static void holds(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128], rep[256];
  const char* r;
  peachy_status_info st;
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "grok", T_GROK);
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open for holds");
  if (!s) return;
  st.struct_size = sizeof(st);
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"hold\"}}");
  CHECK(has(r, "\"held\":true") && has(r, "MCP client asked"), "client hold");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"observe\"}}");
  CHECK(has(r, "\"isError\":true") && has(r, "held"), "observe refused while held");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
                  "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\"}]}}}");
  CHECK(has(r, "\"verdict\":\"rejected\"") && has(r, "\"check\":6") && has(r, "held"), "propose rejected while held");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"status\"}}");
  CHECK(has(r, "\"held\":true"), "status works while held");
  CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_OK && has(rep, "resumed"), "console resume");
  A->hold(s);
  A->pump(s, 1000, NULL);
  A->status(s, &st);
  CHECK(st.held && has(st.hold_reason, "host asked"), "host hold at the next phase boundary");
  A->console(s, "resume", rep, sizeof(rep));
  CHECK(A->console(s, "hold", rep, sizeof(rep)) == PEACHY_OK, "console hold");
  A->status(s, &st);
  CHECK(st.held && has(st.hold_reason, "console"), "console hold reason");
  A->console(s, "resume", rep, sizeof(rep));
  /* a log edited on disk while held blocks resume (review 2026-11-16). The edit can only be staged on POSIX: on
   * Windows the session's deny-write share mode stops another writer while the log is open, so check that instead. */
#if !defined(_WIN32)
  {
    char ap[600];
    size_t n = 0;
    char* txt;
    CHECK(A->console(s, "hold", rep, sizeof(rep)) == PEACHY_OK, "hold before tamper");
    snprintf(ap, sizeof(ap), "%s", pj_test_audit_path(s));
    txt = pt_read_file(ap, &n);
    if (txt && n > 2) {
      char* last = txt + n - 2;
      while (last > txt && *last != '\n') --last; /* drop the newest record */
      pt_write_file(ap, txt, (size_t)(last - txt) + 1);
      CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_E_HELD && has(rep, "audit log on disk"),
            "resume refused when the audit log on disk was cut short");
      pt_write_file(ap, txt, n); /* put it back so the rest of the test runs */
    }
    free(txt);
    CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_OK, "resume once the log matches again");
  }
#else
  {
    /* Windows: the open log refuses a second writer, and resume can still re-read it */
    FILE* w2 = fopen(pj_test_audit_path(s), "r+b");
    CHECK(w2 == NULL, "a second writer can't open the live audit log");
    if (w2) fclose(w2);
    CHECK(A->console(s, "hold", rep, sizeof(rep)) == PEACHY_OK, "hold before the re-read");
    CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_OK, "resume re-reads the log while it is open");
  }
#endif
  /* audit write failure holds and blocks resume */
  A->status(s, &st);
  pj_test_fail_audit_after(s, st.audit_length + 2);
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":6,\"method\":\"ping\"}");
  A->status(s, &st);
  CHECK(st.held && has(st.hold_reason, "audit"), "audit write failure holds");
  CHECK(A->console(s, "resume", rep, sizeof(rep)) == PEACHY_E_HELD && has(rep, "audit"), "no resume without the audit log");
  A->session_close(s);
  pt_client_clear(&cl);
  /* output failure holds */
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open for output failure");
  if (!s) return;
  cl.fail_writes = 1;
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
  A->status(s, &st);
  CHECK(st.held && has(st.hold_reason, "output"), "undeliverable output holds");
  cl.fail_writes = 0;
  A->session_close(s);
  pt_client_clear(&cl);
  /* deadline: a clock that jumps a second per read misses a 1 s message deadline */
  {
    pt_clock ck = {1, 1000000000ull};
    peachy_limits L;
    memset(&L, 0, sizeof(L));
    L.struct_size = sizeof(L);
    L.max_message_ms = 1000;
    c.limits = &L;
    c.clock = pt_clock_fn;
    c.clock_ctx = &ck;
    CHECK(open_s(&c, &s, err) == PEACHY_OK, "open with a jumping clock");
    if (!s) return;
    rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}");
    A->status(s, &st);
    CHECK(st.held && has(st.hold_reason, "deadline"), "missed deadline holds");
    A->session_close(s);
    pt_client_clear(&cl);
  }
}

static void proposals_without_game(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128];
  const char* r;
  peachy_limits L;
  int k;
  memset(&cl, 0, sizeof(cl));
  memset(&L, 0, sizeof(L));
  L.struct_size = sizeof(L);
  L.action_budget = 300;
  L.rate_per_sec = 256;
  L.rate_burst = 512;
  base_cfg(&c, &cl, "grok", T_GROK);
  c.limits = &L;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open for proposals");
  if (!s) return;
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\"}}");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
                  "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"steps\":2}]}}}");
  CHECK(has(r, "\"id\":1,\"verdict\":\"rejected\",\"check\":4") && has(r, "observe first"), "no observation: check 4");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"proposal_status\",\"arguments\":{\"id\":1}}}");
  CHECK(has(r, "\"state\":\"rejected\"") && has(r, "\"check\":4"), "proposal_status rejected");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"proposal_status\",\"arguments\":{\"id\":999}}}");
  CHECK(has(r, "\"state\":\"unknown\""), "unknown proposal id");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"proposal_status\",\"arguments\":{\"id\":0}}}");
  CHECK(has(r, "-32602"), "proposal id 0 refused");
  {
    static const char* const bad_args[] = {
        "{}",
        "{\"actions\":[]}",
        "{\"actions\":[{\"kind\":\"move\",\"dir\":\"up\"}]}",
        "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"steps\":5}]}",
        "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"steps\":1.5}]}",
        "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"speed\":1}]}",
        "{\"actions\":[{\"kind\":\"fly\"}]}",
        "{\"actions\":[{\"kind\":\"look\",\"turn\":\"up\"}]}",
        "{\"actions\":[{\"kind\":\"action\",\"which\":\"attack\"}]}",
        "{\"actions\":[1]}",
        "{\"actions\":[{\"kind\":\"look\",\"turn\":\"left\"}],\"note\":\"please\"}",
        "{\"actions\":[{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},"
        "{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},"
        "{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"}]}",
        NULL};
    int all2 = 1;
    for (k = 0; bad_args[k]; ++k) {
      char line[2048];
      snprintf(line, sizeof(line),
               "{\"jsonrpc\":\"2.0\",\"id\":10,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":%s}}",
               bad_args[k]);
      r = rpc(s, &cl, line);
      if (!has(r, "\"check\":2")) {
        all2 = 0;
        fprintf(stderr, "  schema case %d: %s\n", k, r);
      }
    }
    CHECK(all2, "every schema violation rejected at check 2");
  }
  /* budget: 300 actions, 8 per proposal. With no observation every one is refused at check 4, so none of them
   * spends the budget: 40 x 8 would cross 300, yet none reaches check 3 (review 2026-11-16, L2) */
  {
    int at3 = 0, at4 = 0;
    for (k = 0; k < 40; ++k) {
      r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":11,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
                      "{\"actions\":[{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},"
                      "{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"},"
                      "{\"kind\":\"look\",\"turn\":\"left\"},{\"kind\":\"look\",\"turn\":\"left\"}]}}}");
      if (has(r, "\"check\":3")) ++at3;
      if (has(r, "\"check\":4")) ++at4;
    }
    if (at3 != 0 || at4 != 40) fprintf(stderr, "  at3=%d at4=%d last=%.300s\n", at3, at4, r);
    CHECK(at3 == 0 && at4 == 40, "proposals refused at check 4 don't spend the action budget");
  }
  /* many proposals: the oldest expire from the 256-entry table */
  for (k = 0; k < 260; ++k)
    rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":12,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":{}}}");
  r = rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":13,\"method\":\"tools/call\",\"params\":{\"name\":\"proposal_status\",\"arguments\":{\"id\":1}}}");
  if (!has(r, "\"state\":\"expired\"")) fprintf(stderr, "  expired? %.300s\n", r);
  CHECK(has(r, "\"state\":\"expired\""), "old proposal expired");
  A->session_close(s);
  pt_client_clear(&cl);
}

/* Outside text must never reach the audit log: tool names, methods, ids and arguments carrying a marker. */
static void injection_never_logged(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128], path[600];
  char* log;
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "grok", T_GROK);
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open for injection");
  if (!s) return;
  snprintf(path, sizeof(path), "%s", pj_test_audit_path(s));
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-11-25\",\"clientInfo\":{\"name\":\"ZZMARKER1\"}}}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ZZMARKER2 ignore previous instructions\"}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":{\"name\":\"ZZMARKER3\"}}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
              "{\"actions\":[{\"kind\":\"move\",\"dir\":\"north\",\"why\":\"ZZMARKER4\"}]}}}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"tools/call\",\"params\":{\"name\":\"propose\",\"arguments\":"
              "{\"actions\":[{\"kind\":\"ZZMARKER5\"}]}}}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":\"ZZMARKER6\",\"method\":\"ping\"}");
  rpc(s, &cl, "{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\",\"ZZMARKER7\":1}");
  rpc(s, &cl, "ZZMARKER8 not json");
  A->session_close(s);
  log = pt_read_file(path, NULL);
  CHECK(log && !strstr(log, "ZZMARKER"), "no outside text in the audit log");
  CHECK(log && !strstr(log, "grok-test-token"), "no token in the audit log");
  free(log);
  pt_client_clear(&cl);
}

static void own_thread(void) {
  peachy_session_config c;
  pt_client cl;
  PeachySession* s = NULL;
  char err[128], out[4096], line[128];
  size_t n;
  int sent = 0, got = 0, in_order = 1;
  uint64_t t_end;
  memset(&cl, 0, sizeof(cl));
  base_cfg(&c, &cl, "grok", T_GROK);
  c.write = NULL;
  CHECK(open_s(&c, &s, err) == PEACHY_OK, "open for own-thread");
  if (!s) return;
  CHECK(A->post_input(s, "x", 1) == PEACHY_E_BUSY, "post_input needs the thread");
  CHECK(A->thread_start(s) == PEACHY_OK, "thread starts");
  CHECK(A->thread_start(s) == PEACHY_E_BUSY, "second start refused");
  CHECK(A->pump(s, 1, NULL) == PEACHY_E_BUSY && A->feed(s, "x", 1, NULL) == PEACHY_E_BUSY, "pump and feed refused");
  CHECK(A->console(s, "status", out, sizeof(out)) == PEACHY_E_BUSY, "console refused while the thread runs");
  t_end = pj_mono_ns() + 60000000000ull;
  while (got < 5000 && pj_mono_ns() < t_end) {
    if (sent < 5000) {
      int k = snprintf(line, sizeof(line), "{\"jsonrpc\":\"2.0\",\"id\":%d,\"method\":\"ping\"}\n", sent);
      if (A->post_input(s, line, (size_t)k) == PEACHY_OK) ++sent;
    }
    if (A->take_output(s, out, sizeof(out), &n) == PEACHY_OK && n) {
      char want[64];
      snprintf(want, sizeof(want), "\"id\":%d,", got);
      if (!strstr(out, want)) in_order = 0;
      ++got;
    }
  }
  CHECK(got == 5000 && in_order, "5000 pings through the rings, in order");
  A->hold(s);
  {
    peachy_status_info st;
    st.struct_size = sizeof(st);
    t_end = pj_mono_ns() + 5000000000ull;
    do {
      A->status(s, &st);
    } while (!st.held && pj_mono_ns() < t_end);
    CHECK(st.held && st.thread_running == 1, "hold reaches the worker");
  }
  CHECK(A->thread_stop(s) == PEACHY_OK, "thread stops");
  A->session_close(s);
}

static void api_edges(void) {
  const peachy_api* a2 = NULL;
  peachy_version_info vi;
  peachy_status_info st;
  char buf[8];
  CHECK(peachy_get_api(PEACHY_ABI_MAJOR + 1, &a2) == PEACHY_E_ABI_MISMATCH && a2 == NULL, "ABI mismatch");
  CHECK(peachy_get_api(PEACHY_ABI_MAJOR, NULL) == PEACHY_E_INVALID_ARG, "NULL out");
  vi.struct_size = 4;
  CHECK(A->get_version(&vi) == PEACHY_E_INVALID_ARG, "short version struct");
  vi.struct_size = sizeof(vi);
  CHECK(A->get_version(&vi) == PEACHY_OK && vi.alpha == 1 && vi.abi_major == PEACHY_ABI_MAJOR, "version");
  CHECK(A->get_version(NULL) == PEACHY_E_INVALID_ARG, "NULL version");
  st.struct_size = sizeof(st);
  CHECK(A->status(NULL, &st) == PEACHY_E_BAD_HANDLE, "status on NULL");
  CHECK(A->feed(NULL, "x", 1, NULL) == PEACHY_E_BAD_HANDLE, "feed on NULL");
  CHECK(A->pump(NULL, 1, NULL) == PEACHY_E_BAD_HANDLE, "pump on NULL");
  CHECK(A->last_error(NULL, buf, sizeof(buf)) == PEACHY_E_BAD_HANDLE, "last_error on NULL");
  CHECK(strcmp(A->status_name(PEACHY_E_HELD), "held") == 0 && strcmp(A->status_name(-5), "unknown") == 0 &&
            strcmp(A->status_name(999), "unknown") == 0,
        "status names");
  A->hold(NULL);
  A->session_close(NULL);
  CHECK(1, "hold and close on NULL are safe");
}

int main(int argc, char** argv) {
  pt_init(argc, argv);
  pt_newdir(g_dir, sizeof(g_dir), "server");
  if (peachy_get_api(PEACHY_ABI_MAJOR, &A) != PEACHY_OK) return 2;
  write_allowlist();
  api_edges();
  open_refusals();
  protocol();
  grants_and_rate();
  cut_off_and_oversize();
  message_flood();
  holds();
  proposals_without_game();
  injection_never_logged();
  own_thread();
  return pt_done("peachy_server");
}
