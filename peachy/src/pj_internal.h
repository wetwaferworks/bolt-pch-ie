/* Peachy internals shared between the source files. MIT license. */
#ifndef PJ_INTERNAL_H
#define PJ_INTERNAL_H

#include "peachy/peachy.h"
#include "peachy/planner_abi.h"
#include "pj_audit.h"
#include "pj_base.h"
#include "pj_json.h"
#include "pj_platform.h"
#include "pj_ring.h"

#define PEACHY_LIB_MAJOR 0
#define PEACHY_LIB_MINOR 1
#define PEACHY_LIB_PATCH 0
#define PEACHY_VERSION_STR "0.1.0"

/* ---- tools ---- */
enum {
  PJ_T_INFO = 0,
  PJ_T_STATUS,
  PJ_T_AUDIT_HEAD,
  PJ_T_OBSERVE,
  PJ_T_PLAN,
  PJ_T_PROPOSE,
  PJ_T_PROPOSAL_STATUS,
  PJ_T_HOLD,
  PJ_T_COUNT
};
#define PJ_BIT(t) (1u << (t))
#define PJ_ALWAYS_TOOLS (PJ_BIT(PJ_T_INFO) | PJ_BIT(PJ_T_STATUS) | PJ_BIT(PJ_T_HOLD))
#define PJ_GRANTABLE_TOOLS                                                                                     \
  (PJ_BIT(PJ_T_AUDIT_HEAD) | PJ_BIT(PJ_T_OBSERVE) | PJ_BIT(PJ_T_PLAN) | PJ_BIT(PJ_T_PROPOSE) |                 \
   PJ_BIT(PJ_T_PROPOSAL_STATUS))
extern const char* const pj_tool_names[PJ_T_COUNT];
int pj_tool_index(const char* name, size_t len); /* -1 if unknown */

/* ---- allowlist ---- */
#define PJ_MAX_CLIENTS 32
enum { PJ_TR_STDIO = 1u, PJ_TR_IN_PROCESS = 2u };
typedef struct pj_client {
  char name[33];
  uint32_t transports;
  uint8_t token_sha[32];
  int has_token;
  uint32_t tools;
  uint32_t rate_per_sec, burst;
  uint32_t max_mode;
} pj_client;
typedef struct pj_allow {
  pj_client c[PJ_MAX_CLIENTS];
  uint32_t n;
} pj_allow;
int pj_allow_load(const char* path, pj_allow* out, char* why, size_t why_cap);
int pj_allow_parse(const char* text, size_t len, pj_allow* out, char* why, size_t why_cap);
int pj_allow_write(const pj_allow* a, char* buf, size_t cap, size_t* len);
int pj_allow_save(const char* path, const pj_allow* a);

/* ---- game client ---- */
typedef void (*pj_game_audit_fn)(void* ctx, int outbound, const char* tool, const uint8_t sha[32], int ok);
typedef struct pj_game {
  int kind;
  int connected;
  pj_proc proc;
  peachy_game_io io;
  int64_t next_id;
  char* rbuf;
  size_t rcap, rlen;
  char* line;
  size_t line_cap;
  char* text;
  size_t text_cap;
  void* scratch;
  size_t scratch_cap;
  pj_limits jl;
  pj_doc doc;
  uint64_t skipped_lines;
  char server_name[64];
  peachy_clock_fn clock;
  void* clock_ctx;
  pj_game_audit_fn audit;
  void* audit_ctx;
} pj_game;

/* Outcome of one tool call to the game. */
enum { PJ_G_OK = 0, PJ_G_TOOL_ERROR = 1, PJ_G_NOT_GRANTED = 2, PJ_G_FAIL = -1, PJ_G_TIMEOUT = -2 };
int pj_game_setup(pj_game* g, pj_arena* a, const pj_limits* jl, uint32_t max_line);
int pj_game_open(pj_game* g, const peachy_session_config* cfg, uint64_t deadline_ns);
/* Calls a game tool. On PJ_G_OK, *payload is the parsed JSON payload of the tool's text (in g->doc) or -1. */
int pj_game_call(pj_game* g, const char* tool, const char* args_json, uint64_t deadline_ns, int32_t* payload);
void pj_game_close(pj_game* g);

/* ---- planner ---- */
typedef struct pj_planner {
  int kind;
  int loaded;
  void* lib_pchie;
  void* lib_mj;
  const pchie_api* api;
  const mjolnir_api* mj;
  pchie_runtime* rt;
  pchie_session* ps;
  uint64_t decisions;
  char why[64];
} pj_planner;
int pj_planner_open(pj_planner* p, const peachy_session_config* cfg, uint32_t width, uint32_t budget);
/* 0 and *action on success; the action is HOLD unless it is in legal_mask. */
int pj_planner_decide(pj_planner* p, const char* text, size_t len, uint32_t legal_mask, uint64_t tick, uint32_t* action);
void pj_planner_close(pj_planner* p);

#endif
