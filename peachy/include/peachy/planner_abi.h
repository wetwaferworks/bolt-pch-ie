/* planner_abi.h - the parts of the PCHIE and Mjolnir library interfaces Peachy uses (alpha). MIT license.
 *
 * The PCHIE table below mirrors the draft `pchie.h` in the PCHIE DLL plan, field for field, so a real
 * `pchie.dll` built from that plan drops in unchanged. Peachy calls only get_version, status_name,
 * runtime_create, runtime_destroy, session_create (with no model: a free decide session), session_decide,
 * session_destroy and cancel. When the real `pchie.h` lands, this file includes it instead.
 *
 * The Mjolnir table is a draft for the Mjolnir library: one export, mjolnir_get_api. Peachy asks it to attach
 * the PCHIE library from a path the host gives, then takes the PCHIE table from it. That is the "Mjolnir loads
 * PCHIE, so linking Mjolnir brings PCHIE in" chain. */
#ifndef PEACHY_PLANNER_ABI_H
#define PEACHY_PLANNER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define PCHIE_CALL __cdecl
#else
#define PCHIE_CALL
#endif

#define PCHIE_ABI_MAJOR 0u
#define PCHIE_ABI_MINOR 1u

typedef struct pchie_runtime pchie_runtime;
typedef struct pchie_model pchie_model;
typedef struct pchie_session pchie_session;
typedef struct pchie_agent pchie_agent;
typedef int32_t pchie_status;
enum { PCHIE_OK = 0, PCHIE_E_INVALID_ARG = 1, PCHIE_E_UNSUPPORTED = 10, PCHIE_E_ABI_MISMATCH = 13 };

typedef struct pchie_version_info {
  uint32_t struct_size;
  uint32_t abi_major, abi_minor;
  uint32_t lib_major, lib_minor, lib_patch;
  uint64_t capabilities;
  uint32_t cpu_tier;
  char build_id[64];
} pchie_version_info;

typedef struct pchie_limits {
  uint32_t struct_size;
  uint64_t max_heap_bytes;
  uint64_t max_model_bytes;
  uint32_t max_ctx_tokens;
  uint32_t max_branches;
  uint32_t max_tree_nodes;
  uint32_t max_input_bytes;
  uint32_t max_call_ms;
  uint32_t max_steps;
} pchie_limits;

typedef void(PCHIE_CALL* pchie_task_fn)(void* ctx, int32_t index);
typedef struct pchie_dispatcher {
  uint32_t struct_size;
  void* host_ctx;
  int32_t workers;
  void(PCHIE_CALL* parallel_for)(void* host_ctx, int32_t count, pchie_task_fn fn, void* fn_ctx);
} pchie_dispatcher;

typedef void(PCHIE_CALL* pchie_log_fn)(void* host_ctx, int32_t level, const char* msg, size_t len);

typedef struct pchie_runtime_config {
  uint32_t struct_size;
  pchie_limits limits;
  int32_t threads;
  const pchie_dispatcher* dispatcher;
  pchie_log_fn log;
  void* log_ctx;
  const char* backend_dir;
  size_t backend_dir_len;
  uint64_t enable;
  uint32_t max_cpu_tier;
} pchie_runtime_config;

typedef struct pchie_model_config pchie_model_config;
typedef struct pchie_model_info pchie_model_info;
typedef struct pchie_agent_config pchie_agent_config;
typedef struct pchie_run_result pchie_run_result;

typedef struct pchie_session_config {
  uint32_t struct_size;
  uint32_t fixture; /* 0: free decide session */
  uint64_t seed;
  uint32_t width, height;
  uint32_t search_budget; /* nodes, never time */
  uint32_t flags;         /* reasoning layers off by default; Peachy always passes 0 (official defaults) */
  const char* out_dir;
  size_t out_dir_len;
} pchie_session_config;

typedef enum pchie_action {
  PCHIE_ACT_HOLD = 0,
  PCHIE_ACT_NORTH,
  PCHIE_ACT_EAST,
  PCHIE_ACT_SOUTH,
  PCHIE_ACT_WEST,
  PCHIE_ACT_PICKUP,
  PCHIE_ACT_USE
} pchie_action;

typedef struct pchie_observation {
  uint32_t struct_size;
  uint64_t tick;
  const char* text;
  size_t text_len;
  uint32_t legal_mask;
} pchie_observation;

typedef struct pchie_decision {
  uint32_t struct_size;
  uint32_t action;
  uint64_t obs_tick;
  uint64_t seq;
  uint32_t valid_for_ticks;
  uint8_t chain_head[32];
  uint64_t step_index;
} pchie_decision;

typedef struct pchie_api {
  uint32_t struct_size;
  uint32_t abi_major, abi_minor;
  pchie_status(PCHIE_CALL* get_version)(pchie_version_info* out);
  const char*(PCHIE_CALL* status_name)(pchie_status s);
  pchie_status(PCHIE_CALL* runtime_create)(const pchie_runtime_config* cfg, pchie_runtime** out);
  void(PCHIE_CALL* runtime_destroy)(pchie_runtime* rt);
  pchie_status(PCHIE_CALL* model_open)(pchie_runtime* rt, const pchie_model_config* cfg, pchie_model** out);
  pchie_status(PCHIE_CALL* model_info)(const pchie_model* m, pchie_model_info* out);
  void(PCHIE_CALL* model_close)(pchie_model* m);
  pchie_status(PCHIE_CALL* session_create)(pchie_runtime* rt, pchie_model* m, const pchie_session_config* cfg,
                                           pchie_session** out);
  void(PCHIE_CALL* session_destroy)(pchie_session* s);
  pchie_status(PCHIE_CALL* session_last_error)(const pchie_session* s, char* buf, size_t cap, size_t* needed);
  pchie_status(PCHIE_CALL* session_run)(pchie_session* s, pchie_run_result* out);
  pchie_status(PCHIE_CALL* session_decide)(pchie_session* s, const pchie_observation* obs, pchie_decision* out);
  pchie_status(PCHIE_CALL* session_actions)(const pchie_session* s, uint8_t* buf, size_t cap, size_t* needed);
  pchie_status(PCHIE_CALL* session_export_chain)(const pchie_session* s, uint8_t* buf, size_t cap, size_t* needed);
  pchie_status(PCHIE_CALL* tokenize_untrusted)(pchie_session* s, const char* text, size_t len, int32_t* out,
                                               size_t cap, size_t* n_out);
  pchie_status(PCHIE_CALL* logits_sha256)(pchie_session* s, const int32_t* tokens, size_t n, uint8_t out[32]);
  pchie_status(PCHIE_CALL* agent_start)(pchie_session* s, const pchie_agent_config* cfg, pchie_agent** out);
  pchie_status(PCHIE_CALL* agent_publish_observation)(pchie_agent* a, const pchie_observation* obs);
  pchie_status(PCHIE_CALL* agent_take_decision)(pchie_agent* a, uint64_t boundary_tick, pchie_decision* out);
  pchie_status(PCHIE_CALL* agent_health)(const pchie_agent* a, uint64_t* heartbeat, uint64_t* dropped);
  void(PCHIE_CALL* agent_stop)(pchie_agent* a);
  void(PCHIE_CALL* cancel)(pchie_session* s);
} pchie_api;

typedef pchie_status(PCHIE_CALL* pchie_get_api_fn)(uint32_t abi_major, const pchie_api** out);

/* ---- Mjolnir (draft) ---- */
#define MJOLNIR_ABI_MAJOR 0u
#define MJOLNIR_ABI_MINOR 1u

typedef struct mjolnir_api {
  uint32_t struct_size;
  uint32_t abi_major, abi_minor;
  /* Loads the PCHIE library at path (UTF-8, len bytes). 0 on success. */
  int32_t(PCHIE_CALL* attach_pchie)(const char* path, size_t len);
  /* The attached PCHIE table, or an error when nothing is attached. */
  int32_t(PCHIE_CALL* get_pchie_api)(const pchie_api** out);
  void(PCHIE_CALL* detach_pchie)(void);
} mjolnir_api;

typedef int32_t(PCHIE_CALL* mjolnir_get_api_fn)(uint32_t abi_major, const mjolnir_api** out);

#ifdef __cplusplus
}
#endif
#endif /* PEACHY_PLANNER_ABI_H */
