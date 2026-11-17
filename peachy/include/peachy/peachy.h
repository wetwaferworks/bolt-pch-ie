/* peachy.h - Peachy public C interface (alpha). MIT license.
 *
 * Peachy is a standalone library (peachy.dll / libpeachy.so). It serves PMCP, the PCHIE MCP interface: an
 * outside model connects over MCP and may only propose. Peachy checks every proposal, logs it in a
 * hash-chained audit log, and decides what, if anything, runs. Peachy is also a client of the game's own MCP
 * server (player_get_state, world_get_block, player_release, ...).
 *
 * Rules every function keeps:
 *   - Any argument values are safe: bad input returns an error code, never a crash or abort.
 *   - No exceptions, no longjmp, nothing allocated on one side and freed on the other.
 *   - No environment variables and no global state: the host passes every path, limit and clock.
 *   - No locks of any kind. Pump mode uses no threads at all; own-thread mode talks to the host only through
 *     single-producer, single-consumer rings built on atomics.
 *   - Every unexpected condition ends in the same safe state: nothing moves, the reason is logged, the
 *     session is held.
 *
 * Interface status: alpha. It may change between releases, with notes. */
#ifndef PEACHY_H
#define PEACHY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PEACHY_ABI_MAJOR 0u /* alpha */
#define PEACHY_ABI_MINOR 1u

#if defined(_WIN32)
#define PEACHY_CALL __cdecl
#else
#define PEACHY_CALL
#endif

typedef struct PeachySession PeachySession; /* opaque */

typedef int32_t peachy_status;
enum {
  PEACHY_OK = 0,
  PEACHY_E_INVALID_ARG = 1,      /* NULL where not allowed, field out of range, bad struct_size */
  PEACHY_E_BAD_HANDLE = 2,       /* not a live session */
  PEACHY_E_BUFFER_TOO_SMALL = 3, /* nothing written past capacity */
  PEACHY_E_LIMIT = 4,            /* a ceiling would be exceeded (message size, ring full, ...) */
  PEACHY_E_NO_MEMORY = 5,        /* arena or allocation refused */
  PEACHY_E_TIMEOUT = 6,          /* a deadline passed; the session is held */
  PEACHY_E_HELD = 7,             /* the session is in the safe state */
  PEACHY_E_IO = 8,               /* file, pipe or process error */
  PEACHY_E_ABI_MISMATCH = 9,
  PEACHY_E_UNSUPPORTED = 10,     /* not in this build or this stage */
  PEACHY_E_DENIED = 11,          /* unknown client, wrong token, transport not allowed, disconnected */
  PEACHY_E_PROTOCOL = 12,        /* the other side broke the protocol */
  PEACHY_E_BUSY = 13,            /* wrong mode for this call (pump call while the worker runs, ...) */
  PEACHY_E_INTERNAL = 14
};

/* Modes. Every session starts in RECORD: proposals are checked and logged, nothing runs. ADVISORY reports what
 * Peachy would do; still nothing runs. ACTUATE is not available in this stage (PM-8 needs JD's go). */
enum { PEACHY_MODE_RECORD = 0, PEACHY_MODE_ADVISORY = 1, PEACHY_MODE_ACTUATE = 2 };

/* Where `plan` gets its decisions. DIRECT: Peachy opens the PCHIE library itself. MJOLNIR: Peachy opens the
 * Mjolnir library, which loads the PCHIE library and hands its interface over. Same `plan` behavior either way. */
enum { PEACHY_PLANNER_NONE = 0, PEACHY_PLANNER_PCHIE_DIRECT = 1, PEACHY_PLANNER_MJOLNIR = 2 };

/* How Peachy reaches the game's MCP server. */
enum { PEACHY_GAME_NONE = 0, PEACHY_GAME_SPAWN_STDIO = 1, PEACHY_GAME_CALLBACKS = 2 };

/* Transport the MCP client uses to reach Peachy (checked against the allowlist). */
enum { PEACHY_TRANSPORT_STDIO = 0, PEACHY_TRANSPORT_IN_PROCESS = 1 };

typedef uint64_t(PEACHY_CALL* peachy_clock_fn)(void* ctx); /* monotonic nanoseconds */
typedef void(PEACHY_CALL* peachy_log_fn)(void* ctx, int32_t level, const char* msg, size_t len);
/* Pump mode output: one whole MCP message (no newline). Return 0 on success; anything else holds the session. */
typedef int32_t(PEACHY_CALL* peachy_write_fn)(void* ctx, const char* msg, size_t len);

/* In-process game link (PEACHY_GAME_CALLBACKS), for a host that runs the game's MCP server itself.
 * send_line: one JSON-RPC message, no newline. read_line: copy at most cap bytes of the next message
 * (no newline) into buf, set *len, return 0; return 1 when nothing arrived before timeout_ms; anything else is
 * an error. Both must return within their timeouts. */
typedef struct peachy_game_io {
  uint32_t struct_size;
  void* ctx;
  int32_t(PEACHY_CALL* send_line)(void* ctx, const char* msg, size_t len);
  int32_t(PEACHY_CALL* read_line)(void* ctx, char* buf, size_t cap, size_t* len, uint32_t timeout_ms);
  void(PEACHY_CALL* close)(void* ctx);
} peachy_game_io;

/* Ceilings. Zero means the default; values above the hard maximum are refused with PEACHY_E_INVALID_ARG. */
typedef struct peachy_limits {
  uint32_t struct_size;
  uint32_t max_message_bytes; /* default 1 MiB, max 16 MiB */
  uint32_t max_depth;         /* default 32, max 64 */
  uint32_t max_keys;          /* per object, default 256, max 4096 */
  uint32_t max_items;         /* per array, default 4096, max 65536 */
  uint32_t max_string_bytes;  /* default 64 KiB, max 1 MiB */
  uint32_t max_values;        /* per message, default 65536, max 1048576 */
  uint32_t arena_bytes;       /* default 8 MiB, max 256 MiB; the session never grows past it */
  uint32_t max_message_ms;    /* one inbound message, end to end; default 5000 */
  uint32_t max_tool_ms;       /* one tool call; default 5000 */
  uint32_t max_plan_ms;       /* one planning call; default 2000 */
  uint32_t max_game_ms;       /* one game round trip; default 1000 */
  uint32_t max_game_init_ms;  /* game start and handshake; default 120000 */
  uint32_t rate_per_sec;      /* tool calls per second ceiling; default 16, max 256 */
  uint32_t rate_burst;        /* default 32, max 512 */
  uint32_t action_budget;     /* actions proposed per session; default 1000 */
  uint32_t bad_message_limit; /* malformed messages before the client is cut off; default 16 */
  uint32_t observe_max_age_ms;/* proposals need an observation at most this old; default 2000 */
} peachy_limits;

typedef struct peachy_session_config {
  uint32_t struct_size;
  const peachy_limits* limits; /* NULL: all defaults */
  peachy_clock_fn clock;       /* NULL: the platform monotonic clock */
  void* clock_ctx;
  peachy_log_fn log;           /* NULL: silent. Never receives untrusted text. */
  void* log_ctx;
  peachy_write_fn write;       /* pump mode output; required unless own-thread mode is used */
  void* write_ctx;

  const char* allowlist_path;  /* UTF-8, required */
  const char* audit_dir;       /* UTF-8, required; must exist */
  const char* session_name;    /* [a-z0-9_-]{1,32} */

  /* The MCP client this session serves. Peachy never copies the token: it hashes the host's buffer during
   * session_open, compares against the allowlist's SHA-256, and keeps nothing. It is never logged. The host may
   * wipe its buffer as soon as session_open returns. */
  const char* client_name;
  const uint8_t* client_token;
  size_t client_token_len;
  uint32_t client_transport;   /* PEACHY_TRANSPORT_* */
  uint32_t requested_mode;     /* PEACHY_MODE_*; capped by the client's max_mode; ACTUATE refused */

  uint32_t planner_kind;       /* PEACHY_PLANNER_* */
  const char* pchie_path;      /* PCHIE library path (both planner kinds) */
  const char* mjolnir_path;    /* Mjolnir library path (MJOLNIR kind) */
  uint32_t plan_search_budget; /* nodes, never time; default 64, max 65536 */

  uint32_t game_kind;          /* PEACHY_GAME_* */
  const char* const* game_argv;/* SPAWN_STDIO: program and arguments, UTF-8 */
  uint32_t game_argc;
  const char* game_workdir;    /* may be NULL */
  const char* game_stderr_log; /* the game's stderr goes here; NULL discards it */
  const peachy_game_io* game_io; /* CALLBACKS */
  uint32_t observe_radius_max; /* default 3, max 4 */
} peachy_session_config;

typedef struct peachy_version_info {
  uint32_t struct_size;
  uint32_t abi_major, abi_minor;
  uint32_t lib_major, lib_minor, lib_patch;
  uint32_t alpha;              /* 1 while the interface is alpha */
  uint32_t checking_build;     /* 1 for sanitizer or debug builds */
  char build_id[64];
} peachy_version_info;

typedef struct peachy_status_info {
  uint32_t struct_size;
  uint32_t mode;
  uint32_t held;
  uint32_t game_connected;
  uint32_t planner_loaded;
  uint32_t thread_running;
  uint32_t client_cut_off;
  uint64_t audit_length;
  uint8_t audit_head[32];
  uint64_t messages_in, messages_out, bad_messages, proposals, proposals_rejected;
  uint64_t arena_used, arena_bytes;
  char hold_reason[96];        /* fixed Peachy text, never untrusted input */
} peachy_status_info;

typedef struct peachy_api {
  uint32_t struct_size;
  uint32_t abi_major, abi_minor;

  peachy_status(PEACHY_CALL* get_version)(peachy_version_info* out);
  const char*(PEACHY_CALL* status_name)(peachy_status s); /* static text */

  peachy_status(PEACHY_CALL* session_open)(const peachy_session_config* cfg, PeachySession** out,
                                           char* err, size_t err_cap);
  void(PEACHY_CALL* session_close)(PeachySession* s);

  /* Pump mode. feed copies bytes from the MCP client into Peachy's bounded input buffer (*accepted tells how
   * many fit). pump processes whole messages at a phase boundary until budget_ns has passed. */
  peachy_status(PEACHY_CALL* feed)(PeachySession* s, const char* bytes, size_t len, size_t* accepted);
  peachy_status(PEACHY_CALL* pump)(PeachySession* s, uint64_t budget_ns, uint32_t* messages_done);

  /* Own-thread mode. After thread_start, the host only uses post_input, take_output, hold, status and
   * thread_stop. post_input never waits: a full ring returns PEACHY_E_LIMIT. take_output pops at most one
   * message (returns PEACHY_OK with *len 0 when none is waiting). */
  peachy_status(PEACHY_CALL* thread_start)(PeachySession* s);
  peachy_status(PEACHY_CALL* thread_stop)(PeachySession* s);
  peachy_status(PEACHY_CALL* post_input)(PeachySession* s, const char* bytes, size_t len);
  peachy_status(PEACHY_CALL* take_output)(PeachySession* s, char* buf, size_t cap, size_t* len);

  /* Host console: "grant <tool>...", "revoke <tool>...", "mode record|advisory", "hold", "resume", "status".
   * Grants only ever come from here or the allowlist file, never over MCP. Pump mode only. */
  peachy_status(PEACHY_CALL* console)(PeachySession* s, const char* line, char* reply, size_t reply_cap);

  /* Safe from any thread at any time: one atomic store. The session holds at its next phase boundary. */
  void(PEACHY_CALL* hold)(PeachySession* s);

  peachy_status(PEACHY_CALL* status)(PeachySession* s, peachy_status_info* out);
  peachy_status(PEACHY_CALL* last_error)(PeachySession* s, char* buf, size_t cap);
} peachy_api;

/* The single exported symbol. */
#if defined(_WIN32)
#if defined(PEACHY_BUILDING_DLL)
__declspec(dllexport)
#endif
#elif defined(PEACHY_BUILDING_DLL)
__attribute__((visibility("default")))
#endif
peachy_status PEACHY_CALL peachy_get_api(uint32_t abi_major, const peachy_api** out);

typedef peachy_status(PEACHY_CALL* peachy_get_api_fn)(uint32_t abi_major, const peachy_api** out);

#ifdef __cplusplus
}
#endif
#endif /* PEACHY_H */
