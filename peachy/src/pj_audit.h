/* Peachy audit log: one hash-chained record per message, check and decision. MIT license.
 * Line format: <record JSON> SP <64 hex SHA-256> LF, where the hash is SHA-256(previous hash bytes || record JSON
 * bytes). The record JSON carries "prev" (the previous hash in hex) and "seq" (1, 2, 3, ...). The first record's
 * previous hash is 32 zero bytes. This is a scratch side chain: never the official PCHIE chain, never anchored,
 * never sent to a witness. */
#ifndef PJ_AUDIT_H
#define PJ_AUDIT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct pj_audit {
  FILE* f;
  uint64_t seq;
  uint8_t head[32];
  int failed;
  uint64_t fail_after; /* tests: fail the write of this record number (0 = never) */
  uint64_t bytes;      /* bytes written so far; the log never grows past PJ_AUDIT_MAX_BYTES */
  char path[512];
} pj_audit;

/* Hard ceiling on one session's audit log. Past it, appends fail and the session holds (H_AUDIT). */
#define PJ_AUDIT_MAX_BYTES (256ull << 20)

/* Creates a new file in dir; never overwrites an existing one. 0 on success. */
int pj_audit_open(pj_audit* a, const char* dir, const char* session_name);
/* Appends one record. body is the inside of the JSON object after the fixed fields (may be empty), already
 * escaped. 0 on success; on failure the log is marked failed and every later append fails too. */
int pj_audit_append(pj_audit* a, uint64_t mono_ns, const char* body, size_t body_len);
void pj_audit_close(pj_audit* a);

typedef struct pj_audit_report {
  uint64_t records;
  uint64_t first_bad;   /* 0 when the chain is whole; else the record number (line) of the first broken link */
  char why[64];
  uint8_t head[32];
} pj_audit_report;

/* Re-derives the chain read-only. 0 if whole, 1 if broken (see report), -1 if unreadable. */
int pj_audit_verify(const char* path, pj_audit_report* r);

#endif
