/* Peachy file guard: who could change the allowlist, and who could read a client's token. MIT license.
 *
 * The allowlist decides which tools each outside model may call, so the account those models run under must not
 * be able to change it, directly or by swapping it out through a folder above it. A token file must be readable by
 * no account other than the one serving it (plus SYSTEM and Administrators, or root). The platform layer gathers
 * the facts; pj_guard_check turns them into a verdict, and any fact it could not read counts against the file. */
#ifndef PJ_GUARD_H
#define PJ_GUARD_H

#include <stddef.h>

enum { PJ_GUARD_ALLOWLIST = 1, PJ_GUARD_TOKEN = 2 };

typedef struct pj_file_facts {
  int known;           /* every fact below was read; 0 means they could not be, and the file is refused */
  int regular;         /* an ordinary file, not a folder, device, link or reparse point */
  int link_in_path;    /* the file or a folder above it is a symlink, junction or other reparse point */
  int changeable;      /* this process could write, delete, re-permission or take ownership of the file */
  int path_changeable; /* this process could swap the file out through a folder above it */
  int others_can_read; /* an account other than this one, SYSTEM and Administrators (root) could read it */
  int elevated;        /* this process is elevated (Windows) or root (POSIX) */
} pj_file_facts;

/* Platform: fills f for path. Never fails; f->known says whether the facts are complete. */
void pj_file_facts_get(const char* utf8_path, pj_file_facts* f);
/* Platform: 1 if this process is elevated or root, 0 if not. Unknown counts as elevated. */
int pj_process_elevated(void);
/* Platform, run elevated: locks path down for kind. An allowlist (or its folder) becomes owned by Administrators
 * (root), full control for SYSTEM and Administrators, read-only for this account, inheritance off. A token file
 * keeps its owner and is opened to this account, SYSTEM and Administrators only (POSIX: mode 0600). 0 on success. */
int pj_protect_path(const char* utf8_path, int kind, int is_dir);

/* 0 if the file may be used as kind; otherwise -1 with a short reason (fixed text) in why. */
int pj_guard_check(const pj_file_facts* f, int kind, char* why, size_t why_cap);

#endif
