/* Peachy install folder: the code serve loads, checked before it is loaded. MIT license.
 *
 * serve loads code (the Peachy library, the PCHIE and Mjolnir planner libraries) and starts a game program. Any of
 * those runs with this account's rights the moment it loads, before any check inside it could run, so the host
 * checks them first:
 *   - they must sit directly in one install folder (peachy_host install makes it; on Windows under Program Files)
 *     that this account cannot change: the file guard's PJ_GUARD_CODE verdict for every file and the manifest;
 *   - the folder holds a manifest, peachy-install.manifest, naming every file with its SHA-256 and size, and
 *     nothing else may be in the folder;
 *   - each file is opened once, its identity and real location checked on that handle, and hashed through it.
 *     On Windows the handle denies writers and deletion and stays open while serve runs, so the bytes that were
 *     hashed are the bytes that load.
 *
 * Manifest format (canonical, so it round-trips byte for byte):
 *   PJINSTALL1\n
 *   <64 lowercase hex sha256>  <decimal size>  <name>\n      one line per file, names strictly ascending
 * Names are 1..64 characters of [A-Za-z0-9._-], not starting with '.' or '-'. */
#ifndef PJ_INSTALL_H
#define PJ_INSTALL_H

#include <stddef.h>
#include <stdint.h>

#define PJ_INSTALL_MANIFEST "peachy-install.manifest"
#define PJ_INSTALL_MAX_FILES 64u
#define PJ_INSTALL_NAME_MAX 64u
#define PJ_INSTALL_MANIFEST_MAX 16384u
#define PJ_INSTALL_FILE_MAX (1ull << 36) /* 64 GiB: far above any program, below anything absurd */

typedef struct pj_install_entry {
  char name[PJ_INSTALL_NAME_MAX + 1];
  uint8_t sha256[32];
  uint64_t size;
} pj_install_entry;

typedef struct pj_install_manifest {
  uint32_t n;
  pj_install_entry e[PJ_INSTALL_MAX_FILES];
} pj_install_manifest;

int pj_install_name_ok(const char* name);
/* 0 if buf is a well-formed canonical manifest. */
int pj_install_parse(const char* buf, size_t n, pj_install_manifest* m);
/* Canonical text for m (entries must already be sorted and valid). Returns length, 0 if it does not fit. */
size_t pj_install_format(const pj_install_manifest* m, char* out, size_t cap);
/* Insert keeping names sorted; -1 on a bad or duplicate name or when full. */
int pj_install_add(pj_install_manifest* m, const char* name, const uint8_t sha[32], uint64_t size);
const pj_install_entry* pj_install_find(const pj_install_manifest* m, const char* name);

/* An open, checked file. On Windows the handle denies writers and deletion until it is closed. */
typedef struct pj_held_file {
  intptr_t h; /* HANDLE or file descriptor; -1 when closed */
  char path[1024]; /* the checked location, install folder + '/' + name */
} pj_held_file;

/* Open path, check that it is the file named in m directly inside install_dir (after resolving links and
 * short names on the open handle), and hash it through that handle against m. 0 on success with held open;
 * otherwise -1 with a short fixed reason in why and nothing held. */
int pj_install_open_verified(const char* install_dir, const char* path, const pj_install_manifest* m,
                             pj_held_file* held, char* why, size_t why_cap);
void pj_held_close(pj_held_file* held);

/* Read and parse <install_dir>/peachy-install.manifest. 0 on success. */
int pj_install_read_manifest(const char* install_dir, pj_install_manifest* m, char* why, size_t why_cap);
/* 0 if install_dir holds exactly the manifest's files plus the manifest; otherwise -1, the first stray or
 * missing name in why. */
int pj_install_check_contents(const char* install_dir, const pj_install_manifest* m, char* why, size_t why_cap);
/* SHA-256 and size of a file (for install). 0 on success. */
int pj_install_hash_file(const char* path, uint8_t sha[32], uint64_t* size);

#endif
