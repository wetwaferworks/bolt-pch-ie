/* Peachy planner link: PCHIE directly, or PCHIE through Mjolnir. MIT license.
 * Either way Peachy ends up holding the same PCHIE table and calls it the same way. Planning only: a decision is
 * a suggestion that Peachy checks; nothing here moves anything, writes any official chain, or turns on any
 * reasoning layer (session flags are always 0, the official defaults). */
#include <string.h>

#include "pj_internal.h"

static int why(pj_planner* p, const char* s) {
  pj_copy_cstr(p->why, sizeof(p->why), s);
  return -1;
}

static int table_ok(const pchie_api* a) {
  return a && a->struct_size >= sizeof(pchie_api) && a->struct_size <= 65536u && a->abi_major == PCHIE_ABI_MAJOR &&
         a->get_version && a->runtime_create && a->runtime_destroy && a->session_create && a->session_destroy &&
         a->session_decide;
}

int pj_planner_open(pj_planner* p, const peachy_session_config* cfg, uint32_t width, uint32_t budget) {
  pchie_runtime_config rc;
  pchie_session_config sc;
  pchie_version_info vi;
  memset(p, 0, sizeof(*p));
  p->kind = (int)cfg->planner_kind;
  if (p->kind == PEACHY_PLANNER_NONE) return why(p, "no planner configured");
  if (!cfg->pchie_path) return why(p, "no PCHIE library path");
  if (p->kind == PEACHY_PLANNER_PCHIE_DIRECT) {
    pchie_get_api_fn get;
    p->lib_pchie = pj_dl_open(cfg->pchie_path);
    if (!p->lib_pchie) return why(p, "PCHIE library did not load");
    get = (pchie_get_api_fn)pj_dl_fn(p->lib_pchie, "pchie_get_api");
    if (!get) {
      pj_planner_close(p);
      return why(p, "PCHIE library has no pchie_get_api");
    }
    if (get(PCHIE_ABI_MAJOR, &p->api) != PCHIE_OK || !table_ok(p->api)) {
      pj_planner_close(p);
      return why(p, "PCHIE interface version or table refused");
    }
  } else if (p->kind == PEACHY_PLANNER_MJOLNIR) {
    mjolnir_get_api_fn get;
    if (!cfg->mjolnir_path) return why(p, "no Mjolnir library path");
    p->lib_mj = pj_dl_open(cfg->mjolnir_path);
    if (!p->lib_mj) return why(p, "Mjolnir library not available");
    get = (mjolnir_get_api_fn)pj_dl_fn(p->lib_mj, "mjolnir_get_api");
    if (!get || get(MJOLNIR_ABI_MAJOR, &p->mj) != 0 || !p->mj || p->mj->struct_size < sizeof(mjolnir_api) ||
        p->mj->struct_size > 65536u || p->mj->abi_major != MJOLNIR_ABI_MAJOR || !p->mj->attach_pchie ||
        !p->mj->get_pchie_api || !p->mj->detach_pchie) {
      p->mj = NULL;
      pj_planner_close(p);
      return why(p, "Mjolnir interface version or table refused");
    }
    if (p->mj->attach_pchie(cfg->pchie_path, strlen(cfg->pchie_path)) != 0) {
      pj_planner_close(p);
      return why(p, "Mjolnir could not attach PCHIE");
    }
    if (p->mj->get_pchie_api(&p->api) != 0 || !table_ok(p->api)) {
      pj_planner_close(p);
      return why(p, "PCHIE table from Mjolnir refused");
    }
  } else {
    return why(p, "unknown planner kind");
  }
  memset(&vi, 0, sizeof(vi));
  vi.struct_size = sizeof(vi);
  if (p->api->get_version(&vi) != PCHIE_OK || vi.abi_major != PCHIE_ABI_MAJOR) {
    pj_planner_close(p);
    return why(p, "PCHIE version refused");
  }
  memset(&rc, 0, sizeof(rc));
  rc.struct_size = sizeof(rc);
  rc.limits.struct_size = sizeof(rc.limits);
  rc.threads = 1;
  rc.max_cpu_tier = 0xFFFFFFFFu;
  if (p->api->runtime_create(&rc, &p->rt) != PCHIE_OK || !p->rt) {
    p->rt = NULL;
    pj_planner_close(p);
    return why(p, "PCHIE runtime refused");
  }
  memset(&sc, 0, sizeof(sc));
  sc.struct_size = sizeof(sc);
  sc.fixture = 0;
  sc.width = width;
  sc.height = width;
  sc.search_budget = budget;
  sc.flags = 0;
  if (p->api->session_create(p->rt, NULL, &sc, &p->ps) != PCHIE_OK || !p->ps) {
    p->ps = NULL;
    pj_planner_close(p);
    return why(p, "PCHIE session refused");
  }
  p->loaded = 1;
  return 0;
}

int pj_planner_decide(pj_planner* p, const char* text, size_t len, uint32_t legal_mask, uint64_t tick, uint32_t* action) {
  pchie_observation o;
  pchie_decision d;
  *action = PCHIE_ACT_HOLD;
  if (!p->loaded) return -1;
  memset(&o, 0, sizeof(o));
  o.struct_size = sizeof(o);
  o.tick = tick;
  o.text = text;
  o.text_len = len;
  o.legal_mask = legal_mask;
  memset(&d, 0, sizeof(d));
  d.struct_size = sizeof(d);
  if (p->api->session_decide(p->ps, &o, &d) != PCHIE_OK) return -1;
  ++p->decisions;
  if (d.action <= PCHIE_ACT_USE && (legal_mask & (1u << d.action))) *action = d.action;
  return 0;
}

void pj_planner_close(pj_planner* p) {
  if (p->api && p->ps) p->api->session_destroy(p->ps);
  if (p->api && p->rt) p->api->runtime_destroy(p->rt);
  p->ps = NULL;
  p->rt = NULL;
  p->api = NULL;
  if (p->mj) p->mj->detach_pchie();
  p->mj = NULL;
  if (p->lib_mj) pj_dl_close(p->lib_mj);
  if (p->lib_pchie) pj_dl_close(p->lib_pchie);
  p->lib_mj = p->lib_pchie = NULL;
  p->loaded = 0;
}
