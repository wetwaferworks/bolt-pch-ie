#!/usr/bin/env python3
"""Write L6 v0 stub labels with canonical SHA-256 hashes. Stubs only."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "reasoning" / "data" / "labels" / "v0"


def jq(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def jarr(v):
    return "[" + ",".join(jq(x) for x in v) + "]"


def canonical(e: dict) -> str:
    pref = "null" if e["preferred_method"] is None else jq(e["preferred_method"])
    rev = "null" if e["reviewed_by"] is None else jq(e["reviewed_by"])
    obs = e["observation"]
    pose = obs["pose"]
    o = []
    o.append("{")
    o.append(f"\"example_id\":{jq(e['example_id'])},")
    o.append(f"\"status\":{jq(e['status'])},")
    o.append(f"\"schema_version\":{jq(e['schema_version'])},")
    o.append(f"\"world\":{jq(e['world'])},")
    o.append("\"observation\":{")
    o.append(f"\"radius\":{obs['radius']},")
    o.append(f"\"cells\":{jarr(obs['cells'])},")
    o.append("\"pose\":{")
    o.append(f"\"x\":{pose['x']},")
    o.append(f"\"y\":{pose['y']},")
    o.append(f"\"heading\":{jq(pose['heading'])}")
    o.append("}}")
    o.append(",")
    o.append(f"\"subgoal\":{jq(e['subgoal'])},")
    o.append(f"\"history_tail\":{jarr(e['history_tail'])},")
    o.append(f"\"proposed_plan\":{jarr(e['proposed_plan'])},")
    o.append(f"\"proposal_source\":{jq(e['proposal_source'])},")
    o.append(f"\"coherence_label\":{jq(e['coherence_label'])},")
    o.append(f"\"reason_codes\":{jarr(e['reason_codes'])},")
    o.append(f"\"preferred_method\":{pref},")
    o.append(f"\"labeler\":{jq(e['labeler'])},")
    o.append(f"\"reviewed_by\":{rev},")
    o.append(f"\"notes\":{jq(e['notes'])}")
    o.append("}")
    return "".join(o)


def file_json(e: dict, h: str) -> str:
    c = canonical(e)
    assert c.endswith("}")
    return c[:-1] + f",\"label_hash\":{jq(h)}" + "}"


STUBS = [
    dict(
        example_id="L6-v0-001",
        status="STUB_AWAITING_JD_REVIEW",
        schema_version="l6-label-v0",
        world="synthetic",
        observation={"radius": 1, "cells": ["UWU", "FVF", "UWU"], "pose": {"x": 1, "y": 1, "heading": "E"}},
        subgoal="ReachExit",
        history_tail=["E", "W", "E"],
        proposed_plan=["W"],
        proposal_source="model",
        coherence_label="incoherent",
        reason_codes=["IMMEDIATE_LOOP", "REVISITS_WITHOUT_NEW_INFO"],
        preferred_method="follow_corridor",
        labeler="stub",
        reviewed_by=None,
        notes="STUB. Illustrates the nav1-style back-and-forth loop.",
    ),
    dict(
        example_id="L6-v0-002",
        status="STUB_AWAITING_JD_REVIEW",
        schema_version="l6-label-v0",
        world="synthetic",
        observation={"radius": 1, "cells": ["UFU", "WVW", "UFU"], "pose": {"x": 1, "y": 1, "heading": "N"}},
        subgoal="Explore",
        history_tail=["N"],
        proposed_plan=["follow_corridor"],
        proposal_source="symbolic",
        coherence_label="coherent",
        reason_codes=["LEGAL_UNDER_BELIEF", "EXPLORES_UNKNOWN"],
        preferred_method="follow_corridor",
        labeler="stub",
        reviewed_by=None,
        notes="STUB. Corridor N-S, walls E/W; continuing is coherent.",
    ),
    dict(
        example_id="L6-v0-003",
        status="STUB_AWAITING_JD_REVIEW",
        schema_version="l6-label-v0",
        world="StageA",
        observation={"radius": 1, "cells": ["UFU", "PVF", "UFU"], "pose": {"x": 1, "y": 1, "heading": "W"}},
        subgoal="ReachExit",
        history_tail=["W"],
        proposed_plan=["W"],
        proposal_source="model",
        coherence_label="incoherent",
        reason_codes=["MOVES_INTO_KNOWN_PIT"],
        preferred_method="turn_and_reobserve",
        labeler="stub",
        reviewed_by=None,
        notes="STUB. Pit visible to the west; stepping in is never coherent.",
    ),
    dict(
        example_id="L6-v0-004",
        status="STUB_AWAITING_JD_REVIEW",
        schema_version="l6-label-v0",
        world="LithoKraft_vision",
        observation={"radius": 1, "cells": ["UUU", "WVU", "UWU"], "pose": {"x": 1, "y": 1, "heading": "E"}},
        subgoal="ReachExit",
        history_tail=[],
        proposed_plan=["S", "S", "E"],
        proposal_source="symbolic",
        coherence_label="incoherent",
        reason_codes=["MOVES_INTO_KNOWN_WALL", "USES_HIDDEN_INFO"],
        preferred_method="probe_unknown",
        labeler="stub",
        reviewed_by=None,
        notes="STUB. Plan assumes cells beyond the view; south is a seen wall. Vision path forbids hidden info.",
    ),
    dict(
        example_id="L6-v0-005",
        status="STUB_AWAITING_JD_REVIEW",
        schema_version="l6-label-v0",
        world="synthetic",
        observation={"radius": 1, "cells": ["WFW", "FVU", "WFW"], "pose": {"x": 1, "y": 1, "heading": "N"}},
        subgoal="ReachExit",
        history_tail=["N", "N"],
        proposed_plan=["E"],
        proposal_source="ncm",
        coherence_label="uncertain",
        reason_codes=["LEGAL_UNDER_BELIEF", "EXPLORES_UNKNOWN"],
        preferred_method="probe_unknown",
        labeler="stub",
        reviewed_by=None,
        notes="STUB. Junction; any open direction is defensible without more view. Shows the 'uncertain' label.",
    ),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for e in STUBS:
        c = canonical(e)
        h = hashlib.sha256(c.encode("utf-8")).hexdigest()
        path = OUT / f"{e['example_id']}.json"
        path.write_text(file_json(e, h) + "\n", encoding="utf-8")
        print(f"{e['example_id']} {h} {path}")


if __name__ == "__main__":
    main()
