"""Tests for src/candidates.py (no model). Run: python -m pytest tests/test_candidates.py"""
from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates import build_library, collect_candidates


def sk(name, desc):
    return {"name": name, "description": desc, "applies_when": "ordering puzzles", "does_not_apply_when": "state tracking"}


def op(kind, skill, support=1):
    return {"op": kind, "target_ids": [], "skill": skill, "support_count": support}


def write_logs(tmp):
    a = {"round": 1, "decision": "rejected",
         "proposed_ops": [[op("ADD", sk("Order by constraints", "list every constraint then place each object in order"), 2)],
                          [op("ADD", sk("Order using constraints", "list each constraint then place every object in order"), 1)],
                          [op("ADD", sk("Check parity", "count swaps and track whether the number is odd or even"), 1)]],
         "dropped_ops": [{"op": op("MERGE", sk("Mentions Alice", "alice is leaked wording here"), 1), "reason": "x"}]}
    b = {"round": 2, "decision": "no-op",
         "proposed_ops": [[{"op": "ADD", "target_ids": [], "skill": {"name": "Broken", "description": ""}}]],
         "dropped_ops": []}
    path = Path(tmp) / "log.jsonl"
    path.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    gated = Path(tmp) / "lib.json"
    gated.write_text(json.dumps([{**sk("Gated skill", "write the full ranking before answering"), "skill_id": "skill_001", "support_count": 3, "created_round": 3}]))
    return path, gated


def test_collects_only_usable_skills_and_ranks_gated_first():
    with tempfile.TemporaryDirectory() as tmp:
        log, gated = write_logs(tmp)
        cands = collect_candidates([log], [gated])
        names = [c["skill"]["name"] for c in cands]
        assert "Broken" not in names                 # empty description is skipped
        assert cands[0]["gated"] and cands[0]["skill"]["name"] == "Gated skill"

        lib, prov, report = build_library(cands, leak_check=lambda s: "leak" if "alice" in s["description"] else None)

        kept = [lib.get(i)["name"] for i in lib.ids()]
        assert kept[0] == "Gated skill" and prov[0]["gated"] is True
        assert sum(n.startswith("Order") for n in kept) == 1     # near duplicates grouped into one
        assert "Check parity" in kept
        assert "Mentions Alice" not in kept                       # leakage filter applies
        assert report["gated_kept"] == 1 and len(report["refused"]) == 1


def test_cap_is_respected():
    cands = [{"skill": sk(f"Skill {i}", f"unique{i} words{i} alpha{i} beta{i}"), "support": 1, "source": "s",
              "round": 1, "decision": "rejected", "via": "proposed", "gated": False} for i in range(8)]
    lib, prov, report = build_library(cands, max_skills=3)
    assert len(lib) == 3 and len(prov) == 3
