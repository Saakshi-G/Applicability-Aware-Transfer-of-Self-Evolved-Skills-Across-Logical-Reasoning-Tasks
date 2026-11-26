"""
Build an "ungated candidate" skill library from evolution logs.

The evolution loop logs every skill the model proposed (reflector proposals
in "proposed_ops", curated-but-refused ops in "dropped_ops"). Most were
rejected by the validation gate. This module collects them, groups near
duplicates, ranks them, and builds one library the retrieval and
applicability steps can work with. Skills that passed the gate (found in the
final library files) rank first and are flagged.

No model is called here. The result is a comparison library ("ungated"), not
a replacement for the strict-gate library.
"""
import json

from src.skill_library import SkillLibrary, TEXT_FIELDS, lexical_similarity, skill_text

BUILD_OPS = ("ADD", "EDIT", "MERGE")


def _usable_skill(skill):
    if not isinstance(skill, dict):
        return None
    clean = {}
    for field in TEXT_FIELDS:
        value = skill.get(field)
        if not isinstance(value, str) or not value.strip():
            return None
        clean[field] = " ".join(value.split())
    return clean


def _from_op(op, source, round_idx, decision, via):
    if not isinstance(op, dict):
        return None
    if str(op.get("op", "")).upper() not in BUILD_OPS:
        return None
    skill = _usable_skill(op.get("skill"))
    if skill is None:
        return None
    try:
        support = max(1, int(op.get("support_count", 1)))
    except (TypeError, ValueError):
        support = 1
    return {"skill": skill, "support": support, "source": source,
            "round": round_idx, "decision": decision, "via": via, "gated": False}


def collect_candidates(log_paths, gated_library_paths=()):
    """
    Returns a list of candidate dicts: skill (4 text fields), support,
    source (file name), round, decision, via, gated.
    Gated skills (from final library files) come first.
    """
    cands = []

    for path in gated_library_paths:
        with open(path, encoding="utf-8") as handle:
            for skill in json.load(handle):
                clean = _usable_skill(skill)
                if clean:
                    cands.append({"skill": clean, "support": int(skill.get("support_count", 1) or 1),
                                  "source": str(path), "round": skill.get("created_round"),
                                  "decision": "accepted", "via": "gated_library", "gated": True})

    for path in log_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                decision = record.get("decision", "")
                round_idx = record.get("round")

                for batch in record.get("proposed_ops") or []:
                    for op in batch or []:
                        cand = _from_op(op, str(path), round_idx, decision, "proposed")
                        if cand:
                            cands.append(cand)

                for dropped in record.get("dropped_ops") or []:
                    cand = _from_op(dropped.get("op"), str(path), round_idx, decision, "curated_dropped")
                    if cand:
                        cands.append(cand)

    return cands


def cluster_candidates(cands, threshold=0.5, similarity_fn=lexical_similarity):
    """Greedy grouping of near-duplicate skills. Returns clusters, best first."""
    clusters = []

    for cand in cands:
        text = skill_text(cand["skill"])
        for cluster in clusters:
            if similarity_fn(skill_text(cluster["members"][0]["skill"]), text) >= threshold:
                cluster["members"].append(cand)
                break
        else:
            clusters.append({"members": [cand]})

    for cluster in clusters:
        members = cluster["members"]
        gated = [m for m in members if m["gated"]]
        cluster["has_gated"] = bool(gated)
        cluster["rep"] = gated[0] if gated else max(members, key=lambda m: m["support"])
        cluster["n_proposals"] = len(members)
        cluster["total_support"] = sum(m["support"] for m in members)

    clusters.sort(key=lambda c: (c["has_gated"], c["n_proposals"], c["total_support"]), reverse=True)
    return clusters


def build_library(cands, leak_check=None, threshold=0.5, max_skills=20,
                  source_task="logical_deduction", similarity_fn=lexical_similarity):
    """
    Returns (library, provenance, report).
    provenance: one dict per kept skill (skill_id, name, gated, n_proposals,
    sources). report: counts and the reasons skills were refused.
    """
    clusters = cluster_candidates(cands, threshold, similarity_fn)
    lib = SkillLibrary(source_task=source_task, max_skills=max_skills)
    provenance, refused = [], []

    for cluster in clusters:
        if len(lib) >= max_skills:
            break
        rep = cluster["rep"]
        op = {"op": "ADD", "target_ids": [], "skill": rep["skill"],
              "support_count": cluster["total_support"]}
        lib, applied, dropped = lib.apply_ops([op], 0, 1, leak_check)

        if applied:
            provenance.append({
                "skill_id": lib.ids()[-1],
                "name": rep["skill"]["name"],
                "gated": cluster["has_gated"],
                "n_proposals": cluster["n_proposals"],
                "sources": sorted({m["source"] for m in cluster["members"]}),
            })
        else:
            refused.append({"name": rep["skill"]["name"], "reason": dropped[0]["reason"]})

    report = {
        "candidates_found": len(cands),
        "distinct_clusters": len(clusters),
        "skills_kept": len(lib),
        "gated_kept": sum(p["gated"] for p in provenance),
        "refused": refused,
    }
    return lib, provenance, report
