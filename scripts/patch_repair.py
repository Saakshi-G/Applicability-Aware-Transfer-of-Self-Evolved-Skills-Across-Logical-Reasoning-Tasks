import pathlib, sys
p = pathlib.Path("src/skill_evolution.py")
txt = p.read_text(encoding="utf-8")
if "def repair_ops" in txt:
    print("already patched"); sys.exit(0)

func = '''def repair_ops(ops, library, record=None):
    """
    Fix ops whose target ids do not exist in the library.

    A small model often "merges" proposals that are not in the library yet and
    writes made-up ids ("PROPOSAL_1", "skill_001"). The skill text is usually
    fine; only the bookkeeping is wrong. So:
      - EDIT/MERGE with no valid target id  -> ADD (empty target_ids)
      - EDIT/MERGE with some valid ids      -> keep only the valid ids
                                               (1 valid -> EDIT, 2-3 -> MERGE)
      - REMOVE keeps only valid ids (left untouched if none are valid, so
        the library refuses it)
    Every repair is written to record["op_repairs"] for the log.
    """
    repaired, notes = [], []

    for op in ops or []:
        if not isinstance(op, dict):
            repaired.append(op)
            continue

        kind = str(op.get("op", "")).upper()
        targets = op.get("target_ids") or []

        if kind not in ("EDIT", "MERGE", "REMOVE") or not isinstance(targets, list):
            repaired.append(op)
            continue

        targets = [str(t) for t in targets]
        valid = [t for t in targets if library.get(t) is not None]

        if valid == targets:
            repaired.append(op)
            continue

        new = dict(op)

        if kind == "REMOVE":
            if valid:
                new["target_ids"] = valid
                notes.append(f"REMOVE {targets} -> REMOVE {valid}")
            repaired.append(new)
            continue

        if not isinstance(op.get("skill"), dict):
            repaired.append(op)
            continue

        if not valid:
            new["op"], new["target_ids"] = "ADD", []
            notes.append(f"{kind} {targets} -> ADD")
        elif len(valid) == 1:
            new["op"], new["target_ids"] = "EDIT", valid
            notes.append(f"{kind} {targets} -> EDIT {valid}")
        else:
            new["op"], new["target_ids"] = "MERGE", valid[:3]
            notes.append(f"{kind} {targets} -> MERGE {valid[:3]}")

        repaired.append(new)

    if record is not None and notes:
        record["op_repairs"] = notes

    return repaired


'''

marker = "# ------------------------------------------------------------------- main loop"
assert marker in txt, "main loop marker not found"
txt = txt.replace(marker, func + marker, 1)

old = """                    candidate, applied, dropped = lib.apply_ops(
                        ops, round_idx, op_budget, leak.check
                    )"""
new = """                    ops = repair_ops(ops, lib, record)
                    candidate, applied, dropped = lib.apply_ops(
                        ops, round_idx, op_budget, leak.check
                    )"""
assert old in txt, "apply_ops call not found"
txt = txt.replace(old, new, 1)
p.write_text(txt, encoding="utf-8")
print("patched src/skill_evolution.py")
