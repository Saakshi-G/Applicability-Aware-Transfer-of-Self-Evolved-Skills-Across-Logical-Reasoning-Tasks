import pathlib, sys
p = pathlib.Path("src/skill_evolution.py")
txt = p.read_text(encoding="utf-8")
assert "def repair_ops" in txt, "run patch_repair.py first"
if "targets or kind == \"REMOVE\"" in txt:
    print("already patched"); sys.exit(0)
old = """        if valid == targets:
            repaired.append(op)
            continue
"""
new = """        if valid == targets and (targets or kind == "REMOVE"):
            repaired.append(op)
            continue
"""
assert txt.count(old) == 1, "anchor not found"
p.write_text(txt.replace(old, new, 1), encoding="utf-8")
print("patched: empty target_ids on EDIT/MERGE now become ADD")
