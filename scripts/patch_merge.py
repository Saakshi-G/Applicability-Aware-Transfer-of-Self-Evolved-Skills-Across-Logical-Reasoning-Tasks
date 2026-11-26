import pathlib, shutil, sys
p = pathlib.Path("prompts/merge_skills.txt")
v1 = pathlib.Path("prompts/merge_skills.v1.txt")
txt = p.read_text(encoding="utf-8")
if "NOT skill ids" in txt:
    print("already patched"); sys.exit(0)
anchor = 'Operations use the same format as the proposals: ADD (empty target_ids), EDIT (one id), MERGE (2-3 ids), REMOVE (1-3 ids, no "skill").'
assert anchor in txt, "anchor line not found - paste the file head so I can adjust"
if not v1.exists():
    shutil.copy(p, v1)
extra = anchor + """

Important rules about target_ids:
- target_ids may contain ONLY skill ids exactly as written in the CURRENT SKILL LIBRARY section. Proposal numbers, batch numbers, and words such as "ADD" or "proposal" are NOT skill ids.
- To combine several PROPOSALS (which are not in the library yet) into one skill, output a single ADD with empty target_ids and the combined skill. Never output a MERGE for proposals.
- Use MERGE, EDIT or REMOVE only on skills that already exist in the CURRENT SKILL LIBRARY.
- If the CURRENT SKILL LIBRARY is empty, every operation must be an ADD."""
p.write_text(txt.replace(anchor, extra), encoding="utf-8")
print("patched; old version saved as", v1)
