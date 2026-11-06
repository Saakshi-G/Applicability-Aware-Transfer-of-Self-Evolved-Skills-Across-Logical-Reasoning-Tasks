"""
Tests for src/skill_library.py and src/leakage.py.
Run from the repo root:  python -m pytest tests/   (or: python tests/test_skill_library.py)
"""
from pathlib import Path
import json
import os
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.skill_library import SkillLibrary, lexical_similarity
from src.leakage import LeakageFilter, extract_entities


def skill(name="Chaining", description="Chain the clues into one ordering.",
          applies="Ordering puzzles.", not_applies="State tracking."):
    return {"name": name, "description": description,
            "applies_when": applies, "does_not_apply_when": not_applies}


def add(name="Chaining", description="Chain the clues into one ordering.", support=2):
    return {"op": "ADD", "target_ids": [], "skill": skill(name, description),
            "support_count": support}


def test_add_edit_merge_remove():
    lib = SkillLibrary()
    new, applied, dropped = lib.apply_ops(
        [add("A skill one", "Do the first thing."), add("B skill two", "Do the second thing.")],
        round_idx=1, max_ops=3)
    assert len(lib) == 0 and len(new) == 2 and not dropped  # original untouched
    assert new.ids() == ["skill_001", "skill_002"]

    edit = {"op": "EDIT", "target_ids": ["skill_001"],
            "skill": skill("A skill one", "Do the first thing better."), "support_count": 3}
    new2, applied, dropped = new.apply_ops([edit], 2, 3)
    assert new2.get("skill_001")["description"] == "Do the first thing better."
    assert new2.get("skill_001")["support_count"] == 2 + 3   # original + op's failures
    assert new2.get("skill_001")["last_edited_round"] == 2
    assert new2.get("skill_001")["created_round"] == 1

    merge = {"op": "MERGE", "target_ids": ["skill_001", "skill_002"],
             "skill": skill("Merged skill", "Do both things."), "support_count": 1}
    new3, applied, dropped = new2.apply_ops([merge], 3, 3)
    assert new3.ids() == ["skill_001"] and new3.get("skill_001")["name"] == "Merged skill"
    assert new3.get("skill_001")["support_count"] == 1 + 5 + 2  # op + both merged skills

    remove = {"op": "REMOVE", "target_ids": ["skill_001"]}
    new4, applied, dropped = new3.apply_ops([remove], 4, 3)
    assert len(new4) == 0

    # Ids are not reused after a removal.
    new5, _, _ = new4.apply_ops([add("Later skill", "Something else.")], 5, 3)
    assert new5.ids() == ["skill_003"]


def test_malformed_and_unknown_ops_are_dropped_not_crashing():
    lib = SkillLibrary()
    bad_ops = [
        "not a dict",
        {"op": "TELEPORT"},
        {"op": "EDIT", "target_ids": ["nope"], "skill": skill()},
        {"op": "EDIT", "target_ids": [], "skill": skill()},
        {"op": "MERGE", "target_ids": ["skill_001"], "skill": skill()},
        {"op": "ADD", "skill": "text"},
        {"op": "ADD", "skill": {"name": "", "description": "x"}},
        {"op": "ADD", "skill": {"name": "n", "description": 5}},
    ]
    new, applied, dropped = lib.apply_ops(bad_ops, 1, 10)
    assert len(new) == 0 and not applied and len(dropped) == len(bad_ops)


def test_word_limits_soft_and_hard():
    lib = SkillLibrary()
    ok_desc = " ".join(["word"] * 100)       # over soft limit 80, under 120
    too_long = " ".join(["word"] * 121)      # over 1.5x
    new, applied, dropped = lib.apply_ops(
        [add("Okay length", ok_desc), add("Too long", too_long)], 1, 5)
    assert len(applied) == 1 and "too long" in dropped[0]["reason"]


def test_cap_and_word_budget_are_enforced():
    lib = SkillLibrary(max_skills=2)
    ops = [add(f"Skill number {i}", f"Procedure {i}.") for i in range(3)]
    new, applied, dropped = lib.apply_ops(ops, 1, 5)
    assert len(new) == 2 and "cap" in dropped[0]["reason"]

    lib = SkillLibrary(max_total_words=12)
    new, applied, dropped = lib.apply_ops(
        [add("First skill", "Six words in this description."),
         add("Second skill", "Another description with many many words.")], 1, 5)
    assert len(new) == 1 and "exceed" in dropped[0]["reason"]
    # The refused op left no trace (no half-applied state, id counter restored).
    new2, _, _ = new.apply_ops([add("Third skill", "Short one.")], 2, 5)
    assert new2.ids() == ["skill_001", "skill_002"]


def test_op_budget_and_independence():
    lib = SkillLibrary()
    lib, _, _ = lib.apply_ops([add("One skill", "Do one thing.")], 1, 3)

    ops = [
        {"op": "EDIT", "target_ids": ["skill_001"], "skill": skill("One skill", "Edited once.")},
        {"op": "EDIT", "target_ids": ["skill_001"], "skill": skill("One skill", "Edited twice.")},
        add("Two skill", "Another thing."),
        add("Three skill", "Yet another."),
    ]
    new, applied, dropped = lib.apply_ops(ops, 2, max_ops=2)
    assert len(applied) == 2
    assert new.get("skill_001")["description"] == "Edited once."
    reasons = [d["reason"] for d in dropped]
    assert any("already changed" in r for r in reasons)
    assert any("budget" in r for r in reasons)


def test_leakage_callback_blocks_ops():
    lib = SkillLibrary()
    blocker = lambda s: "blocked" if "owl" in s["description"] else None
    new, applied, dropped = lib.apply_ops(
        [add("Leaky skill", "Mention the owl."), add("Fine skill", "General steps.")],
        1, 5, leakage_check=blocker)
    assert len(new) == 1 and "leakage: blocked" in dropped[0]["reason"]


def test_hash_ignores_bookkeeping_but_tracks_text():
    lib, _, _ = SkillLibrary().apply_ops([add("Hash skill", "Do a thing.")], 1, 3)
    other = lib.copy()
    other.skills[0]["support_count"] = 99
    other.skills[0]["last_edited_round"] = 7
    assert lib.hash() == other.hash()
    other.skills[0]["description"] = "Do a different thing."
    assert lib.hash() != other.hash()


def test_save_load_roundtrip_and_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lib.json"

        path.write_text("")  # the repo ships an empty placeholder file
        assert len(SkillLibrary.load(path)) == 0
        assert len(SkillLibrary.load(Path(tmp) / "missing.json")) == 0

        lib, _, _ = SkillLibrary().apply_ops(
            [add("Round trip", "Survives saving.")], 1, 3)
        lib.save(path)
        loaded = SkillLibrary.load(path)
        assert loaded.skills == lib.skills and loaded.hash() == lib.hash()
        assert isinstance(json.loads(path.read_text()), list)

        # New skills continue numbering after a reload.
        again, _, _ = loaded.apply_ops([add("Another one", "More steps.")], 2, 3)
        assert again.ids() == ["skill_001", "skill_002"]


def test_similarity_and_near_duplicates():
    a = "Constraint chaining. Chain each ordering clue into one full ordering and test every option."
    b = "Constraint chaining. Chain ordering clues into a full ordering, then test each option."
    c = "Position anchoring. Place the leftmost and rightmost objects first."
    assert lexical_similarity(a, b) > 0.5 > lexical_similarity(a, c)

    lib, _, _ = SkillLibrary().apply_ops(
        [add("Constraint chaining", "Chain each ordering clue into one full ordering and test every option."),
         add("Constraint chaining v2", "Chain ordering clues into a full ordering, then test each option."),
         add("Position anchoring", "Place the leftmost and rightmost objects first.")], 1, 5)
    pairs = lib.near_duplicates(0.5)
    assert [(x, y) for x, y, _ in pairs] == [("skill_001", "skill_002")]


# ----------------------------------------------------------------- leakage


QUESTIONS = [
    "On a shelf there are three birds: a quail, an owl, and a raven. The owl is left of the raven. "
    f"The quail is the leftmost of all the birds and statement {i} holds.\nOptions:\n(A) x\n(B) y\n(C) z"
    for i in range(40)
]


def test_extract_entities():
    q = ("On a branch, there are five birds: a quail, an owl, a raven, a falcon, and a robin. "
         "The owl is the leftmost.")
    assert extract_entities(q) == {"quail", "owl", "raven", "falcon", "robin"}
    assert extract_entities("No colon here.") == set()


def test_leakage_filter_flags_leaks_and_passes_general_skills():
    f = LeakageFilter(QUESTIONS)
    check = lambda text: f.check({"name": "n", "description": text})

    assert check("Choose (B) when unsure.")
    assert check("Use option C if stuck.")
    assert check("The owl is usually leftmost.")
    assert check("QUAIL ordering matters.")           # case-insensitive

    # N-gram copy of a rare stretch of a question (no entities, no letters).
    copied = "is the leftmost of all the birds and statement 7 holds"
    assert "copies" in (check(copied) or "")

    assert check("Chain each ordering clue into one full ordering and test every option.") is None
    assert check("Enumerating options (a) and (b) is fine in lower case.") is None


def test_template_phrases_are_not_leakage():
    # An 8-word phrase present in every question is template wording, not leakage.
    qs = [f"Statements are logically consistent within each paragraph here. Item {i} is odd." for i in range(40)]
    f = LeakageFilter(qs)
    assert f.check({"name": "n", "description": "statements are logically consistent within each paragraph here"}) is None


def test_leakage_filter_on_real_bbh_if_available():
    bbh = Path(os.environ.get("BBH_DIR", PROJECT_ROOT / "external" / "Big-Bench-Hard" / "bbh"))
    if not (bbh / "logical_deduction_five_objects.json").exists():
        return  # BBH not cloned here; skip
    from src.dataset import load_family, load_split
    allq = [e["input"] for e in load_family("logical_deduction")]
    assert all(extract_entities(q) for q in allq)     # parsed for all 750

    f = LeakageFilter([e["input"] for e in load_split("logical_deduction", "evolution")])
    assert f.check({"name": "n", "description": "Write each ordering clue as a short relation, "
                    "chain them into one ordering, then test every option."}) is None
    assert f.check({"name": "n", "description": "The quail is the leftmost."})


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
