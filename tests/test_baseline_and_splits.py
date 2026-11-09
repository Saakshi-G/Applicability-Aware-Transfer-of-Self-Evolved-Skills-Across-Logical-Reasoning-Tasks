"""
Run from the repo root:  python -m pytest tests/    (or: python tests/test_baseline_and_splits.py)
Split tests need the BBH clone (external/Big-Bench-Hard or $BBH_DIR).
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import extract_answer, is_correct, build_prompt
from src.dataset import SPLIT_SIZES, make_splits


def test_extract_answer_formats():
    assert extract_answer("...\nFINAL ANSWER: (D)") == "(D)"
    assert extract_answer("final answer: (d)") == "(D)"
    assert extract_answer("**FINAL ANSWER: (D)**") == "(D)"
    assert extract_answer("FINAL ANSWER: **(D)**") == "(D)"
    assert extract_answer("FINAL ANSWER: D") == "(D)"
    assert extract_answer("FINAL ANSWER: D.") == "(D)"


def test_extract_answer_uses_last_final_answer():
    text = "I will end with FINAL ANSWER: (A) format.\nAlice has Lola.\nFINAL ANSWER: (D)"
    assert extract_answer(text) == "(D)"


def test_extract_answer_does_not_guess():
    assert extract_answer("Option (B) is wrong. Option (D) matches.") is None
    assert extract_answer("Step 1: swap, so (A) now has ... then (B)") is None
    assert extract_answer("FINAL ANSWER: A pair of swaps leaves it at Lola") is None
    assert extract_answer("FINAL ANSWER: I think it is Lola") is None
    assert extract_answer(None) is None
    assert extract_answer("") is None


def test_is_correct():
    assert is_correct("(D)", "(D)")
    assert not is_correct("(A)", "(D)")
    assert not is_correct(None, "(D)")


def test_build_prompt_no_skills_vs_skills():
    q = "Alice has {a} and Bob has {b}.\nOptions:\n(A) x\n(B) y"
    plain = build_prompt(q)
    assert "{skills_block}" not in plain and "{question}" not in plain
    assert q in plain  # braces in the question survive
    assert "Learned reasoning skills" not in plain

    skills = [{"name": "Constraint Ordering", "description": "Order items."}]
    with_skills = build_prompt(q, skills)
    assert "- Constraint Ordering: Order items." in with_skills
    # Same template: removing the skills block gives the no-skills prompt.
    assert with_skills.replace("Learned reasoning skills:\n- Constraint Ordering: Order items.\n\n", "") == plain


def test_splits_are_disjoint_sized_and_stratified():
    for family, sizes in SPLIT_SIZES.items():
        splits = make_splits(family, seed=0)

        for name, total in sizes.items():
            assert len(splits[name]) == total, (family, name)

        all_ids = [i for ids in splits.values() for i in ids]
        assert len(all_ids) == len(set(all_ids)) == 750  # disjoint, nothing lost

        for name in sizes:
            configs = [i.split(":")[0] for i in splits[name]]
            counts = {c: configs.count(c) for c in set(configs)}
            assert max(counts.values()) - min(counts.values()) <= 1, (family, name, counts)


def test_splits_are_reproducible_and_seed_dependent():
    a = make_splits("logical_deduction", seed=0)
    b = make_splits("logical_deduction", seed=0)
    c = make_splits("logical_deduction", seed=1)
    assert a == b
    assert a != c


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
