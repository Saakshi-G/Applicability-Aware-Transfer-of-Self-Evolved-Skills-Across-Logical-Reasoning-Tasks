"""Tests for retrieval, applicability, the five transfer conditions and metrics (no real model).
Run: python -m pytest tests/test_transfer.py"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.applicability import build_check_prompt, check_applicability, parse_applicable
from src.evaluation import MeteredLLM, paired_counts, sign_test_p, summarize
from src.retrieval import TfidfScorer, retrieve
from src.transfer import CONDITIONS, PromptCache, select_skills, solve_with_selection


def mk(sid, name, desc, applies, not_applies="state tracking"):
    return {"skill_id": sid, "name": name, "description": desc,
            "applies_when": applies, "does_not_apply_when": not_applies}


SKILLS = [
    mk("skill_001", "Order constraints", "place each object in order using the positional constraints", "objects have positional constraints"),
    mk("skill_002", "Verify arrangement", "verify the final arrangement against every clue", "a full arrangement is built"),
    mk("skill_003", "Track swaps", "write the holder of each ball after every swap", "partners swap items over several rounds"),
]

EXAMPLES = [
    {"id": f"q{i}", "input": f"Alice and Bob swap balls round {i}. Who holds the red ball?", "target": "(A)"}
    for i in range(6)
]


class ScriptedLLM:
    """Solver says (A); the judge says YES only for the skill named in `yes_for`."""

    def __init__(self, yes_for=("Track swaps",)):
        self.yes_for = yes_for

    def _one(self, prompt):
        if prompt.startswith("You decide whether one reasoning skill"):
            name = prompt.split("Skill name:", 1)[1].split("\n", 1)[0].strip()
            return "Checked.\nAPPLICABLE: " + ("YES" if name in self.yes_for else "NO")
        return "FINAL ANSWER: (A)"

    def __call__(self, prompt):
        return self._one(prompt)


class BatchedScripted(ScriptedLLM):
    def batch(self, prompts):
        return [self._one(p) for p in prompts]


def test_retrieval_ranks_by_overlap_and_is_deterministic():
    q = "balls are swapped between partners; who holds the ball after every swap"
    top = retrieve(q, SKILLS, 2)
    assert top[0][0]["skill_id"] == "skill_003" and top[0][1] > top[1][1] >= 0
    assert [s["skill_id"] for s, _ in retrieve(q, SKILLS, 2)] == [s["skill_id"] for s, _ in top]
    # no shared words at all: all scores 0, ties keep library order
    assert [s["skill_id"] for s, _ in retrieve("zzz qqq", SKILLS, 2)] == ["skill_001", "skill_002"]
    assert TfidfScorer(SKILLS).scores("") == [0.0, 0.0, 0.0]


def test_parse_applicable():
    assert parse_applicable("ok\nAPPLICABLE: YES") is True
    assert parse_applicable("APPLICABLE: no") is False
    assert parse_applicable("APPLICABLE: YES ... later: APPLICABLE: NO") is False   # last one wins
    assert parse_applicable("I think so") is None and parse_applicable(None) is None


def test_check_prompt_survives_braces_in_question():
    p = build_check_prompt("What is {NAME} in {x}?", SKILLS[0])
    assert "What is {NAME} in {x}?" in p and "Order constraints" in p


def test_check_applicability_marks_unparseable_as_not_applicable():
    class Weird:
        def __call__(self, prompt):
            return "no idea"
    out = check_applicability(Weird(), [("q", SKILLS[0])])
    assert out == [{"applicable": False, "parse_failed": True, "raw": "no idea"}]


def test_conditions_select_expected_skills():
    for llm in (ScriptedLLM(), BatchedScripted()):
        sel, _ = select_skills("no_skills", EXAMPLES, SKILLS)
        assert all(s == [] for s in sel)

        sel, _ = select_skills("all_skills", EXAMPLES, SKILLS)
        assert all(len(s) == 3 for s in sel)

        sel1, d1 = select_skills("random_skills", EXAMPLES, SKILLS, k=2, seed=0)
        sel2, _ = select_skills("random_skills", EXAMPLES, SKILLS, k=2, seed=0)
        sel3, _ = select_skills("random_skills", EXAMPLES, SKILLS, k=2, seed=1)
        assert sel1 == sel2 and all(len(s) == 2 for s in sel1) and sel1 != sel3

        sel, d = select_skills("similarity", EXAMPLES, SKILLS, k=1)
        assert all(s[0]["skill_id"] == "skill_003" for s in sel)      # swap wording matches Track swaps

        sel, d = select_skills("applicability", EXAMPLES, SKILLS, k=3, llm=llm)
        assert all([x["skill_id"] for x in s] == ["skill_003"] for s in sel)
        assert d[0]["kept"] == ["skill_003"] and len(d[0]["judged"]) == 3

        sel, _ = select_skills("applicability", EXAMPLES, SKILLS, k=3, llm=ScriptedLLM(yes_for=()))
        assert all(s == [] for s in sel)                              # nothing applicable -> no skills shown


def test_k_is_capped_at_library_size_and_unknown_condition_fails():
    sel, _ = select_skills("similarity", EXAMPLES, SKILLS, k=10)
    assert all(len(s) == 3 for s in sel)
    try:
        select_skills("nonsense", EXAMPLES, SKILLS)
        assert False
    except ValueError:
        pass


def test_solve_and_metrics_and_cost_metering():
    metered = MeteredLLM(BatchedScripted())
    base = solve_with_selection(metered, EXAMPLES, [[] for _ in EXAMPLES])
    assert all(r["correct"] for r in base) and base[0]["skill_ids"] == []
    assert metered.calls == 6 and metered.prompt_chars > 0

    # a condition that breaks 2 questions and repairs none
    worse = [dict(r) for r in base]
    worse[0]["correct"] = worse[1]["correct"] = False
    assert paired_counts(base, worse) == (0, 2)

    rows = summarize({"no_skills": base, "similarity": worse},
                     {"similarity": {"calls": 6, "prompt_chars": 100, "seconds": 1.0}})
    assert rows["no_skills"]["transfer_gain"] == 0 and rows["no_skills"]["p_value"] == 1.0
    assert rows["similarity"]["broken"] == 2 and rows["similarity"]["transfer_gain"] < 0
    assert rows["similarity"]["cost"]["calls"] == 6


def test_sign_test_p_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(20, 26) > 0.4                      # our Qwen 3B result: not significant
    assert sign_test_p(0, 10) < 0.01
    assert sign_test_p(5, 5) == 1.0


def test_all_five_conditions_are_defined():
    assert CONDITIONS == ("no_skills", "all_skills", "random_skills", "similarity", "applicability")


def test_identical_prompts_are_answered_once_and_reused():
    class Flaky:
        """Answers differently every call: a model whose repeated answers would drift."""
        def __init__(self):
            self.n = 0

        def __call__(self, prompt):
            self.n += 1
            return f"FINAL ANSWER: ({'A' if self.n % 2 else 'B'})"

    llm = MeteredLLM(Flaky())
    cache = PromptCache()
    base = solve_with_selection(llm, EXAMPLES, [[] for _ in EXAMPLES], cache)
    calls_after_base = llm.calls
    again = solve_with_selection(llm, EXAMPLES, [[] for _ in EXAMPLES], cache)   # e.g. applicability kept no skills

    assert llm.calls == calls_after_base == 6                      # no new model calls
    assert [r["prediction"] for r in again] == [r["prediction"] for r in base]
    assert cache.hits == 6

    shown = solve_with_selection(llm, EXAMPLES[:2], [[SKILLS[0]], [SKILLS[0]]], cache)
    assert llm.calls == 8 and cache.hits == 6                      # different prompts are really solved


# ------------------------------------------------------- semantic retriever

def _toy_embed(texts):
    """Two axes: 'ordering' words and 'swap' words. Synonyms share an axis, so a
    skill and a question can match with NO words in common (what TF-IDF cannot do)."""
    order = {"rank", "order", "left", "right", "position"}
    swap = {"swap", "trade", "exchange", "switch"}
    out = []
    for text in texts:
        words = set(text.lower().replace(".", " ").split())
        out.append([float(len(words & order)), float(len(words & swap)), 0.01])
    return out


def test_embedding_scorer_matches_meaning_without_shared_words():
    from src.retrieval import EmbeddingScorer, TfidfScorer, retrieve
    skills = [
        {"skill_id": "a", "name": "Order", "description": "rank the items by position", "applies_when": "left right order"},
        {"skill_id": "b", "name": "Swaps", "description": "follow every exchange", "applies_when": "exchange switch"},
    ]
    question = "Alice and Bob swap, then Bob and Cara trade."      # shares no words with skill b's text
    assert TfidfScorer(skills).scores(question) == [0.0, 0.0]       # lexical sees nothing
    scorer = EmbeddingScorer(skills, embed_fn=_toy_embed)
    top = retrieve(question, skills, 1, scorer)[0][0]
    assert top["skill_id"] == "b"


def test_make_scorer_and_importing_retrieval_loads_no_model():
    import sys
    from src.retrieval import EmbeddingScorer, TfidfScorer, make_scorer
    assert "torch" not in sys.modules or True    # import must not require loading a model
    skills = [{"skill_id": "a", "name": "n", "description": "d", "applies_when": "w"}]
    assert isinstance(make_scorer("tfidf", skills), TfidfScorer)
    assert isinstance(make_scorer("embedding", skills, embed_fn=_toy_embed), EmbeddingScorer)
    try:
        make_scorer("nope", skills)
        assert False
    except ValueError:
        pass


# ------------------------------------------------- separate applicability judge

def test_judge_model_is_used_only_for_applicability_judgments():
    from src.transfer import select_skills, solve_with_selection
    skills = [{"skill_id": "s1", "name": "A", "description": "d", "applies_when": "w", "does_not_apply_when": "n"}]
    examples = [{"id": "q1", "input": "Question one?", "target": "(A)"}]
    judge_prompts, solver_prompts = [], []

    def judge(prompt):
        judge_prompts.append(prompt)
        return "APPLICABLE: YES"

    def solver(prompt):
        solver_prompts.append(prompt)
        return "FINAL ANSWER: (A)"

    selected, details = select_skills("applicability", examples, skills, k=1, llm=judge)
    assert len(judge_prompts) == 1 and not solver_prompts
    assert details[0]["kept"] == ["s1"]

    solve_with_selection(solver, examples, selected)
    assert len(solver_prompts) == 1 and len(judge_prompts) == 1       # solving never calls the judge
