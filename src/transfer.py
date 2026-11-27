"""
Transfer conditions: which skills the solver sees for each target question.

  no_skills      nothing
  all_skills     every skill in the library
  random_skills  k skills drawn at random (seeded per question)
  similarity     the k skills most similar to the question (src/retrieval.py)
  applicability  skills the model judges applicable to the question, best
                 similarity first, at most k (src/applicability.py)

All five go through the same prompt template (build_prompt); they differ only
in the skill list.
"""
import random

from src.applicability import check_applicability
from src.baseline import build_prompt
from src.retrieval import TfidfScorer, rank_skills
from src.skill_evolution import _result_from_raw

CONDITIONS = ("no_skills", "all_skills", "random_skills", "similarity", "applicability")


class StubLLM:
    """Dry-run model: answers (A) and says every skill is NOT applicable."""

    def _one(self, prompt):
        if prompt.startswith("You decide whether one reasoning skill"):
            return "Not relevant.\nAPPLICABLE: NO"
        return "FINAL ANSWER: (A)"

    def __call__(self, prompt):
        return self._one(prompt)

    def batch(self, prompts):
        return [self._one(p) for p in prompts]


def select_skills(condition, examples, skills, k=3, seed=0, llm=None, pool=None, scorer=None):
    """
    Returns (selected, details), both lists aligned with `examples`.
    selected[i]: list of skill dicts shown for example i.
    details[i]: what was decided (scores, judgments) for the log.
    `pool`: for the applicability condition, judge only the `pool` most
    similar skills (None = judge every skill).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition '{condition}'")

    k = min(k, len(skills))
    selected, details = [], []

    if condition == "no_skills":
        return [[] for _ in examples], [{} for _ in examples]

    if condition == "all_skills":
        return [list(skills) for _ in examples], [{} for _ in examples]

    if condition == "random_skills":
        for e in examples:
            rng = random.Random(f"{seed}-{e['id']}")
            picked = rng.sample(range(len(skills)), k)
            selected.append([skills[i] for i in picked])
            details.append({"picked": [skills[i]["skill_id"] for i in picked]})
        return selected, details

    scorer = scorer or TfidfScorer(skills)
    ranked = [rank_skills(e["input"], skills, scorer) for e in examples]

    if condition == "similarity":
        for r in ranked:
            top = r[:k]
            selected.append([skills[i] for i, _ in top])
            details.append({"scores": {skills[i]["skill_id"]: round(s, 4) for i, s in top}})
        return selected, details

    # applicability
    if llm is None:
        raise ValueError("the applicability condition needs a model")

    pool = len(skills) if pool is None else min(pool, len(skills))
    pairs, owners = [], []
    for ex_idx, (e, r) in enumerate(zip(examples, ranked)):
        for i, score in r[:pool]:
            pairs.append((e["input"], skills[i]))
            owners.append((ex_idx, i, score))

    verdicts = check_applicability(llm, pairs)

    per_example = [[] for _ in examples]
    for (ex_idx, i, score), v in zip(owners, verdicts):
        per_example[ex_idx].append((i, score, v))

    for items in per_example:
        keep = [(i, s) for i, s, v in items if v["applicable"]][:k]
        selected.append([skills[i] for i, _ in keep])
        details.append({
            "judged": {skills[i]["skill_id"]: {
                "applicable": v["applicable"], "parse_failed": v["parse_failed"],
                "score": round(s, 4), "raw": v["raw"]} for i, s, v in items},
            "kept": [skills[i]["skill_id"] for i, _ in keep],
        })
    return selected, details


class PromptCache(dict):
    """prompt text -> raw model reply. Shared across conditions.

    A deterministic model gives one reply per prompt, so a prompt that is
    identical in two conditions (e.g. a question that gets no skills under the
    applicability condition, which is the no-skills prompt) is answered once.
    This also removes batching noise: the same prompt in a differently padded
    batch can flip a borderline answer, which is not an effect of any skill.
    `hits` counts prompts served from the cache.
    """

    def __init__(self):
        super().__init__()
        self.hits = 0


def solve_with_selection(llm, examples, selected, cache=None):
    """Solve every example with its own skill list. Results keep example order."""
    cache = PromptCache() if cache is None else cache
    prompts = [build_prompt(e["input"], s) for e, s in zip(examples, selected)]

    todo = [p for p in dict.fromkeys(prompts) if p not in cache]
    cache.hits += len(prompts) - len(todo)

    if todo:
        batch = getattr(llm, "batch", None)
        raws = batch(todo) if batch is not None else [llm(p) for p in todo]
        cache.update(zip(todo, raws))

    results = []
    for e, s, prompt in zip(examples, selected, prompts):
        r = _result_from_raw(e, cache[prompt])
        r["skill_ids"] = [x.get("skill_id") for x in s]
        results.append(r)
    return results
