"""
A deterministic stand-in for the real model, for dry runs and tests.

It lets the whole pipeline run on a laptop with no GPU and no API key.
It is NOT a model of real behaviour. Its rules:

  * Solver prompts: half of the questions ("easy", decided by a hash of the
    example id) are answered correctly; the other half only when the prompt
    contains the GOOD skill. A prompt containing the BAD skill makes every
    answer wrong (a harmful skill, to exercise rejection).
  * Reflector prompts: propose ADDing the GOOD skill, unless the library
    already has it.
  * Curator prompts: return the proposals de-duplicated by skill name.

Tests can replace any of the three behaviours with their own functions.
"""

import hashlib
import json
import re
import threading


GOOD_NAME = "Constraint chaining"

GOOD_SKILL = {
    "name": GOOD_NAME,
    "description": (
        "Write each ordering clue as a short relation, chain the relations "
        "into one full ordering, then test every answer option against that "
        "ordering before choosing."
    ),
    "applies_when": "Puzzles that give pairwise or positional constraints on a fixed order.",
    "does_not_apply_when": "Puzzles about how a state changes after a sequence of operations.",
}

BAD_NAME = "Always trust the first clue"

BAD_SKILL = {
    "name": BAD_NAME,
    "description": "Ignore all other clues and answer using only the first stated clue.",
    "applies_when": "Any puzzle.",
    "does_not_apply_when": "Never.",
}


def _is_easy(example_id):
    digest = hashlib.md5(str(example_id).encode("utf-8")).hexdigest()
    return int(digest, 16) % 2 == 0


def _section(prompt, start, end):
    i = prompt.find(start)
    if i == -1:
        return ""
    i += len(start)
    j = prompt.find(end, i)
    return prompt[i:j if j != -1 else len(prompt)]


class FakeLLM:

    def __init__(self, examples, solver=None, reflector=None, curator=None):
        self.examples = list(examples)
        self._solver = solver
        self._reflector = reflector
        self._curator = curator
        self.kinds = []
        self._lock = threading.Lock()

    def __call__(self, prompt):
        if prompt.startswith("You are a reasoning-skill analyst"):
            kind, handler = "reflector", self._reflector or self._default_reflector
        elif prompt.startswith("You are a skill-library curator"):
            kind, handler = "curator", self._curator or self._default_curator
        else:
            kind, handler = "solver", self._solver or self._default_solver

        with self._lock:
            self.kinds.append(kind)

        return handler(prompt)

    def count(self, kind):
        return self.kinds.count(kind)

    # ---------------------------------------------------------------- defaults

    def _find_example(self, prompt):
        for example in self.examples:
            if example["input"] in prompt:
                return example
        return None

    def _default_solver(self, prompt):
        example = self._find_example(prompt)

        if example is None:
            return "I could not find the question.\nFINAL ANSWER: (A)"

        skills_part = prompt.split("Problem:", 1)[0]
        gold = example["target"].strip()

        if BAD_NAME in skills_part:
            correct = False
        else:
            correct = _is_easy(example["id"]) or GOOD_NAME in skills_part

        if correct:
            answer = gold
        else:
            letters = re.findall(r"^\(([A-G])\)", example["input"], re.M)
            wrong = [f"({x})" for x in letters if f"({x})" != gold]
            answer = wrong[0] if wrong else "(A)"

        return f"Let me reason step by step.\nFINAL ANSWER: {answer}"

    def _default_reflector(self, prompt):
        library = _section(prompt, "=== CURRENT SKILL LIBRARY ===", "=== REJECTED EDITS")

        if GOOD_NAME in library:
            return json.dumps({"failure_summary": [], "ops": []})

        return json.dumps({
            "failure_summary": [{"pattern": "ignores part of the constraints", "count": 3}],
            "ops": [{
                "op": "ADD",
                "target_ids": [],
                "skill": GOOD_SKILL,
                "support_count": 3,
                "reason": "common error: constraints not combined",
            }],
        })

    def _default_curator(self, prompt):
        text = _section(prompt, "=== PROPOSED OPS ===", "=== POSSIBLE DUPLICATES ===")
        try:
            proposals = json.loads(text)
        except json.JSONDecodeError:
            proposals = []

        seen, ops = set(), []
        for op in proposals:
            name = (op.get("skill") or {}).get("name", "")
            if name in seen:
                continue
            seen.add(name)
            ops.append(op)

        return json.dumps({"reasoning": "dedup by name", "ops": ops})
