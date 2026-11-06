"""
Skill library: the external store of learned reasoning skills.

The LLM is frozen; everything it "learns" lives here as a small list of
skills saved to skills/skill_library.json. Each skill is one dict:

    skill_id, source_task, name, description,
    applies_when, does_not_apply_when,          # read by the applicability check
    support_count, created_round, last_edited_round

Only `name` and `description` are shown to the problem-solving prompt (see
src/baseline.py::format_skills_block), so all five experimental conditions
see skills in the same format. `applies_when` / `does_not_apply_when` are for
the applicability check, not for the solver.

Library edits use four operations (as in ExpeL / SkillOpt-style curation):
ADD, EDIT, MERGE, REMOVE. They are applied to a COPY of the library; the
evolution loop decides whether to keep the copy (see skill_evolution.py).
"""

import copy
import hashlib
import json
import re
from pathlib import Path


TEXT_FIELDS = ("name", "description", "applies_when", "does_not_apply_when")

# Limits that the evolution prompts TELL the LLM about.
SOFT_WORD_LIMITS = {
    "name": 8,
    "description": 80,
    "applies_when": 30,
    "does_not_apply_when": 30,
}

# LLMs overshoot word limits slightly. An op is only rejected when a field is
# more than HARD_LIMIT_FACTOR times the soft limit (we reject, never truncate,
# because cutting a procedure mid-sentence would change its meaning).
HARD_LIMIT_FACTOR = 1.5

VALID_OPS = ("ADD", "EDIT", "MERGE", "REMOVE")

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "be", "by", "it", "its", "that", "this", "as", "at", "from", "then",
    "each", "every", "all", "any", "if", "when", "so", "than", "into", "which",
}


def _words(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


def lexical_similarity(text_a, text_b):
    """
    Jaccard overlap of content words, in [0, 1]. A cheap stand-in for
    embedding similarity, used to flag possible duplicate skills. Flags are
    only HINTS to the curator prompt, never automatic merges, so a rough
    measure is acceptable. Pass a different `similarity_fn` to the evolution
    loop to use sentence embeddings instead.
    """
    a = {w for w in _words(text_a) if w not in _STOPWORDS}
    b = {w for w in _words(text_b) if w not in _STOPWORDS}

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def skill_text(skill):
    """The text of a skill that the solver sees (and that we compare)."""
    return f"{skill.get('name', '')}. {skill.get('description', '')}"


def _count_words(text):
    return len(str(text).split())


class SkillLibrary:

    def __init__(
        self,
        skills=None,
        source_task="logical_deduction",
        max_skills=20,
        max_total_words=1200,
    ):
        self.skills = [dict(s) for s in (skills or [])]
        self.source_task = source_task
        self.max_skills = max_skills
        # ~1,200 words is roughly 1,500 tokens: small enough that the
        # "all skills" condition fits comfortably in a prompt.
        self.max_total_words = max_total_words
        self._next_num = self._max_id_number() + 1

    # ------------------------------------------------------------------ basics

    def __len__(self):
        return len(self.skills)

    def ids(self):
        return [s["skill_id"] for s in self.skills]

    def get(self, skill_id):
        for skill in self.skills:
            if skill["skill_id"] == skill_id:
                return skill
        return None

    def copy(self):
        other = SkillLibrary(
            copy.deepcopy(self.skills),
            source_task=self.source_task,
            max_skills=self.max_skills,
            max_total_words=self.max_total_words,
        )
        other._next_num = self._next_num
        return other

    def total_words(self):
        return sum(
            _count_words(s.get("name", "")) + _count_words(s.get("description", ""))
            for s in self.skills
        )

    def hash(self):
        """
        Stable fingerprint of the library content (ids + text fields; not the
        bookkeeping counters). Used to detect no-op candidates and to cache
        validation results.
        """
        payload = sorted(
            [s["skill_id"]] + [s.get(f, "") for f in TEXT_FIELDS]
            for s in self.skills
        )
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_prompt_json(self):
        """Compact JSON of the library, as shown to the reflector/curator."""
        view = [
            {
                "skill_id": s["skill_id"],
                "name": s.get("name", ""),
                "description": s.get("description", ""),
                "applies_when": s.get("applies_when", ""),
                "does_not_apply_when": s.get("does_not_apply_when", ""),
                "support_count": s.get("support_count", 1),
            }
            for s in self.skills
        ]
        return json.dumps(view, indent=2, ensure_ascii=False)

    # --------------------------------------------------------------- save/load

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.skills, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path, **kwargs):
        """
        Load a library. A missing or EMPTY file gives an empty library (the
        repo ships skills/skill_library.json as an empty placeholder).
        """
        path = Path(path)

        if not path.exists():
            return cls([], **kwargs)

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            return cls([], **kwargs)

        return cls(json.loads(text), **kwargs)

    # --------------------------------------------------------------------- ids

    def _max_id_number(self):
        numbers = []
        for skill in self.skills:
            match = re.search(r"(\d+)$", str(skill.get("skill_id", "")))
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers) if numbers else 0

    def _new_id(self):
        skill_id = f"skill_{self._next_num:03d}"
        self._next_num += 1
        return skill_id

    # ------------------------------------------------------------- similarity

    def near_duplicates(self, threshold, similarity_fn=lexical_similarity):
        """Pairs of existing skills whose text is at least `threshold` similar."""
        pairs = []
        for i in range(len(self.skills)):
            for j in range(i + 1, len(self.skills)):
                score = similarity_fn(
                    skill_text(self.skills[i]), skill_text(self.skills[j])
                )
                if score >= threshold:
                    pairs.append(
                        (self.skills[i]["skill_id"], self.skills[j]["skill_id"], score)
                    )
        return pairs

    # --------------------------------------------------------------------- ops

    def apply_ops(self, ops, round_idx, max_ops, leakage_check=None):
        """
        Apply a list of ops to a COPY of this library.

        Returns (new_library, applied, dropped):
          applied: ops that were applied (with the new/affected skill ids)
          dropped: [{"op": ..., "reason": ...}] for ops that were refused.

        An op is refused (never half-applied) if it is malformed, names a
        skill that does not exist, touches a skill another op already
        touched this round, exceeds the word limits, leaks example-specific
        content (leakage_check), or would push the library over its size caps.
        At most `max_ops` ops are applied (the round's "edit budget").
        """
        new = self.copy()
        applied, dropped = [], []
        touched = set()

        for op in ops or []:
            if len(applied) >= max_ops:
                dropped.append({"op": op, "reason": "over the round's op budget"})
                continue

            ok, info = new._apply_one(op, round_idx, touched, leakage_check)

            if ok:
                applied.append(info)
            else:
                dropped.append({"op": op, "reason": info})

        return new, applied, dropped

    def _clean_skill_fields(self, skill):
        """Validate the text fields of a proposed skill. Returns (dict, error)."""
        if not isinstance(skill, dict):
            return None, "missing 'skill' object"

        clean = {}

        for field in TEXT_FIELDS:
            value = skill.get(field, "")

            if not isinstance(value, str):
                return None, f"field '{field}' must be a string"

            value = " ".join(value.split())
            clean[field] = value

            limit = SOFT_WORD_LIMITS[field] * HARD_LIMIT_FACTOR

            if _count_words(value) > limit:
                return None, (
                    f"field '{field}' too long "
                    f"({_count_words(value)} words, soft limit "
                    f"{SOFT_WORD_LIMITS[field]})"
                )

        if not clean["name"] or not clean["description"]:
            return None, "name and description are required"

        return clean, None

    def _apply_one(self, op, round_idx, touched, leakage_check):
        if not isinstance(op, dict):
            return False, "op is not an object"

        kind = str(op.get("op", "")).upper()

        if kind not in VALID_OPS:
            return False, f"unknown op '{op.get('op')}'"

        targets = op.get("target_ids") or []

        if not isinstance(targets, list):
            return False, "target_ids must be a list"

        targets = [str(t) for t in targets]

        try:
            support = max(1, int(op.get("support_count", 1)))
        except (TypeError, ValueError):
            support = 1

        if kind == "ADD":
            targets = []
        else:
            if kind == "EDIT" and len(targets) != 1:
                return False, "EDIT needs exactly one target id"
            if kind == "MERGE" and not 2 <= len(targets) <= 3:
                return False, "MERGE needs 2-3 target ids"
            if kind == "REMOVE" and not 1 <= len(targets) <= 3:
                return False, "REMOVE needs 1-3 target ids"

            for target in targets:
                if self.get(target) is None:
                    return False, f"unknown skill id '{target}'"
                if target in touched:
                    return False, f"skill '{target}' already changed this round"

        clean = None

        if kind in ("ADD", "EDIT", "MERGE"):
            clean, error = self._clean_skill_fields(op.get("skill"))

            if error:
                return False, error

            if leakage_check is not None:
                leak = leakage_check(clean)
                if leak:
                    return False, f"leakage: {leak}"

        snapshot = copy.deepcopy(self.skills)
        snapshot_next = self._next_num
        info = {"op": kind, "target_ids": targets}

        if kind == "ADD":
            if len(self.skills) + 1 > self.max_skills:
                return False, f"library is at its cap of {self.max_skills} skills"

            new_id = self._new_id()
            self.skills.append(
                {
                    "skill_id": new_id,
                    "source_task": self.source_task,
                    **clean,
                    "support_count": support,
                    "created_round": round_idx,
                    "last_edited_round": round_idx,
                }
            )
            touched.add(new_id)
            info.update(new_id=new_id, name=clean["name"])

        elif kind == "EDIT":
            skill = self.get(targets[0])
            skill.update(clean)
            skill["support_count"] = skill.get("support_count", 1) + support
            skill["last_edited_round"] = round_idx
            touched.add(targets[0])
            info.update(new_id=targets[0], name=clean["name"])

        elif kind == "MERGE":
            first = self.get(targets[0])
            total_support = support + sum(
                self.get(t).get("support_count", 1) for t in targets
            )
            created = min(self.get(t).get("created_round", round_idx) for t in targets)
            first.update(clean)
            first["support_count"] = total_support
            first["created_round"] = created
            first["last_edited_round"] = round_idx
            self.skills = [
                s for s in self.skills if s["skill_id"] not in targets[1:]
            ]
            touched.update(targets)
            info.update(new_id=targets[0], name=clean["name"])

        elif kind == "REMOVE":
            self.skills = [s for s in self.skills if s["skill_id"] not in targets]
            touched.update(targets)

        if self.total_words() > self.max_total_words:
            self.skills = snapshot
            self._next_num = snapshot_next
            for skill_id in [info.get("new_id")] + targets:
                touched.discard(skill_id)
            return False, (
                f"library would exceed {self.max_total_words} words "
                f"(token budget for the 'all skills' condition)"
            )

        return True, info
