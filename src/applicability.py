"""
Applicability check: ask the frozen model, per (question, skill), whether the
skill's "applies_when" / "does_not_apply_when" conditions fit the question.

An unparseable reply counts as NOT applicable (conservative: when unsure,
leave the skill out) and is counted as a parse failure so it can be reported.
"""
import re
from pathlib import Path

from src.skill_evolution import fill_template

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECK_PROMPT_PATH = PROJECT_ROOT / "prompts" / "applicability_check.txt"

_ANSWER = re.compile(r"APPLICABLE\s*:[\s*]*(YES|NO)\b", re.IGNORECASE)


def parse_applicable(text):
    """True / False from the LAST 'APPLICABLE: YES|NO' line, None if missing."""
    if not text:
        return None
    matches = _ANSWER.findall(text)
    if not matches:
        return None
    return matches[-1].upper() == "YES"


def build_check_prompt(question, skill, template=None):
    template = template if template is not None else CHECK_PROMPT_PATH.read_text(encoding="utf-8")
    return fill_template(template, {
        "NAME": skill.get("name", ""),
        "DESCRIPTION": skill.get("description", ""),
        "APPLIES_WHEN": skill.get("applies_when", ""),
        "NOT_APPLIES_WHEN": skill.get("does_not_apply_when", ""),
        "QUESTION": question,
    })


def check_applicability(llm, pairs):
    """
    pairs: [(question, skill), ...]. Returns one dict per pair:
    {"applicable": bool, "parse_failed": bool, "raw": str}.
    Uses llm.batch when the model has it (one GPU pass for all pairs).
    """
    template = CHECK_PROMPT_PATH.read_text(encoding="utf-8")
    prompts = [build_check_prompt(q, s, template) for q, s in pairs]

    batch = getattr(llm, "batch", None)
    raws = batch(prompts) if batch is not None else [llm(p) for p in prompts]

    out = []
    for raw in raws:
        verdict = parse_applicable(raw)
        out.append({
            "applicable": bool(verdict),
            "parse_failed": verdict is None,
            "raw": raw,
        })
    return out
