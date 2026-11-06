import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOLVE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "solve_problem.txt"

# "FINAL ANSWER: (D)", "**FINAL ANSWER: (D)**", "FINAL ANSWER: **(D)**"
_PAREN_PATTERN = re.compile(r"FINAL\s*ANSWER\s*:[\s*]*\(([A-Z])\)")

# Lenient fallback for "FINAL ANSWER: D" (no parentheses). The letter must be
# followed by end of line or punctuation, so "FINAL ANSWER: A pair of..." or
# "FINAL ANSWER: I think..." are not misread as options A / I.
_BARE_PATTERN = re.compile(r"FINAL\s*ANSWER\s*:[\s*]*([A-Z])\s*[.*]*\s*(?:\n|$)")


def extract_answer(response):
    """
    Extract the final multiple-choice answer from the model response.

    Returns a string like "(D)", or None if the response has no parseable
    "FINAL ANSWER" line (e.g. truncated output). It deliberately does NOT
    guess from other "(X)" mentions in the reasoning: those are usually
    options the model was discussing, and guessing would add random noise
    to every transfer metric. Count None results as parse failures and
    report the rate.

    If "FINAL ANSWER" appears more than once (e.g. the model echoes the
    format before reasoning), the LAST one is used.
    """

    if response is None:
        return None

    response = response.upper()

    # Pick whichever form appears last in the text.
    candidates = [
        (m.start(), m.group(1))
        for pattern in (_PAREN_PATTERN, _BARE_PATTERN)
        for m in pattern.finditer(response)
    ]

    if not candidates:
        return None

    _, letter = max(candidates)
    return f"({letter})"


def is_correct(prediction, target):
    """
    Compare model prediction with the BBH target.
    """

    if prediction is None:
        return False

    return prediction.strip().upper() == target.strip().upper()


def format_skills_block(skills):
    """
    Turn a list of skill dicts (name, description) into the text inserted
    into the solve prompt. Returns "" when there are no skills, so the
    no-skills condition uses exactly the same template with the block empty.
    """

    if not skills:
        return ""

    lines = ["Learned reasoning skills:"]
    for skill in skills:
        lines.append(f"- {skill['name']}: {skill['description']}")

    return "\n".join(lines) + "\n\n"


def build_prompt(question, skills=None):
    """
    Build the problem-solving prompt for any experimental condition.

    All five conditions (no skills, all skills, random, semantic similarity,
    applicability-aware) go through this one function; they differ only in
    which `skills` list is passed in.

    Uses str.replace rather than str.format so braces inside BBH questions
    cannot break the template.
    """

    template = SOLVE_PROMPT_PATH.read_text(encoding="utf-8")

    return (
        template
        .replace("{skills_block}", format_skills_block(skills))
        .replace("{question}", question)
    )


if __name__ == "__main__":

    test_response = """
    After tracking all of the swaps, the object is currently
    in the third position.

    FINAL ANSWER: (C)
    """

    prediction = extract_answer(test_response)

    print("Extracted:", prediction)
    print("Correct:", is_correct(prediction, "(C)"))
