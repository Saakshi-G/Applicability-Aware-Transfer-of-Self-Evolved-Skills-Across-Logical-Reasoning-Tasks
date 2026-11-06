"""
Leakage filter for evolved skills.

A skill must be a GENERAL reasoning procedure, never a note about specific
evolution questions. If a skill mentions an option letter, an object from one
of the evolution puzzles, or copies a long stretch of a question, it is
rejected BEFORE it is tested. This follows the guidance in SkillOpt
(Yang et al. 2026, "do not hardcode task-specific values") and
Piao et al. 2026 ("never encode ... literal answers, or one-task exceptions
as reusable policy").

The checks are deterministic string checks, so they cost no LLM calls.
"""

import re


_OPTION_LETTER = re.compile(r"\([A-G]\)|\boption\s+[A-G]\b")
_ARTICLES = ("a ", "an ", "the ")


def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def extract_entities(question):
    """
    Pull the object names out of a Logical Deduction style question, e.g.
    "...there are five birds: a quail, an owl, a raven, a falcon, and a robin."
    -> {"quail", "owl", "raven", "falcon", "robin"}
    Returns lowercase phrases without leading articles.
    """
    if ":" not in question:
        return set()

    after = question.split(":", 1)[1]
    sentence = after.split(".", 1)[0]

    items = re.split(r",\s*(?:and\s+)?|\s+and\s+", sentence)
    entities = set()

    for item in items:
        item = item.strip().lower()

        for article in _ARTICLES:
            if item.startswith(article):
                item = item[len(article):]
                break

        if len(item) >= 3 and len(item.split()) <= 4:
            entities.add(item)

    return entities


class LeakageFilter:

    def __init__(self, questions, ngram_size=8, max_doc_freq_fraction=0.03):
        """
        questions: the evolution-set question texts (never validation or test).

        An n-gram of `ngram_size` words counts as "copied from a question"
        only if it is rare across questions. Phrases that appear in many
        questions are template wording ("the statements are logically
        consistent within each paragraph"), which is not leakage.
        """
        self.ngram_size = ngram_size

        entities = set()
        doc_freq = {}

        for question in questions:
            entities |= extract_entities(question)

            tokens = _tokens(question)
            seen = {
                " ".join(tokens[i:i + ngram_size])
                for i in range(len(tokens) - ngram_size + 1)
            }
            for gram in seen:
                doc_freq[gram] = doc_freq.get(gram, 0) + 1

        limit = max(1, int(max_doc_freq_fraction * max(1, len(questions))))
        self.rare_ngrams = {g for g, c in doc_freq.items() if c <= limit}

        self.entities = entities

        if entities:
            alternation = "|".join(
                re.escape(e) for e in sorted(entities, key=len, reverse=True)
            )
            self._entity_pattern = re.compile(
                r"\b(?:" + alternation + r")\b", re.IGNORECASE
            )
        else:
            self._entity_pattern = None

    def check(self, skill):
        """
        Return a short reason string if the skill leaks example-specific
        content, else None. `skill` is a dict with the text fields.
        """
        text = " ".join(
            str(skill.get(f, ""))
            for f in ("name", "description", "applies_when", "does_not_apply_when")
        )

        match = _OPTION_LETTER.search(text)
        if match:
            return f"mentions an answer option ('{match.group(0)}')"

        if self._entity_pattern is not None:
            match = self._entity_pattern.search(text)
            if match:
                return f"mentions an object from an evolution question ('{match.group(0)}')"

        tokens = _tokens(text)
        for i in range(len(tokens) - self.ngram_size + 1):
            if " ".join(tokens[i:i + self.ngram_size]) in self.rare_ngrams:
                return f"copies {self.ngram_size} words from an evolution question"

        return None
