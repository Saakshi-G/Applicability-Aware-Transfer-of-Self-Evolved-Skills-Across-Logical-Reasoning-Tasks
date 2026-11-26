"""
Similarity retrieval: pick the k library skills most similar to a question.

Similarity is TF-IDF cosine over skill text (name + description +
applies_when) and the question text. It uses no model and no extra packages,
so it is fast, deterministic and easy to explain. It is lexical, not truly
semantic: a question with no words in common with any skill scores 0 for every
skill, and ties are broken by library order.

`EmbeddingScorer` is the semantic version: cosine similarity between sentence
embeddings (default model sentence-transformers/all-MiniLM-L6-v2, loaded with
`transformers`, so no extra package). Use `make_scorer("tfidf" | "embedding",
skills)` to pick one. Any object with .scores(question) -> list of floats, one
per skill, also works as a scorer.
"""
import math
from collections import Counter

from src.skill_library import _STOPWORDS, _words


def retrieval_text(skill):
    return " ".join(skill.get(f, "") for f in ("name", "description", "applies_when"))


def _tokens(text):
    return [w for w in _words(text) if w not in _STOPWORDS]


class TfidfScorer:

    def __init__(self, skills):
        docs = [Counter(_tokens(retrieval_text(s))) for s in skills]
        n = len(docs)
        df = Counter(w for d in docs for w in d)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
        self.unseen_idf = math.log(1 + n) + 1.0      # words no skill uses
        self.vectors = [self._vector(d) for d in docs]

    def _vector(self, counts):
        vec = {w: (1 + math.log(c)) * self.idf.get(w, self.unseen_idf)
               for w, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in vec.values()))
        return vec, norm

    def scores(self, question):
        q_vec, q_norm = self._vector(Counter(_tokens(question)))
        out = []
        for d_vec, d_norm in self.vectors:
            if q_norm == 0 or d_norm == 0:
                out.append(0.0)
                continue
            dot = sum(x * d_vec[w] for w, x in q_vec.items() if w in d_vec)
            out.append(dot / (q_norm * d_norm))
        return out


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def hf_embed_fn(model_name=DEFAULT_EMBEDDING_MODEL, batch_size=32):
    """texts -> list of unit-length vectors (mean-pooled transformer states)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def embed(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            out.extend(torch.nn.functional.normalize(vec, dim=-1).cpu().tolist())
        return out

    return embed


class EmbeddingScorer:
    """Cosine similarity between sentence embeddings of the question and each skill.

    `embed_fn(list[str]) -> list[list[float]]` is injectable so tests need no
    model. By default it loads `model_name` on first use.
    """

    def __init__(self, skills, model_name=DEFAULT_EMBEDDING_MODEL, embed_fn=None):
        self.embed = embed_fn or hf_embed_fn(model_name)
        self.skill_vectors = [_unit(v) for v in self.embed([retrieval_text(s) for s in skills])]
        self._cache = {}

    def scores(self, question):
        if question not in self._cache:
            self._cache[question] = _unit(self.embed([question])[0])
        q = self._cache[question]
        return [sum(a * b for a, b in zip(q, v)) for v in self.skill_vectors]


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else list(vec)


def make_scorer(kind, skills, model_name=DEFAULT_EMBEDDING_MODEL, embed_fn=None):
    """kind: "tfidf" (lexical, default) or "embedding" (semantic)."""
    if kind == "tfidf":
        return TfidfScorer(skills)
    if kind == "embedding":
        return EmbeddingScorer(skills, model_name=model_name, embed_fn=embed_fn)
    raise ValueError(f"unknown retriever '{kind}' (use 'tfidf' or 'embedding')")


def rank_skills(question, skills, scorer=None):
    """All skill indices, best first, with their scores: [(index, score), ...]."""
    scorer = scorer or TfidfScorer(skills)
    scores = scorer.scores(question)
    order = sorted(range(len(skills)), key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order]


def retrieve(question, skills, k, scorer=None):
    """Top-k skills for the question: [(skill, score), ...]."""
    return [(skills[i], s) for i, s in rank_skills(question, skills, scorer)[:k]]
