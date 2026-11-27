"""
Tests that evaluate() gives the same results with and without a model .batch
method, and that CountingLLM counts batched calls. Fake model only.
Run from the repo root:  python -m pytest tests/test_batching.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fake_llm import GOOD_SKILL, FakeLLM
from src.skill_evolution import CountingLLM, EvolutionConfig, evaluate, evolve_skills
from tests.test_skill_evolution import ALL, EVOLUTION, VALIDATION


class BatchedFake:
    """FakeLLM plus a .batch method, like the Hugging Face backend."""

    def __init__(self):
        self._inner = FakeLLM(ALL)
        self.batch_sizes = []

    def __call__(self, prompt):
        return self._inner(prompt)

    def batch(self, prompts):
        self.batch_sizes.append(len(prompts))
        return [self._inner(p) for p in prompts]


def test_batched_evaluate_matches_unbatched():
    plain = evaluate(CountingLLM(FakeLLM(ALL)), VALIDATION, [GOOD_SKILL])
    batched_model = BatchedFake()
    batched = evaluate(CountingLLM(batched_model), VALIDATION, [GOOD_SKILL])

    assert batched == plain
    assert batched_model.batch_sizes == [len(VALIDATION)]   # one batch call, not one per question


def test_counting_llm_exposes_batch_only_when_model_has_it():
    assert not hasattr(CountingLLM(FakeLLM(ALL)), "batch")
    assert hasattr(CountingLLM(BatchedFake()), "batch")


def test_counting_llm_counts_every_batched_prompt():
    llm = CountingLLM(BatchedFake())
    evaluate(llm, VALIDATION, None)
    assert llm.calls == len(VALIDATION)


def test_evolution_gives_same_library_with_and_without_batch():
    cfg = EvolutionConfig(epochs=1, rollout_batch_size=10, max_workers=1, patience=5)
    a = evolve_skills(FakeLLM(ALL), EVOLUTION, VALIDATION, cfg, verbose=False)
    b = evolve_skills(BatchedFake(), EVOLUTION, VALIDATION, cfg, verbose=False)

    assert a["library"].ids() == b["library"].ids()
    assert a["validation_final"] == b["validation_final"]
    assert a["llm_calls"] == b["llm_calls"]
