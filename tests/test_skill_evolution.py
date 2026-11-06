"""
Tests for src/skill_evolution.py using the fake model (no GPU, no API).
Run from the repo root:  python -m pytest tests/   (or: python tests/test_skill_evolution.py)
"""
from pathlib import Path
import json
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fake_llm import (
    BAD_NAME, BAD_SKILL, GOOD_NAME, GOOD_SKILL, FakeLLM,
)
from src.skill_evolution import (
    EvolutionConfig, accuracy, calibrate_noise, evaluate, evolve_skills,
    extract_json, fill_template, paired_gate,
)


def make_examples(n, prefix):
    return [
        {
            "id": f"{prefix}:{i}",
            "input": (
                "On a shelf there are three birds: a quail, an owl, and a raven. "
                f"The owl is left of the raven in case {prefix}{i}.\n"
                "Options:\n(A) The quail is the leftmost\n(B) The owl is the leftmost\n"
                "(C) The raven is the leftmost"
            ),
            "target": "(B)",
        }
        for i in range(n)
    ]


EVOLUTION = make_examples(40, "ev")
VALIDATION = make_examples(20, "va")
ALL = EVOLUTION + VALIDATION


def reply(ops):
    return json.dumps({"failure_summary": [], "ops": ops})


def add_op(skill_dict):
    return {"op": "ADD", "target_ids": [], "skill": skill_dict, "support_count": 3}


def small_config(**overrides):
    values = dict(epochs=1, rollout_batch_size=10, max_workers=1, patience=5)
    values.update(overrides)
    return EvolutionConfig(**values)


# ------------------------------------------------------------- unit pieces


def result(correct):
    return {"correct": correct}


def test_paired_gate_rule():
    def gate(inc, cand):
        return paired_gate([result(x) for x in inc], [result(x) for x in cand])

    # repaired 3, broken 1 -> u = 1 > 0 -> accept
    g = gate([0, 0, 0, 1, 1], [1, 1, 1, 0, 1])
    assert (g["repaired"], g["broken"], g["utility"], g["accept"]) == (3, 1, 1, True)
    # repaired 2, broken 1 -> u = 0 -> reject (ties rejected)
    assert gate([0, 0, 1, 1], [1, 1, 0, 1])["accept"] is False
    # identical -> reject
    assert gate([1, 0, 1], [1, 0, 1])["accept"] is False
    # only gains -> accept
    assert gate([0, 0], [1, 0])["accept"] is True
    # net accuracy equal but with swaps -> reject
    assert gate([1, 0], [0, 1])["accept"] is False


def test_extract_json_variants():
    assert extract_json('{"ops": []}') == {"ops": []}
    assert extract_json('```json\n{"ops": [{"op": "ADD"}]}\n```')["ops"][0]["op"] == "ADD"
    assert extract_json('Sure! Here you go: {"ops": []} Hope this helps.') == {"ops": []}
    # Prefers the object that has "ops" over an earlier unrelated object.
    assert extract_json('{"note": 1} then {"ops": [1]}') == {"ops": [1]}
    assert extract_json("no json here") is None
    assert extract_json('{"ops": [broken') is None
    assert extract_json("") is None and extract_json(None) is None


def test_fill_template_is_single_pass():
    out = fill_template("A {X} B {Y} C {UNKNOWN}", {"X": "{Y}", "Y": "ok"})
    assert out == "A {Y} B ok C {UNKNOWN}"   # inserted braces are not re-expanded


def test_calibrate_noise_reports_stable_runs():
    llm = FakeLLM(ALL)
    report = calibrate_noise(llm, VALIDATION, repeats=3)
    assert report["flipped_questions"] == 0 and len(set(report["accuracies"])) == 1


# -------------------------------------------------------------- full loop


def test_happy_path_accepts_good_skill_and_logs():
    with tempfile.TemporaryDirectory() as tmp:
        log, ckpt = Path(tmp) / "log.jsonl", Path(tmp) / "lib.json"
        llm = FakeLLM(ALL)
        out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(),
                            log_path=log, checkpoint_path=ckpt, verbose=False)

        lib = out["library"]
        assert lib.ids() == ["skill_001"] and lib.get("skill_001")["name"] == GOOD_NAME
        assert out["validation_final"] > out["validation_start"]
        assert out["validation_final"] == 1.0

        records = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(records) == len(out["history"]) == 4          # 40 examples / batches of 10
        first = records[0]
        assert first["decision"] == "accepted" and first["gate"]["broken"] == 0
        assert first["library_hash_before"] != first["library_hash_after"]
        assert first["llm_calls_this_round"] > 0
        assert json.loads(ckpt.read_text())[0]["name"] == GOOD_NAME   # checkpoint written


def test_harmful_skill_is_rejected_and_remembered():
    prompts = []

    def reflector(prompt):
        prompts.append(prompt)
        return reply([add_op(BAD_SKILL)])

    llm = FakeLLM(ALL, reflector=reflector)
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), verbose=False)

    assert len(out["library"]) == 0
    assert out["validation_final"] == out["validation_start"]
    rejected = [r for r in out["history"] if r["decision"] == "rejected"]
    assert rejected and rejected[0]["gate"]["broken"] > 0 and rejected[0]["gate"]["utility"] < 0
    # The rejected edit is shown to the reflector in the NEXT round.
    assert BAD_NAME in prompts[1].split("=== REJECTED EDITS")[1].split("=== FAILED ATTEMPTS")[0]
    assert "(rejected)" in prompts[1]
    assert "(none)" in prompts[0].split("=== REJECTED EDITS")[1].split("=== FAILED ATTEMPTS")[0]


def test_rejected_buffer_resets_each_epoch():
    prompts = []

    def reflector(prompt):
        prompts.append(prompt)
        return reply([add_op(BAD_SKILL)])

    cfg = small_config(epochs=2, rollout_batch_size=40, patience=10)  # one round per epoch
    llm = FakeLLM(ALL, reflector=reflector)
    evolve_skills(llm, EVOLUTION, VALIDATION, cfg, verbose=False)
    section = lambda p: p.split("=== REJECTED EDITS")[1].split("=== FAILED ATTEMPTS")[0]
    assert "(none)" in section(prompts[0]) and "(none)" in section(prompts[1])


def test_leaking_skill_is_dropped_before_validation():
    leaky = dict(GOOD_SKILL, description="The owl is the leftmost, so choose (B).")
    llm = FakeLLM(ALL, reflector=lambda p: reply([add_op(leaky)]))
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), verbose=False)

    first = out["history"][0]
    assert first["decision"] == "no-op" and first["reason"] == "no valid ops after filtering"
    assert "leakage" in first["dropped_ops"][0]["reason"]
    assert len(out["library"]) == 0
    # Only rollouts + the initial validation run were solved (no candidate validation).
    assert llm.count("solver") == len(VALIDATION) + len(EVOLUTION)


def test_unusable_reflector_output_does_not_crash():
    for garbage in ["I cannot help with that.", '{"ops": "nope"}', "", "{broken"]:
        llm = FakeLLM(ALL, reflector=lambda p, g=garbage: g)
        out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), verbose=False)
        assert len(out["library"]) == 0
        assert all(r["decision"] == "no-op" for r in out["history"])
        assert out["history"][0]["reflector_parse_errors"] >= 1


def test_early_stop_after_patience():
    llm = FakeLLM(ALL, reflector=lambda p: reply([]))
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(patience=2), verbose=False)
    assert len(out["history"]) == 2 and "2 rounds in a row" in out["stopped"]


def test_no_failures_means_no_reflection():
    llm = FakeLLM(ALL, solver=lambda prompt: "FINAL ANSWER: (B)")   # always right
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), verbose=False)
    assert llm.count("reflector") == 0
    assert all("no reasoning failures" in r["reason"] for r in out["history"])


def test_parse_failures_are_counted_but_not_reflected_on():
    llm = FakeLLM(ALL, solver=lambda prompt: "I am still thinking and ran out of")
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), verbose=False)
    first = out["history"][0]
    assert first["n_parse_failures"] == 10 and first["n_failures_reflected"] == 0
    assert llm.count("reflector") == 0


def test_curator_runs_when_several_minibatches_propose_edits():
    cfg = small_config(reflection_minibatch_size=2, max_failure_minibatches=3, rollout_batch_size=20)
    llm = FakeLLM(ALL)       # default reflector proposes the same ADD every call
    out = evolve_skills(llm, EVOLUTION, VALIDATION, cfg, verbose=False)

    assert llm.count("curator") >= 1
    first = out["history"][0]
    assert first["curator_called"] and first["failure_minibatches"] == 3
    assert out["library"].ids() == ["skill_001"]      # three identical proposals -> one skill


def test_failure_minibatches_are_capped():
    cfg = small_config(reflection_minibatch_size=1, max_failure_minibatches=2, rollout_batch_size=20)
    llm = FakeLLM(ALL, reflector=lambda p: reply([]))
    out = evolve_skills(llm, EVOLUTION, VALIDATION, cfg, verbose=False)
    assert llm.count("reflector") == 2 * len(out["history"])
    assert out["history"][0]["failure_minibatches_skipped"] > 0


def test_unusable_curator_falls_back_to_proposals():
    cfg = small_config(reflection_minibatch_size=2, max_failure_minibatches=3, rollout_batch_size=20)
    llm = FakeLLM(ALL, curator=lambda p: "not json")
    out = evolve_skills(llm, EVOLUTION, VALIDATION, cfg, verbose=False)
    assert out["history"][0]["curator_parse_error"] is True
    assert len(out["library"]) >= 1          # fell back to the proposals (budget limits the count)


def test_only_evolution_and_validation_examples_are_ever_solved():
    seen_ids = set()
    inner = FakeLLM(ALL)
    by_input = {e["input"]: e["id"] for e in ALL}

    def spy(prompt):
        for question, eid in by_input.items():
            if question in prompt:
                seen_ids.add(eid)
        return inner(prompt)

    test_examples = make_examples(10, "te")
    inner.examples = ALL + test_examples
    by_input.update({e["input"]: e["id"] for e in test_examples})

    evolve_skills(spy, EVOLUTION, VALIDATION, small_config(), verbose=False)
    assert seen_ids and not any(i.startswith("te:") for i in seen_ids)


def test_parallel_workers_give_the_same_result():
    a = evolve_skills(FakeLLM(ALL), EVOLUTION, VALIDATION, small_config(max_workers=1), verbose=False)
    b = evolve_skills(FakeLLM(ALL), EVOLUTION, VALIDATION, small_config(max_workers=4), verbose=False)
    assert a["library"].hash() == b["library"].hash()
    assert [r["decision"] for r in a["history"]] == [r["decision"] for r in b["history"]]


def test_same_seed_is_reproducible_and_seed_changes_batches():
    def run(seed):
        return evolve_skills(FakeLLM(ALL), EVOLUTION, VALIDATION, small_config(seed=seed), verbose=False)

    a, b, c = run(0), run(0), run(1)
    assert [r["batch_ids"] for r in a["history"]] == [r["batch_ids"] for r in b["history"]]
    assert [r["batch_ids"] for r in a["history"]] != [r["batch_ids"] for r in c["history"]]


def test_existing_library_is_continued_not_overwritten():
    from src.skill_library import SkillLibrary
    start, _, _ = SkillLibrary().apply_ops([add_op(GOOD_SKILL)], 0, 3)
    llm = FakeLLM(ALL)
    out = evolve_skills(llm, EVOLUTION, VALIDATION, small_config(), library=start, verbose=False)
    assert out["validation_start"] == 1.0 and out["library"].ids() == ["skill_001"]
    assert len(start) == 1  # caller's object untouched


def test_prompt_contains_skills_only_when_given():
    llm = FakeLLM(ALL)
    prompts = []
    orig = llm._default_solver
    llm._solver = lambda p: (prompts.append(p), orig(p))[1]
    evaluate(llm, VALIDATION[:1], None)
    evaluate(llm, VALIDATION[:1], [GOOD_SKILL])
    assert "Learned reasoning skills" not in prompts[0]
    assert "Learned reasoning skills" in prompts[1] and GOOD_NAME in prompts[1]
    assert "applies_when" not in prompts[1] and "Puzzles that give pairwise" not in prompts[1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
