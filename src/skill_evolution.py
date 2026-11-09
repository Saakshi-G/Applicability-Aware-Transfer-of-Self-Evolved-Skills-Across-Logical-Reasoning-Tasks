"""
Skill evolution: turn Logical Deduction experience into a skill library.

The LLM is frozen. Skills are learned by a validation-gated loop:

  for each round (a batch of evolution examples):
    1. SOLVE   the batch with the current library (the "incumbent").
    2. REFLECT failures (in minibatches) + a few successes for contrast ->
               a few proposed library edits (ADD / EDIT / MERGE / REMOVE).
    3. CURATE  merge the proposals, drop duplicates (only when needed).
    4. FILTER  refuse edits that leak example-specific content, break size
               limits, or exceed the round's edit budget.
    5. GATE    solve the fixed validation set with the candidate library and
               accept it only if  repairs - 2 * regressions > 0  (paired,
               per question). Otherwise keep the incumbent and remember the
               rejected edit so the reflector does not propose it again.

Design sources (see THIRD_PARTY_NOTICES.md; all numbers/rules below are as
reported in those papers, adapted to a small multi-skill JSON library):
  * SkillOpt (Yang et al. 2026, arXiv:2605.23904): minibatch reflection over
    failures AND successes (failures take priority), small edit budget per
    step, hierarchical merge, candidate hashing, rejected-edit buffer,
    per-step reports. Reflection minibatch 8.
  * Rethinking Self-Evolving Agent Skills (Liu et al. 2026, arXiv:2608.02636):
    <= 10 rounds, stop after 5 consecutive rejected/no-op rounds, failures
    are essential (success-only feedback invents incidental rules), and
    validation noise is large (SD ~3.9 points on 49 items).
  * Do Self-Evolving Skills Generalize to Held-Out Tasks? (Piao et al. 2026,
    arXiv:2609.39148): paired acceptance u = |R| - 2|B| > 0 (regressions cost
    double), and the "never encode literal answers / one-task exceptions"
    rule.
  * ExpeL (Zhao et al. 2024): the ADD / EDIT / REMOVE-style operation set.

The model is any function `llm(prompt: str) -> str`. Use decoding with
temperature 0 for every call so that validation scores are repeatable.
"""

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from src.baseline import build_prompt, extract_answer, is_correct
from src.leakage import LeakageFilter
from src.skill_library import SkillLibrary, lexical_similarity, skill_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFLECT_PROMPT_PATH = PROJECT_ROOT / "prompts" / "evolve_skill.txt"
MERGE_PROMPT_PATH = PROJECT_ROOT / "prompts" / "merge_skills.txt"

TASK_DESCRIPTION = (
    "Logical Deduction: several objects are arranged in a fixed order "
    "(for example left to right, or by price or finishing position). A few "
    "statements give pairwise or positional constraints, and the question "
    "asks which answer option is consistent with them."
)


@dataclass
class EvolutionConfig:
    epochs: int = 2                      # passes over the evolution set
    rollout_batch_size: int = 30         # examples solved per round
    reflection_minibatch_size: int = 8   # failures per reflector call (SkillOpt)
    max_failure_minibatches: int = 3     # cap on reflector calls per round
    contrast_successes: int = 8          # successes shown for contrast
    ops_per_round: tuple = (3, 2)        # edit budget per epoch; last value reused
    max_skills: int = 20
    max_total_words: int = 1200
    patience: int = 5                    # stop after this many rejected/no-op rounds in a row
    regression_penalty: int = 2          # u = repaired - penalty * broken
    dedup_threshold: float = 0.5         # similarity above which a pair is flagged
    attempt_chars_failed: int = 2400     # reasoning shown to the reflector (head + tail)
    attempt_chars_success: int = 800
    rejected_buffer_size: int = 5
    seed: int = 0
    max_workers: int = 1                 # parallel solver calls (only if the model wrapper is thread-safe)
    source_task: str = "logical_deduction"
    task_description: str = TASK_DESCRIPTION


# ---------------------------------------------------------------- small helpers


class CountingLLM:
    """Wraps the model function and counts calls (thread-safe)."""

    def __init__(self, llm):
        self._llm = llm
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, prompt):
        with self._lock:
            self.calls += 1
        return self._llm(prompt)


def fill_template(template, values):
    """
    Replace {NAME} placeholders in ONE pass. Inserted text is never
    re-scanned, so braces inside questions or model outputs are safe.
    """
    return re.sub(
        r"\{(\w+)\}",
        lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
        template,
    )


def extract_json(text):
    """
    Find the JSON object in an LLM reply. Tolerates code fences and extra
    prose around it. Prefers an object that has an "ops" key.
    Returns a dict or None.
    """
    if not text:
        return None

    decoder = json.JSONDecoder()
    first_dict = None

    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            if "ops" in obj:
                return obj
            if first_dict is None:
                first_dict = obj

    return first_dict


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _shorten(text, limit):
    """Keep the start and the end of long reasoning (the answer is at the end)."""
    text = (text or "").strip()

    if len(text) <= limit:
        return text

    half = limit // 2
    return text[:half] + "\n[... shortened ...]\n" + text[-half:]


# --------------------------------------------------------------------- solving


def _solve_example(llm, example, skills=None):
    """
    Build the prompt (same template for every condition), ask the model,
    parse the answer, compare with gold.
    """
    raw = llm(build_prompt(example["input"], skills))
    prediction = extract_answer(raw)

    return {
        "id": example["id"],
        "input": example["input"],
        "target": example["target"],
        "prediction": prediction,
        "correct": is_correct(prediction, example["target"]),
        "parse_failed": prediction is None,
        "raw": raw,
    }


def evaluate(llm, examples, skills=None, max_workers=1):
    """Solve every example; results keep the order of `examples`."""

    def one(example):
        return _solve_example(llm, example, skills)

    if max_workers <= 1:
        return [one(e) for e in examples]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, examples))


def accuracy(results):
    if not results:
        return 0.0
    return sum(r["correct"] for r in results) / len(results)


def paired_gate(incumbent_results, candidate_results, penalty=2):
    """
    Paired acceptance rule (Piao et al. 2026): compare the SAME validation
    questions under the incumbent and the candidate library.

      R = repaired    (incumbent wrong, candidate right)
      B = broken      (incumbent right, candidate wrong)
      u = R - penalty * B      accept only if u > 0

    u > 0 means R > penalty * B >= 0, so the candidate's accuracy is strictly
    higher; "accuracy does not drop" is therefore implied. Regressions cost
    double because a skill that breaks questions the model used to solve is
    exactly negative transfer.
    """
    repaired = sum(
        1 for i, c in zip(incumbent_results, candidate_results)
        if not i["correct"] and c["correct"]
    )
    broken = sum(
        1 for i, c in zip(incumbent_results, candidate_results)
        if i["correct"] and not c["correct"]
    )
    utility = repaired - penalty * broken

    return {
        "repaired": repaired,
        "broken": broken,
        "utility": utility,
        "accept": utility > 0,
    }


def calibrate_noise(llm, examples, repeats=3, max_workers=1):
    """
    Step 0 sanity check, run BEFORE evolving: solve the same examples with an
    EMPTY library several times. If outcomes change between runs, validation
    scores are noisy and small gains cannot be trusted. Returns accuracies,
    the number of questions whose outcome flipped, and parse failures.
    """
    runs = [evaluate(llm, examples, None, max_workers) for _ in range(repeats)]

    flipped = sum(
        1 for outcomes in zip(*[[r["correct"] for r in run] for run in runs])
        if len(set(outcomes)) > 1
    )

    return {
        "accuracies": [accuracy(run) for run in runs],
        "flipped_questions": flipped,
        "parse_failures": [sum(r["parse_failed"] for r in run) for run in runs],
        "n": len(examples),
    }


# ------------------------------------------------------------------ reflection


def _format_attempts(results, char_limit, with_gold=True):
    blocks = []

    for index, r in enumerate(results, start=1):
        answer = r["prediction"] if r["prediction"] else "NONE"
        lines = [
            f"--- Attempt {index} ---",
            "Question:",
            r["input"].strip(),
            "Model reasoning:",
            _shorten(r["raw"], char_limit),
            f"Model answer: {answer}",
        ]
        if with_gold:
            lines.append(f"Gold answer: {r['target']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "(none)"


def _format_rejected(buffer):
    if not buffer:
        return "(none)"

    return "\n".join(
        f"- Round {b['round']}: {b['summary']} -> repaired {b['repaired']}, "
        f"broke {b['broken']} validation answers (rejected)"
        for b in buffer
    )


def _summarize_ops(applied):
    parts = []
    for a in applied:
        label = a.get("name") or ",".join(a.get("target_ids", []))
        parts.append(f"{a['op']} '{label}'")
    return "; ".join(parts) if parts else "(no ops)"


def _flag_duplicates(library, proposals, threshold, similarity_fn):
    """
    Text lines describing possible duplicates among library skills and
    proposed skills. Hints for the curator only.
    """
    lines = []

    for a, b, score in library.near_duplicates(threshold, similarity_fn):
        lines.append(f"library {a} ~ library {b} (similarity {score:.2f})")

    flat = [op for batch in proposals for op in batch]

    for i, op in enumerate(flat, start=1):
        skill = op.get("skill") if isinstance(op, dict) else None
        if not isinstance(skill, dict):
            continue

        targets = set(map(str, op.get("target_ids") or []))

        for existing in library.skills:
            if existing["skill_id"] in targets:
                continue
            score = similarity_fn(skill_text(skill), skill_text(existing))
            if score >= threshold:
                lines.append(
                    f"proposal #{i} '{skill.get('name', '')}' ~ library "
                    f"{existing['skill_id']} (similarity {score:.2f})"
                )

        for j in range(i, len(flat)):
            other = flat[j].get("skill") if isinstance(flat[j], dict) else None
            if isinstance(other, dict):
                score = similarity_fn(skill_text(skill), skill_text(other))
                if score >= threshold:
                    lines.append(
                        f"proposal #{i} ~ proposal #{j + 1} (similarity {score:.2f})"
                    )

    return lines


def reflect_and_curate(llm, library, failures, successes, op_budget, config,
                       rejected_buffer, rng, similarity_fn, record):
    """
    Steps 2-3: ask the reflector for edits (one call per failure minibatch),
    then call the curator only if it is needed. Returns the final op list.
    Fills `record` with what happened (for the log).
    """
    reflect_template = REFLECT_PROMPT_PATH.read_text(encoding="utf-8")
    merge_template = MERGE_PROMPT_PATH.read_text(encoding="utf-8")

    shuffled = list(failures)
    rng.shuffle(shuffled)
    minibatches = _chunks(shuffled, config.reflection_minibatch_size)
    skipped_minibatches = max(0, len(minibatches) - config.max_failure_minibatches)
    minibatches = minibatches[:config.max_failure_minibatches]

    record["failure_minibatches"] = len(minibatches)
    record["failure_minibatches_skipped"] = skipped_minibatches

    proposals = []
    summaries = []
    parse_errors = 0

    for minibatch in minibatches:
        pool = list(successes)
        rng.shuffle(pool)
        contrast = pool[:config.contrast_successes]

        prompt = fill_template(reflect_template, {
            "TASK_DESCRIPTION": config.task_description,
            "L": op_budget,
            "CAP": config.max_skills,
            "LIBRARY": library.to_prompt_json(),
            "REJECTED": _format_rejected(rejected_buffer),
            "FAILURES": _format_attempts(minibatch, config.attempt_chars_failed),
            "SUCCESSES": _format_attempts(
                contrast, config.attempt_chars_success, with_gold=False
            ),
        })

        obj = extract_json(llm(prompt))
        ops = obj.get("ops") if isinstance(obj, dict) else None

        if not isinstance(ops, list):
            parse_errors += 1
            proposals.append([])
            continue

        proposals.append([op for op in ops if isinstance(op, dict)])

        if isinstance(obj.get("failure_summary"), list):
            summaries.extend(obj["failure_summary"])

    record["reflector_parse_errors"] = parse_errors
    record["failure_summary"] = summaries
    record["proposed_ops"] = proposals

    flat = [op for batch in proposals for op in batch]

    if not flat:
        return []

    flagged = _flag_duplicates(
        library, proposals, config.dedup_threshold, similarity_fn
    )
    record["duplicate_flags"] = flagged

    needs_curator = len([b for b in proposals if b]) > 1 or flagged or len(flat) > op_budget

    if not needs_curator:
        return flat

    numbered = [
        {"proposal": i, "batch": b + 1, **op}
        for i, (b, op) in enumerate(
            ((b, op) for b, batch in enumerate(proposals) for op in batch), start=1
        )
    ]

    prompt = fill_template(merge_template, {
        "L": op_budget,
        "CAP": config.max_skills,
        "LIBRARY": library.to_prompt_json(),
        "PROPOSALS": json.dumps(numbered, indent=2, ensure_ascii=False),
        "DUPLICATES": "\n".join(flagged) if flagged else "(none flagged)",
    })

    obj = extract_json(llm(prompt))
    curated = obj.get("ops") if isinstance(obj, dict) else None

    record["curator_called"] = True

    if isinstance(curated, list):
        return [op for op in curated if isinstance(op, dict)]

    # Curator output unusable: fall back to the best-supported proposals.
    record["curator_parse_error"] = True

    def support(op):
        try:
            return int(op.get("support_count", 1))
        except (TypeError, ValueError):
            return 1

    return sorted(flat, key=support, reverse=True)


# ------------------------------------------------------------------- main loop


def evolve_skills(
    llm,
    evolution_examples,
    validation_examples,
    config=None,
    library=None,
    log_path=None,
    checkpoint_path=None,
    leakage_filter=None,
    similarity_fn=lexical_similarity,
    verbose=True,
):
    """
    Run the evolution loop. Returns a dict:
      library          final SkillLibrary (the incumbent; it is also the best
                       library, because acceptance requires a strict
                       validation improvement every time)
      history          one record per round (also written to `log_path`)
      validation_start / validation_final   validation accuracy
      llm_calls        total model calls
      stopped          why the loop ended

    Examples are dicts with "id", "input", "target" (see dataset.load_split).
    The evolution set may be used for reflection; the validation set only
    for the accept/reject decision; never pass the test set here.
    """
    cfg = config or EvolutionConfig()
    llm = CountingLLM(llm)

    lib = library.copy() if library is not None else SkillLibrary(
        source_task=cfg.source_task,
        max_skills=cfg.max_skills,
        max_total_words=cfg.max_total_words,
    )

    leak = leakage_filter or LeakageFilter([e["input"] for e in evolution_examples])

    log_file = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    incumbent_val = evaluate(llm, validation_examples, lib.skills, cfg.max_workers)
    val_start = accuracy(incumbent_val)
    val_cache = {lib.hash(): incumbent_val}

    if verbose:
        print(f"Start: validation accuracy {val_start:.3f} "
              f"({sum(r['correct'] for r in incumbent_val)}/{len(validation_examples)})")

    ordered = sorted(evolution_examples, key=lambda e: e["id"])
    history = []
    round_idx = 0
    quiet_rounds = 0
    stopped = "all epochs finished"

    try:
        for epoch in range(cfg.epochs):
            order = list(ordered)
            random.Random(cfg.seed + epoch).shuffle(order)
            rng = random.Random(cfg.seed * 1000 + epoch)
            op_budget = cfg.ops_per_round[min(epoch, len(cfg.ops_per_round) - 1)]
            rejected_buffer = []  # epoch-local, as in SkillOpt

            for batch in _chunks(order, cfg.rollout_batch_size):
                round_idx += 1
                calls_before = llm.calls

                record = {
                    "round": round_idx,
                    "epoch": epoch + 1,
                    "op_budget": op_budget,
                    "batch_ids": [e["id"] for e in batch],
                    "library_hash_before": lib.hash(),
                    "library_size_before": len(lib),
                }

                results = evaluate(llm, batch, lib.skills, cfg.max_workers)
                failures = [r for r in results if not r["correct"] and not r["parse_failed"]]
                successes = [r for r in results if r["correct"]]

                record.update(
                    batch_accuracy=accuracy(results),
                    n_failures_reflected=len(failures),
                    n_parse_failures=sum(r["parse_failed"] for r in results),
                    n_successes=len(successes),
                )

                decision, reason = "no-op", ""
                applied, dropped = [], []

                if not failures:
                    # Failures are essential feedback (Liu et al. 2026); with
                    # none there is nothing to learn from this batch.
                    reason = "no reasoning failures in this batch"
                else:
                    ops = reflect_and_curate(
                        llm, lib, failures, successes, op_budget, cfg,
                        rejected_buffer, rng, similarity_fn, record,
                    )
                    candidate, applied, dropped = lib.apply_ops(
                        ops, round_idx, op_budget, leak.check
                    )

                    if not applied:
                        reason = "no valid ops after filtering"
                    elif candidate.hash() == lib.hash():
                        reason = "candidate identical to incumbent"
                        applied = []
                    else:
                        key = candidate.hash()

                        if key not in val_cache:
                            val_cache[key] = evaluate(
                                llm, validation_examples, candidate.skills,
                                cfg.max_workers,
                            )

                        gate = paired_gate(
                            incumbent_val, val_cache[key], cfg.regression_penalty
                        )
                        record["gate"] = gate
                        record["candidate_val_accuracy"] = accuracy(val_cache[key])

                        if gate["accept"]:
                            decision = "accepted"
                            lib = candidate
                            incumbent_val = val_cache[key]
                            if checkpoint_path:
                                lib.save(checkpoint_path)
                        else:
                            decision = "rejected"
                            reason = (
                                f"u = {gate['repaired']} - "
                                f"{cfg.regression_penalty}*{gate['broken']} = "
                                f"{gate['utility']} <= 0"
                            )
                            rejected_buffer.append({
                                "round": round_idx,
                                "summary": _summarize_ops(applied),
                                "repaired": gate["repaired"],
                                "broken": gate["broken"],
                            })
                            rejected_buffer = rejected_buffer[-cfg.rejected_buffer_size:]

                record.update(
                    decision=decision,
                    reason=reason,
                    applied_ops=applied,
                    dropped_ops=dropped,
                    library_hash_after=lib.hash(),
                    library_size_after=len(lib),
                    validation_accuracy=accuracy(incumbent_val),
                    llm_calls_this_round=llm.calls - calls_before,
                )
                history.append(record)

                if log_file:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()

                if verbose:
                    print(
                        f"Round {round_idx:2d} (epoch {epoch + 1}): "
                        f"batch acc {record['batch_accuracy']:.2f}, "
                        f"{decision:8s} val acc {record['validation_accuracy']:.3f}, "
                        f"{len(lib)} skills"
                        + (f"  [{reason}]" if reason else "")
                    )

                quiet_rounds = 0 if decision == "accepted" else quiet_rounds + 1

                if quiet_rounds >= cfg.patience:
                    stopped = f"{cfg.patience} rounds in a row without an accepted edit"
                    break

            if stopped != "all epochs finished":
                break
    finally:
        if log_file:
            log_file.close()

    return {
        "library": lib,
        "history": history,
        "validation_start": val_start,
        "validation_final": accuracy(incumbent_val),
        "llm_calls": llm.calls,
        "stopped": stopped,
    }
