"""
Transfer metrics, all paired against the no-skills run on the same questions.

  accuracy          fraction correct (parse failures count as wrong)
  transfer_gain     accuracy(condition) - accuracy(no_skills)
  positive transfer repaired: wrong without skills, right with them
  negative transfer broken:   right without skills, wrong with them
  p_value           exact two-sided sign test on (repaired, broken): how likely
                    a split this uneven is if skills changed answers at random
  cost              model calls, prompt characters (a proxy for tokens), seconds
"""
import math
import threading


class MeteredLLM:
    """Wraps the model; counts calls and prompt characters (thread-safe)."""

    def __init__(self, llm):
        self._llm = llm
        self.calls = 0
        self.prompt_chars = 0
        self._lock = threading.Lock()
        if hasattr(llm, "batch"):
            self.batch = self._batch

    def __call__(self, prompt):
        with self._lock:
            self.calls += 1
            self.prompt_chars += len(prompt)
        return self._llm(prompt)

    def _batch(self, prompts):
        prompts = list(prompts)
        with self._lock:
            self.calls += len(prompts)
            self.prompt_chars += sum(len(p) for p in prompts)
        return self._llm.batch(prompts)

    def snapshot(self):
        return {"calls": self.calls, "prompt_chars": self.prompt_chars}


def accuracy(results):
    return sum(r["correct"] for r in results) / len(results) if results else 0.0


def paired_counts(base, other):
    repaired = sum(1 for b, o in zip(base, other) if not b["correct"] and o["correct"])
    broken = sum(1 for b, o in zip(base, other) if b["correct"] and not o["correct"])
    return repaired, broken


def sign_test_p(repaired, broken):
    """Exact two-sided binomial (p=0.5) on the discordant pairs."""
    n = repaired + broken
    if n == 0:
        return 1.0
    k = min(repaired, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(results_by_condition, costs=None, baseline="no_skills"):
    """One row per condition. `costs`: {condition: {"calls", "prompt_chars", "seconds"}}."""
    base = results_by_condition[baseline]
    base_acc = accuracy(base)
    rows = {}

    for name, results in results_by_condition.items():
        acc = accuracy(results)
        repaired, broken = paired_counts(base, results)
        rows[name] = {
            "n": len(results),
            "accuracy": round(acc, 4),
            "transfer_gain": round(acc - base_acc, 4),
            "repaired": repaired,
            "broken": broken,
            "p_value": round(sign_test_p(repaired, broken), 4),
            "parse_failures": sum(r["parse_failed"] for r in results),
            "mean_skills_shown": round(
                sum(len(r.get("skill_ids", [])) for r in results) / max(1, len(results)), 2),
            "cost": (costs or {}).get(name, {}),
        }
    return rows


def format_table(rows):
    header = (f"{'condition':<15}{'acc':>7}{'gain':>8}{'repaired':>10}{'broken':>8}"
              f"{'p':>7}{'parse_fail':>11}{'skills':>8}{'calls':>8}{'sec':>7}")
    lines = [header, "-" * len(header)]
    for name, r in rows.items():
        c = r.get("cost", {})
        lines.append(
            f"{name:<15}{r['accuracy']:>7.3f}{r['transfer_gain']:>+8.3f}{r['repaired']:>10}"
            f"{r['broken']:>8}{r['p_value']:>7.2f}{r['parse_failures']:>11}"
            f"{r['mean_skills_shown']:>8.2f}{c.get('calls', 0):>8}{c.get('seconds', 0):>7.0f}")
    return "\n".join(lines)
