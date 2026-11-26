"""
Screen a candidate model BEFORE running skill evolution.

Skill evolution only works if the model makes mistakes on Logical Deduction
(failures are what skills are learned from), and the transfer experiment only
works if the model is neither at chance nor at ceiling on Tracking Shuffled
Objects. This script runs the NO-SKILLS baseline on:

    Logical Deduction          evolution  (150)
    Logical Deduction          validation (50)
    Tracking Shuffled Objects  dev        (50)

and prints accuracy, parse failures and reasoning failures overall and per
object count (3 / 5 / 7). Test splits are never touched.

Examples
--------
python experiments/screen_model.py --dry-run                       # no model, checks the pipeline
python experiments/screen_model.py --backend hf --model Qwen/Qwen2.5-7B-Instruct
python experiments/screen_model.py --backend hf --model meta-llama/Llama-3.1-8B-Instruct
python experiments/screen_model.py --backend gemini                # the earlier pilot, now on all splits
python experiments/screen_model.py --backend hf --model Qwen/Qwen2.5-3B-Instruct --limit 12   # quick smoke test

Output goes to results/screening/<model>.jsonl (full responses) and
results/screening/<model>.csv (one row per question). Re-running the same
command resumes: questions already in the .jsonl are not asked again.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline import build_prompt, extract_answer, is_correct
from src.dataset import load_split
from src.sampling import subset

# (family, split) pairs that are screened. Test splits are deliberately absent.
SCREEN_SPLITS = [
    ("logical_deduction", "evolution"),
    ("logical_deduction", "validation"),
    ("tracking_shuffled_objects", "dev"),
]
assert all(split != "test" for _, split in SCREEN_SPLITS)

CONFIGS = ("three_objects", "five_objects", "seven_objects")


class DryRunLLM:
    """Stand-in model: answers with a pseudo-random option, and sometimes with
    no FINAL ANSWER line at all. Only for checking that the pipeline runs."""

    description = "dry-run (fake model)"

    def batch(self, prompts):
        outputs = []
        for prompt in prompts:
            digest = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            letters = sorted(set(re.findall(r"\(([A-G])\)", prompt)))
            if digest % 33 == 0:
                outputs.append("I ran out of space while reasoning ...")
            else:
                outputs.append(f"Some reasoning.\nFINAL ANSWER: ({letters[digest % len(letters)]})")
        return outputs


def load_done(jsonl_path):
    records = {}
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["key"]] = record
    return records


def summarize(records):
    """Print the screening report and return the key numbers."""
    print("\n" + "=" * 72)
    print("SCREENING REPORT (no skills)")
    print("=" * 72)

    groups = {}
    for record in records:
        groups.setdefault((record["family"], record["split"]), []).append(record)

    def stats(rows):
        n = len(rows)
        correct = sum(r["correct"] for r in rows)
        parse = sum(r["parse_failed"] for r in rows)
        return n, correct, parse, n - correct - parse  # last = reasoning failures

    for (family, split), rows in sorted(groups.items()):
        n, correct, parse, reasoning = stats(rows)
        print(f"\n{family} / {split}: n={n}  accuracy={correct / n:.3f}  "
              f"reasoning failures={reasoning}  parse failures={parse}")
        for config in CONFIGS:
            sub = [r for r in rows if r["config"] == config]
            if sub:
                n_c, c_c, p_c, r_c = stats(sub)
                print(f"    {config:14s} n={n_c:3d}  acc={c_c / n_c:.3f}  "
                      f"reasoning failures={r_c:3d}  parse failures={p_c}")

    ld = [r for r in records if r["family"] == "logical_deduction"]
    tso = [r for r in records if r["family"] == "tracking_shuffled_objects"]
    everything = ld + tso

    # Parse failures: long responses with no answer line usually mean the
    # response was cut off by MAX_NEW_TOKENS.
    parse_rows = [r for r in everything if r["parse_failed"]]
    parse_rate = len(parse_rows) / len(everything) if everything else 0.0
    mean_all = statistics.mean(r["response_chars"] for r in everything) if everything else 0
    mean_parse = statistics.mean(r["response_chars"] for r in parse_rows) if parse_rows else 0
    seconds = statistics.mean(r["seconds"] for r in everything) if everything else 0

    print(f"\nParse failures overall: {len(parse_rows)}/{len(everything)} ({parse_rate:.1%}). "
          f"Mean response length: {mean_all:.0f} chars overall, {mean_parse:.0f} chars for parse failures.")
    print("  (parse failures much longer than average => responses were cut off; raise MAX_NEW_TOKENS)")
    print(f"Average time per question: {seconds:.1f} s")

    print("\nRule-of-thumb checks for choosing the primary model:")
    ld_acc = sum(r["correct"] for r in ld) / len(ld) if ld else None
    ld_fail = sum((not r["correct"]) and (not r["parse_failed"]) for r in ld if r["split"] == "evolution")
    tso_acc = sum(r["correct"] for r in tso) / len(tso) if tso else None

    def flag(ok, text):
        print(f"  [{'OK   ' if ok else 'CHECK'}] {text}")

    if ld_acc is not None:
        flag(0.40 <= ld_acc <= 0.75,
             f"Logical Deduction accuracy {ld_acc:.2f} (aim for roughly 0.40-0.75 so there are failures to learn from)")
        flag(ld_fail >= 30,
             f"{ld_fail} reasoning failures on the evolution split (more is better; fewer than ~30 gives the reflector little to work with)")
    if tso_acc is not None:
        flag(0.30 < tso_acc < 0.90,
             f"Tracking Shuffled Objects dev accuracy {tso_acc:.2f} (chance is about 0.225; needs room both above chance and below ceiling)")
    flag(parse_rate <= 0.05, f"parse-failure rate {parse_rate:.1%} (aim for 5% or less)")

    return {"ld_acc": ld_acc, "tso_acc": tso_acc, "parse_rate": parse_rate}


def write_csv(records, path):
    columns = ["key", "family", "split", "config", "id", "gold", "pred",
               "correct", "parse_failed", "response_chars", "seconds"]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default=None, help="gemini or hf (default: LLM_BACKEND env var, else gemini)")
    parser.add_argument("--model", default=None, help="model name / Hugging Face repo id")
    parser.add_argument("--dry-run", action="store_true", help="use a fake model (no GPU, no API)")
    parser.add_argument("--limit", type=int, default=None,
                        help="smoke test: only N questions per split, spread across 3/5/7 objects")
    parser.add_argument("--chunk", type=int, default=32, help="questions per saved chunk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "screening"))
    parser.add_argument("--tag", default="", help="extra text for the output file name")
    args = parser.parse_args()

    items = []
    for family, split in SCREEN_SPLITS:
        examples = load_split(family, split)
        if args.limit:
            examples = subset(examples, args.limit, args.seed)
        items.extend((family, split, example) for example in examples)

    if args.dry_run:
        llm = DryRunLLM()
    else:
        from src.llm import get_llm
        llm = get_llm(args.model, args.backend)

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", llm.description)
    if args.limit:
        slug += f"_limit{args.limit}"
    if args.tag:
        slug += f"_{args.tag}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{slug}.jsonl"
    csv_path = out_dir / f"{slug}.csv"

    done = load_done(jsonl_path)
    todo = [(f, s, e) for f, s, e in items if f"{s}|{e['id']}" not in done]
    print(f"Model: {llm.description}")
    print(f"{len(items)} questions in total, {len(done)} already done, {len(todo)} to run.")

    started = time.time()
    with open(jsonl_path, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            prompts = [build_prompt(example["input"]) for _, _, example in chunk]

            t0 = time.time()
            responses = llm.batch(prompts)
            per_item = (time.time() - t0) / len(chunk)

            for (family, split, example), response in zip(chunk, responses):
                prediction = extract_answer(response)
                record = {
                    "key": f"{split}|{example['id']}",
                    "family": family,
                    "split": split,
                    "config": example["config"],
                    "id": example["id"],
                    "gold": example["target"],
                    "pred": prediction,
                    "correct": bool(is_correct(prediction, example["target"])),
                    "parse_failed": prediction is None,
                    "response_chars": len(response),
                    "seconds": round(per_item, 2),
                    "response": response,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[record["key"]] = record

            out.flush()
            print(f"  {min(start + args.chunk, len(todo))}/{len(todo)} done "
                  f"({time.time() - started:.0f} s elapsed)", flush=True)

    # Report on exactly the questions requested in this run.
    wanted = {f"{s}|{e['id']}" for _, s, e in items}
    records = [done[key] for key in done if key in wanted]
    write_csv(records, csv_path)
    summarize(records)
    print(f"\nPer-question results: {csv_path}\nFull responses:      {jsonl_path}")


if __name__ == "__main__":
    main()
