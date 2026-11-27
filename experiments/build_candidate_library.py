"""
Collect every skill proposed during evolution (including the ones the gate
rejected) into one deduplicated, leakage-filtered "ungated" library.

Example (on SOL, from the repo root):
  python experiments/build_candidate_library.py \
      --logs "skills/evolution_log_Qwen_Qwen2.5-3B-Instruct_v*.jsonl" \
      --gated "skills/library_Qwen_Qwen2.5-3B-Instruct_v*.json" \
      --out skills/library_ungated_qwen3b.json

Writes the library plus <out>.provenance.json (where each skill came from,
and whether it passed the gate). Calls no model.
"""
from pathlib import Path
import argparse
import glob
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.candidates import build_library, collect_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="evolution log files or globs (.jsonl)")
    parser.add_argument("--gated", nargs="*", default=[], help="final library files or globs (skills that passed the gate)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threshold", type=float, default=0.5, help="word-overlap above which two skills are duplicates")
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument("--no-leak-filter", action="store_true", help="skip the leakage filter (testing only)")
    args = parser.parse_args()

    logs = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    gated = sorted({p for pattern in args.gated for p in glob.glob(pattern)})

    if not logs:
        sys.exit("No log files matched --logs")

    print(f"{len(logs)} log files, {len(gated)} gated library files")

    leak_check = None
    if not args.no_leak_filter:
        from src.dataset import load_split
        from src.leakage import LeakageFilter
        questions = [e["input"] for e in load_split("logical_deduction", "evolution")]
        leak_check = LeakageFilter(questions).check

    cands = collect_candidates(logs, gated)
    lib, provenance, report = build_library(cands, leak_check, args.threshold, args.max_skills)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lib.save(out)
    prov_path = out.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps({"report": report, "skills": provenance}, indent=2), encoding="utf-8")

    print(f"candidates found: {report['candidates_found']}   distinct groups: {report['distinct_clusters']}")
    print(f"skills kept: {report['skills_kept']} (passed the gate: {report['gated_kept']})   refused: {len(report['refused'])}")
    for item in report["refused"][:8]:
        print(f"  refused: {item['name'][:60]} -> {item['reason']}")
    print(f"library: {out}\nprovenance: {prov_path}")


if __name__ == "__main__":
    main()
