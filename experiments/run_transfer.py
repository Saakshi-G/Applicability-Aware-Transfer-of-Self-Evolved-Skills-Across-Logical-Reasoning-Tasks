"""
Transfer experiment: take a skill library learned on Logical Deduction and
test it on Tracking Shuffled Objects under five conditions
(no skills / all skills / random / similarity / applicability-aware).

Example on SOL (GPU needed):
  sbatch scripts/transfer_sol.sbatch Qwen/Qwen2.5-3B-Instruct ungated \
      --library skills/library_ungated_qwen3b.json

Local smoke test (no model, no GPU):
  python experiments/run_transfer.py --library skills/library_ungated_qwen3b.json --dry-run --limit 10

Writes to results/transfer/<tag>/:
  summary.json          metrics per condition
  per_question.jsonl    prediction, correctness and skills shown, per condition
  applicability.jsonl   the model's verdict for every (question, skill) pair
"""
from pathlib import Path
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_split
from src.evaluation import MeteredLLM, format_table, summarize
from src.retrieval import DEFAULT_EMBEDDING_MODEL, make_scorer
from src.skill_library import SkillLibrary
from src.transfer import CONDITIONS, PromptCache, StubLLM, select_skills, solve_with_selection

FAMILY = "tracking_shuffled_objects"


def _hash_embed(texts, dim=64):
    """Stand-in embedder for --dry-run only (no model download)."""
    import hashlib
    out = []
    for text in texts:
        vec = [0.0] * dim
        for w in text.lower().split():
            vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        out.append(vec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="skill library JSON (from run_skill_evolution or build_candidate_library)")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--family", default=FAMILY, choices=["tracking_shuffled_objects", "logical_deduction"],
                        help="question set to solve (logical_deduction = the skills' own domain, a control)")
    parser.add_argument("--split", default="test", choices=["dev", "validation", "evolution", "test"])
    parser.add_argument("--k", type=int, default=3, help="skills shown by random / similarity / applicability")
    parser.add_argument("--pool", type=int, default=None, help="applicability: judge only the N most similar skills (default: all)")
    parser.add_argument("--retriever", default="tfidf", choices=["tfidf", "embedding"],
                        help="similarity scorer for the similarity condition and for ranking in applicability: "
                             "tfidf = lexical (default), embedding = semantic sentence embeddings")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--judge-model", default=None,
                        help="a separate (usually stronger) model that makes the applicability YES/NO judgments; "
                             "the solver stays the main model. Default: the main model judges itself.")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random-skills condition")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N questions (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="stub model, no GPU")
    args = parser.parse_args()

    examples = load_split(args.family, args.split)
    if args.limit:
        examples = examples[:args.limit]

    library = SkillLibrary.load(args.library)
    skills = library.skills
    if not skills:
        sys.exit(f"{args.library} has no skills")

    if "no_skills" not in args.conditions:
        args.conditions = ["no_skills"] + args.conditions   # every metric is paired against it

    if args.dry_run:
        raw_llm, description = StubLLM(), "stub (dry run)"
    else:
        from src.llm import get_llm
        raw_llm = get_llm()
        description = getattr(raw_llm, "description", "model")

    llm = MeteredLLM(raw_llm)

    if args.judge_model:
        if args.dry_run:
            judge_raw, judge_description = StubLLM(), "stub judge (dry run)"
        else:
            judge_raw = get_llm(args.judge_model)
            judge_description = getattr(judge_raw, "description", args.judge_model)
        judge = MeteredLLM(judge_raw)
    else:
        judge, judge_description = llm, description + " (same as solver)"
    out_dir = PROJECT_ROOT / "results" / "transfer" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {description}")
    print(f"applicability judge: {judge_description}")
    print(f"library: {args.library} ({len(skills)} skills)   target: {args.family}/{args.split} ({len(examples)} questions)   k={args.k}")

    embed_fn = _hash_embed if (args.dry_run and args.retriever == "embedding") else None
    scorer = make_scorer(args.retriever, skills, model_name=args.embedding_model, embed_fn=embed_fn)
    print(f"retriever: {args.retriever}" + (f" ({args.embedding_model})" if args.retriever == "embedding" and not args.dry_run else ""))
    cache = PromptCache()
    results, costs, applic_rows = {}, {}, []

    with open(out_dir / "per_question.jsonl", "w", encoding="utf-8") as per_q:
        for condition in args.conditions:
            before, t0, hits0 = llm.snapshot(), time.perf_counter(), cache.hits
            judge_before = judge.snapshot()

            selected, details = select_skills(
                condition, examples, skills, k=args.k, seed=args.seed,
                llm=judge, pool=args.pool, scorer=scorer,
            )
            res = solve_with_selection(llm, examples, selected, cache)

            after = llm.snapshot()
            costs[condition] = {
                "calls": after["calls"] - before["calls"],
                "prompt_chars": after["prompt_chars"] - before["prompt_chars"],
                "seconds": round(time.perf_counter() - t0, 1),
                "prompts_reused": cache.hits - hits0,
            }
            if judge is not llm:
                costs[condition]["judge_calls"] = judge.snapshot()["calls"] - judge_before["calls"]
            results[condition] = res

            for r, d in zip(res, details):
                per_q.write(json.dumps({"condition": condition, **r, "details": {k: v for k, v in d.items() if k != "judged"}}, ensure_ascii=False) + "\n")
                if "judged" in d:
                    applic_rows.append({"id": r["id"], "judged": d["judged"], "kept": d["kept"]})

            acc = sum(r["correct"] for r in res) / len(res)
            print(f"  {condition:<14} accuracy {acc:.3f}   calls {costs[condition]['calls']}   reused {costs[condition]['prompts_reused']}   {costs[condition]['seconds']:.0f}s", flush=True)

    with open(out_dir / "applicability.jsonl", "w", encoding="utf-8") as f:
        for row in applic_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows = summarize(results, costs)
    (out_dir / "summary.json").write_text(json.dumps({
        "library": args.library, "n_skills": len(skills), "target": f"{args.family}/{args.split}",
        "k": args.k, "seed": args.seed, "model": description, "retriever": args.retriever, "judge_model": judge_description, "conditions": rows,
    }, indent=2), encoding="utf-8")

    print("\n" + format_table(rows))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
