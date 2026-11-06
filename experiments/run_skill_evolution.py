"""
Run skill evolution on Logical Deduction.

  python experiments/run_skill_evolution.py --dry-run     # fake model, runs anywhere
  python experiments/run_skill_evolution.py               # real model (see below)

Real runs need a model hook-up: create src/llm.py with a function

    def get_llm():
        '''Return a function llm(prompt: str) -> str  (greedy decoding).'''

The dry run uses src/fake_llm.py and writes to skills/_dry_run/, never to
skills/skill_library.json.
"""

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_split
from src.skill_evolution import (
    EvolutionConfig, accuracy, calibrate_noise, evaluate, evolve_skills,
    paired_gate,
)

FAMILY = "logical_deduction"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-evolution", type=int, default=None,
                        help="use only the first N evolution examples (smoke test)")
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--noise-repeats", type=int, default=3,
                        help="re-solve validation this many times with no skills first (0 = skip)")
    parser.add_argument("--eval-test", action="store_true",
                        help="after evolving, compare no-skills vs evolved library on the Logical Deduction TEST split")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    evolution = load_split(FAMILY, "evolution")
    validation = load_split(FAMILY, "validation")
    test = load_split(FAMILY, "test")

    if args.limit_evolution:
        evolution = evolution[:args.limit_evolution]
    if args.limit_validation:
        validation = validation[:args.limit_validation]

    if args.dry_run:
        from src.fake_llm import FakeLLM
        llm = FakeLLM(evolution + validation + test)
        out_dir = PROJECT_ROOT / "skills" / "_dry_run"
    else:
        try:
            from src.llm import get_llm
        except ImportError:
            sys.exit(
                "No model hook-up yet: create src/llm.py with get_llm() "
                "(see the docstring at the top of this file), or use --dry-run."
            )
        llm = get_llm()
        out_dir = PROJECT_ROOT / "skills"

    out_path = Path(args.out) if args.out else out_dir / "skill_library.json"
    log_path = Path(args.log) if args.log else out_dir / "evolution_log.jsonl"

    config = EvolutionConfig(
        epochs=args.epochs,
        rollout_batch_size=args.batch_size,
        seed=args.seed,
        max_workers=args.workers,
    )

    print(f"Evolution: {len(evolution)}  Validation: {len(validation)}  Test (not used for evolving): {len(test)}")

    if args.noise_repeats:
        noise = calibrate_noise(llm, validation, args.noise_repeats, args.workers)
        print("\nStep 0 - validation with NO skills, repeated:")
        print("  accuracies:", [round(a, 3) for a in noise["accuracies"]])
        print(f"  questions whose outcome flipped between runs: {noise['flipped_questions']} of {noise['n']}")
        print("  parse failures per run:", noise["parse_failures"])

        base = evaluate(llm, evolution, None, args.workers)
        n_fail = sum(1 for r in base if not r["correct"] and not r["parse_failed"])
        print(f"  evolution-set accuracy with no skills: {accuracy(base):.3f} "
              f"({n_fail} reasoning failures available as feedback)\n")

    result = evolve_skills(
        llm, evolution, validation, config,
        log_path=log_path, checkpoint_path=out_path,
    )

    result["library"].save(out_path)

    print(f"\nStopped: {result['stopped']}")
    print(f"Validation accuracy: {result['validation_start']:.3f} -> {result['validation_final']:.3f}")
    print(f"Skills kept: {len(result['library'])}   Model calls: {result['llm_calls']}")
    print(f"Library: {out_path}\nLog: {log_path}")

    if args.eval_test:
        without = evaluate(llm, test, None, args.workers)
        with_skills = evaluate(llm, test, result["library"].skills, args.workers)
        gate = paired_gate(without, with_skills)
        print("\nIn-domain check on the Logical Deduction TEST split:")
        print(f"  no skills : {accuracy(without):.3f}")
        print(f"  evolved   : {accuracy(with_skills):.3f}   (repaired {gate['repaired']}, broken {gate['broken']})")


if __name__ == "__main__":
    main()
