"""Add three options to experiments/run_skill_evolution.py (run AFTER patch_reflector.py):
  --reflect-prompt PATH   reflector prompt file (relative to the repo root)
  --merge-prompt PATH     curator prompt file
  --extra-validation N    move the last N evolution questions into the validation set (bigger, less noisy gate)
Run from the repo root:  python scripts/patch_v3.py     (safe to run twice)"""
import pathlib, sys

def sub(text, old, new, label):
    if old not in text:
        sys.exit(f"anchor not found for: {label} - send me the file so I can adjust")
    return text.replace(old, new, 1)

p = pathlib.Path("experiments/run_skill_evolution.py"); t = p.read_text(encoding="utf-8")
if "--extra-validation" in t:
    print("already patched"); sys.exit(0)

t = sub(t, "def main():\n",
'''def split_extra_validation(evolution, validation, extra):
    """Move the last `extra` evolution questions into validation. They were never
    used for reflection, so the gate stays honest; the test split is untouched."""
    if not extra:
        return evolution, validation
    if extra < 0 or extra >= len(evolution):
        raise ValueError(f"--extra-validation must be between 1 and {len(evolution) - 1}")
    return evolution[:-extra], validation + evolution[-extra:]


def main():
''', "main def")

t = sub(t, '    parser.add_argument("--out", default=None)',
'''    parser.add_argument("--reflect-prompt", default=None, help="reflector prompt file, relative to the repo root")
    parser.add_argument("--merge-prompt", default=None, help="curator prompt file, relative to the repo root")
    parser.add_argument("--extra-validation", type=int, default=0,
                        help="move the last N evolution questions into the validation set (e.g. 50 -> a 100-question gate)")
    parser.add_argument("--out", default=None)''', "args")

t = sub(t, "    if args.limit_evolution:",
'''    import src.skill_evolution as skill_evolution_module
    if args.reflect_prompt:
        skill_evolution_module.REFLECT_PROMPT_PATH = PROJECT_ROOT / args.reflect_prompt
        print(f"Reflector prompt: {args.reflect_prompt}")
    if args.merge_prompt:
        skill_evolution_module.MERGE_PROMPT_PATH = PROJECT_ROOT / args.merge_prompt
        print(f"Curator prompt: {args.merge_prompt}")
    for path in (skill_evolution_module.REFLECT_PROMPT_PATH, skill_evolution_module.MERGE_PROMPT_PATH):
        if not path.exists():
            sys.exit(f"prompt file not found: {path}")

    evolution, validation = split_extra_validation(evolution, validation, args.extra_validation)

    if args.limit_evolution:''', "limit evolution")
p.write_text(t, encoding="utf-8"); print("patched experiments/run_skill_evolution.py")
