"""Add --reflector-model (a separate, stronger model that WRITES skills).
Run from the repo root:  python scripts/patch_reflector.py
Safe to run twice. Edits src/skill_evolution.py and experiments/run_skill_evolution.py in place."""
import pathlib, sys

def sub(text, old, new, label):
    if old not in text:
        sys.exit(f"anchor not found for: {label} - send me the file so I can adjust")
    return text.replace(old, new, 1)

p = pathlib.Path("src/skill_evolution.py"); t = p.read_text(encoding="utf-8")
if "reflector_llm" in t:
    print("src/skill_evolution.py already patched")
else:
    t = sub(t, "    similarity_fn=lexical_similarity,\n    verbose=True,\n):",
               "    similarity_fn=lexical_similarity,\n    verbose=True,\n    reflector_llm=None,\n):", "signature")
    t = sub(t, "      llm_calls        total model calls\n      stopped          why the loop ended\n",
               "      llm_calls        total solver-model calls\n"
               "      reflector_calls  calls made to `reflector_llm` (0 when it is not used)\n"
               "      stopped          why the loop ended\n\n"
               "    `llm` solves the questions (rollouts and the validation gate). By default\n"
               "    it also writes the skills (reflector + curator). Pass `reflector_llm` to use\n"
               "    a different, usually stronger, model for writing skills only; the solver\n"
               "    that the skills are judged on stays `llm`.\n", "docstring")
    t = sub(t, "    cfg = config or EvolutionConfig()\n    llm = CountingLLM(llm)\n",
               "    cfg = config or EvolutionConfig()\n    llm = CountingLLM(llm)\n"
               "    writer = CountingLLM(reflector_llm) if reflector_llm is not None else llm\n", "writer")
    t = sub(t, "                    ops = reflect_and_curate(\n                        llm, lib, failures,",
               "                    ops = reflect_and_curate(\n                        writer, lib, failures,", "call site")
    t = sub(t, '        "llm_calls": llm.calls,\n        "stopped": stopped,',
               '        "llm_calls": llm.calls,\n        "reflector_calls": writer.calls if writer is not llm else 0,\n        "stopped": stopped,', "return")
    p.write_text(t, encoding="utf-8"); print("patched src/skill_evolution.py")

p = pathlib.Path("experiments/run_skill_evolution.py"); t = p.read_text(encoding="utf-8")
if "--reflector-model" in t:
    print("experiments/run_skill_evolution.py already patched")
else:
    t = sub(t, '    parser.add_argument("--out", default=None)',
               '    parser.add_argument("--reflector-model", default=None,\n'
               '                        help="use this (usually stronger) model to WRITE skills (reflector + curator); "\n'
               '                             "the solver and the validation gate still use the main model")\n'
               '    parser.add_argument("--out", default=None)', "arg")
    t = sub(t, "    out_path = Path(args.out)",
               "    reflector = None\n    if args.reflector_model:\n        if args.dry_run:\n"
               "            reflector = llm  # nothing to load; exercises the code path\n        else:\n"
               "            reflector = get_llm(args.reflector_model)\n"
               "        print(f\"Solver: {getattr(llm, 'description', 'model')}   \"\n"
               "              f\"Skill writer (reflector + curator): {getattr(reflector, 'description', 'dry-run')}\")\n\n"
               "    out_path = Path(args.out)", "reflector block")
    t = sub(t, "        log_path=log_path, checkpoint_path=out_path,\n    )",
               "        log_path=log_path, checkpoint_path=out_path,\n        reflector_llm=reflector,\n    )", "evolve call")
    t = sub(t, "Skills kept: {len(result['library'])}   Model calls: {result['llm_calls']}\")",
               "Skills kept: {len(result['library'])}   Solver calls: {result['llm_calls']}\"\n"
               "          f\"   Skill-writer calls: {result['reflector_calls']}\")", "summary print")
    p.write_text(t, encoding="utf-8"); print("patched experiments/run_skill_evolution.py")
