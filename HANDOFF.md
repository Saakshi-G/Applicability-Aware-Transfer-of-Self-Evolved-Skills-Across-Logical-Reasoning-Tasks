# Handoff: codebase walkthrough (as of 2026-11-26)

Project: Applicability-Aware Transfer of Self-Evolved Skills Across Logical Reasoning Tasks (CSE 572, Group 10).
A frozen LLM solves BBH Logical Deduction (LD). A reflector proposes edits to an external skill library. An edit is
accepted only if it passes a validation gate (R - 2B > 0). The skills are then transferred to Tracking Shuffled Objects
(TSO) under five conditions: no skills, all skills, random skills, similarity retrieval, applicability-aware selection.

This is the team GitHub repo (`main`, commit 8204926, 2026-11-09) plus everything added on SOL since then
(evolution fixes, candidate libraries, the transfer pipeline, metrics and the report script). See section 7.

## 1. Run the tests first
    pip install -r requirements.txt pytest
    git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git external/Big-Bench-Hard
    python -m pytest -q tests          # needs the BBH data for test_baseline_and_splits.py; 51 other tests pass without it
The tests use a fake model (`src/fake_llm.py`), so no GPU or API key is needed.

## 2. File map (read in this order)
| File | What it does |
|---|---|
| `src/dataset.py` | Loads BBH LD and TSO (`load_bbh_task`, `load_family`). `make_splits(seed=0)` builds the fixed splits, saved in `data/splits/*.json` and read with `load_split`. |
| `src/baseline.py` | Prompt builder (one template for all conditions, with a skills slot), answer parser (needs `FINAL ANSWER: (X)`; a missing answer = parse failure = wrong). |
| `src/skill_library.py` | The skill library: add/edit/merge/delete ops, JSON save/load. A skill = name, description, applies_when, does_not_apply_when. |
| `src/leakage.py` | Filter that blocks skills containing answer options, object names, or 8+ consecutive words copied from an example. |
| `src/skill_evolution.py` | The evolution loop: solve a batch, reflect on failures, curate, validation gate, patience stop. Batched `evaluate()` added after Nov 10. |
| `src/llm.py` | `get_llm()`: one interface, two backends. `LLM_BACKEND=gemini` (default) or `hf` (Hugging Face, batched, greedy). Rewritten after Nov 10. |
| `src/sampling.py` | New after Nov 10. `subset()` takes N examples spread evenly over 3/5/7 objects (used for screening). |
| `src/candidates.py` | New. Builds the "ungated candidate library": collects every skill proposed during evolution (`proposed_ops`/`dropped_ops` in the logs), clusters near-duplicates (TF-IDF, threshold 0.3), applies the leakage filter, keeps up to 20. |
| `src/retrieval.py` | New. `TfidfScorer` (lexical TF-IDF cosine over name+description+applies_when), `rank_skills`, `retrieve`. Scorers are swappable. `EmbeddingScorer` (sentence embeddings, default all-MiniLM-L6-v2) is the semantic version; pick with `run_transfer.py --retriever embedding`. |
| `src/applicability.py` | New. Applicability judge: the frozen model answers YES/NO per (skill, question); `parse_applicable` (last YES/NO wins; unparseable = NO), `check_applicability` (batched). |
| `src/transfer.py` | New. The five conditions (`no_skills`, `all_skills`, `random_skills`, `similarity`, `applicability`), `select_skills`, and `PromptCache` so identical prompts get one answer (removes batching noise). |
| `src/evaluation.py` | New. `MeteredLLM` (call counts), accuracy, repaired/broken counts, exact sign test, `summarize`, `format_table`. |
| `src/fake_llm.py` | Deterministic fake model for tests and `--dry-run`. |
| `prompts/` | `solve_problem.txt`, `evolve_skill.txt` (reflector), `merge_skills.txt` (curator, v2 has the id rules; v1 kept as `merge_skills.v1.txt`), `applicability_check.txt` (now written), `evolve_skill_v2.txt`. |
| `experiments/run_baseline.py` | Baseline accuracy run. |
| `experiments/screen_model.py` | Model screening (no skills): LD evolution/validation + TSO dev, prints the "rule-of-thumb" checks. |
| `experiments/run_skill_evolution.py` | Runs evolution; flags `--epochs --batch-size --seed --noise-repeats --eval-test --dry-run --out --log`. |
| `scripts/*.sbatch`, `scripts/check_gpu.py` | SOL (Slurm) jobs: `screen_sol.sbatch <model>`, `evolve_sol.sbatch <model>`, `evolve_sol2.sbatch <model> <tag> [args]`. |
| `experiments/build_candidate_library.py` | New. `--logs ... --gated ... --out ... --threshold 0.3 --max-skills 20`; writes the library and `<out>.provenance.json`. |
| `experiments/run_transfer.py` | New. `--library --tag --family --split --k --conditions --seed --limit --dry-run`; writes `results/transfer/<tag>/{summary.json,per_question.jsonl,applicability.jsonl}`. |
| `experiments/make_report.py` | New. Reads the transfer runs, adds paired bootstrap 95% CIs, writes `results/transfer/report/report.md` and `transfer_gain.png`. Needs matplotlib for the figure. |
| `scripts/transfer_sol.sbatch` | New. `sbatch scripts/transfer_sol.sbatch <model> <tag> [run_transfer args]`. |
| `scripts/patch_*.py` | One-off patches applied on SOL (curator prompt rules, `repair_ops`, `--reflector-model`). Kept for the record. |
| `tests/` | `test_baseline_and_splits.py`, `test_skill_library.py`, `test_skill_evolution.py`, `test_batching.py`, `test_candidates.py`, `test_transfer.py`. |


## 3. Splits (seed 0)
LD: 150 evolution / 50 validation / 150 test (400 reserved). TSO: 50 dev / 150 test (550 reserved).
Each task has 750 questions (250 each of 3/5/7 objects). Test questions are never used while building the method.

## 4. Running on SOL (ASU)
Needs VPN. Slurm headers used: `-p public -q class -A class_cse57268549fall2026 -G a100:1`, `HF_HUB_DISABLE_XET=1`.
    module load mamba/latest; source activate skills
    mkdir -p logs
    sbatch scripts/screen_sol.sbatch Qwen/Qwen2.5-3B-Instruct
    sbatch scripts/evolve_sol2.sbatch Qwen/Qwen2.5-3B-Instruct s1 --seed 1 --batch-size 60 --epochs 3
Output files are named per model (and tag), so runs do not overwrite each other.
Python prints are buffered in Slurm logs unless you use `python -u` (evolve_sol2 does).
Secrets: no API keys are in this folder. Gemini uses `GEMINI_API_KEY` in a local `.env` (ignored by git). Gated
Hugging Face models (Llama) need `huggingface-cli login` with `HF_HOME` set to the same folder the job uses.

## 5. Results so far (all no-skills baselines unless stated; one run each, greedy)
| | Qwen2.5-3B | Qwen2.5-7B |
|---|---|---|
| LD evolution accuracy (n=150) | 0.553 | 0.773 |
| LD validation (n=50) | 0.580 | 0.740 |
| TSO dev (n=50) | 0.500 | 0.880 |
| Parse failures | 3/250 | 0/250 |
| LD test, no skills (n=150) | 0.587 | not run |
- Gemini 3.5 Flash-Lite pilot: 20/20 on 20 seven-object LD questions, and its evolution round had no mistakes to learn
  from, so 0 skills. Too strong for this project.
- Group decision so far: Qwen2.5-3B-Instruct as the primary model (it has room on both tasks).

## 6. Evolution: what happened and how it was fixed
- Curator bug: the 3B/7B curator "merged" proposals that were not in the library and invented target ids (`ADD`,
  `proposal:2`, `MERGE_1`, empty lists), so rounds ended as `no-op`. A prompt rule did not hold, so `repair_ops` in
  `src/skill_evolution.py` now fixes it in code (invalid EDIT/MERGE becomes ADD; partial valid ids are kept; repairs are
  logged in `op_repairs`).
- Gated runs: Qwen 3B (seeds 0-2) kept 0-2 skills; Qwen 7B and Llama kept 0. Both gate-accepted 3B skills LOWERED held-out
  LD accuracy (winner's curse: a 50-question validation set reshuffles 15-25 answers by chance).
- So the transfer experiment uses an ungated candidate library (all proposed skills, deduplicated), built with
  `experiments/build_candidate_library.py`, plus a strict library of only the gated skills.

## 7. Transfer results (Qwen2.5-3B-Instruct, n=150 per run, one run each)
Change in accuracy vs no skills, in points, with 95% paired bootstrap CI (see `results/transfer/report/report.md`).
| Run | no_skills | Condition | Change | 95% CI | repaired/broken | sign-test p |
|---|---|---|---|---|---|---|
| TSO, 10-skill ungated | 0.400 | all_skills | -5.3 | [-15.3, +4.0] | 24/32 | 0.35 |
| | | random_skills | +2.0 | [-8.0, +12.0] | 30/27 | 0.79 |
| | | similarity | -6.0 | [-14.0, +2.0] | 16/25 | 0.21 |
| | | applicability | +2.0 | [-4.7, +8.7] | 14/11 | 0.69 |
| TSO, 2 gated skills | 0.400 | all_skills | -2.7 | [-10.7, +5.3] | 18/22 | 0.64 |
| | | applicability | -3.3 | [-7.3, +0.0] | 2/7 | 0.18 |
| LD control, 10 skills | 0.580 | all_skills | -12.0 | [-21.3, -2.7] | 16/34 | 0.02 |
| | | random_skills | -7.3 | [-16.0, +1.3] | 17/28 | 0.14 |
| | | similarity | -1.3 | [-10.0, +7.3] | 21/23 | 0.88 |
| | | applicability | +0.0 | [-4.0, +4.0] | 5/5 | 1.00 |
Findings: (a) no significant positive transfer on TSO under any selection method; (b) skills learned on LD hurt LD
(all_skills -12 points, p=0.02); (c) the 3B self-judge does not discriminate domains: YES rate 15.8% on TSO but only 2.5%
on LD, so the positive control fails; (d) the noise floor is large (random skills changed 57 of 150 answers, CIs about
+-9 points), so effects below ~8 points cannot be detected; (e) all Qwen 3B proposals converge on one strategy (list
constraints, place objects, verify); (f) TSO similarity scores are tiny (mean 0.015), so lexical retrieval is near
arbitrary there.
Caveats: one model, one run per condition; ungated skills are of mixed quality; the similarity baseline is lexical.

## 7b. Later results (2026-11-26): 7B-written skills, semantic retrieval, 7B judge
All Qwen2.5-3B-Instruct as solver, n=150 per run. Change vs no skills in points (TSO no_skills 0.400, LD no_skills 0.580).

Evolution (held-out LD test, no skills -> evolved; seed 0):
| Skill writer / prompt | Gate | Skills kept | Test |
|---|---|---|---|
| 7B, original prompt (2 seeds) | 50 | 0 | n/a |
| 7B, numbered steps (evolve_skill_v2) | 50 | 1 | 0.580 -> 0.553 (24 repaired, 28 broken, p=0.68) |
| 7B, v3 prompt (diversity + boundaries) | 50 | 2 | 0.580 -> 0.473 (15 / 31, p=0.026) |
| 7B, v3 prompt | 100 | 0 | n/a |
Skills that pass a 50-question gate lowered held-out accuracy (winner's curse); the 100-question gate accepted nothing.

Transfer with the ungated 7B-written library (19 skills, `skills/library_ungated_7bwriter.json`, 135 proposals in 19 groups):
| Run | all | random | similarity | applicability |
|---|---|---|---|---|
| TSO, 3B judge | -3.3 | -1.3 | +4.0 | +4.0 (p=0.42, 1.57 skills shown) |
| LD control, 3B judge | -12.0 (p=0.02) | -7.3 | -6.7 | 0.0 (0.13 skills shown) |
| TSO, 7B judge | | | +4.0 | -4.7 (p=0.43, 3.00 shown) |
| LD control, 7B judge | | | -6.7 | -8.7 (p=0.10, 3.00 shown) |
Semantic retrieval (all-MiniLM-L6-v2) on the 10-skill 3B library, TSO: similarity +2.7, applicability -1.3 (both n.s.).
Judge YES rate (TSO vs LD, which should be LOWER vs HIGHER): 3B judge 11.4% vs 0.7% (backwards); 7B judge 51.2% vs 64.9% (right
direction but weak). The 7B judge answers YES to generic skills on every question (3 skills: 150/150 on TSO) and discriminates
only on specific ones (e.g. 'Price Comparison' 0 vs 35, 'Understand second from left/right' 0 vs 67), so the top-3 kept are generic.
Conclusion: no significant transfer gain on TSO under any selection method; skills hurt in-domain (all skills -12, p=0.02, two libraries).

## 8. Remaining work (not done)
- Semantic retriever and 7B judge experiments are done (section 7b). Not done: more seeds or models, a third task pair, a judge
  prompt that separates generic from specific skills.
- Report and deck: use `results/transfer/report/` (tables and `transfer_gain.png`) and section 7b.
