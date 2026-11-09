# Applicability-Aware Transfer of Self-Evolved Skills Across Logical Reasoning Tasks

## Overview

Large Language Model (LLM) agents can learn reusable reasoning strategies, or **skills**, from previous problem-solving experiences. However, a skill that is useful for one reasoning task may not necessarily be appropriate for another task.

Applying an unsuitable skill may reduce model performance and result in **negative transfer**.

This project investigates whether checking the **applicability of learned skills before transferring them to a new reasoning task** can reduce harmful skill reuse.

We evolve reasoning skills using one logical reasoning task and evaluate whether those skills can be effectively transferred to another related but structurally different reasoning task.

---

## Research Question

**Can applicability-aware selection of self-evolved reasoning skills reduce negative transfer when skills learned from one logical reasoning task are applied to another?**

---

## Hypothesis

We hypothesize that **applicability-aware skill selection** will reduce harmful skill reuse compared with:

- Using no skills
- Applying all learned skills
- Randomly selecting skills
- Selecting skills using semantic similarity alone

This may improve overall reasoning accuracy while reducing negative transfer.

---

## Dataset

We use two publicly available tasks from the **BIG-Bench Hard (BBH)** benchmark.

### Source Task: Logical Deduction

**Logical Deduction** is used for skill evolution.

The LLM solves Logical Deduction problems, and successful and unsuccessful reasoning attempts are used to generate and refine reusable reasoning strategies.

### Target Task: Tracking Shuffled Objects

**Tracking Shuffled Objects** is used to evaluate whether the reasoning skills learned from Logical Deduction can be successfully transferred to another reasoning task.

Both tasks involve reasoning about relationships between objects, but they differ in the reasoning procedure required to solve them.

### Dataset Source

**BIG-Bench Hard (BBH):**

https://github.com/suzgunmirac/BIG-Bench-Hard

---

## Proposed Method

The project consists of two main stages:

1. **Skill Evolution**
2. **Skill Transfer and Applicability-Aware Selection**

---

## Stage 1: Skill Evolution

The LLM first solves problems from **Logical Deduction**.

Successful and unsuccessful reasoning attempts are collected and used as feedback to generate reusable reasoning strategies.

```text
Logical Deduction Problems
          |
          v
      LLM Agent
          |
          v
Successful + Unsuccessful Attempts
          |
          v
 Correctness Feedback
          |
          v
Skill Generation / Refinement
          |
          v
     Skill Library
```

The underlying LLM remains frozen. The learned skills are stored externally in a **skill library** rather than modifying the model's weights.

---

## Stage 2: Skill Transfer

The learned skills are then evaluated on the **Tracking Shuffled Objects** task.

```text
Tracking Shuffled Objects
           |
           v
     Target Question
           |
           v
     Skill Selection
           |
           v
    Candidate Skills
           |
           v
 Applicability Check
           |
      +----+----+
      |         |
   Applicable  Not Applicable
      |         |
     KEEP      REJECT
      |
      v
 Selected Skills
      |
      v
    LLM Agent
      |
      v
 Final Answer
```

---

## Experimental Conditions

We compare **five skill-selection approaches**.

### 1. No Skills

The LLM solves the target problem without receiving any learned skills.

This provides the primary baseline.

### 2. All Skills

All learned skills from the skill library are provided to the LLM.

This tests whether unrestricted reuse of learned skills improves or harms performance.

### 3. Random Skills

A fixed number of skills are randomly selected from the skill library.

This acts as a control for skill retrieval.

### 4. Semantic Similarity

Sentence embeddings are used to compare the target problem with skills stored in the skill library.

The most semantically similar skills are selected.

```text
Target Question
      |
      v
Question Embedding
      |
      +------------------+
                         |
Skill Library            |
      |                  |
      v                  |
Skill Embeddings         |
      |                  |
      +------------------+
              |
              v
       Cosine Similarity
              |
              v
         Top-k Skills
```

### 5. Applicability-Aware Selection

This is our **proposed method**.

Semantic similarity is first used to retrieve potentially relevant candidate skills.

An LLM-based applicability check then determines whether each candidate skill's **reasoning procedure is actually appropriate for the target problem**.

```text
Target Question
       |
       v
Semantic Retrieval
       |
       v
Top-k Candidate Skills
       |
       v
LLM Applicability Check
       |
   +---+---+
   |       |
 KEEP    REJECT
   |
   v
Applicable Skills
   |
   v
LLM Solves Problem
```

Only skills determined to be applicable are provided to the final problem-solving agent.

---

## Skill Representation

Learned reasoning skills are stored externally in a structured **skill library**.

Example:

```json
{
    "skill_id": "skill_001",
    "source_task": "logical_deduction",
    "name": "Constraint Ordering",
    "description": "Extract pairwise ordering constraints, combine them transitively, and eliminate answer choices that violate the resulting ordering."
}
```

The goal is to store **general reasoning procedures rather than answers to individual questions**.

---

## Repository Structure

```text
applicability-aware-skill-transfer/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── logical_deduction/
│   └── tracking_shuffled_objects/
│
├── src/
│   ├── dataset.py
│   ├── baseline.py
│   ├── skill_evolution.py
│   ├── skill_library.py
│   ├── retrieval.py
│   ├── applicability.py
│   └── evaluation.py
│
├── prompts/
│   ├── solve_problem.txt
│   ├── evolve_skill.txt
│   └── applicability_check.txt
│
├── skills/
│   └── skill_library.json
│
├── experiments/
│   ├── run_baseline.py
│   ├── run_transfer.py
│   └── progress.md
│
├── results/
│   ├── baseline_results.csv
│   └── transfer_results.csv
│
└── docs/
    └── progress_report/
```

---

## Module Description

### `dataset.py`

Responsible for:

- Loading BBH dataset files
- Extracting questions and answers
- Standardizing dataset formatting
- Removing duplicates
- Creating experiment splits

### `baseline.py`

Implements the **no-skills baseline**.

The LLM solves Tracking Shuffled Objects questions without access to the learned skill library.

### `skill_evolution.py`

Responsible for generating and refining reusable reasoning skills from Logical Deduction experiences.

The module uses:

- Successful reasoning attempts
- Failed reasoning attempts
- Correct answers
- Feedback from previous attempts

to generate generalizable reasoning strategies.

### `skill_library.py`

Responsible for managing the learned skill library.

Functions include:

- Adding skills
- Updating skills
- Loading skills
- Saving skills
- Managing duplicate skills

### `retrieval.py`

Implements **semantic similarity-based skill retrieval** using sentence embeddings.

Given a target question, the module retrieves the most semantically similar skills from the skill library.

### `applicability.py`

Implements the proposed **applicability-aware skill selection method**.

Each candidate skill is evaluated to determine whether its reasoning procedure is appropriate for solving the target problem.

The output is conceptually:

```text
Skill: Constraint Ordering

Decision: NOT_APPLICABLE

Reason:
The target problem requires tracking state changes after a
sequence of swaps rather than constructing a static ordering
from pairwise constraints.
```

### `evaluation.py`

Implements the evaluation metrics used to compare the five experimental conditions.

---

## Evaluation Metrics

### Accuracy

Accuracy measures the percentage of correctly answered problems.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Transfer Gain

Transfer gain measures the change in performance compared with the **no-skills baseline**.

```text
Transfer Gain = Skill Method Accuracy - No-Skills Accuracy
```

### Positive Transfer

Positive transfer occurs when:

```text
No-Skills Baseline = Incorrect
Skill-Based Method = Correct
```

This indicates that the transferred skill helped solve the problem.

### Negative Transfer

Negative transfer occurs when:

```text
No-Skills Baseline = Correct
Skill-Based Method = Incorrect
```

This indicates that introducing a learned skill harmed performance.

### Negative Transfer Rate

The negative transfer rate measures how often skill reuse causes the model to answer a problem incorrectly that it would otherwise have answered correctly.

Additional measurements include:

- Number of skills selected
- Computational cost
- Examples of successful transfer
- Examples of harmful transfer

---

## Experimental Setup

Initial development experiments will use small subsets of the datasets to validate the complete pipeline.

For example:

```text
Logical Deduction
20–50 examples
      |
      v
Skill Evolution
      |
      v
Skill Library
      |
      v
Tracking Shuffled Objects
20–50 examples
      |
      v
Five Experimental Conditions
      |
      v
Evaluation
```

After validating the pipeline, experiments will be expanded to the planned dataset splits.

Test examples will **not** be used during skill evolution or method development.

---

## Expected Results Table

Results will be reported using a table similar to:

| Method | Accuracy | Transfer Gain | Positive Transfer | Negative Transfer Rate |
|---|---:|---:|---:|---:|
| No Skills | TBD | — | — | — |
| All Skills | TBD | TBD | TBD | TBD |
| Random Skills | TBD | TBD | TBD | TBD |
| Semantic Similarity | TBD | TBD | TBD | TBD |
| **Applicability-Aware** | TBD | TBD | TBD | TBD |

---

## Technologies

The project uses:

- **Python**
- **Hugging Face Transformers**
- **Sentence Transformers**
- **PyTorch**
- **BIG-Bench Hard (BBH)**
- **Git/GitHub**
- **ASU SOL** for computational experiments

---

## Setup

```bash
pip install -r requirements.txt

# BBH is not stored in this repo (external/ is gitignored). Clone it once:
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git external/Big-Bench-Hard
# or point to an existing copy (e.g. on SOL):  export BBH_DIR=/path/to/Big-Bench-Hard/bbh

# Create the fixed, seeded splits (already committed in data/splits/; only rerun on purpose):
python -m src.dataset

# Check the answer parser and splits:
python -m pytest tests/
```

**Splits.** `data/splits/*.json` hold the example IDs for every split (pooled over the 3/5/7-object
configs, stratified by config, seed 0). All experimental conditions must evaluate on exactly these
IDs, because transfer gain and negative transfer compare the same example across conditions.

**Prompts.** Every condition builds its prompt with `src.baseline.build_prompt(question, skills)`.
No-skills is `skills=None`; the other conditions differ only in which skills are passed in.

**Answer parsing.** `extract_answer` returns `None` when there is no `FINAL ANSWER` line instead of
guessing. Report the `None` rate (parse failures) alongside accuracy.

---

## Skill Evolution (Logical Deduction -> skill library)

`src/skill_evolution.py` turns Logical Deduction experience into `skills/skill_library.json`.
The LLM stays frozen; skills are learned by a validation-gated loop. Each round:

1. **Solve** a batch of the 150 evolution examples with the current library.
2. **Reflect**: failures (8 per call) plus a few successes for contrast -> a few proposed edits
   (`ADD`, `EDIT`, `MERGE`, `REMOVE`). Parse failures (no `FINAL ANSWER`) are not reflected on.
3. **Curate** (only when several proposals or duplicates need reconciling).
4. **Filter**: edits that mention answer options, objects from the examples, or copy question text
   (`src/leakage.py`), exceed the word limits, or exceed the size caps (20 skills / 1,200 words) are refused.
5. **Gate** on the fixed 50-example validation set: accept only if `repaired - 2 * broken > 0`
   (paired, per question). Otherwise keep the old library and remember the rejected edit.

Defaults: 2 epochs x 5 rounds (batches of 30), edit budget 3 then 2 per round, stop after 5 rounds
in a row without an accepted edit. Every round is logged to `skills/evolution_log.jsonl`.

**Skill format** (a JSON list). Only `name` and `description` are shown to the solver, in the same
format for all five conditions. `applies_when` / `does_not_apply_when` are for the applicability check.

```json
{
  "skill_id": "skill_001", "source_task": "logical_deduction",
  "name": "Constraint chaining",
  "description": "Write each ordering clue as a short relation, chain ...",
  "applies_when": "...", "does_not_apply_when": "...",
  "support_count": 3, "created_round": 1, "last_edited_round": 4
}
```

**Run it**

```bash
# No model needed: runs the whole loop with a fake model (writes to skills/_dry_run/, not committed)
python experiments/run_skill_evolution.py --dry-run --eval-test

# Real run: first add src/llm.py with  get_llm() -> function(prompt: str) -> str  (greedy decoding)
python experiments/run_skill_evolution.py --eval-test
```

Before evolving, the script re-solves the validation set several times with no skills, because a
50-question set is noisy (one question = 2 points) and the baseline failure count limits how much
there is to learn from. `--eval-test` then checks the evolved library in-domain on the Logical
Deduction test split.

Methods follow SkillOpt, Rethinking Self-Evolving Agent Skills, and Piao et al.; see
`THIRD_PARTY_NOTICES.md`.

---

## Team Workflow

The project is divided into five main technical modules:

| Module | Responsibility |
|---|---|
| Integration & Baseline | Repository integration, dataset pipeline, and baseline experiments |
| Skill Evolution | Generate and refine reusable reasoning skills |
| Semantic Retrieval | Embedding-based skill retrieval |
| Applicability Selection | Determine whether candidate skills apply to target problems |
| Evaluation | Metrics, experimental comparison, and result analysis |

All team members will also contribute to testing, documentation, analysis, and the final report.

---

## References

1. Yifan Yang et al. (2026). **SkillOpt: Executive Strategy for Self-Evolving Agent Skills.**

2. Chang Liu, Xinyu Li, and Artur Dubrawski. (2026). **Notes to Self: Can LLMs Benefit from Experiential Abstractions?**

3. Mirac Suzgun et al. (2026). **Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory.**

4. Yuxuan Liu et al. (2026). **Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds.**

5. Xihao Piao, Zifeng Wang, and Zheng Chen. (2026). **Do Self-Evolving Skills Generalize to Held-Out Tasks?**

6. Mirac Suzgun et al. (2023). **Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them.**
