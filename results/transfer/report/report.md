# Transfer results

Accuracy change is relative to the no-skills run on the same questions. CI = 95% paired bootstrap interval; p = exact sign test on repaired vs broken. A CI that includes 0 means the change cannot be told apart from no change.

## final_ungated: tracking_shuffled_objects/test, 10 skills, k=3

Library: `skills/library_ungated_qwen3b_final.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.400

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| all_skills | 0.347 | -5.3 | [-15.3, +4.0] | 24 / 32 | 0.35 | 10.00 | 150 | 0 |
| random_skills | 0.420 | +2.0 | [-8.0, +12.0] | 30 / 27 | 0.79 | 3.00 | 150 | 0 |
| similarity | 0.340 | -6.0 | [-14.0, +2.0] | 16 / 25 | 0.21 | 3.00 | 150 | 0 |
| applicability | 0.420 | +2.0 | [-4.7, +8.7] | 14 / 11 | 0.69 | 1.19 | 1577 | 73 |

Applicability judge:
- Judge said YES to 237 of 1500 (question, skill) pairs (15.8%); unparseable replies: 0.
- 78 of 150 questions (52%) received at least one skill; on those, accuracy changed by 14 repaired and 11 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 27, skill_002 56, skill_003 34, skill_004 5, skill_005 8, skill_006 17, skill_007 29, skill_008 16, skill_009 45, skill_010 0.

## strict: tracking_shuffled_objects/test, 2 skills, k=3

Library: `skills/library_strict_qwen3b.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.400

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| all_skills | 0.373 | -2.7 | [-10.7, +5.3] | 18 / 22 | 0.64 | 2.00 | 150 | 0 |
| applicability | 0.367 | -3.3 | [-7.3, +0.0] | 2 / 7 | 0.18 | 0.37 | 323 | 127 |

Applicability judge:
- Judge said YES to 55 of 300 (question, skill) pairs (18.3%); unparseable replies: 0.
- 39 of 150 questions (26%) received at least one skill; on those, accuracy changed by 2 repaired and 7 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 25, skill_002 30.

## ld_control: logical_deduction/test, 10 skills, k=3

Library: `skills/library_ungated_qwen3b_final.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.580

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| all_skills | 0.460 | -12.0 | [-21.3, -2.7] | 16 / 34 | 0.02 | 10.00 | 150 | 0 |
| random_skills | 0.507 | -7.3 | [-16.0, +1.3] | 17 / 28 | 0.14 | 3.00 | 150 | 0 |
| similarity | 0.567 | -1.3 | [-10.0, +7.3] | 21 / 23 | 0.88 | 3.00 | 150 | 0 |
| applicability | 0.580 | +0.0 | [-4.0, +4.0] | 5 / 5 | 1.00 | 0.25 | 1530 | 120 |

Applicability judge:
- Judge said YES to 37 of 1500 (question, skill) pairs (2.5%); unparseable replies: 0.
- 30 of 150 questions (20%) received at least one skill; on those, accuracy changed by 5 repaired and 5 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 3, skill_002 27, skill_003 3, skill_004 0, skill_005 0, skill_006 0, skill_007 0, skill_008 4, skill_009 0, skill_010 0.

## ungated_embed: tracking_shuffled_objects/test, 10 skills, k=3

Library: `skills/library_ungated_qwen3b_final.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.400

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| similarity | 0.427 | +2.7 | [-7.3, +12.7] | 32 / 28 | 0.70 | 3.00 | 150 | 0 |
| applicability | 0.387 | -1.3 | [-7.3, +4.7] | 10 / 12 | 0.83 | 1.19 | 1574 | 76 |

Applicability judge:
- Judge said YES to 237 of 1500 (question, skill) pairs (15.8%); unparseable replies: 0.
- 78 of 150 questions (52%) received at least one skill; on those, accuracy changed by 10 repaired and 12 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 27, skill_002 56, skill_003 34, skill_004 5, skill_005 8, skill_006 17, skill_007 29, skill_008 16, skill_009 45, skill_010 0.

## ungated_7bw: tracking_shuffled_objects/test, 19 skills, k=3

Library: `skills/library_ungated_7bwriter.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.400

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| all_skills | 0.367 | -3.3 | [-14.0, +6.7] | 29 / 34 | 0.61 | 19.00 | 150 | 0 |
| random_skills | 0.387 | -1.3 | [-10.7, +8.0] | 25 / 27 | 0.89 | 3.00 | 150 | 0 |
| similarity | 0.440 | +4.0 | [-4.7, +13.3] | 27 / 21 | 0.47 | 3.00 | 150 | 0 |
| applicability | 0.440 | +4.0 | [-4.0, +12.0] | 22 / 16 | 0.42 | 1.57 | 2952 | 48 |

Applicability judge:
- Judge said YES to 326 of 2850 (question, skill) pairs (11.4%); unparseable replies: 0.
- 102 of 150 questions (68%) received at least one skill; on those, accuracy changed by 22 repaired and 16 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 7, skill_002 88, skill_003 0, skill_004 43, skill_005 26, skill_006 0, skill_007 31, skill_008 0, skill_009 15, skill_010 0, skill_011 38, skill_012 11, skill_013 15, skill_014 1, skill_015 0, skill_016 6, skill_017 5, skill_018 4, skill_019 36.

## ld_control_7bw: logical_deduction/test, 19 skills, k=3

Library: `skills/library_ungated_7bwriter.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.580

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| all_skills | 0.460 | -12.0 | [-22.0, -2.7] | 19 / 37 | 0.02 | 19.00 | 150 | 0 |
| random_skills | 0.507 | -7.3 | [-16.7, +2.0] | 22 / 33 | 0.18 | 3.00 | 150 | 0 |
| similarity | 0.513 | -6.7 | [-16.7, +3.3] | 23 / 33 | 0.23 | 3.00 | 150 | 0 |
| applicability | 0.580 | +0.0 | [-2.7, +2.7] | 2 / 2 | 1.00 | 0.13 | 2869 | 131 |

Applicability judge:
- Judge said YES to 19 of 2850 (question, skill) pairs (0.7%); unparseable replies: 0.
- 19 of 150 questions (13%) received at least one skill; on those, accuracy changed by 2 repaired and 2 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 0, skill_002 1, skill_003 0, skill_004 2, skill_005 0, skill_006 0, skill_007 13, skill_008 0, skill_009 0, skill_010 0, skill_011 0, skill_012 0, skill_013 0, skill_014 0, skill_015 0, skill_016 0, skill_017 0, skill_018 0, skill_019 3.

## judge7b_tso: tracking_shuffled_objects/test, 19 skills, k=3

Library: `skills/library_ungated_7bwriter.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.400

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| similarity | 0.440 | +4.0 | [-4.7, +13.3] | 27 / 21 | 0.47 | 3.00 | 150 | 0 |
| applicability | 0.353 | -4.7 | [-14.7, +5.3] | 25 / 32 | 0.43 | 3.00 | 148 (+2850 judge) | 2 |

Applicability judge:
- Judge said YES to 1459 of 2850 (question, skill) pairs (51.2%); unparseable replies: 0.
- 150 of 150 questions (100%) received at least one skill; on those, accuracy changed by 25 repaired and 32 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 30, skill_002 1, skill_003 150, skill_004 137, skill_005 9, skill_006 150, skill_007 29, skill_008 146, skill_009 150, skill_010 9, skill_011 79, skill_012 134, skill_013 147, skill_014 66, skill_015 21, skill_016 55, skill_017 0, skill_018 0, skill_019 146.

## judge7b_ld: logical_deduction/test, 19 skills, k=3

Library: `skills/library_ungated_7bwriter.json`   Model: hf:Qwen/Qwen2.5-3B-Instruct   no_skills accuracy: 0.580

| Condition | Accuracy | Change (pts) | 95% CI (pts) | Repaired / broken | p | Skills shown | Model calls | Prompts reused |
|---|---|---|---|---|---|---|---|---|
| similarity | 0.513 | -6.7 | [-16.7, +3.3] | 23 / 33 | 0.23 | 3.00 | 150 | 0 |
| applicability | 0.493 | -8.7 | [-18.7, +1.3] | 21 / 34 | 0.10 | 3.00 | 24 (+2850 judge) | 126 |

Applicability judge:
- Judge said YES to 1849 of 2850 (question, skill) pairs (64.9%); unparseable replies: 0.
- 150 of 150 questions (100%) received at least one skill; on those, accuracy changed by 21 repaired and 34 broken (the other questions have the no-skills prompt).
- YES count per skill: skill_001 92, skill_002 55, skill_003 150, skill_004 38, skill_005 15, skill_006 139, skill_007 73, skill_008 150, skill_009 128, skill_010 101, skill_011 40, skill_012 139, skill_013 146, skill_014 150, skill_015 46, skill_016 146, skill_017 35, skill_018 67, skill_019 139.
