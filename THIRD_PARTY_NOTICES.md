# Third-party notices

This project's skill-evolution module (`src/skill_evolution.py`, `src/skill_library.py`,
`src/leakage.py`, `prompts/evolve_skill.txt`, `prompts/merge_skills.txt`) is our own code.
No source code was copied from the repositories below. The algorithm and some prompt wording are
adapted from the published papers and, for the prompts, from the SkillOpt repository.

## SkillOpt (Microsoft) - MIT License

- Paper: Yang et al. 2026, *SkillOpt: Executive Strategy for Self-Evolving Agent Skills*, arXiv:2605.23904
- Code: https://github.com/microsoft/SkillOpt
- Adapted: the reflection and merge prompt wording (common patterns rather than edge cases, do not
  hardcode task-specific values, prefer patterns proposed by several analysts, failure-driven
  edits take priority), and the design of reflection minibatches, a small per-round edit budget,
  hash-cached candidates, a rejected-edit buffer and per-round reports.

MIT License

Copyright (c) 2026 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Other sources of ideas (papers only; no code used)

- Liu et al. 2026, *Rethinking Self-Evolving Agent Skills: Feedback Dynamics over Multiple Rounds*,
  arXiv:2608.02636 (code: https://github.com/HKUST-KnowComp/rethinkskill, MIT) - at most ~10 rounds,
  stop after 5 consecutive rejected/no-op rounds, failures are essential feedback, validation noise.
- Piao et al. 2026, *Do Self-Evolving Skills Generalize to Held-Out Tasks?*, arXiv:2609.39148 -
  paired acceptance rule `|R| - 2|B| > 0` and the rule against encoding literal answers or
  one-task exceptions.
- Zhao et al. 2024, *ExpeL: LLM Agents Are Experiential Learners*, AAAI 2024, arXiv:2308.10144 -
  ADD / EDIT style insight operations.
- Suzgun et al. 2026, *Dynamic Cheatsheet*, EACL 2026 - instruction to discard problem-specific
  details when curating memory.
