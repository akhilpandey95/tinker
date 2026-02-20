You are the Environment sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/env
Branch: feat/env-validation

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist section 4 in exp_init.md and the env design described in README.md.

Scope you own:
- tinker_disruption_rl/tinker_disruption_env.py
- tests/test_env_validation.py
- docs/env_validation.md

Requirements:
- Implement DisruptionPredictionEnv, NoveltyPredictionEnv, CombinedImpactEnv, AdversarialDisruptionEnv.
- Reward decomposition must expose R_correctness, R_reasoning, R_adaptation.
- Add robust parsing/sanity checks for malformed outputs and label mismatch edge cases.
- Include a tiny synthetic end-to-end env smoke test.

Validation:
- Run tests for reward behavior and malformed outputs.
- Show one smoke-run output with reward components.

Constraints:
- Do not edit README.md or exp_init.md.
- Do not touch data/training/repro/report files.

Deliverable back to orchestrator:
- Files changed
- Commands/tests run
- Remaining risks
- Commit hash
