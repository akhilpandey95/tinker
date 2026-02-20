You are the SFT baseline sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/sft
Branch: feat/baseline-sft

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist section 5: QDoRA + FSDP SFT baseline on the same task format used for RLVR.

Scope you own:
- training/sft/train_qdora_fsdp.py
- training/sft/prompts.py
- evaluation/sft/eval_sft.py
- configs/sft/**
- results/sft/README.md

Requirements:
- Baseline uses identical label targets and prompt contract as RL task.
- Save metrics for disruption + novelty classification and calibration artifacts.
- Include tiny dry-run mode for local validation.

Validation:
- Run dry-run train/eval command on synthetic/tiny data.
- Save metrics JSON/CSV in results/sft/.

Constraints:
- Do not edit README.md or exp_init.md.
- Do not edit RLVR, env, data, or repro modules.

Deliverable back to orchestrator:
- Files changed
- Commands run
- Produced artifacts
- Commit hash
