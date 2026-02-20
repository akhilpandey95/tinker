You are the RLVR sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/rlvr
Branch: feat/rlvr-training

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist section 6: Just-RL-style RLVR from pretrained base, no SFT scaffold, fair comparison with baseline.

Scope you own:
- tinker_disruption_rl/train_disruption.py
- training/rlvr/**
- evaluation/rlvr/eval_rlvr.py
- configs/rlvr/**
- results/rlvr/README.md

Requirements:
- Use adversarial env reward structure from README.
- Keep reward shaping/environment settings explicit and fixed.
- Evaluate with same metric suite shape as SFT outputs.
- Include tiny dry-run mode.

Validation:
- Run dry-run training command.
- Run eval and emit metrics artifact in results/rlvr/.

Constraints:
- Do not edit README.md or exp_init.md.
- Do not modify SFT/data/env/repro files.

Deliverable back to orchestrator:
- Files changed
- Commands run
- Produced artifacts
- Commit hash
