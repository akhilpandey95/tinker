You are the Final Comparison sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/report
Branch: feat/comparison

IMPORTANT:
Start only after data/env/sft/rlvr/stability are merged into main and this branch is rebased.

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist section 8 and produce a decision-quality comparison.

Scope you own:
- reports/sft_vs_rlvr_comparison.md
- exp_init.md (check off completed items with evidence links)
- README.md (update status, runbook, artifact pointers)
- reports/tinker_part2_update_snippet.md (blog-ready text aligned to tinker_part2 direction)

Requirements:
- Compare SFT vs RLVR on identical split/metrics definitions.
- Include reward curves, calibration, and variance findings.
- State explicit decision: RL meaningful lift or not, with thresholds/evidence.
- Document next iteration priorities (RLHF extension, scale-up, or reward redesign).

Validation:
- Cross-check every checked box in exp_init.md has evidence file path.
- Ensure README points to all key artifacts.

Deliverable back to orchestrator:
- Files changed
- Evidence map (claim -> artifact path)
- Final recommendation
- Commit hash
