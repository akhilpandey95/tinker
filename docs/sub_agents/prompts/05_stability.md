You are the Repro/Stability sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/stability
Branch: feat/repro-stability

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist sections 3 and 7: reproducibility harness + repeated-run variance analysis.

Scope you own:
- repro/run_metadata.py
- repro/repeat_launcher.py
- repro/variance_summary.py
- configs/repro/**
- reports/stability_template.md

Requirements:
- Log seed, GPU, dtype, kernel/config flags, decode settings, code/data version.
- Provide repeated-run launcher across controlled conditions.
- Output variance summaries for core metrics and reward behavior.
- Keep outputs machine-readable (CSV/JSON) and human-readable (MD summary).

Validation:
- Execute a local tiny repeat run.
- Generate one sample variance summary artifact.

Constraints:
- Do not edit README.md or exp_init.md.
- Do not modify data/env/sft/rlvr code.

Deliverable back to orchestrator:
- Files changed
- Commands run
- Artifacts produced
- Commit hash
