You are the Data sub-agent.

Workspace: /Users/akhilpandey/code/writing/tinker_wt/data
Branch: feat/data-pipeline

Read first for alignment:
- /Users/akhilpandey/code/writing/rh/content/notes/tinker.md
- /Users/akhilpandey/code/writing/rh/content/notes/tinker_part2.md
- /Users/akhilpandey/code/writing/tinker/README.md
- /Users/akhilpandey/code/writing/tinker/exp_init.md

Mission:
Implement checklist section 2 in exp_init.md with deterministic data generation and split freezing for disruption/novelty prediction.

Scope you own:
- tinker_disruption_rl/disruption_novelty_dataset.py
- data/** (sample or synthetic outputs allowed, keep small)
- docs/data_pipeline.md

Requirements:
- Support synthetic mode and OpenAlex ingestion mode.
- Emit schema fields defined in README.md.
- Compute/verify disruption and novelty labels.
- Create reproducible train/val/test splits from a seed.
- Write dataset metadata (timestamp, filters, split seed, label thresholds, record counts).

Validation:
- Run a synthetic generation command and verify output rows + split counts.
- Add a short reproducibility check (same seed => same split IDs).

Constraints:
- Do not edit README.md or exp_init.md.
- Do not modify env/training/repro/report files.

Deliverable back to orchestrator:
- Files changed
- Exact commands run
- What is still blocked (if anything)
- Commit hash
