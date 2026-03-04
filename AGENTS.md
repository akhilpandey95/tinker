# AGENTS.md for `tinker/`

## Canonical workflow surface
- This repo is the SciSciNet cookbook experimentation workspace.
- Active scripts (source of truth):
  - `src/sciscinet_cookbook_mini.py`
  - `src/sciscinet_cookbook_hosted_runner.py`
  - `src/viz.py`
- Treat removed paths as legacy and do not add new usage:
  - `tinker_disruption_rl/`
  - `training/`
  - `evaluation/`

## Data conventions
- Metadata and split manifests are tracked in-repo under `data/`.
- Large dataset JSONL files are expected outside git history (external + gitignored).
- Dataset lookup preference in current scripts:
  1. `./data`
  2. `../tinker_smolrl_data`

## Canonical run roots
- Code root: `/Users/akhilpandey/code/writing/tinker`
- Data root: `./data` (fallback: external `tinker_smolrl_data`)
- Experiment outputs:
  - `agent_runs/<exp>/runs`
  - `agent_runs/<exp>/viz`

## Default command templates
- Preflight manifest:
  - `python3 src/sciscinet_cookbook_mini.py --config configs/experiments/base_hosted.json --max-train 1`
- Hosted manifest preflight (no training launch):
  - `python3 src/sciscinet_cookbook_hosted_runner.py --config configs/experiments/base_hosted.json --config-only --output-dir /tmp/hosted_dry_check`
- Hosted run:
  - `python3 src/sciscinet_cookbook_hosted_runner.py --config configs/experiments/base_hosted.json --config configs/experiments/exp2_batch_diversity.json`
- Viz:
  - `python3 src/viz.py --results-root agent_runs/exp2_batch_diversity/runs --run-glob "exp2_*" --output-dir agent_runs/exp2_batch_diversity/viz`

## Guardrail check for stale references
- `rg -n "tinker_disruption_rl|training/|evaluation/" README.md AGENTS.md src/*.py configs/**/*.json /Users/akhilpandey/code/writing/agent_runs/docs/*.md`
