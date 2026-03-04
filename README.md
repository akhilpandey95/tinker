# Tinker RL: Disruption & Novelty Prediction

A toy RL environment for training language models to predict scientific impact metrics (disruption index, novelty/conventionality).

Blog post: [Tinker, smol-RL and QDoRA](https://akhilpandey95.github.io/notes/tinker/)

## Overview

This repo now runs a lean, config-driven SciSciNet workflow built around three canonical scripts:
- **`src/sciscinet_cookbook_mini.py`**: preflight checks, dataset sanity, deterministic manifest
- **`src/sciscinet_cookbook_hosted_runner.py`**: hosted RL launcher + dry-run/config-only manifest path
- **`src/viz.py`**: run diagnostics and visualization

The workflow keeps the same adversarial framing:
- **Agent A (trainable)**: predicts disruptive/consolidating/neutral (and optionally novelty)
- **Agent B (fixed/challenger role in the environment design)**: challenges the prediction with counterarguments

This follows the [Twenty Questions pattern](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/multiplayer_rl/twenty_questions) from Tinker, adapted for science-of-science metrics.

## Files

```text
tinker/
├── src/
│   ├── sciscinet_cookbook_mini.py
│   ├── sciscinet_cookbook_hosted_runner.py
│   └── viz.py
├── configs/
│   ├── experiments/
│   │   ├── base_hosted.json
│   │   ├── exp2_batch_diversity.json
│   │   ├── exp3_reward_reweighting.json
│   │   ├── exp4_optimization_exploration.json
│   │   └── exp5_prompt_contract_upgrade.json
│   └── deprecated/
├── data/
├── agent_runs/
└── README.md
```

## Canonical run-root conventions

- code root: `/Users/akhilpandey/code/writing/tinker`
- data root: `./data` (fallback: external `../tinker_smolrl_data`)
- experiment outputs:
  - `agent_runs/<exp>/runs`
  - `agent_runs/<exp>/viz`

## Dataset strategy

- Metadata and split manifests are tracked in-repo under `data/`.
- Heavy JSONL datasets should remain external + gitignored.
- Script lookup precedence is `./data` first, then `../tinker_smolrl_data`.

## Quick start

### 1) Preflight

```bash
python3 src/sciscinet_cookbook_mini.py \
  --config configs/experiments/base_hosted.json \
  --dataset-key sci_balanced_from2m_no_ovr_rl_balanced \
  --max-train 1
```

### 2) Hosted manifest check (no launch)

```bash
python3 src/sciscinet_cookbook_hosted_runner.py \
  --config configs/experiments/base_hosted.json \
  --config-only \
  --output-dir /tmp/hosted_dry_check
```

### 3) Hosted run (base + variant overlay)

```bash
python3 src/sciscinet_cookbook_hosted_runner.py \
  --config configs/experiments/base_hosted.json \
  --config configs/experiments/exp2_batch_diversity.json
```

### 4) Viz

```bash
python3 src/viz.py \
  --results-root agent_runs/exp2_batch_diversity/runs \
  --run-glob "exp2_*" \
  --output-dir agent_runs/exp2_batch_diversity/viz
```

## Environment Types

| Environment | Description | Turns | Reward |
|-------------|-------------|-------|--------|
| `DisruptionPredictionEnv` | Single-turn disruption prediction | 1 | ±1.0 + 0.5 reasoning bonus |
| `AdversarialDisruptionEnv` | Multi-turn with challenger | 2-3 | ±1.0 + 0.3 reasoning + 0.2 adaptation |
| `NoveltyPredictionEnv` | Single-turn novelty/conventionality | 1 | ±1.0 (partial credit for adjacent) |
| `CombinedImpactEnv` | Both metrics simultaneously | 1 | 0.5 × disruption + 0.5 × novelty |

## References

- [Thinking Machines Lab: Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)
- [Just-RL (arXiv:2512.16649)](https://arxiv.org/abs/2512.16649)
- [Funk & Owen-Smith (2017), CD Index](https://doi.org/10.1287/mnsc.2015.2366)
- [Uzzi et al. (2013), Atypical Combinations and Scientific Impact](https://doi.org/10.1126/science.1240474)
- [AP Akella, HV Siravuri, S Rohatgi (2025), "Pre-review to Peer review: Pitfalls of Automating Reviews using Large Language Models"](https://arxiv.org/abs/2512.22145)

---

Author: Akhil Pandey Akella
<a href="https://github.com/akhilpandey95/tinker">Tinker, smol-RL and QDoRA</a> © 2026 by <a href="https://github.com/akhilpandey95">Akhil Akella</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/sa.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
