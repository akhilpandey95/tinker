# SFT Baseline Artifacts (Checklist Section 5)

This folder stores QDoRA + FSDP SFT baseline artifacts using the same joint prompt contract and label taxonomy as the RL task format.

## Prompt/Label Contract

The baseline uses the exact `CombinedImpactEnv` contract:

```text
Predict both labels for the paper.
Disruption labels: disruptive, consolidating, neutral.
Novelty labels: novel, conventional, balanced.
Return exactly:
disruption: <label>
novelty: <label>
reasoning: <short justification>
```

Targets are the dataset-provided `disruption_label` and `novelty_label`.

## Tiny Dry-Run Commands

Train (tiny synthetic split):

```bash
python3 training/sft/train_qdora_fsdp.py --config configs/sft/dry_run_tiny.json
```

Eval (tiny synthetic test split):

```bash
python3 evaluation/sft/eval_sft.py --config configs/sft/eval_test_tiny.json
```

## Expected Artifacts

Inside `results/sft/<run_name>/`:

- `model_artifact.json`
- `train_examples.jsonl`
- `val_examples.jsonl`
- `predictions_train.csv`
- `predictions_val.csv`
- `metrics_train.json`
- `metrics_train.csv`
- `metrics_val.json`
- `metrics_val.csv`
- `calibration_*_disruption.csv`
- `calibration_*_novelty.csv`
- `metrics_test.json` and `metrics_test.csv` (after eval)
- `predictions_test.csv` (after eval)

Calibration artifacts include reliability-bin CSVs and temperature-search CSVs.

## Current Tiny Run (Feb 20, 2026)

Latest local validation run:

- `results/sft/qdora_fsdp_tiny_dry_run/model_artifact.json`
- `results/sft/qdora_fsdp_tiny_dry_run/metrics_test.json`
- `results/sft/qdora_fsdp_tiny_dry_run/metrics_test.csv`
- `results/sft/qdora_fsdp_tiny_dry_run/calibration_test_disruption.csv`
- `results/sft/qdora_fsdp_tiny_dry_run/calibration_test_novelty.csv`

Key test summary (`3` examples):

- Joint exact-match accuracy: `1.0`
- Disruption accuracy: `1.0`
- Novelty accuracy: `1.0`
