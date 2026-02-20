# RLVR Artifacts (Checklist Section 6)

This folder stores Just-RL-style RLVR outputs trained directly from a pretrained base policy with no SFT scaffold.

## Fixed Environment and Reward Spec

RL training uses `AdversarialDisruptionEnv` with fixed shaping from the project README:

- `R_correctness`: `+1.0` if disruption label is correct, otherwise `-1.0`
- `R_reasoning`: up to `0.3`
- `R_adaptation`: up to `0.2`
- `R_total = R_correctness + R_reasoning + R_adaptation`

The reward/environment spec is explicit in `configs/rlvr/*.json` and persisted in `model_artifact.json`.

## Tiny Dry-Run Commands

Train:

```bash
python3 tinker_disruption_rl/train_disruption.py --config configs/rlvr/dry_run_tiny.json
```

Eval:

```bash
python3 evaluation/rlvr/eval_rlvr.py --config configs/rlvr/eval_test_tiny.json
```

## Expected Artifacts

Inside `results/rlvr/<run_name>/`:

- `model_artifact.json`
- `train_examples.jsonl`
- `val_examples.jsonl`
- `reward_trace.csv`
- `reward_curve.csv`
- `reward_summary.json`
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
- `eval_manifest_test.json` (after eval)

The `metrics_*.json` and `metrics_*.csv` schema matches SFT outputs for direct comparison.
