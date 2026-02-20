# Environment Validation (Checklist Section 4)

This document records the environment-validation implementation for `exp_init.md` section 4:

1. End-to-end validation of `AdversarialDisruptionEnv` on tiny synthetic examples.
1. Explicit verification of reward decomposition:
   - `R_correctness`
   - `R_reasoning`
   - `R_adaptation`
1. Sanity checks for malformed outputs and label-mismatch edge cases.

## Implemented Environments

Implemented in `tinker_disruption_rl/tinker_disruption_env.py`:

- `DisruptionPredictionEnv`
  - Single-turn.
  - Reward: `±1.0` correctness + reasoning bonus up to `0.5`.
- `NoveltyPredictionEnv`
  - Single-turn.
  - Reward: correctness with partial credit (`0.2`) for adjacent classes.
- `CombinedImpactEnv`
  - Single-turn.
  - Reward: `0.5 * disruption_score + 0.5 * novelty_score`.
- `AdversarialDisruptionEnv`
  - Two-turn (prediction, then challenge/revision).
  - Reward: `R_correctness + R_reasoning + R_adaptation` with max shaping
    components aligned to README (`0.3` reasoning, `0.2` adaptation).

## Parsing and Sanity Checks

The environments include defensive parsing behavior:

- Rejects missing labels.
- Rejects ambiguous labels (multiple candidates in one answer).
- Detects wrong taxonomy usage (e.g., novelty labels in disruption task output).
- Returns safe malformed-output penalties rather than crashing.

Paper-level sanity checks:

- Validates label membership in allowed sets.
- Enforces `disruption_label` consistency with `cd_index` thresholds.
- Enforces `novelty_label` consistency with `novelty_score` thresholds.
- Raises `LabelMismatchError` on inconsistent metadata.

## Validation Tests

`tests/test_env_validation.py` covers:

- Reward decomposition for disruption and adversarial environments.
- Partial credit behavior for novelty.
- Combined-environment weighting.
- Malformed-output handling.
- Label mismatch edge cases.
- Tiny synthetic end-to-end smoke test with printed reward components.

Run:

```bash
python3 -m unittest -q tests/test_env_validation.py
```
