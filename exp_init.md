# Tinker Project: Experiment Initialization Checklist

## 1) Scope and Success Criteria
- [ ] Finalize the primary question: does Tinker RLVR (adversarial reasoning) outperform QDoRA SFT on disruption/novelty prediction?
- [ ] Define success metrics (classification quality, calibration, reward behavior, run-to-run stability).
- [ ] Freeze acceptance criteria for claiming RL value over SFT.

## 2) Data Pipeline and Splits
- [ ] Implement OpenAlex data ingestion for paper metadata + citation-derived labels.
- [ ] Compute/verify disruption and novelty labels for all examples.
- [ ] Create and freeze train/validation/test splits.
- [ ] Save dataset version metadata (timestamp, query filters, split seed).

## 3) Reproducibility Harness
- [ ] Add run logging for: seed, GPU type, dtype, kernel/config flags, decode settings, code/data version.
- [ ] Standardize config files so all runs are reproducible from a single source of truth.
- [ ] Add repeated-run launcher for variance measurement.

## 4) Environment Validation (Adversarial Setup)
- [ ] Validate `AdversarialDisruptionEnv` end-to-end on a tiny synthetic dataset.
- [ ] Verify reward components (`R_correctness`, `R_reasoning`, `R_adaptation`) behave as intended.
- [ ] Add sanity checks for malformed outputs and label mismatch edge cases.

## 5) Baseline First: QDoRA + FSDP SFT
- [ ] Train QDoRA SFT baseline on the exact same task and prompt format.
- [ ] Evaluate on frozen validation/test splits.
- [ ] Record baseline metrics and calibration artifacts.

## 6) RL Run: Tinker RLVR (Just-RL Style)
- [ ] Train RLVR from pretrained base (no SFT scaffold) on the same split.
- [ ] Keep reward shaping and environment settings fixed for fair comparison.
- [ ] Evaluate with the same metric suite as baseline.

## 7) Stability Ablations
- [ ] Run controlled ablations across dtype/hardware/config settings.
- [ ] Repeat each condition multiple times with fixed seeds where applicable.
- [ ] Quantify output and metric variance per condition.

## 8) Final Comparison and Decision
- [ ] Build a single comparison report: SFT vs RLVR metrics, reward curves, calibration, variance.
- [ ] Decide whether RL provides meaningful lift beyond QDoRA SFT.
- [ ] Document next iteration priorities (RLHF extension, scale-up, or reward redesign).
