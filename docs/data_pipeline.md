# Data Pipeline: Disruption & Novelty

This document defines the deterministic dataset pipeline for checklist section 2 in `exp_init.md`.

## Scope

Implemented in `tinker_disruption_rl/disruption_novelty_dataset.py`:

- Synthetic dataset generation (`--synthetic`)
- OpenAlex ingestion (`default mode`)
- Label computation + verification for disruption and novelty
- Frozen train/val/test split generation from a seed
- Metadata emission with dataset versioning details

## Dataset Schema

Each JSONL row emits the README schema fields:

1. `openalex_id`
2. `title`
3. `abstract`
4. `publication_year`
5. `cited_by_count`
6. `cd_index`
7. `novelty_score`
8. `conventionality_score`
9. `disruption_label`
10. `novelty_label`
11. `primary_field`
12. `concepts`

## Label Logic

Thresholds are configurable by CLI and captured in metadata.

Disruption label from `cd_index`:

- `disruptive` if `cd_index > disruptive_threshold` (default `0.1`)
- `consolidating` if `cd_index < consolidating_threshold` (default `-0.1`)
- `neutral` otherwise

Novelty label from score delta:

- `novel` if `novelty_score - conventionality_score > novelty_margin` (default `0.15`)
- `conventional` if delta `< -novelty_margin`
- `balanced` otherwise

Every row is re-labeled during build to verify consistency (`--strict-label-check` can fail on mismatch).

In OpenAlex mode, `cd_index`, `novelty_score`, and `conventionality_score` are deterministic citation-derived proxies computed from OpenAlex citation/reference counts and concept distributions, then mapped to labels with the thresholds above.

## Deterministic Splits

Split generation is deterministic for a fixed dataset and seed:

1. IDs are sorted (`openalex_id`)
2. IDs are shuffled with `random.Random(seed)`
3. Train/val/test sizes are computed deterministically from ratios
4. Split IDs are persisted to JSON and hashed (`split_ids_sha256`) in metadata

## Artifacts

The pipeline writes three artifacts:

- `--output`: dataset rows in JSONL
- `--splits-output`: frozen split IDs + split counts
- `--metadata-output`: generation metadata

Metadata includes:

- timestamp (`generated_at_utc`)
- mode (`synthetic` or `openalex`)
- query filters
- split seed + ratios
- label thresholds
- record count + split counts + label counts
- label mismatch counts from verification
- artifact paths + split ID hash

## Commands

Synthetic example:

```bash
python3 tinker_disruption_rl/disruption_novelty_dataset.py \
  --synthetic \
  --n-papers 36 \
  --seed 20260220 \
  --output data/synthetic/disruption_novelty_v1.jsonl \
  --splits-output data/synthetic/disruption_novelty_v1.splits.json \
  --metadata-output data/synthetic/disruption_novelty_v1.metadata.json
```

OpenAlex example:

```bash
python3 tinker_disruption_rl/disruption_novelty_dataset.py \
  --n-papers 500 \
  --seed 20260220 \
  --email your-email@example.com \
  --openalex-filter "type:article,has_abstract:true" \
  --from-year 2000 \
  --to-year 2026 \
  --min-citations 10 \
  --output data/openalex/disruption_novelty_openalex.jsonl \
  --splits-output data/openalex/disruption_novelty_openalex.splits.json \
  --metadata-output data/openalex/disruption_novelty_openalex.metadata.json
```
