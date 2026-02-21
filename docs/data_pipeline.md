# Data Pipeline: Disruption & Novelty

This document defines the deterministic dataset pipeline for checklist section 2 in `exp_init.md`.

## Scope

Implemented in `tinker_disruption_rl/disruption_novelty_dataset.py`:

- Synthetic dataset generation (`--synthetic`)
- OpenAlex ingestion (`default mode`)
- Local SciSciNet parquet ingestion (`--sciscinet-parquet` and/or `--tinker-sciscinet-parquet`)
- Label computation + verification for disruption and novelty
- Frozen train/val/test split generation from a seed
- Metadata emission with dataset versioning details

For parquet mode, install `polars` (`pip install polars pyarrow`).

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

In SciSciNet parquet mode, `cd_index` uses `disruption` (or `cd_index` if present), while novelty/conventionality are derived from Uzzi-style `Atyp_10pct_Z` and `Atyp_Median_Z` when explicit normalized scores are absent.

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
- mode (`synthetic`, `openalex`, or `sciscinet_parquet`)
- query filters
- split seed + ratios
- label thresholds
- record count + split counts + label counts
- label mismatch counts from verification
- artifact paths + split ID hash

## Commands

SciSciNet parquet example (alpha parquet with title + abstract inverted index):

```bash
python3 tinker_disruption_rl/disruption_novelty_dataset.py \
  --n-papers 500000 \
  --tinker-sciscinet-parquet tinker_sciscinet_papers_alpha.parquet \
  --parquet-primary tinker \
  --sciscinet-language en \
  --sciscinet-from-year 1950 \
  --sciscinet-min-citations 0 \
  --output data/sciscinet/disruption_novelty_sciscinet.jsonl \
  --splits-output data/sciscinet/disruption_novelty_sciscinet.splits.json \
  --metadata-output data/sciscinet/disruption_novelty_sciscinet.metadata.json
```

You can also pass both `--sciscinet-parquet` and `--tinker-sciscinet-parquet`; the loader coalesces columns by `paperid`.

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
