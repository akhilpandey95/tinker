# AGENTS.md for `tinker/`

## Project overview
- `tinker/` is a Python-first research/experimentation repo for disruption/novelty prediction with local dataset prep, toy RL environments, RLVR training, SFT baselines, and evaluation scripts.
- Core dataset + env scripts live in `tinker_disruption_rl/`.
- RLVR training logic lives in `training/rlvr/`; SFT baseline logic lives in `training/sft/`.
- Evaluation scripts live in `evaluation/rlvr/` and `evaluation/sft/`.
- Config presets live in `configs/rlvr/` and `configs/sft/`.
- Tests live in `tests/`.
- Generated outputs/artifacts should go in `data/` and `results/` (avoid editing these manually unless the task is specifically about artifacts).

## Build and test commands
- Run all commands from `tinker/` unless noted otherwise.
- **Dataset builder help**: `python tinker_disruption_rl/disruption_novelty_dataset.py --help`
- **Dataset build (local parquet)**: `python tinker_disruption_rl/disruption_novelty_dataset.py --tinker-sciscinet-parquet /path/to/tinker_sciscinet_papers_alpha.parquet --n-papers 1000 --output data/sciscinet/disruption_novelty_sample.jsonl`
- **Field enrichment help**: `python tinker_disruption_rl/enrich_sciscinet_fields.py --help`
- **Field enrichment (post-process JSONL)**: `python tinker_disruption_rl/enrich_sciscinet_fields.py --base-jsonl data/sciscinet/disruption_novelty_sample.jsonl --paper-fields-parquet /path/to/tinker_sciscinet_paper_fields_alpha.parquet --output-jsonl data/sciscinet/disruption_novelty_sample.enriched.jsonl`
- **RLVR train (tiny dry run)**: `python tinker_disruption_rl/train_disruption.py --config configs/rlvr/dry_run_tiny.json`
- **RLVR eval (tiny dry run)**: `python evaluation/rlvr/eval_rlvr.py --config configs/rlvr/eval_test_tiny.json`
- **SFT train (tiny dry run)**: `python training/sft/train_qdora_fsdp.py --config configs/sft/dry_run_tiny.json`
- **SFT eval (tiny dry run)**: `python evaluation/sft/eval_sft.py --config configs/sft/eval_test_tiny.json`
- **Env/unit tests**: `python -m unittest tests.test_env_validation`

## Environment / dependency notes
- There is no local `pyproject.toml` or `package.json` in `tinker/` right now; do not assume `uv sync` / npm scripts are available in this folder.
- For parquet-based dataset/enrichment scripts, install `polars` and `pyarrow` in the environment you are using.
- The dataset/enrichment scripts use Polars lazy scanning and `LazyFrame.collect_batches(...)`; older Polars versions may fail.

## Code style guidelines

### General
- IM NOT ABSOLUTELY RIGHT OR WRONG, I MAKE MISTAKES AND I LEARN FROM THEM.
- DONT BE AFRAID TO CRITICIZE MY WORK OR POINT OUT MY MISTAKES.
- Critically evaluate the code before making any changes.
- Always use the latest version of the code in the working tree (check for local changes before patching).
- Reduce unnecessary lines when it improves clarity, but do not compress code into unreadable one-liners.
- Remove slop code, dead branches, and comments that restate obvious behavior.
- When adding print statements or logs, DO NOT insert emojis.
- Prefer deterministic outputs (stable JSON formatting, sorted keys where already used, reproducible seeds).
- Keep generated artifacts (`results/`, large `data/` outputs, `__pycache__/`) out of code changes unless explicitly requested.

### Python
- Match the style of the file you are editing; this repo already uses type annotations heavily in many modules.
- Preserve existing invariants and validation behavior unless the task explicitly requires changing them.
- Use `argparse` for CLI scripts and keep `if __name__ == "__main__":` guards at the bottom.
- Use `pathlib.Path` for paths in new code.
- Prefer `logging` over `print` for long-running scripts (dataset build / enrichment), matching the existing module’s logging style.
- Prefer generators / batch iteration for large dataset processing.
- Keep memory use in mind for parquet workflows; use lazy scans and streaming/batch collection patterns.
- Use JSONL for large row-wise outputs and JSON for metadata/manifests/configs.
- Add concise comments only where logic is non-obvious.
- Keep helper utilities near the script that uses them unless multiple modules already share a utility layer.

### JSON / config files
- Keep JSON valid, compact, and consistent with surrounding files.
- Prefer adding a new config variant in `configs/rlvr/` or `configs/sft/` instead of mutating shared base configs for one-off experiments.
- Preserve existing key names because training/eval scripts load them directly.

## Code logical guidelines
- Dataset generation and schema/label changes belong in `tinker_disruption_rl/disruption_novelty_dataset.py`.
- Post-enrichment of `primary_field` / `concepts` belongs in `tinker_disruption_rl/enrich_sciscinet_fields.py`.
- Environment prompts, parsing, reward decomposition, and `Paper` validation belong in `tinker_disruption_rl/tinker_disruption_env.py`.
- RLVR training behavior, policy updates, reward-spec plumbing, and artifacts belong in `training/rlvr/`.
- SFT baseline heuristics/calibration logic belongs in `training/sft/`.
- Evaluation metrics/reporting output belongs in `evaluation/rlvr/` and `evaluation/sft/`.
- Test behavior and regression checks belong in `tests/`.
- Config defaults and experiment knobs belong in `configs/`.

## Repo-specific gotchas (important)
- `disruption_novelty_dataset.py` labels novelty using `novelty_score - conventionality_score` with a margin threshold.
- `Paper` in `tinker_disruption_env.py` validates novelty labels using a toy threshold on `novelty_score` alone.
- `training/rlvr/train_rlvr.py` intentionally remaps the novelty label when constructing `Paper` objects; do not "fix" this mismatch casually without tracing downstream effects.
- Dataset `openalex_id` is populated from parquet `paperid` for downstream compatibility; keep this field name stable unless you update all consumers.

## Important comment file on top of each file
- Keep shebang lines (`#!/usr/bin/env ...`) as the first line in executable scripts; place the license comment block immediately after the shebang.
- Unless the repo owner explicitly asks for a license-header migration, do not mass-edit existing files just to add headers.
- For source files in this repo, use the language-appropriate comment style with this text:

```python
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.
```
