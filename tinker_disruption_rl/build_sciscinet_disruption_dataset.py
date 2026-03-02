# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""One-shot dataset creator for local SciSciNet alpha artifacts.

This orchestrates:
1) disruption/novelty JSONL + splits + metadata generation
2) optional field enrichment from the paper-field alpha parquet
"""

from __future__ import annotations

import argparse
import logging
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_BUILDER = REPO_ROOT / "tinker" / "tinker_disruption_rl" / "disruption_novelty_dataset.py"
FIELD_ENRICHER = REPO_ROOT / "tinker" / "tinker_disruption_rl" / "enrich_sciscinet_fields.py"
DEFAULT_PAPERS_PARQUET = REPO_ROOT / "tinker_sciscinet_papers_alpha.parquet"
DEFAULT_PAPER_FIELDS_PARQUET = REPO_ROOT / "tinker_sciscinet_paper_fields_alpha.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tinker_smolrl_data"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build SciSciNet disruption/novelty dataset and optionally enrich "
            "primary_field / concepts in one pass over local alpha parquet files."
        )
    )
    p.add_argument(
        "--tinker-sciscinet-parquet",
        type=Path,
        default=DEFAULT_PAPERS_PARQUET,
        help="Path to tinker_sciscinet_papers_alpha.parquet",
    )
    p.add_argument(
        "--paper-fields-parquet",
        type=Path,
        default=DEFAULT_PAPER_FIELDS_PARQUET,
        help="Path to tinker_sciscinet_paper_fields_alpha.parquet",
    )
    p.add_argument("--name", default="disruption_novelty_sciscinet_500k", help="Output filename prefix")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-papers", type=int, default=500000)
    p.add_argument("--seed", type=int, default=20260220)
    p.add_argument("--sciscinet-language", default="en")
    p.add_argument("--disruptive-threshold", type=float, default=0.1)
    p.add_argument("--consolidating-threshold", type=float, default=-0.1)
    p.add_argument("--novelty-margin", type=float, default=0.15)
    p.add_argument("--sciscinet-from-year", type=int)
    p.add_argument("--sciscinet-to-year", type=int)
    p.add_argument("--sciscinet-min-citations", type=int)
    p.add_argument("--include-retracted", action="store_true")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--row-log-interval", type=int, default=50000)
    p.add_argument("--skip-enrichment", action="store_true", help="Skip field enrichment step")
    p.add_argument("--concepts-k", type=int, default=8)
    p.add_argument("--include-primary-in-concepts", action="store_true")
    p.add_argument(
        "--balance-disruption-labels",
        action="store_true",
        help="Build an additional class-balanced JSONL for RL by balancing disruption labels.",
    )
    p.add_argument(
        "--disruption-balance-target",
        type=int,
        help="Per-label target row count for balanced RL dataset. Overrides --disruption-balance-total.",
    )
    p.add_argument(
        "--disruption-balance-total",
        type=int,
        help="Target total size for balanced dataset. Split evenly across 3 disruption labels.",
    )
    p.add_argument("--disruption-balance-seed", type=int, default=20260220)
    p.add_argument(
        "--disruption-balance-no-oversample",
        action="store_true",
        help="If set, do not oversample minority labels; downsample only to the minimum available count.",
    )
    p.add_argument(
        "--skip-balance-validation",
        action="store_true",
        help="Skip the hard guard that stops if a requested target label is impossible without oversampling.",
    )
    p.add_argument(
        "--disruption-balance-auto-expand",
        action="store_true",
        help=(
            "If set and --disruption-balance-no-oversample is set, rerun the base build with "
            "larger --n-papers until the requested target is feasible without oversampling."
        ),
    )
    p.add_argument(
        "--disruption-balance-auto-expand-factor",
        type=float,
        default=2.0,
        help="Growth factor used between retries when auto-expand is enabled (must be > 1.0).",
    )
    p.add_argument(
        "--disruption-balance-auto-expand-max-attempts",
        type=int,
        default=8,
        help="Maximum number of base-build attempts in auto-expand mode.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.tinker_sciscinet_parquet.exists():
        raise ValueError(f"Missing papers parquet: {args.tinker_sciscinet_parquet}")
    if not args.paper_fields_parquet.exists():
        raise ValueError(f"Missing paper fields parquet: {args.paper_fields_parquet}")
    if args.n_papers <= 0:
        raise ValueError("--n-papers must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.row_log_interval <= 0:
        raise ValueError("--row-log-interval must be > 0")
    if args.concepts_k <= 0:
        raise ValueError("--concepts-k must be > 0")
    if args.disruption_balance_auto_expand and not args.disruption_balance_no_oversample:
        raise ValueError(
            "--disruption-balance-auto-expand requires --disruption-balance-no-oversample "
            "because auto-expand only applies when oversampling is disallowed."
        )
    if args.disruption_balance_auto_expand and not args.balance_disruption_labels:
        raise ValueError("--disruption-balance-auto-expand requires --balance-disruption-labels.")
    if args.disruption_balance_target is not None and args.disruption_balance_target <= 0:
        raise ValueError("--disruption-balance-target must be > 0")
    if args.disruption_balance_total is not None and args.disruption_balance_total <= 0:
        raise ValueError("--disruption-balance-total must be > 0")
    if args.disruption_balance_auto_expand_factor is not None and args.disruption_balance_auto_expand_factor <= 1.0:
        raise ValueError("--disruption-balance-auto-expand-factor must be > 1.0")
    if args.disruption_balance_auto_expand_max_attempts is not None and args.disruption_balance_auto_expand_max_attempts <= 0:
        raise ValueError("--disruption-balance-auto-expand-max-attempts must be > 0")
    if (
        args.disruption_balance_target is not None
        and args.disruption_balance_total is not None
        and args.disruption_balance_total % 3 != 0
    ):
        raise ValueError("--disruption-balance-total should be divisible by 3 for equal label balancing.")


def run(cmd: list[str], workdir: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(workdir), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")


def build_base(args: argparse.Namespace, n_papers: int) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_jsonl = args.output_dir / f"{args.name}.jsonl"
    splits_json = args.output_dir / f"{args.name}.splits.json"
    metadata_json = args.output_dir / f"{args.name}.metadata.json"

    cmd: list[str] = [
        sys.executable,
        str(DATASET_BUILDER),
        "--parquet-primary",
        "tinker",
        "--tinker-sciscinet-parquet",
        str(args.tinker_sciscinet_parquet),
        "--n-papers",
        str(int(n_papers)),
        "--seed",
        str(args.seed),
        "--sciscinet-language",
        str(args.sciscinet_language),
        "--disruptive-threshold",
        str(args.disruptive_threshold),
        "--consolidating-threshold",
        str(args.consolidating_threshold),
        "--novelty-margin",
        str(args.novelty_margin),
        "--batch-size",
        str(args.batch_size),
        "--row-log-interval",
        str(args.row_log_interval),
        "--output",
        str(base_jsonl),
        "--splits-output",
        str(splits_json),
        "--metadata-output",
        str(metadata_json),
    ]

    if args.sciscinet_from_year is not None:
        cmd.extend(["--sciscinet-from-year", str(args.sciscinet_from_year)])
    if args.sciscinet_to_year is not None:
        cmd.extend(["--sciscinet-to-year", str(args.sciscinet_to_year)])
    if args.sciscinet_min_citations is not None:
        cmd.extend(["--sciscinet-min-citations", str(args.sciscinet_min_citations)])
    if args.include_retracted:
        cmd.append("--include-retracted")

    run(cmd, workdir=REPO_ROOT)
    return base_jsonl, splits_json, metadata_json


def enrich_fields(args: argparse.Namespace, base_jsonl: Path) -> tuple[Path, Path]:
    if args.skip_enrichment:
        return base_jsonl, Path()

    enriched_jsonl = args.output_dir / f"{args.name}.field_enriched.jsonl"
    report_json = args.output_dir / f"{args.name}.field_enriched.report.json"

    cmd: list[str] = [
        sys.executable,
        str(FIELD_ENRICHER),
        "--base-jsonl",
        str(base_jsonl),
        "--paper-fields-parquet",
        str(args.paper_fields_parquet),
        "--output-jsonl",
        str(enriched_jsonl),
        "--report-output",
        str(report_json),
        "--concepts-k",
        str(args.concepts_k),
        "--batch-size",
        str(args.batch_size),
        "--row-log-interval",
        str(args.row_log_interval),
    ]
    if args.include_primary_in_concepts:
        cmd.append("--include-primary-in-concepts")

    run(cmd, workdir=REPO_ROOT)
    return enriched_jsonl, report_json


def _iter_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                yield line_no, json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def balance_disruption_labels(
    args: argparse.Namespace,
    source_jsonl: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    if not args.balance_disruption_labels:
        return {"balanced_jsonl": str(source_jsonl), "balanced": False}

    rng = random.Random(int(args.disruption_balance_seed))
    buckets: dict[str, list[dict[str, Any]]] = {
        "disruptive": [],
        "consolidating": [],
        "neutral": [],
    }
    seen = 0
    missing: dict[str, int] = {"disruption_label": 0}

    for _line_no, rec in _iter_jsonl_records(source_jsonl):
        seen += 1
        label = str(rec.get("disruption_label", "")).strip().lower()
        if label in buckets:
            buckets[label].append(rec)
        else:
            missing["disruption_label"] += 1

    avail = {label: len(rows) for label, rows in buckets.items()}
    if args.disruption_balance_total is not None:
        per_label_target = max(1, int(args.disruption_balance_total) // 3)
    elif args.disruption_balance_target is not None:
        per_label_target = int(args.disruption_balance_target)
    else:
        per_label_target = min(avail.values()) if all(avail.values()) else 0

    if per_label_target <= 0:
        raise RuntimeError(f"Could not determine disruption balance target from source labels: {avail}")

    for label, count in avail.items():
        if count < per_label_target and args.disruption_balance_no_oversample and not args.skip_balance_validation:
            raise RuntimeError(
                f"Cannot satisfy target for label={label!r}: available={count}, target={per_label_target}. "
                "Use --disruption-balance-no-oversample=false (default) or lower target."
            )

    selected: list[dict[str, Any]] = []

    for label, rows in buckets.items():
        if not rows:
            raise RuntimeError(f"No rows for disruption label={label!r}; cannot produce balanced dataset.")
        if len(rows) >= per_label_target:
            selected.extend(rng.sample(rows, per_label_target))
        else:
            repeats, rem = divmod(per_label_target, len(rows))
            label_rows = []
            for _ in range(repeats):
                label_rows.extend(rows)
            label_rows.extend(rng.sample(rows, rem))
            selected.extend(label_rows)

    rng.shuffle(selected)
    with output_jsonl.open("w", encoding="utf-8") as out:
        for record in selected:
            out.write(json.dumps(record, ensure_ascii=True))
            out.write("\n")

    final_counts: dict[str, int] = {"disruptive": 0, "consolidating": 0, "neutral": 0}
    for rec in selected:
        final_counts[str(rec.get("disruption_label", "")).strip().lower()] += 1

    return {
        "balanced_jsonl": str(output_jsonl),
        "balanced": True,
        "balanced_seed": int(args.disruption_balance_seed),
        "balanced_source_records": int(seen),
        "balanced_per_label_target": int(per_label_target),
        "balanced_availability": avail,
        "balanced_counts": final_counts,
        "balanced_total": int(len(selected)),
        "balanced_oversampled": any(avail[label] < per_label_target for label in avail),
        "balanced_missing_disruption_label": int(missing["disruption_label"]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    validate_args(args)

    base_jsonl: Path
    splits_json: Path
    metadata_json: Path
    enriched_jsonl: Path = Path()
    report_json: Path = Path()
    max_attempts = max(1, int(args.disruption_balance_auto_expand_max_attempts))
    current_attempt = 0
    current_n_papers = int(args.n_papers)
    balanced_jsonl = args.output_dir / f"{args.name}.rl_balanced.jsonl"
    balance_payload: dict[str, Any] = {"balanced_jsonl": str(balanced_jsonl), "balanced": False}

    auto_expand_enabled = bool(
        args.balance_disruption_labels
        and args.disruption_balance_no_oversample
        and args.disruption_balance_auto_expand
    )

    while True:
        current_attempt += 1
        LOGGER.info(
            "Building source JSONL (attempt=%d/%d, n_papers=%d)",
            current_attempt,
            max_attempts,
            current_n_papers,
        )
        base_jsonl, splits_json, metadata_json = build_base(args, current_n_papers)
        enriched_jsonl, report_json = enrich_fields(args, base_jsonl)

        if not args.balance_disruption_labels:
            break

        try:
            balance_payload = balance_disruption_labels(
                args,
                enriched_jsonl if not args.skip_enrichment else base_jsonl,
                balanced_jsonl,
            )
            break
        except RuntimeError as exc:
            if auto_expand_enabled and (
                "Cannot satisfy target for label=" in str(exc)
                and current_attempt < max_attempts
            ):
                next_n_papers = max(
                    current_n_papers + 1,
                    int(current_n_papers * float(args.disruption_balance_auto_expand_factor)),
                )
                LOGGER.info(
                    f"target not feasible at n-papers={current_n_papers} (attempt {current_attempt}/{max_attempts}); "
                    f"retrying with n-papers={next_n_papers}"
                )
                current_n_papers = next_n_papers
                continue
            raise

        break

    LOGGER.info("base_jsonl=%s", base_jsonl)
    LOGGER.info("splits_json=%s", splits_json)
    LOGGER.info("metadata_json=%s", metadata_json)
    if not args.skip_enrichment:
        LOGGER.info("enriched_jsonl=%s", enriched_jsonl)
        LOGGER.info("enrich_report=%s", report_json)
    if args.balance_disruption_labels:
        LOGGER.info("rl_balanced_jsonl=%s", balance_payload["balanced_jsonl"])
        LOGGER.info(
            "rl_balance_payload=%s",
            json.dumps({k: v for k, v in balance_payload.items() if k != "balanced_jsonl"}, sort_keys=True),
        )


if __name__ == "__main__":
    main()
