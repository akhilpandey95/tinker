#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rlvr.metrics import (
    compute_metrics,
    flatten_metrics,
    load_json,
    load_jsonl,
    select_records_by_split,
    write_generic_csv,
    write_json,
    write_predictions_csv,
)
from training.rlvr.policy import RLVRPolicy
from training.rlvr.reward_spec import validate_reward_spec


def _predict_rows(policy: RLVRPolicy, records: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [policy.predict_joint_row(record) for record in records]


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    config = load_json(bootstrap_args.config) if bootstrap_args.config is not None else {}

    parser = argparse.ArgumentParser(description="Evaluate RLVR artifacts.")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)

    model_artifact_default = config.get("model_artifact")
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=Path(model_artifact_default) if model_artifact_default else None,
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path(config.get("dataset_jsonl", "data/synthetic/disruption_novelty_v1.jsonl")),
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path(config.get("splits_json", "data/synthetic/disruption_novelty_v1.splits.json")),
    )
    parser.add_argument("--split", default=str(config.get("split", "test")))

    output_dir_default = config.get("output_dir")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(output_dir_default) if output_dir_default else None,
        help="Defaults to the model artifact directory.",
    )
    parser.add_argument(
        "--tiny-max-samples",
        type=int,
        default=config.get("tiny_max_samples"),
        help="Optional max sample count for tiny local validation.",
    )
    parser.add_argument(
        "--n-calibration-bins",
        type=int,
        default=int(config.get("n_calibration_bins", 10)),
    )

    args = parser.parse_args(remaining)
    if args.model_artifact is None:
        raise ValueError("--model-artifact is required.")
    return args


def main() -> None:
    args = _parse_args()

    if args.n_calibration_bins <= 0:
        raise ValueError("--n-calibration-bins must be positive.")

    artifact = load_json(args.model_artifact)
    policy = RLVRPolicy.from_dict(artifact["policy_state"])
    reward_spec = artifact.get("reward_spec")
    if reward_spec is not None:
        validate_reward_spec(reward_spec)

    records = load_jsonl(args.dataset_jsonl)
    splits = load_json(args.splits_json)
    split_ids = splits.get("ids", {})

    eval_records = select_records_by_split(
        records,
        split_ids=split_ids,
        split_name=args.split,
        max_items=args.tiny_max_samples,
    )
    if not eval_records:
        raise ValueError("No evaluation records selected. Check --split and data files.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.model_artifact.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _predict_rows(policy, eval_records)
    metrics, calibration = compute_metrics(rows, n_bins=args.n_calibration_bins)

    predictions_csv = output_dir / f"predictions_{args.split}.csv"
    metrics_json = output_dir / f"metrics_{args.split}.json"
    metrics_csv = output_dir / f"metrics_{args.split}.csv"
    calibration_disruption_csv = output_dir / f"calibration_{args.split}_disruption.csv"
    calibration_novelty_csv = output_dir / f"calibration_{args.split}_novelty.csv"
    eval_manifest = output_dir / f"eval_manifest_{args.split}.json"

    write_predictions_csv(predictions_csv, rows)
    write_json(metrics_json, metrics)
    write_generic_csv(metrics_csv, ["scope", "metric", "value"], flatten_metrics(metrics))

    calibration_fields = [
        "task",
        "bin_index",
        "bin_lower",
        "bin_upper",
        "count",
        "avg_confidence",
        "accuracy",
        "gap",
    ]
    write_generic_csv(calibration_disruption_csv, calibration_fields, calibration["disruption"])
    write_generic_csv(calibration_novelty_csv, calibration_fields, calibration["novelty"])

    manifest = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "model_artifact": str(args.model_artifact),
        "dataset_jsonl": str(args.dataset_jsonl),
        "splits_json": str(args.splits_json),
        "record_count": len(eval_records),
        "artifacts": {
            "predictions_csv": str(predictions_csv),
            "metrics_json": str(metrics_json),
            "metrics_csv": str(metrics_csv),
            "calibration_disruption_csv": str(calibration_disruption_csv),
            "calibration_novelty_csv": str(calibration_novelty_csv),
        },
    }
    write_json(eval_manifest, manifest)

    print(f"EVAL_SPLIT={args.split}")
    print(f"RECORD_COUNT={len(eval_records)}")
    print(f"METRICS_JSON={metrics_json}")
    print(f"JOINT_ACCURACY={metrics['overall']['joint_exact_match_accuracy']:.4f}")


if __name__ == "__main__":
    main()
