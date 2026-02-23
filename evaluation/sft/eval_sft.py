#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TRAINING_SFT_DIR = Path(__file__).resolve().parents[2] / "training" / "sft"
if str(TRAINING_SFT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_SFT_DIR))

from prompts import (  # noqa: E402
    DISRUPTION_LABELS,
    NOVELTY_LABELS,
    format_joint_response,
    parse_joint_response,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def _softmax(logits: Mapping[str, float]) -> Dict[str, float]:
    max_logit = max(logits.values())
    exps = {label: math.exp(value - max_logit) for label, value in logits.items()}
    denom = sum(exps.values())
    if denom <= 0:
        size = float(len(logits))
        return {label: 1.0 / size for label in logits}
    return {label: exps[label] / denom for label in logits}


def _argmax(probs: Mapping[str, float], labels: Sequence[str]) -> str:
    best_label = labels[0]
    best_score = float("-inf")
    for label in labels:
        score = float(probs.get(label, 0.0))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _temperature_scale(logits: Mapping[str, float], temperature: float) -> Dict[str, float]:
    t = max(float(temperature), 1e-6)
    return {label: value / t for label, value in logits.items()}


def _heuristic_logits(record: Mapping[str, Any], params: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
    cd_index = float(record["cd_index"])
    novelty_score = float(record["novelty_score"])
    conventionality_score = float(record["conventionality_score"])
    delta = novelty_score - conventionality_score

    disruption_scale = float(params["disruption_scale"])
    disruption_threshold = float(params["disruption_threshold"])
    disruption_neutral_bias = float(params["disruption_neutral_bias"])

    novelty_scale = float(params["novelty_scale"])
    novelty_margin = float(params["novelty_margin"])
    novelty_balanced_bias = float(params["novelty_balanced_bias"])

    disruption_logits = {
        "disruptive": disruption_scale * (cd_index - disruption_threshold),
        "consolidating": disruption_scale * ((-cd_index) - disruption_threshold),
        "neutral": disruption_neutral_bias - disruption_scale * abs(cd_index),
    }
    novelty_logits = {
        "novel": novelty_scale * (delta - novelty_margin),
        "conventional": novelty_scale * ((-delta) - novelty_margin),
        "balanced": novelty_balanced_bias - novelty_scale * abs(delta),
    }
    return {"disruption": disruption_logits, "novelty": novelty_logits}


def _predict_rows(
    records: Sequence[Mapping[str, Any]],
    params: Mapping[str, float],
    temperatures: Mapping[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        logits = _heuristic_logits(record, params)
        disruption_probs = _softmax(_temperature_scale(logits["disruption"], temperatures["disruption"]))
        novelty_probs = _softmax(_temperature_scale(logits["novelty"], temperatures["novelty"]))

        pred_disruption = _argmax(disruption_probs, DISRUPTION_LABELS)
        pred_novelty = _argmax(novelty_probs, NOVELTY_LABELS)

        response = format_joint_response(
            pred_disruption,
            pred_novelty,
            "CD and novelty-conventionality signals support this prediction.",
        )
        parsed = parse_joint_response(response)

        row: Dict[str, Any] = {
            "openalex_id": str(record["openalex_id"]),
            "gold_disruption": str(record["disruption_label"]).lower(),
            "gold_novelty": str(record["novelty_label"]).lower(),
            "pred_disruption": str(parsed["disruption_label"]) if parsed["disruption_label"] else None,
            "pred_novelty": str(parsed["novelty_label"]) if parsed["novelty_label"] else None,
            "disruption_confidence": max(disruption_probs.values()),
            "novelty_confidence": max(novelty_probs.values()),
            "parse_error": parsed["parse_error"],
            "response_text": response,
        }
        for label in DISRUPTION_LABELS:
            row[f"disruption_prob_{label}"] = disruption_probs[label]
        for label in NOVELTY_LABELS:
            row[f"novelty_prob_{label}"] = novelty_probs[label]

        row["disruption_correct"] = int(row["pred_disruption"] == row["gold_disruption"])
        row["novelty_correct"] = int(row["pred_novelty"] == row["gold_novelty"])
        row["joint_correct"] = int(row["disruption_correct"] and row["novelty_correct"])
        rows.append(row)

    return rows


def _build_calibration_bins(rows: Sequence[Mapping[str, Any]], task: str, n_bins: int) -> Tuple[List[Dict[str, Any]], float]:
    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for row in rows:
        conf = float(row[f"{task}_confidence"])
        correct = int(row[f"{task}_correct"])
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    total = max(1, len(rows))
    ece = 0.0
    out: List[Dict[str, Any]] = []
    for idx, bucket in enumerate(bins):
        lower = idx / n_bins
        upper = (idx + 1) / n_bins
        if not bucket:
            out.append(
                {
                    "task": task,
                    "bin_index": idx,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = sum(item[0] for item in bucket) / len(bucket)
        acc = sum(item[1] for item in bucket) / len(bucket)
        gap = abs(avg_conf - acc)
        ece += (len(bucket) / total) * gap
        out.append(
            {
                "task": task,
                "bin_index": idx,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": len(bucket),
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": gap,
            }
        )

    return out, ece


def _compute_task_metrics(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    labels: Sequence[str],
    n_bins: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total = len(rows)
    gold_key = f"gold_{task}"
    pred_key = f"pred_{task}"
    correct_key = f"{task}_correct"

    confusion: Dict[str, Dict[str, int]] = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in rows:
        gold = str(row[gold_key])
        pred = str(row[pred_key])
        if gold in confusion and pred in confusion[gold]:
            confusion[gold][pred] += 1

    accuracy = sum(int(row[correct_key]) for row in rows) / total if total else 0.0

    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label][pred] for pred in labels)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_scores.append(f1)

    nll_sum = 0.0
    brier_sum = 0.0
    eps = 1e-12
    for row in rows:
        gold = str(row[gold_key])
        probs = {label: float(row[f"{task}_prob_{label}"]) for label in labels}
        nll_sum += -math.log(max(probs.get(gold, 0.0), eps))
        for label in labels:
            target = 1.0 if label == gold else 0.0
            brier_sum += (probs[label] - target) ** 2

    bins, ece = _build_calibration_bins(rows, task=task, n_bins=n_bins)

    metrics = {
        "support": total,
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "nll": nll_sum / total if total else 0.0,
        "brier": brier_sum / total if total else 0.0,
        "ece": ece,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }
    return metrics, bins


def _compute_metrics(rows: Sequence[Mapping[str, Any]], n_bins: int) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    disruption_metrics, disruption_bins = _compute_task_metrics(
        rows,
        task="disruption",
        labels=DISRUPTION_LABELS,
        n_bins=n_bins,
    )
    novelty_metrics, novelty_bins = _compute_task_metrics(
        rows,
        task="novelty",
        labels=NOVELTY_LABELS,
        n_bins=n_bins,
    )

    total = len(rows)
    overall = {
        "support": total,
        "joint_exact_match_accuracy": sum(int(row["joint_correct"]) for row in rows) / total if total else 0.0,
        "parse_error_rate": sum(1 for row in rows if row.get("parse_error")) / total if total else 0.0,
    }

    metrics = {
        "overall": overall,
        "disruption": disruption_metrics,
        "novelty": novelty_metrics,
    }
    calibration = {
        "disruption": disruption_bins,
        "novelty": novelty_bins,
    }
    return metrics, calibration


def _flatten_metrics(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "scope": "overall",
            "metric": "joint_exact_match_accuracy",
            "value": metrics["overall"]["joint_exact_match_accuracy"],
        }
    )
    rows.append(
        {
            "scope": "overall",
            "metric": "parse_error_rate",
            "value": metrics["overall"]["parse_error_rate"],
        }
    )
    for task in ("disruption", "novelty"):
        for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            rows.append(
                {
                    "scope": task,
                    "metric": metric,
                    "value": metrics[task][metric],
                }
            )
    return rows


def _write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "openalex_id",
        "gold_disruption",
        "pred_disruption",
        "disruption_correct",
        "disruption_confidence",
        "disruption_prob_disruptive",
        "disruption_prob_consolidating",
        "disruption_prob_neutral",
        "gold_novelty",
        "pred_novelty",
        "novelty_correct",
        "novelty_confidence",
        "novelty_prob_novel",
        "novelty_prob_conventional",
        "novelty_prob_balanced",
        "joint_correct",
        "parse_error",
        "response_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_generic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _select_records(
    records: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Sequence[str]],
    split_name: str,
    max_items: Optional[int],
) -> List[Dict[str, Any]]:
    id_to_record = {str(row["openalex_id"]): dict(row) for row in records}
    ids = list(split_ids.get(split_name, []))
    selected = [id_to_record[record_id] for record_id in ids if record_id in id_to_record]
    if max_items is not None:
        selected = selected[: max(0, int(max_items))]
    return selected


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()
    config = _load_json(bootstrap_args.config) if bootstrap_args.config is not None else {}

    parser = argparse.ArgumentParser(description="Evaluate SFT baseline artifacts.")
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
        help="Optional max example count for tiny local validation.",
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

    model_artifact = _load_json(args.model_artifact)
    heuristic_params = model_artifact["heuristic_params"]
    temperatures = model_artifact.get("temperature_calibration", {"disruption": 1.0, "novelty": 1.0})

    records = _load_jsonl(args.dataset_jsonl)
    splits = _load_json(args.splits_json)
    split_ids = splits.get("ids", {})

    eval_records = _select_records(
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

    rows = _predict_rows(eval_records, params=heuristic_params, temperatures=temperatures)
    metrics, calibration = _compute_metrics(rows, n_bins=args.n_calibration_bins)

    predictions_csv = output_dir / f"predictions_{args.split}.csv"
    metrics_json = output_dir / f"metrics_{args.split}.json"
    metrics_csv = output_dir / f"metrics_{args.split}.csv"
    calibration_disruption_csv = output_dir / f"calibration_{args.split}_disruption.csv"
    calibration_novelty_csv = output_dir / f"calibration_{args.split}_novelty.csv"
    eval_manifest = output_dir / f"eval_manifest_{args.split}.json"

    _write_predictions_csv(predictions_csv, rows)
    _write_json(metrics_json, metrics)
    _write_generic_csv(metrics_csv, ["scope", "metric", "value"], _flatten_metrics(metrics))
    _write_generic_csv(
        calibration_disruption_csv,
        ["task", "bin_index", "bin_lower", "bin_upper", "count", "avg_confidence", "accuracy", "gap"],
        calibration["disruption"],
    )
    _write_generic_csv(
        calibration_novelty_csv,
        ["task", "bin_index", "bin_lower", "bin_upper", "count", "avg_confidence", "accuracy", "gap"],
        calibration["novelty"],
    )

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
    _write_json(eval_manifest, manifest)

    print(f"EVAL_SPLIT={args.split}")
    print(f"RECORD_COUNT={len(eval_records)}")
    print(f"METRICS_JSON={metrics_json}")
    print(f"JOINT_ACCURACY={metrics['overall']['joint_exact_match_accuracy']:.4f}")


if __name__ == "__main__":
    main()
