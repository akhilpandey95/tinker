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
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from prompts import (  # noqa: E402
    DISRUPTION_LABELS,
    NOVELTY_LABELS,
    build_joint_example,
    format_joint_response,
    parse_joint_response,
)


@dataclass(frozen=True)
class HeuristicParams:
    disruption_scale: float
    disruption_neutral_bias: float
    disruption_threshold: float
    novelty_scale: float
    novelty_balanced_bias: float
    novelty_margin: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "disruption_scale": self.disruption_scale,
            "disruption_neutral_bias": self.disruption_neutral_bias,
            "disruption_threshold": self.disruption_threshold,
            "novelty_scale": self.novelty_scale,
            "novelty_balanced_bias": self.novelty_balanced_bias,
            "novelty_margin": self.novelty_margin,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            records.append(item)
    return records


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _parse_float_list(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _softmax(logits: Mapping[str, float]) -> Dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exps = {label: math.exp(val - max_logit) for label, val in logits.items()}
    denom = sum(exps.values())
    if denom <= 0:
        size = float(len(logits))
        return {label: 1.0 / size for label in logits}
    return {label: exps[label] / denom for label in logits}


def _argmax(probs: Mapping[str, float], labels: Sequence[str]) -> str:
    best = labels[0]
    best_score = float("-inf")
    for label in labels:
        score = float(probs.get(label, 0.0))
        if score > best_score:
            best = label
            best_score = score
    return best


def _heuristic_logits(record: Mapping[str, Any], params: HeuristicParams) -> Dict[str, Dict[str, float]]:
    cd_index = float(record["cd_index"])
    novelty_score = float(record["novelty_score"])
    conventionality_score = float(record["conventionality_score"])
    delta = novelty_score - conventionality_score

    disruption_logits = {
        "disruptive": params.disruption_scale * (cd_index - params.disruption_threshold),
        "consolidating": params.disruption_scale * ((-cd_index) - params.disruption_threshold),
        "neutral": params.disruption_neutral_bias - params.disruption_scale * abs(cd_index),
    }
    novelty_logits = {
        "novel": params.novelty_scale * (delta - params.novelty_margin),
        "conventional": params.novelty_scale * ((-delta) - params.novelty_margin),
        "balanced": params.novelty_balanced_bias - params.novelty_scale * abs(delta),
    }
    return {"disruption": disruption_logits, "novelty": novelty_logits}


def _temperature_scale_logits(logits: Mapping[str, float], temperature: float) -> Dict[str, float]:
    t = max(float(temperature), 1e-6)
    return {label: value / t for label, value in logits.items()}


def _predict_row(
    record: Mapping[str, Any],
    params: HeuristicParams,
    temperatures: Mapping[str, float],
) -> Dict[str, Any]:
    logits = _heuristic_logits(record, params)
    disruption_probs = _softmax(_temperature_scale_logits(logits["disruption"], temperatures["disruption"]))
    novelty_probs = _softmax(_temperature_scale_logits(logits["novelty"], temperatures["novelty"]))

    pred_disruption = _argmax(disruption_probs, DISRUPTION_LABELS)
    pred_novelty = _argmax(novelty_probs, NOVELTY_LABELS)

    reasoning = (
        "CD index thresholding and novelty-conventionality delta support this joint label prediction."
    )
    response = format_joint_response(pred_disruption, pred_novelty, reasoning)
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
    return row


def _predict_rows(
    records: Sequence[Mapping[str, Any]],
    params: HeuristicParams,
    temperatures: Mapping[str, float],
) -> List[Dict[str, Any]]:
    return [_predict_row(record, params=params, temperatures=temperatures) for record in records]


def _build_calibration_bins(rows: Sequence[Mapping[str, Any]], task: str, n_bins: int) -> Tuple[List[Dict[str, Any]], float]:
    task = task.lower()
    if task not in {"disruption", "novelty"}:
        raise ValueError(f"Unsupported task for calibration: {task}")

    bins: List[List[Tuple[float, int]]] = [[] for _ in range(n_bins)]
    for row in rows:
        conf = float(row[f"{task}_confidence"])
        correct = int(row[f"{task}_correct"])
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, correct))

    total = max(len(rows), 1)
    ece = 0.0
    result: List[Dict[str, Any]] = []
    for idx, bucket in enumerate(bins):
        lower = idx / n_bins
        upper = (idx + 1) / n_bins
        if not bucket:
            result.append(
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

        avg_conf = sum(pair[0] for pair in bucket) / len(bucket)
        acc = sum(pair[1] for pair in bucket) / len(bucket)
        gap = abs(avg_conf - acc)
        ece += (len(bucket) / total) * gap
        result.append(
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

    return result, ece


def _compute_task_metrics(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    labels: Sequence[str],
    n_bins: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    gold_key = f"gold_{task}"
    pred_key = f"pred_{task}"
    correct_key = f"{task}_correct"

    total = len(rows)
    if total == 0:
        empty = {
            "support": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "nll": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "per_class": {label: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for label in labels},
            "confusion_matrix": {gold: {pred: 0 for pred in labels} for gold in labels},
        }
        bins, _ = _build_calibration_bins(rows, task, n_bins=n_bins)
        return empty, bins

    accuracy = sum(int(row[correct_key]) for row in rows) / total

    confusion: Dict[str, Dict[str, int]] = {gold: {pred: 0 for pred in labels} for gold in labels}
    for row in rows:
        gold = str(row[gold_key])
        pred = str(row[pred_key])
        if gold in confusion and pred in confusion[gold]:
            confusion[gold][pred] += 1

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
        f1_scores.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    nll_sum = 0.0
    brier_sum = 0.0
    eps = 1e-12
    for row in rows:
        gold = str(row[gold_key])
        probs = {label: float(row[f"{task}_prob_{label}"]) for label in labels}
        gold_prob = max(probs.get(gold, 0.0), eps)
        nll_sum += -math.log(gold_prob)
        for label in labels:
            target = 1.0 if label == gold else 0.0
            brier_sum += (probs[label] - target) ** 2

    bins, ece = _build_calibration_bins(rows, task, n_bins=n_bins)

    metrics = {
        "support": total,
        "accuracy": accuracy,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "nll": nll_sum / total,
        "brier": brier_sum / total,
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
        "joint_exact_match_accuracy": (
            sum(int(row["joint_correct"]) for row in rows) / total if total else 0.0
        ),
        "parse_error_rate": (
            sum(1 for row in rows if row.get("parse_error")) / total if total else 0.0
        ),
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


def _fit_temperature(
    logits_rows: Sequence[Mapping[str, Any]],
    gold_key: str,
    logit_key: str,
    labels: Sequence[str],
    candidates: Sequence[float],
) -> Tuple[float, List[Dict[str, float]]]:
    if not logits_rows:
        return 1.0, []

    search: List[Dict[str, float]] = []
    best_temp = 1.0
    best_nll = float("inf")
    eps = 1e-12

    for temp in candidates:
        nll_sum = 0.0
        for row in logits_rows:
            logits = row[logit_key]
            scaled = _temperature_scale_logits(logits, temp)
            probs = _softmax(scaled)
            gold_label = str(row[gold_key])
            gold_prob = max(float(probs.get(gold_label, 0.0)), eps)
            nll_sum += -math.log(gold_prob)
        avg_nll = nll_sum / len(logits_rows)
        item = {"temperature": float(temp), "nll": avg_nll}
        search.append(item)
        if avg_nll < best_nll:
            best_nll = avg_nll
            best_temp = float(temp)

    return best_temp, search


def _collect_raw_logits(
    records: Sequence[Mapping[str, Any]],
    params: HeuristicParams,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        logits = _heuristic_logits(record, params)
        rows.append(
            {
                "openalex_id": str(record["openalex_id"]),
                "gold_disruption": str(record["disruption_label"]).lower(),
                "gold_novelty": str(record["novelty_label"]).lower(),
                "disruption_logits": logits["disruption"],
                "novelty_logits": logits["novelty"],
            }
        )
    return rows


def _write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    ordered_fields = [
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
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ordered_fields})


def _write_calibration_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["task", "bin_index", "bin_lower", "bin_upper", "count", "avg_confidence", "accuracy", "gap"]
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
        task_metrics = metrics[task]
        for metric in ("accuracy", "macro_f1", "nll", "brier", "ece"):
            rows.append(
                {
                    "scope": task,
                    "metric": metric,
                    "value": task_metrics[metric],
                }
            )
    return rows


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["scope", "metric", "value"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _select_records_by_split(
    records: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Sequence[str]],
    split_name: str,
    max_items: Optional[int],
) -> List[Dict[str, Any]]:
    id_to_record = {str(record["openalex_id"]): dict(record) for record in records}
    ids = list(split_ids.get(split_name, []))
    selected = [id_to_record[record_id] for record_id in ids if record_id in id_to_record]
    if max_items is not None:
        selected = selected[: max(0, int(max_items))]
    return selected


def _grid_search_params(
    train_records: Sequence[Mapping[str, Any]],
    disruption_scales: Sequence[float],
    neutral_biases: Sequence[float],
    novelty_scales: Sequence[float],
    balanced_biases: Sequence[float],
    disruption_threshold: float,
    novelty_margin: float,
) -> Tuple[HeuristicParams, List[Dict[str, Any]]]:
    leaderboard: List[Dict[str, Any]] = []
    best: Optional[Tuple[Tuple[float, float, float], HeuristicParams]] = None

    for d_scale in disruption_scales:
        for d_bias in neutral_biases:
            for n_scale in novelty_scales:
                for n_bias in balanced_biases:
                    params = HeuristicParams(
                        disruption_scale=float(d_scale),
                        disruption_neutral_bias=float(d_bias),
                        disruption_threshold=float(disruption_threshold),
                        novelty_scale=float(n_scale),
                        novelty_balanced_bias=float(n_bias),
                        novelty_margin=float(novelty_margin),
                    )
                    rows = _predict_rows(
                        train_records,
                        params=params,
                        temperatures={"disruption": 1.0, "novelty": 1.0},
                    )
                    metrics, _ = _compute_metrics(rows, n_bins=10)
                    score = (
                        float(metrics["overall"]["joint_exact_match_accuracy"]),
                        float(metrics["disruption"]["accuracy"]),
                        float(metrics["novelty"]["accuracy"]),
                    )
                    leaderboard.append(
                        {
                            "disruption_scale": params.disruption_scale,
                            "disruption_neutral_bias": params.disruption_neutral_bias,
                            "novelty_scale": params.novelty_scale,
                            "novelty_balanced_bias": params.novelty_balanced_bias,
                            "joint_exact_match_accuracy": score[0],
                            "disruption_accuracy": score[1],
                            "novelty_accuracy": score[2],
                        }
                    )
                    if best is None or score > best[0]:
                        best = (score, params)

    if best is None:
        raise ValueError("Grid search failed: no parameter combinations evaluated.")

    leaderboard.sort(
        key=lambda item: (
            item["joint_exact_match_accuracy"],
            item["disruption_accuracy"],
            item["novelty_accuracy"],
        ),
        reverse=True,
    )
    return best[1], leaderboard


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()

    config: Dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config = _load_json(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="QDoRA + FSDP SFT baseline (deterministic dry-run backend).")
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.get("output_dir", "results/sft")),
    )
    parser.add_argument("--run-name", default=str(config.get("run_name", "qdora_fsdp_baseline")))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 20260220)))
    parser.add_argument(
        "--train-split",
        default=str(config.get("train_split", "train")),
        help="Split name used for train examples.",
    )
    parser.add_argument(
        "--val-split",
        default=str(config.get("val_split", "val")),
        help="Split name used for calibration/validation metrics.",
    )
    parser.add_argument(
        "--tiny-max-train",
        type=int,
        default=config.get("tiny_max_train"),
        help="Optional max number of train examples for tiny local runs.",
    )
    parser.add_argument(
        "--tiny-max-val",
        type=int,
        default=config.get("tiny_max_val"),
        help="Optional max number of val examples for tiny local runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(config.get("dry_run", False)),
        help="Run tiny deterministic dry-run backend (default for local validation).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=bool(config.get("overwrite", False)),
        help="Overwrite existing run directory if it already exists.",
    )
    parser.add_argument(
        "--n-calibration-bins",
        type=int,
        default=int(config.get("n_calibration_bins", 10)),
    )
    parser.add_argument(
        "--disruption-threshold",
        type=float,
        default=float(config.get("disruption_threshold", 0.1)),
    )
    parser.add_argument(
        "--novelty-margin",
        type=float,
        default=float(config.get("novelty_margin", 0.15)),
    )
    parser.add_argument(
        "--disruption-scales",
        default=str(config.get("disruption_scales", "3,5,8,12")),
        help="Comma-separated scale candidates for disruption logits.",
    )
    parser.add_argument(
        "--neutral-biases",
        default=str(config.get("neutral_biases", "0.0,0.5,1.0")),
        help="Comma-separated bias candidates for disruption neutral class.",
    )
    parser.add_argument(
        "--novelty-scales",
        default=str(config.get("novelty_scales", "3,5,8,12")),
        help="Comma-separated scale candidates for novelty logits.",
    )
    parser.add_argument(
        "--balanced-biases",
        default=str(config.get("balanced_biases", "0.0,0.5,1.0")),
        help="Comma-separated bias candidates for novelty balanced class.",
    )
    parser.add_argument(
        "--temperature-candidates",
        default=str(config.get("temperature_candidates", "0.5,0.75,1.0,1.25,1.5,2.0,3.0")),
        help="Comma-separated temperature candidates for post-hoc calibration.",
    )

    args = parser.parse_args(remaining)
    return args


def main() -> None:
    args = _parse_args()

    if args.n_calibration_bins <= 0:
        raise ValueError("--n-calibration-bins must be positive.")

    records = _load_jsonl(args.dataset_jsonl)
    splits = _load_json(args.splits_json)
    split_ids = splits.get("ids", {})

    train_records = _select_records_by_split(
        records,
        split_ids=split_ids,
        split_name=args.train_split,
        max_items=args.tiny_max_train,
    )
    val_records = _select_records_by_split(
        records,
        split_ids=split_ids,
        split_name=args.val_split,
        max_items=args.tiny_max_val,
    )

    if not train_records:
        raise ValueError("No train records selected. Check split names and data files.")
    if not val_records:
        raise ValueError("No validation records selected. Check split names and data files.")

    run_dir = args.output_dir / args.run_name
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    train_examples = [build_joint_example(record) for record in train_records]
    val_examples = [build_joint_example(record) for record in val_records]

    _write_jsonl(
        run_dir / "train_examples.jsonl",
        (
            {
                "openalex_id": example.openalex_id,
                "prompt": example.prompt,
                "target": example.target,
                "disruption_label": example.disruption_label,
                "novelty_label": example.novelty_label,
            }
            for example in train_examples
        ),
    )
    _write_jsonl(
        run_dir / "val_examples.jsonl",
        (
            {
                "openalex_id": example.openalex_id,
                "prompt": example.prompt,
                "target": example.target,
                "disruption_label": example.disruption_label,
                "novelty_label": example.novelty_label,
            }
            for example in val_examples
        ),
    )

    disruption_scales = _parse_float_list(args.disruption_scales)
    neutral_biases = _parse_float_list(args.neutral_biases)
    novelty_scales = _parse_float_list(args.novelty_scales)
    balanced_biases = _parse_float_list(args.balanced_biases)
    temp_candidates = _parse_float_list(args.temperature_candidates)

    if not disruption_scales or not neutral_biases or not novelty_scales or not balanced_biases:
        raise ValueError("Grid-search candidate lists must be non-empty.")
    if not temp_candidates:
        raise ValueError("Temperature candidate list must be non-empty.")

    best_params, leaderboard = _grid_search_params(
        train_records,
        disruption_scales=disruption_scales,
        neutral_biases=neutral_biases,
        novelty_scales=novelty_scales,
        balanced_biases=balanced_biases,
        disruption_threshold=args.disruption_threshold,
        novelty_margin=args.novelty_margin,
    )

    search_fields = [
        "disruption_scale",
        "disruption_neutral_bias",
        "novelty_scale",
        "novelty_balanced_bias",
        "joint_exact_match_accuracy",
        "disruption_accuracy",
        "novelty_accuracy",
    ]
    with (run_dir / "hyperparameter_search.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=search_fields)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({field: row.get(field) for field in search_fields})

    raw_val_logits = _collect_raw_logits(val_records, best_params)
    best_disruption_temp, disruption_search = _fit_temperature(
        raw_val_logits,
        gold_key="gold_disruption",
        logit_key="disruption_logits",
        labels=DISRUPTION_LABELS,
        candidates=temp_candidates,
    )
    best_novelty_temp, novelty_search = _fit_temperature(
        raw_val_logits,
        gold_key="gold_novelty",
        logit_key="novelty_logits",
        labels=NOVELTY_LABELS,
        candidates=temp_candidates,
    )

    _write_generic_csv(
        run_dir / "temperature_search_disruption.csv",
        fields=["temperature", "nll"],
        rows=disruption_search,
    )
    _write_generic_csv(
        run_dir / "temperature_search_novelty.csv",
        fields=["temperature", "nll"],
        rows=novelty_search,
    )

    temperatures = {
        "disruption": best_disruption_temp,
        "novelty": best_novelty_temp,
    }

    train_rows = _predict_rows(train_records, params=best_params, temperatures=temperatures)
    val_rows = _predict_rows(val_records, params=best_params, temperatures=temperatures)

    train_metrics, train_calibration = _compute_metrics(train_rows, n_bins=args.n_calibration_bins)
    val_metrics, val_calibration = _compute_metrics(val_rows, n_bins=args.n_calibration_bins)

    _write_predictions_csv(run_dir / "predictions_train.csv", train_rows)
    _write_predictions_csv(run_dir / "predictions_val.csv", val_rows)
    _write_calibration_csv(run_dir / "calibration_train_disruption.csv", train_calibration["disruption"])
    _write_calibration_csv(run_dir / "calibration_train_novelty.csv", train_calibration["novelty"])
    _write_calibration_csv(run_dir / "calibration_val_disruption.csv", val_calibration["disruption"])
    _write_calibration_csv(run_dir / "calibration_val_novelty.csv", val_calibration["novelty"])

    _write_json(run_dir / "metrics_train.json", train_metrics)
    _write_json(run_dir / "metrics_val.json", val_metrics)
    _write_metrics_csv(run_dir / "metrics_train.csv", _flatten_metrics(train_metrics))
    _write_metrics_csv(run_dir / "metrics_val.csv", _flatten_metrics(val_metrics))

    model_artifact = {
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "qdora_fsdp_baseline",
        "backend": "heuristic_dry_run" if args.dry_run else "heuristic_reference",
        "prompt_contract": "combined_impact_v1",
        "label_targets": {
            "disruption": list(DISRUPTION_LABELS),
            "novelty": list(NOVELTY_LABELS),
        },
        "heuristic_params": best_params.to_dict(),
        "temperature_calibration": temperatures,
        "dataset": {
            "dataset_jsonl": str(args.dataset_jsonl),
            "splits_json": str(args.splits_json),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "train_count": len(train_records),
            "val_count": len(val_records),
        },
        "dry_run": bool(args.dry_run),
    }
    _write_json(run_dir / "model_artifact.json", model_artifact)

    run_manifest = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "n_calibration_bins": int(args.n_calibration_bins),
        "config": {
            "disruption_threshold": args.disruption_threshold,
            "novelty_margin": args.novelty_margin,
            "disruption_scales": disruption_scales,
            "neutral_biases": neutral_biases,
            "novelty_scales": novelty_scales,
            "balanced_biases": balanced_biases,
            "temperature_candidates": temp_candidates,
        },
        "artifacts": {
            "model_artifact": str(run_dir / "model_artifact.json"),
            "metrics_train_json": str(run_dir / "metrics_train.json"),
            "metrics_val_json": str(run_dir / "metrics_val.json"),
            "predictions_train_csv": str(run_dir / "predictions_train.csv"),
            "predictions_val_csv": str(run_dir / "predictions_val.csv"),
            "calibration_val_disruption_csv": str(run_dir / "calibration_val_disruption.csv"),
            "calibration_val_novelty_csv": str(run_dir / "calibration_val_novelty.csv"),
        },
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)

    print(f"RUN_DIR={run_dir}")
    print(f"MODEL_ARTIFACT={run_dir / 'model_artifact.json'}")
    print(f"VAL_JOINT_ACCURACY={val_metrics['overall']['joint_exact_match_accuracy']:.4f}")


if __name__ == "__main__":
    main()
