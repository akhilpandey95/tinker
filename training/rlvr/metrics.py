# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from training.rlvr.prompt_contract import DISRUPTION_LABELS, NOVELTY_LABELS


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def select_records_by_split(
    records: Sequence[Mapping[str, Any]],
    split_ids: Mapping[str, Sequence[str]],
    split_name: str,
    max_items: int | None,
) -> List[Dict[str, Any]]:
    id_to_record = {str(record["openalex_id"]): dict(record) for record in records}
    ids = list(split_ids.get(split_name, []))
    selected = [id_to_record[record_id] for record_id in ids if record_id in id_to_record]
    if max_items is not None:
        selected = selected[: max(0, int(max_items))]
    return selected


def build_calibration_bins(
    rows: Sequence[Mapping[str, Any]],
    task: str,
    n_bins: int,
) -> Tuple[List[Dict[str, Any]], float]:
    task = task.lower()
    if task not in {"disruption", "novelty"}:
        raise ValueError(f"Unsupported task for calibration: {task}")

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

        avg_conf = sum(pair[0] for pair in bucket) / len(bucket)
        acc = sum(pair[1] for pair in bucket) / len(bucket)
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


def compute_task_metrics(
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
        bins, _ = build_calibration_bins(rows, task=task, n_bins=n_bins)
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

    bins, ece = build_calibration_bins(rows, task=task, n_bins=n_bins)

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


def compute_metrics(rows: Sequence[Mapping[str, Any]], n_bins: int) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    disruption_metrics, disruption_bins = compute_task_metrics(
        rows,
        task="disruption",
        labels=DISRUPTION_LABELS,
        n_bins=n_bins,
    )
    novelty_metrics, novelty_bins = compute_task_metrics(
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


def flatten_metrics(metrics: Mapping[str, Any]) -> List[Dict[str, Any]]:
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


def write_predictions_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def write_generic_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
