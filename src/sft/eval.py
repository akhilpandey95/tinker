# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import wandb
from transformers import TrainerCallback

LABELS = ("disruptive", "consolidating", "neutral")
STRICT_CONFIDENCE_VALUES = ("low", "medium", "high")
CONFIDENCE_TO_SCORE = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
}


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True))
            handle.write("\n")


def normalize_completion(completion: str) -> str:
    text = completion.strip()
    for suffix in ("<|endoftext|>", "<|eot_id|>"):
        text = text.replace(suffix, "").strip()
    return text


def clean_completion(completion: str) -> str:
    import re

    text = normalize_completion(completion)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    elif text.startswith("```"):
        text = text[len("```") :].strip()

    if text.endswith("```"):
        text = text[: -len("```")].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]

    return text.strip()


def validate_payload(payload: object) -> tuple[str, str | None, str | None]:
    if not isinstance(payload, dict):
        return "not_json_object", None, None

    if set(payload) != {"disruption_label", "confidence"}:
        return "bad_keys", None, None

    label = payload.get("disruption_label")
    confidence = payload.get("confidence")

    if not isinstance(label, str) or label.lower() not in LABELS:
        return "bad_label", None, None

    if not isinstance(confidence, str) or confidence.lower() not in STRICT_CONFIDENCE_VALUES:
        return "bad_confidence", None, None

    return "ok", label.lower(), confidence.lower()


def parse_json_text(text: str) -> tuple[str, str | None, str | None, bool]:
    text = normalize_completion(text)
    if not text:
        return "empty", None, None, False

    if not text.startswith("{") or not text.endswith("}"):
        return "not_json_object", None, None, False

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return "invalid_json", None, None, False

    status, label, confidence = validate_payload(payload)
    return status, label, confidence, True


def analyze_raw_completion(completion: str) -> dict[str, Any]:
    raw_text = normalize_completion(completion)
    lowered = raw_text.lower()
    raw_has_think = "<think>" in lowered or "</think>" in lowered
    raw_has_fence = "```" in raw_text
    raw_exact_json_only = (
        raw_text.startswith("{")
        and raw_text.endswith("}")
        and not raw_has_think
        and not raw_has_fence
    )

    analysis: dict[str, Any] = {
        "text": raw_text,
        "status": "not_json_object",
        "label": None,
        "confidence": None,
        "raw_has_think": raw_has_think,
        "raw_has_fence": raw_has_fence,
        "raw_exact_json_only": raw_exact_json_only,
        "raw_json_loads": False,
        "payload_valid": False,
    }

    if not raw_exact_json_only:
        if raw_has_think:
            analysis["status"] = "raw_has_think"
        elif raw_has_fence:
            analysis["status"] = "raw_has_fence"
        return analysis

    status, label, confidence, raw_json_loads = parse_json_text(raw_text)
    analysis["status"] = status
    analysis["label"] = label
    analysis["confidence"] = confidence
    analysis["raw_json_loads"] = raw_json_loads
    analysis["payload_valid"] = status == "ok"
    return analysis


def analyze_clean_completion(completion: str) -> dict[str, Any]:
    clean_text = clean_completion(completion)
    status, label, confidence, json_loads = parse_json_text(clean_text)
    return {
        "text": clean_text,
        "status": status,
        "label": label,
        "confidence": confidence,
        "json_loads": json_loads,
        "payload_valid": status == "ok",
    }


def compute_macro_f1(predictions: list[dict[str, Any]]) -> float:
    f1_scores: list[float] = []
    for label in LABELS:
        tp = 0
        fp = 0
        fn = 0
        for row in predictions:
            pred_label = row["pred_label"] if row["pred_label"] in LABELS else None
            gold_label = row["gold_label"]
            if pred_label == label and gold_label == label:
                tp += 1
            elif pred_label == label and gold_label != label:
                fp += 1
            elif pred_label != label and gold_label == label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_scores) / len(f1_scores))


def compute_calibration(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in predictions
        if row["pred_confidence"] in CONFIDENCE_TO_SCORE
    ]
    if not valid_rows:
        return {
            "n_scored_predictions": 0,
            "ece": None,
            "mean_predicted_confidence": None,
            "mean_accuracy": None,
            "buckets": {name: None for name in STRICT_CONFIDENCE_VALUES},
        }

    buckets: dict[str, dict[str, Any] | None] = {}
    ece = 0.0
    confidence_sum = 0.0
    accuracy_sum = 0.0

    for confidence_name in STRICT_CONFIDENCE_VALUES:
        bucket_rows = [row for row in valid_rows if row["pred_confidence"] == confidence_name]
        if not bucket_rows:
            buckets[confidence_name] = None
            continue

        confidence_value = CONFIDENCE_TO_SCORE[confidence_name]
        accuracy = sum(row["label_match"] for row in bucket_rows) / len(bucket_rows)
        confidence_sum += confidence_value * len(bucket_rows)
        accuracy_sum += accuracy * len(bucket_rows)
        ece += abs(accuracy - confidence_value) * len(bucket_rows) / len(valid_rows)
        buckets[confidence_name] = {
            "count": len(bucket_rows),
            "predicted_confidence": confidence_value,
            "accuracy": accuracy,
        }

    return {
        "n_scored_predictions": len(valid_rows),
        "ece": ece,
        "mean_predicted_confidence": confidence_sum / len(valid_rows),
        "mean_accuracy": accuracy_sum / len(valid_rows),
        "buckets": buckets,
    }


def compute_eval_metrics(predictions: list[dict[str, Any]], split_name: str) -> dict[str, Any]:
    n = len(predictions)
    if n == 0:
        return {
            "split": split_name,
            "n_examples": 0,
            "parse_ok": None,
            "format_strict_ok": None,
            "clean_parse_ok": None,
            "label_match": None,
            "macro_f1": None,
            "per_class_recall": {label: None for label in LABELS},
            "gold_label_counts": {label: 0 for label in LABELS},
            "confusion_matrix": {
                gold_label: {pred_label: 0 for pred_label in (*LABELS, "invalid")}
                for gold_label in LABELS
            },
            "calibration": {
                "n_scored_predictions": 0,
                "ece": None,
                "mean_predicted_confidence": None,
                "mean_accuracy": None,
                "buckets": {name: None for name in STRICT_CONFIDENCE_VALUES},
            },
        }

    parse_ok = sum(row["parse_ok"] for row in predictions) / n
    format_strict_ok = sum(row["format_strict_ok"] for row in predictions) / n
    clean_parse_ok = sum(row["clean_parse_ok"] for row in predictions) / n
    label_match = sum(row["label_match"] for row in predictions) / n

    gold_counts = Counter(row["gold_label"] for row in predictions)
    correct_by_class = Counter(
        row["gold_label"]
        for row in predictions
        if row["label_match"] == 1
    )
    per_class_recall = {
        label: (
            float(correct_by_class.get(label, 0) / gold_counts[label])
            if gold_counts[label]
            else None
        )
        for label in LABELS
    }

    confusion: dict[str, dict[str, int]] = {
        gold_label: {pred_label: 0 for pred_label in (*LABELS, "invalid")}
        for gold_label in LABELS
    }
    for row in predictions:
        pred_label = row["pred_label"] if row["pred_label"] in LABELS else "invalid"
        confusion[row["gold_label"]][pred_label] += 1

    return {
        "split": split_name,
        "n_examples": n,
        "parse_ok": parse_ok,
        "format_strict_ok": format_strict_ok,
        "clean_parse_ok": clean_parse_ok,
        "label_match": label_match,
        "macro_f1": compute_macro_f1(predictions),
        "per_class_recall": per_class_recall,
        "gold_label_counts": {label: int(gold_counts.get(label, 0)) for label in LABELS},
        "confusion_matrix": confusion,
        "calibration": compute_calibration(predictions),
    }


def to_wandb_metrics(split_name: str, metrics: dict[str, Any]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        f"{split_name}/n_examples": int(metrics["n_examples"]),
    }
    for key in ("parse_ok", "format_strict_ok", "clean_parse_ok", "label_match", "macro_f1"):
        value = metrics.get(key)
        if value is not None:
            payload[f"{split_name}/{key}"] = float(value)

    for label, value in metrics.get("per_class_recall", {}).items():
        if value is not None:
            payload[f"{split_name}/recall_{label}"] = float(value)

    calibration = metrics.get("calibration", {})
    for key in ("ece", "mean_predicted_confidence", "mean_accuracy"):
        value = calibration.get(key)
        if value is not None:
            payload[f"{split_name}/{key}"] = float(value)

    return payload


def build_prediction_row(example: dict[str, Any], raw_completion: str) -> dict[str, Any]:
    raw = analyze_raw_completion(raw_completion)
    clean = analyze_clean_completion(raw_completion)
    gold_label = str(example.get("disruption_label", "") or "").strip().lower()
    teacher_confidence = example.get("teacher_confidence")
    if teacher_confidence is not None:
        teacher_confidence = str(teacher_confidence).strip().lower() or None

    label_match = int(
        bool(raw["payload_valid"] and raw["label"] == gold_label)
    )
    return {
        "openalex_id": str(example.get("openalex_id", "") or "").strip(),
        "gold_label": gold_label,
        "teacher_confidence": teacher_confidence,
        "completion_raw": raw_completion,
        "pred_label": raw["label"],
        "pred_confidence": raw["confidence"],
        "raw_status": raw["status"],
        "clean_status": clean["status"],
        "parse_ok": int(raw["raw_json_loads"]),
        "format_strict_ok": int(raw["status"] == "ok"),
        "clean_parse_ok": int(clean["json_loads"]),
        "clean_payload_valid": int(clean["payload_valid"]),
        "raw_payload_valid": int(raw["payload_valid"]),
        "label_match": label_match,
        "raw_has_fence": int(raw["raw_has_fence"]),
        "raw_has_think": int(raw["raw_has_think"]),
        "raw_exact_json_only": int(raw["raw_exact_json_only"]),
    }


def evaluate_completion_rows(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    output_dir: str | Path,
    global_step: int | None = None,
    file_suffix: str = "",
    logger=None,
) -> dict[str, Any]:
    output_root = Path(output_dir) / "eval"
    output_root.mkdir(parents=True, exist_ok=True)

    predictions = [
        build_prediction_row(example=row, raw_completion=str(row.get("completion_raw", "") or ""))
        for row in rows
    ]
    metrics = compute_eval_metrics(predictions, split_name)
    if global_step is not None:
        metrics["global_step"] = int(global_step)

    write_json(output_root / f"metrics_{split_name}{file_suffix}.json", metrics)
    write_jsonl(output_root / f"predictions_{split_name}{file_suffix}.jsonl", predictions)

    wandb_metrics = to_wandb_metrics(split_name, metrics)
    if wandb.run is not None and wandb_metrics:
        wandb.log(wandb_metrics, step=global_step)

    if logger is not None:
        logger.info(
            "%s generation eval: label_match=%.4f macro_f1=%.4f parse_ok=%.4f clean_parse_ok=%.4f",
            split_name,
            metrics["label_match"] or 0.0,
            metrics["macro_f1"] or 0.0,
            metrics["parse_ok"] or 0.0,
            metrics["clean_parse_ok"] or 0.0,
        )

    return metrics


def evaluate_split(
    model,
    tokenizer,
    dataset,
    split_name: str,
    output_dir: str | Path,
    max_length: int,
    max_new_tokens: int,
    per_device_eval_batch_size: int,
    global_step: int | None = None,
    file_suffix: str = "",
    logger=None,
) -> dict[str, Any]:
    output_root = Path(output_dir) / "eval"
    output_root.mkdir(parents=True, exist_ok=True)

    if len(dataset) == 0:
        metrics = compute_eval_metrics([], split_name)
        write_json(output_root / f"metrics_{split_name}{file_suffix}.json", metrics)
        write_jsonl(output_root / f"predictions_{split_name}{file_suffix}.jsonl", [])
        return metrics

    tokenization_kwargs = {
        "add_special_tokens": not bool(getattr(tokenizer, "chat_template", None))
    }
    prompt_max_length = max(128, max_length - max_new_tokens)
    previous_padding_side = tokenizer.padding_side
    previous_use_cache = getattr(model.config, "use_cache", None)
    was_training = model.training
    tokenizer.padding_side = "left"
    if previous_use_cache is not None:
        model.config.use_cache = True
    model.eval()

    model_device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []

    try:
        for start in range(0, len(dataset), per_device_eval_batch_size):
            stop = min(start + per_device_eval_batch_size, len(dataset))
            batch = dataset[start:stop]
            encoded = tokenizer(
                batch["prompt_text"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
                **tokenization_kwargs,
            )
            encoded = {name: tensor.to(model_device) for name, tensor in encoded.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for row_index, raw_completion in enumerate(decoded):
                predictions.append(
                    build_prediction_row(
                        {
                            "openalex_id": batch["openalex_id"][row_index],
                            "disruption_label": batch["disruption_label"][row_index],
                            "teacher_confidence": batch["teacher_confidence"][row_index],
                        },
                        raw_completion,
                    )
                )
    finally:
        tokenizer.padding_side = previous_padding_side
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
        if was_training:
            model.train()

    metrics = compute_eval_metrics(predictions, split_name)
    if global_step is not None:
        metrics["global_step"] = int(global_step)

    write_json(output_root / f"metrics_{split_name}{file_suffix}.json", metrics)
    write_jsonl(output_root / f"predictions_{split_name}{file_suffix}.jsonl", predictions)

    wandb_metrics = to_wandb_metrics(split_name, metrics)
    if wandb.run is not None and wandb_metrics:
        wandb.log(wandb_metrics, step=global_step)

    if logger is not None:
        logger.info(
            "%s generation eval: label_match=%.4f macro_f1=%.4f parse_ok=%.4f clean_parse_ok=%.4f",
            split_name,
            metrics["label_match"] or 0.0,
            metrics["macro_f1"] or 0.0,
            metrics["parse_ok"] or 0.0,
            metrics["clean_parse_ok"] or 0.0,
        )

    return metrics


class HeldOutEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        tokenizer,
        eval_dataset,
        output_dir: str | Path,
        max_length: int,
        max_new_tokens: int,
        per_device_eval_batch_size: int,
        split_name: str = "val",
        logger=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.split_name = split_name
        self.logger = logger

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control
        if model is None:
            return control
        if self.logger is not None:
            self.logger.info(
                "Running %s generation eval at epoch %.2f step %s on %s examples",
                self.split_name,
                float(state.epoch or 0.0),
                state.global_step,
                len(self.eval_dataset),
            )
        evaluate_split(
            model=model,
            tokenizer=self.tokenizer,
            dataset=self.eval_dataset,
            split_name=self.split_name,
            output_dir=self.output_dir,
            max_length=self.max_length,
            max_new_tokens=self.max_new_tokens,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            global_step=state.global_step,
            file_suffix=f"_step{state.global_step}",
            logger=self.logger,
        )
        return control
