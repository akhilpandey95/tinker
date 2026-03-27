# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""
SFT fordisruption classification.

this script:
1. loads sciscinet dataset.
2. data processing and preparation for SFT.
3. QDoRA(4-bit) fine tune using DoRA adapters.
4. eval and check post-training disruption prediction.

usage:
python3 src/sft/qdora_train.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --train-size 100000 \
  --val-size 2000 \
  --test-size 500 \
  --output-dir agent_runs/part2_qdora_sft_baseline/runs/llama31_8b_qdora_sft
"""

# gen
import os
import json
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Any, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

# constants
LABELS = ("disruptive", "consolidating", "neutral")
STRICT_CONFIDENCE_VALUES = ("low", "medium", "high")
CONFIDENCE_TO_SCORE = {
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
}

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TRAIN_SIZE = 100_000
DEFAULT_VAL_SIZE = 2_000
DEFAULT_TEST_SIZE = 500
DEFAULT_MAX_LENGTH = 1024
DEFAULT_EVAL_MAX_NEW_TOKENS = 64
DEFAULT_SAVE_STRATEGY = "epoch"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_dataset_path() -> Path:
    return repo_root() / "sci_balanced_from2m_no_ovr.rl_balanced.jsonl"


def default_splits_path() -> Path:
    return repo_root() / "sci_balanced_from2m_no_ovr.splits.json"


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return repo_root() / "qdora_sft_runs" / f"qdora_sft_{stamp}"

DISRUPTION_LABEL_GUIDANCE = [
    "Interpret the labels using these operational definitions:",
    "disruptive = introduces a new method, system, concept, or finding likely to open a new line of work or replace an existing workflow.",
    "consolidating = strengthens, validates, extends, benchmarks, reviews, or systematizes an existing line of work.",
    "neutral = descriptive, narrow, or limited-impact work without a clear disruptive or consolidating signal.",
    "Use the contribution described in the title, abstract, and metadata.",
    "Do not rely mainly on journal prestige, citation count, or how unusual the topic sounds.",
    "Rare case reports, preliminary findings, or unusual applications are not automatically disruptive.",
]

DISRUPTION_DECISION_GUIDANCE = [
    "Decision checklist:",
    "If the paper likely changes what later researchers can do or introduces a reusable tool/workflow, prefer disruptive.",
    "If the paper mainly validates, benchmarks, extends, applies, or systematizes an existing line of work, prefer consolidating.",
    "If the paper is narrow, descriptive, or limited to a local finding without broader workflow or field effects, prefer neutral.",
    "Do not use consolidating as a default uncertainty label.",
    "If uncertain between disruptive and consolidating, ask whether the work plausibly changes the workflow for later papers.",
    "If uncertain between consolidating and neutral, ask whether it meaningfully strengthens or organizes an existing line of work.",
]

DISRUPTION_MINI_EXAMPLES = [
    "Mini examples:",
    "A paper introducing a new assay or model class that enables previously infeasible measurements across many later studies is disruptive.",
    "A paper benchmarking or incrementally improving existing models on standard tasks is consolidating.",
    "A paper reporting a narrow association, case report, or local application without broader methodological change is neutral.",
]


@dataclass(frozen=True)
class PaperRecord:
    openalex_id: str
    title: str
    abstract: str
    year: int | None
    citations: int | None
    field: str | None
    gold_label: str
    cd_index: float | None


@dataclass(frozen=True)
class SFTExample:
    openalex_id: str
    gold_label: str
    teacher_confidence: str
    prompt_text: str
    full_text: str


@dataclass(frozen=True)
class TokenizedSFTExample:
    openalex_id: str
    gold_label: str
    teacher_confidence: str
    input_ids: list[int]
    attention_mask: list[int]
    prompt_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-path", type=Path, default=default_dataset_path())
    parser.add_argument("--splits-path", type=Path, default=default_splits_path())
    parser.add_argument(
        "--split-source",
        choices=("seeded_jsonl", "manifest"),
        default="seeded_jsonl",
        help=(
            "How to derive train/val/test splits. "
            "'seeded_jsonl' matches the current RL builder style on the RL-balanced JSONL; "
            "'manifest' uses ids from the tracked split manifest."
        ),
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-max-new-tokens", type=int, default=DEFAULT_EVAL_MAX_NEW_TOKENS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, cap training at this many optimizer steps and ignore num_train_epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="If set, prefer explicit warmup steps over warmup_ratio.",
    )
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--save-strategy",
        choices=("no", "steps", "epoch"),
        default=DEFAULT_SAVE_STRATEGY,
    )
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--group-by-length",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated target modules for DoRA adapters.",
    )
    parser.add_argument(
        "--teacher-confidence-mode",
        choices=("constant_medium", "cd_index_margin"),
        default="cd_index_margin",
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help='Trainer reporting target, for example "none" or "wandb".',
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} already exists and is not empty. "
            "Pass --overwrite-output-dir to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_split_ids(splits_path: Path, split_name: str, limit: int) -> list[str]:
    payload = json.loads(splits_path.read_text())
    ids = payload.get("ids", {}).get(split_name)
    if not isinstance(ids, list):
        raise KeyError(f"Split {split_name!r} not found in {splits_path}")
    if limit <= 0:
        return []
    if len(ids) < limit:
        raise ValueError(
            f"Split {split_name!r} has only {len(ids)} ids, but {limit} were requested"
        )
    return [str(x) for x in ids[:limit]]


def load_selected_records(
    dataset_path: Path,
    split_to_ids: dict[str, list[str]],
) -> dict[str, list[PaperRecord]]:
    selected_ids = {paper_id for ids in split_to_ids.values() for paper_id in ids}
    if not selected_ids:
        return {split: [] for split in split_to_ids}

    found: dict[str, PaperRecord] = {}
    with dataset_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            paper_id = str(row.get("openalex_id", "")).strip()
            if paper_id not in selected_ids or paper_id in found:
                continue

            label = str(row.get("disruption_label", "")).strip().lower()
            if label not in LABELS:
                continue

            found[paper_id] = PaperRecord(
                openalex_id=paper_id,
                title=str(row.get("title", "") or "").strip(),
                abstract=str(row.get("abstract", "") or "").strip(),
                year=_coerce_int(row.get("publication_year")),
                citations=_coerce_int(row.get("cited_by_count")),
                field=_coerce_optional_str(row.get("primary_field")),
                gold_label=label,
                cd_index=_coerce_float(row.get("cd_index")),
            )
            if len(found) == len(selected_ids):
                break

    split_to_records: dict[str, list[PaperRecord]] = {}
    for split_name, ids in split_to_ids.items():
        missing = [paper_id for paper_id in ids if paper_id not in found]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Missing {len(missing)} records for split {split_name!r} in {dataset_path}. "
                f"First missing ids: {preview}"
            )
        split_to_records[split_name] = [found[paper_id] for paper_id in ids]
    return split_to_records


def load_seeded_record_splits(
    dataset_path: Path,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[PaperRecord]]:
    total_needed = train_size + val_size + test_size
    if total_needed <= 0:
        return {"train": [], "val": [], "test": []}

    records: list[PaperRecord] = []
    with dataset_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            label = str(row.get("disruption_label", "")).strip().lower()
            if label not in LABELS:
                continue

            records.append(
                PaperRecord(
                    openalex_id=str(row.get("openalex_id", "")).strip(),
                    title=str(row.get("title", "") or "").strip(),
                    abstract=str(row.get("abstract", "") or "").strip(),
                    year=_coerce_int(row.get("publication_year")),
                    citations=_coerce_int(row.get("cited_by_count")),
                    field=_coerce_optional_str(row.get("primary_field")),
                    gold_label=label,
                    cd_index=_coerce_float(row.get("cd_index")),
                )
            )
            if len(records) == total_needed:
                break

    if len(records) < total_needed:
        raise ValueError(
            f"Need at least {total_needed} labeled records in {dataset_path}, found {len(records)}"
        )

    random.Random(seed).shuffle(records)
    train_stop = train_size
    # Keep test immediately after train so the held-out slice stays closest to
    # the current RL builder convention.
    test_stop = train_stop + test_size
    val_stop = test_stop + val_size

    return {
        "train": records[:train_stop],
        "test": records[train_stop:test_stop],
        "val": records[test_stop:val_stop],
    }


def load_experiment_record_splits(args: argparse.Namespace) -> dict[str, list[PaperRecord]]:
    split_source = getattr(args, "split_source", "seeded_jsonl")
    if split_source == "manifest":
        split_to_ids = {
            "train": load_split_ids(args.splits_path, args.train_split, args.train_size),
            "val": load_split_ids(args.splits_path, args.val_split, args.val_size),
            "test": load_split_ids(args.splits_path, args.test_split, args.test_size),
        }
        return load_selected_records(args.dataset_path, split_to_ids)

    if split_source != "seeded_jsonl":
        raise ValueError(f"Unsupported split source: {split_source}")

    return load_seeded_record_splits(
        args.dataset_path,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_messages(record: PaperRecord) -> list[dict[str, str]]:
    system_text = "\n".join(
        [
            "You are a careful scientific literature analyst.",
            "Classify the paper using only the user-provided record.",
            *DISRUPTION_LABEL_GUIDANCE,
            *DISRUPTION_DECISION_GUIDANCE,
            *DISRUPTION_MINI_EXAMPLES,
            "Return one JSON object with exactly two keys.",
            "Keys: disruption_label, confidence.",
            "Allowed disruption_label values: disruptive, consolidating, neutral.",
            "confidence must be one of: low, medium, high.",
            "Use low confidence when the label boundary is genuinely uncertain.",
            "Do not use markdown fences.",
            "Do not include <think> tags.",
        ]
    )

    user_lines = [
        "Paper record:",
        f"Title: {record.title}",
        f"Abstract: {record.abstract}",
    ]

    for key, value in (
        ("Year", record.year),
        ("Citations", record.citations),
        ("Field", record.field),
    ):
        if value is not None:
            user_lines.append(f"{key}: {value}")

    user_lines.extend(["", "Return JSON only.", "{"])

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def derive_teacher_confidence(record: PaperRecord, mode: str) -> str:
    if mode == "constant_medium":
        return "medium"

    if mode != "cd_index_margin":
        raise ValueError(f"Unsupported teacher confidence mode: {mode}")

    cd_index = record.cd_index
    if cd_index is None:
        return "medium"

    if record.gold_label == "disruptive":
        if cd_index >= 0.05:
            return "high"
        if cd_index >= 0.01:
            return "medium"
        return "low"

    if record.gold_label == "consolidating":
        if cd_index <= -0.05:
            return "high"
        if cd_index <= -0.01:
            return "medium"
        return "low"

    distance_to_zero = abs(cd_index)
    if distance_to_zero <= 0.00025:
        return "high"
    if distance_to_zero <= 0.00075:
        return "medium"
    return "low"


def build_target_json(record: PaperRecord, teacher_confidence_mode: str) -> str:
    payload = {
        "disruption_label": record.gold_label,
        "confidence": derive_teacher_confidence(record, teacher_confidence_mode),
    }
    return json.dumps(payload, separators=(",", ":"))


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered: list[str] = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content'].strip()}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def build_sft_examples(
    records: Iterable[PaperRecord],
    tokenizer: Any,
    teacher_confidence_mode: str,
) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for record in records:
        messages = build_messages(record)
        target_json = build_target_json(record, teacher_confidence_mode)
        prompt_text = render_messages(
            tokenizer,
            messages,
            add_generation_prompt=True,
        )
        full_text = render_messages(
            tokenizer,
            messages + [{"role": "assistant", "content": target_json}],
            add_generation_prompt=False,
        )
        examples.append(
            SFTExample(
                openalex_id=record.openalex_id,
                gold_label=record.gold_label,
                teacher_confidence=json.loads(target_json)["confidence"],
                prompt_text=prompt_text,
                full_text=full_text,
            )
        )
    return examples


def filter_examples_by_length(
    tokenizer: Any,
    examples: list[SFTExample],
    max_length: int,
    batch_size: int = 256,
) -> tuple[list[SFTExample], dict[str, int]]:
    tokenization_kwargs = {"add_special_tokens": not bool(getattr(tokenizer, "chat_template", None))}
    kept: list[SFTExample] = []
    dropped_prompt_too_long = 0
    dropped_full_too_long = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompt_batch = tokenizer(
            [example.prompt_text for example in batch],
            truncation=False,
            **tokenization_kwargs,
        )["input_ids"]
        full_batch = tokenizer(
            [example.full_text for example in batch],
            truncation=False,
            **tokenization_kwargs,
        )["input_ids"]

        for example, prompt_ids, full_ids in zip(batch, prompt_batch, full_batch, strict=True):
            if len(prompt_ids) >= max_length:
                dropped_prompt_too_long += 1
                continue
            if len(full_ids) > max_length:
                dropped_full_too_long += 1
                continue
            kept.append(example)

    stats = {
        "requested_examples": len(examples),
        "kept_examples": len(kept),
        "dropped_prompt_too_long": dropped_prompt_too_long,
        "dropped_full_too_long": dropped_full_too_long,
    }
    return kept, stats


def tokenize_examples_for_training(
    tokenizer: Any,
    examples: list[SFTExample],
    max_length: int,
    batch_size: int = 256,
) -> list[TokenizedSFTExample]:
    tokenization_kwargs = {
        "add_special_tokens": not bool(getattr(tokenizer, "chat_template", None))
    }
    tokenized_examples: list[TokenizedSFTExample] = []

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompt_batch = tokenizer(
            [example.prompt_text for example in batch],
            truncation=True,
            max_length=max_length,
            **tokenization_kwargs,
        )["input_ids"]
        full_batch = tokenizer(
            [example.full_text for example in batch],
            truncation=True,
            max_length=max_length,
            **tokenization_kwargs,
        )
        full_input_ids = full_batch["input_ids"]
        full_attention_masks = full_batch["attention_mask"]

        for example, prompt_ids, full_ids, full_mask in zip(
            batch,
            prompt_batch,
            full_input_ids,
            full_attention_masks,
            strict=True,
        ):
            prompt_length = min(len(prompt_ids), len(full_ids))
            tokenized_examples.append(
                TokenizedSFTExample(
                    openalex_id=example.openalex_id,
                    gold_label=example.gold_label,
                    teacher_confidence=example.teacher_confidence,
                    input_ids=list(full_ids),
                    attention_mask=list(full_mask),
                    prompt_length=prompt_length,
                )
            )

    return tokenized_examples


class PromptCompletionDataset:
    def __init__(self, examples: list[TokenizedSFTExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        return {
            "openalex_id": example.openalex_id,
            "gold_label": example.gold_label,
            "teacher_confidence": example.teacher_confidence,
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "prompt_length": example.prompt_length,
            "length": len(example.input_ids),
        }


class CompletionOnlyCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.tokenizer.pad(
            [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ],
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for row_index, feature in enumerate(features):
            prompt_len = min(int(feature["prompt_length"]), labels.shape[1])
            labels[row_index, :prompt_len] = -100

        batch["labels"] = labels
        return batch


class DisruptionJSONContract:
    @staticmethod
    def normalize_completion(completion: str) -> str:
        text = completion.strip()
        for suffix in ("<|endoftext|>", "<|eot_id|>"):
            text = text.replace(suffix, "").strip()
        return text

    @staticmethod
    def clean_completion(completion: str) -> str:
        import re

        text = DisruptionJSONContract.normalize_completion(completion)
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

    @staticmethod
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

    @staticmethod
    def parse_json_text(text: str) -> tuple[str, str | None, str | None, bool]:
        text = DisruptionJSONContract.normalize_completion(text)
        if not text:
            return "empty", None, None, False

        if not text.startswith("{") or not text.endswith("}"):
            return "not_json_object", None, None, False

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json", None, None, False

        status, label, confidence = DisruptionJSONContract.validate_payload(payload)
        return status, label, confidence, True

    @staticmethod
    def analyze_raw_completion(completion: str) -> dict[str, Any]:
        raw_text = DisruptionJSONContract.normalize_completion(completion)
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

        status, label, confidence, raw_json_loads = DisruptionJSONContract.parse_json_text(raw_text)
        analysis["status"] = status
        analysis["label"] = label
        analysis["confidence"] = confidence
        analysis["raw_json_loads"] = raw_json_loads
        analysis["payload_valid"] = status == "ok"
        return analysis

    @staticmethod
    def analyze_clean_completion(completion: str) -> dict[str, Any]:
        clean_text = DisruptionJSONContract.clean_completion(completion)
        status, label, confidence, json_loads = DisruptionJSONContract.parse_json_text(clean_text)
        return {
            "text": clean_text,
            "status": status,
            "label": label,
            "confidence": confidence,
            "json_loads": json_loads,
            "payload_valid": status == "ok",
        }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True))
            handle.write("\n")


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


def trainer_output_dir(output_dir: Path) -> Path:
    return output_dir / "trainer"


def list_trainer_checkpoints(output_dir: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    root = trainer_output_dir(output_dir)
    if not root.exists():
        return []

    for path in root.iterdir():
        if not path.is_dir():
            continue
        prefix = "checkpoint-"
        if not path.name.startswith(prefix):
            continue
        step_text = path.name[len(prefix) :]
        if not step_text.isdigit():
            continue
        checkpoints.append((int(step_text), path))

    checkpoints.sort(key=lambda item: item[0])
    return [path for _, path in checkpoints]


def find_latest_trainer_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = list_trainer_checkpoints(output_dir)
    if not checkpoints:
        return None
    return checkpoints[-1]


def count_labels(records: Iterable[PaperRecord]) -> dict[str, int]:
    counts = Counter(record.gold_label for record in records)
    return {label: int(counts.get(label, 0)) for label in LABELS}


def count_example_confidences(examples: Iterable[SFTExample]) -> dict[str, int]:
    counts = Counter(example.teacher_confidence for example in examples)
    return {name: int(counts.get(name, 0)) for name in STRICT_CONFIDENCE_VALUES}


def load_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to run qdora_train.py"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    return tokenizer


def load_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(
            "torch, transformers, peft, and bitsandbytes are required to run qdora_train.py"
        ) from exc

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required for 4-bit QDoRA training.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model_kwargs: dict[str, Any] = {
        "quantization_config": quant_config,
        "device_map": {"": local_rank},
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    except Exception as exc:
        if args.attn_implementation != "flash_attention_2":
            raise
        print(
            "flash_attention_2 unavailable during model load; retrying with sdpa.",
            f"Original error: {exc}",
        )
        model_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=args.gradient_checkpointing,
    )

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_target_modules(args.target_modules),
        use_dora=True,
    )
    model = get_peft_model(model, peft_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    summary = {
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "trainable_parameters": int(trainable_params),
        "total_parameters": int(total_params),
        "trainable_fraction": float(trainable_params / total_params) if total_params else 0.0,
    }
    return model, summary


def parse_target_modules(raw: str) -> list[str]:
    modules = [part.strip() for part in raw.split(",") if part.strip()]
    if not modules:
        raise ValueError("At least one target module is required for DoRA.")
    return modules


def build_training_arguments(args: argparse.Namespace, output_dir: Path) -> Any:
    import inspect
    import torch
    from transformers import TrainingArguments

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    use_fp16 = torch.cuda.is_available() and not use_bf16
    max_steps = getattr(args, "max_steps", -1)
    warmup_steps = getattr(args, "warmup_steps", None)
    dataloader_num_workers = getattr(args, "dataloader_num_workers", 0)
    dataloader_persistent_workers = bool(
        getattr(args, "dataloader_persistent_workers", True)
    ) and dataloader_num_workers > 0

    report_to = [] if args.report_to == "none" else [args.report_to]
    raw_kwargs: dict[str, Any] = {
        "output_dir": str(trainer_output_dir(output_dir)),
        "overwrite_output_dir": args.overwrite_output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": max_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": args.gradient_checkpointing,
        "max_grad_norm": args.max_grad_norm,
        "remove_unused_columns": False,
        "report_to": report_to,
        "seed": args.seed,
        "data_seed": args.seed,
        "logging_first_step": True,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": getattr(args, "dataloader_pin_memory", True),
        "dataloader_persistent_workers": dataloader_persistent_workers,
        "group_by_length": getattr(args, "group_by_length", False),
        "ddp_find_unused_parameters": False,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    supported_params = set(signature.parameters)

    if "eval_strategy" in supported_params:
        raw_kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in supported_params:
        raw_kwargs["evaluation_strategy"] = "no"

    if warmup_steps is not None:
        raw_kwargs["warmup_steps"] = warmup_steps
    elif args.warmup_ratio is not None:
        raw_kwargs["warmup_ratio"] = args.warmup_ratio

    if args.save_strategy == "steps":
        raw_kwargs["save_steps"] = args.save_steps

    kwargs = {
        key: value for key, value in raw_kwargs.items() if key in supported_params
    }
    unsupported = sorted(set(raw_kwargs) - set(kwargs))
    if unsupported:
        print(
            "Skipping unsupported TrainingArguments keys for this transformers build:",
            ", ".join(unsupported),
        )

    return TrainingArguments(**kwargs)


def train_model(
    model: Any,
    tokenizer: Any,
    train_examples: list[SFTExample],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    import inspect
    from transformers import Trainer

    tokenized_train_examples = tokenize_examples_for_training(
        tokenizer,
        train_examples,
        args.max_length,
    )
    train_dataset = PromptCompletionDataset(tokenized_train_examples)
    collator = CompletionOnlyCollator(
        tokenizer,
        pad_to_multiple_of=getattr(args, "pad_to_multiple_of", None),
    )
    training_args = build_training_arguments(args, output_dir)

    raw_trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    trainer_supported_params = set(trainer_signature.parameters)
    if "processing_class" in trainer_supported_params:
        raw_trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_supported_params:
        raw_trainer_kwargs["tokenizer"] = tokenizer

    trainer_kwargs = {
        key: value
        for key, value in raw_trainer_kwargs.items()
        if key in trainer_supported_params
    }
    unsupported = sorted(set(raw_trainer_kwargs) - set(trainer_kwargs))
    if unsupported:
        print(
            "Skipping unsupported Trainer keys for this transformers build:",
            ", ".join(unsupported),
        )

    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    metrics = dict(train_result.metrics)
    metrics["train_examples"] = len(train_examples)
    metrics["global_step"] = trainer.state.global_step
    latest_checkpoint = find_latest_trainer_checkpoint(output_dir)
    metrics["latest_trainer_checkpoint"] = latest_checkpoint
    metrics["all_trainer_checkpoints"] = list_trainer_checkpoints(output_dir)
    metrics["final_adapter_dir"] = final_adapter_dir
    metrics["requested_max_steps"] = getattr(args, "max_steps", -1)
    trainer.save_state()
    return trainer, metrics


def prepare_model_for_eval(model: Any) -> None:
    # Generation can break if the PEFT wrapper or wrapped model still has
    # gradient checkpointing enabled from training.
    model.eval()
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "gradient_checkpointing_disable"):
            candidate.gradient_checkpointing_disable()
        if hasattr(candidate, "config"):
            candidate.config.use_cache = True
        generation_config = getattr(candidate, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = True


def evaluate_split(
    model: Any,
    tokenizer: Any,
    examples: list[SFTExample],
    split_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    import torch

    if not examples:
        empty_metrics = {
            "split": split_name,
            "n_examples": 0,
            "parse_ok": None,
            "format_strict_ok": None,
            "label_match": None,
            "macro_f1": None,
            "per_class_recall": {label: None for label in LABELS},
            "calibration": None,
        }
        write_json(output_dir / f"metrics_{split_name}.json", empty_metrics)
        write_jsonl(output_dir / f"predictions_{split_name}.jsonl", [])
        return empty_metrics

    prepare_model_for_eval(model)
    tokenization_kwargs = {
        "add_special_tokens": not bool(getattr(tokenizer, "chat_template", None))
    }
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    prompt_max_length = max(128, args.max_length - args.eval_max_new_tokens)

    predictions: list[dict[str, Any]] = []
    try:
        for start in range(0, len(examples), args.per_device_eval_batch_size):
            batch = examples[start : start + args.per_device_eval_batch_size]
            encoded = tokenizer(
                [example.prompt_text for example in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
                **tokenization_kwargs,
            )
            encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for example, raw_completion in zip(batch, decoded, strict=True):
                raw = DisruptionJSONContract.analyze_raw_completion(raw_completion)
                clean = DisruptionJSONContract.analyze_clean_completion(raw_completion)
                label_match = int(
                    bool(raw["payload_valid"] and raw["label"] == example.gold_label)
                )
                predictions.append(
                    {
                        "openalex_id": example.openalex_id,
                        "gold_label": example.gold_label,
                        "teacher_confidence": example.teacher_confidence,
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
                )
    finally:
        tokenizer.padding_side = previous_padding_side

    metrics = compute_eval_metrics(predictions, split_name)
    write_json(output_dir / f"metrics_{split_name}.json", metrics)
    write_jsonl(output_dir / f"predictions_{split_name}.jsonl", predictions)
    return metrics


def compute_eval_metrics(
    predictions: list[dict[str, Any]],
    split_name: str,
) -> dict[str, Any]:
    n = len(predictions)
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

    macro_f1 = _compute_macro_f1(predictions)
    calibration = _compute_calibration(predictions)

    return {
        "split": split_name,
        "n_examples": n,
        "parse_ok": parse_ok,
        "format_strict_ok": format_strict_ok,
        "clean_parse_ok": clean_parse_ok,
        "label_match": label_match,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "gold_label_counts": {label: int(gold_counts.get(label, 0)) for label in LABELS},
        "confusion_matrix": confusion,
        "calibration": calibration,
    }


def _compute_macro_f1(predictions: list[dict[str, Any]]) -> float:
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


def _compute_calibration(predictions: list[dict[str, Any]]) -> dict[str, Any]:
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

    buckets: dict[str, dict[str, Any]] = {}
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


def build_run_manifest(
    args: argparse.Namespace,
    train_records: list[PaperRecord],
    val_records: list[PaperRecord],
    test_records: list[PaperRecord],
    train_examples: list[SFTExample],
    val_examples: list[SFTExample],
    test_examples: list[SFTExample],
    filter_stats: dict[str, dict[str, int]],
    model_summary: dict[str, Any],
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "dataset": {
            "dataset_path": str(args.dataset_path),
            "splits_path": str(args.splits_path),
            "split_source": getattr(args, "split_source", "seeded_jsonl"),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "requested_sizes": {
                "train": args.train_size,
                "val": args.val_size,
                "test": args.test_size,
            },
            "loaded_records": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
            },
            "kept_examples": {
                "train": len(train_examples),
                "val": len(val_examples),
                "test": len(test_examples),
            },
            "label_counts": {
                "train": count_labels(train_records),
                "val": count_labels(val_records),
                "test": count_labels(test_records),
            },
            "teacher_confidence_counts": {
                "train": count_example_confidences(train_examples),
                "val": count_example_confidences(val_examples),
                "test": count_example_confidences(test_examples),
            },
            "length_filter_stats": filter_stats,
        },
        "model": {
            "model_name": args.model_name,
            "target_modules": parse_target_modules(args.target_modules),
            **model_summary,
        },
        "training": train_metrics,
        "evaluation": eval_metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_output_dir(args.output_dir, overwrite=args.overwrite_output_dir)

    tokenizer = load_tokenizer(args)
    split_to_records = load_experiment_record_splits(args)
    train_records = split_to_records["train"]
    val_records = split_to_records["val"]
    test_records = split_to_records["test"]

    # prep train, val and test examples
    raw_train_examples = build_sft_examples(train_records, tokenizer, args.teacher_confidence_mode,)
    raw_val_examples = build_sft_examples(val_records, tokenizer, args.teacher_confidence_mode)
    raw_test_examples = build_sft_examples(test_records, tokenizer, args.teacher_confidence_mode)

    # data processing train, val and test examples
    train_examples, train_filter_stats = filter_examples_by_length(tokenizer, raw_train_examples, args.max_length)
    val_examples, val_filter_stats = filter_examples_by_length(tokenizer, raw_val_examples, args.max_length)
    test_examples, test_filter_stats = filter_examples_by_length(tokenizer, raw_test_examples, args.max_length)

    model, model_summary = load_model(args)
    trainer, train_metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        train_examples=train_examples,
        args=args,
        output_dir=args.output_dir,
    )

    prepare_model_for_eval(trainer.model)
    val_metrics = evaluate_split(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=val_examples,
        split_name="val",
        args=args,
        output_dir=args.output_dir,
    )
    test_metrics = evaluate_split(
        model=trainer.model,
        tokenizer=tokenizer,
        examples=test_examples,
        split_name="test",
        args=args,
        output_dir=args.output_dir,
    )

    manifest = build_run_manifest(
        args=args,
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
        filter_stats={
            "train": train_filter_stats,
            "val": val_filter_stats,
            "test": test_filter_stats,
        },
        model_summary=model_summary,
        train_metrics=train_metrics,
        eval_metrics={"val": val_metrics, "test": test_metrics},
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    write_json(args.output_dir / "train_metrics.json", train_metrics)


if __name__ == "__main__":
    main()
