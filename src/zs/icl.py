# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import json
import logging
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import polars as pl
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

THIS_DIR = pathlib.Path(__file__).resolve().parent
SFT_DIR = THIS_DIR.parent / "sft"
if str(SFT_DIR) not in sys.path:
    sys.path.insert(0, str(SFT_DIR))

from eval import evaluate_completion_rows, write_json

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

LABELS = ("disruptive", "consolidating", "neutral")
STRICT_CONFIDENCE_VALUES = ("low", "medium", "high")

# setup wandb
wandb_project = "tinker_rl_disruption_novelty"
experiment_name = "zs_icl"
num_shots = 0
prompt_mode = f"{num_shots}shot" if num_shots else "zeroshot"
wandb_run = f"qwen3_sm_{prompt_mode}_EXP1"
if int(os.environ.get("LOCAL_RANK", "0")) != 0:
    os.environ["WANDB_MODE"] = "disabled"
    logging.getLogger().setLevel(logging.WARNING)
else:
    wandb.init(project=wandb_project, name=wandb_run)
    logger.info("WandDB setup init to %s", wandb_project)

# set polars threads and eager engine
os.environ["POLARS_MAX_THREADS"] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger.info("Polars concurrency set to %s threads.", pl.thread_pool_size())

# simple experiment config
sglang_base_url = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000").rstrip("/")
sglang_model_name = os.environ.get("SGLANG_MODEL_NAME", "").strip()
sglang_api_key = os.environ.get("SGLANG_API_KEY", "").strip()
tokenizer_name_or_path = (
    os.environ.get("SGLANG_TOKENIZER_PATH", "").strip()
    or os.environ.get("SGLANG_MODEL_PATH", "").strip()
    or "/root/models/Qwen3-8B"
)
temperature = 0.0
top_p = 1.0
max_completion_tokens = 64
request_timeout_seconds = 300
request_parallelism = max(1, min(os.cpu_count() or 1, 32))
retry_count = 3
retry_backoff_seconds = 2.0
dataset_num_proc = max(1, min(os.cpu_count() or 1, 24))
val_eval_size = 2000
test_eval_size = 2000
shot_seed = 2026
enable_thinking = False

# set data directory
repo_root = THIS_DIR.parents[1]
primary_data_root = repo_root / "data"
fallback_data_root = repo_root.parent / "tinker_smolrl_data"
dataset_name_candidates = [
    "sci_balanced_from2m_no_ovr.rl_balanced.with_teacher_confidence.jsonl",
    "sci_balanced_from2m_no_ovr.rl_balanced.jsonl",
]
splits_path = primary_data_root / "sci_balanced_from2m_no_ovr.splits.json"
output_dir = repo_root / "agent_runs" / experiment_name / "runs" / wandb_run
output_dir.mkdir(parents=True, exist_ok=True)
tokenizer = None

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

DISRUPTION_JSON_SCHEMA = {
    "name": "disruption_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "disruption_label": {
                "type": "string",
                "enum": list(LABELS),
            },
            "confidence": {
                "type": "string",
                "enum": list(STRICT_CONFIDENCE_VALUES),
            },
        },
        "required": ["disruption_label", "confidence"],
        "additionalProperties": False,
    },
}


def resolve_dataset_path() -> pathlib.Path:
    for root in (primary_data_root, fallback_data_root):
        for dataset_name in dataset_name_candidates:
            candidate = root / dataset_name
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(
        "Could not find an RL-balanced JSONL under ./data or ../tinker_smolrl_data."
    )


def infer_teacher_confidence(example: dict[str, Any]) -> str:
    existing = str(example.get("teacher_confidence", "") or "").strip().lower()
    if existing in STRICT_CONFIDENCE_VALUES:
        return existing

    label = str(example.get("disruption_label", "") or "").strip().lower()
    cd_index = example.get("cd_index")
    try:
        cd_value = float(cd_index) if cd_index is not None else None
    except (TypeError, ValueError):
        cd_value = None

    if cd_value is None:
        return "medium"

    if label == "disruptive":
        if cd_value >= 0.05:
            return "high"
        if cd_value >= 0.01:
            return "medium"
        return "low"

    if label == "consolidating":
        if cd_value <= -0.05:
            return "high"
        if cd_value <= -0.01:
            return "medium"
        return "low"

    if abs(cd_value) <= 0.00025:
        return "high"
    if abs(cd_value) <= 0.00075:
        return "medium"
    return "low"


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "openalex_id": str(example.get("openalex_id", "") or "").strip(),
        "title": str(example.get("title", "") or "").strip(),
        "abstract": str(example.get("abstract", "") or "").strip(),
        "primary_field": str(example.get("primary_field", "") or "").strip() or None,
        "publication_year": example.get("publication_year"),
        "cited_by_count": example.get("cited_by_count"),
        "cd_index": example.get("cd_index"),
        "disruption_label": str(example.get("disruption_label", "") or "").strip().lower(),
        "teacher_confidence": infer_teacher_confidence(example),
    }


def load_tinker_splits() -> DatasetDict:
    data_path = resolve_dataset_path()
    logger.info("Loading %s dataset...", data_path)
    raw_dataset = load_dataset("json", data_files=str(data_path), split="train")
    raw_dataset = raw_dataset.filter(
        lambda example: (
            str(example.get("openalex_id", "") or "").strip()
            and str(example.get("title", "") or "").strip()
            and str(example.get("abstract", "") or "").strip()
            and str(example.get("disruption_label", "") or "").strip().lower() in LABELS
        ),
        desc="Filter usable Tinker ICL rows",
        num_proc=dataset_num_proc,
    )
    raw_dataset = raw_dataset.map(
        normalize_example,
        desc="Normalize Tinker ICL rows",
        num_proc=dataset_num_proc,
    )

    split_payload = json.loads(splits_path.read_text())
    manifest_ids = split_payload.get("ids", {})
    available_ids = set(raw_dataset["openalex_id"])
    split_coverage = {}
    use_manifest_splits = True

    for split_name in ("train", "val", "test"):
        ids = manifest_ids.get(split_name)
        if not isinstance(ids, list) or not ids:
            use_manifest_splits = False
            split_coverage[split_name] = 0.0
            continue
        matched_ids = sum(1 for paper_id in ids if paper_id in available_ids)
        split_coverage[split_name] = matched_ids / len(ids)
        if split_coverage[split_name] < 0.95:
            use_manifest_splits = False

    if use_manifest_splits:
        id_to_index = {paper_id: idx for idx, paper_id in enumerate(raw_dataset["openalex_id"])}
        dataset = DatasetDict(
            {
                split_name: raw_dataset.select(
                    [id_to_index[paper_id] for paper_id in manifest_ids[split_name] if paper_id in id_to_index]
                )
                for split_name in ("train", "val", "test")
            }
        )
        logger.info("Using manifest splits from %s", splits_path)
        return dataset

    split_ratios = split_payload.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed = int(split_payload.get("split_seed", 2026))
    shuffled_dataset = raw_dataset.shuffle(seed=split_seed)
    total_rows = len(shuffled_dataset)
    train_end = int(total_rows * float(split_ratios.get("train", 0.8)))
    val_end = train_end + int(total_rows * float(split_ratios.get("val", 0.1)))
    logger.warning(
        "Manifest split coverage is too low for the RL-balanced JSONL: %s. Falling back to deterministic seeded JSONL splits.",
        {split: round(coverage, 4) for split, coverage in split_coverage.items()},
    )
    return DatasetDict(
        {
            "train": shuffled_dataset.select(range(0, train_end)),
            "val": shuffled_dataset.select(range(train_end, min(val_end, total_rows))),
            "test": shuffled_dataset.select(range(min(val_end, total_rows), total_rows)),
        }
    )


def select_shot_examples(train_dataset: Dataset) -> list[dict[str, Any]]:
    if num_shots <= 0:
        return []
    selected = train_dataset.shuffle(seed=shot_seed).select(range(min(num_shots, len(train_dataset))))
    examples = [selected[index] for index in range(len(selected))]
    logger.info(
        "Using %s in-context examples with label counts=%s",
        len(examples),
        {
            label: sum(1 for row in examples if row["disruption_label"] == label)
            for label in LABELS
        },
    )
    return examples


def build_system_prompt() -> str:
    return "\n".join(
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


def build_user_text(example: dict[str, Any]) -> str:
    user_lines = [
        "Paper record:",
        f"Title: {example['title']}",
        f"Abstract: {example['abstract']}",
    ]
    for key, value in (
        ("Year", example.get("publication_year")),
        ("Citations", example.get("cited_by_count")),
        ("Field", example.get("primary_field")),
    ):
        if value is not None:
            user_lines.append(f"{key}: {value}")
    user_lines.append("")
    user_lines.append("Return JSON only.")
    return "\n".join(user_lines)


def build_assistant_json(example: dict[str, Any]) -> str:
    return json.dumps(
        {
            "disruption_label": example["disruption_label"],
            "confidence": example["teacher_confidence"],
        },
        separators=(",", ":"),
    )


def build_messages(example: dict[str, Any], shots: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": build_system_prompt()}]
    for shot in shots:
        messages.append({"role": "user", "content": build_user_text(shot)})
        messages.append({"role": "assistant", "content": build_assistant_json(shot)})
    messages.append({"role": "user", "content": build_user_text(example)})
    return messages


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    return tokenizer


def render_messages(messages: list[dict[str, str]]) -> str:
    local_tokenizer = get_tokenizer()
    chat_template = getattr(local_tokenizer, "chat_template", None)
    if chat_template:
        try:
            return local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return local_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content'].strip()}")
    rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def request_json(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if sglang_api_key:
        headers["Authorization"] = f"Bearer {sglang_api_key}"
    request_bytes = None
    if payload is not None:
        request_bytes = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=request_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=request_timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def detect_sglang_model_name() -> str:
    if sglang_model_name:
        return sglang_model_name
    payload = request_json(method="GET", url=f"{sglang_base_url}/v1/models")
    model_rows = payload.get("data", [])
    if not model_rows:
        raise RuntimeError(
            "Could not detect a served model from /v1/models. Set SGLANG_MODEL_NAME explicitly."
        )
    return str(model_rows[0].get("id", "") or "").strip()


def request_completion(messages: list[dict[str, str]], model_name: str) -> tuple[str, dict[str, Any]]:
    del model_name
    prompt_text = render_messages(messages)
    payload = {
        "text": prompt_text,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_completion_tokens,
            "json_schema": json.dumps(DISRUPTION_JSON_SCHEMA["schema"]),
        },
    }
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            response = request_json(
                method="POST",
                url=f"{sglang_base_url}/generate",
                payload=payload,
            )
            completion = str(response.get("text", "") or "")
            return completion, response
        except Exception as exc:
            last_error = exc
            if attempt == retry_count:
                break
            time.sleep(retry_backoff_seconds * attempt)
    raise RuntimeError(f"SGLang request failed after {retry_count} attempts: {last_error}") from last_error


def run_split(
    *,
    dataset: Dataset,
    split_name: str,
    shots: list[dict[str, Any]],
    model_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(dataset) == 0:
        metrics = evaluate_completion_rows(
            [],
            split_name=split_name,
            output_dir=output_dir,
            file_suffix="_final",
            logger=logger,
        )
        return metrics, {"n_request_errors": 0, "examples_per_second": None, "mean_latency_seconds": None}

    started_at = time.time()
    failures = 0
    latencies: list[float] = []
    rows: list[dict[str, Any] | None] = [None] * len(dataset)

    def invoke(index: int) -> tuple[int, dict[str, Any], float]:
        example = dataset[index]
        begin = time.time()
        completion_raw, response = request_completion(build_messages(example, shots), model_name)
        latency = time.time() - begin
        meta_info = response.get("meta_info", {})
        return (
            index,
            {
                "openalex_id": example["openalex_id"],
                "disruption_label": example["disruption_label"],
                "teacher_confidence": example.get("teacher_confidence"),
                "completion_raw": completion_raw,
                "usage_prompt_tokens": meta_info.get("prompt_tokens"),
                "usage_completion_tokens": meta_info.get("completion_tokens"),
            },
            latency,
        )

    logger.info(
        "Running %s SGLang %s eval on %s examples with request_parallelism=%s",
        prompt_mode,
        split_name,
        len(dataset),
        request_parallelism,
    )

    with ThreadPoolExecutor(max_workers=request_parallelism) as executor:
        futures = {executor.submit(invoke, index): index for index in range(len(dataset))}
        for completed_index, future in enumerate(as_completed(futures), start=1):
            dataset_index = futures[future]
            try:
                row_index, row, latency = future.result()
                rows[row_index] = row
                latencies.append(latency)
            except Exception as exc:
                failures += 1
                logger.warning(
                    "%s request failed for example %s (%s): %s",
                    split_name,
                    dataset_index,
                    dataset[dataset_index]["openalex_id"],
                    exc,
                )
                example = dataset[dataset_index]
                rows[dataset_index] = {
                    "openalex_id": example["openalex_id"],
                    "disruption_label": example["disruption_label"],
                    "teacher_confidence": example.get("teacher_confidence"),
                    "completion_raw": "",
                }
            if completed_index % 50 == 0 or completed_index == len(dataset):
                logger.info(
                    "%s progress: %s/%s examples completed",
                    split_name,
                    completed_index,
                    len(dataset),
                )

    completion_rows = [row for row in rows if row is not None]
    metrics = evaluate_completion_rows(
        completion_rows,
        split_name=split_name,
        output_dir=output_dir,
        file_suffix="_final",
        logger=logger,
    )
    elapsed = max(time.time() - started_at, 1e-6)
    summary = {
        "split": split_name,
        "model_name": model_name,
        "num_shots": num_shots,
        "n_examples": len(completion_rows),
        "n_request_errors": failures,
        "examples_per_second": len(completion_rows) / elapsed,
        "mean_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "total_wall_time_seconds": elapsed,
    }
    write_json(output_dir / "eval" / f"request_summary_{split_name}_final.json", summary)

    if wandb.run is not None:
        wandb_payload = {f"{split_name}/request_errors": failures}
        if summary["examples_per_second"] is not None:
            wandb_payload[f"{split_name}/examples_per_second"] = summary["examples_per_second"]
        if summary["mean_latency_seconds"] is not None:
            wandb_payload[f"{split_name}/mean_latency_seconds"] = summary["mean_latency_seconds"]
        wandb.log(wandb_payload)
    return metrics, summary


def main() -> None:
    model_name = detect_sglang_model_name()
    dataset = load_tinker_splits()
    train_dataset = dataset["train"]
    val_dataset = dataset["val"].select(range(min(val_eval_size, len(dataset["val"]))))
    test_dataset = dataset["test"].select(range(min(test_eval_size, len(dataset["test"]))))
    shot_examples = select_shot_examples(train_dataset)

    config_payload = {
        "wandb_project": wandb_project,
        "wandb_run": wandb_run,
        "experiment_name": experiment_name,
        "prompt_mode": prompt_mode,
        "num_shots": num_shots,
        "sglang_base_url": sglang_base_url,
        "sglang_model_name": model_name,
        "tokenizer_name_or_path": tokenizer_name_or_path,
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "enable_thinking": enable_thinking,
        "request_parallelism": request_parallelism,
        "val_eval_size": len(val_dataset),
        "test_eval_size": len(test_dataset),
        "dataset_num_proc": dataset_num_proc,
    }
    write_json(output_dir / "config.json", config_payload)
    if wandb.run is not None:
        wandb.config.update(config_payload, allow_val_change=True)

    logger.info("Using SGLang model %s via %s", model_name, sglang_base_url)
    logger.info(
        "Loaded Tinker ICL splits with train=%s val=%s test=%s",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    val_metrics, val_summary = run_split(
        dataset=val_dataset,
        split_name="val",
        shots=shot_examples,
        model_name=model_name,
    )
    logger.info("Final validation metrics: %s", val_metrics)
    logger.info("Validation request summary: %s", val_summary)

    test_metrics, test_summary = run_split(
        dataset=test_dataset,
        split_name="test",
        shots=shot_examples,
        model_name=model_name,
    )
    logger.info("Final test metrics: %s", test_metrics)
    logger.info("Test request summary: %s", test_summary)

    if wandb.run is not None:
        wandb.finish()
    logger.info("ICL evaluation completed successfully!")


if __name__ == "__main__":
    main()
