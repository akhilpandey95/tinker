# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import os
import gc
import time
import json
import inspect
import torch
import wandb
import pprint
import pathlib
import logging
import datasets
import numpy as np
import polars as pl
from typing import Any, List
from outlines import models, generate
from trl import SFTConfig, SFTTrainer
from bitsandbytes.optim import Adam8bit
from pydantic import BaseModel, Field, constr, conlist
from transformers.trainer_utils import is_main_process
from transformers import BitsAndBytesConfig, set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainerCallback, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Features, Sequence, Value, load_dataset, DatasetDict, concatenate_datasets
from eval import HeldOutEvalCallback, evaluate_split

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# setup wandb
# sm for me is 8b/9b parameters
wandb_project = "tinker_rl_disruption_novelty"
wandb_run = "qwen3_sm_EXP1"
if not is_main_process(int(os.environ.get("LOCAL_RANK", 0))):
    os.environ["WANDB_MODE"] = "disabled"
    logging.getLogger().setLevel(logging.WARNING)
else:
    wandb.init(project=wandb_project, name=wandb_run)
    logger.info(f"WandDB setup init to {wandb_project}")

# set polars threads and new eager engine
os.environ['POLARS_MAX_THREADS'] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "oops"
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# set seed
set_seed(2026)

# set LLM
#model_id = "/root/models/Qwen3.5-9B"
model_id = "/root/models/Qwen3-8B"
experiment_name = "qdora_sft"

# simple experiment config
max_seq_length = 2048
learning_rate = 2e-5
warmup_ratio = 0.03
weight_decay = 0.0
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 10
save_total_limit = 2
dataset_num_proc = max(1, min(os.cpu_count() or 1, 24))
dataset_batch_size = 1000
per_device_eval_batch_size = 8
eval_max_new_tokens = 64
val_eval_size = 2000
test_eval_size = 2000
use_liger_kernel = True
max_train_examples = 50000 
train_subset_seed = 2026
lora_rank = 64
lora_alpha = 128
lora_dropout = 0.05
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# get device status
device = torch.device("cuda:0" if torch.cuda.is_available() else
                      ("mps" if torch.backends.mps.is_available() else "cpu"))
logger.info("----------------------------------")
logger.info(f"Using {device} to run {model_id}")
logger.info("----------------------------------")
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is required for 4-bit QDoRA training.")

# init the tokenizer
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"

# tokenizer quirks
logger.info(f"EOS token-id: {tokenizer.eos_token_id}")
logger.info(f"PAD token-id: {tokenizer.pad_token_id}")

# load the model
attn_implementation = "flash_attention_2"
use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype,
)

# load quantized model for QDoRA
model = AutoModelForCausalLM.from_pretrained(model_id,
                                            dtype=compute_dtype,
                                            quantization_config=quantization_config,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            attn_implementation=attn_implementation,
                                            device_map="auto")
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
model = prepare_model_for_kbit_training(model)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

if "use_dora" not in inspect.signature(LoraConfig.__init__).parameters:
    raise RuntimeError(
        "This PEFT build does not support use_dora. Upgrade peft or disable DoRA."
    )

peft_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
    use_dora=True,
)
model = get_peft_model(model, peft_config)


# tokenizer
logger.info(f"Tokenizer length: {len(tokenizer)}")

trainable_parameters = 0
total_parameters = 0
for parameter in model.parameters():
    total_parameters += parameter.numel()
    if parameter.requires_grad:
        trainable_parameters += parameter.numel()
trainable_fraction = trainable_parameters / total_parameters if total_parameters else 0.0
logger.info(
    "Applied QDoRA config: rank=%s alpha=%s dropout=%.3f target_modules=%s",
    lora_rank,
    lora_alpha,
    lora_dropout,
    ",".join(target_modules),
)
logger.info(
    "Trainable parameters: %s/%s (%.4f%%)",
    trainable_parameters,
    total_parameters,
    100.0 * trainable_fraction,
)
logger.info(
    "Dataset preprocessing will use up to %s processes with batch size %s",
    dataset_num_proc,
    dataset_batch_size,
)
logger.info("Liger kernel enabled: %s", use_liger_kernel)

# calc model-tokenizer time
end = time.time()
logger.info(f"Model-tokenizer load time: {end - start} seconds")
logger.info("----------------------------------")

# set data directory
data_dir = "../data/sci_balanced_from2m_no_ovr.rl_balanced.with_teacher_confidence.jsonl"
splits_dir = "../data/sci_balanced_from2m_no_ovr.splits.json"
logger.info(f"Loading {data_dir} dataset...")

# start load data
valid_labels = {"disruptive", "consolidating", "neutral"}
valid_confidences = {"low", "medium", "high"}

repo_root = pathlib.Path(__file__).resolve().parents[2]
data_path = repo_root / "data" / pathlib.Path(data_dir).name
splits_path = repo_root / "data" / pathlib.Path(splits_dir).name
output_dir = repo_root / "agent_runs" / experiment_name / "runs" / wandb_run
output_dir.mkdir(parents=True, exist_ok=True)


def _build_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    system_text = "\n".join(
        [
            "You are a careful scientific literature analyst.",
            "Classify the paper using only the user-provided record.",
            "Disruptive papers clearly change subsequent work away from prior lines.",
            "Consolidating papers mainly synthesize or reinforce existing directions.",
            "Neutral papers fit between those extremes.",
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
        f"Title: {str(example.get('title', '') or '').strip()}",
        f"Abstract: {str(example.get('abstract', '') or '').strip()}",
    ]

    for key, value in (
        ("Year", example.get("publication_year")),
        ("Citations", example.get("cited_by_count")),
        ("Field", str(example.get("primary_field", "") or "").strip() or None),
    ):
        if value is not None:
            user_lines.append(f"{key}: {value}")

    user_lines.extend(["", "Return JSON only.", "{"])
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def _render_messages(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content'].strip()}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:\n")
    return "\n\n".join(rendered)


def _format_sft_example(example: dict[str, Any]) -> dict[str, Any]:
    label = str(example.get("disruption_label", "") or "").strip().lower()
    confidence = str(example.get("teacher_confidence", "") or "").strip().lower()
    messages = _build_messages(example)
    prompt_text = _render_messages(messages, add_generation_prompt=True)
    target_json = json.dumps(
        {
            "disruption_label": label,
            "confidence": confidence,
        },
        separators=(",", ":"),
    )

    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str) and "<think>" in chat_template:
        eos_token = getattr(tokenizer, "eos_token", None) or ""
        eos_suffix = f"{eos_token}\n" if eos_token else ""
        full_text = f"{prompt_text}{target_json}{eos_suffix}"
    else:
        full_text = _render_messages(
            messages + [{"role": "assistant", "content": target_json}],
            add_generation_prompt=False,
        )

    return {
        "openalex_id": str(example.get("openalex_id", "") or "").strip(),
        "disruption_label": label,
        "teacher_confidence": confidence,
        "prompt_text": prompt_text,
        "target_json": target_json,
        "text": full_text,
    }


def _build_sft_config() -> tuple[SFTConfig, dict[str, Any]]:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    steps_per_epoch = max(
        1,
        len(train_dataset) // max(1, per_device_train_batch_size * gradient_accumulation_steps * world_size),
    )
    warmup_steps = max(1, int(steps_per_epoch * num_train_epochs * warmup_ratio))
    report_targets = [] if os.environ.get("WANDB_MODE") == "disabled" else ["wandb"]
    raw_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": wandb_run,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "num_train_epochs": num_train_epochs,
        "logging_steps": logging_steps,
        "report_to": report_targets,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "bf16": use_bf16,
        "fp16": torch.cuda.is_available() and not use_bf16,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "save_strategy": "epoch",
        "save_total_limit": save_total_limit,
        "save_safetensors": True,
        "remove_unused_columns": True,
        "logging_first_step": True,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 1.0,
    }
    trainer_dataset_kwargs: dict[str, Any] = {}

    supported = set(inspect.signature(SFTConfig.__init__).parameters)
    if "dataset_text_field" in supported:
        raw_kwargs["dataset_text_field"] = "text"
    else:
        trainer_dataset_kwargs["dataset_text_field"] = "text"

    if "max_length" in supported:
        raw_kwargs["max_length"] = max_seq_length
    elif "max_seq_length" in supported:
        raw_kwargs["max_seq_length"] = max_seq_length
    else:
        trainer_dataset_kwargs["max_seq_length"] = max_seq_length

    if "packing" in supported:
        raw_kwargs["packing"] = False
    else:
        trainer_dataset_kwargs["packing"] = False

    if "dataset_num_proc" in supported:
        raw_kwargs["dataset_num_proc"] = dataset_num_proc
    else:
        trainer_dataset_kwargs["dataset_num_proc"] = dataset_num_proc

    if "dataset_batch_size" in supported:
        raw_kwargs["dataset_batch_size"] = dataset_batch_size
    else:
        trainer_dataset_kwargs["dataset_batch_size"] = dataset_batch_size

    if use_liger_kernel:
        if "use_liger_kernel" in supported:
            raw_kwargs["use_liger_kernel"] = True
        else:
            raise RuntimeError(
                "This TRL build does not support use_liger_kernel in SFTConfig. "
                "Upgrade trl or disable use_liger_kernel."
            )

    if "eval_strategy" in supported:
        raw_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in supported:
        raw_kwargs["evaluation_strategy"] = "epoch"

    config_kwargs = {
        key: value for key, value in raw_kwargs.items() if key in supported
    }
    skipped = sorted(set(raw_kwargs) - set(config_kwargs))
    if skipped:
        logger.info("Skipping unsupported SFTConfig keys: %s", ", ".join(skipped))

    return SFTConfig(**config_kwargs), trainer_dataset_kwargs


def _build_sft_trainer(
    train_dataset_for_trainer,
    eval_dataset_for_trainer,
    training_args: SFTConfig,
    trainer_dataset_kwargs: dict[str, Any],
    data_collator: DataCollatorForLanguageModeling,
) -> SFTTrainer:
    raw_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset_for_trainer,
        "eval_dataset": eval_dataset_for_trainer,
        "data_collator": data_collator,
        **trainer_dataset_kwargs,
    }
    supported = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in supported:
        raw_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in supported:
        raw_kwargs["tokenizer"] = tokenizer

    trainer_kwargs = {
        key: value for key, value in raw_kwargs.items() if key in supported
    }
    skipped = sorted(set(raw_kwargs) - set(trainer_kwargs))
    if skipped:
        logger.info("Skipping unsupported SFTTrainer keys: %s", ", ".join(skipped))

    return SFTTrainer(**trainer_kwargs)


raw_dataset = load_dataset("json", data_files=str(data_path), split="train")
raw_dataset = raw_dataset.filter(
    lambda example: (
        str(example.get("openalex_id", "") or "").strip()
        and str(example.get("title", "") or "").strip()
        and str(example.get("abstract", "") or "").strip()
        and str(example.get("disruption_label", "") or "").strip().lower() in valid_labels
        and str(example.get("teacher_confidence", "") or "").strip().lower() in valid_confidences
    ),
    desc="Filter usable Tinker SFT rows",
)
raw_dataset = raw_dataset.map(
    lambda example: {
        "openalex_id": str(example.get("openalex_id", "") or "").strip(),
        "title": str(example.get("title", "") or "").strip(),
        "abstract": str(example.get("abstract", "") or "").strip(),
        "primary_field": str(example.get("primary_field", "") or "").strip() or None,
        "disruption_label": str(example.get("disruption_label", "") or "").strip().lower(),
        "teacher_confidence": str(example.get("teacher_confidence", "") or "").strip().lower(),
    },
    desc="Normalize Tinker SFT rows",
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
    sft_dataset = DatasetDict(
        {
            split_name: raw_dataset.select(
                [id_to_index[paper_id] for paper_id in manifest_ids[split_name] if paper_id in id_to_index]
            )
            for split_name in ("train", "val", "test")
        }
    )
    logger.info("Using manifest splits from %s", splits_path)
else:
    split_ratios = split_payload.get("split_ratios", {"train": 0.8, "val": 0.1, "test": 0.1})
    split_seed = int(split_payload.get("split_seed", 2026))
    shuffled_dataset = raw_dataset.shuffle(seed=split_seed)
    total_rows = len(shuffled_dataset)
    train_end = int(total_rows * float(split_ratios.get("train", 0.8)))
    val_end = train_end + int(total_rows * float(split_ratios.get("val", 0.1)))
    sft_dataset = DatasetDict(
        {
            "train": shuffled_dataset.select(range(0, train_end)),
            "val": shuffled_dataset.select(range(train_end, min(val_end, total_rows))),
            "test": shuffled_dataset.select(range(min(val_end, total_rows), total_rows)),
        }
    )
    logger.warning("Manifest split coverage is too low for the RL-balanced JSONL: %s. "
        "Falling back to deterministic seeded JSONL splits.",
        {split: round(coverage, 4) for split, coverage in split_coverage.items()},
    )

sft_dataset = DatasetDict(
    {
        split_name: split_dataset.map(
            _format_sft_example,
            desc=f"Format {split_name} split for QDoRA SFT",
            num_proc=dataset_num_proc,
        )
        for split_name, split_dataset in sft_dataset.items()
    }
)

dataset = sft_dataset
train_dataset = sft_dataset["train"]
val_dataset = sft_dataset["val"]
test_dataset = sft_dataset["test"]

if max_train_examples is not None:
    limited_train_size = min(int(max_train_examples), len(train_dataset))
    train_dataset = train_dataset.shuffle(seed=train_subset_seed).select(range(limited_train_size))
    logger.info(
        "Using a deterministic train subset of %s examples (seed=%s)",
        len(train_dataset),
        train_subset_seed,
    )

val_metrics_dataset = val_dataset.select(range(min(val_eval_size, len(val_dataset))))
test_metrics_dataset = test_dataset.select(range(min(test_eval_size, len(test_dataset))))

logger.info("Loaded Tinker SFT splits from %s with train=%s val=%s test=%s", data_path, len(train_dataset), len(val_dataset), len(test_dataset))
logger.info(
    "Training text field is available at dataset['train']['text']; example id=%s label=%s",
    train_dataset[0]["openalex_id"] if len(train_dataset) else "n/a",
    train_dataset[0]["disruption_label"] if len(train_dataset) else "n/a",
)
logger.info("Held-out generation eval will use val=%s test=%s examples", len(val_metrics_dataset), len(test_metrics_dataset),)

trainer_train_dataset = train_dataset.remove_columns([column for column in train_dataset.column_names if column != "text"])
trainer_val_dataset = val_metrics_dataset.remove_columns([column for column in val_metrics_dataset.column_names if column != "text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
training_args, trainer_dataset_kwargs = _build_sft_config()

logger.info("Initializing SFTTrainer...")
trainer_start = time.time()
trainer = _build_sft_trainer(
    train_dataset_for_trainer=trainer_train_dataset,
    eval_dataset_for_trainer=trainer_val_dataset,
    training_args=training_args,
    trainer_dataset_kwargs=trainer_dataset_kwargs,
    data_collator=data_collator,
)
trainer.add_callback(
    HeldOutEvalCallback(
        tokenizer=tokenizer,
        eval_dataset=val_metrics_dataset,
        output_dir=output_dir,
        max_length=max_seq_length,
        max_new_tokens=eval_max_new_tokens,
        per_device_eval_batch_size=per_device_eval_batch_size,
        split_name="val",
        logger=logger,
    )
)
logger.info("SFTTrainer initialization completed in %.2f seconds", time.time() - trainer_start)

# train
logger.info("Starting QDoRA training...")
train_start = time.time()
train_result = trainer.train()
logger.info("Training completed in %.2f seconds", time.time() - train_start)
logger.info("Training metrics: %s", train_result.metrics)

if is_main_process(int(os.environ.get("LOCAL_RANK", 0))):
    logger.info("Running final validation generation eval...")
    val_metrics = evaluate_split(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=val_metrics_dataset,
        split_name="val",
        output_dir=output_dir,
        max_length=max_seq_length,
        max_new_tokens=eval_max_new_tokens,
        per_device_eval_batch_size=per_device_eval_batch_size,
        global_step=trainer.state.global_step,
        file_suffix="_final",
        logger=logger,
    )
    logger.info("Final validation metrics: %s", val_metrics)

    logger.info("Running final test generation eval...")
    test_metrics = evaluate_split(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=test_metrics_dataset,
        split_name="test",
        output_dir=output_dir,
        max_length=max_seq_length,
        max_new_tokens=eval_max_new_tokens,
        per_device_eval_batch_size=per_device_eval_batch_size,
        global_step=trainer.state.global_step,
        file_suffix="_final",
        logger=logger,
    )
    logger.info("Final test metrics: %s", test_metrics)

# save weights
final_adapter_dir = output_dir / "final_adapter"
logger.info("Saving final adapter to %s", final_adapter_dir)
trainer.save_model(str(final_adapter_dir))
trainer.save_state()
tokenizer.save_pretrained(str(final_adapter_dir))

# fin
wandb.finish()
logger.info("Training and evaluation completed successfully!")
