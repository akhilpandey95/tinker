#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import math
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Optional hosted SDK import.
    import tinker as tinker_sdk  # type: ignore
except Exception:  # pragma: no cover
    tinker_sdk = None

from training.rlvr.metrics import write_generic_csv, write_json, write_jsonl
from training.rlvr.reward_spec import get_fixed_reward_spec
from training.tinker_rl.tinker_dataset import (
    SciSciNetRLDataset,
    build_batch_histogram_summary,
)
from training.tinker_rl.tinker_env_adapter import (
    Action,
    COOKBOOK_TYPES_AVAILABLE,
    DISRUPTION_LABELS,
    SimpleMessageRenderer,
    TinkerDisruptionEnv,
    find_prompt_leakage_markers,
)


def _load_json(path: Path) -> Dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()

    config: Dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config = _load_json(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description="Tinker-hosted RL bridge for disruption prediction (v1 single-turn smoke)."
    )
    parser.add_argument("--config", type=Path, default=bootstrap_args.config)
    parser.add_argument("--base-model", default=str(config.get("base_model", "Qwen/Qwen3-8B")))
    parser.add_argument("--model-family", default=str(config.get("model_family", "qwen3")))
    parser.add_argument("--env-name", default=str(config.get("env_name", "disruption")))
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=Path(config.get("dataset_jsonl", "data/sciscinet/disruption_novelty_sciscinet_500k.jsonl")),
    )
    parser.add_argument(
        "--splits-json",
        type=Path,
        default=Path(config.get("splits_json", "data/sciscinet/disruption_novelty_sciscinet_500k.splits.json")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.get("output_dir", "results/tinker_rl")),
    )
    parser.add_argument("--run-name", default=str(config.get("run_name", "tinker_rl_dry_run")))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 20260220)))
    parser.add_argument("--train-split", default=str(config.get("train_split", "train")))
    parser.add_argument("--val-split", default=str(config.get("val_split", "val")))
    parser.add_argument("--lora-rank", type=int, default=int(config.get("lora_rank", 32)))
    parser.add_argument("--base-lr", type=float, default=float(config.get("base_lr", 1e-5)))
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 1)))
    parser.add_argument("--group-size", type=int, default=int(config.get("group_size", 4)))
    parser.add_argument("--batch-size", type=int, default=int(config.get("batch_size", 4)))
    parser.add_argument("--max-env-tokens", type=int, default=int(config.get("max_env_tokens", 256)))
    parser.add_argument("--max-train", type=int, default=config.get("max_train"))
    parser.add_argument("--max-val", type=int, default=config.get("max_val"))
    parser.add_argument(
        "--sampling-strategy",
        default=str(config.get("sampling_strategy", "stratified")),
        choices=("stratified", "natural"),
    )
    parser.add_argument("--prompt-max-chars", type=int, default=int(config.get("prompt_max_chars", 2048)))
    parser.add_argument("--system-prompt", default=str(config.get("system_prompt", "")))
    parser.add_argument(
        "--include-concepts",
        action="store_true",
        default=bool(config.get("include_concepts", False)),
    )
    parser.add_argument("--dry-run", action="store_true", default=bool(config.get("dry_run", False)))
    parser.add_argument("--overwrite", action="store_true", default=bool(config.get("overwrite", False)))
    parser.add_argument(
        "--sample-trajectory-limit",
        type=int,
        default=int(config.get("sample_trajectory_limit", 8)),
    )
    parser.add_argument(
        "--prompt-check-limit",
        type=int,
        default=int(config.get("prompt_check_limit", 8)),
    )
    parser.add_argument(
        "--offline-policy",
        default=str(config.get("offline_policy", "uniform_valid_format")),
        choices=("uniform_valid_format",),
    )
    args = parser.parse_args(remaining)
    args.config_payload = config
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.env_name.strip().lower() != "disruption":
        raise ValueError("v1 supports only --env-name disruption.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_env_tokens <= 0:
        raise ValueError("--max-env-tokens must be positive.")
    if args.prompt_max_chars <= 0:
        raise ValueError("--prompt-max-chars must be positive.")
    if args.lora_rank <= 0:
        raise ValueError("--lora-rank must be positive.")
    if args.base_lr <= 0:
        raise ValueError("--base-lr must be positive.")
    if args.max_train is not None and int(args.max_train) <= 0:
        raise ValueError("--max-train must be positive when set.")
    if args.max_val is not None and int(args.max_val) <= 0:
        raise ValueError("--max-val must be positive when set.")


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _heuristic_lora_lr_multiplier(base_model: str) -> tuple[float, str]:
    name = base_model.strip().lower()
    if "qwen3-8b" in name:
        return 64.0, "heuristic_qwen3_8b"
    if "qwen3" in name:
        return 64.0, "heuristic_qwen3_family"
    return 32.0, "heuristic_default"


def _resolved_config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "base_model": args.base_model,
        "model_family": args.model_family,
        "env_name": args.env_name,
        "dataset_jsonl": str(args.dataset_jsonl),
        "splits_json": str(args.splits_json),
        "output_dir": str(args.output_dir),
        "run_name": args.run_name,
        "seed": args.seed,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "lora_rank": args.lora_rank,
        "base_lr": args.base_lr,
        "epochs": args.epochs,
        "group_size": args.group_size,
        "batch_size": args.batch_size,
        "max_env_tokens": args.max_env_tokens,
        "max_train": args.max_train,
        "max_val": args.max_val,
        "sampling_strategy": args.sampling_strategy,
        "prompt_max_chars": args.prompt_max_chars,
        "system_prompt": args.system_prompt,
        "include_concepts": bool(args.include_concepts),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "sample_trajectory_limit": args.sample_trajectory_limit,
        "prompt_check_limit": args.prompt_check_limit,
        "offline_policy": args.offline_policy,
    }


def _build_renderer(args: argparse.Namespace) -> tuple[Any, Dict[str, Any]]:
    # Local dry-run path: byte-level shim. Hosted path can be upgraded to cookbook renderer
    # without touching the env adapter or dataset interfaces.
    renderer = SimpleMessageRenderer()
    info = {
        "renderer_backend": "simple_bytes_fallback",
        "model_family": args.model_family,
        "cookbook_types_available": bool(COOKBOOK_TYPES_AVAILABLE),
        "validation_scope": "offline_smoke_prompt_serialization_only",
    }
    return renderer, info


def _detect_hosted_prereqs() -> Dict[str, Any]:
    tinker_available = bool(importlib.util.find_spec("tinker"))
    cookbook_available = bool(importlib.util.find_spec("tinker_cookbook"))
    api_key_present = bool(os.environ.get("TINKER_API_KEY"))
    return {
        "tinker_sdk_available": tinker_available,
        "tinker_sdk_version": getattr(tinker_sdk, "__version__", None) if tinker_sdk is not None else None,
        "tinker_cookbook_available": cookbook_available,
        "tinker_api_key_present": api_key_present,
        "tinker_base_url": os.environ.get("TINKER_BASE_URL"),
    }


def _bootstrap_hosted_clients(args: argparse.Namespace, *, attempt_remote: bool) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "sdk_imported": tinker_sdk is not None,
        "sdk_version": getattr(tinker_sdk, "__version__", None) if tinker_sdk is not None else None,
        "attempt_remote": bool(attempt_remote),
        "service_client_created": False,
        "sampling_client_created": False,
        "training_client_created": False,
    }
    if tinker_sdk is None:
        status["reason"] = "tinker SDK not importable"
        return status
    if not attempt_remote:
        status["reason"] = "dry_run: remote client bootstrap skipped"
        return status
    if not os.environ.get("TINKER_API_KEY"):
        status["reason"] = "TINKER_API_KEY not set"
        return status

    try:
        service_client = tinker_sdk.ServiceClient()
        status["service_client_created"] = True
    except Exception as exc:  # pragma: no cover - depends on env.
        status["service_client_error"] = repr(exc)
        return status

    try:
        _ = service_client.create_sampling_client(base_model=args.base_model)
        status["sampling_client_created"] = True
    except Exception as exc:  # pragma: no cover - depends on env.
        status["sampling_client_error"] = repr(exc)

    try:
        _ = service_client.create_lora_training_client(
            base_model=args.base_model,
            rank=int(args.lora_rank),
            seed=int(args.seed),
            train_attn=True,
            train_mlp=True,
            train_unembed=True,
        )
        status["training_client_created"] = True
    except Exception as exc:  # pragma: no cover - depends on env.
        status["training_client_error"] = repr(exc)

    return status


def _make_run_dir(output_dir: Path, run_name: str, overwrite: bool) -> Path:
    run_dir = output_dir / run_name
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _offline_response_text(rng: random.Random) -> tuple[str, str]:
    label = DISRUPTION_LABELS[rng.randrange(len(DISRUPTION_LABELS))]
    reasoning = (
        "Because citation and reference patterns in the abstract and field context can signal "
        "disruptive versus consolidating behavior, this label is the most plausible estimate."
    )
    return label, reasoning


async def _run_single_env_rollout(
    env: TinkerDisruptionEnv,
    renderer: SimpleMessageRenderer,
    rng: random.Random,
) -> Dict[str, Any]:
    _observation, _stop = await env.initial_observation()
    label, reasoning = _offline_response_text(rng)
    response_text = f"disruption: {label}\nreasoning: {reasoning}"
    action = Action(tokens=renderer.encode_assistant_content(response_text))
    step_result = await env.step(action)

    inner = env.last_inner_result
    if inner is None:
        raise RuntimeError("Adapter did not capture inner StepResult.")

    reward_components = dict(inner.reward_components)
    parse_error = inner.info.get("parse_error")
    predicted = inner.info.get("predicted_disruption")
    return {
        "reward": float(step_result.reward),
        "reward_components": reward_components,
        "parse_error": parse_error,
        "predicted_disruption": predicted,
        "response_text": response_text,
        "prompt_text": env.last_prompt_text,
        "adapter_parse_success": bool(env.last_parse_success),
        "adapter_parsed_text": env.last_parsed_text,
    }


def _title_abstract_char_sum(prompt_text: str) -> Optional[int]:
    title_match = None
    abstract_match = None
    for line in prompt_text.splitlines():
        if title_match is None and line.startswith("Title: "):
            title_match = line[len("Title: ") :]
            continue
        if abstract_match is None and line.startswith("Abstract: "):
            abstract_match = line[len("Abstract: ") :]
            continue
        if title_match is not None and abstract_match is not None:
            break
    if title_match is None or abstract_match is None:
        return None
    return len(title_match) + len(abstract_match)


def _run_offline_smoke(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset: SciSciNetRLDataset,
    renderer: SimpleMessageRenderer,
    dataset_info: Mapping[str, Any],
    renderer_info: Mapping[str, Any],
    hosted_prereqs: Mapping[str, Any],
    hosted_client_bootstrap: Mapping[str, Any],
    lora_lr_multiplier: float,
    lora_lr_source: str,
) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    reward_trace_rows: List[Dict[str, Any]] = []
    sample_trajectories: List[Dict[str, Any]] = []
    batch_histograms: List[Dict[str, Any]] = []
    prompt_checks: List[Dict[str, Any]] = []
    prompt_checked_ids: set[str] = set()
    global_step = 0

    train_batches_per_epoch = max(1, math.ceil(len(dataset.train_records) / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        for batch_idx in range(train_batches_per_epoch):
            builders = dataset.get_batch(batch_idx)
            batch_hist = dataset.observed_batch_label_histogram(builders)
            batch_histograms.append(
                {
                    "epoch": epoch,
                    "batch_index": batch_idx,
                    **batch_hist,
                }
            )

            for builder in builders:
                if len(prompt_checks) < args.prompt_check_limit and builder.openalex_id not in prompt_checked_ids:
                    prompt_text = builder.build_prompt_text()
                    leakage_markers = find_prompt_leakage_markers(prompt_text)
                    prompt_checks.append(
                        {
                            "openalex_id": builder.openalex_id,
                            "prompt_char_len": len(prompt_text),
                            "title_abstract_char_sum": _title_abstract_char_sum(prompt_text),
                            "has_leakage": bool(leakage_markers),
                            "leakage_markers": leakage_markers,
                            "prompt_preview": prompt_text[:500],
                        }
                    )
                    prompt_checked_ids.add(builder.openalex_id)

                envs = asyncio.run(builder.make_envs())
                for group_index, env in enumerate(envs):
                    rollout = asyncio.run(_run_single_env_rollout(env, renderer, rng))
                    global_step += 1
                    inner_reward_components = dict(rollout["reward_components"])
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "batch_index": batch_idx,
                        "group_index": group_index,
                        "openalex_id": builder.openalex_id,
                        "gold_disruption": builder.disruption_label,
                        "predicted_disruption": rollout.get("predicted_disruption"),
                        "reward": float(rollout["reward"]),
                        "R_correctness": float(inner_reward_components.get("R_correctness", 0.0)),
                        "R_reasoning": float(inner_reward_components.get("R_reasoning", 0.0)),
                        "R_adaptation": float(inner_reward_components.get("R_adaptation", 0.0)),
                        "parse_error": rollout.get("parse_error"),
                        "prompt_char_len": len(str(rollout["prompt_text"])),
                        "adapter_parse_success": int(bool(rollout["adapter_parse_success"])),
                    }
                    reward_trace_rows.append(row)

                    if len(sample_trajectories) < args.sample_trajectory_limit:
                        sample_trajectories.append(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "batch_index": batch_idx,
                                "group_index": group_index,
                                "openalex_id": builder.openalex_id,
                                "gold_disruption": builder.disruption_label,
                                "predicted_disruption": rollout.get("predicted_disruption"),
                                "response_text": rollout["response_text"],
                                "reward": float(rollout["reward"]),
                                "reward_components": inner_reward_components,
                                "parse_error": rollout.get("parse_error"),
                                "prompt_has_leakage": bool(find_prompt_leakage_markers(str(rollout["prompt_text"]))),
                                "prompt_preview": str(rollout["prompt_text"])[:500],
                            }
                        )

    reward_values = [float(row["reward"]) for row in reward_trace_rows]
    parse_error_rate = _mean([1.0 if row.get("parse_error") else 0.0 for row in reward_trace_rows])
    correctness_values = [float(row["R_correctness"]) for row in reward_trace_rows]
    reasoning_values = [float(row["R_reasoning"]) for row in reward_trace_rows]
    adaptation_values = [float(row["R_adaptation"]) for row in reward_trace_rows]
    gold_counts = Counter(str(row["gold_disruption"]) for row in reward_trace_rows)
    pred_counts = Counter(str(row["predicted_disruption"]) for row in reward_trace_rows if row.get("predicted_disruption"))

    prompt_leakage_rows = [row for row in prompt_checks if row["has_leakage"]]
    title_abstract_sums = [
        int(row["title_abstract_char_sum"])
        for row in prompt_checks
        if row.get("title_abstract_char_sum") is not None
    ]
    prompt_leakage_summary = {
        "checked_prompts": len(prompt_checks),
        "with_leakage": len(prompt_leakage_rows),
        "forbidden_markers": [
            "CD Index:",
            "Novelty Score:",
            "Conventionality Score:",
            "cd_index",
            "novelty_score",
            "conventionality_score",
        ],
        "title_abstract_char_sum_max": max(title_abstract_sums) if title_abstract_sums else None,
        "title_abstract_cap": int(args.prompt_max_chars),
        "title_abstract_cap_respected": (
            all(total <= int(args.prompt_max_chars) for total in title_abstract_sums)
            if title_abstract_sums
            else None
        ),
        "status": "pass" if not prompt_leakage_rows else "fail",
    }

    reward_summary = {
        "reward_count": len(reward_trace_rows),
        "mean_reward": _mean(reward_values),
        "min_reward": min(reward_values) if reward_values else 0.0,
        "max_reward": max(reward_values) if reward_values else 0.0,
        "mean_R_correctness": _mean(correctness_values),
        "mean_R_reasoning": _mean(reasoning_values),
        "mean_R_adaptation": _mean(adaptation_values),
        "parse_error_rate": parse_error_rate,
        "rewards_non_degenerate": bool(reward_values and (min(reward_values) != max(reward_values))),
        "gold_label_counts_in_rollouts": {label: int(gold_counts.get(label, 0)) for label in DISRUPTION_LABELS},
        "predicted_label_counts_in_rollouts": {label: int(pred_counts.get(label, 0)) for label in DISRUPTION_LABELS},
    }

    batch_hist_summary = build_batch_histogram_summary(batch_histograms)
    write_json(run_dir / "config_input.json", dict(args.config_payload))
    write_json(run_dir / "config_resolved.json", _resolved_config_dict(args))
    write_json(run_dir / "dataset_info.json", dict(dataset_info))
    write_json(run_dir / "renderer_info.json", dict(renderer_info))
    write_json(run_dir / "hosted_prereqs.json", dict(hosted_prereqs))
    write_json(run_dir / "hosted_client_bootstrap.json", dict(hosted_client_bootstrap))
    write_json(run_dir / "sampling_policy.json", dataset.sampling_manifest())
    write_json(run_dir / "train_batch_histograms.json", batch_hist_summary)
    write_json(run_dir / "prompt_leakage_check.json", prompt_leakage_summary)
    write_jsonl(run_dir / "prompt_leakage_samples.jsonl", prompt_checks)
    write_json(run_dir / "reward_summary.json", reward_summary)
    write_jsonl(run_dir / "sample_trajectories.jsonl", sample_trajectories)
    write_generic_csv(
        run_dir / "reward_trace.csv",
        fields=[
            "step",
            "epoch",
            "batch_index",
            "group_index",
            "openalex_id",
            "gold_disruption",
            "predicted_disruption",
            "reward",
            "R_correctness",
            "R_reasoning",
            "R_adaptation",
            "parse_error",
            "prompt_char_len",
            "adapter_parse_success",
        ],
        rows=reward_trace_rows,
    )

    run_manifest = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "backend_mode": "offline_smoke",
        "dry_run": bool(args.dry_run),
        "env_name": args.env_name,
        "model": {
            "base_model": args.base_model,
            "model_family": args.model_family,
        },
        "lora": {
            "rank": args.lora_rank,
            "base_lr": args.base_lr,
            "lora_lr_multiplier": lora_lr_multiplier,
            "lora_lr_multiplier_source": lora_lr_source,
            "effective_lr": args.base_lr * lora_lr_multiplier,
            "train_all_weight_matrices_target": True,
        },
        "rl_hyperparams": {
            "epochs": args.epochs,
            "group_size": args.group_size,
            "batch_size": args.batch_size,
            "max_env_tokens": args.max_env_tokens,
            "sampling_strategy": args.sampling_strategy,
            "seed": args.seed,
        },
        "task_contract": {
            "turns": 1,
            "response_format": [
                "disruption: <label>",
                "reasoning: <short justification>",
            ],
            "prompt_max_chars_title_plus_abstract": args.prompt_max_chars,
            "leakage_markers_removed": [
                "CD Index",
                "Novelty Score",
                "Conventionality Score",
            ],
        },
        "reference_reward_spec_adversarial_local": get_fixed_reward_spec(),
        "v1_reward_spec": {
            "environment": "DisruptionPredictionEnvV1",
            "formula": "R = R_correctness + R_reasoning",
            "R_correctness": {"correct": 1.0, "incorrect": -1.0},
            "R_reasoning": {"max": 0.5},
            "R_adaptation": {"max": 0.0},
            "total_min": -1.0,
            "total_max": 1.5,
        },
        "dataset": dict(dataset_info),
        "sampling_policy": dataset.sampling_manifest(),
        "renderer": dict(renderer_info),
        "hosted_prereqs": dict(hosted_prereqs),
        "hosted_client_bootstrap": dict(hosted_client_bootstrap),
        "artifacts": {
            "config_resolved_json": str(run_dir / "config_resolved.json"),
            "config_input_json": str(run_dir / "config_input.json"),
            "dataset_info_json": str(run_dir / "dataset_info.json"),
            "renderer_info_json": str(run_dir / "renderer_info.json"),
            "hosted_prereqs_json": str(run_dir / "hosted_prereqs.json"),
            "hosted_client_bootstrap_json": str(run_dir / "hosted_client_bootstrap.json"),
            "sampling_policy_json": str(run_dir / "sampling_policy.json"),
            "train_batch_histograms_json": str(run_dir / "train_batch_histograms.json"),
            "prompt_leakage_check_json": str(run_dir / "prompt_leakage_check.json"),
            "prompt_leakage_samples_jsonl": str(run_dir / "prompt_leakage_samples.jsonl"),
            "reward_trace_csv": str(run_dir / "reward_trace.csv"),
            "reward_summary_json": str(run_dir / "reward_summary.json"),
            "sample_trajectories_jsonl": str(run_dir / "sample_trajectories.jsonl"),
        },
        "hosted_status": {
            "state": "not_attempted",
            "reason": "dry_run uses offline smoke backend for env/prompt/sampling validation",
        },
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    return {
        "run_manifest": run_manifest,
        "reward_summary": reward_summary,
        "prompt_leakage_summary": prompt_leakage_summary,
        "batch_hist_summary": batch_hist_summary,
    }


def _hosted_blocker_message(hosted_prereqs: Mapping[str, Any]) -> str:
    missing: List[str] = []
    if not hosted_prereqs.get("tinker_sdk_available"):
        missing.append("tinker SDK")
    if not hosted_prereqs.get("tinker_cookbook_available"):
        missing.append("tinker_cookbook")
    if not hosted_prereqs.get("tinker_api_key_present"):
        missing.append("TINKER_API_KEY")
    if not missing:
        missing.append("hosted RL loop integration is dependency-gated in this workspace")
    return "Hosted Tinker RL path blocked: missing " + ", ".join(missing)


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    run_dir = _make_run_dir(args.output_dir, args.run_name, overwrite=bool(args.overwrite))
    renderer, renderer_info = _build_renderer(args)
    hosted_prereqs = _detect_hosted_prereqs()
    hosted_client_bootstrap = _bootstrap_hosted_clients(args, attempt_remote=not bool(args.dry_run))
    lora_lr_multiplier, lora_lr_source = _heuristic_lora_lr_multiplier(args.base_model)

    dataset, dataset_info = SciSciNetRLDataset.from_files(
        dataset_jsonl=args.dataset_jsonl,
        splits_json=args.splits_json,
        renderer=renderer,
        group_size=args.group_size,
        batch_size=args.batch_size,
        system_prompt=args.system_prompt,
        max_env_tokens=args.max_env_tokens,
        prompt_max_chars=args.prompt_max_chars,
        sampling_strategy=args.sampling_strategy,
        seed=args.seed,
        train_split=args.train_split,
        val_split=args.val_split,
        max_train=args.max_train,
        max_val=args.max_val,
        include_concepts=bool(args.include_concepts),
    )

    if args.dry_run:
        result = _run_offline_smoke(
            args=args,
            run_dir=run_dir,
            dataset=dataset,
            renderer=renderer,
            dataset_info=dataset_info,
            renderer_info=renderer_info,
            hosted_prereqs=hosted_prereqs,
            hosted_client_bootstrap=hosted_client_bootstrap,
            lora_lr_multiplier=lora_lr_multiplier,
            lora_lr_source=lora_lr_source,
        )
        print(f"RUN_DIR={run_dir}")
        print("BACKEND=offline_smoke")
        print(f"PROMPT_LEAKAGE_STATUS={result['prompt_leakage_summary']['status']}")
        print(f"MEAN_REWARD={result['reward_summary']['mean_reward']:.4f}")
        print(f"PARSE_ERROR_RATE={result['reward_summary']['parse_error_rate']:.4f}")
        return

    blocker = _hosted_blocker_message(hosted_prereqs)
    write_json(
        run_dir / "hosted_blocker.json",
        {
            "blocked": True,
            "message": blocker,
            "hosted_prereqs": hosted_prereqs,
            "hosted_client_bootstrap": hosted_client_bootstrap,
            "renderer_info": renderer_info,
            "dataset_info": dataset_info,
            "lora_rank": args.lora_rank,
            "base_lr": args.base_lr,
            "lora_lr_multiplier": lora_lr_multiplier,
            "effective_lr": args.base_lr * lora_lr_multiplier,
        },
    )
    raise RuntimeError(
        blocker
        + ". Run with --dry-run for offline validation here, or install tinker_cookbook and set "
        "TINKER_API_KEY in a hosted-enabled environment."
    )


if __name__ == "__main__":
    main()
