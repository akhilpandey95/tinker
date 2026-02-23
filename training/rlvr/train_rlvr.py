#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import random
import shutil
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinker_disruption_rl.tinker_disruption_env import (
    AdversarialDisruptionEnv,
    Paper,
    derive_novelty_label,
)
from training.rlvr.metrics import (
    compute_metrics,
    flatten_metrics,
    load_json,
    load_jsonl,
    select_records_by_split,
    write_generic_csv,
    write_json,
    write_jsonl,
    write_predictions_csv,
)
from training.rlvr.policy import RLVRPolicy, build_policy_from_config
from training.rlvr.prompt_contract import DISRUPTION_LABELS, NOVELTY_LABELS, build_joint_example
from training.rlvr.reward_spec import get_fixed_reward_spec, validate_reward_spec


def _paper_from_record(record: Mapping[str, Any]) -> Paper:
    # Paper enforces novelty_label == derive_novelty_label(novelty_score), while the
    # dataset novelty label uses delta margin thresholding. RLVR reward here is only
    # disruption-based, so we keep env construction consistent with its invariant.
    env_novelty_label = derive_novelty_label(float(record["novelty_score"]))
    return Paper(
        openalex_id=str(record["openalex_id"]),
        title=str(record["title"]),
        abstract=str(record["abstract"]),
        publication_year=int(record["publication_year"]),
        cited_by_count=int(record["cited_by_count"]),
        cd_index=float(record["cd_index"]),
        novelty_score=float(record["novelty_score"]),
        conventionality_score=float(record["conventionality_score"]),
        disruption_label=str(record["disruption_label"]),
        novelty_label=env_novelty_label,
        primary_field=str(record["primary_field"]),
    )


async def _run_single_rollout(
    policy: RLVRPolicy,
    record: Mapping[str, Any],
    rng: random.Random,
    exploration_temperature: float,
    max_env_tokens: int,
) -> Dict[str, Any]:
    paper = _paper_from_record(record)
    env = AdversarialDisruptionEnv(paper=paper, max_tokens=max_env_tokens)
    await env.initial_observation()

    initial = policy.sample_initial_action(
        record=record,
        rng=rng,
        exploration_temperature=exploration_temperature,
    )
    first_result = await env.step(initial.initial_response)
    if first_result.done:
        return {
            "openalex_id": str(record["openalex_id"]),
            "reward": float(first_result.reward),
            "R_correctness": float(first_result.R_correctness),
            "R_reasoning": float(first_result.R_reasoning),
            "R_adaptation": float(first_result.R_adaptation),
            "initial_label": initial.initial_label,
            "final_label": initial.initial_label,
            "initial_probs": dict(initial.initial_probs),
            "final_probs": dict(initial.initial_probs),
            "revised": False,
            "revision_probability": 0.0,
            "parse_error": first_result.info.get("parse_error"),
        }

    challenge_text = str(first_result.observation or "")
    final = policy.sample_final_action(
        record=record,
        challenge_text=challenge_text,
        initial_label=initial.initial_label,
        rng=rng,
        exploration_temperature=exploration_temperature,
    )
    second_result = await env.step(final.final_response)

    return {
        "openalex_id": str(record["openalex_id"]),
        "reward": float(second_result.reward),
        "R_correctness": float(second_result.R_correctness),
        "R_reasoning": float(second_result.R_reasoning),
        "R_adaptation": float(second_result.R_adaptation),
        "initial_label": initial.initial_label,
        "final_label": final.final_label,
        "initial_probs": dict(initial.initial_probs),
        "final_probs": dict(final.final_probs),
        "revised": bool(final.revised),
        "revision_probability": float(final.revision_probability),
        "parse_error": second_result.info.get("parse_error"),
    }


def _parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    bootstrap_args, remaining = bootstrap.parse_known_args()

    config: Dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config = load_json(bootstrap_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Just-RL-style RLVR training from a pretrained base "
            "(deterministic local backend)."
        )
    )
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
        default=Path(config.get("output_dir", "results/rlvr")),
    )
    parser.add_argument("--run-name", default=str(config.get("run_name", "just_rlvr_base")))
    parser.add_argument("--seed", type=int, default=int(config.get("seed", 20260220)))
    parser.add_argument("--train-split", default=str(config.get("train_split", "train")))
    parser.add_argument("--val-split", default=str(config.get("val_split", "val")))
    parser.add_argument(
        "--tiny-max-train",
        type=int,
        default=config.get("tiny_max_train"),
        help="Optional max train examples for tiny local runs.",
    )
    parser.add_argument(
        "--tiny-max-val",
        type=int,
        default=config.get("tiny_max_val"),
        help="Optional max validation examples for tiny local runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=bool(config.get("dry_run", False)),
        help="Enable tiny deterministic dry-run mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=bool(config.get("overwrite", False)),
        help="Overwrite existing run directory if it exists.",
    )
    parser.add_argument(
        "--n-calibration-bins",
        type=int,
        default=int(config.get("n_calibration_bins", 10)),
    )
    parser.add_argument("--epochs", type=int, default=int(config.get("epochs", 4)))
    parser.add_argument("--group-size", type=int, default=int(config.get("group_size", 4)))
    parser.add_argument("--learning-rate", type=float, default=float(config.get("learning_rate", 0.06)))
    parser.add_argument(
        "--revision-learning-rate",
        type=float,
        default=float(config.get("revision_learning_rate", 0.03)),
    )
    parser.add_argument(
        "--exploration-temperature",
        type=float,
        default=float(config.get("exploration_temperature", 1.0)),
    )
    parser.add_argument(
        "--inference-temperature-disruption",
        type=float,
        default=float(config.get("inference_temperature_disruption", 0.8)),
    )
    parser.add_argument(
        "--inference-temperature-novelty",
        type=float,
        default=float(config.get("inference_temperature_novelty", 1.0)),
    )
    parser.add_argument(
        "--max-env-tokens",
        type=int,
        default=int(config.get("max_env_tokens", 320)),
    )

    args = parser.parse_args(remaining)
    args.reward_spec = config.get("reward_spec", get_fixed_reward_spec())
    args.pretrained_base = config.get("pretrained_base", {})
    return args


def _predict_rows(policy: RLVRPolicy, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [policy.predict_joint_row(record) for record in records]


def _mean(items: Sequence[float]) -> float:
    if not items:
        return 0.0
    return sum(items) / len(items)


def main() -> None:
    args = _parse_args()

    if args.n_calibration_bins <= 0:
        raise ValueError("--n-calibration-bins must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if args.revision_learning_rate <= 0:
        raise ValueError("--revision-learning-rate must be positive.")
    if args.exploration_temperature <= 0:
        raise ValueError("--exploration-temperature must be positive.")

    if not isinstance(args.reward_spec, Mapping):
        raise ValueError("Config reward_spec must be an object.")
    validate_reward_spec(args.reward_spec)

    records = load_jsonl(args.dataset_jsonl)
    splits = load_json(args.splits_json)
    split_ids = splits.get("ids", {})

    train_records = select_records_by_split(
        records,
        split_ids=split_ids,
        split_name=args.train_split,
        max_items=args.tiny_max_train,
    )
    val_records = select_records_by_split(
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

    write_jsonl(
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
    write_jsonl(
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

    policy = build_policy_from_config({"pretrained_base": args.pretrained_base})
    rng = random.Random(args.seed)

    reward_trace_rows: List[Dict[str, Any]] = []
    reward_curve_rows: List[Dict[str, Any]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        for record in train_records:
            group_rollouts: List[Dict[str, Any]] = []
            for group_index in range(args.group_size):
                rollout = asyncio.run(
                    _run_single_rollout(
                        policy=policy,
                        record=record,
                        rng=rng,
                        exploration_temperature=args.exploration_temperature,
                        max_env_tokens=args.max_env_tokens,
                    )
                )
                rollout["group_index"] = group_index
                group_rollouts.append(rollout)

            mean_reward = _mean([float(row["reward"]) for row in group_rollouts])
            mean_correctness = _mean([float(row["R_correctness"]) for row in group_rollouts])
            mean_reasoning = _mean([float(row["R_reasoning"]) for row in group_rollouts])
            mean_adaptation = _mean([float(row["R_adaptation"]) for row in group_rollouts])

            global_step += 1
            reward_curve_rows.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "openalex_id": str(record["openalex_id"]),
                    "group_size": args.group_size,
                    "mean_reward": mean_reward,
                    "mean_R_correctness": mean_correctness,
                    "mean_R_reasoning": mean_reasoning,
                    "mean_R_adaptation": mean_adaptation,
                }
            )

            for rollout in group_rollouts:
                advantage = float(rollout["reward"]) - mean_reward
                policy.policy_gradient_update_disruption(
                    record=record,
                    action_label=str(rollout["final_label"]),
                    action_probs=rollout["final_probs"],
                    advantage=advantage,
                    learning_rate=args.learning_rate,
                )
                policy.update_revision_bias(
                    revised=bool(rollout["revised"]),
                    revision_probability=float(rollout["revision_probability"]),
                    advantage=advantage,
                    learning_rate=args.revision_learning_rate,
                )

                reward_trace_rows.append(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "openalex_id": str(rollout["openalex_id"]),
                        "group_index": int(rollout["group_index"]),
                        "reward": float(rollout["reward"]),
                        "R_correctness": float(rollout["R_correctness"]),
                        "R_reasoning": float(rollout["R_reasoning"]),
                        "R_adaptation": float(rollout["R_adaptation"]),
                        "advantage": float(advantage),
                        "initial_label": str(rollout["initial_label"]),
                        "final_label": str(rollout["final_label"]),
                        "revised": int(bool(rollout["revised"])),
                        "revision_probability": float(rollout["revision_probability"]),
                        "parse_error": rollout.get("parse_error"),
                    }
                )

    policy.disruption_temperature = float(args.inference_temperature_disruption)
    policy.novelty_temperature = float(args.inference_temperature_novelty)

    train_rows = _predict_rows(policy, train_records)
    val_rows = _predict_rows(policy, val_records)

    train_metrics, train_calibration = compute_metrics(train_rows, n_bins=args.n_calibration_bins)
    val_metrics, val_calibration = compute_metrics(val_rows, n_bins=args.n_calibration_bins)

    write_predictions_csv(run_dir / "predictions_train.csv", train_rows)
    write_predictions_csv(run_dir / "predictions_val.csv", val_rows)

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
    write_generic_csv(
        run_dir / "calibration_train_disruption.csv",
        fields=calibration_fields,
        rows=train_calibration["disruption"],
    )
    write_generic_csv(
        run_dir / "calibration_train_novelty.csv",
        fields=calibration_fields,
        rows=train_calibration["novelty"],
    )
    write_generic_csv(
        run_dir / "calibration_val_disruption.csv",
        fields=calibration_fields,
        rows=val_calibration["disruption"],
    )
    write_generic_csv(
        run_dir / "calibration_val_novelty.csv",
        fields=calibration_fields,
        rows=val_calibration["novelty"],
    )

    write_json(run_dir / "metrics_train.json", train_metrics)
    write_json(run_dir / "metrics_val.json", val_metrics)
    write_generic_csv(
        run_dir / "metrics_train.csv",
        fields=["scope", "metric", "value"],
        rows=flatten_metrics(train_metrics),
    )
    write_generic_csv(
        run_dir / "metrics_val.csv",
        fields=["scope", "metric", "value"],
        rows=flatten_metrics(val_metrics),
    )

    write_generic_csv(
        run_dir / "reward_trace.csv",
        fields=[
            "step",
            "epoch",
            "openalex_id",
            "group_index",
            "reward",
            "R_correctness",
            "R_reasoning",
            "R_adaptation",
            "advantage",
            "initial_label",
            "final_label",
            "revised",
            "revision_probability",
            "parse_error",
        ],
        rows=reward_trace_rows,
    )
    write_generic_csv(
        run_dir / "reward_curve.csv",
        fields=[
            "step",
            "epoch",
            "openalex_id",
            "group_size",
            "mean_reward",
            "mean_R_correctness",
            "mean_R_reasoning",
            "mean_R_adaptation",
        ],
        rows=reward_curve_rows,
    )

    reward_summary = {
        "reward_count": len(reward_trace_rows),
        "mean_reward": _mean([float(row["reward"]) for row in reward_trace_rows]),
        "mean_R_correctness": _mean([float(row["R_correctness"]) for row in reward_trace_rows]),
        "mean_R_reasoning": _mean([float(row["R_reasoning"]) for row in reward_trace_rows]),
        "mean_R_adaptation": _mean([float(row["R_adaptation"]) for row in reward_trace_rows]),
        "revision_rate": _mean([float(row["revised"]) for row in reward_trace_rows]),
        "parse_error_rate": _mean(
            [1.0 if row.get("parse_error") else 0.0 for row in reward_trace_rows]
        ),
    }
    write_json(run_dir / "reward_summary.json", reward_summary)

    model_artifact = {
        "artifact_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": "just_rl_rlvr_base",
        "backend": "rlvr_dry_run" if args.dry_run else "rlvr_reference",
        "prompt_contract": "combined_impact_v1",
        "label_targets": {
            "disruption": list(DISRUPTION_LABELS),
            "novelty": list(NOVELTY_LABELS),
        },
        "policy_state": policy.to_dict(),
        "reward_spec": args.reward_spec,
        "training": {
            "epochs": args.epochs,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "revision_learning_rate": args.revision_learning_rate,
            "exploration_temperature": args.exploration_temperature,
            "max_env_tokens": args.max_env_tokens,
        },
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
    write_json(run_dir / "model_artifact.json", model_artifact)

    run_manifest = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "n_calibration_bins": int(args.n_calibration_bins),
        "config": {
            "epochs": args.epochs,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "revision_learning_rate": args.revision_learning_rate,
            "exploration_temperature": args.exploration_temperature,
            "inference_temperature_disruption": args.inference_temperature_disruption,
            "inference_temperature_novelty": args.inference_temperature_novelty,
            "max_env_tokens": args.max_env_tokens,
        },
        "artifacts": {
            "model_artifact": str(run_dir / "model_artifact.json"),
            "metrics_train_json": str(run_dir / "metrics_train.json"),
            "metrics_val_json": str(run_dir / "metrics_val.json"),
            "predictions_train_csv": str(run_dir / "predictions_train.csv"),
            "predictions_val_csv": str(run_dir / "predictions_val.csv"),
            "calibration_val_disruption_csv": str(run_dir / "calibration_val_disruption.csv"),
            "calibration_val_novelty_csv": str(run_dir / "calibration_val_novelty.csv"),
            "reward_trace_csv": str(run_dir / "reward_trace.csv"),
            "reward_curve_csv": str(run_dir / "reward_curve.csv"),
            "reward_summary_json": str(run_dir / "reward_summary.json"),
        },
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    print(f"RUN_DIR={run_dir}")
    print(f"MODEL_ARTIFACT={run_dir / 'model_artifact.json'}")
    print(f"TRAIN_MEAN_REWARD={reward_summary['mean_reward']:.4f}")
    print(f"VAL_JOINT_ACCURACY={val_metrics['overall']['joint_exact_match_accuracy']:.4f}")


if __name__ == "__main__":
    main()
