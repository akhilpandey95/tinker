#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""Lean hosted RL runner for the SciSciNet cookbook workflow.

This script is the "run launcher".

Refactor goals (v2):
- Fix boolean CLI flags that were previously impossible to disable.
- Fix a subtle logging bias bug for stratified sampling: when the trainer logs a fixed
  prefix of env-groups each batch, a fixed label ordering can permanently hide a class.
  We now rotate (or shuffle) label order by batch index.
- Detect forbidden markers (e.g. <think>) on the raw model output by default.
  Previously the renderer could strip these before scoring, making the metric lie.
- Emit per-episode one-hot confusion indicators into metrics.jsonl so viz.py can compute
  calibration drift from metrics alone (no HTML scraping required).

The runner intentionally reuses dataset + prompt + reward logic from sciscinet_cookbook_mini.py.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


# ----------------------------
# Environment / path helpers
# ----------------------------

def load_env_file(path: Path, *, override: bool = False) -> dict[str, Any]:
    status = {"path": str(path), "exists": path.exists(), "keys_loaded": 0, "loader": None}
    if not path.exists():
        return status
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=override)
        status["loader"] = "python-dotenv"
        return status
    except Exception:
        loaded = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            before = key in os.environ
            if override or not before:
                os.environ[key] = value
            loaded += int((override or not before) and bool(key))
        status["loader"] = "manual-parser"
        status["keys_loaded"] = loaded
        return status


def bootstrap_paths(script_dir: Path, repo_root: Path) -> list[str]:
    candidates: list[Path] = [
        script_dir,
        repo_root,
        repo_root.parent,
        repo_root.parent / "tinker-cookbook",
        repo_root.parent / "tinker_cookbook",
        repo_root / "tinker-cookbook",
        repo_root / "tinker_cookbook",
    ]
    env_cookbook = os.environ.get("TINKER_COOKBOOK_ROOT")
    if env_cookbook:
        candidates.append(Path(env_cookbook).expanduser())

    added: list[str] = []
    for p in candidates:
        rp = p.resolve()
        if not rp.exists():
            continue
        rp_s = str(rp)
        if rp_s not in sys.path:
            sys.path.insert(0, rp_s)
            added.append(rp_s)
    return added


def try_git_commit(root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str) -> None:
    """Add --foo / --no-foo boolean flag (compatible across Python versions)."""

    dest = name.lstrip("-").replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, default=default, action=argparse.BooleanOptionalAction, help=help)
        return

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


# ----------------------------
# CLI
# ----------------------------

def parse_args(dataset_choices: Sequence[str], default_data_root: Path, default_output_root: Path) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hosted RL runner (lean) for SciSciNet cookbook")

    p.add_argument("--data-root", type=Path, default=default_data_root)
    p.add_argument("--dataset-key", choices=sorted(dataset_choices), default="sci_balanced_from2m_no_ovr_rl_balanced")

    p.add_argument(
        "--split-mode",
        choices=["use_official_or_generate", "force_generate", "train_only"],
        default="use_official_or_generate",
    )
    p.add_argument(
        "--loading-mode",
        choices=["use_splits_head", "use_splits_label_aware_quotas", "rl_balanced_stream_sample"],
        default="rl_balanced_stream_sample",
    )

    p.add_argument("--seed", type=int, default=20260220)
    p.add_argument("--dataset-seed", type=int, default=20260220)

    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)

    p.add_argument("--max-train", type=int)
    p.add_argument("--max-val", type=int)
    p.add_argument("--max-test", type=int)

    p.add_argument("--train-quota-consolidating", type=int, default=512)
    p.add_argument("--train-quota-disruptive", type=int, default=512)
    p.add_argument("--train-quota-neutral", type=int, default=512)
    p.add_argument("--val-quota-consolidating", type=int, default=128)
    p.add_argument("--val-quota-disruptive", type=int, default=128)
    p.add_argument("--val-quota-neutral", type=int, default=128)

    p.add_argument("--working-set-per-label", type=int, default=1024)
    p.add_argument("--working-set-total", type=int)
    add_bool_arg(p, "--shuffle-selected", default=True, help="Shuffle selected working-set records")

    p.add_argument(
        "--env-variant",
        choices=[
            "single_turn_disruption",
            "single_turn_disruption_novelty",
            "adversarial_two_turn_disruption_novelty",
        ],
        default="single_turn_disruption",
    )
    p.add_argument("--prompt-max-chars", type=int, default=1800)
    add_bool_arg(p, "--include-concepts", default=False, help="Include concepts string if available")
    add_bool_arg(p, "--include-definitions", default=False, help="Include label definitions in the prompt")

    # Reward weights and shaping.
    p.add_argument("--weight-disruption", type=float, default=1.0)
    p.add_argument("--weight-novelty", type=float, default=1.0)
    p.add_argument("--reward-disruption-consolidating", type=float, default=3.0)
    p.add_argument("--reward-disruption-disruptive", type=float, default=2.0)
    p.add_argument("--reward-disruption-neutral", type=float, default=1.0)

    add_bool_arg(p, "--strict-output-format", default=True, help="Penalize format violations")
    p.add_argument("--format-check", choices=["exact", "keys"], default="exact")
    p.add_argument("--format-violation-penalty", type=float, default=1.0)
    p.add_argument("--forbidden-marker-policy", choices=["penalize", "ignore"], default="penalize")

    p.add_argument(
        "--forbidden-marker-detection",
        choices=["raw", "sanitized", "both"],
        default="raw",
        help="Where to detect forbidden markers like <think>. Default: raw (what the model emitted).",
    )

    p.add_argument(
        "--reasoning-bonus-mode",
        choices=["always", "only_if_correct", "only_if_strict_ok", "disabled"],
        default="always",
    )
    p.add_argument("--reasoning-bonus-weight", type=float, default=1.0)

    # Model / RL config.
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer-name", default="")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)

    # Batch structure.
    p.add_argument("--groups-per-batch", type=int, default=6)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--n-batches", type=int, default=12)

    p.add_argument("--sampling-strategy", choices=["natural", "stratified"], default="stratified")
    p.add_argument(
        "--stratified-order",
        choices=["rotate", "shuffle", "fixed"],
        default="rotate",
        help=(
            "How to order labels in stratified sampling. 'rotate' avoids logging bias when only a prefix "
            "of groups is logged each batch."
        ),
    )

    # Logging / eval.
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument(
        "--num-groups-to-log",
        type=int,
        default=3,
        help="How many env-groups per batch to log in detail (recommend >=3 for 3-way labels).",
    )

    add_bool_arg(p, "--eval-on-val", default=False, help="Provide a separate eval dataset built from val split")
    p.add_argument("--eval-n-batches", type=int, default=4, help="How many eval batches to run when eval_on_val")

    p.add_argument("--base-url", default=os.environ.get("TINKER_BASE_URL"))
    p.add_argument("--system-prompt", default="")

    p.add_argument("--experiment-slug", default="sciscinet_hosted_rl")
    p.add_argument("--run-note", default="")
    p.add_argument("--run-name", default="")
    p.add_argument("--output-dir", type=Path, default=default_output_root)

    add_bool_arg(p, "--clean-logdir-before-run", default=False, help="Delete run_dir before starting")
    add_bool_arg(p, "--dry-run", default=False, help="Build manifest and exit without training")

    return p.parse_args()


def default_system_prompt(env_variant: str) -> str:
    if env_variant == "single_turn_disruption":
        return (
            "You are a scientific impact prediction model. "
            "Return exactly two lines. "
            "First line must be: disruption: <label>. "
            "Second line must be: reasoning: <short justification>. "
            "Do not output <think> or hidden reasoning."
        )
    return (
        "You are a scientific impact prediction model. "
        "Return exactly three lines. "
        "Line 1: disruption: <disruptive|consolidating|neutral>. "
        "Line 2: novelty: <novel|conventional|balanced>. "
        "Line 3: reasoning: <short justification>. "
        "Do not output <think> or hidden reasoning."
    )


# ----------------------------
# Runner
# ----------------------------

def _has_forbidden_marker(text: str) -> bool:
    t = (text or "").lower()
    return ("<think>" in t) or ("</think>" in t)


def _action_to_text(action: Any) -> str:
    """Best-effort conversion of the model 'action' into text.

    We intentionally avoid renderer parsing here because we want *raw* output.
    """

    if action is None:
        return ""
    if isinstance(action, str):
        return action
    if isinstance(action, dict):
        # common shapes: {'content': '...'}
        if "content" in action:
            try:
                return str(action.get("content") or "")
            except Exception:
                return ""
    return str(action)


async def run(args: argparse.Namespace, mini: Any) -> int:
    script_dir = Path(__file__).resolve().parent
    repo = mini.repo_root()

    load_status_script = load_env_file(script_dir / ".env", override=False)
    load_status_repo = load_env_file(repo / ".env", override=False)

    added_paths = bootstrap_paths(script_dir, repo)

    if not importlib.util.find_spec("tinker") or not importlib.util.find_spec("tinker_cookbook"):
        raise ImportError(
            "Missing `tinker` and/or `tinker_cookbook` in active environment. "
            f"python={sys.executable} added_paths={added_paths}"
        )

    import tinker
    from tinker_cookbook import model_info, renderers
    from tinker_cookbook.rl import train as cookbook_rl_train
    from tinker_cookbook.rl.types import Env as CookbookEnv
    from tinker_cookbook.rl.types import EnvGroupBuilder as CookbookEnvGroupBuilder
    from tinker_cookbook.rl.types import RLDataset as CookbookRLDataset
    from tinker_cookbook.rl.types import StepResult as CookbookStepResult
    from tinker_cookbook.rl.types import Trajectory
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    # Stable RNG for python-side sampling.
    random.seed(int(args.seed))

    catalog = mini.dataset_catalog()
    spec = catalog[args.dataset_key]

    data_root = Path(args.data_root).expanduser().resolve()
    dataset_jsonl, metadata_json, splits_json = mini.resolve_dataset_paths(data_root, spec)
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Missing dataset JSONL: {dataset_jsonl}")

    active_splits_json, active_splits_payload = mini.ensure_splits(
        dataset_jsonl=dataset_jsonl,
        explicit_splits=splits_json,
        split_mode=args.split_mode,
        seed=int(args.dataset_seed),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
    )

    train_quotas = {
        "consolidating": int(args.train_quota_consolidating),
        "disruptive": int(args.train_quota_disruptive),
        "neutral": int(args.train_quota_neutral),
    }
    val_quotas = {
        "consolidating": int(args.val_quota_consolidating),
        "disruptive": int(args.val_quota_disruptive),
        "neutral": int(args.val_quota_neutral),
    }

    train_records, val_records, test_records, data_info = mini.load_records(
        dataset_jsonl=dataset_jsonl,
        splits_payload=active_splits_payload,
        mode=args.loading_mode,
        seed=int(args.seed),
        train_split="train",
        val_split="val",
        test_split="test",
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        train_label_quotas=train_quotas,
        val_label_quotas=val_quotas,
        shuffle_selected=bool(args.shuffle_selected),
        working_set_per_label=args.working_set_per_label,
        working_set_total=args.working_set_total,
        split_ratios=(float(args.train_ratio), float(args.val_ratio), float(args.test_ratio)),
    )

    if not train_records:
        raise RuntimeError("No train records loaded after preflight")

    train_cov = mini.label_coverage_report(data_info["train_label_histogram"], mini.DISRUPTION_LABELS)
    val_cov = mini.label_coverage_report(data_info["val_label_histogram"], mini.DISRUPTION_LABELS)

    if args.sampling_strategy == "stratified":
        missing = [label for label, c in data_info["train_label_histogram"].items() if int(c) <= 0]
        if missing:
            raise RuntimeError(f"Cannot stratify with missing train labels: {missing}")

    renderer_name = (
        args.renderer_name.strip()
        if str(args.renderer_name).strip()
        else model_info.get_recommended_renderer_name(args.model_name)
    )

    system_prompt = args.system_prompt.strip() if str(args.system_prompt).strip() else default_system_prompt(args.env_variant)

    disruption_weights = {
        "consolidating": float(args.reward_disruption_consolidating),
        "disruptive": float(args.reward_disruption_disruptive),
        "neutral": float(args.reward_disruption_neutral),
    }
    novelty_weights = {"novel": 1.0, "conventional": 1.0, "balanced": 1.0}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if str(args.run_name).strip() else f"{args.experiment_slug}_{args.dataset_key}_{args.env_variant}_{ts}"
    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / run_name

    # ----------------------------
    # Env / Dataset definitions
    # ----------------------------

    @dataclass(frozen=True)
    class HostedEnvGroupBuilder(CookbookEnvGroupBuilder):
        record: Mapping[str, Any]
        group_size: int
        renderer: Any

        async def make_envs(self) -> Sequence[CookbookEnv]:
            return [HostedEnv(record=self.record, renderer=self.renderer) for _ in range(int(self.group_size))]

        async def compute_group_rewards(
            self,
            trajectory_group: list[Trajectory],
            env_group: Sequence[CookbookEnv],
        ) -> list[tuple[float, dict[str, float | int]]]:
            # We compute rewards in env.step(). Group-level reward shaping is intentionally off.
            _ = trajectory_group, env_group
            return [(0.0, {}) for _ in range(int(self.group_size))]

        def logging_tags(self) -> list[str]:
            tags = ["sciscinet", "disruption"]
            if args.env_variant != "single_turn_disruption":
                tags.append("novelty")
            if args.env_variant == "adversarial_two_turn_disruption_novelty":
                tags.append("adversarial_stub")
            return tags

    class HostedEnv(CookbookEnv):
        def __init__(self, *, record: Mapping[str, Any], renderer: Any) -> None:
            self.record = dict(record)
            self.renderer = renderer
            self._done = False
            self._prompt = ""

        def _stop_condition(self):
            return self.renderer.get_stop_sequences()

        async def initial_observation(self):
            self._prompt = mini.build_prompt(
                self.record,
                env_variant=args.env_variant,
                prompt_max_chars=int(args.prompt_max_chars),
                include_concepts=bool(args.include_concepts),
                include_definitions=bool(args.include_definitions),
            )
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": self._prompt})
            return self.renderer.build_generation_prompt(messages), self._stop_condition()

        async def step(self, action):
            if self._done:
                raise RuntimeError("Environment already finished")
            self._done = True

            raw_action_text = _action_to_text(action)
            raw_has_forbidden = _has_forbidden_marker(raw_action_text)

            message, parse_success = self.renderer.parse_response(action)
            try:
                content = renderers.get_text_content(message)
            except Exception:
                if isinstance(message, dict):
                    content = str(message.get("content", ""))
                else:
                    content = str(getattr(message, "content", ""))
            content = str(content or "")
            sanitized_has_forbidden = _has_forbidden_marker(content)

            # Which marker signal should be used for reward?
            detection = str(args.forbidden_marker_detection)
            if detection == "raw":
                has_forbidden_for_reward = raw_has_forbidden
            elif detection == "sanitized":
                has_forbidden_for_reward = sanitized_has_forbidden
            else:  # both
                has_forbidden_for_reward = raw_has_forbidden or sanitized_has_forbidden

            parsed = mini.parse_prediction_output(content, env_variant=args.env_variant)

            # Choose which format check counts as "format pass".
            format_check_mode = str(args.format_check)
            if format_check_mode == "keys":
                format_ok = bool(parsed.get("keys_ok"))
            else:
                format_ok = bool(parsed.get("strict_ok"))

            gold_disruption = mini.normalize_label(self.record.get("disruption_label"))
            raw_novelty = self.record.get("novelty_label")
            gold_novelty = mini.normalize_novelty_label(raw_novelty) if raw_novelty is not None else None
            if gold_novelty not in mini.NOVELTY_LABELS:
                gold_novelty = None

            reward = mini.compute_reward(
                env_variant=args.env_variant,
                pred_disruption=parsed.get("disruption_label"),
                pred_novelty=parsed.get("novelty_label"),
                gold_disruption=gold_disruption,
                gold_novelty=gold_novelty,
                reasoning=str(parsed.get("reasoning") or ""),
                format_ok=bool(format_ok),
                has_forbidden_marker=bool(has_forbidden_for_reward),
                strict_output_format=bool(args.strict_output_format),
                format_violation_penalty=float(args.format_violation_penalty),
                forbidden_marker_policy=str(args.forbidden_marker_policy),
                disruption_weights=disruption_weights,
                novelty_weights=novelty_weights,
                weight_disruption=float(args.weight_disruption),
                weight_novelty=float(args.weight_novelty),
                reasoning_bonus_mode=str(args.reasoning_bonus_mode),
                reasoning_bonus_weight=float(args.reasoning_bonus_weight),
            )

            # Extra metrics to enable confusion-matrix reconstruction from metrics.jsonl.
            gold_lbl = gold_disruption if gold_disruption in mini.DISRUPTION_LABELS else None
            pred_lbl = parsed.get("disruption_label") if parsed.get("disruption_label") in mini.DISRUPTION_LABELS else None
            cm_metrics = mini.disruption_confusion_indicators(gold_lbl, pred_lbl, prefix="cm")
            oh_metrics = mini.disruption_one_hot_labels(gold_lbl, pred_lbl)

            return CookbookStepResult(
                reward=float(reward["reward_total"]),
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self._stop_condition(),
                metrics={
                    "R_correctness": float(reward["R_correctness"]),
                    "R_correct_disruption": float(reward["R_correct_disruption"]),
                    "R_correct_novelty": float(reward["R_correct_novelty"]),
                    "R_reasoning": float(reward["R_reasoning"]),
                    "R_format": float(reward["R_format"]),
                    "R_adaptation": 0.0,
                    "parse_success": int(bool(parse_success)),
                    "label_parsed": int(parsed.get("disruption_label") is not None),
                    "novelty_label_parsed": int(parsed.get("novelty_label") is not None),
                    "label_correct": int(reward["disruption_correct"]),
                    "novelty_label_correct": int(reward["novelty_correct"]),
                    # Format diagnostics
                    "format_strict_ok": int(bool(parsed.get("strict_ok"))),
                    "format_keys_ok": int(bool(parsed.get("keys_ok"))),
                    "format_ok": int(bool(format_ok)),
                    "has_forbidden_output_marker": int(bool(has_forbidden_for_reward)),
                    "raw_has_forbidden_output_marker": int(bool(raw_has_forbidden)),
                    "sanitized_has_forbidden_output_marker": int(bool(sanitized_has_forbidden)),
                    **cm_metrics,
                    **oh_metrics,
                },
                logs={
                    "openalex_id": str(self.record.get("openalex_id", "")),
                    "env_variant": args.env_variant,
                    "gold_disruption": gold_disruption,
                    "pred_disruption": str(parsed.get("disruption_label") or ""),
                    "gold_novelty": str(gold_novelty or ""),
                    "pred_novelty": str(parsed.get("novelty_label") or ""),
                    "prompt_chars": len(self._prompt),
                    "response_chars": len(content),
                    "raw_response_chars": len(raw_action_text),
                },
            )

    class HostedDataset(CookbookRLDataset):
        def __init__(
            self,
            *,
            records: Sequence[Mapping[str, Any]],
            renderer: Any,
            groups_per_batch: int,
            group_size: int,
            n_batches: int,
            sampling_strategy: str,
            stratified_order: str,
            seed: int,
        ) -> None:
            self.records = [dict(r) for r in records]
            self.renderer = renderer
            self.groups_per_batch = int(groups_per_batch)
            self.group_size = int(group_size)
            self.n_batches = int(n_batches)
            self.sampling_strategy = str(sampling_strategy).strip().lower()
            self.stratified_order = str(stratified_order).strip().lower()
            self.seed = int(seed)
            self.rng = random.Random(int(seed))

            if not self.records:
                raise ValueError("records empty")
            if self.groups_per_batch <= 0 or self.group_size <= 0 or self.n_batches <= 0:
                raise ValueError("group and batch settings must be positive")

            # Natural stream (shuffled and looped).
            self.natural = [dict(r) for r in self.records]
            self.rng.shuffle(self.natural)
            self.npos = 0

            # Label buckets for stratified sampling.
            self.buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in mini.DISRUPTION_LABELS}
            for r in self.records:
                lbl = mini.normalize_label(r.get("disruption_label"))
                if lbl in self.buckets:
                    self.buckets[lbl].append(dict(r))
            self.bpos = {label: 0 for label in mini.DISRUPTION_LABELS}
            for label, bucket in self.buckets.items():
                if bucket:
                    self.rng.shuffle(bucket)

        def __len__(self) -> int:
            return self.n_batches

        def _next_natural(self) -> dict[str, Any]:
            if self.npos >= len(self.natural):
                self.npos = 0
                self.rng.shuffle(self.natural)
            r = dict(self.natural[self.npos])
            self.npos += 1
            return r

        def _next_label(self, label: str) -> dict[str, Any]:
            bucket = self.buckets.get(label) or []
            if not bucket:
                return self._next_natural()
            pos = self.bpos[label]
            if pos >= len(bucket):
                self.rng.shuffle(bucket)
                pos = 0
            self.bpos[label] = pos + 1
            return dict(bucket[pos])

        def _label_for_group(self, batch_index: int, group_index: int) -> str:
            labels = list(mini.DISRUPTION_LABELS)
            if self.stratified_order == "fixed":
                return labels[group_index % len(labels)]
            if self.stratified_order == "shuffle":
                # Deterministic shuffle per batch.
                r = random.Random(self.seed + 9973 * int(batch_index))
                r.shuffle(labels)
                return labels[group_index % len(labels)]
            # Default: rotate by batch index to avoid systematic logging bias.
            offset = int(batch_index) % len(labels)
            return labels[(group_index + offset) % len(labels)]

        def get_batch(self, index: int) -> Sequence[CookbookEnvGroupBuilder]:
            out: list[HostedEnvGroupBuilder] = []
            for i in range(self.groups_per_batch):
                if self.sampling_strategy == "stratified":
                    label = self._label_for_group(index, i)
                    rec = self._next_label(label)
                else:
                    rec = self._next_natural()
                out.append(
                    HostedEnvGroupBuilder(
                        record=rec,
                        group_size=self.group_size,
                        renderer=self.renderer,
                    )
                )
            return out

    @dataclass
    class HostedDatasetBuilder:
        model_name_for_tokenizer: str
        renderer_name: str
        train_records: Sequence[Mapping[str, Any]]
        val_records: Sequence[Mapping[str, Any]]

        async def __call__(self) -> tuple[HostedDataset, HostedDataset | None]:
            tokenizer = get_tokenizer(self.model_name_for_tokenizer)
            renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

            train_ds = HostedDataset(
                records=self.train_records,
                renderer=renderer,
                groups_per_batch=int(args.groups_per_batch),
                group_size=int(args.group_size),
                n_batches=int(args.n_batches),
                sampling_strategy=str(args.sampling_strategy),
                stratified_order=str(args.stratified_order),
                seed=int(args.seed),
            )

            eval_ds: HostedDataset | None = None
            if bool(args.eval_on_val) and self.val_records:
                eval_ds = HostedDataset(
                    records=self.val_records,
                    renderer=renderer,
                    groups_per_batch=int(args.groups_per_batch),
                    group_size=1,  # eval doesn't need multiple rollouts per prompt
                    n_batches=max(1, int(args.eval_n_batches)),
                    sampling_strategy=str(args.sampling_strategy),
                    stratified_order=str(args.stratified_order),
                    seed=int(args.seed) + 4242,
                )
            return train_ds, eval_ds

    dataset_builder = HostedDatasetBuilder(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        train_records=train_records,
        val_records=val_records,
    )

    cfg = cookbook_rl_train.Config(
        learning_rate=float(args.learning_rate),
        dataset_builder=dataset_builder,
        model_name=args.model_name,
        lora_rank=int(args.lora_rank),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        log_path=str(run_dir),
        eval_every=int(args.eval_every),
        save_every=int(args.save_every),
        base_url=args.base_url,
        num_groups_to_log=int(args.num_groups_to_log),
    )

    manifest = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "git_commit": try_git_commit(repo),
        "repo_root": str(repo),
        "env_load": {"script_dir": load_status_script, "repo_root": load_status_repo},
        "sys_path_added": added_paths,
        "dataset_key": args.dataset_key,
        "dataset_description": spec.description,
        "data_root": str(data_root),
        "dataset_jsonl": str(dataset_jsonl),
        "metadata_json": str(metadata_json) if metadata_json else None,
        "splits_json": str(splits_json) if splits_json else None,
        "active_splits_json": str(active_splits_json) if active_splits_json is not None else None,
        "split_mode": args.split_mode,
        "loading_mode": args.loading_mode,
        "seed": int(args.seed),
        "dataset_seed": int(args.dataset_seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(args.test_ratio),
        "max_train": args.max_train,
        "max_val": args.max_val,
        "max_test": args.max_test,
        "train_label_quotas": train_quotas,
        "val_label_quotas": val_quotas,
        "working_set_per_label": args.working_set_per_label,
        "working_set_total": args.working_set_total,
        "data_info": data_info,
        "train_coverage": train_cov,
        "val_coverage": val_cov,
        "env_variant": args.env_variant,
        "prompt_max_chars": int(args.prompt_max_chars),
        "include_concepts": bool(args.include_concepts),
        "include_definitions": bool(args.include_definitions),
        "reward": {
            "weight_disruption": float(args.weight_disruption),
            "weight_novelty": float(args.weight_novelty),
            "disruption_label_weights": disruption_weights,
            "format_check": str(args.format_check),
            "strict_output_format": bool(args.strict_output_format),
            "format_violation_penalty": float(args.format_violation_penalty),
            "forbidden_marker_policy": str(args.forbidden_marker_policy),
            "forbidden_marker_detection": str(args.forbidden_marker_detection),
            "reasoning_bonus_mode": str(args.reasoning_bonus_mode),
            "reasoning_bonus_weight": float(args.reasoning_bonus_weight),
        },
        "runner_config": {
            "model_name": args.model_name,
            "renderer_name": renderer_name,
            "learning_rate": float(args.learning_rate),
            "lora_rank": int(args.lora_rank),
            "max_tokens": int(args.max_tokens),
            "temperature": float(args.temperature),
            "groups_per_batch": int(args.groups_per_batch),
            "group_size": int(args.group_size),
            "n_batches": int(args.n_batches),
            "sampling_strategy": str(args.sampling_strategy),
            "stratified_order": str(args.stratified_order),
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "num_groups_to_log": int(args.num_groups_to_log),
            "eval_on_val": bool(args.eval_on_val),
            "eval_n_batches": int(args.eval_n_batches),
            "base_url": args.base_url,
            "run_name": run_name,
            "run_note": args.run_note,
            "experiment_slug": args.experiment_slug,
            "dry_run": bool(args.dry_run),
        },
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = mini.sha256_text(
        mini.stable_json({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    )

    if bool(args.clean_logdir_before_run) and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print("dataset_key:", args.dataset_key)
    print("dataset_jsonl:", dataset_jsonl)
    print("active_splits_json:", active_splits_json)
    print("loading_mode:", args.loading_mode)
    print("env_variant:", args.env_variant)
    print("train/val/test:", data_info["loaded_train_records"], data_info["loaded_val_records"], data_info["loaded_test_records"])
    print("train_hist:", data_info["train_label_histogram"])
    print("val_hist:", data_info["val_label_histogram"])
    print("run_dir:", run_dir)
    print("manifest:", manifest_path)

    if bool(args.dry_run):
        print("dry_run=True -> skipping hosted training launch")
        return 0

    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("TINKER_API_KEY is not set; cannot launch hosted training")

    await cookbook_rl_train.main(cfg)
    return 0


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    mini = importlib.import_module("sciscinet_cookbook_mini")

    args = parse_args(
        dataset_choices=tuple(mini.dataset_catalog().keys()),
        default_data_root=mini.default_data_root(),
        default_output_root=mini.repo_root() / "results" / "tinker_rl_cookbook",
    )

    rc = asyncio.run(run(args, mini))
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
