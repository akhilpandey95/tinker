from __future__ import annotations

# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from tinker_cookbook import renderers
from tinker_cookbook.rl import train as cookbook_rl_train
from tinker_cookbook.rl.types import Env as CookbookEnv
from tinker_cookbook.rl.types import StepResult as CookbookStepResult

from data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET_PATH,
    DEFAULT_GROUP_SIZE,
    DEFAULT_RENDERER_NAME,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DisruptionDatasetBuilder,
    LABELS,
    LLM,
    PaperRecord,
)

DEFAULT_LOG_ROOT = Path(
    os.environ.get("TINKER_RL_LOG_ROOT", "/tmp/tinker-disruption-rl-runs")
)
LAST_RUN_LOG_PATH: Path | None = None
STRICT_CONFIDENCE_VALUES = {"low", "medium", "high"}
# Exp2 reward shaping:
# keep contract pressure, but move most of the signal onto label correctness.
RAW_JSON_FAIL_PENALTY = 1.0
RAW_THINK_PENALTY = 1.0
PAYLOAD_VALID_BONUS = 0.05
PAYLOAD_INVALID_PENALTY = 0.75
TASK_CORRECT_REWARD = 1.0
TASK_INCORRECT_PENALTY = 0.6
DEFAULT_MAX_TOKENS = 64
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


class DisruptionJSONContract:
    @staticmethod
    def build_messages(record: PaperRecord) -> list[renderers.Message]:
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

    @staticmethod
    def normalize_completion(completion: str) -> str:
        return completion.strip().replace("<|endoftext|>", "").strip()

    @staticmethod
    def clean_completion(completion: str) -> str:
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
    def get_raw_text_content(message: renderers.Message) -> str:
        content = message["content"]
        if isinstance(content, str):
            return content

        chunks: list[str] = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                chunks.append(part["text"])
            elif part_type == "thinking":
                chunks.append(f"<think>{part['thinking']}</think>")

        return "".join(chunks)

    @staticmethod
    def validate_payload(payload: object) -> tuple[str, str | None]:
        if not isinstance(payload, dict):
            return "not_json_object", None

        if set(payload) != {"disruption_label", "confidence"}:
            return "bad_keys", None

        label = payload.get("disruption_label")
        confidence = payload.get("confidence")

        if not isinstance(label, str) or label.lower() not in LABELS:
            return "bad_label", None

        if not isinstance(confidence, str) or confidence.lower() not in STRICT_CONFIDENCE_VALUES:
            return "bad_confidence", None

        return "ok", label.lower()

    @staticmethod
    def parse_json_text(text: str) -> tuple[str, str | None, bool]:
        text = DisruptionJSONContract.normalize_completion(text)
        if not text:
            return "empty", None, False

        if not text.startswith("{") or not text.endswith("}"):
            return "not_json_object", None, False

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json", None, False

        status, label = DisruptionJSONContract.validate_payload(payload)
        return status, label, True

    @staticmethod
    def analyze_raw_completion(completion: str) -> dict[str, object]:
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

        analysis = {
            "text": raw_text,
            "status": "not_json_object",
            "label": None,
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

        status, label, raw_json_loads = DisruptionJSONContract.parse_json_text(raw_text)
        analysis["status"] = status
        analysis["label"] = label
        analysis["raw_json_loads"] = raw_json_loads
        analysis["payload_valid"] = status == "ok"
        return analysis

    @staticmethod
    def analyze_clean_completion(completion: str) -> dict[str, object]:
        clean_text = DisruptionJSONContract.clean_completion(completion)
        status, label, json_loads = DisruptionJSONContract.parse_json_text(clean_text)
        return {
            "text": clean_text,
            "status": status,
            "label": label,
            "json_loads": json_loads,
            "payload_valid": status == "ok",
        }

    @staticmethod
    def score(record: PaperRecord, completion: str) -> dict[str, object]:
        raw = DisruptionJSONContract.analyze_raw_completion(completion)
        clean = DisruptionJSONContract.analyze_clean_completion(completion)

        gold_label = record.gold_label.lower() if record.gold_label else None
        label_correct = bool(raw["payload_valid"] and gold_label and raw["label"] == gold_label)

        r_parse = 0.0 if raw["raw_json_loads"] else -RAW_JSON_FAIL_PENALTY
        if raw["payload_valid"]:
            r_contract = PAYLOAD_VALID_BONUS
        elif raw["raw_json_loads"]:
            r_contract = -PAYLOAD_INVALID_PENALTY
        else:
            r_contract = 0.0
        r_think = -RAW_THINK_PENALTY if raw["raw_has_think"] else 0.0

        if gold_label is None or not raw["payload_valid"]:
            r_task = 0.0
        else:
            r_task = TASK_CORRECT_REWARD if label_correct else -TASK_INCORRECT_PENALTY

        reward = float(r_parse + r_contract + r_think + r_task)
        return {
            "reward": reward,
            "label": raw["label"],
            "label_correct": label_correct,
            "raw": raw,
            "clean": clean,
            "components": {
                "R_parse": float(r_parse),
                "R_contract": float(r_contract),
                "R_think": float(r_think),
                "R_task": float(r_task),
            },
        }


class DisruptionRLEnv(CookbookEnv):
    def __init__(self, record: PaperRecord, renderer: renderers.Renderer) -> None:
        self.record = record
        self.renderer = renderer
        self.messages = DisruptionJSONContract.build_messages(record)

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    def _observation(self):
        return self.renderer.build_generation_prompt(self.messages)

    async def initial_observation(self):
        return self._observation(), self.stop_condition

    async def step(self, action):
        action_message, parse_success = self.renderer.parse_response(action)
        raw_content = DisruptionJSONContract.get_raw_text_content(action_message)
        scored = DisruptionJSONContract.score(self.record, raw_content)
        raw = scored["raw"]
        clean = scored["clean"]
        label = scored["label"]
        components = scored["components"]

        label_match = int(
            bool(scored["label_correct"])
        )

        return CookbookStepResult(
            reward=scored["reward"],
            episode_done=True,
            next_observation=self._observation(),
            next_stop_condition=self.stop_condition,
            metrics={
                "render_parse_success": int(bool(parse_success)),
                "parse_ok": int(raw["raw_json_loads"]),
                "format_strict_ok": int(raw["status"] == "ok"),
                "payload_valid": int(raw["payload_valid"]),
                "raw_payload_valid": int(raw["payload_valid"]),
                "label_match": label_match,
                "label_correct_given_valid_payload": label_match,
                "raw_exact_json_only": int(raw["raw_exact_json_only"]),
                "raw_json_loads": int(raw["raw_json_loads"]),
                "raw_has_fence": int(raw["raw_has_fence"]),
                "raw_has_think": int(raw["raw_has_think"]),
                "had_think_tags": int(raw["raw_has_think"]),
                "has_forbidden_output_marker": int(raw["raw_has_think"]),
                "clean_parse_ok": int(clean["json_loads"]),
                "clean_payload_valid": int(clean["payload_valid"]),
                "reward_parse": components["R_parse"],
                "reward_contract": components["R_contract"],
                "reward_think": components["R_think"],
                "reward_task": components["R_task"],
                "reward_total": scored["reward"],
            },
            logs={
                "raw_status": str(raw["status"]),
                "clean_status": str(clean["status"]),
                "gold_label": self.record.gold_label or "",
                "pred_label": label or "",
                "title": self.record.title[:120],
            },
        )


def make_log_path(prefix: str = "disruption-rl") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOG_ROOT / f"{prefix}_{stamp}"


def build_disruption_rl_config(
    *,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    log_path: str | Path | None = None,
    model_name: str = LLM,
    renderer_name: str = DEFAULT_RENDERER_NAME,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    group_size: int = DEFAULT_GROUP_SIZE,
    learning_rate: float = 8e-6,
    temperature: float = 0.3,
    max_tokens: int = DEFAULT_MAX_TOKENS,
):
    resolved_log_path = Path(log_path) if log_path is not None else make_log_path()
    builder = DisruptionDatasetBuilder(
        dataset_path=str(Path(dataset_path)),
        batch_size=batch_size,
        group_size=group_size,
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        train_size=train_size,
        test_size=test_size,
    )

    return cookbook_rl_train.Config(
        learning_rate=learning_rate,
        dataset_builder=builder,
        model_name=model_name,
        max_tokens=max_tokens,
        log_path=str(resolved_log_path),
        eval_every=250,
        save_every=250,
        loss_fn="importance_sampling",
        num_substeps=1,
        lora_rank=32,
        temperature=temperature,
    )


async def preview_disruption_rl(cfg=None) -> dict[str, object]:
    cfg = cfg or build_disruption_rl_config()
    train_rl_ds, test_rl_ds = await cfg.dataset_builder()
    first_group = train_rl_ds.get_batch(0)[0]
    first_env = (await first_group.make_envs())[0]

    return {
        "train_batches": len(train_rl_ds),
        "test_batches": len(test_rl_ds),
        "group_size": train_rl_ds.group_size,
        "gold_label": first_env.record.gold_label,
        "model_name": cfg.model_name,
        "renderer_name": cfg.dataset_builder.renderer_name,
        "stop_condition": first_env.stop_condition,
        "prompt_preview": first_env.messages[-1]["content"][:300],
        "log_path": cfg.log_path,
    }


def require_api_key() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise EnvironmentError("Set TINKER_API_KEY before running RL training.")


async def run_disruption_rl(cfg=None) -> Path:
    global LAST_RUN_LOG_PATH

    require_api_key()
    cfg = cfg or build_disruption_rl_config()
    LAST_RUN_LOG_PATH = Path(cfg.log_path)
    await cookbook_rl_train.main(cfg)
    return LAST_RUN_LOG_PATH


def plot_rl_metrics(metrics_path: str | Path | None = None, *, show: bool = False):
    if metrics_path is None:
        if LAST_RUN_LOG_PATH is None:
            raise ValueError("Pass metrics_path or call run_disruption_rl first.")
        metrics_path = LAST_RUN_LOG_PATH / "metrics.jsonl"

    df = pl.read_ndjson(Path(metrics_path))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    reward_candidates = ["env/all/reward/total", "env/all/reward_total"]
    kl_candidates = ["optim/kl_sample_train_v1", "kl_sample_train_v1"]
    reward_col = next((col for col in reward_candidates if col in df.columns), None)
    kl_col = next((col for col in kl_candidates if col in df.columns), None)

    if reward_col is not None:
        axes[0].plot(df[reward_col].to_list(), marker="o", linewidth=1.5)
        axes[0].set_title("Reward Total")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel(reward_col)
    else:
        axes[0].set_title("Reward Total")
        axes[0].text(0.5, 0.5, "missing metric", ha="center", va="center")

    if kl_col is not None:
        axes[1].plot(
            df[kl_col].to_list(),
            marker="o",
            linewidth=1.5,
            color="tab:orange",
        )
        axes[1].set_title("KL Sample Train v1")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel(kl_col)
    else:
        axes[1].set_title("KL Sample Train v1")
        axes[1].text(0.5, 0.5, "missing metric", ha="center", va="center")

    plt.tight_layout()
    if show:
        plt.show()
    return fig
