#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""Mini CLI + library utilities for the SciSciNet RL cookbook workflow.

This file is intentionally "batteries included": the hosted RL runner imports
it as a library, and the CLI (`python sciscinet_cookbook_mini.py ...`) is used
as a deterministic preflight / dataset sanity tool.

Refactor goals (v2):
- Fix boolean CLI flags that were previously impossible to disable.
- Make format checks configurable ("exact line count" vs "keys present").
- Make reasoning bonus configurable / optionally conditional.
- Add helper metrics to reconstruct confusion matrices directly from metrics.jsonl
  (so we don't depend on HTML logs for calibration plots).
- Improve RL-balanced stream sampling split to be label-stratified.

Nothing in this file should depend on `tinker` / hosted training; it should stay
pure-Python and runnable anywhere you have the dataset JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

# ----------------------------
# Constants / schema
# ----------------------------

DISRUPTION_LABELS = ("disruptive", "consolidating", "neutral")
NOVELTY_LABELS = ("novel", "conventional", "balanced")
ENV_VARIANTS = (
    "single_turn_disruption",
    "single_turn_disruption_novelty",
    "adversarial_two_turn_disruption_novelty",
)

REASONING_BONUS_MODES = (
    "always",            # always add reasoning_bonus()
    "only_if_correct",   # only if disruption label is correct (and novelty if applicable)
    "only_if_strict_ok", # only if chosen format check passes
    "disabled",          # always 0.0
)

FORBIDDEN_MARKER_POLICIES = (
    "penalize",  # forbidden marker counts as a format violation
    "ignore",    # forbidden marker does not affect format reward
)

FORMAT_CHECK_MODES = (
    "exact",  # exact non-empty line count + keys
    "keys",   # keys present + labels valid (ignores extra lines)
)

FORBIDDEN_PROMPT_MARKERS = (
    "CD Index:",
    "Novelty Score:",
    "Conventionality Score:",
    "cd_index",
    "novelty_score",
    "conventionality_score",
)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    jsonl: str
    metadata: str | None
    splits: str | None
    description: str
    has_official_splits: bool
    is_rl_balanced: bool
    expected_disruption_counts: dict[str, int] | None
    thresholds: dict[str, float] | None


# ----------------------------
# Paths / dataset catalog
# ----------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    """Best-effort default data root.

    Tries ../tinker_smolrl_data first (typical layout), then ./tinker_smolrl_data.
    """

    r = repo_root()
    c1 = (r.parent / "tinker_smolrl_data").resolve()
    c2 = (r / "tinker_smolrl_data").resolve()
    if c1.exists():
        return c1
    return c2


def dataset_catalog() -> dict[str, DatasetSpec]:
    """Hard-coded dataset variants known to this cookbook."""

    return {
        "baseline_500k": DatasetSpec(
            key="baseline_500k",
            jsonl="disruption_novelty_sciscinet_500k.jsonl",
            metadata="disruption_novelty_sciscinet_500k.metadata.json",
            splits="disruption_novelty_sciscinet_500k.splits.json",
            description="Root baseline (imbalanced).",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 739, "disruptive": 2407, "neutral": 488674},
            thresholds={"consolidating_max": -0.1, "disruptive_min": 0.1, "novelty_margin": 0.15},
        ),
        "sci_balanced_500k_t1": DatasetSpec(
            key="sci_balanced_500k_t1",
            jsonl="sci_balanced_500k_t1.jsonl",
            metadata="sci_balanced_500k_t1.metadata.json",
            splits="sci_balanced_500k_t1.splits.json",
            description="500k t1 cohort.",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 126374, "disruptive": 45237, "neutral": 320209},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_500k_t1_rl_balanced": DatasetSpec(
            key="sci_balanced_500k_t1_rl_balanced",
            jsonl="sci_balanced_500k_t1.rl_balanced.jsonl",
            metadata="sci_balanced_500k_t1.metadata.json",
            splits=None,
            description="500k t1 RL-balanced subset.",
            has_official_splits=False,
            is_rl_balanced=True,
            expected_disruption_counts={"consolidating": 100000, "disruptive": 100000, "neutral": 100000},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_2m_t1": DatasetSpec(
            key="sci_balanced_2m_t1",
            jsonl="sci_balanced_2m_t1.jsonl",
            metadata="sci_balanced_2m_t1.metadata.json",
            splits="sci_balanced_2m_t1.splits.json",
            description="2M t1 cohort.",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 583215, "disruptive": 247090, "neutral": 1142492},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_2m_t1_rl_balanced": DatasetSpec(
            key="sci_balanced_2m_t1_rl_balanced",
            jsonl="sci_balanced_2m_t1.rl_balanced.jsonl",
            metadata="sci_balanced_2m_t1.metadata.json",
            splits=None,
            description="2M t1 RL-balanced subset.",
            has_official_splits=False,
            is_rl_balanced=True,
            expected_disruption_counts={"consolidating": 100000, "disruptive": 100000, "neutral": 100000},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_from2m_no_ovr": DatasetSpec(
            key="sci_balanced_from2m_no_ovr",
            jsonl="sci_balanced_from2m_no_ovr.jsonl",
            metadata="sci_balanced_from2m_no_ovr.metadata.json",
            splits="sci_balanced_from2m_no_ovr.splits.json",
            description="2M no-oversample source.",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 583215, "disruptive": 247090, "neutral": 1142492},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_from2m_no_ovr_rl_balanced": DatasetSpec(
            key="sci_balanced_from2m_no_ovr_rl_balanced",
            jsonl="sci_balanced_from2m_no_ovr.rl_balanced.jsonl",
            metadata="sci_balanced_from2m_no_ovr.metadata.json",
            splits=None,
            description="2M no-oversample RL-balanced subset.",
            has_official_splits=False,
            is_rl_balanced=True,
            expected_disruption_counts={"consolidating": 200000, "disruptive": 200000, "neutral": 200000},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "test_auto_expand_600k": DatasetSpec(
            key="test_auto_expand_600k",
            jsonl="test_auto_expand_600k.jsonl",
            metadata="test_auto_expand_600k.metadata.json",
            splits="test_auto_expand_600k.splits.json",
            description="Auto-expand source cohort.",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 583215, "disruptive": 247090, "neutral": 1142492},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "test_auto_expand_600k_rl_balanced": DatasetSpec(
            key="test_auto_expand_600k_rl_balanced",
            jsonl="test_auto_expand_600k.rl_balanced.jsonl",
            metadata="test_auto_expand_600k.metadata.json",
            splits=None,
            description="Auto-expand RL-balanced subset.",
            has_official_splits=False,
            is_rl_balanced=True,
            expected_disruption_counts={"consolidating": 200000, "disruptive": 200000, "neutral": 200000},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_from3m_no_ovr": DatasetSpec(
            key="sci_balanced_from3m_no_ovr",
            jsonl="sci_balanced_from3m_no_ovr.jsonl",
            metadata="sci_balanced_from3m_no_ovr.metadata.json",
            splits="sci_balanced_from3m_no_ovr.splits.json",
            description="3M no-oversample source.",
            has_official_splits=True,
            is_rl_balanced=False,
            expected_disruption_counts={"consolidating": 892777, "disruptive": 402418, "neutral": 1663793},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
        "sci_balanced_from3m_no_ovr_rl_balanced": DatasetSpec(
            key="sci_balanced_from3m_no_ovr_rl_balanced",
            jsonl="sci_balanced_from3m_no_ovr.rl_balanced.jsonl",
            metadata="sci_balanced_from3m_no_ovr.metadata.json",
            splits=None,
            description="3M no-oversample RL-balanced subset.",
            has_official_splits=False,
            is_rl_balanced=True,
            expected_disruption_counts={"consolidating": 300000, "disruptive": 300000, "neutral": 300000},
            thresholds={"consolidating_max": -0.001, "disruptive_min": 0.001, "novelty_margin": 0.15},
        ),
    }


# ----------------------------
# Small utilities
# ----------------------------

def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[str]) -> str:
    return sha256_text("\n".join(sorted(str(x) for x in ids)))


def file_size_mib(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024**2), 3)


def norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def record_id(record: Mapping[str, Any], fallback_idx: int) -> str:
    rid = record.get("openalex_id") or record.get("paperid") or record.get("id")
    return str(rid) if rid is not None and str(rid).strip() else f"row_{fallback_idx}"


def normalize_label(value: Any) -> str:
    v = norm_text(value).lower()
    if v in {"disruptive", "disruption", "d"}:
        return "disruptive"
    if v in {"consolidating", "consolidation", "c"}:
        return "consolidating"
    if v in {"neutral", "n"}:
        return "neutral"
    return v


def normalize_novelty_label(value: Any) -> str:
    v = norm_text(value).lower()
    if v in {"novel", "n"}:
        return "novel"
    if v in {"conventional", "conv", "c"}:
        return "conventional"
    if v in {"balanced", "balance", "b"}:
        return "balanced"
    return v


def resolve_dataset_paths(data_root: Path, spec: DatasetSpec) -> tuple[Path, Path | None, Path | None]:
    return (
        (data_root / spec.jsonl).resolve(),
        (data_root / spec.metadata).resolve() if spec.metadata else None,
        (data_root / spec.splits).resolve() if spec.splits else None,
    )


def iter_jsonl_records(path: Path, max_rows: int | None = None):
    with path.open("r", encoding="utf-8") as handle:
        seen = 0
        for line_no, raw in enumerate(handle, start=1):
            s = raw.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            seen += 1
            yield seen, rec
            if max_rows is not None and seen >= max_rows:
                break


def infer_splits_path(dataset_jsonl: Path) -> Path | None:
    candidates = [dataset_jsonl.with_suffix(".splits.json")]
    s = str(dataset_jsonl)
    if s.endswith(".rl_balanced.jsonl"):
        base = Path(s[: -len(".rl_balanced.jsonl")])
        candidates.append(Path(str(base) + ".splits.json"))
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def scan_dataset_stats(path: Path, max_rows: int | None) -> dict[str, Any]:
    disruption_hist = Counter()
    novelty_hist = Counter()
    missing = Counter()
    rows = 0
    for i, rec in iter_jsonl_records(path, max_rows=max_rows):
        rows = i
        disruption_hist[normalize_label(rec.get("disruption_label")) or "<missing>"] += 1
        if rec.get("novelty_label") is not None:
            novelty_hist[normalize_novelty_label(rec.get("novelty_label")) or "<missing>"] += 1
        for field in ("title", "abstract", "publication_year", "cited_by_count", "primary_field"):
            if rec.get(field) in (None, "", []):
                missing[field] += 1
    missing_pct = {k: (100.0 * v / rows if rows else 0.0) for k, v in missing.items()}
    return {
        "rows_scanned": rows,
        "disruption_hist": dict(disruption_hist),
        "novelty_hist": dict(novelty_hist),
        "missing_pct": missing_pct,
    }


# ----------------------------
# Splits
# ----------------------------

def ensure_splits(
    *,
    dataset_jsonl: Path,
    explicit_splits: Path | None,
    split_mode: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Return (splits_path, splits_payload) or (None, None) for train_only."""

    mode = str(split_mode)
    official = explicit_splits if explicit_splits and explicit_splits.exists() else infer_splits_path(dataset_jsonl)

    if mode == "train_only":
        return None, None

    if mode == "use_official_or_generate" and official is not None:
        payload = json.loads(official.read_text(encoding="utf-8"))
        return official, payload

    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    s = sum(ratios)
    if s <= 0:
        raise ValueError("Split ratios must sum to > 0")
    ratios = [x / s for x in ratios]

    ids: list[str] = []
    seen: set[str] = set()
    for i, rec in iter_jsonl_records(dataset_jsonl, max_rows=None):
        rid = record_id(rec, i)
        if rid in seen:
            continue
        seen.add(rid)
        ids.append(rid)

    rng = random.Random(int(seed))
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    ids_train = ids[:n_train]
    ids_val = ids[n_train : n_train + n_val]
    ids_test = ids[n_train + n_val :]

    payload = {
        "counts": {"train": len(ids_train), "val": len(ids_val), "test": len(ids_test)},
        "ids": {"train": ids_train, "val": ids_val, "test": ids_test},
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "split_seed": int(seed),
        "split_ids_sha256": sha256_ids(ids),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_jsonl": str(dataset_jsonl),
        "mode": "generated",
    }

    out = Path(str(dataset_jsonl).replace(".jsonl", ".splits.generated.json"))
    out.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return out, payload


# ----------------------------
# Prompt building
# ----------------------------

def truncate_with_ellipsis(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def truncate_title_abstract(title: str, abstract: str, max_total_chars: int) -> tuple[str, str]:
    t = norm_text(title)
    a = norm_text(abstract)
    if max_total_chars <= 0:
        return "", ""
    if len(t) + len(a) <= max_total_chars:
        return t, a
    if len(t) >= max_total_chars:
        return truncate_with_ellipsis(t, max_total_chars), ""
    budget = max_total_chars - len(t)
    return t, truncate_with_ellipsis(a, budget)


def format_concepts(concepts: Any) -> str | None:
    if concepts is None:
        return None
    if isinstance(concepts, (str, bytes, bytearray)):
        raw_items = [concepts]
    elif isinstance(concepts, Sequence):
        raw_items = list(concepts)
    else:
        raw_items = [concepts]

    vals: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        v = norm_text(item)
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        vals.append(v)
    return ", ".join(vals[:8]) if vals else None


def label_definitions(env_variant: str) -> list[str]:
    """Short, non-technical label definitions to reduce ambiguity."""

    lines = [
        "Label definitions (use your best judgment):",
        "- disruptive: introduces a new approach or reframes the problem in a way that shifts future work.",
        "- consolidating: synthesizes, refines, or extends an existing paradigm without a major shift.",
        "- neutral: incremental or mixed; not a clear shift and not a clear synthesis.",
    ]
    if env_variant != "single_turn_disruption":
        lines.extend(
            [
                "Novelty definitions:",
                "- novel: unusual combination or new conceptual direction.",
                "- conventional: follows common patterns in the field.",
                "- balanced: in-between; some novelty but mostly conventional.",
            ]
        )
    return lines


def build_prompt(
    record: Mapping[str, Any],
    *,
    env_variant: str,
    prompt_max_chars: int,
    include_concepts: bool,
    include_definitions: bool = False,
) -> str:
    """Build a leak-checked prompt from a record."""

    title, abstract = truncate_title_abstract(
        str(record.get("title", "")),
        str(record.get("abstract", "")),
        int(prompt_max_chars),
    )
    field = norm_text(record.get("primary_field", "Unknown")) or "Unknown"

    if env_variant == "single_turn_disruption":
        task = [
            "Predict the disruption label for the paper.",
            "Allowed labels: disruptive, consolidating, neutral.",
            "Return exactly:",
            "disruption: <label>",
            "reasoning: <short justification>",
        ]
    else:
        task = [
            "Predict both disruption and novelty labels for the paper.",
            "Allowed disruption labels: disruptive, consolidating, neutral.",
            "Allowed novelty labels: novel, conventional, balanced.",
            "Return exactly:",
            "disruption: <label>",
            "novelty: <label>",
            "reasoning: <short justification>",
        ]

    lines: list[str] = [*task]
    if include_definitions:
        lines.extend(["", *label_definitions(env_variant)])

    lines.extend(
        [
            "",
            f"Title: {title}",
            f"Abstract: {abstract}",
            f"Year: {int(record.get('publication_year', 0) or 0)}",
            f"Citations: {int(record.get('cited_by_count', 0) or 0)}",
            f"Field: {field}",
        ]
    )

    if include_concepts:
        concepts_text = format_concepts(record.get("concepts"))
        if concepts_text:
            lines.append(f"Concepts: {concepts_text}")

    prompt = "\n".join(lines)
    leak = [m for m in FORBIDDEN_PROMPT_MARKERS if m in prompt]
    if leak:
        raise ValueError(f"Prompt leakage markers detected: {leak}")
    return prompt


# ----------------------------
# Output parsing
# ----------------------------

def extract_line_value(text: str, key: str) -> str | None:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"
    m = re.search(pattern, str(text or ""))
    return norm_text(m.group(1)) if m else None


def parse_prediction_output(text: str, env_variant: str) -> dict[str, Any]:
    """Parse model output.

    Returns both:
      - strict_ok: requires exact line count + valid labels + non-empty reasoning.
      - keys_ok: requires required keys/labels, ignores extra non-empty lines.

    The runner can choose which one counts as a "format pass".
    """

    raw = str(text or "")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    disruption = extract_line_value(raw, "disruption")
    novelty = extract_line_value(raw, "novelty")
    reasoning = extract_line_value(raw, "reasoning") or ""

    disruption_norm = normalize_label(disruption) if disruption else None
    novelty_norm = normalize_novelty_label(novelty) if novelty else None

    if env_variant == "single_turn_disruption":
        keys_ok = (disruption_norm in DISRUPTION_LABELS) and bool(reasoning) and (novelty_norm is None)
        strict_ok = keys_ok and (len(lines) == 2)
    else:
        keys_ok = (disruption_norm in DISRUPTION_LABELS) and (novelty_norm in NOVELTY_LABELS) and bool(reasoning)
        strict_ok = keys_ok and (len(lines) == 3)

    return {
        "raw": raw,
        "disruption_label": disruption_norm,
        "novelty_label": novelty_norm,
        "reasoning": reasoning,
        "strict_ok": bool(strict_ok),
        "keys_ok": bool(keys_ok),
        "line_count": len(lines),
    }


# ----------------------------
# Reward
# ----------------------------

def reasoning_bonus(reasoning: str) -> float:
    """Small shaping signal based on reasoning length only."""

    n = len(norm_text(reasoning))
    if n == 0:
        return -0.05
    if n < 24:
        return 0.0
    if n < 80:
        return 0.1
    return 0.2


def score_classification(
    pred: str | None,
    gold: str | None,
    weights: Mapping[str, float],
    default: float = 1.0,
) -> tuple[float, int]:
    if not gold:
        return 0.0, 0
    w = float(weights.get(gold, default))
    ok = int(pred is not None and pred == gold)
    return (w if ok else -w), ok


def compute_reward(
    *,
    env_variant: str,
    pred_disruption: str | None,
    pred_novelty: str | None,
    gold_disruption: str | None,
    gold_novelty: str | None,
    reasoning: str,
    # format-related
    format_ok: bool,
    has_forbidden_marker: bool,
    strict_output_format: bool,
    format_violation_penalty: float,
    forbidden_marker_policy: str = "penalize",
    # correctness weights
    disruption_weights: Mapping[str, float],
    novelty_weights: Mapping[str, float],
    weight_disruption: float,
    weight_novelty: float,
    # reasoning shaping
    reasoning_bonus_mode: str = "always",
    reasoning_bonus_weight: float = 1.0,
) -> dict[str, float | int]:
    """Compute total reward + a diagnostic decomposition.

    Notes:
    - `format_ok` is provided by the caller and can mean either strict_ok or keys_ok.
    - `has_forbidden_marker` should be measured on the *raw* model output if you care about
      forbidding `<think>` in what the model actually emits.
    """

    if reasoning_bonus_mode not in REASONING_BONUS_MODES:
        raise ValueError(f"Unknown reasoning_bonus_mode={reasoning_bonus_mode}")
    if forbidden_marker_policy not in FORBIDDEN_MARKER_POLICIES:
        raise ValueError(f"Unknown forbidden_marker_policy={forbidden_marker_policy}")

    r_dis, ok_dis = score_classification(pred_disruption, gold_disruption, disruption_weights)
    r_nov, ok_nov = score_classification(pred_novelty, gold_novelty, novelty_weights)

    r_correctness = float(weight_disruption) * float(r_dis)
    if env_variant != "single_turn_disruption" and gold_novelty:
        r_correctness += float(weight_novelty) * float(r_nov)

    # Reasoning shaping: optionally conditional.
    base_reason = float(reasoning_bonus(reasoning)) * float(reasoning_bonus_weight)
    if reasoning_bonus_mode == "disabled":
        r_reason = 0.0
    elif reasoning_bonus_mode == "only_if_correct":
        all_correct = bool(ok_dis) and (bool(ok_nov) if env_variant != "single_turn_disruption" and gold_novelty else True)
        r_reason = base_reason if all_correct else 0.0
    elif reasoning_bonus_mode == "only_if_strict_ok":
        r_reason = base_reason if bool(format_ok) else 0.0
    else:  # "always"
        r_reason = base_reason

    # Format penalty.
    forbid_counts = bool(has_forbidden_marker) and (forbidden_marker_policy == "penalize")
    format_violation = (not bool(format_ok)) or forbid_counts
    r_format = 0.0
    if strict_output_format and format_violation:
        r_format = -abs(float(format_violation_penalty))

    total = float(r_correctness) + float(r_reason) + float(r_format)

    return {
        "reward_total": float(total),
        "R_correctness": float(r_correctness),
        "R_correct_disruption": float(r_dis),
        "R_correct_novelty": float(r_nov),
        "R_reasoning": float(r_reason),
        "R_format": float(r_format),
        "disruption_correct": int(ok_dis),
        "novelty_correct": int(ok_nov),
    }


# ----------------------------
# Confusion-matrix helper metrics
# ----------------------------

def disruption_confusion_indicators(
    gold: str | None,
    pred: str | None,
    *,
    prefix: str = "cm",
) -> dict[str, int]:
    """One-hot indicators for confusion matrix cells.

    Produces 9 keys (3x3):
      cm_g_disruptive_p_neutral = 1/0, etc.

    When aggregated as means over a batch, you can reconstruct counts by:
      count(cell) ~= mean(cell) * episodes_per_batch

    This lets `viz.py` compute calibration drift from metrics.jsonl alone.
    """

    out: dict[str, int] = {}
    g = gold if gold in DISRUPTION_LABELS else None
    p = pred if pred in DISRUPTION_LABELS else None
    for gg in DISRUPTION_LABELS:
        for pp in DISRUPTION_LABELS:
            out[f"{prefix}_g_{gg}_p_{pp}"] = int(g == gg and p == pp)
    return out


def disruption_one_hot_labels(gold: str | None, pred: str | None) -> dict[str, int]:
    out: dict[str, int] = {}
    g = gold if gold in DISRUPTION_LABELS else None
    p = pred if pred in DISRUPTION_LABELS else None
    for lbl in DISRUPTION_LABELS:
        out[f"gold_is_{lbl}"] = int(g == lbl)
        out[f"pred_is_{lbl}"] = int(p == lbl)
    return out


# ----------------------------
# Record selection / loading
# ----------------------------

def select_head(dataset_jsonl: Path, wanted_ids: Sequence[str]) -> list[dict[str, Any]]:
    wanted = set(str(x) for x in wanted_ids)
    if not wanted:
        return []
    by_id: dict[str, dict[str, Any]] = {}
    for i, rec in iter_jsonl_records(dataset_jsonl, max_rows=None):
        rid = record_id(rec, i)
        if rid not in wanted:
            continue
        if rid not in by_id:
            by_id[rid] = dict(rec)
        if len(by_id) >= len(wanted):
            break
    return [by_id[rid] for rid in wanted_ids if rid in by_id]


def select_label_aware(
    *,
    dataset_jsonl: Path,
    split_ids: Sequence[str],
    label_quotas: Mapping[str, int],
    shuffle_selected: bool,
    seed: int,
) -> list[dict[str, Any]]:
    split_id_set = set(str(x) for x in split_ids)
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in DISRUPTION_LABELS}
    has_unbounded = any(int(q) < 0 for q in label_quotas.values())

    for i, rec in iter_jsonl_records(dataset_jsonl, max_rows=None):
        rid = record_id(rec, i)
        if rid not in split_id_set:
            continue
        label = normalize_label(rec.get("disruption_label"))
        if label not in by_label:
            continue
        quota = int(label_quotas.get(label, 0))
        if quota == 0:
            continue
        if quota > 0 and len(by_label[label]) >= quota:
            continue
        by_label[label].append(dict(rec))
        if not has_unbounded:
            done = all(len(by_label[l]) >= max(0, int(label_quotas.get(l, 0))) for l in DISRUPTION_LABELS)
            if done:
                break

    rows: list[dict[str, Any]] = []
    for label in DISRUPTION_LABELS:
        rows.extend(by_label[label])

    if shuffle_selected and rows:
        random.Random(int(seed)).shuffle(rows)
    return rows


def stream_sample_rl_balanced(
    *,
    dataset_jsonl: Path,
    working_set_per_label: int | None,
    working_set_total: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Reservoir sample a balanced working set directly from a (potentially huge) JSONL."""

    if working_set_per_label is None:
        total = int(working_set_total) if working_set_total is not None else 6000
        working_set_per_label = max(1, total // len(DISRUPTION_LABELS))

    target = int(working_set_per_label)
    rng = random.Random(int(seed))
    reservoirs: dict[str, list[dict[str, Any]]] = {label: [] for label in DISRUPTION_LABELS}
    seen: dict[str, int] = {label: 0 for label in DISRUPTION_LABELS}

    for _, rec in iter_jsonl_records(dataset_jsonl, max_rows=None):
        label = normalize_label(rec.get("disruption_label"))
        if label not in reservoirs:
            continue
        seen[label] += 1
        bucket = reservoirs[label]
        if len(bucket) < target:
            bucket.append(dict(rec))
        else:
            j = rng.randint(0, seen[label] - 1)
            if j < target:
                bucket[j] = dict(rec)

    rows: list[dict[str, Any]] = []
    for label in DISRUPTION_LABELS:
        rows.extend(reservoirs[label])
    rng.shuffle(rows)
    return rows


def stratified_split_records(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split rows into train/val/test with per-label stratification."""

    r_train, r_val, r_test = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    s = r_train + r_val + r_test
    if s <= 0:
        raise ValueError("Split ratios must sum to > 0")
    r_train, r_val, r_test = (r_train / s, r_val / s, r_test / s)

    rng = random.Random(int(seed))
    buckets: dict[str, list[dict[str, Any]]] = {label: [] for label in DISRUPTION_LABELS}
    for rec in rows:
        lbl = normalize_label(rec.get("disruption_label"))
        if lbl in buckets:
            buckets[lbl].append(dict(rec))

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for lbl in DISRUPTION_LABELS:
        bucket = buckets[lbl]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * r_train)
        n_val = int(n * r_val)
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    # Shuffle within each split to avoid label-block ordering.
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def load_records(
    *,
    dataset_jsonl: Path,
    splits_payload: Mapping[str, Any] | None,
    mode: str,
    seed: int,
    train_split: str,
    val_split: str,
    test_split: str,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
    train_label_quotas: Mapping[str, int] | None,
    val_label_quotas: Mapping[str, int] | None,
    shuffle_selected: bool,
    working_set_per_label: int | None,
    working_set_total: int | None,
    split_ratios: tuple[float, float, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load train/val/test working sets."""

    m = str(mode).strip().lower()
    train_records: list[dict[str, Any]] = []
    val_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    if m in {"use_splits_head", "use_splits_label_aware_quotas"}:
        if splits_payload is None:
            raise ValueError(f"mode={m} requires splits payload")

        ids_by_split = dict((splits_payload.get("ids") or {}))
        train_ids_all = [str(x) for x in ids_by_split.get(train_split, [])]
        val_ids_all = [str(x) for x in ids_by_split.get(val_split, [])]
        test_ids_all = [str(x) for x in ids_by_split.get(test_split, [])]

        if m == "use_splits_head":
            train_ids = train_ids_all[: max(0, int(max_train))] if max_train is not None else list(train_ids_all)
            val_ids = val_ids_all[: max(0, int(max_val))] if max_val is not None else list(val_ids_all)
            test_ids = test_ids_all[: max(0, int(max_test))] if max_test is not None else list(test_ids_all)

            train_records = select_head(dataset_jsonl, train_ids)
            val_records = select_head(dataset_jsonl, val_ids)
            test_records = select_head(dataset_jsonl, test_ids)
        else:
            if train_label_quotas is None or val_label_quotas is None:
                raise ValueError("use_splits_label_aware_quotas requires train and val quotas")
            train_records = select_label_aware(
                dataset_jsonl=dataset_jsonl,
                split_ids=train_ids_all,
                label_quotas=train_label_quotas,
                shuffle_selected=shuffle_selected,
                seed=int(seed) + 11,
            )
            val_records = select_label_aware(
                dataset_jsonl=dataset_jsonl,
                split_ids=val_ids_all,
                label_quotas=val_label_quotas,
                shuffle_selected=shuffle_selected,
                seed=int(seed) + 29,
            )
            test_ids = test_ids_all[: max(0, int(max_test))] if max_test is not None else list(test_ids_all)
            test_records = select_head(dataset_jsonl, test_ids)

    elif m == "rl_balanced_stream_sample":
        sampled = stream_sample_rl_balanced(
            dataset_jsonl=dataset_jsonl,
            working_set_per_label=working_set_per_label,
            working_set_total=working_set_total,
            seed=int(seed),
        )
        if sampled and shuffle_selected:
            random.Random(int(seed)).shuffle(sampled)

        # IMPORTANT: split in a label-stratified way so each split remains balanced.
        train_records, val_records, test_records = stratified_split_records(
            sampled,
            seed=int(seed) + 101,
            ratios=split_ratios,
        )

        if max_train is not None:
            train_records = train_records[: max(0, int(max_train))]
        if max_val is not None:
            val_records = val_records[: max(0, int(max_val))]
        if max_test is not None:
            test_records = test_records[: max(0, int(max_test))]
    else:
        raise ValueError("Unknown mode. Use use_splits_head/use_splits_label_aware_quotas/rl_balanced_stream_sample")

    def disruption_hist(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        cnt = Counter(normalize_label(r.get("disruption_label")) for r in rows)
        return {label: int(cnt.get(label, 0)) for label in DISRUPTION_LABELS}

    def novelty_hist(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        cnt = Counter(
            normalize_novelty_label(r.get("novelty_label"))
            for r in rows
            if r.get("novelty_label") is not None
        )
        return {label: int(cnt.get(label, 0)) for label in NOVELTY_LABELS}

    info = {
        "mode": m,
        "loaded_train_records": len(train_records),
        "loaded_val_records": len(val_records),
        "loaded_test_records": len(test_records),
        "train_label_histogram": disruption_hist(train_records),
        "val_label_histogram": disruption_hist(val_records),
        "test_label_histogram": disruption_hist(test_records),
        "train_novelty_histogram": novelty_hist(train_records),
        "val_novelty_histogram": novelty_hist(val_records),
        "test_novelty_histogram": novelty_hist(test_records),
        "train_ids_sha256": sha256_ids([record_id(r, i) for i, r in enumerate(train_records, start=1)]),
        "val_ids_sha256": sha256_ids([record_id(r, i) for i, r in enumerate(val_records, start=1)]),
        "test_ids_sha256": sha256_ids([record_id(r, i) for i, r in enumerate(test_records, start=1)]),
    }
    return train_records, val_records, test_records, info


def label_coverage_report(hist: Mapping[str, int], labels: Sequence[str]) -> dict[str, Any]:
    missing = [label for label in labels if int(hist.get(label, 0)) <= 0]
    return {
        "hist": {label: int(hist.get(label, 0)) for label in labels},
        "all_labels_present": not missing,
        "missing_labels": missing,
    }


# ----------------------------
# CLI
# ----------------------------

def add_bool_arg(parser: argparse.ArgumentParser, name: str, *, default: bool, help: str) -> None:
    """Add --foo / --no-foo boolean flag (compatible across Python versions)."""

    dest = name.lstrip("-").replace("-", "_")
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, default=default, action=argparse.BooleanOptionalAction, help=help)
        return

    # Fallback for older Python.
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name.lstrip('-')}", dest=dest, action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    cat = dataset_catalog()

    p = argparse.ArgumentParser(description="Mini CLI preflight for SciSciNet cookbook workflow")
    p.add_argument("--data-root", type=Path, default=default_data_root())
    p.add_argument("--dataset-key", choices=sorted(cat.keys()), default="sci_balanced_from2m_no_ovr_rl_balanced")

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
    add_bool_arg(
        p,
        "--shuffle-selected",
        default=True,
        help="Shuffle selected working-set records (recommended).",
    )

    p.add_argument("--env-variant", choices=ENV_VARIANTS, default="single_turn_disruption")
    p.add_argument("--prompt-max-chars", type=int, default=1800)
    add_bool_arg(p, "--include-concepts", default=False, help="Include concepts string if available")
    add_bool_arg(p, "--include-definitions", default=False, help="Include label definitions in the prompt")

    # Reward format settings (for parser sanity only in mini).
    add_bool_arg(p, "--strict-output-format", default=True, help="Apply a format penalty when output violates the format")
    p.add_argument("--format-check", choices=FORMAT_CHECK_MODES, default="exact")
    p.add_argument("--format-violation-penalty", type=float, default=1.0)
    p.add_argument("--forbidden-marker-policy", choices=FORBIDDEN_MARKER_POLICIES, default="penalize")

    p.add_argument("--weight-disruption", type=float, default=1.0)
    p.add_argument("--weight-novelty", type=float, default=1.0)

    p.add_argument("--reasoning-bonus-mode", choices=REASONING_BONUS_MODES, default="always")
    p.add_argument("--reasoning-bonus-weight", type=float, default=1.0)

    p.add_argument("--scan-max-rows", type=int, default=200000)
    add_bool_arg(p, "--full-scan-if-no-metadata", default=False, help="If metadata missing, scan full JSONL")
    add_bool_arg(p, "--print-sample-prompt", default=False, help="Print a sample prompt preview")

    p.add_argument("--write-manifest", type=Path)
    add_bool_arg(p, "--print-catalog", default=False, help="Print dataset catalog and exit")
    return p.parse_args()


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


def main() -> None:
    args = parse_args()
    cat = dataset_catalog()

    if args.print_catalog:
        print(json.dumps({k: v.__dict__ for k, v in cat.items()}, ensure_ascii=True, indent=2, sort_keys=True))
        return

    spec = cat[args.dataset_key]
    data_root = Path(args.data_root).expanduser().resolve()
    dataset_jsonl, metadata_json, splits_json = resolve_dataset_paths(data_root, spec)

    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"Missing dataset JSONL: {dataset_jsonl}")

    metadata_payload: dict[str, Any] | None = None
    if metadata_json and metadata_json.exists():
        metadata_payload = json.loads(metadata_json.read_text(encoding="utf-8"))

    if metadata_payload is not None:
        stats_payload = {
            "source": "metadata",
            "record_count": metadata_payload.get("record_count"),
            "label_counts": metadata_payload.get("label_counts"),
            "thresholds": metadata_payload.get("label_thresholds"),
            "split_counts": metadata_payload.get("split_counts"),
        }
    else:
        limit = None if args.full_scan_if_no_metadata else max(1, int(args.scan_max_rows))
        scanned = scan_dataset_stats(dataset_jsonl, max_rows=limit)
        stats_payload = {"source": "scan", **scanned}

    active_splits_json, active_splits_payload = ensure_splits(
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

    records_train, records_val, records_test, data_info = load_records(
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

    train_cov = label_coverage_report(data_info["train_label_histogram"], DISRUPTION_LABELS)
    val_cov = label_coverage_report(data_info["val_label_histogram"], DISRUPTION_LABELS)

    prompt_preview = ""
    parser_sanity: dict[str, Any] | None = None
    if records_train:
        sample = records_train[0]
        prompt_preview = build_prompt(
            sample,
            env_variant=args.env_variant,
            prompt_max_chars=int(args.prompt_max_chars),
            include_concepts=bool(args.include_concepts),
            include_definitions=bool(args.include_definitions),
        )

        gold_disruption = normalize_label(sample.get("disruption_label"))
        gold_novelty_raw = sample.get("novelty_label")
        gold_novelty = normalize_novelty_label(gold_novelty_raw) if gold_novelty_raw is not None else None

        if args.env_variant == "single_turn_disruption":
            synthetic_output = (
                f"disruption: {gold_disruption}\n"
                "reasoning: plausible field/year/citation evidence supports this label."
            )
        else:
            synthetic_output = (
                f"disruption: {gold_disruption}\n"
                f"novelty: {gold_novelty or 'balanced'}\n"
                "reasoning: plausible field/year/citation evidence supports both labels."
            )

        parsed = parse_prediction_output(synthetic_output, env_variant=args.env_variant)
        format_ok = bool(parsed["strict_ok"] if args.format_check == "exact" else parsed["keys_ok"])
        has_forbidden = any(m.lower() in synthetic_output.lower() for m in ("<think>", "</think>"))

        reward = compute_reward(
            env_variant=args.env_variant,
            pred_disruption=parsed.get("disruption_label"),
            pred_novelty=parsed.get("novelty_label"),
            gold_disruption=gold_disruption,
            gold_novelty=gold_novelty,
            reasoning=str(parsed.get("reasoning") or ""),
            format_ok=format_ok,
            has_forbidden_marker=bool(has_forbidden),
            strict_output_format=bool(args.strict_output_format),
            format_violation_penalty=float(args.format_violation_penalty),
            forbidden_marker_policy=str(args.forbidden_marker_policy),
            disruption_weights={"consolidating": 3.0, "disruptive": 2.0, "neutral": 1.0},
            novelty_weights={"novel": 1.0, "conventional": 1.0, "balanced": 1.0},
            weight_disruption=float(args.weight_disruption),
            weight_novelty=float(args.weight_novelty),
            reasoning_bonus_mode=str(args.reasoning_bonus_mode),
            reasoning_bonus_weight=float(args.reasoning_bonus_weight),
        )
        parser_sanity = {
            "synthetic_output": synthetic_output,
            "parsed": parsed,
            "format_ok": format_ok,
            "reward": reward,
        }

    manifest = {
        "script": str(Path(__file__).resolve()),
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.executable,
        "git_commit": try_git_commit(repo_root()),
        "repo_root": str(repo_root()),
        "data_root": str(data_root),
        "dataset_key": spec.key,
        "dataset_description": spec.description,
        "dataset_jsonl": str(dataset_jsonl),
        "metadata_json": str(metadata_json) if metadata_json else None,
        "splits_json": str(splits_json) if splits_json else None,
        "active_splits_json": str(active_splits_json) if active_splits_json else None,
        "dataset_jsonl_size_mib": file_size_mib(dataset_jsonl),
        "metadata_size_mib": file_size_mib(metadata_json),
        "active_splits_size_mib": file_size_mib(active_splits_json),
        "split_mode": args.split_mode,
        "loading_mode": args.loading_mode,
        "seed": int(args.seed),
        "dataset_seed": int(args.dataset_seed),
        "env_variant": args.env_variant,
        "prompt_max_chars": int(args.prompt_max_chars),
        "include_concepts": bool(args.include_concepts),
        "include_definitions": bool(args.include_definitions),
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
        "stats": stats_payload,
        "data_info": data_info,
        "train_coverage": train_cov,
        "val_coverage": val_cov,
        "reward_preview": {
            "strict_output_format": bool(args.strict_output_format),
            "format_check": str(args.format_check),
            "format_violation_penalty": float(args.format_violation_penalty),
            "forbidden_marker_policy": str(args.forbidden_marker_policy),
            "reasoning_bonus_mode": str(args.reasoning_bonus_mode),
            "reasoning_bonus_weight": float(args.reasoning_bonus_weight),
        },
        "prompt_preview_sha256": sha256_text(prompt_preview) if prompt_preview else None,
        "prompt_preview_chars": len(prompt_preview),
        "parser_reward_sanity": parser_sanity,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = sha256_text(stable_json({k: v for k, v in manifest.items() if k != "manifest_sha256"}))

    out = args.write_manifest
    if out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = repo_root() / "results" / "tinker_rl_cookbook" / f"mini_preflight_{spec.key}_{ts}.manifest.json"
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print("dataset_key:", spec.key)
    print("dataset_jsonl:", dataset_jsonl)
    print("active_splits_json:", active_splits_json)
    print("loading_mode:", args.loading_mode)
    print("train/val/test:", data_info["loaded_train_records"], data_info["loaded_val_records"], data_info["loaded_test_records"])
    print("train_hist:", data_info["train_label_histogram"])
    print("val_hist:", data_info["val_label_histogram"])
    print("train_coverage:", train_cov)
    print("val_coverage:", val_cov)
    print("manifest:", out)

    if args.print_sample_prompt and prompt_preview:
        print("\n=== SAMPLE PROMPT PREVIEW ===")
        print(prompt_preview[:1200] + ("..." if len(prompt_preview) > 1200 else ""))


if __name__ == "__main__":
    main()
