#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""Decision-oriented visual diagnostics for hosted RL experiment runs.

What changed in this refactor (v2):
- Prefer computing confusion matrices / calibration drift from metrics.jsonl directly
  using per-episode one-hot confusion indicators emitted by the runner.
  (Fallback to HTML scraping for older runs.)
- Fix effective accuracy computation to use the full gate
  (parse_success * format_ok * (1 - forbidden_marker)).
- Use format gate (not just parse_success) in the cross-run landscape.
- Expand per-run dynamics: show pred-share for all labels in the calibration panel.

The aim is to help you answer:
- "Is the run learning anything beyond format compliance?"
- "Is the model collapsing to one label (calibration drift)?"
- "Are gains real or coming from reward hacking / reasoning shaping?"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = ("consolidating", "disruptive", "neutral")
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

# Legacy HTML parsing fallback.
GOLD_PRED_RE = re.compile(
    r"Gold disruption:\s*([A-Za-z_]+)\s*\|\s*Pred disruption:\s*([A-Za-z_]+|None)",
    re.IGNORECASE,
)
ITER_FILE_RE = re.compile(r"train_iteration_(\d+)\.html$")

RANDOM_BASELINE = 1.0 / 3.0


@dataclass
class RunStats:
    name: str
    path: Path
    run_at_utc: str
    metrics: list[dict[str, Any]]
    confusion: np.ndarray
    batch_confusions: dict[int, np.ndarray]
    episodes_per_batch: int | None

    mean_parse_success: float
    mean_format_ok: float
    mean_format_strict_ok: float
    mean_format_keys_ok: float
    mean_forbidden_marker: float
    mean_raw_forbidden_marker: float

    mean_label_parsed: float
    mean_label_correct: float

    mean_reward_total: float
    mean_r_correctness: float
    mean_r_reasoning: float
    mean_r_format: float

    mean_ac_tokens: float
    mean_train_time: float

    format_gate: float
    effective_accuracy: float

    js_divergence: float
    pred_entropy: float

    recall_consolidating: float
    recall_disruptive: float
    recall_neutral: float
    balanced_accuracy: float

    workflow_score: float
    n_predictions: int

    @property
    def run_tag(self) -> str:
        m = re.search(r"(20\d{6}_\d{6})$", self.name)
        if m:
            return m.group(1)
        return self.name[-14:]


def normalize_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in LABEL_TO_IDX:
        return text
    return None


def mean_metric(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        raw = row.get(key)
        if isinstance(raw, (int, float)):
            vals.append(float(raw))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def metric_series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = []
    for row in rows:
        raw = row.get(key)
        if isinstance(raw, (int, float)):
            vals.append(float(raw))
        else:
            vals.append(float("nan"))
    return np.asarray(vals, dtype=float)


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def parse_manifest(run_dir: Path) -> dict[str, Any]:
    manifest = run_dir / "run_manifest.json"
    if not manifest.exists():
        return {}
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def safe_probs(counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(counts))
    if total <= 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / total


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p2 = np.clip(p.astype(float), eps, 1.0)
    q2 = np.clip(q.astype(float), eps, 1.0)
    return float(np.sum(p2 * np.log2(p2 / q2)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if np.sum(p) <= 0 or np.sum(q) <= 0:
        return float("nan")
    p2 = p / np.sum(p)
    q2 = q / np.sum(q)
    m = 0.5 * (p2 + q2)
    return 0.5 * kl_divergence(p2, m) + 0.5 * kl_divergence(q2, m)


def normalized_entropy(p: np.ndarray) -> float:
    if np.sum(p) <= 0:
        return float("nan")
    p2 = np.clip(p / np.sum(p), 1e-12, 1.0)
    h = -np.sum(p2 * np.log2(p2))
    return float(h / math.log2(len(p2)))


def confusion_to_recalls(cm: np.ndarray) -> tuple[float, float, float, float]:
    recalls = []
    for i in range(len(LABELS)):
        denom = float(np.sum(cm[i, :]))
        if denom <= 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float(cm[i, i] / denom))
    balanced = float(np.nanmean(recalls)) if recalls else float("nan")
    return recalls[0], recalls[1], recalls[2], balanced


def compute_workflow_score(format_gate: float, label_correct: float, jsd: float) -> float:
    f = float(np.clip(format_gate, 0.0, 1.0)) if not np.isnan(format_gate) else 0.0
    l = float(np.clip(label_correct, 0.0, 1.0)) if not np.isnan(label_correct) else 0.0
    c = 1.0 - float(np.clip(jsd, 0.0, 1.0)) if not np.isnan(jsd) else 0.0
    return 0.45 * f + 0.35 * l + 0.20 * c


# ----------------------------
# Confusion matrix extraction
# ----------------------------

def parse_batch_confusions_from_html(run_dir: Path) -> dict[int, np.ndarray]:
    batch_confusions: dict[int, np.ndarray] = {}
    for html_path in sorted(run_dir.glob("train_iteration_*.html")):
        m = ITER_FILE_RE.search(html_path.name)
        if not m:
            continue
        batch_idx = int(m.group(1))
        text = html_path.read_text(encoding="utf-8", errors="ignore")
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
        for gold_raw, pred_raw in GOLD_PRED_RE.findall(text):
            gold = normalize_label(gold_raw)
            pred = normalize_label(pred_raw)
            if gold is None or pred is None:
                continue
            cm[LABEL_TO_IDX[gold], LABEL_TO_IDX[pred]] += 1
        if int(np.sum(cm)) > 0:
            batch_confusions[batch_idx] = cm
    return batch_confusions


def _has_confusion_metrics(metrics: list[dict[str, Any]]) -> bool:
    if not metrics:
        return False
    row0 = metrics[0]
    for g in LABELS:
        for p in LABELS:
            k = f"env/all/cm_g_{g}_p_{p}"
            if k in row0:
                return True
    return False


def parse_batch_confusions_from_metrics(
    metrics: list[dict[str, Any]],
    *,
    episodes_per_batch: int,
) -> dict[int, np.ndarray]:
    """Reconstruct per-batch confusion matrices from per-episode indicators.

    Each confusion indicator is a 0/1 metric per episode. When the trainer aggregates
    env metrics, it averages across episodes; multiply the mean by episodes_per_batch
    to get an approximate count.
    """

    out: dict[int, np.ndarray] = {}
    n_batches = len(metrics)
    if n_batches == 0:
        return out

    for b in range(n_batches):
        row = metrics[b]
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=float)
        for gi, g in enumerate(LABELS):
            for pi, p in enumerate(LABELS):
                k = f"env/all/cm_g_{g}_p_{p}"
                v = row.get(k)
                if isinstance(v, (int, float)):
                    cm[gi, pi] = float(v) * float(episodes_per_batch)
        if float(np.sum(cm)) <= 0.0:
            continue
        # Round to integers for display, but keep non-negativity.
        out[b] = np.maximum(0.0, np.rint(cm)).astype(int)
    return out


# ----------------------------
# Run loading
# ----------------------------

def load_run_stats(run_dir: Path) -> RunStats | None:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    metrics = parse_jsonl(metrics_path)
    if not metrics:
        return None

    manifest = parse_manifest(run_dir)
    run_at_utc = str(manifest.get("run_at_utc") or "")

    groups_per_batch = None
    group_size = None
    try:
        rcfg = manifest.get("runner_config") or {}
        groups_per_batch = int(rcfg.get("groups_per_batch")) if rcfg.get("groups_per_batch") is not None else None
        group_size = int(rcfg.get("group_size")) if rcfg.get("group_size") is not None else None
    except Exception:
        groups_per_batch, group_size = None, None

    episodes_per_batch = None
    if groups_per_batch is not None and group_size is not None and groups_per_batch > 0 and group_size > 0:
        episodes_per_batch = int(groups_per_batch) * int(group_size)

    # Prefer metrics-based confusion; fallback to HTML parsing.
    cm_by_batch: dict[int, np.ndarray] = {}
    if episodes_per_batch is not None and _has_confusion_metrics(metrics):
        cm_by_batch = parse_batch_confusions_from_metrics(metrics, episodes_per_batch=episodes_per_batch)
    else:
        cm_by_batch = parse_batch_confusions_from_html(run_dir)

    cm_total = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for cm in cm_by_batch.values():
        cm_total += cm

    # Core scalar metrics.
    mean_parse = mean_metric(metrics, "env/all/parse_success")

    mean_format_ok = mean_metric(metrics, "env/all/format_ok")
    mean_format_strict = mean_metric(metrics, "env/all/format_strict_ok")
    mean_format_keys = mean_metric(metrics, "env/all/format_keys_ok")

    # Backward compat: older runs won't have format_ok.
    if np.isnan(mean_format_ok):
        mean_format_ok = mean_format_strict

    mean_forbidden = mean_metric(metrics, "env/all/has_forbidden_output_marker")
    mean_raw_forbidden = mean_metric(metrics, "env/all/raw_has_forbidden_output_marker")

    mean_label_parsed = mean_metric(metrics, "env/all/label_parsed")
    mean_label_correct = mean_metric(metrics, "env/all/label_correct")

    mean_reward_total = mean_metric(metrics, "env/all/reward/total")
    mean_r_correctness = mean_metric(metrics, "env/all/R_correctness")
    mean_r_reasoning = mean_metric(metrics, "env/all/R_reasoning")
    mean_r_format = mean_metric(metrics, "env/all/R_format")
    mean_ac_tokens = mean_metric(metrics, "env/all/ac_tokens_per_turn")
    mean_train_time = mean_metric(metrics, "time/train")

    # Series for gate computations.
    parse_series = metric_series(metrics, "env/all/parse_success")
    forb_series = metric_series(metrics, "env/all/has_forbidden_output_marker")

    format_ok_series = metric_series(metrics, "env/all/format_ok")
    if np.all(np.isnan(format_ok_series)):
        # Older run: approximate with strict_ok
        format_ok_series = metric_series(metrics, "env/all/format_strict_ok")

    label_series = metric_series(metrics, "env/all/label_correct")

    gate_series = parse_series * format_ok_series * (1.0 - forb_series)
    format_gate = float(np.nanmean(gate_series))
    effective_accuracy = float(np.nanmean(label_series * gate_series))

    gold_dist = safe_probs(np.sum(cm_total, axis=1))
    pred_dist = safe_probs(np.sum(cm_total, axis=0))
    jsd = js_divergence(gold_dist, pred_dist)
    pred_h = normalized_entropy(pred_dist)

    rec_cons, rec_disr, rec_neut, bal_acc = confusion_to_recalls(cm_total)
    score = compute_workflow_score(format_gate, mean_label_correct, jsd)

    return RunStats(
        name=run_dir.name,
        path=run_dir,
        run_at_utc=run_at_utc,
        metrics=metrics,
        confusion=cm_total,
        batch_confusions=cm_by_batch,
        episodes_per_batch=episodes_per_batch,
        mean_parse_success=mean_parse,
        mean_format_ok=mean_format_ok,
        mean_format_strict_ok=mean_format_strict,
        mean_format_keys_ok=mean_format_keys,
        mean_forbidden_marker=mean_forbidden,
        mean_raw_forbidden_marker=mean_raw_forbidden,
        mean_label_parsed=mean_label_parsed,
        mean_label_correct=mean_label_correct,
        mean_reward_total=mean_reward_total,
        mean_r_correctness=mean_r_correctness,
        mean_r_reasoning=mean_r_reasoning,
        mean_r_format=mean_r_format,
        mean_ac_tokens=mean_ac_tokens,
        mean_train_time=mean_train_time,
        format_gate=format_gate,
        effective_accuracy=effective_accuracy,
        js_divergence=jsd,
        pred_entropy=pred_h,
        recall_consolidating=rec_cons,
        recall_disruptive=rec_disr,
        recall_neutral=rec_neut,
        balanced_accuracy=bal_acc,
        workflow_score=score,
        n_predictions=int(np.sum(cm_total)),
    )


def discover_run_dirs(results_root: Path, run_glob: str, max_runs: int) -> list[Path]:
    run_dirs = [p for p in results_root.glob(run_glob) if p.is_dir()]
    run_dirs = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    if max_runs > 0:
        run_dirs = run_dirs[:max_runs]
    return run_dirs


def sort_stats(stats: list[RunStats]) -> list[RunStats]:
    def key_fn(s: RunStats) -> tuple[str, str]:
        return s.run_at_utc, s.name

    return sorted(stats, key=key_fn)


def pareto_indices(xs: np.ndarray, ys: np.ndarray) -> list[int]:
    idxs: list[int] = []
    for i in range(len(xs)):
        dominated = False
        for j in range(len(xs)):
            if i == j:
                continue
            better_or_equal = xs[j] <= xs[i] and ys[j] >= ys[i]
            strictly_better = xs[j] < xs[i] or ys[j] > ys[i]
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


# ----------------------------
# Plots
# ----------------------------

def plot_workflow_landscape(stats: list[RunStats], out_path: Path) -> None:
    if not stats:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Gate map: format reliability vs learning signal.
    x1 = np.asarray([s.format_gate for s in stats], dtype=float)
    y1 = np.asarray([s.mean_label_correct for s in stats], dtype=float)
    c1 = np.asarray([s.js_divergence if not np.isnan(s.js_divergence) else 1.0 for s in stats], dtype=float)
    s1 = np.asarray([70.0 + 2.0 * max(1.0, s.mean_ac_tokens) for s in stats], dtype=float)

    sc1 = axes[0].scatter(x1, y1, c=c1, s=s1, cmap="viridis_r", alpha=0.9, edgecolor="black", linewidth=0.4)
    for i, st in enumerate(stats):
        axes[0].annotate(st.run_tag, (x1[i], y1[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].axvline(0.95, color="tab:red", linestyle="--", linewidth=1, label="format gate target (0.95)")
    axes[0].axhline(RANDOM_BASELINE, color="tab:orange", linestyle="--", linewidth=1, label="random baseline (1/3)")
    axes[0].set_xlabel("Format Gate (mean parse * format_ok * (1-forbidden))")
    axes[0].set_ylabel("Mean Label Accuracy")
    axes[0].set_title("Workflow Gate Map: Format Reliability vs Learning Signal")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right", fontsize=8)
    cb1 = fig.colorbar(sc1, ax=axes[0], shrink=0.92)
    cb1.set_label("JS Divergence (Gold vs Pred Label Distribution)")

    # Efficiency frontier: quality vs compute.
    x2 = np.asarray([s.mean_train_time for s in stats], dtype=float)
    y2 = np.asarray([s.mean_label_correct for s in stats], dtype=float)
    c2 = np.asarray([s.workflow_score for s in stats], dtype=float)
    s2 = np.asarray([120.0 + 280.0 * np.clip(s.format_gate, 0.0, 1.0) for s in stats], dtype=float)

    sc2 = axes[1].scatter(x2, y2, c=c2, s=s2, cmap="plasma", alpha=0.92, edgecolor="black", linewidth=0.4)
    for i, st in enumerate(stats):
        axes[1].annotate(st.run_tag, (x2[i], y2[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    if np.nanmax(x2) > 10.0 * max(1e-6, np.nanmin(x2)):
        axes[1].set_xscale("log")
    idxs = pareto_indices(x2, y2)
    if idxs:
        px = x2[idxs]
        py = y2[idxs]
        order = np.argsort(px)
        axes[1].plot(px[order], py[order], color="black", linestyle="--", linewidth=1.2, label="Pareto frontier")
    axes[1].axhline(RANDOM_BASELINE, color="tab:orange", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Mean Train Time per Batch (sec)")
    axes[1].set_ylabel("Mean Label Accuracy")
    axes[1].set_title("Efficiency Frontier: Quality vs Compute Cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower right", fontsize=8)
    cb2 = fig.colorbar(sc2, ax=axes[1], shrink=0.92)
    cb2.set_label("Workflow Score")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_recall_calibration(stats: list[RunStats], out_path: Path) -> None:
    if not stats:
        return
    labels = [s.run_tag for s in stats]
    x = np.arange(len(stats))
    w = 0.22

    r_cons = np.asarray([s.recall_consolidating for s in stats], dtype=float)
    r_disr = np.asarray([s.recall_disruptive for s in stats], dtype=float)
    r_neut = np.asarray([s.recall_neutral for s in stats], dtype=float)
    jsd = np.asarray([s.js_divergence for s in stats], dtype=float)

    fig, ax1 = plt.subplots(figsize=(max(10, 1.25 * len(stats)), 6), constrained_layout=True)
    ax1.bar(x - w, r_cons, width=w, label="Recall(consolidating)", color="#4c78a8")
    ax1.bar(x, r_disr, width=w, label="Recall(disruptive)", color="#e45756")
    ax1.bar(x + w, r_neut, width=w, label="Recall(neutral)", color="#72b7b2")
    ax1.axhline(RANDOM_BASELINE, color="tab:orange", linestyle="--", linewidth=1, label="random baseline (1/3)")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Per-Class Recall")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Class Recall Stability and Calibration Drift Across Runs")
    ax1.grid(axis="y", alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(x, jsd, color="black", marker="o", linewidth=1.4, label="JS divergence")
    ax2.set_ylabel("JS Divergence (lower is better)")
    ax2.set_ylim(bottom=0.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_panels(stats: list[RunStats], out_path: Path, panel_runs: int) -> None:
    selected = [s for s in sort_stats(stats)][-panel_runs:]
    selected = [s for s in selected if s.n_predictions > 0]
    if not selected:
        return

    n = len(selected)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    for i, run in enumerate(selected):
        ax = axes_arr[i]
        cm = run.confusion.astype(float)
        row_sums = np.sum(cm, axis=1, keepdims=True)
        norm = np.divide(cm, np.maximum(row_sums, 1.0), where=True)
        im = ax.imshow(norm, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(LABELS)))
        ax.set_yticks(np.arange(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=20, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(
            f"{run.run_tag}\nacc={run.mean_label_correct:.3f}, bal_acc={run.balanced_accuracy:.3f}, "
            f"JS={run.js_divergence:.3f}"
        )
        for r in range(len(LABELS)):
            for c in range(len(LABELS)):
                value = norm[r, c]
                count = int(cm[r, c])
                color = "white" if value < 0.65 else "black"
                ax.text(c, r, f"{value:.2f}\n({count})", ha="center", va="center", color=color, fontsize=8)

    for j in range(i + 1, len(axes_arr)):
        axes_arr[j].axis("off")
    fig.colorbar(im, ax=axes_arr.tolist(), fraction=0.02, pad=0.02, label="Row-Normalized Rate")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def batch_diagnostics(run: RunStats) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_batches = len(run.metrics)
    js = np.full(n_batches, np.nan, dtype=float)
    pred_shares: dict[str, np.ndarray] = {lbl: np.full(n_batches, np.nan, dtype=float) for lbl in LABELS}

    for batch_idx, cm in run.batch_confusions.items():
        if batch_idx < 0 or batch_idx >= n_batches:
            continue
        gold_dist = safe_probs(np.sum(cm, axis=1))
        pred_dist = safe_probs(np.sum(cm, axis=0))
        js[batch_idx] = js_divergence(gold_dist, pred_dist)
        for lbl in LABELS:
            pred_shares[lbl][batch_idx] = float(pred_dist[LABEL_TO_IDX[lbl]])

    return js, pred_shares


def plot_run_dynamics(run: RunStats, out_path: Path) -> None:
    batches = np.arange(len(run.metrics))
    if len(batches) == 0:
        return

    reward_total = metric_series(run.metrics, "env/all/reward/total")
    r_corr = metric_series(run.metrics, "env/all/R_correctness")
    r_reason = metric_series(run.metrics, "env/all/R_reasoning")
    r_format = metric_series(run.metrics, "env/all/R_format")

    parse = metric_series(run.metrics, "env/all/parse_success")
    forb = metric_series(run.metrics, "env/all/has_forbidden_output_marker")

    fmt_ok = metric_series(run.metrics, "env/all/format_ok")
    if np.all(np.isnan(fmt_ok)):
        fmt_ok = metric_series(run.metrics, "env/all/format_strict_ok")

    label_acc = metric_series(run.metrics, "env/all/label_correct")
    gate = parse * fmt_ok * (1.0 - forb)
    eff_acc = label_acc * gate

    js, pred_shares = batch_diagnostics(run)

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True, constrained_layout=True)

    # Panel 1: reward decomposition.
    axes[0].plot(batches, reward_total, marker="o", linewidth=2.0, label="reward_total", color="black")
    axes[0].plot(batches, r_corr, marker="o", linewidth=1.6, label="R_correctness", color="#1f77b4")
    axes[0].plot(batches, r_reason, marker="o", linewidth=1.6, label="R_reasoning", color="#2ca02c")
    axes[0].plot(batches, r_format, marker="o", linewidth=1.6, label="R_format", color="#d62728")
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Reward")
    axes[0].set_title(f"Run Dynamics: {run.name}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", ncols=2, fontsize=8)

    # Panel 2: format gate & effective accuracy.
    axes[1].plot(batches, parse, marker="o", label="parse_success", color="#9467bd")
    axes[1].plot(batches, fmt_ok, marker="o", label="format_ok", color="#8c564b")
    axes[1].plot(batches, 1.0 - forb, marker="o", label="1 - forbidden_marker", color="#17becf")
    axes[1].plot(batches, label_acc, marker="o", label="label_correct", color="#ff7f0e")
    axes[1].plot(batches, gate, marker="o", label="effective_gate", color="#2f4f4f")
    axes[1].plot(batches, eff_acc, marker="o", label="effective_accuracy", color="#444444")
    axes[1].axhline(RANDOM_BASELINE, color="tab:orange", linestyle="--", linewidth=1, label="random baseline")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Format Gating and Effective Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", ncols=3, fontsize=8)

    # Panel 3: calibration drift & predicted label shares.
    axes[2].plot(batches, js, marker="o", linewidth=1.8, color="black", label="JS divergence")
    axes[2].set_ylabel("JS Divergence")
    axes[2].grid(alpha=0.25)
    axes[2].set_xlabel("Batch Index")
    axes[2].set_title("Calibration Drift and Predicted Label Distribution")
    axr = axes[2].twinx()
    axr.set_ylabel("Predicted Share")
    for lbl, arr in pred_shares.items():
        axr.plot(batches, arr, marker="o", linewidth=1.4, label=f"pred_share({lbl})")
    axr.axhline(RANDOM_BASELINE, color="tab:orange", linestyle="--", linewidth=1)
    axr.set_ylim(0.0, 1.0)

    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc="best", fontsize=8, ncols=2)

    # Avoid absurdly dense x tick labels on long runs.
    if len(batches) <= 60:
        axes[2].set_xticks(batches)
    else:
        step = max(1, len(batches) // 20)
        axes[2].set_xticks(batches[::step])

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ----------------------------
# Summary CSV
# ----------------------------

def write_summary_csv(stats: list[RunStats], out_path: Path) -> None:
    fieldnames = [
        "run_name",
        "run_tag",
        "run_at_utc",
        "path",
        "n_batches",
        "episodes_per_batch",
        "n_predictions",
        "mean_parse_success",
        "mean_format_ok",
        "mean_format_strict_ok",
        "mean_format_keys_ok",
        "mean_forbidden_marker",
        "mean_raw_forbidden_marker",
        "mean_label_parsed",
        "mean_label_correct",
        "mean_reward_total",
        "mean_r_correctness",
        "mean_r_reasoning",
        "mean_r_format",
        "mean_ac_tokens",
        "mean_train_time",
        "format_gate",
        "effective_accuracy",
        "js_divergence",
        "pred_entropy",
        "recall_consolidating",
        "recall_disruptive",
        "recall_neutral",
        "balanced_accuracy",
        "workflow_score",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "run_name": s.name,
                    "run_tag": s.run_tag,
                    "run_at_utc": s.run_at_utc,
                    "path": str(s.path),
                    "n_batches": len(s.metrics),
                    "episodes_per_batch": s.episodes_per_batch,
                    "n_predictions": s.n_predictions,
                    "mean_parse_success": s.mean_parse_success,
                    "mean_format_ok": s.mean_format_ok,
                    "mean_format_strict_ok": s.mean_format_strict_ok,
                    "mean_format_keys_ok": s.mean_format_keys_ok,
                    "mean_forbidden_marker": s.mean_forbidden_marker,
                    "mean_raw_forbidden_marker": s.mean_raw_forbidden_marker,
                    "mean_label_parsed": s.mean_label_parsed,
                    "mean_label_correct": s.mean_label_correct,
                    "mean_reward_total": s.mean_reward_total,
                    "mean_r_correctness": s.mean_r_correctness,
                    "mean_r_reasoning": s.mean_r_reasoning,
                    "mean_r_format": s.mean_r_format,
                    "mean_ac_tokens": s.mean_ac_tokens,
                    "mean_train_time": s.mean_train_time,
                    "format_gate": s.format_gate,
                    "effective_accuracy": s.effective_accuracy,
                    "js_divergence": s.js_divergence,
                    "pred_entropy": s.pred_entropy,
                    "recall_consolidating": s.recall_consolidating,
                    "recall_disruptive": s.recall_disruptive,
                    "recall_neutral": s.recall_neutral,
                    "balanced_accuracy": s.balanced_accuracy,
                    "workflow_score": s.workflow_score,
                }
            )


# ----------------------------
# Entrypoint
# ----------------------------

def default_results_root() -> Path:
    outside = Path("../results/tinker_rl_cookbook")
    inside = Path("results/tinker_rl_cookbook")
    if outside.exists():
        return outside
    return inside


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual diagnostics for hosted RL run artifacts.")
    p.add_argument("--results-root", type=Path, default=default_results_root())
    p.add_argument("--run-glob", default="*")
    p.add_argument("--max-runs", type=int, default=30)
    p.add_argument("--detail-runs", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("results/analysis_plots"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    run_dirs = discover_run_dirs(results_root, args.run_glob, int(args.max_runs))
    stats: list[RunStats] = []
    for run_dir in run_dirs:
        st = load_run_stats(run_dir)
        if st is not None:
            stats.append(st)
    stats = sort_stats(stats)
    if not stats:
        raise RuntimeError(f"No valid run directories with metrics.jsonl found under: {results_root}")

    write_summary_csv(stats, output_dir / "run_summary.csv")
    plot_workflow_landscape(stats, output_dir / "workflow_landscape.png")
    plot_recall_calibration(stats, output_dir / "class_recall_and_calibration.png")
    plot_confusion_panels(stats, output_dir / "confusion_panels_recent.png", int(args.detail_runs))

    recent = stats[-int(args.detail_runs) :]
    for run in recent:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.name)
        plot_run_dynamics(run, output_dir / f"run_dynamics_{safe_name}.png")

    print(f"Processed runs: {len(stats)}")
    print(f"Results root: {results_root}")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    print(f"- {output_dir / 'run_summary.csv'}")
    print(f"- {output_dir / 'workflow_landscape.png'}")
    print(f"- {output_dir / 'class_recall_and_calibration.png'}")
    print(f"- {output_dir / 'confusion_panels_recent.png'}")
    for run in recent:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run.name)
        print(f"- {output_dir / f'run_dynamics_{safe_name}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
