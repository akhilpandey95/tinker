# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tinker_disruption_rl.tinker_disruption_env import Paper, derive_novelty_label
from training.rlvr.metrics import load_json

from training.tinker_rl.tinker_env_adapter import (
    DISRUPTION_LABELS,
    DisruptionPredictionEnvV1,
    TinkerDisruptionEnv,
    TinkerEnvGroupBuilderBase,
    TinkerRLDatasetBase,
)


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value]
    return [str(value)]


def _paper_from_record(record: Mapping[str, Any]) -> Paper:
    # Preserve the RLVR trainer workaround: Paper validates novelty from novelty_score,
    # while dataset novelty labels are delta-margin based.
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
        primary_field=str(record.get("primary_field", "Unknown")),
    )


def label_histogram(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(_normalize_label(record.get("disruption_label")) for record in records)
    return {label: int(counts.get(label, 0)) for label in DISRUPTION_LABELS}


def _clip_ids(ids: Sequence[str], max_items: Optional[int]) -> List[str]:
    out = list(ids)
    if max_items is not None:
        out = out[: max(0, int(max_items))]
    return out


def load_split_records_streaming(
    *,
    dataset_jsonl: Path,
    splits_json: Path,
    train_split: str = "train",
    val_split: str = "val",
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    stratified_train_limit: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    splits = load_json(splits_json)
    split_ids = dict(splits.get("ids", {}))
    raw_train_ids = list(split_ids.get(train_split, []))
    train_ids = _clip_ids(raw_train_ids, max_train)
    val_ids = _clip_ids(split_ids.get(val_split, []), max_val)

    want_all_train_ids = bool(stratified_train_limit and max_train is not None)
    wanted_train = set(str(item) for item in (raw_train_ids if want_all_train_ids else train_ids))
    wanted_val = set(str(item) for item in val_ids)
    wanted_all = wanted_train | wanted_val
    by_id: Dict[str, Dict[str, Any]] = {}
    val_found = 0
    val_found_ids: set[str] = set()

    train_records_stratified: List[Dict[str, Any]] = []
    train_counts = {label: 0 for label in DISRUPTION_LABELS}
    train_quota = {label: 0 for label in DISRUPTION_LABELS}
    if want_all_train_ids:
        for idx in range(int(max_train)):
            train_quota[DISRUPTION_LABELS[idx % len(DISRUPTION_LABELS)]] += 1

    with dataset_jsonl.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {dataset_jsonl}:{line_no}: {exc}") from exc
            record_id = str(record.get("openalex_id"))
            if record_id not in wanted_all:
                continue

            if record_id in wanted_val and record_id not in val_found_ids:
                by_id[record_id] = record
                val_found_ids.add(record_id)
                val_found += 1

            if want_all_train_ids and record_id in wanted_train:
                label = _normalize_label(record.get("disruption_label"))
                if label in train_quota and train_counts[label] < train_quota[label]:
                    train_records_stratified.append(dict(record))
                    train_counts[label] += 1
            elif record_id in wanted_train:
                by_id[record_id] = record

            if want_all_train_ids:
                train_done = sum(train_counts.values()) >= int(max_train)
                if train_done and val_found >= len(val_ids):
                    break
            elif len(by_id) == len(wanted_all):
                break

    if want_all_train_ids:
        train_records = train_records_stratified[: int(max_train)]
    else:
        train_records = [dict(by_id[record_id]) for record_id in train_ids if record_id in by_id]
    val_records = [dict(by_id[record_id]) for record_id in val_ids if record_id in by_id]

    dataset_info = {
        "dataset_jsonl": str(dataset_jsonl),
        "splits_json": str(splits_json),
        "train_split": train_split,
        "val_split": val_split,
        "requested_train_ids": len(train_ids),
        "requested_val_ids": len(val_ids),
        "loaded_train_records": len(train_records),
        "loaded_val_records": len(val_records),
        "train_limit_mode": "stratified_scan" if want_all_train_ids else "split_head",
        "split_counts": splits.get("counts"),
        "split_seed": splits.get("split_seed"),
    }
    return train_records, val_records, dataset_info


@dataclass
class DisruptionEnvGroupBuilder(TinkerEnvGroupBuilderBase):
    record: Mapping[str, Any]
    group_size: int
    renderer: Any
    system_prompt: str | None
    max_env_tokens: int
    prompt_max_chars: int
    include_concepts: bool = False

    @property
    def openalex_id(self) -> str:
        return str(self.record["openalex_id"])

    @property
    def disruption_label(self) -> str:
        return _normalize_label(self.record.get("disruption_label"))

    def build_inner_env(self) -> DisruptionPredictionEnvV1:
        paper = _paper_from_record(self.record)
        concepts = _safe_list(self.record.get("concepts")) if self.include_concepts else None
        return DisruptionPredictionEnvV1(
            paper=paper,
            max_tokens=int(self.max_env_tokens),
            prompt_max_chars=int(self.prompt_max_chars),
            concepts=concepts,
        )

    def build_prompt_text(self) -> str:
        inner_env = self.build_inner_env()
        # Reuse the inner env's v1 prompt construction without invoking async execution.
        paper = inner_env.paper
        concepts = _safe_list(self.record.get("concepts")) if self.include_concepts else None
        from training.tinker_rl.tinker_env_adapter import build_disruption_v1_prompt_from_paper

        return build_disruption_v1_prompt_from_paper(
            paper,
            prompt_max_chars=int(self.prompt_max_chars),
            concepts=concepts,
        )

    async def make_envs(self) -> List[TinkerDisruptionEnv]:
        envs: List[TinkerDisruptionEnv] = []
        for _ in range(int(self.group_size)):
            envs.append(
                TinkerDisruptionEnv(
                    inner_env=self.build_inner_env(),
                    renderer=self.renderer,
                    system_prompt=self.system_prompt,
                )
            )
        return envs


class SciSciNetRLDataset(TinkerRLDatasetBase):
    """Tinker-style RL dataset wrapper with stratified disruption sampling."""

    def __init__(
        self,
        *,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]],
        renderer: Any,
        group_size: int,
        batch_size: int,
        system_prompt: str | None,
        max_env_tokens: int,
        prompt_max_chars: int,
        sampling_strategy: str = "stratified",
        seed: int = 20260220,
        include_concepts: bool = False,
    ) -> None:
        self.train_records = [dict(record) for record in train_records]
        self.val_records = [dict(record) for record in val_records]
        self.renderer = renderer
        self.group_size = int(group_size)
        self.batch_size = int(batch_size)
        self.system_prompt = system_prompt
        self.max_env_tokens = int(max_env_tokens)
        self.prompt_max_chars = int(prompt_max_chars)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.seed = int(seed)
        self.include_concepts = bool(include_concepts)
        self._rng = random.Random(self.seed)
        self._batch_calls = 0

        if self.group_size <= 0:
            raise ValueError("group_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not self.train_records:
            raise ValueError("train_records is empty.")
        if self.sampling_strategy not in {"stratified", "natural"}:
            raise ValueError("sampling_strategy must be one of: stratified, natural")

        self._label_buckets: Dict[str, List[Dict[str, Any]]] = {label: [] for label in DISRUPTION_LABELS}
        for record in self.train_records:
            label = _normalize_label(record.get("disruption_label"))
            if label in self._label_buckets:
                self._label_buckets[label].append(dict(record))
        for label, bucket in self._label_buckets.items():
            if not bucket:
                raise ValueError(
                    f"Training split has no records for disruption label {label!r}; "
                    "stratified sampling cannot preserve all classes."
                )
            self._rng.shuffle(bucket)

        self._bucket_positions = {label: 0 for label in DISRUPTION_LABELS}
        self._natural_position = 0

    @classmethod
    def from_files(
        cls,
        *,
        dataset_jsonl: Path,
        splits_json: Path,
        renderer: Any,
        group_size: int,
        batch_size: int,
        system_prompt: str | None,
        max_env_tokens: int,
        prompt_max_chars: int,
        sampling_strategy: str = "stratified",
        seed: int = 20260220,
        train_split: str = "train",
        val_split: str = "val",
        max_train: Optional[int] = None,
        max_val: Optional[int] = None,
        include_concepts: bool = False,
    ) -> tuple["SciSciNetRLDataset", Dict[str, Any]]:
        train_records, val_records, dataset_info = load_split_records_streaming(
            dataset_jsonl=dataset_jsonl,
            splits_json=splits_json,
            train_split=train_split,
            val_split=val_split,
            max_train=max_train,
            max_val=max_val,
            stratified_train_limit=(str(sampling_strategy).strip().lower() == "stratified"),
        )
        if not train_records:
            raise ValueError("No train records selected. Check split names/files.")
        if not val_records:
            raise ValueError("No val records selected. Check split names/files.")
        dataset = cls(
            train_records=train_records,
            val_records=val_records,
            renderer=renderer,
            group_size=group_size,
            batch_size=batch_size,
            system_prompt=system_prompt,
            max_env_tokens=max_env_tokens,
            prompt_max_chars=prompt_max_chars,
            sampling_strategy=sampling_strategy,
            seed=seed,
            include_concepts=include_concepts,
        )
        dataset_info = dict(dataset_info)
        dataset_info.update(
            {
                "train_label_histogram": label_histogram(train_records),
                "val_label_histogram": label_histogram(val_records),
            }
        )
        return dataset, dataset_info

    def _next_record_for_label(self, label: str) -> Dict[str, Any]:
        bucket = self._label_buckets[label]
        pos = self._bucket_positions[label]
        if pos >= len(bucket):
            self._rng.shuffle(bucket)
            pos = 0
        self._bucket_positions[label] = pos + 1
        return dict(bucket[pos])

    def _next_natural_record(self) -> Dict[str, Any]:
        if self._natural_position >= len(self.train_records):
            self._natural_position = 0
        record = dict(self.train_records[self._natural_position])
        self._natural_position += 1
        return record

    def get_batch(self, index: int) -> List[DisruptionEnvGroupBuilder]:
        _ = index  # The sampler is stateful but deterministic given seed + call order.
        builders: List[DisruptionEnvGroupBuilder] = []
        label_targets: List[str] = []

        if self.sampling_strategy == "stratified":
            for item_idx in range(self.batch_size):
                label_targets.append(DISRUPTION_LABELS[item_idx % len(DISRUPTION_LABELS)])
        else:
            label_targets = []

        for item_idx in range(self.batch_size):
            if self.sampling_strategy == "stratified":
                record = self._next_record_for_label(label_targets[item_idx])
            else:
                record = self._next_natural_record()
            builders.append(
                DisruptionEnvGroupBuilder(
                    record=record,
                    group_size=self.group_size,
                    renderer=self.renderer,
                    system_prompt=self.system_prompt,
                    max_env_tokens=self.max_env_tokens,
                    prompt_max_chars=self.prompt_max_chars,
                    include_concepts=self.include_concepts,
                )
            )

        self._batch_calls += 1
        return builders

    def observed_batch_label_histogram(self, builders: Sequence[DisruptionEnvGroupBuilder]) -> Dict[str, int]:
        counts = Counter(builder.disruption_label for builder in builders)
        return {label: int(counts.get(label, 0)) for label in DISRUPTION_LABELS}

    def sampling_manifest(self) -> Dict[str, Any]:
        bucket_sizes = {label: len(self._label_buckets[label]) for label in DISRUPTION_LABELS}
        return {
            "sampling_strategy": self.sampling_strategy,
            "seed": self.seed,
            "label_order_cycle": list(DISRUPTION_LABELS) if self.sampling_strategy == "stratified" else None,
            "group_size": self.group_size,
            "batch_size": self.batch_size,
            "train_label_histogram_natural": label_histogram(self.train_records),
            "val_label_histogram_natural": label_histogram(self.val_records),
            "train_bucket_sizes": bucket_sizes,
            "batch_calls_so_far": self._batch_calls,
            "novelty_label_handling": (
                "Paper.__post_init__ compatibility: novelty_label is re-derived from novelty_score "
                "using derive_novelty_label() at env construction time."
            ),
        }

    def preview_train_prompt(self, index: int = 0) -> str:
        record = dict(self.train_records[index % len(self.train_records)])
        builder = DisruptionEnvGroupBuilder(
            record=record,
            group_size=self.group_size,
            renderer=self.renderer,
            system_prompt=self.system_prompt,
            max_env_tokens=self.max_env_tokens,
            prompt_max_chars=self.prompt_max_chars,
            include_concepts=self.include_concepts,
        )
        return builder.build_prompt_text()


def build_batch_histogram_summary(
    batch_histograms: Sequence[Mapping[str, int]],
) -> Dict[str, Any]:
    total = defaultdict(int)
    for row in batch_histograms:
        for label in DISRUPTION_LABELS:
            total[label] += int(row.get(label, 0))
    return {
        "num_batches": len(batch_histograms),
        "aggregate_label_counts": {label: int(total[label]) for label in DISRUPTION_LABELS},
        "per_batch": [dict(row) for row in batch_histograms],
    }


__all__ = [
    "DisruptionEnvGroupBuilder",
    "SciSciNetRLDataset",
    "build_batch_histogram_summary",
    "label_histogram",
    "load_split_records_streaming",
]
