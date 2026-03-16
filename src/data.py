from __future__ import annotations

# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import chz
import polars as pl
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import Env as CookbookEnv
from tinker_cookbook.rl.types import EnvGroupBuilder as CookbookEnvGroupBuilder
from tinker_cookbook.rl.types import RLDataset as CookbookRLDataset
from tinker_cookbook.rl.types import RLDatasetBuilder as CookbookRLDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

LABELS = ("disruptive", "consolidating", "neutral")
# Default to a dense instruction model to keep the launch path simple.
LLM = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_RENDERER_NAME = "llama3"
DEFAULT_BATCH_SIZE = 16
DEFAULT_GROUP_SIZE = 2
DEFAULT_TRAIN_SIZE = 100_000
DEFAULT_TEST_SIZE = 2_000
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sci_balanced_from2m_no_ovr.rl_balanced.jsonl"
)
DATA_COLUMNS = [
    "title",
    "abstract",
    "publication_year",
    "cited_by_count",
    "primary_field",
    "disruption_label",
]


@dataclass
class PaperRecord:
    title: str
    abstract: str
    year: int | None = None
    citations: int | None = None
    field: str | None = None
    gold_label: str | None = None


def load_rl_records(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    limit: int | None = None,
    seed: int = 0,
) -> list[PaperRecord]:
    lf = (
        pl.scan_ndjson(Path(dataset_path))
        .select(DATA_COLUMNS)
        .filter(pl.col("disruption_label").is_in(LABELS))
    )
    if limit is not None:
        lf = lf.head(limit)

    df = lf.collect(engine="streaming")
    if len(df) > 1:
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    records: list[PaperRecord] = []
    for row in df.to_dicts():
        label = row.get("disruption_label")
        if label not in LABELS:
            continue

        records.append(
            PaperRecord(
                title=row["title"],
                abstract=row["abstract"],
                year=row.get("publication_year"),
                citations=row.get("cited_by_count"),
                field=row.get("primary_field"),
                gold_label=label,
            )
        )

    return records


@dataclass(frozen=True)
class DisruptionEnvGroupBuilder(CookbookEnvGroupBuilder):
    record: PaperRecord
    renderer: renderers.Renderer
    num_envs: int

    async def make_envs(self) -> Sequence[CookbookEnv]:
        from util import DisruptionRLEnv

        return [
            DisruptionRLEnv(record=self.record, renderer=self.renderer)
            for _ in range(self.num_envs)
        ]


@dataclass(frozen=True)
class DisruptionRLDataset(CookbookRLDataset):
    records: list[PaperRecord]
    renderer: renderers.Renderer
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[CookbookEnvGroupBuilder]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self.records))
        return [
            DisruptionEnvGroupBuilder(
                record=record,
                renderer=self.renderer,
                num_envs=self.group_size,
            )
            for record in self.records[start:stop]
        ]

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


@chz.chz
class DisruptionDatasetBuilder(CookbookRLDatasetBuilder):
    dataset_path: str
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str = DEFAULT_RENDERER_NAME
    train_size: int = DEFAULT_TRAIN_SIZE
    test_size: int = DEFAULT_TEST_SIZE
    seed: int = 0

    async def __call__(self) -> tuple[CookbookRLDataset, CookbookRLDataset]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        total_rows = self.train_size + self.test_size
        records = load_rl_records(self.dataset_path, limit=total_rows, seed=self.seed)

        if len(records) < total_rows:
            raise ValueError(
                f"Need at least {total_rows} labeled records, found {len(records)}"
            )

        train_records = records[: self.train_size]
        test_records = records[self.train_size : total_rows]
        test_batch_size = max(1, min(self.batch_size, len(test_records)))

        train_ds = DisruptionRLDataset(
            records=train_records,
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )
        test_ds = DisruptionRLDataset(
            records=test_records,
            renderer=renderer,
            batch_size=test_batch_size,
            group_size=1,
        )
        return train_ds, test_ds
