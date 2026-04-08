# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import pathlib
import logging
import polars as pl

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

src = pathlib.Path("../../data/sci_balanced_from2m_no_ovr.rl_balanced.jsonl")
dst = pathlib.Path("../../data/sci_balanced_from2m_no_ovr.rl_balanced.with_teacher_confidence.jsonl")

cd = pl.col("cd_index")
label = pl.col("disruption_label")

teacher_confidence = (
  pl.when(cd.is_null())
  .then(pl.lit("medium"))
  .when(label == "disruptive")
  .then(
      pl.when(cd >= 0.05)
      .then(pl.lit("high"))
      .when(cd >= 0.01)
      .then(pl.lit("medium"))
      .otherwise(pl.lit("low"))
  )
  .when(label == "consolidating")
  .then(
      pl.when(cd <= -0.05)
      .then(pl.lit("high"))
      .when(cd <= -0.01)
      .then(pl.lit("medium"))
      .otherwise(pl.lit("low"))
  )
  .otherwise(
      pl.when(cd.abs() <= 0.00025)
      .then(pl.lit("high"))
      .when(cd.abs() <= 0.00075)
      .then(pl.lit("medium"))
      .otherwise(pl.lit("low"))
  )
  .alias("teacher_confidence")
)

preview = (
  pl.scan_ndjson(src)
  .with_columns(teacher_confidence)
  .select(["openalex_id", "disruption_label", "cd_index", "teacher_confidence"])
  .limit(10)
  .collect()
)
print(preview)

(
  pl.scan_ndjson(src)
  .with_columns(teacher_confidence)
  .sink_ndjson(dst)
)

print(f"wrote {dst}")


