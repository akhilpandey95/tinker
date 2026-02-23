#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

"""Post-enrich SciSciNet JSONL `primary_field` / `concepts` from paper-field parquet."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
os.environ.setdefault("POLARS_FORCE_NEW_STREAMING", "1")

REQUIRED_BASE_COLUMNS = ("openalex_id", "primary_field", "concepts")
PLACEHOLDER_PRIMARYS = {"article", "unknown"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrich SciSciNet JSONL field metadata from paper-field parquet.")
    p.add_argument("--base-jsonl", type=Path, required=True, help="Existing disruption/novelty JSONL to enrich.")
    p.add_argument("--paper-fields-parquet", type=Path, required=True, help="Paper->field hierarchy parquet (alpha).")
    p.add_argument("--output-jsonl", type=Path, required=True, help="New enriched JSONL output path (base is not overwritten).")
    p.add_argument("--report-output", type=Path, help="Optional enrichment report JSON path (defaults from output).")
    p.add_argument("--concepts-k", type=int, default=8, help="Max number of concepts to retain per paper.")
    p.add_argument(
        "--include-primary-in-concepts",
        action="store_true",
        help="Keep selected primary field inside the concepts list (default: remove duplicate primary when possible).",
    )
    p.add_argument("--base-limit", type=int, help="Optional smoke-test limit on base JSONL rows before enrichment.")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--row-log-interval", type=int, default=50000)
    p.add_argument("--example-limit", type=int, default=8)
    p.add_argument("--infer-schema-length", type=int, default=10000, help="NDJSON schema inference rows for Polars scan.")
    a = p.parse_args()
    if not a.base_jsonl.exists():
        raise ValueError(f"--base-jsonl not found: {a.base_jsonl}")
    if not a.paper_fields_parquet.exists():
        raise ValueError(f"--paper-fields-parquet not found: {a.paper_fields_parquet}")
    if a.concepts_k <= 0:
        raise ValueError("--concepts-k must be > 0")
    if a.base_limit is not None and a.base_limit <= 0:
        raise ValueError("--base-limit must be > 0 when provided")
    if a.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if a.row_log_interval <= 0:
        raise ValueError("--row-log-interval must be > 0")
    if a.example_limit < 0:
        raise ValueError("--example-limit must be >= 0")
    a.report_output = a.report_output or a.output_jsonl.with_suffix(".enrichment_report.json")
    return a


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def text(v: Any) -> str:
    s = "" if v is None else str(v).strip()
    return "" if (not s or s.lower() == "nan") else s


def to_int(v: Any, d: int | None = None) -> int | None:
    try:
        if v is None:
            return d
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return d


def normalize_str_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        vals = [text(x) for x in raw]
    else:
        vals = [text(x) for x in text(raw).replace("|", ",").split(",")]
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def placeholder_primary(s: str) -> bool:
    return text(s).lower() in PLACEHOLDER_PRIMARYS


def placeholder_concepts(vals: Sequence[str]) -> bool:
    xs = [text(x) for x in vals if text(x)]
    return len(xs) == 1 and placeholder_primary(xs[0])


def dedupe_keep_order(vals: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        s = text(v)
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def materialize_concepts(
    concepts_candidates: Any,
    primary_field: str,
    *,
    concepts_k: int,
    include_primary_in_concepts: bool,
    fallback_existing: Sequence[str],
) -> list[str]:
    raw_vals = dedupe_keep_order(concepts_candidates if concepts_candidates is not None else [])
    vals = list(raw_vals)
    if not include_primary_in_concepts and primary_field:
        vals = [v for v in vals if v.lower() != primary_field.lower()]
    vals = vals[: int(concepts_k)]
    if vals:
        return vals
    if raw_vals and primary_field:
        # If the only enriched concept is the chosen primary, keep that signal rather than falling back to placeholders.
        return [primary_field]
    existing = dedupe_keep_order(fallback_existing)
    if existing:
        return existing[: int(concepts_k)]
    return [primary_field] if primary_field else ["Unknown"]


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def import_polars():
    try:
        import polars as pl  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Polars is required for enrichment (`pip install polars pyarrow`). "
            f"Import failed: {e}"
        ) from e
    return pl


def collect_schema_names(lf: Any) -> list[str]:
    try:
        return list(lf.collect_schema().names())
    except Exception:
        schema = getattr(lf, "schema", None)
        if isinstance(schema, Mapping):
            return list(schema.keys())
        raise


def scan_ndjson_compat(pl: Any, path: Path, infer_schema_length: int) -> Any:
    if not hasattr(pl, "scan_ndjson"):
        raise RuntimeError(
            "This Polars version does not support `scan_ndjson`. "
            "Please upgrade Polars to use lazy/memory-safe enrichment."
        )
    for kwargs in (
        {"infer_schema_length": int(infer_schema_length)},
        {},
    ):
        try:
            return pl.scan_ndjson(str(path), **kwargs)
        except TypeError:
            continue
    return pl.scan_ndjson(str(path))


def with_row_index_compat(lf: Any, name: str) -> Any:
    if hasattr(lf, "with_row_index"):
        try:
            return lf.with_row_index(name)
        except TypeError:
            pass
    return lf.with_row_count(name)


def iter_batches(lf: Any, batch_size: int):
    if hasattr(lf, "collect_batches"):
        for kw in ({"engine": "streaming"}, {"streaming": True}, {}):
            try:
                try:
                    return lf.collect_batches(chunk_size=int(batch_size), **kw)
                except TypeError:
                    return lf.collect_batches(**kw)
            except (TypeError, ValueError):
                continue
    raise RuntimeError(
        "Polars lazy batch collection is unavailable. "
        "Upgrade Polars to a version that supports `LazyFrame.collect_batches(...)`."
    )


def collect_df(lf: Any) -> Any:
    for kw in ({"engine": "streaming"}, {"streaming": True}, {}):
        try:
            return lf.collect(**kw)
        except (TypeError, ValueError):
            continue
    return lf.collect()


def scalar_from_lf(lf: Any) -> Any:
    df = collect_df(lf)
    if getattr(df, "height", 0) != 1 or getattr(df, "width", 0) != 1:
        raise RuntimeError("Expected a 1x1 scalar collect")
    try:
        return df.item()
    except Exception:
        return df.to_series(0).item()


def build_base_lazy(pl: Any, args: argparse.Namespace) -> tuple[Any, list[str]]:
    base = scan_ndjson_compat(pl, args.base_jsonl, args.infer_schema_length)
    base_cols = collect_schema_names(base)
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in base_cols]
    if missing:
        raise ValueError(f"Base JSONL missing required columns: {missing}")
    base = with_row_index_compat(base, "row_idx").with_columns(
        pl.col("openalex_id").cast(pl.Utf8, strict=False).alias("openalex_id")
    )
    if args.base_limit:
        base = base.limit(int(args.base_limit))
    return base, base_cols


def build_enrichment_lazy(pl: Any, args: argparse.Namespace, base_ids_df: Any) -> Any:
    base_ids = base_ids_df.lazy().select(pl.col("openalex_id").cast(pl.Utf8, strict=False).alias("paperid"))

    pf = pl.scan_parquet(str(args.paper_fields_parquet)).select(
        [
            pl.col("paperid").cast(pl.Utf8, strict=False).alias("paperid"),
            pl.col("fieldid").cast(pl.Utf8, strict=False).alias("fieldid"),
            pl.col("display_name").cast(pl.Utf8, strict=False).alias("display_name"),
            pl.col("level").cast(pl.Int64, strict=False).alias("level"),
        ]
    )
    pf = pf.with_columns(
        [
            pl.col("paperid").fill_null("").str.strip_chars().alias("paperid"),
            pl.col("fieldid").fill_null("").str.strip_chars().alias("fieldid"),
            pl.col("display_name").fill_null("").str.strip_chars().alias("display_name"),
        ]
    ).filter((pl.col("paperid").str.len_bytes() > 0) & (pl.col("display_name").str.len_bytes() > 0))

    relevant = pf.join(base_ids, on="paperid", how="semi")

    # Collapse duplicate display names per paper, retaining the best level and deterministic fieldid tie-break.
    per_name = relevant.group_by(["paperid", "display_name"]).agg(
        [
            pl.col("level").min().alias("best_level"),
            pl.col("fieldid").sort().first().alias("best_fieldid"),
        ]
    )
    per_name = per_name.with_columns(
        [
            pl.col("display_name").str.to_lowercase().alias("display_name_lc"),
            pl.when(pl.col("best_level").is_null())
            .then(pl.lit(1_000_000))
            .otherwise(pl.col("best_level"))
            .cast(pl.Int64)
            .alias("best_level_sort"),
            pl.when(pl.col("best_level") == 0).then(pl.lit(0)).otherwise(pl.lit(1)).alias("primary_pref"),
            pl.col("best_fieldid").fill_null("").alias("best_fieldid"),
        ]
    )

    enrich = per_name.group_by("paperid").agg(
        [
            pl.col("display_name")
            .sort_by(["primary_pref", "best_level_sort", "display_name_lc", "best_fieldid"])
            .first()
            .alias("enriched_primary_field"),
            pl.col("display_name")
            .sort_by(["best_level_sort", "display_name_lc", "best_fieldid"])
            .alias("concepts_candidates"),
            (pl.col("best_level") == 0).any().alias("has_level0_assignment"),
            pl.col("display_name").count().alias("unique_concept_count_pre_trunc"),
        ]
    )
    return enrich.with_columns(pl.lit(True).alias("has_any_field_assignment"))


def choose_primary_field(enriched: Any, existing: Any) -> str:
    return text(enriched) or text(existing) or "Unknown"


def process_row(
    row: Mapping[str, Any],
    *,
    base_columns: Sequence[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_primary = text(row.get("primary_field"))
    old_concepts = normalize_str_list(row.get("concepts"))

    new_primary = choose_primary_field(row.get("enriched_primary_field"), old_primary)
    new_concepts = materialize_concepts(
        row.get("concepts_candidates"),
        new_primary,
        concepts_k=int(args.concepts_k),
        include_primary_in_concepts=bool(args.include_primary_in_concepts),
        fallback_existing=old_concepts,
    )

    out: dict[str, Any] = {}
    for c in base_columns:
        if c == "primary_field":
            out[c] = new_primary
        elif c == "concepts":
            out[c] = [str(x) for x in new_concepts]
        else:
            out[c] = row.get(c)

    meta = {
        "old_primary": old_primary,
        "new_primary": new_primary,
        "old_concepts": old_concepts,
        "new_concepts": new_concepts,
        "has_any_field_assignment": bool(row.get("has_any_field_assignment") or False),
        "has_level0_assignment": bool(row.get("has_level0_assignment") or False),
    }
    return out, meta


def enrich(args: argparse.Namespace) -> dict[str, Any]:
    pl = import_polars()
    logger.info("Scanning base JSONL: %s", args.base_jsonl)
    base_lf, base_columns = build_base_lazy(pl, args)

    logger.info("Collecting base IDs for semi-join filter")
    base_ids_df = collect_df(base_lf.select("openalex_id").unique())
    base_row_count = int(scalar_from_lf(base_lf.select(pl.len())))
    unique_base_ids = int(getattr(base_ids_df, "height", 0))
    logger.info("Base rows=%d unique_ids=%d", base_row_count, unique_base_ids)

    logger.info("Building paper-field enrichment lazy plan from %s", args.paper_fields_parquet)
    enrich_lf = build_enrichment_lazy(pl, args, base_ids_df)

    # Select base columns explicitly to preserve schema ordering and append helper columns for validation/reporting.
    joined = (
        base_lf.select([pl.col(c) for c in base_columns] + [pl.col("row_idx")])
        .join(enrich_lf, left_on="openalex_id", right_on="paperid", how="left")
        .with_columns(
            [
                pl.col("has_any_field_assignment").fill_null(False).alias("has_any_field_assignment"),
                pl.col("has_level0_assignment").fill_null(False).alias("has_level0_assignment"),
            ]
        )
        .sort("row_idx")
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "started_at_utc": now_utc_iso(),
        "base_jsonl": str(args.base_jsonl),
        "paper_fields_parquet": str(args.paper_fields_parquet),
        "output_jsonl": str(args.output_jsonl),
        "report_output": str(args.report_output),
        "base_limit": int(args.base_limit) if args.base_limit else None,
        "concepts_k": int(args.concepts_k),
        "include_primary_in_concepts": bool(args.include_primary_in_concepts),
        "base_row_count": int(base_row_count),
        "base_unique_openalex_ids": int(unique_base_ids),
        "output_row_count": 0,
        "with_any_field_assignment_count": 0,
        "with_level0_assignment_count": 0,
        "primary_field_changed_count": 0,
        "concepts_changed_count": 0,
        "placeholder_primary_before_count": 0,
        "placeholder_primary_after_count": 0,
        "placeholder_concepts_before_count": 0,
        "placeholder_concepts_after_count": 0,
        "row_idx_contiguous": True,
        "row_idx_first": None,
        "row_idx_last": None,
        "examples": [],
        "notes": [
            "primary_field tie-break: prefer level==0, else lowest level, then display_name alphabetical, then fieldid",
            f"concepts ordering: unique display_name sorted by level/display_name/fieldid, truncated to top {int(args.concepts_k)}",
            "concepts exclude selected primary by default (use --include-primary-in-concepts to disable)",
        ],
    }

    expected_row_idx = 0
    logger.info("Writing enriched JSONL to %s", args.output_jsonl)
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for batch in iter_batches(joined, args.batch_size):
            for row in batch.iter_rows(named=True):
                row_idx = to_int(row.get("row_idx"), -1)
                if stats["row_idx_first"] is None:
                    stats["row_idx_first"] = row_idx
                stats["row_idx_last"] = row_idx
                if row_idx != expected_row_idx:
                    stats["row_idx_contiguous"] = False
                expected_row_idx += 1

                out_row, meta = process_row(row, base_columns=base_columns, args=args)
                out_f.write(json.dumps(out_row, ensure_ascii=True))
                out_f.write("\n")

                stats["output_row_count"] += 1
                if meta["has_any_field_assignment"]:
                    stats["with_any_field_assignment_count"] += 1
                if meta["has_level0_assignment"]:
                    stats["with_level0_assignment_count"] += 1
                if meta["old_primary"] != meta["new_primary"]:
                    stats["primary_field_changed_count"] += 1
                if meta["old_concepts"] != meta["new_concepts"]:
                    stats["concepts_changed_count"] += 1
                if placeholder_primary(meta["old_primary"]):
                    stats["placeholder_primary_before_count"] += 1
                if placeholder_primary(meta["new_primary"]):
                    stats["placeholder_primary_after_count"] += 1
                if placeholder_concepts(meta["old_concepts"]):
                    stats["placeholder_concepts_before_count"] += 1
                if placeholder_concepts(meta["new_concepts"]):
                    stats["placeholder_concepts_after_count"] += 1

                if (
                    len(stats["examples"]) < int(args.example_limit)
                    and (meta["old_primary"] != meta["new_primary"] or meta["old_concepts"] != meta["new_concepts"])
                ):
                    stats["examples"].append(
                        {
                            "openalex_id": text(out_row.get("openalex_id")),
                            "has_any_field_assignment": meta["has_any_field_assignment"],
                            "has_level0_assignment": meta["has_level0_assignment"],
                            "primary_field_before": meta["old_primary"],
                            "primary_field_after": meta["new_primary"],
                            "concepts_before": meta["old_concepts"][:12],
                            "concepts_after": meta["new_concepts"][:12],
                        }
                    )

                if stats["output_row_count"] % int(args.row_log_interval) == 0:
                    logger.info(
                        "Wrote %d rows (coverage any=%.2f%% level0=%.2f%%)",
                        stats["output_row_count"],
                        100.0 * stats["with_any_field_assignment_count"] / max(1, stats["output_row_count"]),
                        100.0 * stats["with_level0_assignment_count"] / max(1, stats["output_row_count"]),
                    )

    if int(stats["output_row_count"]) != int(stats["base_row_count"]):
        raise RuntimeError(
            f"Row count mismatch after enrichment: base={stats['base_row_count']} output={stats['output_row_count']}"
        )

    n = max(1, int(stats["output_row_count"]))
    stats["with_any_field_assignment_pct"] = round(100.0 * stats["with_any_field_assignment_count"] / n, 4)
    stats["with_level0_assignment_pct"] = round(100.0 * stats["with_level0_assignment_count"] / n, 4)
    stats["primary_field_changed_pct"] = round(100.0 * stats["primary_field_changed_count"] / n, 4)
    stats["concepts_changed_pct"] = round(100.0 * stats["concepts_changed_count"] / n, 4)
    stats["placeholder_primary_before_pct"] = round(100.0 * stats["placeholder_primary_before_count"] / n, 4)
    stats["placeholder_primary_after_pct"] = round(100.0 * stats["placeholder_primary_after_count"] / n, 4)
    stats["placeholder_concepts_before_pct"] = round(100.0 * stats["placeholder_concepts_before_count"] / n, 4)
    stats["placeholder_concepts_after_pct"] = round(100.0 * stats["placeholder_concepts_after_count"] / n, 4)
    stats["order_preserved"] = bool(stats["row_idx_contiguous"]) and int(stats["row_idx_first"] or 0) == 0 and int(
        stats["row_idx_last"] or -1
    ) == int(stats["output_row_count"]) - 1
    stats["completed_at_utc"] = now_utc_iso()
    stats["duration_note"] = "Runtime/memory depend on Polars version, thread count, and disk throughput; use /usr/bin/time -v on the target host."

    dump_json(args.report_output, stats)
    logger.info(
        "Completed enrichment rows=%d any_field=%d (%.2f%%) level0=%d (%.2f%%) order_preserved=%s",
        stats["output_row_count"],
        stats["with_any_field_assignment_count"],
        stats["with_any_field_assignment_pct"],
        stats["with_level0_assignment_count"],
        stats["with_level0_assignment_pct"],
        stats["order_preserved"],
    )
    return stats


def main() -> None:
    args = parse_args()
    stats = enrich(args)
    print(
        "field_enrichment "
        f"rows={stats['output_row_count']} "
        f"any_field_pct={stats['with_any_field_assignment_pct']} "
        f"level0_pct={stats['with_level0_assignment_pct']} "
        f"order_preserved={stats['order_preserved']} "
        f"output={args.output_jsonl}"
    )
    print(f"report={args.report_output}")


if __name__ == "__main__":
    main()
