#!/usr/bin/env python3
"""Slim local-only SciSciNet parquet dataset builder (JSONL + splits + metadata)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_FIELDS = ("openalex_id", "title", "abstract", "publication_year", "cited_by_count", "cd_index", "novelty_score", "conventionality_score", "disruption_label", "novelty_label", "primary_field", "concepts")
PARQUET_COLUMNS = ("paperid", "year", "doctype", "cited_by_count", "citation_count", "is_retracted", "language", "title", "abstract", "abstract_inverted_index", "primary_field", "concepts", "cd_index", "disruption", "novelty_score", "conventionality_score", "Atyp_Median_Z", "Atyp_10pct_Z")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Prefer Polars' newer streaming engine for large parquet scans.
os.environ.setdefault("POLARS_FORCE_NEW_STREAMING", "1")


@dataclass(frozen=True)
class LabelThresholds:
    disruptive_min: float = 0.1
    consolidating_max: float = -0.1
    novelty_margin: float = 0.15


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for local SciSciNet parquet dataset generation.

    Parameters
    ------------
        None

    Returns
    ------------
    argparse.Namespace
        Parsed arguments for parquet input, filtering, outputs, and split config.
    """
    p = argparse.ArgumentParser(description="Build disruption/novelty dataset from local SciSciNet parquet.")
    p.add_argument("--sciscinet-parquet", type=Path); p.add_argument("--tinker-sciscinet-parquet", type=Path)
    p.add_argument("--parquet-primary", choices=("sciscinet", "tinker"), default="tinker")
    p.add_argument("--n-papers", type=int, default=200); p.add_argument("--seed", type=int, default=20260220)
    p.add_argument("--output", type=Path, default=Path("data/disruption_novelty_dataset.jsonl"))
    p.add_argument("--splits-output", type=Path); p.add_argument("--metadata-output", type=Path)
    p.add_argument("--train-ratio", type=float, default=0.8); p.add_argument("--val-ratio", type=float, default=0.1); p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--disruptive-threshold", type=float, default=0.1); p.add_argument("--consolidating-threshold", type=float, default=-0.1); p.add_argument("--novelty-margin", type=float, default=0.15)
    p.add_argument("--sciscinet-from-year", type=int); p.add_argument("--sciscinet-to-year", type=int); p.add_argument("--sciscinet-min-citations", type=int)
    p.add_argument("--include-retracted", action="store_true"); p.add_argument("--sciscinet-language")
    p.add_argument("--batch-size", type=int, default=5000); p.add_argument("--row-log-interval", type=int, default=50000)
    a = p.parse_args()
    if not (a.sciscinet_parquet or a.tinker_sciscinet_parquet): raise ValueError("Provide --sciscinet-parquet and/or --tinker-sciscinet-parquet")
    for k in ("sciscinet_parquet", "tinker_sciscinet_parquet"):
        if getattr(a, k) and not getattr(a, k).exists(): raise ValueError(f"--{k.replace('_','-')} not found: {getattr(a, k)}")
    if a.n_papers <= 0: raise ValueError("--n-papers must be > 0")
    if a.consolidating_threshold >= a.disruptive_threshold: raise ValueError("--consolidating-threshold must be < --disruptive-threshold")
    if a.sciscinet_from_year and a.sciscinet_to_year and a.sciscinet_from_year > a.sciscinet_to_year: raise ValueError("--sciscinet-from-year must be <= --sciscinet-to-year")
    if a.train_ratio + a.val_ratio + a.test_ratio <= 0: raise ValueError("Split ratios must sum to > 0")
    a.sciscinet_language = (str(a.sciscinet_language).strip().lower() or None) if a.sciscinet_language else None
    a.splits_output = a.splits_output or a.output.with_suffix(".splits.json")
    a.metadata_output = a.metadata_output or a.output.with_suffix(".metadata.json")
    return a


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clamp(x: float, lo: float, hi: float) -> float: return max(lo, min(hi, x))
def text(v: Any) -> str: s = "" if v is None else str(v).strip(); return "" if (not s or s.lower() == "nan") else s


def to_int(v: Any, d: int = 0) -> int:
    try:
        if isinstance(v, bool): return int(v)
        if v is None or (isinstance(v, float) and not math.isfinite(v)): return d
        return int(float(v))
    except (TypeError, ValueError):
        return d


def to_float(v: Any, d: float = float("nan")) -> float:
    try:
        if isinstance(v, bool): return float(int(v))
        x = d if v is None else float(v)
        return x if math.isfinite(x) else d
    except (TypeError, ValueError):
        return d


def label_disruption(cd: float, t: LabelThresholds) -> str:
    return "disruptive" if cd > t.disruptive_min else ("consolidating" if cd < t.consolidating_max else "neutral")


def label_novelty(n: float, c: float, t: LabelThresholds) -> str:
    delta = n - c
    return "novel" if delta > t.novelty_margin else ("conventional" if delta < -t.novelty_margin else "balanced")


def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid helper.

    Parameters
    ------------
    x: float
        Input scalar.

    Returns
    ------------
    float
        Sigmoid-transformed value in `[0, 1]`.
    """
    z = math.exp(-x) if x >= 0 else math.exp(x)
    return (1.0 / (1.0 + z)) if x >= 0 else (z / (1.0 + z))


def year_from(row: Mapping[str, Any]) -> int:
    """
    Read publication year from a curated parquet row.

    Parameters
    ------------
    row: Mapping[str, Any]
        A single parquet row.

    Returns
    ------------
    int
        Publication year.
    """
    return to_int(row.get("year"), 0)


def concepts_from(raw: Any, primary: str, doctype: str) -> list[str]:
    """
    Normalize concepts into a short unique list of strings.

    Parameters
    ------------
    raw: Any
        Concepts column value from parquet (list-like or delimited string).
    primary: str
        Primary field fallback if concepts are empty.
    doctype: str
        Document type appended as a lightweight concept tag.

    Returns
    ------------
    list[str]
        Up to 8 unique concept strings.
    """
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = [text(x) for x in raw if text(x)]
    else:
        items = [text(x) for x in text(raw).replace("|", ",").split(",") if text(x)] if raw is not None else []
    if not items and primary and primary != "Unknown": items.append(primary)
    if doctype: items.append(doctype)
    out, seen = [], set()
    for x in items:
        k = x.lower()
        if k not in seen: seen.add(k); out.append(x)
    return out[:8] or ["Unknown"]


def decode_abstract_inverted_index(raw: Any) -> str:
    """
    Decode a SciSciNet/OpenAlex-style abstract inverted index payload.

    Parameters
    ------------
    raw: Any
        JSON string payload (or already-decoded mapping) of token -> positions.

    Returns
    ------------
    str
        Reconstructed abstract text, or empty string on parse failure.
    """
    if raw is None:
        return ""
    payload: Any = raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return ""
        try:
            payload = json.loads(s)
        except Exception:
            return ""
    if not isinstance(payload, Mapping):
        return ""

    slots: dict[int, str] = {}
    max_pos = -1
    for token, positions in payload.items():
        tok = text(token)
        if not tok or not isinstance(positions, Sequence) or isinstance(positions, (str, bytes, bytearray)):
            continue
        for pos in positions:
            i = to_int(pos, -1)
            if i < 0:
                continue
            if i not in slots:
                slots[i] = tok
            if i > max_pos:
                max_pos = i
    if max_pos < 0:
        return ""
    return " ".join(slots.get(i, "") for i in range(max_pos + 1)).strip()


def novelty_from_row(row: Mapping[str, Any]) -> tuple[float, float] | None:
    """
    Read novelty/conventionality scores from curated columns, or derive from Atyp metrics.

    Parameters
    ------------
    row: Mapping[str, Any]
        A single parquet row.

    Returns
    ------------
    tuple[float, float] | None
        `(novelty_score, conventionality_score)` or `None` if unavailable.
    """
    n = to_float(row.get("novelty_score"))
    c = to_float(row.get("conventionality_score"))
    if math.isfinite(n) and math.isfinite(c):
        return clamp(float(n), 0.0, 1.0), clamp(float(c), 0.0, 1.0)

    atyp10 = to_float(row.get("Atyp_10pct_Z"))
    atypm = to_float(row.get("Atyp_Median_Z"))
    if math.isfinite(atyp10) and math.isfinite(atypm):
        n = clamp(sigmoid(clamp(-float(atyp10), -8.0, 8.0)), 0.0, 1.0)
        c = clamp(sigmoid(clamp(float(atypm), -8.0, 8.0)), 0.0, 1.0)
        return n, c
    return None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    s, t = args.sciscinet_parquet, args.tinker_sciscinet_parquet
    primary = (s or t) if args.parquet_primary == "sciscinet" else (t or s)
    secondary = (t if (s and t and args.parquet_primary == "sciscinet") else (s if (s and t and args.parquet_primary == "tinker") else None))
    if primary is None: raise ValueError("No parquet inputs")
    if secondary == primary: secondary = None
    return Path(primary), (Path(secondary) if secondary else None)


def build_lazyframe(args: argparse.Namespace) -> tuple[Any, Any]:
    """
    Build the filtered Polars lazyframe from one or two local parquet inputs.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI arguments containing parquet paths and row filters.

    Returns
    ------------
    tuple[Any, Any]
        `(polars_module, filtered_lazyframe_limited_to_n_papers)`.
    """
    import polars as pl  # type: ignore
    primary, secondary = resolve_paths(args)
    logger.info("Scanning parquet (primary=%s secondary=%s)", primary, secondary)

    def scan_prefixed(path: Path, prefix: str) -> tuple[Any, set[str]]:
        lf = pl.scan_parquet(str(path)); cols = set(lf.collect_schema().names())
        if "paperid" not in cols: raise ValueError(f"Missing `paperid` in {path}")
        return lf.select([pl.col(c).alias(f"{prefix}{c}") for c in sorted(cols)]), cols

    left, left_cols = scan_prefixed(primary, "l_"); merged = left; right_cols: set[str] = set()
    if secondary:
        right, right_cols = scan_prefixed(secondary, "r_")
        merged = left.join(right.unique(subset=["r_paperid"], keep="first"), left_on="l_paperid", right_on="r_paperid", how="left")

    def pick(c: str) -> Any:
        exprs = ([pl.col(f"l_{c}")] if c in left_cols else []) + ([pl.col(f"r_{c}")] if c in right_cols else [])
        return (pl.coalesce(exprs) if exprs else pl.lit(None)).alias(c)

    lf = merged.select([pick(c) for c in PARQUET_COLUMNS]).filter(pl.col("paperid").cast(pl.Utf8, strict=False).fill_null("").str.len_bytes() > 0)
    if args.sciscinet_from_year is not None: lf = lf.filter(pl.col("year").cast(pl.Int64, strict=False) >= int(args.sciscinet_from_year))
    if args.sciscinet_to_year is not None: lf = lf.filter(pl.col("year").cast(pl.Int64, strict=False) <= int(args.sciscinet_to_year))
    if args.sciscinet_min_citations is not None:
        cites = pl.coalesce([pl.col("cited_by_count"), pl.col("citation_count"), pl.lit(0)])
        lf = lf.filter(cites.cast(pl.Int64, strict=False).fill_null(0) >= int(args.sciscinet_min_citations))
    if not args.include_retracted: lf = lf.filter(~pl.col("is_retracted").cast(pl.Boolean, strict=False).fill_null(False))
    if args.sciscinet_language: lf = lf.filter(pl.col("language").cast(pl.Utf8, strict=False).str.to_lowercase() == args.sciscinet_language)
    return pl, lf.limit(int(args.n_papers))


def iter_batches(lf: Any, batch_size: int):
    """
    Yield Polars record batches using lazy/streaming execution.

    Parameters
    ------------
    lf: Any
        Polars LazyFrame.
    batch_size: int
        Desired batch size for batch collection APIs.

    Returns
    ------------
    iterable
        Iterable of Polars DataFrames/batches.
    """
    if hasattr(lf, "collect_batches"):
        for kw in ({"engine": "streaming"}, {"streaming": True}, {}):
            try:
                try: return lf.collect_batches(chunk_size=int(batch_size), **kw)
                except TypeError: return lf.collect_batches(**kw)
            except (TypeError, ValueError):
                pass
    raise RuntimeError(
        "Polars lazy batch collection is unavailable in this environment. "
        "Upgrade Polars to a version that supports `LazyFrame.collect_batches(...)` "
        "to avoid full in-memory collect on large datasets."
    )


def row_to_record(row: Mapping[str, Any], t: LabelThresholds) -> dict[str, Any] | None:
    """
    Convert one curated parquet row into the standardized ML-ready example.

    Parameters
    ------------
    row: Mapping[str, Any]
        A single parquet row from the filtered SciSciNet/Tinker source.
    t: LabelThresholds
        Thresholds used to map continuous scores into class labels.

    Returns
    ------------
    dict[str, Any] | None
        Normalized record ready for JSONL output, or `None` if the row is unusable.
    """
    pid = text(row.get("paperid"))
    if not pid: return None
    year = year_from(row)
    cited = max(0, to_int(row.get("cited_by_count"), to_int(row.get("citation_count"), 0)))
    cd = to_float(row.get("cd_index"))
    if not math.isfinite(cd): cd = to_float(row.get("disruption"))
    nc = novelty_from_row(row)
    if nc is None:
        return None
    novelty, conv = nc
    title = text(row.get("title"))
    abstract = text(row.get("abstract")) or decode_abstract_inverted_index(row.get("abstract_inverted_index"))
    if not title or not abstract or not math.isfinite(cd): return None
    cd = clamp(float(cd), -1.0, 1.0)
    doctype = text(row.get("doctype"))
    primary = text(row.get("primary_field")) or (doctype.title() if doctype else "Unknown")
    concepts = concepts_from(row.get("concepts"), primary, doctype)
    cd, novelty, conv = round(float(cd), 4), round(float(novelty), 4), round(float(conv), 4)
    return {"openalex_id": pid, "title": title, "abstract": abstract, "publication_year": int(year), "cited_by_count": int(cited), "cd_index": cd, "novelty_score": novelty, "conventionality_score": conv, "disruption_label": label_disruption(cd, t), "novelty_label": label_novelty(novelty, conv, t), "primary_field": primary, "concepts": [str(x) for x in concepts]}


def write_jsonl_and_collect_stats(args: argparse.Namespace, t: LabelThresholds) -> tuple[int, dict[str, dict[str, int]], list[str]]:
    """
    Write the normalized dataset JSONL and collect lightweight summary stats.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI arguments.
    t: LabelThresholds
        Thresholds used to compute labels.

    Returns
    ------------
    tuple[int, dict[str, dict[str, int]], list[str]]
        `(record_count, label_counts, paper_ids)` for downstream splits/metadata.
    """
    _, lf = build_lazyframe(args); args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"disruption": {"disruptive": 0, "neutral": 0, "consolidating": 0}, "novelty": {"novel": 0, "balanced": 0, "conventional": 0}}
    ids: list[str] = []; n = 0; logger.info("Writing JSONL to %s", args.output)
    with args.output.open("w", encoding="utf-8") as f:
        for batch in iter_batches(lf, args.batch_size):
            for row in batch.iter_rows(named=True):
                rec = row_to_record(row, t)
                if rec is None: continue
                f.write(json.dumps(rec, ensure_ascii=True)); f.write("\n")
                ids.append(str(rec["openalex_id"])); counts["disruption"][rec["disruption_label"]] += 1; counts["novelty"][rec["novelty_label"]] += 1; n += 1
                if n % int(args.row_log_interval) == 0: logger.info("Wrote %d records...", n)
    if n == 0: raise RuntimeError("No records produced. Check parquet inputs/filters.")
    logger.info("Finished JSONL write (%d records)", n)
    return n, counts, ids


def split_counts(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    """
    Convert split ratios into integer train/val/test counts.

    Parameters
    ------------
    total: int
        Total number of records.
    ratios: Sequence[float]
        `(train_ratio, val_ratio, test_ratio)`.

    Returns
    ------------
    tuple[int, int, int]
        `(train_count, val_count, test_count)`.
    """
    norm = [r / sum(ratios) for r in ratios]; raw = [total * r for r in norm]; out = [math.floor(x) for x in raw]
    for i in sorted(range(3), key=lambda i: (raw[i] - out[i], -i), reverse=True)[: total - sum(out)]: out[i] += 1
    return int(out[0]), int(out[1]), int(out[2])


def make_splits(ids: Sequence[str], seed: int, ratios: Sequence[float]) -> dict[str, Any]:
    """
    Create deterministic train/val/test splits from paper IDs.

    Parameters
    ------------
    ids: Sequence[str]
        Paper IDs collected during JSONL writing.
    seed: int
        Random seed for shuffling.
    ratios: Sequence[float]
        `(train_ratio, val_ratio, test_ratio)`.

    Returns
    ------------
    dict[str, Any]
        Split payload with IDs and counts.
    """
    ids = sorted(str(x) for x in ids); random.Random(seed).shuffle(ids); n1, n2, n3 = split_counts(len(ids), ratios)
    train, val, test = ids[:n1], ids[n1:n1 + n2], ids[n1 + n2:n1 + n2 + n3]
    return {"split_seed": seed, "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]}, "counts": {"train": len(train), "val": len(val), "test": len(test)}, "ids": {"train": train, "val": val, "test": test}}


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write a JSON payload to disk with stable formatting.

    Parameters
    ------------
    path: Path
        Output path.
    payload: Mapping[str, Any]
        JSON-serializable payload.

    Returns
    ------------
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(payload, f, indent=2, sort_keys=True); f.write("\n")


def sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_metadata(
    args: argparse.Namespace,
    t: LabelThresholds,
    record_count: int,
    label_counts: Mapping[str, Any],
    splits: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build metadata for reproducibility and downstream bookkeeping.

    Parameters
    ------------
    args: argparse.Namespace
        Parsed CLI arguments.
    t: LabelThresholds
        Thresholds used for label generation.
    record_count: int
        Number of JSONL records written.
    label_counts: Mapping[str, Any]
        Aggregate counts for disruption and novelty labels.
    splits: Mapping[str, Any]
        Split payload returned by `make_splits`.

    Returns
    ------------
    dict[str, Any]
        Metadata payload for the dataset generation run.
    """
    primary, secondary = resolve_paths(args)
    return {
        "generated_at_utc": now_utc_iso(), "mode": "sciscinet_parquet", "schema_fields": list(SCHEMA_FIELDS), "record_count": int(record_count),
        "label_thresholds": {"disruptive_min": t.disruptive_min, "consolidating_max": t.consolidating_max, "novelty_margin": t.novelty_margin},
        "label_counts": label_counts, "split_seed": int(args.seed), "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "split_counts": splits["counts"], "split_ids_sha256": sha256_json(splits["ids"]),
        "query_filters": {"primary_parquet": str(primary), "secondary_parquet": str(secondary) if secondary else None, "parquet_primary": args.parquet_primary, "n_papers": int(args.n_papers), "from_year": args.sciscinet_from_year, "to_year": args.sciscinet_to_year, "min_citations": args.sciscinet_min_citations, "include_retracted": bool(args.include_retracted), "language": args.sciscinet_language},
        "artifacts": {"dataset_jsonl": str(args.output), "splits_json": str(args.splits_output), "metadata_json": str(args.metadata_output)},
    }


def main() -> None:
    """
    Run the local parquet -> JSONL/splits/metadata data preparation pipeline.

    Parameters
    ------------
        None

    Returns
    ------------
        None
    """
    args = parse_args()
    t = LabelThresholds(args.disruptive_threshold, args.consolidating_threshold, args.novelty_margin)
    logger.info("Starting local SciSciNet build (limit=%d)", args.n_papers)
    record_count, label_counts, paper_ids = write_jsonl_and_collect_stats(args, t)
    splits = make_splits(paper_ids, args.seed, (args.train_ratio, args.val_ratio, args.test_ratio))
    dump_json(args.splits_output, splits)
    metadata = build_metadata(args, t, record_count, label_counts, splits)
    dump_json(args.metadata_output, metadata)
    print(f"mode=sciscinet_parquet records={record_count} output={args.output}")
    print(f"splits train={splits['counts']['train']} val={splits['counts']['val']} test={splits['counts']['test']}")
    print(f"split_ids_sha256={metadata['split_ids_sha256']}")


if __name__ == "__main__":
    main()
