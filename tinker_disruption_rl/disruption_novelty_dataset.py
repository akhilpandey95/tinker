#!/usr/bin/env python3
"""Deterministic dataset builder for disruption/novelty prediction."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
CURRENT_YEAR = 2026

SCHEMA_FIELDS = [
    "openalex_id",
    "title",
    "abstract",
    "publication_year",
    "cited_by_count",
    "cd_index",
    "novelty_score",
    "conventionality_score",
    "disruption_label",
    "novelty_label",
    "primary_field",
    "concepts",
]

FIELD_CONCEPTS: Dict[str, List[str]] = {
    "Computer Science": [
        "machine learning",
        "deep learning",
        "information retrieval",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "algorithms",
        "distributed systems",
        "human-computer interaction",
    ],
    "Biology": [
        "genomics",
        "molecular biology",
        "ecology",
        "bioinformatics",
        "evolutionary biology",
        "cell signaling",
        "protein engineering",
        "microbiology",
        "systems biology",
    ],
    "Physics": [
        "condensed matter",
        "quantum computing",
        "optics",
        "particle physics",
        "statistical mechanics",
        "materials science",
        "nanotechnology",
        "astrophysics",
    ],
    "Medicine": [
        "clinical trials",
        "epidemiology",
        "medical imaging",
        "public health",
        "health informatics",
        "immunology",
        "precision medicine",
        "neuroscience",
    ],
    "Economics": [
        "econometrics",
        "labor economics",
        "innovation policy",
        "network economics",
        "behavioral economics",
        "development economics",
        "applied microeconomics",
        "macroeconomic modeling",
    ],
}

TITLE_PREFIXES = [
    "A Framework for",
    "Towards",
    "Benchmarking",
    "Robust",
    "Interpretable",
    "Scalable",
    "Data-Driven",
    "Causal",
]

TITLE_SUFFIXES = [
    "in Dynamic Systems",
    "for Scientific Discovery",
    "with Limited Supervision",
    "under Distribution Shift",
    "using Multi-Modal Signals",
    "at Scale",
    "in Real-World Data",
    "through Representation Learning",
]


@dataclass(frozen=True)
class LabelThresholds:
    disruptive_min: float = 0.1
    consolidating_max: float = -0.1
    novelty_margin: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "disruptive_min": self.disruptive_min,
            "consolidating_max": self.consolidating_max,
            "novelty_margin": self.novelty_margin,
        }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build disruption/novelty datasets with frozen splits.")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic papers instead of OpenAlex data.")
    parser.add_argument("--n-papers", type=int, default=200, help="Number of papers to generate or ingest.")
    parser.add_argument("--seed", type=int, default=20260220, help="Global random seed for generation and splits.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/disruption_novelty_dataset.jsonl"),
        help="Output JSONL path for records.",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=None,
        help="Output JSON path for frozen split IDs. Defaults to <output>_splits.json.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Output JSON path for metadata. Defaults to <output>_metadata.json.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--strict-label-check", action="store_true", help="Error if provided labels mismatch computed labels.")

    parser.add_argument(
        "--openalex-filter",
        default="type:article,has_abstract:true",
        help="Base OpenAlex filter expression.",
    )
    parser.add_argument("--from-year", type=int, default=2000, help="Minimum publication year for OpenAlex mode.")
    parser.add_argument("--to-year", type=int, default=CURRENT_YEAR, help="Maximum publication year for OpenAlex mode.")
    parser.add_argument("--min-citations", type=int, default=5, help="Minimum cited_by_count for OpenAlex mode.")
    parser.add_argument("--email", default="", help="Contact email for OpenAlex mailto parameter.")
    parser.add_argument("--openalex-per-page", type=int, default=200, help="OpenAlex page size (max 200).")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="HTTP timeout (seconds).")

    parser.add_argument(
        "--disruptive-threshold",
        type=float,
        default=0.1,
        help="Disruption threshold above which label=disruptive.",
    )
    parser.add_argument(
        "--consolidating-threshold",
        type=float,
        default=-0.1,
        help="Disruption threshold below which label=consolidating.",
    )
    parser.add_argument(
        "--novelty-margin",
        type=float,
        default=0.15,
        help="novelty_score - conventionality_score margin for novel/conventional labels.",
    )

    args = parser.parse_args()
    if args.n_papers <= 0:
        raise ValueError("--n-papers must be > 0.")

    for ratio_name in ("train_ratio", "val_ratio", "test_ratio"):
        if getattr(args, ratio_name) < 0:
            raise ValueError(f"--{ratio_name.replace('_', '-')} must be >= 0.")

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive value.")

    if args.consolidating_threshold >= args.disruptive_threshold:
        raise ValueError("--consolidating-threshold must be lower than --disruptive-threshold.")

    if args.from_year and args.to_year and args.from_year > args.to_year:
        raise ValueError("--from-year must be <= --to-year.")

    if args.splits_output is None:
        args.splits_output = args.output.with_suffix(".splits.json")
    if args.metadata_output is None:
        args.metadata_output = args.output.with_suffix(".metadata.json")

    return args


def decode_abstract_inverted_index(inverted_index: Mapping[str, Sequence[int]]) -> str:
    if not inverted_index:
        return ""
    max_position = -1
    for positions in inverted_index.values():
        if not positions:
            continue
        max_position = max(max_position, max(positions))
    if max_position < 0:
        return ""

    words = [""] * (max_position + 1)
    for token, positions in inverted_index.items():
        for position in positions:
            if 0 <= position <= max_position and not words[position]:
                words[position] = token
    return " ".join(word for word in words if word).strip()


def extract_openalex_id(openalex_url: str) -> str:
    if not openalex_url:
        return ""
    return openalex_url.rstrip("/").split("/")[-1]


def normalized_entropy(values: Sequence[float]) -> float:
    positives = [value for value in values if value > 0]
    if len(positives) <= 1:
        return 0.0
    total = sum(positives)
    probs = [value / total for value in positives]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    return clamp(entropy / math.log(len(probs)), 0.0, 1.0)


def label_disruption(cd_index: float, thresholds: LabelThresholds) -> str:
    if cd_index > thresholds.disruptive_min:
        return "disruptive"
    if cd_index < thresholds.consolidating_max:
        return "consolidating"
    return "neutral"


def label_novelty(novelty_score: float, conventionality_score: float, thresholds: LabelThresholds) -> str:
    delta = novelty_score - conventionality_score
    if delta > thresholds.novelty_margin:
        return "novel"
    if delta < -thresholds.novelty_margin:
        return "conventional"
    return "balanced"


def compute_cd_index_proxy(cited_by_count: int, referenced_works_count: int, publication_year: int) -> float:
    denominator = max(cited_by_count + referenced_works_count, 1)
    citation_ratio = (cited_by_count - referenced_works_count) / denominator
    recency = clamp((CURRENT_YEAR - publication_year) / 25.0, 0.0, 1.0)
    recency_adjustment = 0.1 * (0.5 - recency)
    return clamp(citation_ratio + recency_adjustment, -1.0, 1.0)


def compute_novelty_conventionality_proxy(
    concept_scores: Sequence[float], referenced_works_count: int, cited_by_count: int, publication_year: int
) -> Tuple[float, float]:
    entropy = normalized_entropy(concept_scores)
    concept_cardinality = clamp(len(concept_scores) / 8.0, 0.0, 1.0)
    reference_density = clamp(referenced_works_count / 60.0, 0.0, 1.0)
    citation_popularity = clamp(math.log1p(cited_by_count) / math.log(5000.0), 0.0, 1.0)
    age = clamp((CURRENT_YEAR - publication_year) / 30.0, 0.0, 1.0)

    novelty_score = 0.45 * entropy + 0.25 * concept_cardinality + 0.2 * (1.0 - reference_density) + 0.1 * (1.0 - age)
    conventionality_score = 0.45 * reference_density + 0.35 * citation_popularity + 0.2 * (1.0 - entropy)

    return clamp(novelty_score, 0.0, 1.0), clamp(conventionality_score, 0.0, 1.0)


def normalize_schema(record: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {field: record[field] for field in SCHEMA_FIELDS}
    normalized["publication_year"] = int(normalized["publication_year"])
    normalized["cited_by_count"] = int(normalized["cited_by_count"])
    normalized["cd_index"] = round(float(normalized["cd_index"]), 4)
    normalized["novelty_score"] = round(float(normalized["novelty_score"]), 4)
    normalized["conventionality_score"] = round(float(normalized["conventionality_score"]), 4)
    normalized["concepts"] = [str(concept) for concept in normalized["concepts"]]
    return normalized


def apply_and_verify_labels(
    records: Iterable[Dict[str, Any]], thresholds: LabelThresholds, strict: bool = False
) -> Dict[str, int]:
    mismatches = {"disruption_label": 0, "novelty_label": 0}
    for record in records:
        computed_disruption = label_disruption(record["cd_index"], thresholds)
        computed_novelty = label_novelty(record["novelty_score"], record["conventionality_score"], thresholds)

        if "disruption_label" in record and record["disruption_label"] != computed_disruption:
            mismatches["disruption_label"] += 1
            if strict:
                raise ValueError(
                    f"Disruption label mismatch for {record['openalex_id']}: "
                    f"found={record['disruption_label']} expected={computed_disruption}"
                )
        if "novelty_label" in record and record["novelty_label"] != computed_novelty:
            mismatches["novelty_label"] += 1
            if strict:
                raise ValueError(
                    f"Novelty label mismatch for {record['openalex_id']}: "
                    f"found={record['novelty_label']} expected={computed_novelty}"
                )

        record["disruption_label"] = computed_disruption
        record["novelty_label"] = computed_novelty

    return mismatches


def make_synthetic_title(rng: random.Random, concepts: Sequence[str]) -> str:
    left = rng.choice(TITLE_PREFIXES)
    right = rng.choice(TITLE_SUFFIXES)
    anchor = concepts[0].title() if concepts else "Scientific Impact"
    return f"{left} {anchor} {right}"


def make_synthetic_abstract(
    title: str, primary_field: str, concepts: Sequence[str], publication_year: int, cited_by_count: int
) -> str:
    joined_concepts = ", ".join(concepts[:3]) if concepts else "interdisciplinary methods"
    return (
        f"{title}. We study {joined_concepts} within {primary_field.lower()} and report a reproducible benchmark. "
        f"The study was published in {publication_year} and currently has {cited_by_count} citations. "
        "We include ablations, error analysis, and robust baselines for downstream evaluation."
    )


def generate_synthetic_records(n_papers: int, seed: int, thresholds: LabelThresholds) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    fields = sorted(FIELD_CONCEPTS.keys())
    records: List[Dict[str, Any]] = []

    for index in range(n_papers):
        primary_field = rng.choice(fields)
        concept_pool = FIELD_CONCEPTS[primary_field]
        concept_count = rng.randint(3, min(6, len(concept_pool)))
        concepts = sorted(rng.sample(concept_pool, k=concept_count))
        concept_scores = sorted([rng.uniform(0.2, 1.0) for _ in concepts], reverse=True)

        publication_year = rng.randint(1995, CURRENT_YEAR)
        cited_by_count = int(max(0, round(rng.lognormvariate(3.1, 0.9))))
        referenced_works_count = rng.randint(8, 70)

        cd_index = compute_cd_index_proxy(cited_by_count, referenced_works_count, publication_year)
        novelty_score, conventionality_score = compute_novelty_conventionality_proxy(
            concept_scores, referenced_works_count, cited_by_count, publication_year
        )

        style = rng.choice(["novel", "balanced", "conventional"])
        if style == "novel":
            novelty_score += 0.18
            conventionality_score -= 0.12
        elif style == "conventional":
            novelty_score -= 0.18
            conventionality_score += 0.18

        cd_index = clamp(cd_index + rng.uniform(-0.08, 0.08), -1.0, 1.0)
        novelty_score = clamp(novelty_score + rng.uniform(-0.08, 0.08), 0.0, 1.0)
        conventionality_score = clamp(conventionality_score + rng.uniform(-0.08, 0.08), 0.0, 1.0)

        title = make_synthetic_title(rng, concepts)
        record = {
            "openalex_id": f"SYNTHW{index + 1:07d}",
            "title": title,
            "abstract": make_synthetic_abstract(title, primary_field, concepts, publication_year, cited_by_count),
            "publication_year": publication_year,
            "cited_by_count": cited_by_count,
            "cd_index": cd_index,
            "novelty_score": novelty_score,
            "conventionality_score": conventionality_score,
            "disruption_label": label_disruption(cd_index, thresholds),
            "novelty_label": label_novelty(novelty_score, conventionality_score, thresholds),
            "primary_field": primary_field,
            "concepts": concepts,
        }
        records.append(normalize_schema(record))

    return records


def openalex_request(url: str, timeout_s: float) -> Mapping[str, Any]:
    request = Request(url, headers={"User-Agent": "tinker-disruption-data-pipeline/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def infer_primary_field(work: Mapping[str, Any], concept_rows: Sequence[Mapping[str, Any]]) -> str:
    primary_topic = work.get("primary_topic")
    if isinstance(primary_topic, Mapping):
        field = primary_topic.get("field")
        if isinstance(field, Mapping) and field.get("display_name"):
            return str(field["display_name"])
    for concept in concept_rows:
        level = concept.get("level")
        name = concept.get("display_name")
        if level == 0 and name:
            return str(name)
    if concept_rows and concept_rows[0].get("display_name"):
        return str(concept_rows[0]["display_name"])
    return "Unknown"


def transform_openalex_work(work: Mapping[str, Any], thresholds: LabelThresholds) -> Optional[Dict[str, Any]]:
    openalex_id = extract_openalex_id(str(work.get("id", "")))
    title = str(work.get("display_name") or "").strip()
    abstract = decode_abstract_inverted_index(work.get("abstract_inverted_index") or {})
    publication_year = int(work.get("publication_year") or 0)
    cited_by_count = int(work.get("cited_by_count") or 0)
    referenced_works = work.get("referenced_works") or []
    referenced_works_count = len(referenced_works)

    if not openalex_id or not title or not abstract or publication_year <= 0:
        return None

    concept_rows = [
        concept
        for concept in (work.get("concepts") or [])
        if isinstance(concept, Mapping) and concept.get("display_name")
    ]
    concept_rows = sorted(
        concept_rows,
        key=lambda concept: (-float(concept.get("score", 0.0)), str(concept.get("display_name"))),
    )
    concepts = [str(concept["display_name"]).strip() for concept in concept_rows[:8]]
    concept_scores = [float(concept.get("score", 0.0)) for concept in concept_rows[:8]]

    if not concepts:
        concepts = ["Unknown"]
        concept_scores = [0.0]

    cd_index = compute_cd_index_proxy(cited_by_count, referenced_works_count, publication_year)
    novelty_score, conventionality_score = compute_novelty_conventionality_proxy(
        concept_scores, referenced_works_count, cited_by_count, publication_year
    )

    record = {
        "openalex_id": openalex_id,
        "title": title,
        "abstract": abstract,
        "publication_year": publication_year,
        "cited_by_count": cited_by_count,
        "cd_index": cd_index,
        "novelty_score": novelty_score,
        "conventionality_score": conventionality_score,
        "disruption_label": label_disruption(cd_index, thresholds),
        "novelty_label": label_novelty(novelty_score, conventionality_score, thresholds),
        "primary_field": infer_primary_field(work, concept_rows),
        "concepts": concepts,
    }
    return normalize_schema(record)


def build_openalex_filter(args: argparse.Namespace) -> str:
    filters = [args.openalex_filter.strip()] if args.openalex_filter.strip() else []
    if args.from_year:
        filters.append(f"from_publication_date:{args.from_year}-01-01")
    if args.to_year:
        filters.append(f"to_publication_date:{args.to_year}-12-31")
    if args.min_citations is not None:
        filters.append(f"cited_by_count:>{int(args.min_citations)}")
    return ",".join(filters)


def ingest_openalex_records(args: argparse.Namespace, thresholds: LabelThresholds) -> List[Dict[str, Any]]:
    filter_expression = build_openalex_filter(args)
    records: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    cursor = "*"
    per_page = max(1, min(200, int(args.openalex_per_page)))

    while len(records) < args.n_papers:
        remaining = args.n_papers - len(records)
        query_params = {
            "filter": filter_expression,
            "per-page": min(per_page, remaining),
            "cursor": cursor,
            "sort": "cited_by_count:desc",
        }
        if args.email:
            query_params["mailto"] = args.email

        url = f"{OPENALEX_WORKS_ENDPOINT}?{urlencode(query_params)}"
        payload = openalex_request(url, timeout_s=args.request_timeout)
        works = payload.get("results") or []
        if not works:
            break

        for work in works:
            record = transform_openalex_work(work, thresholds)
            if record is None:
                continue
            if record["openalex_id"] in seen_ids:
                continue
            seen_ids.add(record["openalex_id"])
            records.append(record)
            if len(records) >= args.n_papers:
                break

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    return records


def compute_split_counts(total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    normalized_ratios = [ratio / sum(ratios) for ratio in ratios]
    raw_counts = [total * ratio for ratio in normalized_ratios]
    counts = [math.floor(raw) for raw in raw_counts]
    remainder = total - sum(counts)

    fractional_order = sorted(
        range(len(raw_counts)),
        key=lambda idx: (raw_counts[idx] - counts[idx], -idx),
        reverse=True,
    )
    for idx in fractional_order:
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def freeze_splits(records: Sequence[Mapping[str, Any]], seed: int, ratios: Sequence[float]) -> Dict[str, Any]:
    paper_ids = sorted(str(record["openalex_id"]) for record in records)
    rng = random.Random(seed)
    rng.shuffle(paper_ids)

    train_count, val_count, test_count = compute_split_counts(len(paper_ids), ratios)
    train_ids = paper_ids[:train_count]
    val_ids = paper_ids[train_count : train_count + val_count]
    test_ids = paper_ids[train_count + val_count : train_count + val_count + test_count]

    return {
        "split_seed": seed,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "counts": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "ids": {"train": train_ids, "val": val_ids, "test": test_ids},
    }


def write_jsonl(records: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=False))
            handle.write("\n")


def write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def compute_label_counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    disruption_counts = {"disruptive": 0, "neutral": 0, "consolidating": 0}
    novelty_counts = {"novel": 0, "balanced": 0, "conventional": 0}
    for record in records:
        disruption_counts[str(record["disruption_label"])] += 1
        novelty_counts[str(record["novelty_label"])] += 1
    return {"disruption": disruption_counts, "novelty": novelty_counts}


def sha256_json(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_metadata(
    args: argparse.Namespace,
    mode: str,
    thresholds: LabelThresholds,
    records: Sequence[Mapping[str, Any]],
    splits_payload: Mapping[str, Any],
    mismatches: Mapping[str, int],
) -> Dict[str, Any]:
    label_counts = compute_label_counts(records)
    query_filters: Dict[str, Any]
    if mode == "synthetic":
        query_filters = {"synthetic_seed": args.seed, "n_papers": args.n_papers}
    else:
        query_filters = {
            "openalex_filter": build_openalex_filter(args),
            "email": args.email or None,
            "n_papers": args.n_papers,
            "per_page": args.openalex_per_page,
        }

    return {
        "generated_at_utc": now_utc_iso(),
        "mode": mode,
        "schema_fields": SCHEMA_FIELDS,
        "query_filters": query_filters,
        "split_seed": args.seed,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "label_thresholds": thresholds.as_dict(),
        "record_count": len(records),
        "split_counts": splits_payload["counts"],
        "label_counts": label_counts,
        "label_verification_mismatches": dict(mismatches),
        "split_ids_sha256": sha256_json(splits_payload["ids"]),
        "artifacts": {
            "dataset_jsonl": str(args.output),
            "splits_json": str(args.splits_output),
            "metadata_json": str(args.metadata_output),
        },
    }


def main() -> None:
    args = parse_args()
    thresholds = LabelThresholds(
        disruptive_min=args.disruptive_threshold,
        consolidating_max=args.consolidating_threshold,
        novelty_margin=args.novelty_margin,
    )

    mode = "synthetic" if args.synthetic else "openalex"
    if mode == "synthetic":
        records = generate_synthetic_records(args.n_papers, seed=args.seed, thresholds=thresholds)
    else:
        records = ingest_openalex_records(args, thresholds)
        if not records:
            raise RuntimeError("OpenAlex ingestion returned no records. Adjust filters or n-papers and retry.")

    records = sorted((normalize_schema(record) for record in records), key=lambda row: row["openalex_id"])
    mismatches = apply_and_verify_labels(records, thresholds=thresholds, strict=args.strict_label_check)

    splits_payload = freeze_splits(
        records,
        seed=args.seed,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    metadata_payload = build_metadata(
        args=args,
        mode=mode,
        thresholds=thresholds,
        records=records,
        splits_payload=splits_payload,
        mismatches=mismatches,
    )

    write_jsonl(records, args.output)
    write_json(splits_payload, args.splits_output)
    write_json(metadata_payload, args.metadata_output)

    counts = splits_payload["counts"]
    print(f"mode={mode} records={len(records)} output={args.output}")
    print(f"splits train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"split_ids_sha256={metadata_payload['split_ids_sha256']}")


if __name__ == "__main__":
    main()
