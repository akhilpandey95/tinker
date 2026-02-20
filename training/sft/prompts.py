from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Mapping, Optional


DISRUPTION_LABELS = ("disruptive", "consolidating", "neutral")
NOVELTY_LABELS = ("novel", "conventional", "balanced")


class ParseError(ValueError):
    """Raised when a response cannot be parsed into the expected contract."""


@dataclass(frozen=True)
class JointPromptExample:
    openalex_id: str
    prompt: str
    target: str
    disruption_label: str
    novelty_label: str


def normalize_label(value: str) -> str:
    return value.strip().lower()


def _require(record: Mapping[str, Any], key: str) -> Any:
    if key not in record:
        raise KeyError(f"Missing required key in record: {key}")
    return record[key]


def build_paper_prompt_block(record: Mapping[str, Any]) -> str:
    return (
        f"Title: {_require(record, 'title')}\n"
        f"Abstract: {_require(record, 'abstract')}\n"
        f"Year: {int(_require(record, 'publication_year'))}\n"
        f"Citations: {int(_require(record, 'cited_by_count'))}\n"
        f"Field: {_require(record, 'primary_field')}\n"
        f"CD Index: {float(_require(record, 'cd_index')):.3f}\n"
        f"Novelty Score: {float(_require(record, 'novelty_score')):.3f}\n"
        f"Conventionality Score: {float(_require(record, 'conventionality_score')):.3f}"
    )


def build_joint_prompt(record: Mapping[str, Any]) -> str:
    """Matches CombinedImpactEnv prompt contract exactly."""
    return (
        "Predict both labels for the paper.\n"
        "Disruption labels: disruptive, consolidating, neutral.\n"
        "Novelty labels: novel, conventional, balanced.\n"
        "Return exactly:\n"
        "disruption: <label>\n"
        "novelty: <label>\n"
        "reasoning: <short justification>\n\n"
        f"{build_paper_prompt_block(record)}"
    )


def build_joint_target(
    record: Mapping[str, Any],
    reasoning_template: str = (
        "CD index and novelty-conventionality evidence support these labels."
    ),
) -> str:
    disruption = normalize_label(str(_require(record, "disruption_label")))
    novelty = normalize_label(str(_require(record, "novelty_label")))
    return (
        f"disruption: {disruption}\n"
        f"novelty: {novelty}\n"
        f"reasoning: {reasoning_template}"
    )


def build_joint_example(record: Mapping[str, Any]) -> JointPromptExample:
    openalex_id = str(_require(record, "openalex_id"))
    disruption = normalize_label(str(_require(record, "disruption_label")))
    novelty = normalize_label(str(_require(record, "novelty_label")))
    return JointPromptExample(
        openalex_id=openalex_id,
        prompt=build_joint_prompt(record),
        target=build_joint_target(record),
        disruption_label=disruption,
        novelty_label=novelty,
    )


def _extract_label_for_task(text: str, task: str) -> str:
    lowered = text.lower()
    if task == "disruption":
        allowed = DISRUPTION_LABELS
        wrong_set = NOVELTY_LABELS
        keys = ("disruption", "disruption_label", "label", "prediction")
    elif task == "novelty":
        allowed = NOVELTY_LABELS
        wrong_set = DISRUPTION_LABELS
        keys = ("novelty", "novelty_label", "label", "prediction")
    else:
        raise ValueError(f"Unsupported task: {task}")

    explicit = re.compile(
        rf"\b(?:{'|'.join(re.escape(key) for key in keys)})\s*[:=]\s*"
        rf"({'|'.join(re.escape(label) for label in allowed)})\b"
    )
    explicit_match = explicit.search(lowered)
    if explicit_match:
        return explicit_match.group(1)

    hits = re.findall(rf"\b({'|'.join(re.escape(label) for label in allowed)})\b", lowered)
    unique_hits = sorted(set(hits))
    if len(unique_hits) == 1:
        return unique_hits[0]
    if len(unique_hits) > 1:
        raise ParseError(f"Ambiguous {task} labels in output: {unique_hits}")

    wrong_hits = re.findall(rf"\b({'|'.join(re.escape(label) for label in wrong_set)})\b", lowered)
    unique_wrong = sorted(set(wrong_hits))
    if unique_wrong:
        raise ParseError(f"Found labels for wrong task while parsing {task}: {unique_wrong}")

    raise ParseError(f"Missing {task} label in model output.")


def _extract_reasoning(text: str) -> str:
    match = re.search(
        r"(?:reasoning|rationale)\s*[:=]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_joint_response(text: str) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {
        "disruption_label": None,
        "novelty_label": None,
        "reasoning": _extract_reasoning(text),
        "parse_error": None,
    }

    disruption_err: Optional[str] = None
    novelty_err: Optional[str] = None

    try:
        result["disruption_label"] = _extract_label_for_task(text, "disruption")
    except ParseError as exc:
        disruption_err = str(exc)

    try:
        result["novelty_label"] = _extract_label_for_task(text, "novelty")
    except ParseError as exc:
        novelty_err = str(exc)

    errors = [err for err in (disruption_err, novelty_err) if err]
    if errors:
        result["parse_error"] = " | ".join(errors)

    return result


def format_joint_response(disruption_label: str, novelty_label: str, reasoning: str) -> str:
    return (
        f"disruption: {normalize_label(disruption_label)}\n"
        f"novelty: {normalize_label(novelty_label)}\n"
        f"reasoning: {reasoning.strip()}"
    )
