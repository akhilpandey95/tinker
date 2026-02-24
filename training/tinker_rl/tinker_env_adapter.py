# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from tinker_disruption_rl.tinker_disruption_env import (
    DisruptionPredictionEnv,
    Paper,
    StopCondition as InnerStopCondition,
    StepResult as InnerStepResult,
)


try:  # pragma: no cover - exercised only when tinker_cookbook is installed.
    from tinker_cookbook.rl.types import (  # type: ignore
        Action,
        Env as TinkerEnvBase,
        EnvGroupBuilder as TinkerEnvGroupBuilderBase,
        Observation,
        RLDataset as TinkerRLDatasetBase,
        StepResult,
        StopCondition,
    )
    COOKBOOK_TYPES_AVAILABLE = True
except Exception:  # pragma: no cover - local fallback path is covered.
    COOKBOOK_TYPES_AVAILABLE = False

    class TinkerEnvBase:  # Minimal local shim for dry-run validation.
        async def initial_observation(self) -> Tuple["Observation", "StopCondition"]:
            raise NotImplementedError

        async def step(self, action: "Action") -> "StepResult":
            raise NotImplementedError

    class TinkerEnvGroupBuilderBase:
        async def make_envs(self) -> List[TinkerEnvBase]:
            raise NotImplementedError

    class TinkerRLDatasetBase:
        def get_batch(self, index: int) -> List[TinkerEnvGroupBuilderBase]:
            raise NotImplementedError

    @dataclass(frozen=True)
    class Observation:
        tokens: List[int]

    @dataclass(frozen=True)
    class StopCondition:
        max_tokens: int
        stop_sequences: List[List[int]] | None = None

    @dataclass(frozen=True)
    class Action:
        tokens: List[int]

    @dataclass(frozen=True)
    class StepResult:
        reward: float = 0.0
        stop_reason: str | None = None
        observation: Observation | None = None
        stop_condition: StopCondition | None = None


DISRUPTION_LABELS = ("disruptive", "consolidating", "neutral")
_FORBIDDEN_PROMPT_MARKERS = (
    "CD Index:",
    "Novelty Score:",
    "Conventionality Score:",
    "cd_index",
    "novelty_score",
    "conventionality_score",
)


class MessageRenderer(Protocol):
    def build_generation_prompt(self, messages: Sequence[Mapping[str, str]]) -> List[int]:
        ...

    def get_stop_sequences(self) -> List[List[int]]:
        ...

    def parse_response(self, tokens: Sequence[int]) -> Tuple[Dict[str, str], bool]:
        ...


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def truncate_title_abstract(
    title: str,
    abstract: str,
    max_total_chars: int,
) -> Tuple[str, str]:
    """Deterministically cap `len(title) + len(abstract)` to `max_total_chars`."""
    title_clean = normalize_text(title)
    abstract_clean = normalize_text(abstract)
    if max_total_chars <= 0:
        return "", ""
    if len(title_clean) + len(abstract_clean) <= max_total_chars:
        return title_clean, abstract_clean

    if len(title_clean) >= max_total_chars:
        return _truncate_with_ellipsis(title_clean, max_total_chars), ""

    abstract_budget = max_total_chars - len(title_clean)
    return title_clean, _truncate_with_ellipsis(abstract_clean, abstract_budget)


def _format_concepts(concepts: Optional[Iterable[Any]]) -> Optional[str]:
    if concepts is None:
        return None
    values: List[str] = []
    seen: set[str] = set()
    for item in concepts:
        value = normalize_text(item)
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    if not values:
        return None
    return ", ".join(values[:8])


def build_disruption_v1_prompt(
    *,
    title: str,
    abstract: str,
    publication_year: int,
    cited_by_count: int,
    primary_field: str,
    prompt_max_chars: int,
    concepts: Optional[Iterable[Any]] = None,
) -> str:
    title_out, abstract_out = truncate_title_abstract(
        title=title,
        abstract=abstract,
        max_total_chars=prompt_max_chars,
    )
    field = normalize_text(primary_field) or "Unknown"
    lines = [
        "Predict the disruption label for the paper.",
        "Allowed labels: disruptive, consolidating, neutral.",
        "Return exactly:",
        "disruption: <label>",
        "reasoning: <short justification>",
        "",
        f"Title: {title_out}",
        f"Abstract: {abstract_out}",
        f"Year: {int(publication_year)}",
        f"Citations: {int(cited_by_count)}",
        f"Field: {field}",
    ]
    concepts_text = _format_concepts(concepts)
    if concepts_text:
        lines.append(f"Concepts: {concepts_text}")
    return "\n".join(lines)


def build_disruption_v1_prompt_from_paper(
    paper: Paper,
    *,
    prompt_max_chars: int,
    concepts: Optional[Iterable[Any]] = None,
) -> str:
    return build_disruption_v1_prompt(
        title=paper.title,
        abstract=paper.abstract,
        publication_year=paper.publication_year,
        cited_by_count=paper.cited_by_count,
        primary_field=paper.primary_field,
        prompt_max_chars=prompt_max_chars,
        concepts=concepts,
    )


def find_prompt_leakage_markers(prompt: str) -> List[str]:
    return [marker for marker in _FORBIDDEN_PROMPT_MARKERS if marker in prompt]


def prompt_has_target_metric_leakage(prompt: str) -> bool:
    return bool(find_prompt_leakage_markers(prompt))


class DisruptionPredictionEnvV1(DisruptionPredictionEnv):
    """Single-turn disruption env using a leak-free prompt contract."""

    def __init__(
        self,
        paper: Paper,
        *,
        max_tokens: int = 256,
        prompt_max_chars: int = 2048,
        concepts: Optional[Iterable[Any]] = None,
    ) -> None:
        super().__init__(paper=paper, max_tokens=max_tokens)
        self._prompt_max_chars = int(prompt_max_chars)
        self._concepts = list(concepts) if concepts is not None else None

    async def initial_observation(self) -> Tuple[str, InnerStopCondition]:
        prompt = build_disruption_v1_prompt_from_paper(
            self.paper,
            prompt_max_chars=self._prompt_max_chars,
            concepts=self._concepts,
        )
        return prompt, InnerStopCondition(max_tokens=self.max_tokens)


class SimpleMessageRenderer:
    """Local renderer shim for dry-run validation when cookbook renderers are unavailable."""

    has_extension_property = False

    def __init__(self) -> None:
        self._stop_sequences: List[List[int]] = []

    def build_generation_prompt(self, messages: Sequence[Mapping[str, str]]) -> List[int]:
        lines: List[str] = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower() or "user"
            content = str(message.get("content", ""))
            lines.append(f"<{role}>")
            lines.append(content)
        lines.append("<assistant>")
        text = "\n".join(lines) + "\n"
        return self._encode(text)

    def get_stop_sequences(self) -> List[List[int]]:
        return list(self._stop_sequences)

    def parse_response(self, tokens: Sequence[int]) -> Tuple[Dict[str, str], bool]:
        try:
            text = self._decode(tokens).strip()
            return {"role": "assistant", "content": text}, True
        except Exception:
            return {"role": "assistant", "content": ""}, False

    def encode_assistant_content(self, text: str) -> List[int]:
        return self._encode(text)

    @staticmethod
    def _encode(text: str) -> List[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def _decode(tokens: Sequence[int]) -> str:
        bytes_out = bytearray()
        for token in tokens:
            try:
                value = int(token)
            except Exception:
                continue
            if 0 <= value <= 255:
                bytes_out.append(value)
        return bytes(bytes_out).decode("utf-8", errors="ignore")


def _extract_tokens(action: Any) -> List[int]:
    if hasattr(action, "tokens"):
        raw = getattr(action, "tokens")
        if isinstance(raw, Sequence):
            return [int(token) for token in raw]
    if isinstance(action, Sequence) and not isinstance(action, (str, bytes, bytearray)):
        return [int(token) for token in action]
    if isinstance(action, (bytes, bytearray)):
        return list(action)
    if isinstance(action, str):
        return list(action.encode("utf-8"))
    return []


class TinkerDisruptionEnv(TinkerEnvBase):
    """Adapter from string-based local envs to token-based Tinker-style envs."""

    def __init__(
        self,
        *,
        inner_env: Any,
        renderer: MessageRenderer,
        system_prompt: str | None = None,
    ) -> None:
        self._inner = inner_env
        self._renderer = renderer
        self._system_prompt = system_prompt.strip() if system_prompt else None
        self._messages: List[Dict[str, str]] = []
        self.last_inner_result: InnerStepResult | None = None
        self.last_parsed_text: str = ""
        self.last_parse_success: bool = False
        self.last_prompt_text: str = ""

    async def initial_observation(self) -> Tuple[Observation, StopCondition]:
        prompt_str, stop_cond = await self._inner.initial_observation()
        self.last_prompt_text = str(prompt_str)
        self._messages = []
        if self._system_prompt:
            self._messages.append({"role": "system", "content": self._system_prompt})
        self._messages.append({"role": "user", "content": self.last_prompt_text})
        tokens = self._renderer.build_generation_prompt(self._messages)
        return Observation(tokens=tokens), StopCondition(
            max_tokens=int(stop_cond.max_tokens),
            stop_sequences=self._renderer.get_stop_sequences(),
        )

    async def step(self, action: Any) -> StepResult:
        tokens = _extract_tokens(action)
        message, success = self._renderer.parse_response(tokens)
        text = str(message.get("content", "")) if success else ""
        self.last_parsed_text = text
        self.last_parse_success = bool(success)
        self._messages.append({"role": "assistant", "content": text})

        inner_result = await self._inner.step(text)
        self.last_inner_result = inner_result

        if (not inner_result.done) and inner_result.observation:
            self._messages.append({"role": "user", "content": str(inner_result.observation)})
            next_tokens = self._renderer.build_generation_prompt(self._messages)
            next_stop = inner_result.stop_condition.max_tokens if inner_result.stop_condition else 256
            return StepResult(
                reward=float(inner_result.reward),
                stop_reason=None,
                observation=Observation(tokens=next_tokens),
                stop_condition=StopCondition(
                    max_tokens=int(next_stop),
                    stop_sequences=self._renderer.get_stop_sequences(),
                ),
            )

        return StepResult(
            reward=float(inner_result.reward),
            stop_reason="stop",
            observation=None,
            stop_condition=None,
        )


def build_json_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """Small helper for debugging prompt serialization in manifests."""
    return json.dumps([dict(message) for message in messages], sort_keys=True)


__all__ = [
    "Action",
    "COOKBOOK_TYPES_AVAILABLE",
    "DISRUPTION_LABELS",
    "DisruptionPredictionEnvV1",
    "MessageRenderer",
    "Observation",
    "SimpleMessageRenderer",
    "StepResult",
    "StopCondition",
    "TinkerDisruptionEnv",
    "TinkerEnvBase",
    "TinkerEnvGroupBuilderBase",
    "TinkerRLDatasetBase",
    "build_disruption_v1_prompt",
    "build_disruption_v1_prompt_from_paper",
    "find_prompt_leakage_markers",
    "prompt_has_target_metric_leakage",
    "truncate_title_abstract",
]
