# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional, Sequence, Tuple


DISRUPTION_LABELS = ("disruptive", "consolidating", "neutral")
NOVELTY_LABELS = ("novel", "conventional", "balanced")
_NOVELTY_PARTIAL_PAIRS = {
    frozenset(("novel", "balanced")),
    frozenset(("conventional", "balanced")),
}


class LabelMismatchError(ValueError):
    """Raised when paper metadata and labels are internally inconsistent."""


class MalformedOutputError(ValueError):
    """Raised when a model output cannot be safely parsed into a task label."""


@dataclass(frozen=True)
class StopCondition:
    max_tokens: int = 256


@dataclass
class StepResult:
    reward: float
    done: bool
    observation: Optional[str] = None
    stop_condition: Optional[StopCondition] = None
    reward_components: Dict[str, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def R_correctness(self) -> float:
        return float(self.reward_components.get("R_correctness", 0.0))

    @property
    def R_reasoning(self) -> float:
        return float(self.reward_components.get("R_reasoning", 0.0))

    @property
    def R_adaptation(self) -> float:
        return float(self.reward_components.get("R_adaptation", 0.0))


@dataclass(frozen=True)
class Paper:
    openalex_id: str
    title: str
    abstract: str
    publication_year: int
    cited_by_count: int
    cd_index: float
    novelty_score: float
    conventionality_score: float
    disruption_label: str
    novelty_label: str
    primary_field: str

    def __post_init__(self) -> None:
        d_label = _normalize_label(self.disruption_label)
        n_label = _normalize_label(self.novelty_label)
        object.__setattr__(self, "disruption_label", d_label)
        object.__setattr__(self, "novelty_label", n_label)

        if d_label not in DISRUPTION_LABELS:
            raise LabelMismatchError(f"Invalid disruption_label: {self.disruption_label!r}")
        if n_label not in NOVELTY_LABELS:
            raise LabelMismatchError(f"Invalid novelty_label: {self.novelty_label!r}")

        expected_disruption = derive_disruption_label(self.cd_index)
        if expected_disruption != d_label:
            raise LabelMismatchError(
                "disruption_label does not match cd_index thresholding: "
                f"expected={expected_disruption!r}, got={d_label!r}, cd_index={self.cd_index}"
            )

        expected_novelty = derive_novelty_label(self.novelty_score)
        if expected_novelty != n_label:
            raise LabelMismatchError(
                "novelty_label does not match novelty_score thresholding: "
                f"expected={expected_novelty!r}, got={n_label!r}, novelty_score={self.novelty_score}"
            )

    def prompt_block(self) -> str:
        return (
            f"Title: {self.title}\n"
            f"Abstract: {self.abstract}\n"
            f"Year: {self.publication_year}\n"
            f"Citations: {self.cited_by_count}\n"
            f"Field: {self.primary_field}\n"
            f"CD Index: {self.cd_index:.3f}\n"
            f"Novelty Score: {self.novelty_score:.3f}\n"
            f"Conventionality Score: {self.conventionality_score:.3f}"
        )


class Env:
    async def initial_observation(self) -> Tuple[str, StopCondition]:
        raise NotImplementedError

    async def step(self, action: Any) -> StepResult:
        raise NotImplementedError


class DisruptionPredictionEnv(Env):
    """Single-turn disruption prediction: ±1.0 correctness + up to 0.5 reasoning."""

    def __init__(self, paper: Paper, max_tokens: int = 256):
        self.paper = paper
        self.max_tokens = max_tokens
        self._done = False

    async def initial_observation(self) -> Tuple[str, StopCondition]:
        prompt = (
            "Predict the disruption label for the paper.\n"
            "Allowed labels: disruptive, consolidating, neutral.\n"
            "Return exactly:\n"
            "disruption: <label>\n"
            "reasoning: <short justification>\n\n"
            f"{self.paper.prompt_block()}"
        )
        return prompt, StopCondition(max_tokens=self.max_tokens)

    async def step(self, action: Any) -> StepResult:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._done = True
        text = _action_to_text(action)
        try:
            predicted = _extract_label_for_task(text, "disruption")
            correctness = 1.0 if predicted == self.paper.disruption_label else -1.0
            reasoning = _score_reasoning(_extract_reasoning_text(text), max_reward=0.5)
            reward = correctness + reasoning
            return _result(
                reward=reward,
                done=True,
                correctness=correctness,
                reasoning=reasoning,
                adaptation=0.0,
                info={"predicted_disruption": predicted},
            )
        except MalformedOutputError as exc:
            return _malformed_result(exc)


class NoveltyPredictionEnv(Env):
    """Single-turn novelty prediction: ±1.0 with partial credit for adjacent classes."""

    def __init__(self, paper: Paper, max_tokens: int = 256):
        self.paper = paper
        self.max_tokens = max_tokens
        self._done = False

    async def initial_observation(self) -> Tuple[str, StopCondition]:
        prompt = (
            "Predict the novelty label for the paper.\n"
            "Allowed labels: novel, conventional, balanced.\n"
            "Return exactly:\n"
            "novelty: <label>\n"
            "reasoning: <short justification>\n\n"
            f"{self.paper.prompt_block()}"
        )
        return prompt, StopCondition(max_tokens=self.max_tokens)

    async def step(self, action: Any) -> StepResult:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._done = True
        text = _action_to_text(action)
        try:
            predicted = _extract_label_for_task(text, "novelty")
            correctness = _novelty_correctness(predicted, self.paper.novelty_label)
            return _result(
                reward=correctness,
                done=True,
                correctness=correctness,
                reasoning=0.0,
                adaptation=0.0,
                info={"predicted_novelty": predicted},
            )
        except MalformedOutputError as exc:
            return _malformed_result(exc)


class CombinedImpactEnv(Env):
    """Single-turn joint prediction: 0.5 * disruption + 0.5 * novelty correctness."""

    def __init__(self, paper: Paper, max_tokens: int = 320):
        self.paper = paper
        self.max_tokens = max_tokens
        self._done = False

    async def initial_observation(self) -> Tuple[str, StopCondition]:
        prompt = (
            "Predict both labels for the paper.\n"
            "Disruption labels: disruptive, consolidating, neutral.\n"
            "Novelty labels: novel, conventional, balanced.\n"
            "Return exactly:\n"
            "disruption: <label>\n"
            "novelty: <label>\n"
            "reasoning: <short justification>\n\n"
            f"{self.paper.prompt_block()}"
        )
        return prompt, StopCondition(max_tokens=self.max_tokens)

    async def step(self, action: Any) -> StepResult:
        if self._done:
            raise RuntimeError("Environment already finished.")
        self._done = True
        text = _action_to_text(action)

        disruption_pred, disruption_err = _safe_extract(text, "disruption")
        novelty_pred, novelty_err = _safe_extract(text, "novelty")

        disruption_score = (
            1.0 if disruption_pred == self.paper.disruption_label else -1.0
            if disruption_pred is not None
            else -1.0
        )
        novelty_score = (
            _novelty_correctness(novelty_pred, self.paper.novelty_label)
            if novelty_pred is not None
            else -1.0
        )
        correctness = 0.5 * disruption_score + 0.5 * novelty_score
        info: Dict[str, Any] = {
            "predicted_disruption": disruption_pred,
            "predicted_novelty": novelty_pred,
        }
        if disruption_err or novelty_err:
            info["parse_errors"] = [err for err in (disruption_err, novelty_err) if err]

        return _result(
            reward=correctness,
            done=True,
            correctness=correctness,
            reasoning=0.0,
            adaptation=0.0,
            info=info,
        )


class AdversarialDisruptionEnv(Env):
    """Two-turn disruption environment with challenge and adaptation scoring."""

    def __init__(self, paper: Paper, max_tokens: int = 320):
        self.paper = paper
        self.max_tokens = max_tokens
        self._turn = 0
        self._done = False
        self._initial_prediction: Optional[str] = None

    async def initial_observation(self) -> Tuple[str, StopCondition]:
        prompt = (
            "Round 1: predict the disruption label.\n"
            "Allowed labels: disruptive, consolidating, neutral.\n"
            "Return exactly:\n"
            "disruption: <label>\n"
            "reasoning: <short justification>\n\n"
            f"{self.paper.prompt_block()}"
        )
        return prompt, StopCondition(max_tokens=self.max_tokens)

    async def step(self, action: Any) -> StepResult:
        if self._done:
            raise RuntimeError("Environment already finished.")

        text = _action_to_text(action)
        if self._turn == 0:
            try:
                predicted = _extract_label_for_task(text, "disruption")
            except MalformedOutputError as exc:
                self._done = True
                return _malformed_result(exc)

            self._initial_prediction = predicted
            self._turn = 1
            challenge = self._build_challenge_prompt(predicted)
            return _result(
                reward=0.0,
                done=False,
                observation=challenge,
                stop_condition=StopCondition(max_tokens=self.max_tokens),
                correctness=0.0,
                reasoning=0.0,
                adaptation=0.0,
                info={"stage": "challenge", "initial_prediction": predicted},
            )

        try:
            final_prediction = _extract_label_for_task(text, "disruption")
            correctness = 1.0 if final_prediction == self.paper.disruption_label else -1.0
            reasoning = _score_reasoning(_extract_reasoning_text(text), max_reward=0.3)
            adaptation = _adaptation_score(
                initial=self._initial_prediction,
                final=final_prediction,
                gold=self.paper.disruption_label,
            )
            reward = correctness + reasoning + adaptation
            self._done = True
            self._turn = 2
            return _result(
                reward=reward,
                done=True,
                correctness=correctness,
                reasoning=reasoning,
                adaptation=adaptation,
                info={
                    "stage": "final",
                    "initial_prediction": self._initial_prediction,
                    "final_prediction": final_prediction,
                },
            )
        except MalformedOutputError as exc:
            self._done = True
            return _malformed_result(exc)

    def _build_challenge_prompt(self, initial_prediction: str) -> str:
        gold = self.paper.disruption_label
        if initial_prediction == gold:
            alternatives = [label for label in DISRUPTION_LABELS if label != gold]
            counter_label = alternatives[0]
            challenge = (
                f"Round 2 challenge: Are you sure this is {gold}? "
                f"One could argue it is {counter_label} if forward citations still "
                "co-cite prior references. Defend or revise your answer."
            )
        else:
            challenge = (
                f"Round 2 challenge: Your prior label was {initial_prediction}, but CD index "
                f"{self.paper.cd_index:.3f} maps to {gold}. Revise if needed."
            )
        return (
            f"{challenge}\n"
            "Return exactly:\n"
            "disruption: <label>\n"
            "reasoning: <short justification>"
        )


def derive_disruption_label(cd_index: float) -> str:
    if cd_index > 0.1:
        return "disruptive"
    if cd_index < -0.1:
        return "consolidating"
    return "neutral"


def derive_novelty_label(novelty_score: float) -> str:
    # Simple proxy thresholds for toy environments and synthetic data.
    if novelty_score >= 0.66:
        return "novel"
    if novelty_score <= 0.33:
        return "conventional"
    return "balanced"


def _normalize_label(value: str) -> str:
    return value.strip().lower()


def _action_to_text(action: Any) -> str:
    if isinstance(action, str):
        return action
    if isinstance(action, Sequence):
        if all(isinstance(item, str) for item in action):
            return " ".join(action)
        if all(isinstance(item, int) for item in action):
            return "".join(chr(item) for item in action if 32 <= item <= 126)
    return str(action)


def _extract_label_for_task(text: str, task: str) -> str:
    lowered = text.lower()
    if task == "disruption":
        allowed = DISRUPTION_LABELS
        wrong_set = NOVELTY_LABELS
        key_patterns = ("disruption", "disruption_label", "label", "prediction")
    elif task == "novelty":
        allowed = NOVELTY_LABELS
        wrong_set = DISRUPTION_LABELS
        key_patterns = ("novelty", "novelty_label", "label", "prediction")
    else:
        raise ValueError(f"Unsupported task: {task}")

    explicit_pattern = re.compile(
        rf"\b(?:{'|'.join(re.escape(key) for key in key_patterns)})\s*[:=]\s*"
        rf"({'|'.join(re.escape(label) for label in allowed)})\b"
    )
    explicit_match = explicit_pattern.search(lowered)
    if explicit_match:
        return explicit_match.group(1)

    label_hits = re.findall(rf"\b({'|'.join(re.escape(label) for label in allowed)})\b", lowered)
    unique_hits = sorted(set(label_hits))
    if len(unique_hits) == 1:
        return unique_hits[0]
    if len(unique_hits) > 1:
        raise MalformedOutputError(f"Ambiguous {task} labels in output: {unique_hits}")

    wrong_hits = re.findall(rf"\b({'|'.join(re.escape(label) for label in wrong_set)})\b", lowered)
    unique_wrong_hits = sorted(set(wrong_hits))
    if unique_wrong_hits:
        raise MalformedOutputError(
            f"Found labels for wrong task while parsing {task}: {unique_wrong_hits}"
        )
    raise MalformedOutputError(f"Missing {task} label in model output.")


def _safe_extract(text: str, task: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        return _extract_label_for_task(text, task), None
    except MalformedOutputError as exc:
        return None, str(exc)


def _extract_reasoning_text(text: str) -> str:
    match = re.search(
        r"(?:reasoning|rationale)\s*[:=]\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return text.strip()


def _score_reasoning(reasoning_text: str, max_reward: float) -> float:
    text = reasoning_text.strip()
    if not text:
        return 0.0

    words = text.split()
    if len(words) < 6:
        return 0.0

    score = 0.0
    if len(words) >= 12:
        score += 0.4 * max_reward
    if len(words) >= 24:
        score += 0.2 * max_reward

    lowered = text.lower()
    discourse_markers = ("because", "therefore", "however", "since", "thus")
    metric_markers = (
        "cd",
        "citation",
        "reference",
        "novelty",
        "conventional",
        "disruptive",
        "consolidating",
    )
    if any(marker in lowered for marker in discourse_markers):
        score += 0.2 * max_reward
    if any(marker in lowered for marker in metric_markers):
        score += 0.2 * max_reward

    return round(min(max_reward, score), 4)


def _novelty_correctness(predicted: str, gold: str) -> float:
    if predicted == gold:
        return 1.0
    if frozenset((predicted, gold)) in _NOVELTY_PARTIAL_PAIRS:
        return 0.2
    return -1.0


def _adaptation_score(initial: Optional[str], final: str, gold: str) -> float:
    if initial is None:
        return 0.0
    if initial != gold and final == gold and final != initial:
        return 0.2
    if initial == gold and final == gold:
        return 0.2
    return 0.0


def _result(
    reward: float,
    done: bool,
    correctness: float,
    reasoning: float,
    adaptation: float,
    observation: Optional[str] = None,
    stop_condition: Optional[StopCondition] = None,
    info: Optional[Dict[str, Any]] = None,
) -> StepResult:
    components = {
        "R_correctness": round(correctness, 4),
        "R_reasoning": round(reasoning, 4),
        "R_adaptation": round(adaptation, 4),
    }
    return StepResult(
        reward=round(reward, 4),
        done=done,
        observation=observation,
        stop_condition=stop_condition,
        reward_components=components,
        info=info or {},
    )


def _malformed_result(exc: Exception) -> StepResult:
    return _result(
        reward=-1.0,
        done=True,
        correctness=-1.0,
        reasoning=0.0,
        adaptation=0.0,
        info={"parse_error": str(exc)},
    )
