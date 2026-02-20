from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from training.rlvr.prompt_contract import (
    DISRUPTION_LABELS,
    NOVELTY_LABELS,
    format_disruption_response,
    format_joint_response,
    parse_joint_response,
)


def _softmax(logits: Mapping[str, float], temperature: float) -> Dict[str, float]:
    t = max(float(temperature), 1e-6)
    scaled = {label: value / t for label, value in logits.items()}
    max_logit = max(scaled.values())
    exps = {label: math.exp(value - max_logit) for label, value in scaled.items()}
    denom = sum(exps.values())
    if denom <= 0:
        size = float(len(logits))
        return {label: 1.0 / size for label in logits}
    return {label: exps[label] / denom for label in logits}


def _argmax(probs: Mapping[str, float], labels: Sequence[str]) -> str:
    best_label = labels[0]
    best_value = float("-inf")
    for label in labels:
        value = float(probs.get(label, 0.0))
        if value > best_value:
            best_label = label
            best_value = value
    return best_label


def _sample_label(probs: Mapping[str, float], labels: Sequence[str], rng: random.Random) -> str:
    draw = rng.random()
    cumulative = 0.0
    for label in labels:
        cumulative += float(probs.get(label, 0.0))
        if draw <= cumulative:
            return label
    return labels[-1]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _dot(values: Sequence[float], weights: Sequence[float]) -> float:
    return sum(float(v) * float(w) for v, w in zip(values, weights))


def _disruption_features(record: Mapping[str, Any]) -> Tuple[float, float, float]:
    cd_index = float(record["cd_index"])
    return (cd_index, abs(cd_index), 1.0)


def _novelty_features(record: Mapping[str, Any]) -> Tuple[float, float, float]:
    novelty_score = float(record["novelty_score"])
    conventionality_score = float(record["conventionality_score"])
    delta = novelty_score - conventionality_score
    return (delta, abs(delta), 1.0)


def _extract_challenge_label(challenge_text: str) -> str | None:
    lowered = challenge_text.lower()
    patterns = (
        r"maps to\s+(disruptive|consolidating|neutral)",
        r"is\s+(disruptive|consolidating|neutral)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return str(match.group(1))
    return None


@dataclass
class InitialAction:
    initial_label: str
    initial_probs: Dict[str, float]
    initial_response: str


@dataclass
class FinalAction:
    final_label: str
    final_probs: Dict[str, float]
    final_response: str
    revised: bool
    revision_probability: float


@dataclass
class RLVRPolicy:
    disruption_weights: Dict[str, list[float]]
    novelty_weights: Dict[str, list[float]]
    disruption_temperature: float
    novelty_temperature: float
    revision_bias: float

    @classmethod
    def default_pretrained_base(cls) -> "RLVRPolicy":
        return cls(
            disruption_weights={
                "disruptive": [2.4, -0.8, -0.2],
                "consolidating": [-2.4, -0.8, -0.2],
                "neutral": [0.0, 1.3, 0.3],
            },
            novelty_weights={
                "novel": [2.0, -0.6, -0.1],
                "conventional": [-2.0, -0.6, -0.1],
                "balanced": [0.0, 1.0, 0.2],
            },
            disruption_temperature=1.0,
            novelty_temperature=1.0,
            revision_bias=0.0,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RLVRPolicy":
        return cls(
            disruption_weights={
                label: [float(v) for v in payload["disruption_weights"][label]]
                for label in DISRUPTION_LABELS
            },
            novelty_weights={
                label: [float(v) for v in payload["novelty_weights"][label]]
                for label in NOVELTY_LABELS
            },
            disruption_temperature=float(payload.get("disruption_temperature", 1.0)),
            novelty_temperature=float(payload.get("novelty_temperature", 1.0)),
            revision_bias=float(payload.get("revision_bias", 0.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disruption_weights": {
                label: [float(v) for v in self.disruption_weights[label]]
                for label in DISRUPTION_LABELS
            },
            "novelty_weights": {
                label: [float(v) for v in self.novelty_weights[label]]
                for label in NOVELTY_LABELS
            },
            "disruption_temperature": float(self.disruption_temperature),
            "novelty_temperature": float(self.novelty_temperature),
            "revision_bias": float(self.revision_bias),
        }

    def _disruption_logits(self, record: Mapping[str, Any]) -> Dict[str, float]:
        features = _disruption_features(record)
        return {
            label: _dot(features, self.disruption_weights[label])
            for label in DISRUPTION_LABELS
        }

    def _novelty_logits(self, record: Mapping[str, Any]) -> Dict[str, float]:
        features = _novelty_features(record)
        return {
            label: _dot(features, self.novelty_weights[label])
            for label in NOVELTY_LABELS
        }

    def disruption_probs(self, record: Mapping[str, Any], temperature: float) -> Dict[str, float]:
        return _softmax(self._disruption_logits(record), temperature=temperature)

    def novelty_probs(self, record: Mapping[str, Any], temperature: float) -> Dict[str, float]:
        return _softmax(self._novelty_logits(record), temperature=temperature)

    def sample_initial_action(
        self,
        record: Mapping[str, Any],
        rng: random.Random,
        exploration_temperature: float,
    ) -> InitialAction:
        initial_probs = self.disruption_probs(record, temperature=exploration_temperature)
        initial_label = _sample_label(initial_probs, DISRUPTION_LABELS, rng)
        initial_response = format_disruption_response(
            initial_label,
            (
                "Because CD index and citation-reference overlap provide the first-pass "
                "signal, I commit an initial disruption label before seeing the challenge."
            ),
        )
        return InitialAction(
            initial_label=initial_label,
            initial_probs=initial_probs,
            initial_response=initial_response,
        )

    def sample_final_action(
        self,
        record: Mapping[str, Any],
        challenge_text: str,
        initial_label: str,
        rng: random.Random,
        exploration_temperature: float,
    ) -> FinalAction:
        challenge_label = _extract_challenge_label(challenge_text)
        final_probs = self.disruption_probs(record, temperature=exploration_temperature)
        sampled_final = _sample_label(final_probs, DISRUPTION_LABELS, rng)

        revision_probability = _sigmoid(self.revision_bias)
        revised = False
        final_label = sampled_final
        if challenge_label is not None and challenge_label != initial_label:
            if rng.random() < revision_probability:
                final_label = challenge_label
                revised = True

        if revised:
            reasoning = (
                "Because CD index thresholding and citation ancestry cues were stronger after "
                "the challenge, I revise the disruption label to match the higher-confidence "
                "evidence rather than keeping my original guess."
            )
        else:
            reasoning = (
                "Because CD index thresholding and citation ancestry cues still support this "
                "decision after challenge, I defend the label with the same evidence basis "
                "instead of revising prematurely."
            )

        final_response = format_disruption_response(final_label, reasoning)

        return FinalAction(
            final_label=final_label,
            final_probs=final_probs,
            final_response=final_response,
            revised=revised,
            revision_probability=revision_probability,
        )

    def policy_gradient_update_disruption(
        self,
        record: Mapping[str, Any],
        action_label: str,
        action_probs: Mapping[str, float],
        advantage: float,
        learning_rate: float,
    ) -> None:
        features = _disruption_features(record)
        for label in DISRUPTION_LABELS:
            coeff = (1.0 if label == action_label else 0.0) - float(action_probs.get(label, 0.0))
            weights = self.disruption_weights[label]
            for idx, feature in enumerate(features):
                weights[idx] += float(learning_rate) * float(advantage) * coeff * float(feature)

    def update_revision_bias(
        self,
        revised: bool,
        revision_probability: float,
        advantage: float,
        learning_rate: float,
    ) -> None:
        action = 1.0 if revised else 0.0
        grad = action - float(revision_probability)
        self.revision_bias += float(learning_rate) * float(advantage) * grad

    def predict_joint_row(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        disruption_probs = self.disruption_probs(record, temperature=self.disruption_temperature)
        novelty_probs = self.novelty_probs(record, temperature=self.novelty_temperature)

        pred_disruption = _argmax(disruption_probs, DISRUPTION_LABELS)
        pred_novelty = _argmax(novelty_probs, NOVELTY_LABELS)

        response = format_joint_response(
            pred_disruption,
            pred_novelty,
            "CD index and novelty-conventionality evidence support this prediction.",
        )
        parsed = parse_joint_response(response)

        row: Dict[str, Any] = {
            "openalex_id": str(record["openalex_id"]),
            "gold_disruption": str(record["disruption_label"]).lower(),
            "gold_novelty": str(record["novelty_label"]).lower(),
            "pred_disruption": str(parsed["disruption_label"]) if parsed["disruption_label"] else None,
            "pred_novelty": str(parsed["novelty_label"]) if parsed["novelty_label"] else None,
            "disruption_confidence": max(disruption_probs.values()),
            "novelty_confidence": max(novelty_probs.values()),
            "parse_error": parsed["parse_error"],
            "response_text": response,
        }

        for label in DISRUPTION_LABELS:
            row[f"disruption_prob_{label}"] = disruption_probs[label]
        for label in NOVELTY_LABELS:
            row[f"novelty_prob_{label}"] = novelty_probs[label]

        row["disruption_correct"] = int(row["pred_disruption"] == row["gold_disruption"])
        row["novelty_correct"] = int(row["pred_novelty"] == row["gold_novelty"])
        row["joint_correct"] = int(row["disruption_correct"] and row["novelty_correct"])
        return row


def build_policy_from_config(config: Mapping[str, Any]) -> RLVRPolicy:
    base = RLVRPolicy.default_pretrained_base()
    overrides = config.get("pretrained_base", {})

    if not overrides:
        return base

    disruption_weights: MutableMapping[str, list[float]] = {
        label: list(base.disruption_weights[label])
        for label in DISRUPTION_LABELS
    }
    novelty_weights: MutableMapping[str, list[float]] = {
        label: list(base.novelty_weights[label])
        for label in NOVELTY_LABELS
    }

    if isinstance(overrides.get("disruption_weights"), Mapping):
        for label in DISRUPTION_LABELS:
            if label in overrides["disruption_weights"]:
                disruption_weights[label] = [
                    float(v) for v in overrides["disruption_weights"][label]
                ]
    if isinstance(overrides.get("novelty_weights"), Mapping):
        for label in NOVELTY_LABELS:
            if label in overrides["novelty_weights"]:
                novelty_weights[label] = [
                    float(v) for v in overrides["novelty_weights"][label]
                ]

    return RLVRPolicy(
        disruption_weights=dict(disruption_weights),
        novelty_weights=dict(novelty_weights),
        disruption_temperature=float(
            overrides.get("disruption_temperature", base.disruption_temperature)
        ),
        novelty_temperature=float(
            overrides.get("novelty_temperature", base.novelty_temperature)
        ),
        revision_bias=float(overrides.get("revision_bias", base.revision_bias)),
    )
