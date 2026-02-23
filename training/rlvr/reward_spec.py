# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping


_FIXED_REWARD_SPEC: Dict[str, Any] = {
    "environment": {
        "name": "AdversarialDisruptionEnv",
        "max_tokens": 320,
        "turns": 2,
        "label_space": {
            "disruption": ["disruptive", "consolidating", "neutral"],
            "novelty": ["novel", "conventional", "balanced"],
        },
        "thresholds": {
            "disruptive_min": 0.1,
            "consolidating_max": -0.1,
        },
    },
    "reward": {
        "formula": "R = R_correctness + R_reasoning + R_adaptation",
        "R_correctness": {
            "correct": 1.0,
            "incorrect": -1.0,
        },
        "R_reasoning": {
            "max": 0.3,
            "source": "quality of reasoning in final response",
        },
        "R_adaptation": {
            "max": 0.2,
            "source": "appropriate revision or valid defense after challenge",
        },
        "total_min": -1.0,
        "total_max": 1.5,
    },
    "fixed_for_fair_comparison": True,
}


def get_fixed_reward_spec() -> Dict[str, Any]:
    return copy.deepcopy(_FIXED_REWARD_SPEC)


def _normalize(payload: Mapping[str, Any]) -> Dict[str, Any]:
    # Convert to plain dict recursively to avoid surprises with custom mappings.
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping):
            out[str(key)] = _normalize(value)
        elif isinstance(value, list):
            out[str(key)] = [
                _normalize(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            out[str(key)] = value
    return out


def validate_reward_spec(candidate: Mapping[str, Any]) -> None:
    normalized_candidate = _normalize(candidate)
    normalized_expected = _normalize(_FIXED_REWARD_SPEC)
    if normalized_candidate != normalized_expected:
        raise ValueError(
            "Reward/environment spec differs from fixed section-6 setup. "
            "Keep shaping and environment settings unchanged for fair comparison."
        )
