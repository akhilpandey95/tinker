# This Source Code Form is subject to the terms of the
# CC BY-NC-SA 4.0 License. If a copy of the same was not
# distributed with this file, You can obtain one at
# https://github.com/akhilpandey95/tinker/blob/main/LICENSE.

import asyncio
import unittest

from tinker_disruption_rl.tinker_disruption_env import (
    AdversarialDisruptionEnv,
    CombinedImpactEnv,
    DisruptionPredictionEnv,
    LabelMismatchError,
    NoveltyPredictionEnv,
    Paper,
)


def _paper(**overrides):
    base = {
        "openalex_id": "W0",
        "title": "Synthetic Paper",
        "abstract": "Synthetic abstract for testing environment behavior.",
        "publication_year": 2020,
        "cited_by_count": 42,
        "cd_index": 0.45,
        "novelty_score": 0.80,
        "conventionality_score": 0.20,
        "disruption_label": "disruptive",
        "novelty_label": "novel",
        "primary_field": "Computer Science",
    }
    base.update(overrides)
    return Paper(**base)


class TestEnvValidation(unittest.TestCase):
    def test_disruption_reward_decomposition(self):
        env = DisruptionPredictionEnv(_paper())
        asyncio.run(env.initial_observation())
        result = asyncio.run(
            env.step(
                "disruption: disruptive\n"
                "reasoning: Because CD is high and forward citation patterns indicate"
                " references are bypassed, the work is disruptive."
            )
        )

        self.assertTrue(result.done)
        self.assertEqual(result.R_correctness, 1.0)
        self.assertGreater(result.R_reasoning, 0.0)
        self.assertEqual(result.R_adaptation, 0.0)
        self.assertAlmostEqual(
            result.reward,
            result.R_correctness + result.R_reasoning + result.R_adaptation,
        )

    def test_novelty_partial_credit_adjacent_class(self):
        paper = _paper(novelty_score=0.75, novelty_label="novel")
        env = NoveltyPredictionEnv(paper)
        asyncio.run(env.initial_observation())
        result = asyncio.run(env.step("novelty: balanced\nreasoning: mixed pattern."))

        self.assertTrue(result.done)
        self.assertEqual(result.R_correctness, 0.2)
        self.assertEqual(result.reward, 0.2)
        self.assertEqual(result.R_reasoning, 0.0)
        self.assertEqual(result.R_adaptation, 0.0)

    def test_combined_impact_weighting_and_partial_parse_failure(self):
        env = CombinedImpactEnv(_paper())
        asyncio.run(env.initial_observation())
        result = asyncio.run(
            env.step(
                "disruption: disruptive\n"
                "reasoning: I only provide one label so novelty should fail parsing."
            )
        )

        # disruption correct (+1.0), novelty parse failure treated as -1.0 => weighted 0.0
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.R_correctness, 0.0)
        self.assertEqual(result.R_reasoning, 0.0)
        self.assertEqual(result.R_adaptation, 0.0)
        self.assertIn("parse_errors", result.info)

    def test_malformed_output_sanity_check_wrong_label_set(self):
        env = DisruptionPredictionEnv(_paper())
        asyncio.run(env.initial_observation())
        result = asyncio.run(
            env.step(
                "novelty: novel\n"
                "reasoning: Output uses novelty taxonomy instead of disruption labels."
            )
        )

        self.assertTrue(result.done)
        self.assertEqual(result.reward, -1.0)
        self.assertEqual(result.R_correctness, -1.0)
        self.assertIn("wrong task", result.info["parse_error"].lower())

    def test_label_mismatch_edge_case_rejected_at_paper_construction(self):
        with self.assertRaises(LabelMismatchError):
            _paper(cd_index=0.65, disruption_label="consolidating")

        with self.assertRaises(LabelMismatchError):
            _paper(novelty_score=0.75, novelty_label="conventional")

    def test_adversarial_reward_components_for_revision(self):
        paper = _paper(cd_index=0.70, disruption_label="disruptive")
        env = AdversarialDisruptionEnv(paper)
        asyncio.run(env.initial_observation())

        first = asyncio.run(
            env.step(
                "disruption: consolidating\n"
                "reasoning: I think downstream citations remain attached to prior work."
            )
        )
        self.assertFalse(first.done)
        self.assertEqual(first.reward, 0.0)

        second = asyncio.run(
            env.step(
                "disruption: disruptive\n"
                "reasoning: Because CD is strongly positive, citations decouple from"
                " reference ancestry, so I revise to disruptive."
            )
        )
        self.assertTrue(second.done)
        self.assertEqual(second.R_correctness, 1.0)
        self.assertEqual(second.R_adaptation, 0.2)
        self.assertGreater(second.R_reasoning, 0.0)
        self.assertAlmostEqual(
            second.reward,
            second.R_correctness + second.R_reasoning + second.R_adaptation,
        )

    def test_tiny_synthetic_end_to_end_env_smoke(self):
        papers = [
            _paper(openalex_id="W1", cd_index=0.50, disruption_label="disruptive"),
            _paper(
                openalex_id="W2",
                cd_index=-0.55,
                disruption_label="consolidating",
                novelty_score=0.40,
                novelty_label="balanced",
            ),
        ]

        smoke_components = None
        for paper in papers:
            env = AdversarialDisruptionEnv(paper)
            asyncio.run(env.initial_observation())
            first = asyncio.run(
                env.step(
                    "disruption: neutral\n"
                    "reasoning: Initial guess before challenge."
                )
            )
            self.assertFalse(first.done)

            final = asyncio.run(
                env.step(
                    f"disruption: {paper.disruption_label}\n"
                    "reasoning: Because CD thresholding supports this final label,"
                    " I adapt after challenge."
                )
            )
            self.assertTrue(final.done)
            smoke_components = final.reward_components

        print(f"SMOKE_REWARD_COMPONENTS={smoke_components}")
        self.assertIsNotNone(smoke_components)


if __name__ == "__main__":
    unittest.main()
