from __future__ import annotations

import json
import math
import unittest
from unittest import mock

from ai_scientist.utils.control_evaluator import (
    SCHEMA,
    MAX_CONTAINER_ITEMS,
    MAX_DEPTH,
    MAX_STRING_CHARS,
    canonical_json,
    evaluate_control_replay,
    evaluate_control_replay_json,
)


def _episode(episode_id: str, actions: list[str], rewards: list[float]) -> dict:
    return {"episode_id": episode_id, "actions": actions, "rewards": rewards}


def _payload() -> dict:
    return {
        "schema": SCHEMA,
        "protocol_id": "matched_latent_nonlatent_design_v2",
        "horizon": 2,
        "action_space": ["a", "b"],
        "oracle_reference": {
            "arm_id": "oracle",
            "episodes": [
                _episode("e1", ["a", "b"], [2.0, 1.0]),
                _episode("e2", ["b", "a"], [1.0, 2.0]),
            ],
        },
        "zero_data_reference": {
            "arm_id": "zero_data",
            "episodes": [
                _episode("e1", ["a", "a"], [0.0, 0.0]),
                _episode("e2", ["b", "b"], [0.0, 0.0]),
            ],
        },
        "arms": [
            {
                "arm_id": "candidate",
                "episodes": [
                    _episode("e1", ["a", "b"], [1.0, 1.0]),
                    _episode("e2", ["b", "a"], [1.0, 1.0]),
                ],
            }
        ],
    }


class ControlEvaluatorTests(unittest.TestCase):
    def test_recomputes_trace_metrics_and_alignment_without_environment_claim(
        self,
    ) -> None:
        report = evaluate_control_replay(
            _payload(), expected_horizon=2, expected_actions=("a", "b")
        )
        self.assertEqual(report["status"], "verified")
        arm = report["arms"][0]
        self.assertEqual(arm["action_occupancy"], {"a": 2, "b": 2})
        self.assertEqual(arm["per_action"]["a"]["count"], 2)
        self.assertAlmostEqual(arm["metrics"]["true_environment_value"], 2.0)
        self.assertAlmostEqual(arm["metrics"]["oracle_normalized_regret"], 1.0 / 3.0)
        self.assertAlmostEqual(arm["diagnostics"]["aligned_step_reward_gap"], 0.5)
        self.assertNotIn("action_occupancy_weighted_regret", arm["metrics"])
        self.assertIn("action_occupancy_weighted_regret", report["unsupported_metrics"])
        self.assertEqual(report["verification_scope"], "artifact_internal_consistency")
        for key in (
            "experiment_authorized",
            "training_authorized",
            "claim_promotion_allowed",
            "environment_provenance_verified",
            "reference_optimality_verified",
            "comparison_ready",
        ):
            self.assertFalse(report[key])
        self.assertEqual(report["references"]["oracle_value"], 3.0)
        self.assertEqual(report["references"]["zero_data_value"], 0.0)

    def test_report_and_input_hashes_are_self_consistent_and_deterministic(
        self,
    ) -> None:
        first = evaluate_control_replay(_payload())
        second = evaluate_control_replay(json.loads(canonical_json(_payload())))
        self.assertEqual(first, second)
        report_hash_payload = dict(first)
        recorded_report_hash = report_hash_payload.pop("report_sha256")
        self.assertEqual(
            "sha256:"
            + __import__("hashlib")
            .sha256(canonical_json(report_hash_payload).encode("utf-8"))
            .hexdigest(),
            recorded_report_hash,
        )
        self.assertRegex(first["input_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(first["evaluator_hash"], r"^sha256:[0-9a-f]{64}$")

    def test_rejects_length_action_reward_and_reference_mismatches(self) -> None:
        cases = []
        candidate = _payload()
        candidate["arms"][0]["episodes"][0]["actions"] = ["a"]
        cases.append(candidate)

        candidate = _payload()
        candidate["arms"][0]["episodes"][0]["actions"][0] = "invalid"
        cases.append(candidate)

        candidate = _payload()
        candidate["arms"][0]["episodes"][0]["rewards"][0] = float("nan")
        cases.append(candidate)

        candidate = _payload()
        candidate["zero_data_reference"]["episodes"][0]["episode_id"] = "other"
        cases.append(candidate)

        candidate = _payload()
        candidate["arms"][0]["episodes"][0]["episode_id"] = "other"
        cases.append(candidate)

        for candidate in cases:
            with self.subTest(candidate=candidate):
                report = evaluate_control_replay(candidate)
                self.assertEqual(report["status"], "invalid")
                self.assertNotIn("arms", report)
                self.assertRegex(report["report_sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_rejects_unknown_keys_reserved_ids_and_bound_mismatches(self) -> None:
        candidate = _payload()
        candidate["arms"][0]["extra"] = True
        self.assertEqual(evaluate_control_replay(candidate)["status"], "invalid")

        candidate = _payload()
        candidate["arms"][0]["arm_id"] = "oracle"
        self.assertEqual(evaluate_control_replay(candidate)["status"], "invalid")

        candidate = _payload()
        self.assertEqual(
            evaluate_control_replay(candidate, expected_horizon=4)["status"], "invalid"
        )
        self.assertEqual(
            evaluate_control_replay(candidate, expected_actions=("b", "a"))["status"],
            "invalid",
        )

    def test_zero_denominator_uses_positive_floor_without_nan(self) -> None:
        candidate = _payload()
        candidate["oracle_reference"]["episodes"] = [
            _episode("e1", ["a", "b"], [0.0, 0.0]),
            _episode("e2", ["b", "a"], [0.0, 0.0]),
        ]
        report = evaluate_control_replay(candidate)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["references"]["normalization_denominator"], 1e-6)
        value = report["arms"][0]["metrics"]["oracle_normalized_regret"]
        self.assertTrue(math.isfinite(value))

    def test_normalization_floor_is_protocol_bound(self) -> None:
        report = evaluate_control_replay(_payload(), regret_floor=1e-3)
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["error"]["code"], "regret_floor_binding")

    def test_json_parser_rejects_nonfinite_constants(self) -> None:
        text = json.dumps(_payload()).replace("2.0", "NaN", 1)
        report = evaluate_control_replay_json(text)
        self.assertEqual(report["status"], "invalid")
        self.assertEqual(report["error"]["code"], "invalid_json")

    def test_duplicate_keys_are_rejected_at_all_levels(self) -> None:
        for old, new in (
            ('"horizon": 2', '"horizon": 999, "horizon": 2'),
            ('"episode_id": "e1"', '"episode_id": "fake", "episode_id": "e1"'),
        ):
            text = json.dumps(_payload()).replace(old, new, 1)
            report = evaluate_control_replay_json(text)
            self.assertEqual(report["status"], "invalid")
            self.assertNotIn("arms", report)

    def test_reference_roles_cannot_collide(self) -> None:
        payload = _payload()
        payload["zero_data_reference"]["arm_id"] = "oracle"
        self.assertEqual(
            evaluate_control_replay(payload)["error"]["code"], "arm_id_collision"
        )

    def test_binding_configuration_is_strict_and_hash_visible(self) -> None:
        for horizon in (True, False, 2.0, "2", 0, -1):
            self.assertEqual(
                evaluate_control_replay(_payload(), expected_horizon=horizon)["status"],
                "invalid",
            )
        self.assertEqual(
            evaluate_control_replay(_payload(), expected_actions="ab")["status"],
            "invalid",
        )
        self.assertEqual(
            evaluate_control_replay(_payload(), expected_protocol_id="other")["status"],
            "invalid",
        )
        plain = evaluate_control_replay(_payload())
        bound = evaluate_control_replay_json(
            json.dumps(_payload()),
            expected_horizon=2,
            expected_protocol_id=_payload()["protocol_id"],
        )
        self.assertEqual(bound["status"], "verified")
        self.assertEqual(plain["input_sha256"], bound["input_sha256"])
        self.assertNotEqual(plain["report_sha256"], bound["report_sha256"])

    def test_limits_cycles_and_non_json_values_fail_closed(self) -> None:
        payload = _payload()
        payload["protocol_id"] = "x" * (MAX_STRING_CHARS + 1)
        self.assertEqual(
            evaluate_control_replay(payload)["error"]["code"], "input_limit"
        )
        self.assertEqual(
            evaluate_control_replay([0] * (MAX_CONTAINER_ITEMS + 1))["status"],
            "invalid",
        )
        nested = []
        for _ in range(MAX_DEPTH + 1):
            nested = [nested]
        self.assertEqual(evaluate_control_replay(nested)["status"], "invalid")
        cyclic = []
        cyclic.append(cyclic)
        self.assertEqual(evaluate_control_replay(cyclic)["status"], "invalid")
        self.assertEqual(evaluate_control_replay({1: object()})["status"], "invalid")
        self.assertEqual(
            evaluate_control_replay_json("[" * 2000 + "]" * 2000)["status"], "invalid"
        )

    def test_json_byte_limit_is_applied_before_parsing(self) -> None:
        with mock.patch("ai_scientist.utils.control_evaluator.MAX_JSON_BYTES", 16):
            self.assertEqual(
                evaluate_control_replay_json(" " * 17)["error"]["code"], "input_limit"
            )

    def test_numeric_overflow_never_returns_partial_metrics(self) -> None:
        for oracle_rewards, candidate_rewards in (
            ([1e308, 1e308], [1, 1]),
            ([1e308, 0], [-1e308, 0]),
        ):
            payload = _payload()
            payload["oracle_reference"]["episodes"][0]["rewards"] = oracle_rewards
            payload["arms"][0]["episodes"][0]["rewards"] = candidate_rewards
            report = evaluate_control_replay(payload)
            self.assertEqual(report["status"], "invalid")
            self.assertNotIn("arms", report)

    def test_trace_gap_is_return_gap_per_step_not_counterfactual_regret(self) -> None:
        payload = _payload()
        before = evaluate_control_replay(payload)
        for episode in payload["arms"][0]["episodes"]:
            episode["actions"] = ["b", "b"]
        after = evaluate_control_replay(payload)
        arm = after["arms"][0]
        self.assertEqual(arm["diagnostics"], before["arms"][0]["diagnostics"])
        self.assertEqual(
            arm["per_action"]["a"], {"count": 0, "aligned_reward_gap": None}
        )
        self.assertEqual(
            arm["diagnostics"]["aligned_step_reward_gap"],
            (
                after["references"]["oracle_value"]
                - arm["metrics"]["true_environment_value"]
            )
            / 2,
        )

    def test_missing_source_hash_does_not_certify_metrics(self) -> None:
        with mock.patch(
            "ai_scientist.utils.control_evaluator.Path.read_bytes", side_effect=OSError
        ):
            report = evaluate_control_replay(_payload())
        self.assertEqual(report["status"], "invalid")
        self.assertNotIn("arms", report)


if __name__ == "__main__":
    unittest.main()
