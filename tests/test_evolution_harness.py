from __future__ import annotations

import unittest
from copy import deepcopy

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_harness import (
    EvolutionHarnessError,
    bind_skill_backtest,
    bind_version_evidence,
    build_evolution_harness_audit,
    build_harness_policy_hash,
    validate_evolution_harness_audit,
)


def _digest(label: str) -> str:
    return canonical_content_hash({"label": label})


def _comparison(policy: dict | None = None) -> dict[str, str]:
    return {
        "harness_hash": _digest("harness-v1"),
        "policy_hash": build_harness_policy_hash(policy),
        "evaluator_hash": _digest("evaluator-v1"),
        "task_hash": _digest("task-slice-v1"),
        "resource_hash": _digest("resource-envelope-v1"),
        "seed_policy_hash": _digest("seed-policy-v1"),
    }


def _thresholds() -> dict[str, dict[str, float | str]]:
    return {
        "loop_rate": {
            "direction": "lower",
            "healthy_bound": 0.25,
            "max_regression": 0.10,
        },
        "test_rate": {
            "direction": "higher",
            "healthy_bound": 0.50,
            "max_regression": 0.10,
        },
    }


def _integrity(**overrides: bool) -> dict[str, bool]:
    checks = {
        "evidence_bound": True,
        "environment_isolated": True,
        "evaluation_frozen": True,
        "git_leakage_absent": True,
    }
    checks.update(overrides)
    return checks


def _version(
    version_id: str,
    parent_version_id: str | None,
    *,
    score: float,
    reward_groups: list[list[float]] | None = None,
    behavior: dict[str, float] | None = None,
    comparison: dict[str, str] | None = None,
    integrity: dict[str, bool] | None = None,
    observed_cost: float = 5.0,
    budget: float = 10.0,
    epoch_id: str = "epoch:test",
) -> dict:
    return bind_version_evidence(
        {
            "epoch_id": epoch_id,
            "version_id": version_id,
            "parent_version_id": parent_version_id,
            "scores": {"objective": score, "reproducibility": 0.95},
            "reward_groups": reward_groups or [[0.0, 0.5, 1.0], [0.1, 0.4, 0.9]],
            "behavior": behavior or {"loop_rate": 0.10, "test_rate": 0.75},
            "behavior_thresholds": _thresholds(),
            "integrity_checks": integrity or _integrity(),
            "comparison_hashes": comparison or _comparison(),
            "cost": {
                "observed": observed_cost,
                "budget": budget,
                "unit": "token_equivalent",
            },
        }
    )


def _healthy_versions() -> list[dict]:
    return [
        _version(
            "v1",
            None,
            score=0.50,
            behavior={"loop_rate": 0.18, "test_rate": 0.60},
        ),
        _version(
            "v2",
            "v1",
            score=0.65,
            behavior={"loop_rate": 0.10, "test_rate": 0.75},
        ),
    ]


def _backtest(
    skill_id: str,
    *,
    domain: str,
    split: str,
    skill_type: str = "analyzer",
    passed: bool = True,
    producer_id: str = "agent:trainer",
    evaluator_id: str = "service:independent-judge",
    authority: str = "independent_evaluator",
) -> dict:
    return bind_skill_backtest(
        {
            "skill_id": skill_id,
            "skill_type": skill_type,
            "skill_artifact_hash": _digest(f"skill:{skill_id}"),
            "domain": domain,
            "split": split,
            "passed": passed,
            "producer_id": producer_id,
            "evaluator_id": evaluator_id,
            "evaluator_authority": authority,
            "evaluator_protocol_hash": _digest("skill-evaluator-v1"),
        }
    )


def _risk_codes(audit: dict) -> set[str]:
    return {row["code"] for row in audit["risks"]}


def _rehash_audit(audit: dict) -> None:
    core = {key: deepcopy(value) for key, value in audit.items() if key != "audit_hash"}
    audit["audit_hash"] = canonical_content_hash(core)


class EvolutionHarnessAuditTests(unittest.TestCase):
    def test_high_score_with_behavior_degradation_is_held(self) -> None:
        versions = [
            _version(
                "v1",
                None,
                score=0.50,
                behavior={"loop_rate": 0.10, "test_rate": 0.80},
            ),
            _version(
                "v2",
                "v1",
                score=0.75,
                behavior={"loop_rate": 0.42, "test_rate": 0.30},
            ),
        ]
        audit = build_evolution_harness_audit(versions)

        self.assertTrue(audit["layers"]["score"]["improved"])
        self.assertTrue(audit["layers"]["behavior"]["score_behavior_divergence"])
        self.assertTrue(
            {
                "BEHAVIOR.THRESHOLD_VIOLATION",
                "BEHAVIOR.REGRESSION",
                "BEHAVIOR.SCORE_DIVERGENCE",
            }.issubset(_risk_codes(audit))
        )
        self.assertEqual(audit["progression"]["decision"], "hold")
        self.assertFalse(audit["progression"]["controlled_progression_allowed"])

    def test_all_tied_groups_are_low_information_and_block_progression(self) -> None:
        versions = _healthy_versions()
        versions[-1] = _version(
            "v2",
            "v1",
            score=0.70,
            reward_groups=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            behavior={"loop_rate": 0.10, "test_rate": 0.75},
        )
        audit = build_evolution_harness_audit(versions)

        self.assertTrue(audit["layers"]["signal"]["current"]["all_tied"])
        self.assertEqual(
            audit["layers"]["signal"]["current"]["low_information_ratio"],
            1.0,
        )
        self.assertTrue(
            {"SIGNAL.ALL_TIED", "SIGNAL.LOW_INFORMATION"}.issubset(_risk_codes(audit))
        )
        targets = {
            row["target_component"] for row in audit["next_epoch_harness_challenges"]
        }
        self.assertIn("signal_metric_or_filter", targets)

    def test_signal_threshold_uses_exact_ratio_not_rounded_display(self) -> None:
        policy = {"low_information_ratio": 0.3333333333331}
        comparison = _comparison(policy)
        audit = build_evolution_harness_audit(
            [
                _version("v1", None, score=0.50, comparison=comparison),
                _version(
                    "v2",
                    "v1",
                    score=0.70,
                    comparison=comparison,
                    reward_groups=[
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 1.0],
                        [0.1, 0.4, 0.9],
                    ],
                ),
            ],
            policy=policy,
        )

        self.assertEqual(
            audit["layers"]["signal"]["current"]["low_information_ratio"],
            0.333333333333,
        )
        self.assertIn("SIGNAL.LOW_INFORMATION", _risk_codes(audit))

    def test_harness_and_evaluator_changes_are_incomparable_next_epoch_only(
        self,
    ) -> None:
        versions = _healthy_versions()
        changed = _comparison()
        changed["harness_hash"] = _digest("harness-v2")
        changed["evaluator_hash"] = _digest("evaluator-v2")
        versions[-1] = _version(
            "v2",
            "v1",
            score=0.80,
            comparison=changed,
            behavior={"loop_rate": 0.10, "test_rate": 0.75},
        )
        audit = build_evolution_harness_audit(versions)

        self.assertFalse(audit["layers"]["version"]["latest_transition_comparable"])
        self.assertTrue(
            {
                "VERSION.HARNESS_MISMATCH",
                "VERSION.EVALUATOR_MISMATCH",
            }.issubset(_risk_codes(audit))
        )
        evaluator_challenge = next(
            row
            for row in audit["next_epoch_harness_challenges"]
            if row["target_component"] == "evaluation_policy"
        )
        self.assertEqual(evaluator_challenge["status"], "next_epoch_human_review")
        self.assertTrue(evaluator_challenge["protected_component"])
        self.assertFalse(evaluator_challenge["same_epoch_evaluator_mutation_allowed"])
        self.assertFalse(evaluator_challenge["automatic_application_allowed"])

    def test_git_leakage_integrity_failure_blocks_false_promotion(self) -> None:
        versions = _healthy_versions()
        versions[-1] = _version(
            "v2",
            "v1",
            score=0.99,
            integrity=_integrity(git_leakage_absent=False),
            behavior={"loop_rate": 0.10, "test_rate": 0.75},
        )
        audit = build_evolution_harness_audit(versions)

        self.assertTrue(
            {"VERSION.GIT_LEAKAGE", "VERSION.INTEGRITY_FAILURE"}.issubset(
                _risk_codes(audit)
            )
        )
        self.assertEqual(audit["progression"]["decision"], "hold")
        current = audit["layers"]["version"]["versions"][-1]
        self.assertFalse(current["integrity_passed"])
        self.assertIn("git_leakage_absent", current["failed_integrity_checks"])

    def test_healthy_version_is_only_eligible_for_human_review(self) -> None:
        audit = build_evolution_harness_audit(_healthy_versions())

        self.assertEqual(audit["risks"], [])
        self.assertEqual(audit["progression"]["decision"], "eligible_for_human_review")
        self.assertTrue(audit["progression"]["controlled_progression_allowed"])
        self.assertFalse(audit["progression"]["automatic_progression_allowed"])
        self.assertFalse(audit["progression"]["automatic_production_mutation_allowed"])
        self.assertFalse(audit["progression"]["same_epoch_evaluator_mutation_allowed"])
        self.assertEqual(audit["next_epoch_harness_challenges"], [])
        self.assertTrue(validate_evolution_harness_audit(audit)["ok"])
        unsigned = {key: value for key, value in audit.items() if key != "audit_hash"}
        self.assertEqual(audit["audit_hash"], canonical_content_hash(unsigned))

    def test_skill_stays_quarantined_without_both_splits_and_independence(self) -> None:
        only_historical = _backtest("loop-analyzer", domain="swe", split="historical")
        self_evaluated_holdout = _backtest(
            "loop-analyzer",
            domain="swe",
            split="holdout",
            evaluator_id="agent:trainer",
        )
        audit = build_evolution_harness_audit(
            _healthy_versions(),
            backtests=[only_historical, self_evaluated_holdout],
        )
        skill = audit["skills"][0]

        self.assertEqual(skill["status"], "quarantined")
        self.assertEqual(skill["validation_scope"], "quarantine")
        self.assertIn("holdout_validation_incomplete", skill["quarantine_reasons"])
        self.assertIn("independent_evaluator_missing", skill["quarantine_reasons"])
        self.assertFalse(skill["automatic_application_allowed"])
        self.assertFalse(skill["production_promotion_allowed"])

    def test_skill_becomes_domain_validated_after_historical_and_holdout(self) -> None:
        backtests = [
            _backtest("loop-analyzer", domain="swe", split="historical"),
            _backtest("loop-analyzer", domain="swe", split="holdout"),
        ]
        audit = build_evolution_harness_audit(_healthy_versions(), backtests=backtests)
        skill = audit["skills"][0]

        self.assertEqual(skill["status"], "domain_validated")
        self.assertEqual(skill["validation_scope"], "domain")
        self.assertEqual(skill["validated_domains"], ["swe"])
        self.assertEqual(skill["quarantine_reasons"], [])
        self.assertTrue(skill["domain_results"][0]["validated"])

    def test_skill_requires_two_fully_validated_domains_for_cross_domain(self) -> None:
        backtests = [
            _backtest("variance-filter", domain=domain, split=split)
            for domain in ("coding", "swe")
            for split in ("historical", "holdout")
        ]
        audit = build_evolution_harness_audit(_healthy_versions(), backtests=backtests)
        skill = audit["skills"][0]

        self.assertEqual(skill["status"], "cross_domain_validated")
        self.assertEqual(skill["validation_scope"], "cross_domain")
        self.assertEqual(skill["validated_domains"], ["coding", "swe"])
        self.assertFalse(skill["automatic_application_allowed"])
        self.assertTrue(skill["requires_fresh_evolution_gate"])

    def test_content_hash_tampering_and_unknown_fields_are_rejected(self) -> None:
        versions = _healthy_versions()
        tampered = deepcopy(versions)
        tampered[-1]["scores"]["objective"] = 0.999
        with self.assertRaises(EvolutionHarnessError):
            build_evolution_harness_audit(tampered)

        core = {
            key: deepcopy(value)
            for key, value in versions[0].items()
            if key != "evidence_hash"
        }
        core["raw_prompt"] = "not allowed"
        with self.assertRaises(EvolutionHarnessError):
            bind_version_evidence(core)

    def test_policy_and_epoch_are_cryptographically_bound(self) -> None:
        wrong_policy = _comparison()
        wrong_policy["policy_hash"] = _digest("different-policy")
        versions = _healthy_versions()
        versions[-1] = _version("v2", "v1", score=0.70, comparison=wrong_policy)
        with self.assertRaisesRegex(EvolutionHarnessError, "policy_hash"):
            build_evolution_harness_audit(versions)

        mixed_epoch = _healthy_versions()
        mixed_epoch[-1] = _version("v2", "v1", score=0.70, epoch_id="epoch:next")
        with self.assertRaisesRegex(EvolutionHarnessError, "frozen epoch_id"):
            build_evolution_harness_audit(mixed_epoch)

    def test_audit_validation_detects_governance_tampering(self) -> None:
        audit = build_evolution_harness_audit(_healthy_versions())
        audit["progression"]["automatic_progression_allowed"] = True
        check = validate_evolution_harness_audit(audit)
        self.assertFalse(check["ok"])
        self.assertIn("automatic_progression_allowed_forbidden", check["errors"])
        self.assertIn("audit_hash_invalid", check["errors"])

    def test_rehashing_cannot_hide_behavior_risks(self) -> None:
        audit = build_evolution_harness_audit(
            [
                _version(
                    "v1",
                    None,
                    score=0.50,
                    behavior={"loop_rate": 0.10, "test_rate": 0.80},
                ),
                _version(
                    "v2",
                    "v1",
                    score=0.90,
                    behavior={"loop_rate": 0.50, "test_rate": 0.20},
                ),
            ]
        )
        audit["risks"] = []
        audit["next_epoch_harness_challenges"] = []
        audit["progression"].update(
            {
                "decision": "eligible_for_human_review",
                "controlled_progression_allowed": True,
                "blocking_risk_codes": [],
                "next_action": "review_progression",
            }
        )
        _rehash_audit(audit)

        check = validate_evolution_harness_audit(audit)

        self.assertFalse(check["ok"])
        self.assertTrue(
            any(error.startswith("behavior_risk_missing:") for error in check["errors"])
        )
        self.assertIn("audit_semantics_mismatch", check["errors"])

    def test_rehashing_layers_and_risks_cannot_bypass_replay_validation(self) -> None:
        audit = build_evolution_harness_audit(
            [
                _version("v1", None, score=0.50),
                _version(
                    "v2",
                    "v1",
                    score=0.90,
                    behavior={"loop_rate": 0.50, "test_rate": 0.20},
                ),
            ]
        )
        audit["layers"]["behavior"].update(
            {
                "threshold_violations": [],
                "regressed_metrics": [],
                "score_behavior_divergence": False,
                "behavior_eligible": True,
            }
        )
        audit["risks"] = []
        audit["next_epoch_harness_challenges"] = []
        audit["progression"].update(
            {
                "decision": "eligible_for_human_review",
                "controlled_progression_allowed": True,
                "blocking_risk_codes": [],
                "next_action": "review_progression",
            }
        )
        _rehash_audit(audit)

        check = validate_evolution_harness_audit(audit)

        self.assertFalse(check["ok"])
        self.assertIn("audit_semantics_mismatch", check["errors"])

    def test_rehashed_governance_and_skill_gate_bypasses_are_rejected(self) -> None:
        audit = build_evolution_harness_audit(
            _healthy_versions(),
            backtests=[
                _backtest("loop-analyzer", domain="swe", split="historical"),
                _backtest("loop-analyzer", domain="swe", split="holdout"),
            ],
        )
        audit["progression"]["human_confirmation_required"] = False
        audit["policy"]["automatic_evaluator_mutation_allowed"] = True
        skill = audit["skills"][0]
        skill["requires_fresh_evolution_gate"] = False
        skill_core = {
            key: deepcopy(value) for key, value in skill.items() if key != "skill_hash"
        }
        skill["skill_hash"] = canonical_content_hash(skill_core)
        _rehash_audit(audit)

        check = validate_evolution_harness_audit(audit)

        self.assertFalse(check["ok"])
        self.assertIn("human_confirmation_required", check["errors"])
        self.assertIn("policy_evaluator_mutation_forbidden", check["errors"])
        self.assertIn("skill_automatic_application_forbidden", check["errors"])


if __name__ == "__main__":
    unittest.main()
