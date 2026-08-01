from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)
from ai_scientist.utils.research_integrity import (
    ResearchIntegrityError,
    assert_claim_promotion_allowed,
    build_preregistration,
    build_verification_report,
    lock_preregistration,
    save_preregistration,
    save_verification_report,
    validate_preregistration,
)


def _idea_card() -> dict:
    return {
        "idea_id": "idea_0",
        "title": "A falsifiable reliability study",
        "core_hypothesis": "The intervention improves accuracy over baseline A.",
        "failure_criteria": ["Accuracy does not exceed baseline A."],
    }


def _research_plan(*, dataset: str = "benchmark-v1") -> dict:
    return {
        "plan_id": "idea_0_plan",
        "tasks": [
            {
                "task_id": "task_0",
                "dataset": dataset,
                "metric": "accuracy",
                "baseline": "baseline-a",
            }
        ],
    }


def _digest(char: str) -> str:
    return "sha256:" + char * 64


class ResearchIntegrityTests(unittest.TestCase):
    def test_preregistration_must_resolve_placeholders_before_locking(self) -> None:
        draft = build_preregistration(
            _idea_card(), _research_plan(dataset="dataset_to_be_selected")
        )

        with self.assertRaises(ResearchIntegrityError):
            lock_preregistration(
                draft,
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
            )

    def test_locked_registration_detects_post_hoc_mutation(self) -> None:
        draft = build_preregistration(_idea_card(), _research_plan())
        locked = lock_preregistration(
            draft,
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        self.assertTrue(validate_preregistration(locked, require_locked=True)["ok"])

        locked["hypotheses"]["alternative"] = "A more convenient post-hoc claim."
        report = validate_preregistration(locked, require_locked=True)
        self.assertFalse(report["ok"])
        self.assertIn("registration_hash_mismatch", report["errors"])

    def test_verification_requires_blind_deterministic_multiseed_reproduction(
        self,
    ) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        records = []
        for seed in (11, 22, 33):
            records.append(
                {
                    "record_id": f"confirm-{seed}",
                    "task_id": "task_0",
                    "dataset": "benchmark-v1",
                    "metric": "accuracy",
                    "baseline_ref": "baseline-a",
                    "seed": seed,
                    "study_phase": "confirmatory",
                    "dataset_split_hash": _digest("a"),
                    "metric_provenance": "deterministic_verified",
                    "evaluator_input_hash": _digest("b"),
                    "evaluator_result_hash": _digest("c"),
                    "holdout_access": "verifier_only",
                    "producer_id": "experiment-agent",
                }
            )

        blocked = build_verification_report(
            locked,
            records,
            verifier_id="verification-agent",
            clean_room=True,
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("independent_reproduction", blocked["required_failures"])
        with self.assertRaises(ResearchIntegrityError):
            assert_claim_promotion_allowed(blocked)

        records.append(
            {
                "record_id": "reproduction-1",
                "independent_reproduction": True,
                "replicates_record_id": "confirm-11",
                "clean_room": True,
                "verifier_id": "verification-agent",
            }
        )
        verified = build_verification_report(
            locked,
            records,
            verifier_id="verification-agent",
            clean_room=True,
        )
        self.assertEqual(verified["status"], "verified")
        self.assertTrue(verified["claim_promotion_allowed"])
        assert_claim_promotion_allowed(verified)

    def test_saved_integrity_artifacts_update_pipeline_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            initialize_pipeline_contracts(root)
            locked = lock_preregistration(
                build_preregistration(_idea_card(), _research_plan()),
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
            )
            save_preregistration(root, locked, producer="test")
            blocked = build_verification_report(
                locked, [], verifier_id="verifier", clean_room=True
            )
            save_verification_report(root, blocked, producer="test")

            manifest = load_pipeline_manifest(root)
            self.assertEqual(
                manifest["artifacts"]["preregistration"]["status"], "ready"
            )
            self.assertEqual(
                manifest["artifacts"]["verification_report"]["status"],
                "blocked",
            )


if __name__ == "__main__":
    unittest.main()
