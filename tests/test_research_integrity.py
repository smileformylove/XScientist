from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import math
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
    _canonical_hash,
    _protocol_fidelity_hash,
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
    def test_locked_preregistration_rejects_missing_identity_and_nonfinite_alpha(
        self,
    ) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        locked["analysis_plan"]["alpha"] = math.nan
        locked["registration_hash"] = _canonical_hash(
            {
                key: value
                for key, value in locked.items()
                if key
                not in {
                    "created_at",
                    "locked_at",
                    "registration_hash",
                    "status",
                    "deviations",
                }
            }
        )
        locked["registered_by"] = ""
        report = validate_preregistration(locked, require_locked=True)
        self.assertFalse(report["ok"])
        self.assertIn("invalid_alpha", report["errors"])
        self.assertIn("registered_by_missing", report["errors"])

    def test_storyline_flag_cannot_promote_exploratory_record(self) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        report = build_verification_report(
            locked,
            [
                {
                    "record_id": "exploratory-1",
                    "task_id": "task_0",
                    "dataset": "benchmark-v1",
                    "metric": "accuracy",
                    "baseline_ref": "baseline-a",
                    "status": "completed",
                    "study_phase": "exploratory",
                    "entered_storyline": True,
                }
            ],
            verifier_id="verification-agent",
            clean_room=True,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertIn("confirmatory_records", report["required_failures"])

    def test_integrity_boolean_flags_are_not_truthy_strings(self) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        report = build_verification_report(
            locked,
            [
                {
                    "record_id": "reproduction-1",
                    "independent_reproduction": "false",
                    "clean_room": "false",
                    "status": "completed",
                }
            ],
            verifier_id="verification-agent",
            clean_room="false",
        )

        self.assertFalse(report["clean_room"])
        self.assertNotIn("reproduction-1", report["reproduction_record_ids"])
        self.assertIn("independent_reproduction", report["required_failures"])
        with self.assertRaises(ResearchIntegrityError):
            assert_claim_promotion_allowed(
                {"status": "verified", "claim_promotion_allowed": "false"}
            )

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

    def test_duplicate_registered_task_ids_are_rejected(self) -> None:
        draft = build_preregistration(
            _idea_card(),
            {
                "plan_id": "plan",
                "tasks": [
                    {
                        "task_id": "task_0",
                        "dataset": "benchmark-v1",
                        "metric": "accuracy",
                        "baseline": "baseline-a",
                    },
                    {
                        "task_id": "task_0",
                        "dataset": "benchmark-v1",
                        "metric": "accuracy",
                        "baseline": "baseline-a",
                    },
                ],
            },
        )
        self.assertIn(
            "duplicate_outcome_task_id", validate_preregistration(draft)["errors"]
        )

    def test_duplicate_confirmatory_record_ids_block_verification(self) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        records = [
            {
                "record_id": "same",
                "task_id": "task_0",
                "dataset": "benchmark-v1",
                "metric": "accuracy",
                "baseline_ref": "baseline-a",
                "status": "completed",
                "study_phase": "confirmatory",
                "producer_id": "experiment-agent",
            }
            for _ in range(3)
        ]
        for index, record in enumerate(records):
            record["seed"] = index
        report = build_verification_report(
            locked, records, verifier_id="verification-agent", clean_room=True
        )
        self.assertIn("record_id_integrity", report["required_failures"])

    def test_non_scalar_seed_blocks_instead_of_crashing(self) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        report = build_verification_report(
            locked,
            [
                {
                    "record_id": "confirm-1",
                    "task_id": "task_0",
                    "dataset": "benchmark-v1",
                    "metric": "accuracy",
                    "baseline_ref": "baseline-a",
                    "status": "completed",
                    "study_phase": "confirmatory",
                    "producer_id": "experiment-agent",
                    "seed": {"value": 1},
                }
            ],
            verifier_id="verification-agent",
            clean_room=True,
        )
        self.assertIn("seed_coverage", report["required_failures"])

    def test_verification_requires_blind_deterministic_multiseed_reproduction(
        self,
    ) -> None:
        root_ctx = tempfile.TemporaryDirectory()
        self.addCleanup(root_ctx.cleanup)
        root = Path(root_ctx.name)
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        records = []
        for seed in (11, 22, 33):
            result_summary = {
                "metric_mean": 0.82,
                "baseline_metric_mean": 0.70,
                "delta_vs_baseline": 0.12,
                "effect_size": 0.12,
            }
            input_path = root / f"input-{seed}.json"
            result_path = root / f"result-{seed}.json"
            input_path.write_text(json.dumps({"seed": seed}), encoding="utf-8")
            result_path.write_text(
                json.dumps(result_summary, sort_keys=True), encoding="utf-8"
            )
            input_hash = "sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest()
            result_hash = (
                "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest()
            )
            records.append(
                {
                    "record_id": f"confirm-{seed}",
                    "task_id": "task_0",
                    "dataset": "benchmark-v1",
                    "metric": "accuracy",
                    "baseline_ref": "baseline-a",
                    "status": "completed",
                    "finished_at": "2026-01-01T00:00:00+00:00",
                    "artifacts": {
                        "input": str(input_path),
                        "result": str(result_path),
                        "artifact_hashes": {
                            "input": input_hash,
                            "result": result_hash,
                        },
                    },
                    "seed": seed,
                    "study_phase": "confirmatory",
                    "dataset_split_hash": _digest("a"),
                    "metric_provenance": "deterministic_verified",
                    "evaluator_input_hash": input_hash,
                    "evaluator_result_hash": result_hash,
                    "verification_recomputed": True,
                    "verification_metric_hash": _canonical_hash(result_summary),
                    "verification_output_hash": result_hash,
                    "verification_command": "python verify_results.py",
                    "holdout_access": "verifier_only",
                    "producer_id": "experiment-agent",
                    "result_summary": result_summary,
                    "preregistration_id": locked["preregistration_id"],
                    "protocol_fidelity_hash": _protocol_fidelity_hash(locked, "task_0"),
                }
            )

        blocked = build_verification_report(
            locked,
            records,
            verifier_id="verification-agent",
            clean_room=True,
            verification_root=root,
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("independent_reproduction", blocked["required_failures"])
        with self.assertRaises(ResearchIntegrityError):
            assert_claim_promotion_allowed(blocked)

        reproduction_summary = {
            "metric_mean": 0.81,
            "baseline_metric_mean": 0.70,
            "delta_vs_baseline": 0.11,
            "effect_size": 0.11,
        }
        reproduction_input = root / "reproduction-input.json"
        reproduction_result = root / "reproduction-result.json"
        reproduction_input.write_text('{"seed":44}', encoding="utf-8")
        reproduction_result.write_text(
            json.dumps(reproduction_summary, sort_keys=True), encoding="utf-8"
        )
        reproduction_input_hash = (
            "sha256:" + hashlib.sha256(reproduction_input.read_bytes()).hexdigest()
        )
        reproduction_result_hash = (
            "sha256:" + hashlib.sha256(reproduction_result.read_bytes()).hexdigest()
        )
        records.append(
            {
                "record_id": "reproduction-1",
                "independent_reproduction": True,
                "replicates_record_id": "confirm-11",
                "task_id": "task_0",
                "dataset": "benchmark-v1",
                "metric": "accuracy",
                "baseline_ref": "baseline-a",
                "preregistration_id": locked["preregistration_id"],
                "protocol_fidelity_hash": _protocol_fidelity_hash(locked, "task_0"),
                "dataset_split_hash": _digest("a"),
                "holdout_access": "verifier_only",
                "producer_id": "reproduction-agent",
                "clean_room": True,
                "verifier_id": "verification-agent",
                "status": "completed",
                "finished_at": "2026-01-02T00:00:00+00:00",
                "artifacts": {
                    "input": str(reproduction_input),
                    "result": str(reproduction_result),
                    "artifact_hashes": {
                        "input": reproduction_input_hash,
                        "result": reproduction_result_hash,
                    },
                },
                "result_summary": reproduction_summary,
                "verification_recomputed": True,
                "evaluator_input_hash": reproduction_input_hash,
                "evaluator_result_hash": reproduction_result_hash,
                "verification_metric_hash": _canonical_hash(reproduction_summary),
                "verification_output_hash": reproduction_result_hash,
                "verification_command": "python verify_results.py",
            }
        )
        verified = build_verification_report(
            locked,
            records,
            verifier_id="verification-agent",
            clean_room=True,
            verification_root=root,
        )
        self.assertEqual(verified["status"], "verified")
        self.assertTrue(verified["claim_promotion_allowed"])
        assert_claim_promotion_allowed(verified)

        # A reproduction that points at the right primary record but changes
        # the registered task context is not an independent replay.
        records[-1]["dataset"] = "unregistered-dataset"
        mismatched = build_verification_report(
            locked,
            records,
            verifier_id="verification-agent",
            clean_room=True,
            verification_root=root,
        )
        self.assertIn("independent_reproduction", mismatched["required_failures"])

    def test_incomplete_confirmatory_attempt_cannot_be_silently_ignored(self) -> None:
        locked = lock_preregistration(
            build_preregistration(_idea_card(), _research_plan()),
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
        )
        report = build_verification_report(
            locked,
            [
                {
                    "record_id": "confirmatory-failed",
                    "task_id": "task_0",
                    "study_phase": "confirmatory",
                    "status": "failed",
                }
            ],
            verifier_id="verification-agent",
            clean_room=True,
        )
        self.assertIn("confirmatory_attempt_completeness", report["required_failures"])

    def test_malformed_preregistration_shapes_fail_closed(self) -> None:
        report = build_verification_report(
            {
                "outcomes": "not-a-list",
                "analysis_plan": "not-an-object",
                "data_policy": "not-an-object",
                "deviations": "not-a-list",
            },
            [],
            verifier_id="verification-agent",
            clean_room=True,
        )

        self.assertEqual(report["status"], "blocked")
        self.assertIn("locked_preregistration", report["required_failures"])

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
