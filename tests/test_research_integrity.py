from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import math
from pathlib import Path
from unittest import mock

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)
from ai_scientist.utils.research_integrity import (
    ResearchIntegrityError,
    assert_claim_promotion_allowed,
    build_adaptive_state_freeze,
    build_preregistration,
    build_verification_report,
    derive_adaptive_state_hashes,
    lock_preregistration,
    save_preregistration,
    save_verification_report,
    validate_adaptive_state_freeze,
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
    def test_failed_reproduction_blocks_even_when_trajectory_structure_is_valid(
        self,
    ) -> None:
        preregistration = {
            "adaptive_state_freeze": {"state_hash": _digest("a")},
            "analysis_plan": {},
        }
        records = [
            {
                "record_id": "primary-complete",
                "study_phase": "confirmatory",
                "status": "completed",
                "producer_id": "agent:runner",
            },
            {
                "record_id": "reproduction-failed",
                "study_phase": "exploratory",
                "independent_reproduction": True,
                "status": "failed",
                "producer_id": "human:reviewer",
            },
        ]
        attestation = {
            "ok": True,
            "errors": [],
            "trajectory_hash": _digest("b"),
            "publication_ready": False,
            "publication_blocking_attempt_record_ids": ["reproduction-failed"],
            "disposed_attempt_record_ids": [],
        }
        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch(
                "ai_scientist.utils.trajectory_binding.attest_structured_trajectory",
                return_value=attestation,
            ),
        ):
            report = build_verification_report(
                preregistration,
                records,
                verifier_id="human:reviewer",
                clean_room=True,
                verification_root=td,
            )

        criteria = {item["id"]: item for item in report["criteria"]}
        self.assertFalse(criteria["confirmatory_attempt_completeness"]["passed"])
        self.assertFalse(criteria["trajectory_binding"]["passed"])
        self.assertFalse(criteria["independent_verifier"]["passed"])

    def test_verification_report_binds_each_record_to_locked_host_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = b"x,y\n1,2\n"
            files = [
                {
                    "path": "observations.csv",
                    "size_bytes": len(data),
                    "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                }
            ]
            snapshot_id = canonical_content_hash({"files": files})
            snapshot = (
                root / ".ara-store" / "datasets" / snapshot_id.removeprefix("sha256:")
            )
            snapshot.mkdir(parents=True)
            data_path = snapshot / "observations.csv"
            data_path.write_bytes(data)
            data_path.chmod(0o444)
            snapshot.chmod(0o555)
            manifest_core = {
                "schema_version": "xscientist.data-contract.v1",
                "mode": "content_addressed_snapshot_read_only",
                "ready": True,
                "source_path_disclosed": False,
                "snapshot_id": snapshot_id,
                "file_count": 1,
                "total_bytes": len(data),
                "files": files,
                "scientific_boundary": "fixed test observations",
            }
            manifest_hash = canonical_content_hash(manifest_core)
            config = root / "00_config"
            config.mkdir()
            (config / "data_manifest.json").write_text(
                json.dumps({**manifest_core, "manifest_hash": manifest_hash}),
                encoding="utf-8",
            )
            draft = build_preregistration(_idea_card(), _research_plan())
            draft["data_policy"].update(
                {
                    "data_manifest_hash": manifest_hash,
                    "data_snapshot_id": snapshot_id,
                }
            )
            locked = lock_preregistration(
                draft,
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
                        "producer_id": "agent:executor",
                        "data_manifest_hash": _digest("8"),
                        "data_snapshot_id": _digest("9"),
                    }
                ],
                verifier_id="human:reviewer",
                clean_room=True,
                verification_root=root,
            )

        criterion = {item["id"]: item for item in report["criteria"]}[
            "data_contract_binding"
        ]
        self.assertTrue(criterion["required"])
        self.assertFalse(criterion["passed"])
        self.assertIn("data_contract_binding", report["required_failures"])
        self.assertEqual(report["data_manifest_hash"], manifest_hash)
        self.assertEqual(report["data_snapshot_id"], snapshot_id)

    def test_adaptive_state_freeze_is_content_addressed_and_tamper_evident(
        self,
    ) -> None:
        draft = build_preregistration(_idea_card(), _research_plan())
        draft["data_policy"]["split_hashes"] = {"task_0": _digest("a")}
        derived = derive_adaptive_state_hashes(
            draft,
            research_vcs_head="1" * 40,
        )
        locked = lock_preregistration(
            draft,
            split_hashes={"task_0": _digest("a")},
            registered_by="planner",
            freeze_inputs={
                "research_vcs_head": "1" * 40,
                **derived,
            },
        )
        freeze = locked["adaptive_state_freeze"]

        self.assertTrue(validate_adaptive_state_freeze(freeze)["ok"])
        self.assertTrue(validate_preregistration(locked, require_locked=True)["ok"])
        protocol_before = _protocol_fidelity_hash(locked, "task_0")

        freeze["memory_state_hash"] = _digest("f")
        self.assertNotEqual(protocol_before, _protocol_fidelity_hash(locked, "task_0"))
        freeze_report = validate_adaptive_state_freeze(freeze)
        self.assertFalse(freeze_report["ok"])
        self.assertIn("adaptive_state_freeze_hash_mismatch", freeze_report["errors"])
        locked_report = validate_preregistration(locked, require_locked=True)
        self.assertFalse(locked_report["ok"])

    def test_adaptive_state_freeze_rejects_non_content_hash_inputs(self) -> None:
        with self.assertRaisesRegex(ResearchIntegrityError, "requires content hashes"):
            build_adaptive_state_freeze(
                build_preregistration(_idea_card(), _research_plan()),
                research_vcs_head="1" * 40,
                code_state_hash="git-sha-is-not-a-content-hash",
                memory_state_hash=_digest("b"),
                evaluator_spec_hash=_digest("c"),
                research_state_hash=_digest("d"),
            )

    def test_mutable_legacy_deviation_cannot_authorize_frozen_publication(
        self,
    ) -> None:
        draft = build_preregistration(_idea_card(), _research_plan())
        draft["data_policy"]["split_hashes"] = {"task_0": _digest("a")}
        derived = derive_adaptive_state_hashes(
            draft,
            research_vcs_head="1" * 40,
        )
        locked = lock_preregistration(
            draft,
            split_hashes={"task_0": _digest("a")},
            registered_by="recorder:planner",
            freeze_inputs={"research_vcs_head": "1" * 40, **derived},
        )
        locked["deviations"] = [
            {
                "reason": "added after observing the result",
                "approved_before_unblinding": True,
            }
        ]

        report = build_verification_report(
            locked,
            [],
            verifier_id="human:reviewer",
            clean_room=True,
        )

        criterion = {item["id"]: item for item in report["criteria"]}[
            "deviation_control"
        ]
        self.assertFalse(criterion["passed"])
        self.assertIn("deviation_control", report["required_failures"])

    def test_adaptive_state_freeze_rejects_unbound_aggregate_state(self) -> None:
        with self.assertRaisesRegex(ResearchIntegrityError, "must bind"):
            build_adaptive_state_freeze(
                build_preregistration(_idea_card(), _research_plan()),
                research_vcs_head="1" * 40,
                code_state_hash=_digest("a"),
                memory_state_hash=_digest("b"),
                evaluator_spec_hash=_digest("c"),
                research_state_hash=_digest("d"),
            )

    def test_adaptive_state_freeze_rejects_invented_bound_components(self) -> None:
        invented = {
            "code_state_hash": _digest("a"),
            "memory_state_hash": _digest("b"),
            "evaluator_spec_hash": _digest("c"),
        }
        with self.assertRaisesRegex(ResearchIntegrityError, "must be derived"):
            build_adaptive_state_freeze(
                build_preregistration(_idea_card(), _research_plan()),
                research_vcs_head="1" * 40,
                **invented,
                research_state_hash=_canonical_hash(
                    {"kind": "confirmatory_research_state", **invented}
                ),
            )

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
                "standard_error": 0.01,
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
            "confidence_interval": [0.08, 0.14],
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
