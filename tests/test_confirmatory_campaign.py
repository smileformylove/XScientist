from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.experiment_registry import (
    append_experiment_record,
    build_experiment_record,
)
from ai_scientist.utils.research_integrity import build_preregistration
from ai_scientist.utils.trajectory_binding import (
    attempt_registry_contract_errors,
    attest_structured_trajectory,
)
from xscientist.entrypoints import research_main
from xscientist.research_commands import (
    bind_experiment_trajectory,
    confirm_paper_research,
    record_attempt_disposition,
    save_experiment,
    save_evidence,
    save_hypothesis,
    save_preregistration,
)
from xscientist.research_git import (
    ResearchGitError,
    create_checkpoint,
    reproduce_checkpoint,
)
from xscientist.research_lifecycle import ResearchLifecycle
from xscientist.research_vcs import ResearchRepository


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _prepare_data(root: Path) -> dict[str, str]:
    data = b"row,value\n1,2\n"
    files = [
        {
            "path": "observations.csv",
            "size_bytes": len(data),
            "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
    ]
    snapshot_id = canonical_content_hash({"files": files})
    snapshot = root / ".ara-store" / "datasets" / snapshot_id.removeprefix("sha256:")
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
        "scientific_boundary": "fixed held-out test observations",
    }
    manifest = {
        **manifest_core,
        "manifest_hash": canonical_content_hash(manifest_core),
    }
    config = root / "00_config"
    config.mkdir()
    (config / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "manifest_hash": manifest["manifest_hash"],
        "snapshot_id": snapshot_id,
    }


def _multi_task_plan() -> tuple[dict, dict]:
    idea = {
        "idea_id": "idea_0",
        "title": "A fixed multi-task study",
        "core_hypothesis": "The full method improves accuracy over baseline-a.",
        "failure_criteria": ["The primary effect is not positive."],
    }
    tasks = [
        {
            "task_id": "task_0",
            "goal": "Run the primary comparison",
            "dataset": "benchmark-v1",
            "metric": "accuracy",
            "baseline": "baseline-a",
            "evidence_role": "primary",
            "paired_control_task_id": None,
            "intervention_variant": "full_method",
            "stress_condition": None,
            "dependencies": [],
        },
        {
            "task_id": "task_1",
            "goal": "Remove the key component",
            "dataset": "benchmark-v1",
            "metric": "accuracy",
            "baseline": "baseline-a",
            "evidence_role": "ablation",
            "paired_control_task_id": "task_0",
            "intervention_variant": "without_key_component",
            "stress_condition": None,
            "dependencies": ["task_0"],
        },
        {
            "task_id": "task_2",
            "goal": "Run the declared shift",
            "dataset": "benchmark-shift-v1",
            "metric": "accuracy",
            "baseline": "baseline-a",
            "evidence_role": "robustness",
            "paired_control_task_id": "task_0",
            "intervention_variant": "full_method",
            "stress_condition": "declared_distribution_shift",
            "dependencies": ["task_0"],
        },
    ]
    plan = {
        "plan_id": "idea_0_plan",
        "idea_id": "idea_0",
        "tasks": tasks,
        "evidence_portfolio": {
            "required": True,
            "required_roles": ["primary", "ablation", "robustness"],
        },
    }
    return plan, build_preregistration(idea, plan)


class ConfirmatoryCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")

    def _run_json(self, args: list[str]) -> tuple[int, dict]:
        with redirect_stdout(io.StringIO()) as output:
            code = research_main([*args, "--json"])
        return code, json.loads(output.getvalue())

    def test_experiment_cli_preserves_numeric_like_result_artifact_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "123"
            artifact.write_bytes(b"durable result")
            code, result = self._run_json(
                [
                    "experiment",
                    "numeric-like result artifact path",
                    "--status",
                    "completed",
                    "--result-artifact",
                    f"result={artifact}",
                    "--repo",
                    str(root),
                ]
            )

            self.assertEqual(code, 0, result)
            payload = repository.get(result["object"]["object_id"])["payload"]
            artifact_hash = (
                "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            self.assertEqual(
                payload["result_artifact_hashes"], {"result": artifact_hash}
            )
            pointer = payload["result_artifacts"]["result"]
            self.assertEqual(pointer["logical_name"], "result")
            self.assertEqual(pointer["content_hash"], artifact_hash)
            self.assertTrue(pointer["pointer_path"].startswith("research-objects/"))
            checkpoint = repository.show()["checkpoint"]
            self.assertIn(artifact_hash, checkpoint["object_refs"])

            artifact.unlink()
            destination = Path(td) / "reproduced"
            reproduced = reproduce_checkpoint(root, destination=destination)
            self.assertTrue(reproduced["objects_complete"])
            self.assertEqual(
                (destination / pointer["logical_path"]).read_bytes(),
                b"durable result",
            )

    def test_invalid_confirmatory_attempt_does_not_ingest_result_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "result.json"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                ResearchGitError, "requires a locked preregistration"
            ):
                save_experiment(
                    str(root),
                    summary="invalid confirmatory result",
                    status="completed",
                    study_phase="confirmatory",
                    result_artifact_paths={"result": artifact},
                )

            self.assertEqual(list((root / "research-objects").glob("*.json")), [])
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertTrue(repository.status()["worktree_clean"])

    def test_multi_artifact_privacy_failure_rolls_back_new_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            accepted = Path(td) / "accepted.json"
            rejected = Path(td) / "rejected.json"
            accepted.write_text('{"score": 1}\n', encoding="utf-8")
            rejected.write_text("sk-" + "A" * 32 + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "privacy gate refused"):
                save_experiment(
                    str(root),
                    summary="atomic artifact ingest",
                    status="completed",
                    result_artifact_paths={
                        "a-accepted": accepted,
                        "b-rejected": rejected,
                    },
                )

            self.assertEqual(list((root / "research-objects").glob("*.json")), [])
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertTrue(repository.status()["worktree_clean"])

    def test_checkpoint_failure_rolls_back_attempt_and_new_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "result.json"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")
            from xscientist import research_commands

            with mock.patch.object(
                research_commands,
                "_finish",
                side_effect=ResearchGitError("injected checkpoint failure"),
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "injected checkpoint failure"
                ):
                    save_experiment(
                        str(root),
                        summary="rollback failed checkpoint",
                        status="completed",
                        result_artifact_paths={"result": artifact},
                    )

            self.assertEqual(list((root / "research-objects").glob("*.json")), [])
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertTrue(repository.status()["worktree_clean"])

    def test_post_commit_error_preserves_committed_attempt_and_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "result.json"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")
            from xscientist import research_commands

            original_head = repository.status()["head"]
            real_finish = research_commands._finish

            def commit_then_raise(*args, **kwargs):
                real_finish(*args, **kwargs)
                raise ResearchGitError("injected post-commit interruption")

            with mock.patch.object(
                research_commands,
                "_finish",
                side_effect=commit_then_raise,
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "injected post-commit interruption"
                ):
                    save_experiment(
                        str(root),
                        summary="preserve committed transition",
                        status="completed",
                        result_artifact_paths={"result": artifact},
                    )

            self.assertNotEqual(repository.status()["head"], original_head)
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 1)
            self.assertEqual(len(list((root / "research-objects").glob("*.json"))), 1)
            self.assertTrue(repository.status()["worktree_clean"])

    def test_experiment_transaction_lock_spans_validation_through_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "result.json"
            contender_source = Path(td) / "contender.bin"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")
            contender_source.write_bytes(b"contender")
            started = threading.Event()
            completed = threading.Event()
            contender: threading.Thread | None = None
            real_validate = ResearchLifecycle.validate_experiment_attempt

            def run_contender() -> None:
                from xscientist.research_git import add_research_object

                started.set()
                add_research_object(
                    root,
                    contender_source,
                    logical_path="results/contender.bin",
                )
                completed.set()

            def validate_then_contend(lifecycle, *args, **kwargs):
                nonlocal contender
                prepared = real_validate(lifecycle, *args, **kwargs)
                contender = threading.Thread(target=run_contender)
                contender.start()
                self.assertTrue(started.wait(timeout=2))
                time.sleep(0.1)
                self.assertFalse(completed.is_set())
                return prepared

            with mock.patch.object(
                ResearchLifecycle,
                "validate_experiment_attempt",
                new=validate_then_contend,
            ):
                saved = save_experiment(
                    str(root),
                    summary="transaction lock",
                    status="completed",
                    result_artifact_paths={"result": artifact},
                )

            self.assertIsNotNone(saved["checkpoint"].commit)
            self.assertIsNotNone(contender)
            contender.join(timeout=5)
            self.assertTrue(completed.is_set())

    def test_experiment_result_artifact_reports_damaged_and_missing_cas(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "result.json"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")
            code, result = self._run_json(
                [
                    "experiment",
                    "CAS integrity",
                    "--status",
                    "completed",
                    "--result-artifact",
                    f"metrics={artifact}",
                    "--repo",
                    str(root),
                ]
            )

            self.assertEqual(code, 0, result)
            payload = repository.get(result["object"]["object_id"])["payload"]
            artifact_hash = payload["result_artifact_hashes"]["metrics"]
            pointer_path = root / payload["result_artifacts"]["metrics"]["pointer_path"]
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            store_path = root / pointer["store_relpath"]

            store_path.write_bytes(b"tampered")
            damaged = reproduce_checkpoint(root)
            self.assertFalse(damaged["objects_complete"])
            self.assertEqual(damaged["damaged_objects"], [artifact_hash])

            store_path.unlink()
            missing = reproduce_checkpoint(root)
            self.assertFalse(missing["objects_complete"])
            self.assertEqual(missing["missing_objects"], [artifact_hash])

    def test_experiment_result_artifact_rejects_nonportable_filename(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "NUL.txt"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")

            code, result = self._run_json(
                [
                    "experiment",
                    "unsafe result artifact filename",
                    "--status",
                    "completed",
                    "--result-artifact",
                    f"metrics={artifact}",
                    "--repo",
                    str(root),
                ]
            )

            self.assertEqual(code, 2, result)
            self.assertFalse(result["ok"])
            self.assertIn(
                "unsafe cross-platform research object filename",
                result["error"]["message"],
            )
            self.assertFalse(list((root / "research-objects").glob("*.json")))

    def test_duplicate_result_bytes_preserve_each_logical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            first = Path(td) / "first.json"
            second = Path(td) / "second.json"
            first.write_text('{"score": 1}\n', encoding="utf-8")
            second.write_bytes(first.read_bytes())

            code, result = self._run_json(
                [
                    "experiment",
                    "duplicate content with distinct meanings",
                    "--status",
                    "completed",
                    "--result-artifact",
                    f"primary={first}",
                    "--result-artifact",
                    f"replicate={second}",
                    "--repo",
                    str(root),
                ]
            )

            self.assertEqual(code, 0, result)
            payload = repository.get(result["object"]["object_id"])["payload"]
            pointers = payload["result_artifacts"]
            self.assertEqual(
                pointers["primary"]["content_hash"],
                pointers["replicate"]["content_hash"],
            )
            self.assertNotEqual(
                pointers["primary"]["pointer_path"],
                pointers["replicate"]["pointer_path"],
            )
            destination = Path(td) / "reproduction"
            reproduction = reproduce_checkpoint(root, destination=destination)
            self.assertTrue(reproduction["objects_complete"])
            for pointer in pointers.values():
                self.assertEqual(
                    (destination / pointer["logical_path"]).read_bytes(),
                    first.read_bytes(),
                )

    def test_multiple_confirmatory_records_preserve_every_frozen_component(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            hypothesis = save_hypothesis(
                str(root),
                statement="Method A improves accuracy",
                falsifier="delta <= 0",
            )
            registration = save_preregistration(
                str(root),
                hypothesis_id=hypothesis["object"].object_id,
                dataset="benchmark-v1",
                metric="accuracy",
                baseline="baseline-a",
                split_hash=_digest("a"),
                registered_by="lead-researcher",
            )
            registration_id = registration["object"].object_id
            plan_id = registration["related"][0].object_id

            first = save_experiment(
                str(root),
                summary="first sealed run",
                status="completed",
                study_phase="confirmatory",
                task_id="primary",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                configuration={"task": "primary", "seed": 1},
                result_artifact_hashes={"result": _digest("d")},
            )
            second = save_experiment(
                str(root),
                summary="second sealed run",
                status="failed",
                study_phase="confirmatory",
                task_id="primary",
                plan_id=plan_id,
                preregistration_id=registration_id,
                failure_class="negative_result",
                producer_id="executor:sealed-runner",
            )

            saved = repository.get(second["object"].object_id)["payload"]
            attestation = saved["frozen_path_attestation"]
            self.assertEqual(
                attestation["prior_confirmatory_attempt_ids"],
                [first["object"].object_id],
            )
            self.assertEqual(
                set(attestation["frozen_components"]),
                {"hypothesis", "method", "code", "memory", "protocol", "evaluator"},
            )
            self.assertTrue(
                all(
                    item["unchanged"]
                    for item in attestation["frozen_components"].values()
                )
            )

            save_hypothesis(
                str(root),
                statement="Post-freeze adaptive hypothesis",
                falsifier="unsupported",
            )
            with self.assertRaisesRegex(
                ResearchGitError, "non-confirmatory post-freeze transition"
            ):
                save_experiment(
                    str(root),
                    summary="must be blocked",
                    status="completed",
                    study_phase="confirmatory",
                    task_id="primary",
                    plan_id=plan_id,
                    preregistration_id=registration_id,
                )

    def test_terminal_negative_requires_host_rehashed_artifact_and_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            data = _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "preregistration.json").write_text(
                json.dumps(draft), encoding="utf-8"
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated multi-task plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )
            confirmed = confirm_paper_research(
                str(root),
                registered_by="lead-researcher",
                split_hashes={
                    "task_0": _digest("a"),
                    "task_1": _digest("b"),
                    "task_2": _digest("c"),
                },
                data_manifest_hash=data["manifest_hash"],
                data_snapshot_id=data["snapshot_id"],
            )
            registration_id = confirmed["object"].object_id
            plan_id = confirmed["related"][0].object_id
            locked = json.loads((root / "preregistration.json").read_text())

            artifact_path = root / ".ara-store" / "terminal-negative.json"
            artifact_path.write_text(
                json.dumps({"accuracy": 0.49, "conclusion": "negative"}),
                encoding="utf-8",
            )
            artifact_hash = (
                "sha256:" + hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            )
            saved = save_experiment(
                str(root),
                summary="sealed run reached the registered negative outcome",
                status="failed",
                study_phase="confirmatory",
                task_id="task_0",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                failure_class="scientific_negative_result",
                result_artifact_hashes={"result": artifact_hash},
            )
            attempt_id = saved["object"].object_id
            attempt_payload = repository.get(attempt_id)["payload"]
            row = build_experiment_record(
                record_id="task_0-negative-1",
                task_id="task_0",
                dataset="benchmark-v1",
                metric="accuracy",
                baseline_ref="baseline-a",
                status="failed",
                result_summary={"accuracy": 0.49},
                artifacts={"artifact_hashes": {"result": artifact_hash}},
                error_type="scientific_negative_result",
                error_message="registered negative terminal outcome",
                study_phase="confirmatory",
                preregistration_id=locked["preregistration_id"],
                protocol_fidelity_hash=attempt_payload["protocol_fidelity_hash"],
                adaptive_state_hash=attempt_payload["adaptive_state_hash"],
                research_state_hash=attempt_payload["research_state_hash"],
                post_freeze_adaptation=False,
                dataset_split_hash=attempt_payload["dataset_split_hash"],
                data_manifest_hash=attempt_payload["data_manifest_hash"],
                data_snapshot_id=attempt_payload["data_snapshot_id"],
                evidence_role=attempt_payload["evidence_role"],
                paired_control_task_id=attempt_payload["paired_control_task_id"],
                intervention_variant=attempt_payload["intervention_variant"],
                stress_condition=attempt_payload["stress_condition"],
                producer_id=attempt_payload["producer_id"],
            )
            append_experiment_record(root, row)
            bind_experiment_trajectory(
                str(root),
                record_id="task_0-negative-1",
                attempt_id=attempt_id,
            )
            with self.assertRaisesRegex(
                ResearchGitError,
                "requires --negative-result-artifact",
            ):
                record_attempt_disposition(
                    str(root),
                    record_id="task_0-negative-1",
                    disposition="terminal_negative",
                    reason="A caller assertion alone must not clear the blocker.",
                )
            evidence = save_evidence(
                str(root),
                result_summary="The registered accuracy outcome is negative.",
                attempt_ids=[attempt_id],
                metrics={"accuracy": 0.49},
            )

            code, disposition = self._run_json(
                [
                    "attempt-disposition",
                    "--paper-dir",
                    str(root),
                    "--record-id",
                    "task_0-negative-1",
                    "--disposition",
                    "terminal_negative",
                    "--reason",
                    "The preregistered scientific negative result is retained.",
                    # This is intentionally repository-relative while pytest's cwd
                    # is outside the temporary repository.
                    "--negative-result-artifact",
                    ".ara-store/terminal-negative.json",
                    "--negative-result-evidence",
                    evidence["object"].object_id,
                ]
            )
            self.assertEqual(code, 0, disposition)
            disposition_id = disposition["object"]["object_id"]
            payload = repository.get(disposition_id)["payload"]
            self.assertEqual(
                payload["negative_result_artifact"]["content_hash"], artifact_hash
            )
            self.assertEqual(
                payload["negative_result_evidence_id"], evidence["object"].object_id
            )
            ready = attest_structured_trajectory(root, locked, [row])
            self.assertTrue(ready["ok"], ready["errors"])
            self.assertTrue(ready["publication_ready"], ready)

            artifact_path.write_text('{"accuracy": 1.0}', encoding="utf-8")
            tampered = attest_structured_trajectory(root, locked, [row])
            self.assertFalse(tampered["ok"])
            self.assertFalse(tampered["publication_ready"])
            self.assertTrue(
                any(
                    "terminal_negative_artifact_receipt_invalid" in error
                    for error in tampered["errors"]
                ),
                tampered["errors"],
            )

    def test_attestation_rejects_raw_git_gap_before_normal_confirmatory_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")

            raw_path = root / "raw-history.txt"
            raw_path.write_text("unattested backend transition\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", "--", raw_path.name],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "raw transition without checkpoint"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            migration_note = root / "migration-note.md"
            migration_note.write_text(
                "Explicitly retain the legacy boundary for repair.\n",
                encoding="utf-8",
            )
            create_checkpoint(
                root,
                stage="migration",
                subject="record explicit legacy boundary",
                only_paths=[migration_note.name],
            )

            data = _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(
                json.dumps(plan),
                encoding="utf-8",
            )
            (root / "preregistration.json").write_text(
                json.dumps(draft),
                encoding="utf-8",
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated multi-task plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )
            confirmed = confirm_paper_research(
                str(root),
                registered_by="lead-researcher",
                split_hashes={
                    "task_0": _digest("a"),
                    "task_1": _digest("b"),
                    "task_2": _digest("c"),
                },
                data_manifest_hash=data["manifest_hash"],
                data_snapshot_id=data["snapshot_id"],
            )
            registration_id = confirmed["object"].object_id
            plan_id = confirmed["related"][0].object_id
            locked = json.loads((root / "preregistration.json").read_text())

            configuration = {"task": "task_0", "seed": 7}
            result_hash = _digest("f")
            saved = save_experiment(
                str(root),
                summary="sealed runner completed normally",
                status="completed",
                study_phase="confirmatory",
                task_id="task_0",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                configuration=configuration,
                result_artifact_hashes={"result": result_hash},
            )
            attempt_id = saved["object"].object_id
            attempt_payload = repository.get(attempt_id)["payload"]
            row = build_experiment_record(
                record_id="task_0-completed-raw-gap",
                task_id="task_0",
                dataset="benchmark-v1",
                metric="accuracy",
                baseline_ref="baseline-a",
                config=configuration,
                status="completed",
                result_summary={"accuracy": 0.8},
                artifacts={"artifact_hashes": {"result": result_hash}},
                study_phase="confirmatory",
                preregistration_id=locked["preregistration_id"],
                protocol_fidelity_hash=attempt_payload["protocol_fidelity_hash"],
                adaptive_state_hash=attempt_payload["adaptive_state_hash"],
                research_state_hash=attempt_payload["research_state_hash"],
                post_freeze_adaptation=False,
                dataset_split_hash=attempt_payload["dataset_split_hash"],
                data_manifest_hash=attempt_payload["data_manifest_hash"],
                data_snapshot_id=attempt_payload["data_snapshot_id"],
                evidence_role=attempt_payload["evidence_role"],
                paired_control_task_id=attempt_payload["paired_control_task_id"],
                intervention_variant=attempt_payload["intervention_variant"],
                stress_condition=attempt_payload["stress_condition"],
                producer_id=attempt_payload["producer_id"],
            )
            append_experiment_record(root, row)
            bind_experiment_trajectory(
                str(root),
                record_id="task_0-completed-raw-gap",
                attempt_id=attempt_id,
            )

            attestation = attest_structured_trajectory(root, locked, [row])

            self.assertFalse(attestation["ok"])
            self.assertFalse(attestation["publication_ready"])
            self.assertIsNone(attestation["trajectory_hash"])
            self.assertIsNone(attestation["trajectory_projection"])
            self.assertTrue(
                any(
                    error.startswith("trajectory_projection_invalid:")
                    for error in attestation["errors"]
                ),
                attestation["errors"],
            )

    def test_failed_queue_checkpoint_reports_one_locked_object_and_recovery(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            save_hypothesis(
                str(root),
                statement="The full method improves accuracy over baseline-a.",
                falsifier="The primary effect is not positive.",
            )
            _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "preregistration.json").write_text(
                json.dumps(draft), encoding="utf-8"
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )

            from xscientist import research_commands

            real_checkpoint = research_commands.create_checkpoint
            calls = 0

            def fail_second_checkpoint(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise ResearchGitError("simulated mirror checkpoint failure")
                return real_checkpoint(*args, **kwargs)

            with mock.patch.object(
                research_commands,
                "create_checkpoint",
                side_effect=fail_second_checkpoint,
            ):
                with self.assertRaisesRegex(
                    ResearchGitError,
                    "INCOMPLETE_CONFIRMATORY_CAMPAIGN.*Do not create a second lock",
                ) as caught:
                    confirm_paper_research(
                        str(root),
                        registered_by="lead-researcher",
                        split_hashes={
                            "task_0": _digest("a"),
                            "task_1": _digest("b"),
                            "task_2": _digest("c"),
                        },
                    )

            self.assertIn("xscientist research stage", str(caught.exception))
            registrations = repository.objects(kind="preregistration", state="locked")
            self.assertEqual(len(registrations), 1)
            self.assertEqual(
                json.loads((root / "preregistration.json").read_text())["status"],
                "locked",
            )
            self.assertTrue((root / "confirmatory_queue.json").is_file())
            self.assertFalse(repository.status()["worktree_clean"])

    def test_confirm_rejects_symlinked_queue_before_committing_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            save_hypothesis(
                str(root),
                statement="The full method improves accuracy over baseline-a.",
                falsifier="The primary effect is not positive.",
            )
            _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "preregistration.json").write_text(
                json.dumps(draft), encoding="utf-8"
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )
            outside = Path(td) / "outside-queue.json"
            sentinel = b'{"secret":"must-not-be-materialized"}'
            outside.write_bytes(sentinel)
            queue_path = root / "confirmatory_queue.json"

            from xscientist import research_commands

            real_guard = research_commands._ensure_direct_save_is_safe

            def validate_then_inject_symlink(*args, **kwargs):
                result = real_guard(*args, **kwargs)
                try:
                    queue_path.symlink_to(outside)
                except OSError as exc:  # pragma: no cover - platform capability
                    self.skipTest(f"file symlinks unavailable: {exc}")
                return result

            with mock.patch.object(
                research_commands,
                "_ensure_direct_save_is_safe",
                side_effect=validate_then_inject_symlink,
            ):
                with self.assertRaisesRegex(
                    ResearchGitError,
                    "existing confirmatory queue is unsafe or unreadable",
                ):
                    confirm_paper_research(
                        str(root),
                        registered_by="lead-researcher",
                        split_hashes={
                            "task_0": _digest("a"),
                            "task_1": _digest("b"),
                            "task_2": _digest("c"),
                        },
                    )

            self.assertEqual(
                repository.objects(kind="preregistration", state="locked"),
                [],
            )
            self.assertEqual(
                json.loads((root / "preregistration.json").read_text())["status"],
                "draft",
            )
            self.assertTrue(queue_path.is_symlink())
            self.assertEqual(outside.read_bytes(), sentinel)

    def test_confirm_command_locks_data_tasks_and_queue_then_records_in_sequence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            data = _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "preregistration.json").write_text(
                json.dumps(draft), encoding="utf-8"
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated multi-task plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )

            code, result = self._run_json(
                [
                    "confirm",
                    "--paper-dir",
                    str(root),
                    "--registered-by",
                    "lead-researcher",
                    "--split",
                    f"task_0={_digest('a')}",
                    "--split",
                    f"task_1={_digest('b')}",
                    "--split",
                    f"task_2={_digest('c')}",
                    "--data-manifest-hash",
                    data["manifest_hash"],
                    "--data-snapshot-id",
                    data["snapshot_id"],
                ]
            )

            self.assertEqual(code, 0, result)
            self.assertEqual(
                result["queue"]["data_manifest_hash"], data["manifest_hash"]
            )
            self.assertEqual(result["queue"]["data_snapshot_id"], data["snapshot_id"])
            by_role = {item["evidence_role"]: item for item in result["queue"]["tasks"]}
            self.assertEqual(set(by_role), {"primary", "ablation", "robustness"})
            self.assertEqual(by_role["ablation"]["paired_control_task_id"], "task_0")
            self.assertEqual(
                by_role["ablation"]["transformation_contract"]["evidence_role"],
                "ablation",
            )
            self.assertEqual(
                by_role["robustness"]["stress_condition"],
                "declared_distribution_shift",
            )
            self.assertTrue(repository.status()["worktree_clean"])

            registration_id = result["object"]["object_id"]
            plan_id = result["related_objects"][0]["object_id"]
            self.assertNotIn(
                "completed",
                by_role["primary"]["record_command_template"],
            )
            terminal_arguments = by_role["primary"]["record_command_terminal_arguments"]
            self.assertIn("--result-artifact", terminal_arguments["completed"])
            self.assertNotIn("--result-artifact", terminal_arguments["unsuccessful"])
            self.assertEqual(
                terminal_arguments["unsuccessful"][:2],
                ["--failure-class", "{FAILURE_CLASS}"],
            )
            self.assertEqual(
                result["queue"]["queue_contract_hash"],
                json.loads((root / "preregistration.json").read_text())[
                    "confirmatory_campaign"
                ]["queue_contract_hash"],
            )

            for task_id in ("task_0", "task_1", "task_2"):
                result_path = Path(td) / f"{task_id}-result.json"
                result_path.write_text(
                    json.dumps({"task_id": task_id, "accuracy": 0.8}),
                    encoding="utf-8",
                )
                saved = save_experiment(
                    str(root),
                    summary=f"sealed result for {task_id}",
                    status="completed",
                    study_phase="confirmatory",
                    task_id=task_id,
                    plan_id=plan_id,
                    preregistration_id=registration_id,
                    producer_id="executor:sealed-runner",
                    configuration={"task": task_id, "seed": 1},
                    result_artifact_hashes={
                        "result": "sha256:"
                        + hashlib.sha256(result_path.read_bytes()).hexdigest()
                    },
                    result_artifact_paths={"result": result_path},
                )
                payload = repository.get(saved["object"].object_id)["payload"]
                self.assertEqual(payload["data_manifest_hash"], data["manifest_hash"])
                self.assertEqual(payload["data_snapshot_id"], data["snapshot_id"])
                self.assertIn("result", payload["result_artifacts"])

    def test_registry_attempt_binding_checkpoint_bijection_and_failed_disposition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "study"
            repository = ResearchRepository.init(root, question="Fixed question")
            data = _prepare_data(root)
            plan, draft = _multi_task_plan()
            (root / "research_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "preregistration.json").write_text(
                json.dumps(draft), encoding="utf-8"
            )
            create_checkpoint(
                root,
                stage="planning",
                subject="checkpoint generated multi-task plan",
                only_paths=[
                    "00_config/data_manifest.json",
                    "research_plan.json",
                    "preregistration.json",
                ],
            )
            confirmed = confirm_paper_research(
                str(root),
                registered_by="lead-researcher",
                split_hashes={
                    "task_0": _digest("a"),
                    "task_1": _digest("b"),
                    "task_2": _digest("c"),
                },
                data_manifest_hash=data["manifest_hash"],
                data_snapshot_id=data["snapshot_id"],
            )
            registration_id = confirmed["object"].object_id
            plan_id = confirmed["related"][0].object_id
            locked = json.loads((root / "preregistration.json").read_text())

            saved_attempt = save_experiment(
                str(root),
                summary="sealed runner failed before producing a result",
                status="failed",
                study_phase="confirmatory",
                task_id="task_0",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                failure_class="runtime_error",
            )
            attempt_id = saved_attempt["object"].object_id
            attempt = repository.get(attempt_id)
            attempt_payload = attempt["payload"]
            row = build_experiment_record(
                record_id="task_0-failed-1",
                task_id="task_0",
                dataset="benchmark-v1",
                metric="accuracy",
                baseline_ref="baseline-a",
                status="failed",
                error_type="runtime_error",
                error_message="sealed runner exited non-zero",
                study_phase="confirmatory",
                preregistration_id=locked["preregistration_id"],
                protocol_fidelity_hash=attempt_payload["protocol_fidelity_hash"],
                adaptive_state_hash=attempt_payload["adaptive_state_hash"],
                research_state_hash=attempt_payload["research_state_hash"],
                post_freeze_adaptation=False,
                dataset_split_hash=attempt_payload["dataset_split_hash"],
                data_manifest_hash=attempt_payload["data_manifest_hash"],
                data_snapshot_id=attempt_payload["data_snapshot_id"],
                evidence_role=attempt_payload["evidence_role"],
                paired_control_task_id=attempt_payload["paired_control_task_id"],
                intervention_variant=attempt_payload["intervention_variant"],
                stress_condition=attempt_payload["stress_condition"],
                producer_id=attempt_payload["producer_id"],
            )
            append_experiment_record(root, row)

            binding = bind_experiment_trajectory(
                str(root),
                record_id="task_0-failed-1",
                attempt_id=attempt_id,
            )
            binding_id = binding["object"].object_id
            binding_checkpoint = binding["checkpoint"]
            binding_origin = repository.blame(binding_id)["origin"]
            binding_commit = str(binding_checkpoint.commit)
            shown_binding = repository.show(binding_commit)

            self.assertEqual(binding_origin["commit"], binding_commit)
            self.assertTrue(shown_binding["checkpoint_hash_valid"])
            self.assertIn(
                "experiment_registry.jsonl",
                shown_binding["checkpoint"]["changed_paths"],
            )
            self.assertEqual(
                [
                    path
                    for path in shown_binding["checkpoint"]["changed_paths"]
                    if path.startswith(".xscientist/objects/")
                ],
                [f".xscientist/objects/gate_decision/{binding_id}.json"],
            )

            quarantined = attest_structured_trajectory(root, locked, [row])
            self.assertFalse(quarantined["ok"])
            self.assertIsNone(quarantined["trajectory_hash"])
            self.assertIn(
                "trajectory_disposition_count_invalid:task_0-failed-1",
                quarantined["errors"],
            )

            configuration = {"task": "task_0", "seed": 7}
            result_hash = _digest("f")
            completed_saved = save_experiment(
                str(root),
                summary="sealed runner completed the retry",
                status="completed",
                study_phase="confirmatory",
                task_id="task_0",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                configuration=configuration,
                result_artifact_hashes={"result": result_hash},
            )
            completed_attempt_id = completed_saved["object"].object_id
            completed_attempt = repository.get(completed_attempt_id)
            completed_payload = completed_attempt["payload"]
            completed_row = build_experiment_record(
                record_id="task_0-completed-2",
                task_id="task_0",
                dataset="benchmark-v1",
                metric="accuracy",
                baseline_ref="baseline-a",
                config=configuration,
                status="completed",
                result_summary={"accuracy": 0.8},
                artifacts={"artifact_hashes": {"result": result_hash}},
                study_phase="confirmatory",
                preregistration_id=locked["preregistration_id"],
                protocol_fidelity_hash=completed_payload["protocol_fidelity_hash"],
                adaptive_state_hash=completed_payload["adaptive_state_hash"],
                research_state_hash=completed_payload["research_state_hash"],
                post_freeze_adaptation=False,
                dataset_split_hash=completed_payload["dataset_split_hash"],
                data_manifest_hash=completed_payload["data_manifest_hash"],
                data_snapshot_id=completed_payload["data_snapshot_id"],
                evidence_role=completed_payload["evidence_role"],
                paired_control_task_id=completed_payload["paired_control_task_id"],
                intervention_variant=completed_payload["intervention_variant"],
                stress_condition=completed_payload["stress_condition"],
                producer_id=completed_payload["producer_id"],
            )
            self.assertEqual(
                attempt_registry_contract_errors(
                    completed_row,
                    completed_attempt,
                    registration_object_id=registration_id,
                ),
                [],
            )
            append_experiment_record(root, completed_row)
            completed_binding = bind_experiment_trajectory(
                str(root),
                record_id="task_0-completed-2",
                attempt_id=completed_attempt_id,
            )

            retry_not_yet_disposed = attest_structured_trajectory(
                root,
                locked,
                [row, completed_row],
            )
            self.assertFalse(retry_not_yet_disposed["ok"])
            self.assertFalse(retry_not_yet_disposed["publication_ready"])
            self.assertIn(
                "trajectory_disposition_count_invalid:task_0-failed-1",
                retry_not_yet_disposed["errors"],
            )
            disposition = record_attempt_disposition(
                str(root),
                record_id="task_0-failed-1",
                disposition="technical_failure_retried",
                retry_record_id="task_0-completed-2",
                reason="The sealed process failed before output; the bound retry completed.",
            )
            disposition_id = disposition["object"].object_id
            disposition_object = repository.get(disposition_id)
            self.assertEqual(
                repository.get(binding_id)["actor"]["authority"],
                "deterministic_gate",
            )
            self.assertEqual(disposition_object["actor"]["authority"], "recorder")
            self.assertEqual(
                disposition_object["actor"]["actor_id"],
                "recorder:xscientist-user",
            )
            disposition_commit = str(
                repository.blame(disposition_id)["origin"]["commit"]
            )
            disposition_path = (
                f".xscientist/objects/gate_decision/{disposition_id}.json"
            )
            lineage = ResearchLifecycle(repository)
            self.assertEqual(
                lineage._validate_confirmatory_gate_path(
                    disposition_path,
                    commit=disposition_commit,
                    current_head=str(repository.status()["head"]),
                    preregistration_id=registration_id,
                ),
                (disposition_id, "xscientist.attempt-disposition.v1"),
            )
            real_get = repository.get
            tamper_cases = (
                (
                    "actor authority",
                    lambda item: item["actor"].update(
                        {"authority": "deterministic_gate"}
                    ),
                    "actor authority is invalid",
                ),
                (
                    "payload hash",
                    lambda item: item["payload"].update({"reason": "tampered"}),
                    "integrity hash is invalid",
                ),
                (
                    "retry relation",
                    lambda item: item.update(
                        {
                            "relations": [
                                relation
                                for relation in item["relations"]
                                if relation.get("role") != "retry"
                            ]
                        }
                    ),
                    "retry binding is invalid",
                ),
            )
            for label, mutate, expected_error in tamper_cases:
                with self.subTest(lifecycle_tamper=label):

                    def tampered_get(candidate_id: str) -> dict:
                        item = json.loads(json.dumps(real_get(candidate_id)))
                        if candidate_id == disposition_id:
                            mutate(item)
                        return item

                    with mock.patch.object(
                        repository,
                        "get",
                        side_effect=tampered_get,
                    ):
                        with self.assertRaisesRegex(
                            ResearchGitError,
                            expected_error,
                        ):
                            lineage._validate_confirmatory_gate_path(
                                disposition_path,
                                commit=disposition_commit,
                                current_head=str(repository.status()["head"]),
                                preregistration_id=registration_id,
                            )
            complete = attest_structured_trajectory(
                root,
                locked,
                [row, completed_row],
            )
            self.assertTrue(complete["ok"], complete["errors"])
            self.assertTrue(complete["publication_ready"])
            self.assertEqual(
                complete["attempt_object_ids"],
                sorted([attempt_id, completed_attempt_id]),
            )
            self.assertEqual(
                complete["binding_object_ids"],
                sorted([binding_id, completed_binding["object"].object_id]),
            )
            self.assertEqual(len(complete["bindings"]), 2)
            self.assertEqual(len(complete["checkpoint_hashes"]), 5)
            self.assertEqual(
                complete["disposed_attempt_record_ids"], ["task_0-failed-1"]
            )
            self.assertEqual(complete["publication_blocking_attempt_record_ids"], [])
            self.assertEqual(
                complete["disposition_object_ids"], [disposition["object"].object_id]
            )
            disposition_checkpoint = repository.show(disposition_commit)["checkpoint"]
            disposition_summary = complete["dispositions"][0]
            self.assertEqual(
                disposition_summary["disposition_origin_commit"],
                disposition_commit,
            )
            self.assertEqual(
                disposition_summary["disposition_checkpoint_hash"],
                disposition_checkpoint["content_hash"],
            )
            self.assertIn(
                disposition_checkpoint["content_hash"],
                complete["checkpoint_hashes"],
            )
            binding_by_record = {
                item["record_id"]: item for item in complete["bindings"]
            }
            self.assertEqual(
                binding_by_record["task_0-failed-1"]["binding_origin_commit"],
                binding_commit,
            )
            self.assertEqual(
                binding_by_record["task_0-failed-1"]["binding_checkpoint_hash"],
                binding_checkpoint.content_hash,
            )

            configuration_tamper = {
                **completed_row,
                "configuration_hash": _digest("e"),
            }
            self.assertIn(
                "trajectory_attempt_configuration_mismatch",
                attempt_registry_contract_errors(
                    configuration_tamper,
                    completed_attempt,
                    registration_object_id=registration_id,
                ),
            )
            artifact_tamper = {
                **completed_row,
                "artifacts": {"artifact_hashes": {"result": _digest("e")}},
            }
            self.assertIn(
                "trajectory_attempt_artifact_mismatch",
                attempt_registry_contract_errors(
                    artifact_tamper,
                    completed_attempt,
                    registration_object_id=registration_id,
                ),
            )
            for tampered_row in (configuration_tamper, artifact_tamper):
                tampered = attest_structured_trajectory(
                    root,
                    locked,
                    [row, tampered_row],
                )
                self.assertFalse(tampered["ok"])
                self.assertIsNone(tampered["trajectory_hash"])
                self.assertIn(
                    "trajectory_binding_payload_invalid:task_0-completed-2",
                    tampered["errors"],
                )

            excluded_saved = save_experiment(
                str(root),
                summary="a second sealed runner failed",
                status="failed",
                study_phase="confirmatory",
                task_id="task_1",
                plan_id=plan_id,
                preregistration_id=registration_id,
                producer_id="executor:sealed-runner",
                failure_class="runtime_error",
            )
            excluded_attempt_id = excluded_saved["object"].object_id
            excluded_attempt = repository.get(excluded_attempt_id)
            excluded_payload = excluded_attempt["payload"]
            excluded_row = build_experiment_record(
                record_id="task_1-failed-1",
                task_id="task_1",
                dataset="benchmark-v1",
                metric="accuracy",
                baseline_ref="baseline-a",
                status="failed",
                error_type="runtime_error",
                error_message="sealed runner exited non-zero",
                study_phase="confirmatory",
                preregistration_id=locked["preregistration_id"],
                protocol_fidelity_hash=excluded_payload["protocol_fidelity_hash"],
                adaptive_state_hash=excluded_payload["adaptive_state_hash"],
                research_state_hash=excluded_payload["research_state_hash"],
                post_freeze_adaptation=False,
                dataset_split_hash=excluded_payload["dataset_split_hash"],
                data_manifest_hash=excluded_payload["data_manifest_hash"],
                data_snapshot_id=excluded_payload["data_snapshot_id"],
                evidence_role=excluded_payload["evidence_role"],
                paired_control_task_id=excluded_payload["paired_control_task_id"],
                intervention_variant=excluded_payload["intervention_variant"],
                stress_condition=excluded_payload["stress_condition"],
                producer_id=excluded_payload["producer_id"],
            )
            append_experiment_record(root, excluded_row)
            bind_experiment_trajectory(
                str(root),
                record_id="task_1-failed-1",
                attempt_id=excluded_attempt_id,
            )
            excluded_disposition = record_attempt_disposition(
                str(root),
                record_id="task_1-failed-1",
                disposition="excluded_with_reason",
                reason="This explanation remains auditable but cannot erase a failure.",
            )
            non_resolving = attest_structured_trajectory(
                root,
                locked,
                [row, completed_row, excluded_row],
            )
            self.assertTrue(non_resolving["ok"], non_resolving["errors"])
            self.assertFalse(non_resolving["publication_ready"])
            self.assertEqual(
                non_resolving["recorded_disposition_attempt_record_ids"],
                ["task_0-failed-1", "task_1-failed-1"],
            )
            self.assertEqual(
                non_resolving["disposed_attempt_record_ids"],
                ["task_0-failed-1"],
            )
            self.assertEqual(
                non_resolving["publication_blocking_attempt_record_ids"],
                ["task_1-failed-1"],
            )
            disposition_by_attempt = {
                item["attempt_record_id"]: item
                for item in non_resolving["dispositions"]
            }
            self.assertTrue(
                disposition_by_attempt["task_0-failed-1"]["resolves_publication_block"]
            )
            self.assertFalse(
                disposition_by_attempt["task_1-failed-1"]["resolves_publication_block"]
            )
            self.assertEqual(len(non_resolving["checkpoint_hashes"]), 8)
            self.assertIn(
                excluded_disposition["object"].object_id,
                non_resolving["disposition_object_ids"],
            )


if __name__ == "__main__":
    unittest.main()
