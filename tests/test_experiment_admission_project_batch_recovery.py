from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from ai_scientist.apps.project import (
    _compact_research_result,
    _experiment_admission_execution_lease,
    _preflight_local_research_experiment_admission,
    _reconcile_required_experiment_admission_runtime,
    _record_local_research_attempt_objects,
)
from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.resources import resolve_bfts_config_path
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist import experiment_admission_runtime as runtime
from xscientist.experiment_admission import (
    EXPERIMENT_ADMISSION_PROTOCOL,
    XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
)
from xscientist.research_git import (
    ResearchGitError,
    _fsync_directory,
    _load_experiment_admission_terminal_completion_ref,
    _publish_experiment_admission_terminal_completion_refs,
    capture_environment_receipt,
    create_checkpoint,
    list_research_objects_at_ref,
    show_checkpoint,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ProjectAdmissionBatchRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_trust_store = os.environ.get(
            XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE
        )
        self._boot_id_patch = mock.patch.object(
            runtime, "_boot_id", return_value="1" * 64
        )
        self._process_identity_patch = mock.patch.object(
            runtime, "_process_identity", return_value="2" * 64
        )
        self._process_identity_matches_patch = mock.patch.object(
            runtime, "_process_identity_matches", return_value=True
        )
        self._boot_id_patch.start()
        self._process_identity_patch.start()
        self._process_identity_matches_patch.start()

    def tearDown(self) -> None:
        self._process_identity_matches_patch.stop()
        self._process_identity_patch.stop()
        self._boot_id_patch.stop()
        if self._original_trust_store is None:
            os.environ.pop(XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE, None)
        else:
            os.environ[XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE] = (
                self._original_trust_store
            )

    def _batch(
        self,
        root: Path,
        *,
        size: int,
        claim_starts: bool = True,
        ideas_hash: str | None = None,
        configuration_hash: str | None = None,
        budget_hash: str | None = None,
    ) -> tuple[ResearchRepository, argparse.Namespace, dict[int, dict]]:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("cryptography is required for Ed25519 admission tests")

        repository = ResearchRepository.init(
            root,
            question="Can a required terminal batch recover exactly?",
            policy="manual",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )
        config_path = root / "research.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment_admission"] = {"mode": "required"}
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        hypothesis = repository.record(
            "hypothesis",
            {"statement": "Exact terminal batches recover after a crash."},
            state="draft",
        )
        plan = repository.record(
            "research_plan",
            {
                "summary": "Run the bounded recovery batch.",
                "hypothesis_id": hypothesis.object_id,
            },
            state="locked",
            relations=[
                {
                    "type": "depends_on",
                    "target": hypothesis.object_id,
                    "role": "hypothesis",
                }
            ],
        )
        repository.commit(stage="plan", subject="lock recovery test plan")

        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        authority_root = root.parent / "admission-authority"
        authority_root.mkdir()
        public_key_path = authority_root / "reviewer.pub"
        public_key_path.write_bytes(public_pem)
        trust_store = authority_root / "trust.json"
        trust_store.write_text(
            json.dumps(
                {
                    "keys": {
                        "key:batch-recovery-reviewer": {
                            "identity": "human:batch-recovery-reviewer",
                            "algorithm": "ed25519",
                            "public_key_file": public_key_path.name,
                            "revoked": False,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        os.environ[XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE] = str(trust_store)

        lifecycle = ResearchLifecycle(repository)
        for idea_index in range(size):
            scope = {
                "idea_index": idea_index,
                "study_phase": "exploratory",
                "environment_hash": capture_environment_receipt(repository.path)[
                    "content_hash"
                ],
            }
            for field, value in (
                ("ideas_hash", ideas_hash),
                ("configuration_hash", configuration_hash),
                ("budget_hash", budget_hash),
            ):
                if value is not None:
                    scope[field] = value
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan.object_id,
                execution_scope=scope,
                decision="authorize",
            )
            attestation = sign_attestation(
                payload,
                purpose=EXPERIMENT_ADMISSION_PROTOCOL,
                identity="human:batch-recovery-reviewer",
                key_id="key:batch-recovery-reviewer",
                algorithm="ed25519",
                key=private_pem,
                issued_at="2026-09-04T00:00:00+00:00",
            )
            lifecycle.ingest_experiment_admission(
                payload,
                attestation=attestation,
                trust_store=trust_store,
            )

        args = argparse.Namespace(
            project_dir=str(repository.path),
            _research_git_active=True,
            _research_vcs_ids={
                "plans": {index: plan.object_id for index in range(size)}
            },
            research_git_strict=True,
        )
        receipts = _preflight_local_research_experiment_admission(
            args,
            idea_indices=list(range(size)),
            ideas_hash=ideas_hash,
            configuration_hash=configuration_hash,
            budget_hash=budget_hash,
        )
        if claim_starts:
            for receipt in receipts.values():
                with _experiment_admission_execution_lease(repository.path, receipt):
                    pass
        return repository, args, receipts

    @staticmethod
    def _result(idea_index: int, *, status: str = "success") -> dict:
        return {"idea_idx": idea_index, "status": status}

    @staticmethod
    def _default_configuration_hash() -> str:
        config_path = resolve_bfts_config_path()
        return canonical_content_hash(config_path.read_text(encoding="utf-8"))

    @staticmethod
    def _persist_uncommitted(
        repository: ResearchRepository,
        args: argparse.Namespace,
        receipt: dict,
        result: dict,
    ) -> dict:
        idea_index = int(result["idea_idx"])
        payload = _compact_research_result(result)
        payload["pipeline_status"] = str(result.get("status") or "failed")
        payload["status"] = str(
            result.get("registry_status") or result.get("status") or "failed"
        )
        for field, value in receipt["execution_scope"].items():
            if field != "code_commit":
                payload.setdefault(str(field), value)
        lifecycle = ResearchLifecycle(repository)
        prepared = lifecycle.validate_experiment_attempt(
            payload,
            plan_id=args._research_vcs_ids["plans"][idea_index],
            admission_gate_id=args._research_vcs_ids["admission_gates"][idea_index],
            admission_preflight=receipt,
            provenance={},
        )
        return lifecycle.record_validated_experiment_attempt(prepared, commit=False)

    @staticmethod
    def _record_project_batch(args: argparse.Namespace, results: list[dict]) -> None:
        with mock.patch(
            "ai_scientist.apps.project._research_attempt_provenance",
            return_value={},
        ):
            _record_local_research_attempt_objects(args, results=results)

    def _commit_batch_then_interrupt_ref_publication(
        self,
        repository: ResearchRepository,
        args: argparse.Namespace,
        results: list[dict],
        *,
        published_count: int,
    ) -> str:
        terminal_commit: str | None = None

        def publish_prefix_then_crash(
            repo: str | Path,
            *,
            start_object_ids: list[str],
            commit: str,
        ) -> str:
            nonlocal terminal_commit
            terminal_commit = commit
            if published_count:
                _publish_experiment_admission_terminal_completion_refs(
                    repo,
                    start_object_ids=start_object_ids[:published_count],
                    commit=commit,
                )
            raise ResearchGitError("simulated crash during completion-ref publication")

        with mock.patch(
            "xscientist.research_git."
            "_publish_experiment_admission_terminal_completion_refs",
            side_effect=publish_prefix_then_crash,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulated crash during completion-ref publication",
            ):
                self._record_project_batch(args, results)
        self.assertIsNotNone(terminal_commit)
        self.assertEqual(repository.status()["head"], terminal_commit)
        return str(terminal_commit)

    def test_exact_crash_residue_commits_as_one_terminal_batch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(Path(td) / "research", size=2)
            results = [self._result(0), self._result(1)]
            first = self._persist_uncommitted(repository, args, receipts[0], results[0])
            second = self._persist_uncommitted(
                repository, args, receipts[1], results[1]
            )
            second["attempt"].path.unlink()
            _fsync_directory(second["attempt"].path.parent)
            self.assertTrue(first["attempt"].path.is_file())
            self.assertFalse(second["attempt"].path.exists())
            self.assertTrue(second["terminal_reservation"].is_file())

            start_head = repository.status()["head"]
            self._record_project_batch(args, results)

            terminal_head = repository.status()["head"]
            self.assertNotEqual(terminal_head, start_head)
            checkpoint = show_checkpoint(repository.path, terminal_head)["checkpoint"]
            expected_paths = {
                first["attempt"].path.relative_to(repository.path).as_posix(),
                second["attempt"].path.relative_to(repository.path).as_posix(),
            }
            self.assertEqual(checkpoint["stage"], "experiment_admission_terminal")
            self.assertEqual(set(checkpoint["changed_paths"]), expected_paths)
            self.assertEqual(len(checkpoint["changed_paths"]), 2)
            self.assertEqual(
                set(args._research_vcs_ids["attempts"].values()),
                {first["attempt"].object_id, second["attempt"].object_id},
            )

    def test_start_batch_receipts_reconstruct_without_parent_args(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _args, receipts = self._batch(Path(td) / "research", size=2)
            start_checkpoint = receipts[0]["start_checkpoint"]

            reconstructed = ResearchLifecycle(
                repository
            ).reconstruct_experiment_admission_batch(start_checkpoint)

            self.assertEqual(reconstructed, receipts)

    def test_single_terminal_checkpoint_preserves_exact_attempt_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, _receipts = self._batch(Path(td) / "research", size=1)

            self._record_project_batch(
                args,
                [
                    {
                        "idea_idx": 0,
                        "status": "failed",
                        "registry_status": "timed_out",
                    }
                ],
            )

            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment_admission_terminal")
            self.assertEqual(checkpoint["status"], "timed_out")
            attempt = repository.objects(kind="experiment_attempt")[0]
            self.assertEqual(attempt["state"], "timed_out")

    def test_v2_worker_outbox_freezes_attempt_result_and_provenance(self) -> None:
        from xscientist.experiment_admission_runtime import publish_worker_terminal

        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(
                Path(td) / "research", size=1, claim_starts=False
            )
            experiment_dir = repository.path / "02_experiments" / "idea-0"
            experiment_dir.mkdir(parents=True)
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[0],
                experiment_dir=experiment_dir,
                idea_key="idea-0",
            ) as claim:
                self.assertIsNotNone(claim)
            result = {
                "idea_idx": 0,
                "status": "failed",
                "registry_status": "completed",
                "research_status": "failed",
                "stage": "writeup",
                "resumable": False,
                "error": "must-not-enter-research-vcs",
            }
            frozen_provenance = {
                "environment_hash": receipts[0]["execution_scope"]["environment_hash"]
            }
            terminal = publish_worker_terminal(
                repository.path,
                claim,
                result,
                provenance=frozen_provenance,
            )

            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                side_effect=AssertionError("live provenance must not be recomputed"),
            ):
                _record_local_research_attempt_objects(args, results=[result])

            attempt = repository.objects(kind="experiment_attempt")[0]
            self.assertEqual(attempt["provenance"], frozen_provenance)
            self.assertIsNone(attempt["payload"].get("error"))
            self.assertNotIn("must-not-enter-research-vcs", json.dumps(attempt))
            self.assertEqual(
                attempt["payload"]["admission_runtime_claim_hash"],
                claim["claim_hash"],
            )
            self.assertEqual(
                attempt["payload"]["admission_worker_terminal_hash"],
                terminal["terminal_hash"],
            )

    def test_committed_v2_batch_rebuild_ignores_deleted_worker_artifact(self) -> None:
        from xscientist import experiment_admission_runtime as runtime

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, args, receipts = self._batch(
                base / "research", size=1, claim_starts=False
            )
            start_head = repository.status()["head"]
            linked = base / "linked"
            added = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "worktree",
                    "add",
                    "--detach",
                    str(linked),
                    start_head,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)

            experiment_dir = repository.path / "02_experiments" / "idea-0"
            experiment_dir.mkdir(parents=True)
            checkpoint_path = experiment_dir / "checkpoint.json"
            checkpoint_path.write_text('{"step":1}\n', encoding="utf-8")
            result = {
                "idea_idx": 0,
                "status": "failed",
                "registry_status": "completed",
                "research_status": "failed",
                "stage": "writeup",
                "resumable": True,
                "checkpoint_path": str(checkpoint_path),
            }
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[0],
                experiment_dir=experiment_dir,
                idea_key="idea-0",
            ) as claim:
                self.assertIsNotNone(claim)
            terminal = runtime.publish_worker_terminal(
                repository.path,
                claim,
                result,
                provenance={
                    "environment_hash": receipts[0]["execution_scope"][
                        "environment_hash"
                    ]
                },
            )
            _record_local_research_attempt_objects(args, results=[result])
            terminal_commit = repository.status()["head"]
            attempt = repository.objects(kind="experiment_attempt")[0]
            self.assertEqual(
                attempt["payload"]["admission_worker_terminal_hash"],
                terminal["terminal_hash"],
            )

            checkpoint_path.unlink()
            _fsync_directory(checkpoint_path.parent)
            linked_args = argparse.Namespace(**copy.deepcopy(vars(args)))
            linked_args.project_dir = str(linked)
            with mock.patch.object(
                runtime,
                "verify_terminal_artifacts",
                side_effect=AssertionError(
                    "committed VCS recovery must not revalidate mutable artifacts"
                ),
            ) as verify_artifacts:
                reconciled = _reconcile_required_experiment_admission_runtime(
                    linked_args,
                    selected_indices=[0],
                    ideas_hash=None,
                    configuration_hash=None,
                    budget_hash=None,
                )

            verify_artifacts.assert_not_called()
            self.assertEqual(ResearchRepository(linked).status()["head"], start_head)
            self.assertEqual(repository.status()["head"], terminal_commit)
            self.assertEqual(reconciled["execute_indices"], [])
            self.assertEqual(reconciled["recovered_results"], [])
            self.assertEqual(len(reconciled["committed_results"]), 1)
            committed = reconciled["committed_results"][0]
            self.assertEqual(committed["admission_terminal_commit"], terminal_commit)
            self.assertEqual(
                Path(committed["checkpoint_path"]).resolve(),
                (linked / "02_experiments" / "idea-0" / "checkpoint.json").resolve(),
            )
            self.assertFalse(Path(committed["checkpoint_path"]).exists())

    def test_reconciliation_reuses_unclaimed_sibling_and_rebuilds_from_vcs(
        self,
    ) -> None:
        from xscientist import experiment_admission_runtime as runtime

        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(
                Path(td) / "research", size=2, claim_starts=False
            )
            start_head = repository.status()["head"]
            result_zero = {
                "idea_idx": 0,
                "status": "failed",
                "registry_status": "completed",
                "research_status": "failed",
                "stage": "writeup",
                "resumable": False,
            }
            experiment_zero = repository.path / "02_experiments" / "idea-0"
            experiment_zero.mkdir(parents=True)
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[0],
                experiment_dir=experiment_zero,
                idea_key="idea-0",
            ) as claim_zero:
                pass
            runtime.publish_worker_terminal(
                repository.path,
                claim_zero,
                result_zero,
                provenance={
                    "environment_hash": receipts[0]["execution_scope"][
                        "environment_hash"
                    ]
                },
            )

            with mock.patch.object(
                runtime, "_process_identity_matches", return_value=False
            ):
                recovered = _reconcile_required_experiment_admission_runtime(
                    args,
                    selected_indices=[0, 1],
                    ideas_hash=None,
                    configuration_hash=None,
                    budget_hash=None,
                )

            self.assertEqual(repository.status()["head"], start_head)
            self.assertEqual(recovered["execute_indices"], [1])
            self.assertEqual(
                [item["idea_idx"] for item in recovered["recovered_results"]], [0]
            )
            reused = _preflight_local_research_experiment_admission(
                args,
                idea_indices=[1],
            )
            self.assertEqual(reused, receipts)
            self.assertEqual(repository.status()["head"], start_head)

            result_one = {
                "idea_idx": 1,
                "status": "success",
                "registry_status": "completed",
                "research_status": "manuscript_draft",
                "stage": "paper",
                "resumable": False,
            }
            experiment_one = repository.path / "02_experiments" / "idea-1"
            experiment_one.mkdir(parents=True)
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[1],
                experiment_dir=experiment_one,
                idea_key="idea-1",
            ) as claim_one:
                pass
            runtime.publish_worker_terminal(
                repository.path,
                claim_one,
                result_one,
                provenance={
                    "environment_hash": receipts[1]["execution_scope"][
                        "environment_hash"
                    ]
                },
            )
            _record_local_research_attempt_objects(
                args,
                results=[*recovered["recovered_results"], result_one],
            )

            terminal_head = repository.status()["head"]
            self.assertNotEqual(terminal_head, start_head)
            for receipt in receipts.values():
                self.assertEqual(
                    _load_experiment_admission_terminal_completion_ref(
                        repository.path,
                        start_object_id=receipt["start_object_id"],
                    ),
                    terminal_head,
                )
            progress = repository.path / "04_logs" / "progress.json"
            progress.parent.mkdir(parents=True, exist_ok=True)
            progress.write_text("{damaged", encoding="utf-8")
            rebuilt = _reconcile_required_experiment_admission_runtime(
                args,
                selected_indices=[0, 1],
                ideas_hash=None,
                configuration_hash=None,
                budget_hash=None,
            )
            self.assertEqual(rebuilt["recovered_results"], [])
            self.assertEqual(
                [item["idea_idx"] for item in rebuilt["committed_results"]],
                [0, 1],
            )
            self.assertEqual(
                [item["status"] for item in rebuilt["committed_results"]],
                ["failed", "success"],
            )

    def test_reconciliation_blocks_the_batch_while_v2_owner_is_alive(self) -> None:
        from xscientist import experiment_admission_runtime as runtime

        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(
                Path(td) / "research", size=1, claim_starts=False
            )
            experiment_dir = repository.path / "02_experiments" / "idea-0"
            experiment_dir.mkdir(parents=True)
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[0],
                experiment_dir=experiment_dir,
                idea_key="idea-0",
            ) as claim:
                pass
            runtime.publish_worker_terminal(
                repository.path,
                claim,
                {
                    "idea_idx": 0,
                    "status": "failed",
                    "registry_status": "failed",
                    "research_status": "failed",
                    "stage": "worker_aborted",
                    "resumable": False,
                },
                provenance={
                    "environment_hash": receipts[0]["execution_scope"][
                        "environment_hash"
                    ]
                },
            )

            with self.assertRaisesRegex(
                ResearchGitError, "whole batch remains blocked"
            ):
                _reconcile_required_experiment_admission_runtime(
                    args,
                    selected_indices=[0],
                    ideas_hash=None,
                    configuration_hash=None,
                    budget_hash=None,
                )

    def test_reconciliation_atomically_records_a_fully_recoverable_batch(self) -> None:
        from xscientist import experiment_admission_runtime as runtime

        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(
                Path(td) / "research", size=1, claim_starts=False
            )
            start_head = repository.status()["head"]
            experiment_dir = repository.path / "02_experiments" / "idea-0"
            experiment_dir.mkdir(parents=True)
            with _experiment_admission_execution_lease(
                repository.path,
                receipts[0],
                experiment_dir=experiment_dir,
                idea_key="idea-0",
            ) as claim:
                pass
            runtime.publish_worker_terminal(
                repository.path,
                claim,
                {
                    "idea_idx": 0,
                    "status": "success",
                    "registry_status": "completed",
                    "research_status": "manuscript_draft",
                    "stage": "paper",
                    "resumable": False,
                },
                provenance={
                    "environment_hash": receipts[0]["execution_scope"][
                        "environment_hash"
                    ]
                },
            )

            with mock.patch.object(
                runtime, "_process_identity_matches", return_value=False
            ):
                recovered = _reconcile_required_experiment_admission_runtime(
                    args,
                    selected_indices=[0],
                    ideas_hash=None,
                    configuration_hash=None,
                    budget_hash=None,
                )

            terminal_head = repository.status()["head"]
            self.assertNotEqual(terminal_head, start_head)
            self.assertEqual(recovered["execute_indices"], [])
            self.assertEqual(recovered["recovered_results"], [])
            self.assertEqual(
                [item["status"] for item in recovered["committed_results"]],
                ["success"],
            )
            self.assertEqual(
                _load_experiment_admission_terminal_completion_ref(
                    repository.path,
                    start_object_id=receipts[0]["start_object_id"],
                ),
                terminal_head,
            )

    def test_committed_exact_batch_retry_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, _receipts = self._batch(Path(td) / "research", size=2)
            results = [self._result(0), self._result(1)]
            self._record_project_batch(args, results)
            terminal_head = repository.status()["head"]
            first_ids = dict(args._research_vcs_ids["attempts"])

            self._record_project_batch(args, results)

            self.assertEqual(repository.status()["head"], terminal_head)
            self.assertEqual(args._research_vcs_ids["attempts"], first_ids)
            checkpoint = show_checkpoint(repository.path, terminal_head)["checkpoint"]
            self.assertEqual(len(checkpoint["changed_paths"]), 2)

    def test_linked_worktree_retry_reuses_one_terminal_batch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, args, receipts = self._batch(base / "research", size=2)
            results = [self._result(0), self._result(1)]
            start_head = repository.status()["head"]
            linked = base / "linked"
            added = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "worktree",
                    "add",
                    "--detach",
                    str(linked),
                    start_head,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)

            self._record_project_batch(args, results)
            terminal_head = repository.status()["head"]
            linked_args = argparse.Namespace(**copy.deepcopy(vars(args)))
            linked_args.project_dir = str(linked)

            self._record_project_batch(linked_args, results)

            self.assertEqual(ResearchRepository(linked).status()["head"], start_head)
            self.assertEqual(
                linked_args._research_vcs_ids["attempts"],
                args._research_vcs_ids["attempts"],
            )
            for idea_index, attempt_id in args._research_vcs_ids["attempts"].items():
                attempt_path = (
                    linked
                    / ".xscientist"
                    / "objects"
                    / "experiment_attempt"
                    / f"{attempt_id}.json"
                )
                self.assertFalse(attempt_path.exists())
                self.assertEqual(
                    _load_experiment_admission_terminal_completion_ref(
                        repository.path,
                        start_object_id=receipts[idea_index]["start_object_id"],
                    ),
                    terminal_head,
                )
                history = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repository.path),
                        "log",
                        "--all",
                        "--format=%H",
                        "--",
                        attempt_path.relative_to(linked).as_posix(),
                    ],
                    text=True,
                    capture_output=True,
                    check=True,
                ).stdout.splitlines()
                self.assertEqual(history, [terminal_head])

    def test_partial_completion_ref_publication_recovers_across_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, args, receipts = self._batch(base / "research", size=2)
            results = [self._result(0), self._result(1)]
            start_head = repository.status()["head"]
            linked = base / "linked"
            added = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "worktree",
                    "add",
                    "--detach",
                    str(linked),
                    start_head,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)

            published_commit: str | None = None

            def publish_one_then_crash(
                repo: str | Path,
                *,
                start_object_ids: list[str],
                commit: str,
            ) -> str:
                nonlocal published_commit
                published_commit = commit
                _publish_experiment_admission_terminal_completion_refs(
                    repo,
                    start_object_ids=[start_object_ids[0]],
                    commit=commit,
                )
                raise ResearchGitError(
                    "simulated crash after partial completion-ref publication"
                )

            with mock.patch(
                "xscientist.research_git."
                "_publish_experiment_admission_terminal_completion_refs",
                side_effect=publish_one_then_crash,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated crash after partial completion-ref publication",
                ):
                    self._record_project_batch(args, results)

            terminal_head = repository.status()["head"]
            self.assertEqual(published_commit, terminal_head)
            self.assertNotEqual(terminal_head, start_head)
            self.assertEqual(
                _load_experiment_admission_terminal_completion_ref(
                    repository.path,
                    start_object_id=receipts[0]["start_object_id"],
                ),
                terminal_head,
            )
            self.assertIsNone(
                _load_experiment_admission_terminal_completion_ref(
                    repository.path,
                    start_object_id=receipts[1]["start_object_id"],
                )
            )

            linked_args = argparse.Namespace(**copy.deepcopy(vars(args)))
            linked_args.project_dir = str(linked)
            self._record_project_batch(linked_args, results)

            self.assertEqual(repository.status()["head"], terminal_head)
            self.assertEqual(ResearchRepository(linked).status()["head"], start_head)
            self.assertEqual(len(linked_args._research_vcs_ids["attempts"]), 2)
            for idea_index, attempt_id in linked_args._research_vcs_ids[
                "attempts"
            ].items():
                self.assertEqual(
                    _load_experiment_admission_terminal_completion_ref(
                        repository.path,
                        start_object_id=receipts[idea_index]["start_object_id"],
                    ),
                    terminal_head,
                )
                self.assertFalse(
                    (
                        linked
                        / ".xscientist"
                        / "objects"
                        / "experiment_attempt"
                        / f"{attempt_id}.json"
                    ).exists()
                )

    def test_reconciliation_atomically_repairs_zero_or_partial_completion_refs(
        self,
    ) -> None:
        ideas_hash = "sha256:" + "1" * 64
        configuration_hash = self._default_configuration_hash()
        budget_hash = "sha256:" + "3" * 64
        for published_count in (0, 1):
            with self.subTest(published_count=published_count):
                with tempfile.TemporaryDirectory() as td:
                    repository, args, receipts = self._batch(
                        Path(td) / "research",
                        size=2,
                        ideas_hash=ideas_hash,
                        configuration_hash=configuration_hash,
                        budget_hash=budget_hash,
                    )
                    results = [self._result(0), self._result(1)]
                    terminal_commit = self._commit_batch_then_interrupt_ref_publication(
                        repository,
                        args,
                        results,
                        published_count=published_count,
                    )
                    start_object_ids = [
                        str(receipts[index]["start_object_id"])
                        for index in sorted(receipts)
                    ]
                    self.assertEqual(
                        [
                            _load_experiment_admission_terminal_completion_ref(
                                repository.path,
                                start_object_id=start_object_id,
                            )
                            for start_object_id in start_object_ids
                        ],
                        [terminal_commit] * published_count
                        + [None] * (len(start_object_ids) - published_count),
                    )

                    publication_batches: list[tuple[tuple[str, ...], str]] = []

                    def publish_reconciled_batch(
                        repo: str | Path,
                        *,
                        start_object_ids: list[str],
                        commit: str,
                    ) -> str:
                        publication_batches.append((tuple(start_object_ids), commit))
                        return _publish_experiment_admission_terminal_completion_refs(
                            repo,
                            start_object_ids=start_object_ids,
                            commit=commit,
                        )

                    with mock.patch(
                        "xscientist.research_git."
                        "_publish_experiment_admission_terminal_completion_refs",
                        side_effect=publish_reconciled_batch,
                    ):
                        reconciled = _reconcile_required_experiment_admission_runtime(
                            args,
                            selected_indices=[0, 1],
                            ideas_hash=ideas_hash,
                            configuration_hash=configuration_hash,
                            budget_hash=budget_hash,
                        )

                    self.assertEqual(repository.status()["head"], terminal_commit)
                    self.assertEqual(
                        publication_batches,
                        [(tuple(start_object_ids), terminal_commit)],
                    )
                    self.assertTrue(reconciled["active"])
                    self.assertFalse(reconciled["pending_batch"])
                    self.assertEqual(reconciled["recovered_results"], [])
                    self.assertEqual(
                        [item["idea_idx"] for item in reconciled["committed_results"]],
                        [0, 1],
                    )
                    self.assertTrue(
                        all(
                            item["admission_terminal_commit"] == terminal_commit
                            for item in reconciled["committed_results"]
                        )
                    )
                    self.assertEqual(
                        [
                            _load_experiment_admission_terminal_completion_ref(
                                repository.path,
                                start_object_id=start_object_id,
                            )
                            for start_object_id in start_object_ids
                        ],
                        [terminal_commit, terminal_commit],
                    )
                    self.assertEqual(reconciled["execute_indices"], [])

    def test_zero_completion_refs_recover_from_stale_linked_worktree(self) -> None:
        ideas_hash = "sha256:" + "8" * 64
        configuration_hash = self._default_configuration_hash()
        budget_hash = "sha256:" + "9" * 64
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, args, receipts = self._batch(
                base / "research",
                size=2,
                ideas_hash=ideas_hash,
                configuration_hash=configuration_hash,
                budget_hash=budget_hash,
            )
            start_head = repository.status()["head"]
            linked = base / "linked"
            added = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "worktree",
                    "add",
                    "--detach",
                    str(linked),
                    start_head,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)
            terminal_commit = self._commit_batch_then_interrupt_ref_publication(
                repository,
                args,
                [self._result(0), self._result(1)],
                published_count=0,
            )
            self.assertTrue(
                all(
                    _load_experiment_admission_terminal_completion_ref(
                        repository.path,
                        start_object_id=receipt["start_object_id"],
                    )
                    is None
                    for receipt in receipts.values()
                )
            )

            linked_args = argparse.Namespace(**copy.deepcopy(vars(args)))
            linked_args.project_dir = str(linked)
            reconciled = _reconcile_required_experiment_admission_runtime(
                linked_args,
                selected_indices=[0, 1],
                ideas_hash=ideas_hash,
                configuration_hash=configuration_hash,
                budget_hash=budget_hash,
            )

            self.assertEqual(ResearchRepository(linked).status()["head"], start_head)
            self.assertEqual(repository.status()["head"], terminal_commit)
            self.assertEqual(reconciled["execute_indices"], [])
            self.assertEqual(
                [item["idea_idx"] for item in reconciled["committed_results"]],
                [0, 1],
            )
            self.assertTrue(
                all(
                    _load_experiment_admission_terminal_completion_ref(
                        linked,
                        start_object_id=receipt["start_object_id"],
                    )
                    == terminal_commit
                    for receipt in receipts.values()
                )
            )

    def test_completed_subset_preserves_newly_selected_indices(self) -> None:
        ideas_hash = "sha256:" + "a" * 64
        configuration_hash = self._default_configuration_hash()
        budget_hash = "sha256:" + "b" * 64
        with tempfile.TemporaryDirectory() as td:
            repository, args, _receipts = self._batch(
                Path(td) / "research",
                size=1,
                ideas_hash=ideas_hash,
                configuration_hash=configuration_hash,
                budget_hash=budget_hash,
            )
            self._record_project_batch(args, [self._result(0)])

            reconciled = _reconcile_required_experiment_admission_runtime(
                args,
                selected_indices=[0, 1],
                ideas_hash=ideas_hash,
                configuration_hash=configuration_hash,
                budget_hash=budget_hash,
            )

            self.assertEqual(
                [item["idea_idx"] for item in reconciled["committed_results"]], [0]
            )
            self.assertEqual(reconciled["execute_indices"], [1])
            self.assertEqual(reconciled["batch_indices"], [0, 1])

    def test_completed_batch_reuse_binds_configuration_and_budget_hashes(
        self,
    ) -> None:
        ideas_hash = "sha256:" + "4" * 64
        configuration_hash = self._default_configuration_hash()
        budget_hash = "sha256:" + "6" * 64
        with tempfile.TemporaryDirectory() as td:
            repository, args, _receipts = self._batch(
                Path(td) / "research",
                size=1,
                ideas_hash=ideas_hash,
                configuration_hash=configuration_hash,
                budget_hash=budget_hash,
            )
            self._record_project_batch(args, [self._result(0)])
            terminal_commit = repository.status()["head"]

            for mismatched_field in ("configuration_hash", "budget_hash"):
                with self.subTest(mismatched_field=mismatched_field):
                    current_scope = {
                        "ideas_hash": ideas_hash,
                        "configuration_hash": configuration_hash,
                        "budget_hash": budget_hash,
                    }
                    current_scope[mismatched_field] = "sha256:" + "7" * 64
                    reconciled = _reconcile_required_experiment_admission_runtime(
                        args,
                        selected_indices=[0],
                        ideas_hash=current_scope["ideas_hash"],
                        configuration_hash=current_scope["configuration_hash"],
                        budget_hash=current_scope["budget_hash"],
                    )

                    self.assertEqual(repository.status()["head"], terminal_commit)
                    self.assertTrue(reconciled["active"])
                    self.assertFalse(reconciled["pending_batch"])
                    self.assertEqual(reconciled["committed_results"], [])
                    self.assertEqual(reconciled["recovered_results"], [])
                    self.assertEqual(reconciled["execute_indices"], [0])

    def test_conflicting_retry_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(Path(td) / "research", size=1)
            original = self._persist_uncommitted(
                repository, args, receipts[0], self._result(0)
            )
            head = repository.status()["head"]

            with self.assertRaisesRegex(RuntimeError, "conflicting attempt identity"):
                self._record_project_batch(args, [self._result(0, status="failed")])

            self.assertEqual(repository.status()["head"], head)
            self.assertTrue(original["attempt"].path.is_file())
            self.assertTrue(original["terminal_reservation"].is_file())

    def test_partially_committed_batch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(Path(td) / "research", size=2)
            results = [self._result(0), self._result(1)]
            first = self._persist_uncommitted(repository, args, receipts[0], results[0])
            first_path = first["attempt"].path.relative_to(repository.path).as_posix()
            partial = create_checkpoint(
                repository.path,
                stage="experiment_admission_terminal",
                subject="simulate partial terminal crash",
                status="completed",
                only_paths=[first_path],
            )
            self.assertTrue(partial.committed)
            head = repository.status()["head"]

            with self.assertRaisesRegex(RuntimeError, "partially committed outcomes"):
                self._record_project_batch(args, results)

            self.assertEqual(repository.status()["head"], head)

    def test_terminal_checkpoint_with_foreign_object_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, args, receipts = self._batch(Path(td) / "research", size=2)
            results = [self._result(0), self._result(1)]
            attempts = [
                self._persist_uncommitted(repository, args, receipts[index], result)[
                    "attempt"
                ]
                for index, result in enumerate(results)
            ]
            foreign = repository.record(
                "observation",
                {
                    "summary": "Foreign material must not join a terminal batch.",
                    "measurement": 1.0,
                },
            )
            paths = [
                attempt.path.relative_to(repository.path).as_posix()
                for attempt in attempts
            ]
            paths.append(foreign.path.relative_to(repository.path).as_posix())
            checkpoint = create_checkpoint(
                repository.path,
                stage="experiment_admission_terminal",
                subject="simulate foreign terminal material",
                status="completed",
                only_paths=paths,
            )
            self.assertTrue(checkpoint.committed)
            head = repository.status()["head"]

            with self.assertRaisesRegex(
                RuntimeError, "does not exactly match the batch"
            ):
                self._record_project_batch(args, results)

            self.assertEqual(repository.status()["head"], head)
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertTrue(
                {attempt.object_id for attempt in attempts} <= committed_ids
            )


if __name__ == "__main__":
    unittest.main()
