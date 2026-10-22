from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from ai_scientist.apps.project import (
    _experiment_admission_execution_lease,
    _experiment_checkpoint_status,
    _preflight_local_research_experiment_admission,
    _record_local_research_attempt_objects,
    _revalidate_experiment_admission_preflight,
    process_single_idea,
)
from ai_scientist.protocol.attestation import sign_attestation
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import schema_validator
from ai_scientist.utils.privacy import redact_sensitive_text
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist.experiment_admission import (
    EXPERIMENT_ADMISSION_ACTOR,
    EXPERIMENT_ADMISSION_PROTOCOL,
    XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
)
from xscientist import research_commands as research_commands_module
from xscientist.entrypoints import research_main
from xscientist.research_commands import save_experiment
from xscientist.research_git import (
    CheckpointResult,
    ResearchGitError,
    _load_experiment_admission_terminal_reservation,
    _repository_common_git_dir,
    _repository_lock,
    _reserve_experiment_admission_terminal,
    capture_environment_receipt,
    create_checkpoint,
    list_research_objects_at_ref,
    record_research_object,
    show_checkpoint,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ExperimentAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._admission_signers: dict[tuple[str, str, str], tuple[bytes, bytes]] = {}
        self._original_admission_trust_store = os.environ.get(
            XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE
        )

    def tearDown(self) -> None:
        if self._original_admission_trust_store is None:
            os.environ.pop(XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE, None)
        else:
            os.environ[XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE] = (
                self._original_admission_trust_store
            )

    def _repository(
        self,
        root: Path,
        *,
        admission_mode: str | None,
        checkpoint_policy: str = "milestone",
    ) -> tuple[ResearchRepository, dict, dict]:
        repository = ResearchRepository.init(
            root,
            question="Should this exact experiment run?",
            policy=checkpoint_policy,
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )
        if admission_mode is not None:
            config_path = root / "research.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["experiment_admission"] = {"mode": admission_mode}
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
        hypothesis_result = repository.record(
            "hypothesis",
            {"statement": "The intervention improves the outcome."},
            state="draft",
        )
        plan_result = repository.record(
            "research_plan",
            {
                "summary": "Run the bounded intervention.",
                "hypothesis_id": hypothesis_result.object_id,
            },
            state="locked",
            relations=[
                {
                    "type": "depends_on",
                    "target": hypothesis_result.object_id,
                    "role": "hypothesis",
                }
            ],
        )
        repository.commit(stage="plan", subject="lock exact test plan")
        return (
            repository,
            repository.get(hypothesis_result.object_id),
            repository.get(plan_result.object_id),
        )

    @staticmethod
    def _execution_scope(
        repository: ResearchRepository, **scope: object
    ) -> dict[str, object]:
        return {
            **scope,
            "environment_hash": capture_environment_receipt(repository.path)[
                "content_hash"
            ],
        }

    def _gate(
        self,
        repository: ResearchRepository,
        *,
        hypothesis: dict,
        plan: dict,
        decision: str = "authorize",
        scope: dict | None = None,
        extra_relations: list[dict] | None = None,
        commit: bool = True,
    ) -> dict:
        normalized_scope = dict(scope or {})
        if commit:
            lifecycle = ResearchLifecycle(repository)
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope=normalized_scope,
                decision=decision,
            )
            attestation, trust_store = self._external_admission_signature(
                repository.path.parent,
                payload,
            )
            ingested = lifecycle.ingest_experiment_admission(
                payload,
                attestation=attestation,
                trust_store=trust_store,
            )
            gate = repository.get(ingested["gate"].object_id)
            additions = [
                relation
                for relation in (extra_relations or [])
                if relation not in (gate.get("relations") or [])
            ]
            if not additions:
                return gate
            forged = repository._record(
                "gate_decision",
                gate["payload"],
                state="verified",
                relations=[*(gate.get("relations") or []), *additions],
                actor=EXPERIMENT_ADMISSION_ACTOR,
            )
            repository.commit(stage="review", subject="forge graph-only supersession")
            return repository.get(forged.object_id)

        core = {
            "protocol_kind": "xscientist.experiment-admission.v1",
            "gate_id": f"gate-{decision}",
            "decision": decision,
            "experiment_authorized": decision == "authorize",
            "claim_promotion_allowed": False,
            "input_bindings": {
                "hypothesis": {
                    "object_id": hypothesis["object_id"],
                    "content_hash": hypothesis["content_hash"],
                },
                "plan": {
                    "object_id": plan["object_id"],
                    "content_hash": plan["content_hash"],
                },
                "preregistration": None,
            },
            "scope": normalized_scope,
            "scope_hash": canonical_content_hash(normalized_scope),
        }
        # Most consumer tests use this private fixture to isolate matching and
        # receipt semantics. Public signed ingestion has dedicated tests below.
        result = repository._record(
            "gate_decision",
            {**core, "gate_hash": canonical_content_hash(core)},
            state="verified",
            relations=[
                {"type": "evaluates", "target": hypothesis["object_id"]},
                {
                    "type": "depends_on",
                    "target": plan["object_id"],
                    "role": "plan",
                },
                *(extra_relations or []),
            ],
            actor=EXPERIMENT_ADMISSION_ACTOR,
        )
        return repository.get(result.object_id)

    def _external_admission_signature(
        self,
        root: Path,
        payload: dict,
        *,
        identity: str = "human:independent-admission-reviewer",
        key_id: str = "key:independent-admission-reviewer",
        revoked: bool = False,
    ) -> tuple[dict, Path]:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("cryptography is required for Ed25519 admission tests")
        authority_root = root / "external-authority"
        authority_root.mkdir(parents=True, exist_ok=True)
        cache_key = (str(authority_root.resolve()), identity, key_id)
        signing_material = self._admission_signers.get(cache_key)
        if signing_material is None:
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
            signing_material = (private_pem, public_pem)
            self._admission_signers[cache_key] = signing_material
        private_pem, public_pem = signing_material
        public_key_path = authority_root / "admission-reviewer.pub"
        public_key_path.write_bytes(public_pem)
        trust_store = authority_root / "trust.json"
        trust_store.write_text(
            json.dumps(
                {
                    "keys": {
                        key_id: {
                            "identity": identity,
                            "algorithm": "ed25519",
                            "public_key_file": public_key_path.name,
                            "revoked": revoked,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        os.environ[XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE] = str(trust_store)
        return (
            sign_attestation(
                payload,
                purpose=EXPERIMENT_ADMISSION_PROTOCOL,
                identity=identity,
                key_id=key_id,
                algorithm="ed25519",
                key=private_pem,
                issued_at="2026-09-04T00:00:00+00:00",
            ),
            trust_store,
        )

    def _finalized_start(
        self,
        lifecycle: ResearchLifecycle,
        *,
        plan_id: str,
        execution_scope: dict,
        preregistration_id: str | None = None,
        code_commit: str | None = None,
        claim: bool = False,
    ) -> dict:
        issued = lifecycle.issue_experiment_admission_start(
            plan_id=plan_id,
            preregistration_id=preregistration_id,
            execution_scope=execution_scope,
            code_commit=code_commit,
        )
        relative_start = (
            issued["start"]
            .path.resolve()
            .relative_to(lifecycle.repository.path)
            .as_posix()
        )
        checkpoint = create_checkpoint(
            lifecycle.repository.path,
            stage="experiment_admission_start",
            subject="commit experiment admission start",
            status="locked",
            only_paths=[relative_start],
        )
        self.assertTrue(checkpoint.committed)
        self.assertIsNotNone(checkpoint.commit)
        finalized = lifecycle.finalize_experiment_admission_start(
            issued["receipt"],
            start_checkpoint=checkpoint.commit,
        )
        if claim:
            with _experiment_admission_execution_lease(
                lifecycle.repository.path,
                finalized,
            ):
                pass
        return finalized

    @staticmethod
    def _pipeline_worker_args(
        root: Path,
        *,
        high_quality_mode: bool = False,
    ) -> tuple:
        return (
            str(root),
            str(root),
            0,
            {
                "Name": "Terminal status separation",
                "Short Hypothesis": "Paper failures do not rewrite experiment truth.",
                "Experiments": ["Run one bounded experiment."],
            },
            None,
            "model-writeup",
            "model-citation",
            "model-review",
            "model-plots",
            "model-small",
            1,
            1,
            "normal",
            0,
            0,
            1,
            0,
            0.0,
            "depth",
            high_quality_mode,
            "publishable",
            "model-quality",
            "iclr",
            8.0,
            8.0,
            0,
            0,
            high_quality_mode,
            None,
            None,
            "default",
            0,
            False,
            0,
            "classic_pipeline",
            "open_ended",
            "adaptive",
            False,
            False,
        )

    def test_legacy_repository_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode=None
            )

            result = ResearchLifecycle(repository).experiment_attempt(
                {"status": "completed", "summary": "legacy run"},
                plan_id=plan["object_id"],
                commit=False,
            )

            saved = repository.get(result["attempt"].object_id)
            self.assertNotIn("admission_gate_id", saved["payload"])

    def test_legacy_direct_attempt_keeps_legacy_checkpoint_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode=None
            )

            result = ResearchLifecycle(repository).experiment_attempt(
                {"status": "completed", "summary": "legacy committed run"},
                plan_id=plan["object_id"],
                commit=True,
            )

            self.assertTrue(result["checkpoint"].committed)
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment")

    def test_legacy_save_experiment_keeps_legacy_checkpoint_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode=None
            )

            result = save_experiment(
                str(repository.path),
                summary="legacy CLI-recorded run",
                status="completed",
                plan_id=plan["object_id"],
                commit=True,
            )

            self.assertTrue(result["checkpoint"].committed)
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment")

    def test_required_mode_rejects_missing_gate_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )

            with self.assertRaisesRegex(ResearchGitError, "admission is required"):
                save_experiment(
                    str(repository.path),
                    summary="unauthorized run",
                    status="completed",
                    plan_id=plan["object_id"],
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_required_mode_closes_raw_sdk_and_cli_record_bypasses(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )

            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment_attempt recording is disabled"
            ):
                repository.record(
                    "experiment_attempt",
                    {"status": "completed", "summary": "raw SDK bypass"},
                    state="completed",
                )

            output = io.StringIO()
            with redirect_stdout(output):
                returncode = research_main(
                    [
                        "record",
                        "experiment_attempt",
                        "--repo",
                        str(repository.path),
                        "--state",
                        "completed",
                        "--data",
                        json.dumps(
                            {"status": "completed", "summary": "raw CLI bypass"}
                        ),
                        "--json",
                    ]
                )
            self.assertEqual(returncode, 2)
            self.assertIn(
                "raw experiment_attempt recording is disabled",
                json.loads(output.getvalue())["error"]["message"],
            )
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_low_level_private_record_cannot_bypass_required_admission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )

            with self.assertRaisesRegex(
                ResearchGitError, "repository-local validated attempt issuer"
            ):
                repository._record(
                    "experiment_attempt",
                    {"status": "failed", "summary": "private bypass"},
                    state="failed",
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_public_sdk_cannot_self_sign_an_admission_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            core = {
                "protocol_kind": "xscientist.experiment-admission.v1",
                "gate_id": "self-signed",
                "decision": "authorize",
                "experiment_authorized": True,
                "claim_promotion_allowed": False,
                "input_bindings": {
                    "hypothesis": {
                        "object_id": hypothesis["object_id"],
                        "content_hash": hypothesis["content_hash"],
                    },
                    "plan": {
                        "object_id": plan["object_id"],
                        "content_hash": plan["content_hash"],
                    },
                },
                "scope": {},
                "scope_hash": canonical_content_hash({}),
            }

            with self.assertRaisesRegex(
                ResearchGitError, "admission authority.*trusted verifier"
            ):
                repository.record(
                    "gate_decision",
                    {**core, "gate_hash": canonical_content_hash(core)},
                    state="verified",
                    actor={
                        "actor_id": "caller-controlled",
                        "authority": "deterministic_gate",
                    },
                )
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment-admission start recording is disabled"
            ):
                repository.record(
                    "context_snapshot",
                    {"protocol_kind": "xscientist.experiment-admission-start.v1"},
                    state="locked",
                )

    def test_low_level_recorder_cannot_bypass_required_admission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            core = {
                "protocol_kind": "xscientist.experiment-admission.v1",
                "gate_id": "low-level-self-signed",
                "decision": "authorize",
                "experiment_authorized": True,
                "claim_promotion_allowed": False,
                "input_bindings": {
                    "hypothesis": {
                        "object_id": hypothesis["object_id"],
                        "content_hash": hypothesis["content_hash"],
                    },
                    "plan": {
                        "object_id": plan["object_id"],
                        "content_hash": plan["content_hash"],
                    },
                },
                "scope": {},
                "scope_hash": canonical_content_hash({}),
            }

            with self.assertRaisesRegex(
                ResearchGitError, "admission authority.*trusted verifier"
            ):
                record_research_object(
                    repository.path,
                    kind="gate_decision",
                    payload={**core, "gate_hash": canonical_content_hash(core)},
                    state="verified",
                )
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment_attempt recording is disabled"
            ):
                record_research_object(
                    repository.path,
                    kind="experiment_attempt",
                    payload={"status": "completed"},
                    state="completed",
                )
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment-admission start recording is disabled"
            ):
                record_research_object(
                    repository.path,
                    kind="context_snapshot",
                    payload={
                        "protocol_kind": "xscientist.experiment-admission-start.v1"
                    },
                    state="locked",
                )

            self.assertEqual(repository.objects(kind="gate_decision"), [])
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertEqual(repository.objects(kind="context_snapshot"), [])

    def test_verified_exact_gate_authorizes_and_is_bound_to_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = self._execution_scope(
                repository,
                task_id="t1",
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            preflight = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
                claim=True,
            )

            result = save_experiment(
                str(repository.path),
                summary="authorized run",
                status="completed",
                task_id="t1",
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=preflight,
                commit=False,
            )

            saved = repository.get(result["object"].object_id)
            self.assertIs(saved["payload"]["experiment_authorized"], True)
            self.assertEqual(saved["payload"]["admission_gate_id"], gate["object_id"])
            self.assertEqual(
                saved["payload"]["admission_gate_hash"], gate["content_hash"]
            )
            self.assertEqual(
                saved["payload"]["admission_preflight_hash"],
                preflight["receipt_hash"],
            )
            self.assertIn(
                {
                    "type": "depends_on",
                    "target": gate["object_id"],
                    "role": "experiment_admission",
                },
                saved["relations"],
            )
            self.assertIn(
                {
                    "type": "consumes",
                    "target": preflight["start_object_id"],
                    "role": "experiment_admission_start",
                },
                saved["relations"],
            )

            with self.assertRaisesRegex(
                ResearchGitError, "conflicting attempt identity"
            ):
                ResearchLifecycle(repository).experiment_attempt(
                    {
                        "status": "completed",
                        "summary": "replayed authorized run",
                        "task_id": "t1",
                        "study_phase": "exploratory",
                    },
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=preflight,
                    provenance={"environment_hash": scope["environment_hash"]},
                    commit=False,
                )

    def test_side_effect_free_preflight_is_not_a_consumable_start_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository, hypothesis=hypothesis, plan=plan, scope={"task_id": "t1"}
            )
            lifecycle = ResearchLifecycle(repository)
            preflight = lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
            )

            with self.assertRaisesRegex(ResearchGitError, "receipt shape is invalid"):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=preflight,
                    commit=False,
                )

    def test_issued_start_must_be_committed_and_finalized_before_use(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            issued = lifecycle.issue_experiment_admission_start(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
            )

            with self.assertRaisesRegex(
                ResearchGitError, "finalized experiment admission receipt shape"
            ):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=issued["receipt"],
                    commit=False,
                )
            with self.assertRaisesRegex(
                ResearchGitError, "start checkpoint is not a dedicated transition"
            ):
                lifecycle.finalize_experiment_admission_start(
                    issued["receipt"],
                    start_checkpoint=repository.status()["head"],
                )
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertNotIn(issued["start"].object_id, committed_ids)

    def test_uncommitted_gate_cannot_authorize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                commit=False,
            )

            with self.assertRaisesRegex(
                ResearchGitError, "gate must be committed at HEAD"
            ):
                ResearchLifecycle(repository).experiment_attempt(
                    {"status": "completed"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_allowed_seeds_scope_requires_nonempty_declared_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"allowed_seeds": [7, 11]},
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "execution scope"):
                lifecycle.experiment_attempt(
                    {"status": "completed"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    commit=False,
                )

            preflight = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"seeds": [7]},
                claim=True,
            )
            authorized = lifecycle.experiment_attempt(
                {"status": "completed", "seeds": [7]},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=preflight,
                commit=False,
            )
            self.assertTrue(
                repository.get(authorized["attempt"].object_id)["payload"][
                    "experiment_authorized"
                ]
            )

    def test_sparse_scope_rejects_unsigned_command_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "execution scope"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    execution_scope={
                        "task_id": "t1",
                        "command_hash": "sha256:" + "1" * 64,
                    },
                )
            with self.assertRaisesRegex(ResearchGitError, "execution scope"):
                lifecycle.validate_experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    provenance={"environment_hash": "sha256:" + "2" * 64},
                )

    def test_seed_scopes_reject_bool_float_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            invalid_seeds = ([True], [1.0], [7, 7])

            for field in ("allowed_seeds", "seeds"):
                for seeds in invalid_seeds:
                    with self.subTest(boundary="signed", field=field, seeds=seeds):
                        with self.assertRaisesRegex(
                            ResearchGitError, "non-empty unique integer array"
                        ):
                            lifecycle.prepare_experiment_admission_payload(
                                plan_id=plan["object_id"],
                                execution_scope={field: seeds},
                                decision="authorize",
                            )

            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"allowed_seeds": [7, 11]},
            )
            for seeds in invalid_seeds:
                with self.subTest(boundary="runtime", seeds=seeds):
                    with self.assertRaisesRegex(ResearchGitError, "execution scope"):
                        lifecycle.preflight_experiment_admission(
                            plan_id=plan["object_id"],
                            admission_gate_id=gate["object_id"],
                            execution_scope={"seeds": seeds},
                        )

    def test_explicit_older_authorization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"task_id": "t1"}
            older = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            current = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "older than the latest"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    admission_gate_id=older["object_id"],
                    execution_scope=scope,
                )
            authorized = lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                admission_gate_id=current["object_id"],
                execution_scope=scope,
            )
            self.assertEqual(authorized["admission_gate_id"], current["object_id"])

    def test_overlapping_narrow_denial_blocks_explicit_broad_authorization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            broad = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"allowed_seeds": [7, 11]},
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope={"allowed_seeds": [7]},
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    admission_gate_id=broad["object_id"],
                    execution_scope={"seeds": [7]},
                )
            authorized = lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                admission_gate_id=broad["object_id"],
                execution_scope={"seeds": [11]},
            )
            self.assertEqual(authorized["admission_gate_id"], broad["object_id"])

    def test_cross_scope_superseder_does_not_neutralize_exact_denial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            denial = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope={"task_id": "t1"},
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t2"},
                extra_relations=[{"type": "supersedes", "target": denial["object_id"]}],
            )

            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                ResearchLifecycle(repository).experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=authorization["object_id"],
                    commit=False,
                )

    def test_record_revalidates_gate_under_lock_after_preparation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            preflight = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
            )
            prepared = lifecycle.validate_experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=authorization["object_id"],
                admission_preflight=preflight,
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope={"task_id": "t1"},
            )

            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                lifecycle.record_validated_experiment_attempt(
                    prepared,
                    commit=False,
                )
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_side_effect_free_preflight_feeds_project_runner_gate_binding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = self._execution_scope(
                repository,
                idea_index=0,
                ideas_hash="sha256:" + "1" * 64,
                configuration_hash="sha256:" + "2" * 64,
                budget_hash="sha256:" + "3" * 64,
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            head_before = repository.status()["head"]
            objects_before = repository.objects()
            lifecycle = ResearchLifecycle(repository)

            receipt = lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            self.assertEqual(receipt["status"], "authorized")
            self.assertEqual(receipt["admission_gate_id"], gate["object_id"])
            self.assertEqual(repository.status()["head"], head_before)
            self.assertEqual(repository.objects(), objects_before)

            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {0: plan["object_id"]},
                },
            )
            project_receipts = _preflight_local_research_experiment_admission(
                args,
                idea_indices=[0],
                ideas_hash=scope["ideas_hash"],
                configuration_hash=scope["configuration_hash"],
                budget_hash=scope["budget_hash"],
            )
            self.assertEqual(
                project_receipts[0]["admission_gate_id"], gate["object_id"]
            )
            self.assertEqual(
                args._research_vcs_ids["admission_gates"][0], gate["object_id"]
            )
            self.assertEqual(
                args._research_vcs_ids["admission_starts"][0],
                project_receipts[0]["start_object_id"],
            )
            self.assertNotEqual(repository.status()["head"], head_before)
            self.assertEqual(
                project_receipts[0]["admission_snapshot_head"], head_before
            )
            self.assertEqual(
                project_receipts[0]["start_checkpoint"],
                repository.status()["head"],
            )
            start_checkpoint = show_checkpoint(repository.path)
            self.assertEqual(
                start_checkpoint["checkpoint"]["stage"],
                "experiment_admission_start",
            )
            self.assertEqual(
                start_checkpoint["checkpoint"]["changed_paths"],
                [
                    ".xscientist/objects/context_snapshot/"
                    f"{project_receipts[0]['start_object_id']}.json"
                ],
            )
            objects_after = repository.objects()
            self.assertEqual(len(objects_after), len(objects_before) + 1)
            start = repository.get(project_receipts[0]["start_object_id"])
            self.assertEqual(start["kind"], "context_snapshot")
            self.assertEqual(
                start["payload"]["protocol_kind"],
                "xscientist.experiment-admission-start.v1",
            )

    def test_preflight_rejects_unknown_execution_scope_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"idea_index": 0},
            )

            with self.assertRaisesRegex(
                ResearchGitError, "unsupported fields: untrusted_dimension"
            ):
                ResearchLifecycle(repository).preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope={
                        "idea_index": 0,
                        "untrusted_dimension": "silently ignored before this guard",
                    },
                )

    def test_multi_idea_preflight_failure_leaves_no_orphaned_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=0,
                    study_phase="exploratory",
                ),
            )
            head_before = repository.status()["head"]
            objects_before = repository.objects()
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {0: plan["object_id"], 1: plan["object_id"]}
                },
            )

            with self.assertRaises(ResearchGitError):
                _preflight_local_research_experiment_admission(
                    args,
                    idea_indices=[0, 1],
                )

            self.assertEqual(repository.status()["head"], head_before)
            self.assertEqual(repository.objects(), objects_before)
            status = subprocess.run(
                ["git", "-C", str(repository.path), "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(status.stdout, "")

    def test_project_runner_commits_multi_idea_starts_in_one_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis0, plan0 = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            hypothesis1_result = repository.record(
                "hypothesis",
                {"statement": "The second intervention improves the outcome."},
            )
            plan1_result = repository.record(
                "research_plan",
                {
                    "summary": "Run the second bounded intervention.",
                    "hypothesis_id": hypothesis1_result.object_id,
                },
                state="locked",
                relations=[
                    {
                        "type": "depends_on",
                        "target": hypothesis1_result.object_id,
                        "role": "hypothesis",
                    }
                ],
            )
            repository.commit(stage="plan", subject="lock second exact test plan")
            hypothesis1 = repository.get(hypothesis1_result.object_id)
            plan1 = repository.get(plan1_result.object_id)
            scope0 = self._execution_scope(
                repository,
                idea_index=0,
                study_phase="exploratory",
            )
            scope1 = {**scope0, "idea_index": 1}
            self._gate(
                repository,
                hypothesis=hypothesis0,
                plan=plan0,
                scope=scope0,
            )
            self._gate(
                repository,
                hypothesis=hypothesis1,
                plan=plan1,
                scope=scope1,
            )
            admission_snapshot = repository.status()["head"]
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {
                        0: plan0["object_id"],
                        1: plan1["object_id"],
                    }
                },
            )

            receipts = _preflight_local_research_experiment_admission(
                args,
                idea_indices=[0, 1],
            )

            self.assertEqual(set(receipts), {0, 1})
            self.assertNotEqual(
                receipts[0]["start_object_id"], receipts[1]["start_object_id"]
            )
            self.assertEqual(
                {receipt["admission_snapshot_head"] for receipt in receipts.values()},
                {admission_snapshot},
            )
            self.assertEqual(
                {receipt["start_checkpoint"] for receipt in receipts.values()},
                {repository.status()["head"]},
            )
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["parent_commit"], admission_snapshot)
            self.assertEqual(checkpoint["stage"], "experiment_admission_start")
            self.assertEqual(
                set(checkpoint["changed_paths"]),
                {
                    ".xscientist/objects/context_snapshot/"
                    f"{receipt['start_object_id']}.json"
                    for receipt in receipts.values()
                },
            )

    def test_worker_recheck_rejects_head_change_after_parent_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"idea_index": 0, "study_phase": "exploratory"}
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope=scope,
            )

            with self.assertRaisesRegex(
                ResearchGitError, "HEAD changed after the experiment admission start"
            ):
                _revalidate_experiment_admission_preflight(
                    repository.path,
                    receipt,
                )

    def test_worker_recheck_accepts_unchanged_authorized_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"idea_index": 0, "study_phase": "exploratory"}
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )

            self.assertEqual(
                set(receipt["execution_scope"]),
                {"idea_index", "study_phase"},
            )
            revalidated = _revalidate_experiment_admission_preflight(
                repository.path,
                receipt,
            )
            self.assertEqual(revalidated["admission_gate_id"], gate["object_id"])
            self.assertEqual(revalidated["execution_scope"], receipt["execution_scope"])

    def test_execution_lease_blocks_supported_history_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"idea_index": 0, "study_phase": "exploratory"}
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )

            with _experiment_admission_execution_lease(repository.path, receipt):
                with self.assertRaisesRegex(
                    ResearchGitError, "active experiment lease"
                ):
                    repository.record("observation", {"summary": "concurrent drift"})

    def test_execution_lease_claim_is_durable_and_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"idea_index": 0, "study_phase": "exploratory"}
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            receipt = self._finalized_start(
                ResearchLifecycle(repository),
                plan_id=plan["object_id"],
                execution_scope=scope,
            )

            with _experiment_admission_execution_lease(repository.path, receipt):
                pass
            with self.assertRaisesRegex(
                ResearchGitError, "already been claimed for execution"
            ):
                with _experiment_admission_execution_lease(repository.path, receipt):
                    pass

    def test_project_terminal_attempt_accepts_supported_first_parent_writer_after_bfts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            scope = self._execution_scope(
                repository,
                idea_index=0,
                study_phase="exploratory",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args,
                idea_indices=[0],
            )
            receipt = receipts[0]
            start_checkpoint = str(receipt["start_checkpoint"])
            writer_started = threading.Event()
            writer_finished = threading.Event()
            writer_errors: list[BaseException] = []

            def write_supported_checkpoint() -> None:
                writer_started.set()
                try:
                    repository.record(
                        "observation",
                        {
                            "measurement": 1.0,
                            "summary": "supported post-BFTS analysis",
                        },
                    )
                    repository.commit(
                        stage="analysis",
                        subject="record supported post-BFTS analysis",
                    )
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    writer_finished.set()

            writer = threading.Thread(target=write_supported_checkpoint)
            with _experiment_admission_execution_lease(repository.path, receipt):
                writer.start()
                self.assertTrue(writer_started.wait(timeout=1.0))
                self.assertFalse(writer_finished.wait(timeout=0.2))

            self.assertTrue(writer_finished.wait(timeout=5.0))
            writer.join(timeout=1.0)
            self.assertFalse(writer.is_alive())
            self.assertEqual(writer_errors, [])
            intervening_checkpoint = repository.status()["head"]
            self.assertNotEqual(intervening_checkpoint, start_checkpoint)

            result = {
                "idea_idx": 0,
                "status": "failed",
                "registry_status": "completed",
                "stage": "writeup",
            }
            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                _record_local_research_attempt_objects(args, results=[result])

            attempts = repository.objects(kind="experiment_attempt")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["state"], "completed")
            self.assertEqual(attempts[0]["payload"]["status"], "completed")
            self.assertEqual(attempts[0]["payload"]["pipeline_status"], "failed")
            self.assertEqual(
                [
                    relation
                    for relation in attempts[0]["relations"]
                    if relation.get("type") == "consumes"
                    and relation.get("role") == "experiment_admission_start"
                ],
                [
                    {
                        "type": "consumes",
                        "target": receipt["start_object_id"],
                        "role": "experiment_admission_start",
                    }
                ],
            )
            terminal_checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(
                terminal_checkpoint["stage"], "experiment_admission_terminal"
            )
            self.assertEqual(terminal_checkpoint["status"], "completed")
            self.assertEqual(
                terminal_checkpoint["parent_commit"], intervening_checkpoint
            )

    def test_terminal_attempt_rejects_start_reachable_only_through_second_parent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            start_checkpoint = str(receipt["start_checkpoint"])
            admission_snapshot = str(receipt["admission_snapshot_head"])
            for command in (
                ["checkout", "--detach", admission_snapshot],
                ["commit", "--allow-empty", "-m", "divergent first parent"],
                ["merge", "--no-ff", start_checkpoint, "-m", "second-parent start"],
            ):
                completed = subprocess.run(
                    ["git", "-C", str(repository.path), *command],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            ancestry = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "merge-base",
                    "--is-ancestor",
                    start_checkpoint,
                    "HEAD",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(ancestry.returncode, 0, ancestry.stderr)

            with self.assertRaisesRegex(
                ResearchGitError,
                "HEAD changed after the experiment admission start checkpoint",
            ):
                lifecycle.validate_experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=receipt["admission_gate_id"],
                    admission_preflight=receipt,
                )

    def test_terminal_attempt_preserves_claimed_outcome_after_descendant_denial(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
            )
            scope = {"task_id": "t1"}
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            with _experiment_admission_execution_lease(repository.path, receipt):
                pass
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope=scope,
                extra_relations=[
                    {"type": "supersedes", "target": authorization["object_id"]}
                ],
            )

            recorded = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=authorization["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            saved = repository.get(recorded["attempt"].object_id)
            self.assertTrue(saved["payload"]["admitted_at_start"])
            self.assertEqual(
                saved["payload"]["current_policy_status"],
                "denied_after_start",
            )
            self.assertTrue(saved["payload"]["quarantined"])
            self.assertFalse(saved["payload"]["claim_promotion_allowed"])
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 1)

    def test_terminal_attempt_preserves_claimed_outcome_after_signer_revocation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
            )
            scope = {"task_id": "t1"}
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            with _experiment_admission_execution_lease(repository.path, receipt):
                pass

            trust_store = Path(os.environ[XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE])
            trust_payload = json.loads(trust_store.read_text(encoding="utf-8"))
            for key in trust_payload["keys"].values():
                key["revoked"] = True
            trust_store.write_text(json.dumps(trust_payload), encoding="utf-8")

            recorded = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=authorization["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            saved = repository.get(recorded["attempt"].object_id)
            self.assertEqual(
                saved["payload"]["current_policy_status"],
                "signer_revoked_after_start",
            )
            self.assertTrue(saved["payload"]["quarantined"])

    def test_required_terminal_attempt_requires_durable_execution_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
            )

            with self.assertRaisesRegex(
                ResearchGitError,
                "no durable execution claim",
            ):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=True,
                )
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_execution_claim_and_lease_are_shared_across_linked_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, hypothesis, plan = self._repository(
                base / "research", admission_mode="required"
            )
            scope = {"idea_index": 0, "study_phase": "exploratory"}
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            receipt = self._finalized_start(
                ResearchLifecycle(repository),
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
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
                    repository.status()["head"],
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)

            with _experiment_admission_execution_lease(repository.path, receipt):
                linked_repository = ResearchRepository(linked)
                with self.assertRaisesRegex(
                    ResearchGitError, "active experiment lease"
                ):
                    linked_repository.record(
                        "observation", {"summary": "cross-worktree drift"}
                    )
            with self.assertRaisesRegex(
                ResearchGitError, "already been claimed for execution"
            ):
                with _experiment_admission_execution_lease(linked, receipt):
                    pass

    def test_terminal_reservation_is_shared_across_linked_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repository, hypothesis, plan = self._repository(
                base / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
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
                    repository.status()["head"],
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(added.returncode, 0, added.stderr)

            recorded = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            reservation_path = recorded["terminal_reservation"]
            self.assertEqual(
                reservation_path.parent,
                _repository_common_git_dir(repository.path)
                / "xscientist"
                / "admission-terminal-reservations",
            )
            reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
            self.assertEqual(reservation["receipt_hash"], receipt["receipt_hash"])
            self.assertEqual(
                reservation["attempt_object_id"], recorded["attempt"].object_id
            )
            self.assertEqual(
                reservation["attempt_object_hash"], recorded["attempt"].object_hash
            )
            linked_lifecycle = ResearchLifecycle(linked)
            with self.assertRaisesRegex(
                ResearchGitError, "already been reserved for terminal recording"
            ):
                linked_lifecycle.validate_finalized_experiment_admission_start(receipt)

            linked_head = linked_lifecycle.repository.status()["head"]
            exact_retry = linked_lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            self.assertEqual(linked_lifecycle.repository.status()["head"], linked_head)
            self.assertEqual(
                exact_retry["checkpoint"].commit, recorded["checkpoint"].commit
            )
            self.assertFalse(exact_retry["attempt"].created)
            self.assertFalse(exact_retry["attempt"].path.exists())
            terminal_history = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository.path),
                    "log",
                    "--all",
                    "--format=%H",
                    "--",
                    recorded["attempt"].path.relative_to(repository.path).as_posix(),
                ],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.splitlines()
            self.assertEqual(terminal_history, [recorded["checkpoint"].commit])
            with self.assertRaisesRegex(
                ResearchGitError, "already been reserved for terminal recording"
            ):
                linked_lifecycle.experiment_attempt(
                    {
                        "status": "completed",
                        "summary": "stale linked-worktree replay",
                        "task_id": "t1",
                    },
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=True,
                )
            with self.assertRaisesRegex(
                ResearchGitError, "already been reserved for terminal recording"
            ):
                with _experiment_admission_execution_lease(linked, receipt):
                    self.fail("a terminally reserved start acquired an execution lease")
            self.assertEqual(
                [
                    item
                    for item in list_research_objects_at_ref(linked, "HEAD")
                    if item.get("kind") == "experiment_attempt"
                ],
                [],
            )

    def test_terminal_attempt_is_committed_after_its_start_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            start_checkpoint = repository.status()["head"]

            result = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            terminal_checkpoint = repository.status()["head"]
            self.assertNotEqual(terminal_checkpoint, start_checkpoint)
            shown = show_checkpoint(repository.path, terminal_checkpoint)["checkpoint"]
            self.assertEqual(shown["parent_commit"], start_checkpoint)
            self.assertEqual(shown["stage"], "experiment_admission_terminal")
            start_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(
                    repository.path, start_checkpoint
                )
            }
            self.assertNotIn(result["attempt"].object_id, start_ids)
            attempt = repository.get(result["attempt"].object_id)
            self.assertEqual(
                [
                    relation
                    for relation in attempt["relations"]
                    if relation.get("type") == "consumes"
                    and relation.get("role") == "experiment_admission_start"
                ],
                [
                    {
                        "type": "consumes",
                        "target": receipt["start_object_id"],
                        "role": "experiment_admission_start",
                    }
                ],
            )

    def test_required_save_experiment_uses_terminal_checkpoint_stage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = self._execution_scope(
                repository,
                task_id="t1",
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            receipt = self._finalized_start(
                ResearchLifecycle(repository),
                plan_id=plan["object_id"],
                execution_scope=scope,
                claim=True,
            )

            result = save_experiment(
                str(repository.path),
                summary="required CLI-recorded run",
                status="completed",
                task_id="t1",
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            self.assertTrue(result["checkpoint"].committed)
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment_admission_terminal")

    def test_required_save_checkpoint_refusal_releases_terminal_reservation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = self._execution_scope(
                repository,
                task_id="t1",
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            receipt = self._finalized_start(
                ResearchLifecycle(repository),
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            with _experiment_admission_execution_lease(repository.path, receipt):
                pass
            arguments = {
                "repo": str(repository.path),
                "summary": "required CLI-recorded run",
                "status": "completed",
                "task_id": "t1",
                "plan_id": plan["object_id"],
                "admission_gate_id": gate["object_id"],
                "admission_preflight": receipt,
                "commit": True,
            }

            with mock.patch.object(
                research_commands_module,
                "create_checkpoint",
                return_value=CheckpointResult(
                    created=False,
                    committed=False,
                    reason="simulated checkpoint refusal",
                ),
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "terminal checkpoint was not committed"
                ):
                    save_experiment(**arguments)

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            retried = save_experiment(**arguments)
            self.assertTrue(retried["checkpoint"].committed)

    def test_required_save_post_commit_exception_preserves_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = self._execution_scope(
                repository,
                task_id="t1",
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            receipt = self._finalized_start(
                ResearchLifecycle(repository),
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            with _experiment_admission_execution_lease(repository.path, receipt):
                pass
            original_create_checkpoint = create_checkpoint

            def commit_then_raise(*checkpoint_args, **checkpoint_kwargs):
                checkpoint = original_create_checkpoint(
                    *checkpoint_args, **checkpoint_kwargs
                )
                self.assertTrue(checkpoint.committed)
                raise RuntimeError("simulated exception after CLI Git commit")

            with mock.patch.object(
                research_commands_module,
                "create_checkpoint",
                side_effect=commit_then_raise,
            ):
                with self.assertRaisesRegex(RuntimeError, "after CLI Git commit"):
                    save_experiment(
                        str(repository.path),
                        summary="required CLI-recorded run",
                        status="completed",
                        task_id="t1",
                        plan_id=plan["object_id"],
                        admission_gate_id=gate["object_id"],
                        admission_preflight=receipt,
                        commit=True,
                    )

            attempts = repository.objects(kind="experiment_attempt")
            self.assertEqual(len(attempts), 1)
            reservation_path = (
                _repository_common_git_dir(repository.path)
                / "xscientist"
                / "admission-terminal-reservations"
                / f"{receipt['start_object_id']}.json"
            )
            self.assertTrue(reservation_path.is_file())
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertIn(attempts[0]["object_id"], committed_ids)

    def test_direct_attempt_checkpoint_excludes_unrelated_worktree_objects(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            unrelated = repository.record(
                "hypothesis",
                {"statement": "This draft must remain outside the attempt checkpoint."},
            )

            result = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            self.assertTrue(result["checkpoint"].committed)
            shown = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(
                shown["changed_paths"],
                [
                    ".xscientist/objects/experiment_attempt/"
                    f"{result['attempt'].object_id}.json"
                ],
            )
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertNotIn(unrelated.object_id, committed_ids)
            self.assertTrue(unrelated.path.is_file())

    def test_direct_attempt_checkpoint_failure_removes_created_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            start_checkpoint = repository.status()["head"]
            question_path = repository.path / "question.md"
            question_path.write_text(
                question_path.read_text(encoding="utf-8") + "\nstaged conflict\n",
                encoding="utf-8",
            )
            staged = subprocess.run(
                ["git", "-C", str(repository.path), "add", "question.md"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(staged.returncode, 0, staged.stderr)

            with self.assertRaisesRegex(
                ResearchGitError, "index already contains staged changes"
            ):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=True,
                )

            self.assertEqual(repository.status()["head"], start_checkpoint)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            status = subprocess.run(
                ["git", "-C", str(repository.path), "status", "--short"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(status.stdout, "M  question.md\n")

    def test_direct_attempt_checkpoint_refusal_releases_terminal_reservation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )

            with mock.patch(
                "xscientist.research_lifecycle.create_checkpoint",
                return_value=CheckpointResult(
                    created=False,
                    committed=False,
                    reason="simulated checkpoint refusal",
                ),
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "checkpoint was not committed"
                ):
                    lifecycle.experiment_attempt(
                        {"status": "completed", "task_id": "t1"},
                        plan_id=plan["object_id"],
                        admission_gate_id=gate["object_id"],
                        admission_preflight=receipt,
                        commit=True,
                    )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            retried = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            self.assertTrue(retried["checkpoint"].committed)

    def test_terminal_retry_recovers_reservation_without_attempt_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )

            def reserve_then_crash(*args, **kwargs):
                _reserve_experiment_admission_terminal(*args, **kwargs)
                raise RuntimeError("simulated crash after terminal reservation")

            with mock.patch(
                "xscientist.research_lifecycle._reserve_experiment_admission_terminal",
                side_effect=reserve_then_crash,
            ):
                with self.assertRaisesRegex(RuntimeError, "after terminal reservation"):
                    lifecycle.experiment_attempt(
                        {"status": "completed", "task_id": "t1"},
                        plan_id=plan["object_id"],
                        admission_gate_id=gate["object_id"],
                        admission_preflight=receipt,
                        commit=True,
                    )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            recovered = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            self.assertTrue(recovered["attempt"].created)
            self.assertTrue(recovered["checkpoint"].committed)

    def test_failed_terminal_retry_preserves_preexisting_reservation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )

            def reserve_then_crash(*args, **kwargs):
                _reserve_experiment_admission_terminal(*args, **kwargs)
                raise RuntimeError("simulated crash after terminal reservation")

            arguments = {
                "attempt": {"status": "completed", "task_id": "t1"},
                "plan_id": plan["object_id"],
                "admission_gate_id": gate["object_id"],
                "admission_preflight": receipt,
                "commit": True,
            }
            with mock.patch(
                "xscientist.research_lifecycle._reserve_experiment_admission_terminal",
                side_effect=reserve_then_crash,
            ):
                with self.assertRaisesRegex(RuntimeError, "after terminal reservation"):
                    lifecycle.experiment_attempt(**arguments)

            reserved = _load_experiment_admission_terminal_reservation(
                repository.path,
                start_object_id=receipt["start_object_id"],
            )
            self.assertIsNotNone(reserved)
            self.assertEqual(
                reserved["schema"],
                "xscientist.experiment-admission-terminal-reservation.v2",
            )
            with mock.patch(
                "xscientist.research_lifecycle.create_checkpoint",
                return_value=CheckpointResult(
                    created=False,
                    committed=False,
                    reason="simulated retry checkpoint refusal",
                ),
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "checkpoint was not committed"
                ):
                    lifecycle.experiment_attempt(**arguments)

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertEqual(
                _load_experiment_admission_terminal_reservation(
                    repository.path,
                    start_object_id=receipt["start_object_id"],
                ),
                reserved,
            )
            recovered = lifecycle.experiment_attempt(**arguments)
            self.assertTrue(recovered["attempt"].created)
            self.assertTrue(recovered["checkpoint"].committed)

    def test_terminal_retry_reuses_reserved_policy_snapshot_after_denial(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            scope = {"task_id": "t1"}
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
                claim=True,
            )

            def reserve_then_crash(*args, **kwargs):
                _reserve_experiment_admission_terminal(*args, **kwargs)
                raise RuntimeError("simulated crash after terminal reservation")

            with mock.patch(
                "xscientist.research_lifecycle._reserve_experiment_admission_terminal",
                side_effect=reserve_then_crash,
            ):
                with self.assertRaisesRegex(RuntimeError, "after terminal reservation"):
                    lifecycle.experiment_attempt(
                        {"status": "completed", "task_id": "t1"},
                        plan_id=plan["object_id"],
                        admission_gate_id=authorization["object_id"],
                        admission_preflight=receipt,
                        commit=True,
                    )

            reservation = _load_experiment_admission_terminal_reservation(
                repository.path,
                start_object_id=receipt["start_object_id"],
            )
            self.assertIsNotNone(reservation)
            self.assertEqual(
                reservation["terminal_policy_snapshot"]["current_policy_status"],
                "authorized",
            )
            self.assertFalse(reservation["terminal_policy_snapshot"]["quarantined"])

            denial = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope=scope,
                extra_relations=[
                    {"type": "supersedes", "target": authorization["object_id"]}
                ],
            )
            recovered = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=authorization["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            saved = repository.get(recovered["attempt"].object_id)
            self.assertEqual(saved["payload"]["current_policy_status"], "authorized")
            self.assertFalse(saved["payload"]["quarantined"])
            self.assertTrue(saved["payload"]["admitted_at_start"])
            self.assertFalse(saved["payload"]["claim_promotion_allowed"])
            self.assertEqual(
                repository.get(denial["object_id"])["payload"]["decision"], "deny"
            )
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 1)

    def test_terminal_retry_commits_existing_uncommitted_attempt_object(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            first = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=False,
            )
            committed_before = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertNotIn(first["attempt"].object_id, committed_before)

            recovered = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            self.assertFalse(recovered["attempt"].created)
            self.assertEqual(recovered["attempt"].object_id, first["attempt"].object_id)
            self.assertTrue(recovered["checkpoint"].committed)

    def test_terminal_retry_of_committed_attempt_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            first = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            terminal_head = repository.status()["head"]

            recovered = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )
            self.assertEqual(repository.status()["head"], terminal_head)
            self.assertEqual(recovered["attempt"].object_id, first["attempt"].object_id)
            self.assertFalse(recovered["attempt"].created)
            self.assertTrue(recovered["checkpoint"].committed)
            self.assertFalse(recovered["checkpoint"].created)
            self.assertEqual(recovered["checkpoint"].commit, terminal_head)

    def test_terminal_retry_rejects_checkpoint_status_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            arguments = {
                "attempt": {"status": "failed", "task_id": "t1"},
                "plan_id": plan["object_id"],
                "admission_gate_id": gate["object_id"],
                "admission_preflight": receipt,
            }
            first = lifecycle.experiment_attempt(**arguments, commit=False)
            relative_attempt = (
                first["attempt"].path.resolve().relative_to(repository.path).as_posix()
            )
            mismatched = create_checkpoint(
                repository.path,
                stage="experiment_admission_terminal",
                subject="commit failed attempt with completed status",
                status="completed",
                only_paths=[relative_attempt],
            )
            self.assertTrue(mismatched.committed)

            with self.assertRaisesRegex(
                ResearchGitError, "valid terminal-only checkpoint"
            ):
                lifecycle.experiment_attempt(**arguments, commit=True)
            self.assertEqual(repository.status()["head"], mismatched.commit)

    def test_terminal_retry_with_conflicting_attempt_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            first = lifecycle.experiment_attempt(
                {"status": "completed", "task_id": "t1"},
                plan_id=plan["object_id"],
                admission_gate_id=gate["object_id"],
                admission_preflight=receipt,
                commit=True,
            )

            with self.assertRaisesRegex(
                ResearchGitError, "conflicting attempt identity"
            ):
                lifecycle.experiment_attempt(
                    {
                        "status": "completed",
                        "summary": "conflicting retry",
                        "task_id": "t1",
                    },
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=True,
                )
            self.assertEqual(
                [
                    item["object_id"]
                    for item in repository.objects(kind="experiment_attempt")
                ],
                [first["attempt"].object_id],
            )

    def test_direct_attempt_post_commit_exception_preserves_committed_object(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            original_create_checkpoint = create_checkpoint
            committed_checkpoint = None

            def commit_then_raise(*args, **kwargs):
                nonlocal committed_checkpoint
                committed_checkpoint = original_create_checkpoint(*args, **kwargs)
                self.assertTrue(committed_checkpoint.committed)
                raise RuntimeError("simulated exception after Git commit")

            with mock.patch(
                "xscientist.research_lifecycle.create_checkpoint",
                side_effect=commit_then_raise,
            ):
                with self.assertRaisesRegex(RuntimeError, "after Git commit"):
                    lifecycle.experiment_attempt(
                        {"status": "completed", "task_id": "t1"},
                        plan_id=plan["object_id"],
                        admission_gate_id=gate["object_id"],
                        admission_preflight=receipt,
                        commit=True,
                    )

            self.assertIsNotNone(committed_checkpoint)
            attempts = repository.objects(kind="experiment_attempt")
            self.assertEqual(len(attempts), 1)
            attempt_path = (
                repository.path
                / ".xscientist"
                / "objects"
                / "experiment_attempt"
                / f"{attempts[0]['object_id']}.json"
            )
            self.assertTrue(attempt_path.is_file())
            reservation_path = (
                _repository_common_git_dir(repository.path)
                / "xscientist"
                / "admission-terminal-reservations"
                / f"{receipt['start_object_id']}.json"
            )
            self.assertTrue(reservation_path.is_file())
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertIn(attempts[0]["object_id"], committed_ids)
            status = subprocess.run(
                ["git", "-C", str(repository.path), "status", "--short"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(status.stdout, "")

    def test_direct_attempt_holds_repository_lock_through_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "t1"},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                claim=True,
            )
            contender_started = threading.Event()
            contender_acquired = threading.Event()
            contender_finished = threading.Event()
            contender_errors: list[BaseException] = []

            def contend_for_lock() -> None:
                contender_started.set()
                try:
                    with _repository_lock(repository.path, timeout_seconds=5.0):
                        contender_acquired.set()
                except BaseException as exc:
                    contender_errors.append(exc)
                finally:
                    contender_finished.set()

            contender = threading.Thread(target=contend_for_lock)
            original_create_checkpoint = create_checkpoint

            def checkpoint_while_contended(*args, **kwargs):
                contender.start()
                self.assertTrue(contender_started.wait(timeout=1.0))
                self.assertFalse(contender_acquired.wait(timeout=0.2))
                return original_create_checkpoint(*args, **kwargs)

            with mock.patch(
                "xscientist.research_lifecycle.create_checkpoint",
                side_effect=checkpoint_while_contended,
            ):
                result = lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=True,
                )

            self.assertTrue(result["checkpoint"].committed)
            self.assertTrue(contender_finished.wait(timeout=2.0))
            contender.join(timeout=1.0)
            self.assertFalse(contender.is_alive())
            self.assertEqual(contender_errors, [])
            self.assertTrue(contender_acquired.is_set())

    def test_project_terminal_checkpoint_is_forced_under_manual_policy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            scope = self._execution_scope(
                repository,
                idea_index=0,
                study_phase="exploratory",
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=scope,
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            with _experiment_admission_execution_lease(repository.path, receipt):
                pass
            start_checkpoint = repository.status()["head"]
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {0: plan["object_id"]},
                    "admission_gates": {0: gate["object_id"]},
                    "admission_starts": {0: receipt["start_object_id"]},
                },
                _experiment_admission_preflights={0: receipt},
                research_git_strict=True,
            )

            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                _record_local_research_attempt_objects(
                    args,
                    results=[{"idea_idx": 0, "status": "success"}],
                )

            terminal_checkpoint = repository.status()["head"]
            self.assertNotEqual(terminal_checkpoint, start_checkpoint)
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment_admission_terminal")
            self.assertEqual(checkpoint["parent_commit"], start_checkpoint)
            self.assertEqual(len(checkpoint["changed_paths"]), 1)
            self.assertTrue(
                checkpoint["changed_paths"][0].startswith(
                    ".xscientist/objects/experiment_attempt/"
                )
            )

    def test_project_attempt_batch_failure_is_atomic_and_requires_all_starts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            for idea_index in (0, 1):
                self._gate(
                    repository,
                    hypothesis=hypothesis,
                    plan=plan,
                    scope=self._execution_scope(
                        repository,
                        idea_index=idea_index,
                        study_phase="exploratory",
                    ),
                )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {0: plan["object_id"], 1: plan["object_id"]}
                },
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args,
                idea_indices=[0, 1],
            )
            start_checkpoint = repository.status()["head"]

            args._experiment_admission_preflights = {0: receipts[0]}
            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "required Research VCS experiment recording failed"
                ):
                    _record_local_research_attempt_objects(
                        args,
                        results=[
                            {"idea_idx": 0, "status": "success"},
                            {"idea_idx": 1, "status": "success"},
                        ],
                    )
            self.assertEqual(repository.status()["head"], start_checkpoint)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

            args._experiment_admission_preflights = receipts
            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "must exactly match every committed admission start"
                ):
                    _record_local_research_attempt_objects(
                        args,
                        results=[{"idea_idx": 0, "status": "success"}],
                    )
            self.assertEqual(repository.status()["head"], start_checkpoint)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

            args._research_vcs_ids["admission_starts"].pop(1)
            args._research_vcs_ids["admission_gates"].pop(1)
            args._experiment_admission_preflights = {0: receipts[0]}
            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "must exactly match every committed admission start"
                ):
                    _record_local_research_attempt_objects(
                        args,
                        results=[{"idea_idx": 0, "status": "success"}],
                    )
            self.assertEqual(repository.status()["head"], start_checkpoint)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            status = subprocess.run(
                ["git", "-C", str(repository.path), "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(status.stdout, "")

    def test_project_checkpoint_refusal_releases_terminal_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            for idea_index in (0, 1):
                self._gate(
                    repository,
                    hypothesis=hypothesis,
                    plan=plan,
                    scope=self._execution_scope(
                        repository,
                        idea_index=idea_index,
                        study_phase="exploratory",
                    ),
                )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={
                    "plans": {0: plan["object_id"], 1: plan["object_id"]}
                },
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args, idea_indices=[0, 1]
            )
            for receipt in receipts.values():
                with _experiment_admission_execution_lease(repository.path, receipt):
                    pass
            results = [
                {"idea_idx": 0, "status": "success"},
                {"idea_idx": 1, "status": "failed"},
            ]

            with (
                mock.patch(
                    "ai_scientist.apps.project._research_attempt_provenance",
                    return_value={},
                ),
                mock.patch(
                    "xscientist.research_git.create_checkpoint",
                    return_value=CheckpointResult(
                        created=False,
                        committed=False,
                        reason="simulated checkpoint refusal",
                    ),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "terminal checkpoint was not committed"
                ):
                    _record_local_research_attempt_objects(
                        args,
                        results=results,
                    )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                _record_local_research_attempt_objects(
                    args,
                    results=results,
                )
            self.assertEqual(len(repository.objects(kind="experiment_attempt")), 2)

    def test_project_post_commit_exception_preserves_terminal_reservations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=0,
                    study_phase="exploratory",
                ),
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args, idea_indices=[0]
            )
            with _experiment_admission_execution_lease(repository.path, receipts[0]):
                pass
            original_create_checkpoint = create_checkpoint

            def commit_then_raise(*checkpoint_args, **checkpoint_kwargs):
                checkpoint = original_create_checkpoint(
                    *checkpoint_args, **checkpoint_kwargs
                )
                self.assertTrue(checkpoint.committed)
                raise RuntimeError("simulated exception after batch Git commit")

            with (
                mock.patch(
                    "ai_scientist.apps.project._research_attempt_provenance",
                    return_value={},
                ),
                mock.patch(
                    "xscientist.research_git.create_checkpoint",
                    side_effect=commit_then_raise,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "after batch Git commit"):
                    _record_local_research_attempt_objects(
                        args,
                        results=[{"idea_idx": 0, "status": "success"}],
                    )

            attempts = repository.objects(kind="experiment_attempt")
            self.assertEqual(len(attempts), 1)
            reservation_path = (
                _repository_common_git_dir(repository.path)
                / "xscientist"
                / "admission-terminal-reservations"
                / f"{receipts[0]['start_object_id']}.json"
            )
            self.assertTrue(reservation_path.is_file())
            committed_ids = {
                item["object_id"]
                for item in list_research_objects_at_ref(repository.path, "HEAD")
            }
            self.assertIn(attempts[0]["object_id"], committed_ids)

    def test_required_empty_result_resume_is_a_noop_without_issued_starts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                _experiment_admission_preflights={},
                research_git_strict=True,
            )
            head = repository.status()["head"]

            _record_local_research_attempt_objects(args, results=[])

            self.assertEqual(repository.status()["head"], head)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])
            self.assertNotIn("attempts", args._research_vcs_ids)

    def test_required_empty_result_does_not_ignore_issued_start(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=0,
                    study_phase="exploratory",
                ),
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                research_git_strict=True,
            )
            _preflight_local_research_experiment_admission(args, idea_indices=[0])

            with self.assertRaisesRegex(
                RuntimeError, "must exactly match every committed admission start"
            ):
                _record_local_research_attempt_objects(args, results=[])
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_failed_project_attempt_sets_failed_terminal_checkpoint_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=0,
                    study_phase="exploratory",
                ),
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args, idea_indices=[0]
            )
            with _experiment_admission_execution_lease(repository.path, receipts[0]):
                pass

            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                _record_local_research_attempt_objects(
                    args,
                    results=[{"idea_idx": 0, "status": "failed"}],
                )

            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment_admission_terminal")
            self.assertEqual(checkpoint["status"], "failed")

    def test_required_attempt_rejects_idea_index_type_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=1,
                    study_phase="exploratory",
                ),
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {1: plan["object_id"]}},
                research_git_strict=True,
            )
            _preflight_local_research_experiment_admission(args, idea_indices=[1])
            start_checkpoint = repository.status()["head"]

            for alias in (True, 1.9, "1"):
                with self.subTest(alias=alias):
                    with mock.patch(
                        "ai_scientist.apps.project._research_attempt_provenance",
                        return_value={},
                    ):
                        with self.assertRaisesRegex(RuntimeError, "invalid idea_idx"):
                            _record_local_research_attempt_objects(
                                args,
                                results=[{"idea_idx": alias, "status": "success"}],
                            )
                    self.assertEqual(repository.status()["head"], start_checkpoint)
                    self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_required_attempt_rejects_aliased_mapping_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research",
                admission_mode="required",
                checkpoint_policy="manual",
            )
            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope=self._execution_scope(
                    repository,
                    idea_index=0,
                    study_phase="exploratory",
                ),
            )
            args = argparse.Namespace(
                project_dir=str(repository.path),
                _research_git_active=True,
                _research_vcs_ids={"plans": {0: plan["object_id"]}},
                research_git_strict=True,
            )
            receipts = _preflight_local_research_experiment_admission(
                args, idea_indices=[0]
            )
            start_id = args._research_vcs_ids["admission_starts"][0]
            args._research_vcs_ids["admission_starts"]["0"] = start_id
            args._experiment_admission_preflights = {
                0: receipts[0],
                "0": receipts[0],
            }
            start_checkpoint = repository.status()["head"]

            with mock.patch(
                "ai_scientist.apps.project._research_attempt_provenance",
                return_value={},
            ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous idea index"):
                    _record_local_research_attempt_objects(
                        args,
                        results=[{"idea_idx": 0, "status": "success"}],
                    )

            self.assertEqual(repository.status()["head"], start_checkpoint)
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_scope_comparison_distinguishes_boolean_from_integer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"idea_index": 0},
            )
            lifecycle = ResearchLifecycle(repository)
            receipt = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"idea_index": 0},
            )
            forged = json.loads(json.dumps(receipt))
            forged["execution_scope"]["idea_index"] = False
            final_core = {
                key: value for key, value in forged.items() if key != "receipt_hash"
            }
            forged["receipt_hash"] = canonical_content_hash(final_core)

            with self.assertRaisesRegex(
                ResearchGitError, "finalized experiment admission receipt is invalid"
            ):
                lifecycle.validate_finalized_experiment_admission_start(forged)
            with self.assertRaises(ResearchGitError):
                lifecycle.experiment_attempt(
                    {"status": "completed", "idea_index": False},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    admission_preflight=receipt,
                    commit=False,
                )
            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_uncommitted_policy_downgrade_cannot_bypass_required_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            config_path = repository.path / "research.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["experiment_admission"] = {"mode": "legacy"}
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )

            self.assertEqual(repository.experiment_admission_mode(), "required")
            with self.assertRaisesRegex(
                ResearchGitError, "raw experiment_attempt recording is disabled"
            ):
                record_research_object(
                    repository.path,
                    kind="experiment_attempt",
                    payload={"status": "completed"},
                    state="completed",
                )
            with self.assertRaisesRegex(
                ResearchGitError, "preflight receipt is missing"
            ):
                _revalidate_experiment_admission_preflight(repository.path, None)

    def test_deleted_worktree_policy_cannot_hide_committed_required_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            (repository.path / "research.yaml").unlink()

            with self.assertRaisesRegex(
                ResearchGitError, "not an XScientist research repository"
            ):
                _revalidate_experiment_admission_preflight(repository.path, None)

    def test_required_worker_rejects_missing_receipt_before_any_experiment_setup(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            worker_args = (
                str(repository.path),
                str(repository.path),
                0,
                {
                    "Name": "Unauthorized worker",
                    "Short Hypothesis": "The worker must fail before setup.",
                    "Experiments": ["This experiment must not run."],
                },
                None,
                "model-writeup",
                "model-citation",
                "model-review",
                "model-plots",
                "model-small",
                1,
                1,
                "normal",
                1,
                0,
                1,
                0,
                0.0,
                "depth",
                False,
                "publishable",
                "model-quality",
                "iclr",
                8.0,
                8.0,
                0,
                0,
                False,
                None,
                None,
                "default",
                0,
                False,
                0,
                "classic_pipeline",
                "open_ended",
                "adaptive",
                False,
                False,
                "classic_pipeline",
                None,
                None,
            )
            self.assertEqual(len(worker_args), 42)

            with mock.patch(
                "ai_scientist.apps.project.perform_experiments_bfts"
            ) as perform_bfts_mock:
                result = process_single_idea(worker_args)

            perform_bfts_mock.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "experiment_admission")
            self.assertIn("preflight receipt is missing", result["error"])
            self.assertFalse((repository.path / "02_experiments").exists())

    def test_post_bfts_pipeline_failures_preserve_completed_experiment_status(
        self,
    ) -> None:
        cases = (
            ("sample_gate", False),
            ("writeup", False),
            ("hostile_critic", False),
            ("final_submission_bar", True),
        )
        for expected_stage, high_quality_mode in cases:
            with (
                self.subTest(stage=expected_stage),
                tempfile.TemporaryDirectory() as td,
            ):
                root = Path(td) / "project"
                root.mkdir()

                def write_paper(*, base_folder: str, **_kwargs) -> bool:
                    if expected_stage == "writeup":
                        return False
                    Path(base_folder, "paper.pdf").write_bytes(
                        b"%PDF-1.4\nmock paper\n%%EOF\n"
                    )
                    return True

                def final_review(**kwargs) -> dict:
                    pdf_path = Path(kwargs["paper_dir"]) / "paper.pdf"
                    return {
                        "found": False,
                        "pdf_path": str(pdf_path),
                        "review_text": None,
                        "review_img": None,
                    }

                quality_result = {
                    "quality_score_after": 9.0,
                    "rigor_score_after": 9.0,
                    "claim_support_after": 9.0,
                    "quality_gate_passed": True,
                    "submission_priority_score": 9.0,
                    "submission_priority_tier": "P1",
                    "blocker_count": 0,
                }
                quality_pass = {
                    "quality_result": quality_result,
                    "acceptance": {"accepted": True, "reasons": []},
                    "summary": "High-quality pass: mocked",
                    "effective_priority_bar": None,
                    "effective_blocker_bar": None,
                }
                critic_result = {
                    "ran": expected_stage == "hostile_critic",
                    "review_roles_used": [],
                    "active_issue_count": (
                        1 if expected_stage == "hostile_critic" else 0
                    ),
                    "blocking_issue_count": (
                        1 if expected_stage == "hostile_critic" else 0
                    ),
                    "critic_findings_file": None,
                }
                sample_gate = {
                    "full_generation_allowed": expected_stage != "sample_gate",
                    "sample_task_count": 1,
                    "result": {
                        "reasons": (
                            ["sample failed"] if expected_stage == "sample_gate" else []
                        )
                    },
                }
                research_plan = {
                    "tasks": [],
                    "workflow_mode": "classic_pipeline",
                    "execution_policy": {},
                }

                with ExitStack() as stack:
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.activate_llm_tracing")
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.deactivate_llm_tracing")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.initialize_pipeline_contracts"
                        )
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.record_decision")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project._write_experiment_pipeline_seed_artifacts",
                            return_value=({}, research_plan, {}),
                        )
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.idea_to_markdown")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.write_bfts_task_descriptor"
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.edit_bfts_config_file",
                            return_value="/tmp/mock-bfts.yaml",
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.perform_experiments_bfts",
                            return_value={"status": "completed"},
                        )
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.write_experiment_report")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project._build_experiment_registry_rows",
                            return_value=[],
                        )
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.save_experiment_registry")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.evaluate_and_save_sample_gate",
                            return_value=sample_gate,
                        )
                    )
                    stack.enter_context(
                        mock.patch("ai_scientist.apps.project.aggregate_plots")
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project._save_manuscript_contract_state"
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.build_writeup_execution_plan",
                            return_value={
                                "target_venue": "iclr",
                                "strategy_feedback": {"rationale": []},
                                "num_cite_rounds": 1,
                                "writeup_retries": 1,
                                "writeup_engine": "normal",
                                "page_limit": 9,
                            },
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.gather_citations",
                            return_value="",
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.perform_normal_writeup",
                            side_effect=write_paper,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.load_guardrail_artifacts",
                            return_value=([], []),
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.execute_quality_workflow_with_followups",
                            return_value=quality_pass,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.record_quality_fallback_if_needed",
                            return_value=None,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.build_review_execution_plan",
                            return_value={"strategy_feedback": {"rationale": []}},
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.load_experiment_todo_payload",
                            return_value={"tasks": []},
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.execute_review_suite",
                            side_effect=final_review,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.run_independent_critic_pass",
                            return_value=critic_result,
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.run_integrity_forensics_for_manuscript",
                            return_value={"enabled": False, "status": "disabled"},
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.finalize_ara_for_idea",
                            return_value=SimpleNamespace(
                                export=None,
                                claim_coverage=None,
                            ),
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.summarise_finalize",
                            return_value=[],
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.evaluate_todo_progress_snapshot",
                            return_value={"counts": {"total_tasks": 0}},
                        )
                    )
                    stack.enter_context(
                        mock.patch(
                            "ai_scientist.apps.project.evaluate_final_submission_readiness",
                            return_value={
                                "accepted": False,
                                "reasons": ["mock submission block"],
                                "signals": {},
                            },
                        )
                    )
                    result = process_single_idea(
                        self._pipeline_worker_args(
                            root,
                            high_quality_mode=high_quality_mode,
                        )
                    )

                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["stage"], expected_stage)
                self.assertEqual(result["registry_status"], "completed")
                self.assertEqual(_experiment_checkpoint_status([result]), "completed")

    def test_gate_scope_mismatch_and_live_exact_deny_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            gate = self._gate(
                repository, hypothesis=hypothesis, plan=plan, scope={"task_id": "t1"}
            )
            lifecycle = ResearchLifecycle(repository)

            with self.assertRaisesRegex(ResearchGitError, "execution scope"):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t2"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    commit=False,
                )

            self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                decision="deny",
                scope={"task_id": "t1"},
            )
            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                lifecycle.experiment_attempt(
                    {"status": "completed", "task_id": "t1"},
                    plan_id=plan["object_id"],
                    admission_gate_id=gate["object_id"],
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_orphaned_receipt_remains_available_for_incident_audit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )

            result = ResearchLifecycle(repository).experiment_attempt(
                {"status": "orphaned", "summary": "observed unmanaged process"},
                plan_id=plan["object_id"],
                commit=False,
            )

            saved = repository.get(result["attempt"].object_id)
            self.assertEqual(saved["state"], "failed")
            self.assertIs(saved["payload"]["experiment_authorized"], False)
            self.assertIs(saved["payload"]["claim_promotion_allowed"], False)
            self.assertEqual(saved["payload"]["admission"]["status"], "orphaned")

    def test_record_validated_cannot_bypass_required_admission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode="required"
            )

            with self.assertRaisesRegex(
                ResearchGitError, "must be returned by validate_experiment_attempt"
            ):
                ResearchLifecycle(repository).record_validated_experiment_attempt(
                    {
                        "payload": {"status": "completed", "summary": "forged"},
                        "state": "completed",
                        "relations": [],
                        "actor": None,
                        "provenance": None,
                    },
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_signed_admission_schema_is_published(self) -> None:
        validator = schema_validator("experiment_admission")
        self.assertEqual(
            validator.schema["properties"]["purpose"]["const"],
            EXPERIMENT_ADMISSION_PROTOCOL,
        )

    def test_external_ed25519_authorization_is_committed_usable_and_idempotent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            head_before = repository.status()["head"]
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                decision="authorize",
                expected_head=head_before,
            )
            self.assertTrue(schema_validator("experiment_admission").is_valid(payload))
            self.assertNotIn("experiment_authorized", payload)
            self.assertEqual(repository.objects(kind="gate_decision"), [])
            attestation, trust_store = self._external_admission_signature(root, payload)

            ingested = lifecycle.ingest_experiment_admission(
                payload,
                attestation=attestation,
                trust_store=trust_store,
            )

            self.assertTrue(ingested["checkpoint"].committed)
            gate = repository.get(ingested["gate"].object_id)
            self.assertEqual(
                gate["actor"],
                {
                    "actor_id": "xscientist-experiment-admission-verifier",
                    "authority": "deterministic_gate",
                },
            )
            self.assertIs(gate["payload"]["experiment_authorized"], True)
            self.assertIs(gate["payload"]["claim_promotion_allowed"], False)
            authority = gate["payload"]["authority_receipt"]
            attestation_record = authority["attestation_record"]
            self.assertEqual(attestation_record["signature_encoding"], "hex")
            self.assertEqual(
                attestation_record["envelope"],
                {
                    key: value
                    for key, value in attestation.items()
                    if key != "signature"
                },
            )
            self.assertEqual(authority["signed_payload"], payload)
            self.assertEqual(authority["key_id"], attestation["key_id"])
            self.assertEqual(
                authority["receipt_hash"],
                canonical_content_hash(
                    {
                        key: value
                        for key, value in authority.items()
                        if key != "receipt_hash"
                    }
                ),
            )
            checkpoint = show_checkpoint(repository.path)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "experiment_admission")
            self.assertEqual(
                checkpoint["changed_paths"],
                [".xscientist/objects/gate_decision/" f"{gate['object_id']}.json"],
            )
            preflight = lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
            )
            self.assertEqual(preflight["admission_gate_id"], gate["object_id"])

            committed_head = repository.status()["head"]
            repeated = lifecycle.ingest_experiment_admission(
                payload,
                attestation=attestation,
                trust_store=trust_store,
            )
            self.assertTrue(repeated["idempotent"])
            self.assertIsNone(repeated["checkpoint"])
            self.assertEqual(repeated["gate"].object_id, gate["object_id"])
            self.assertEqual(repository.status()["head"], committed_head)
            self.assertEqual(len(repository.objects(kind="gate_decision")), 1)

    def test_signed_denial_supersedes_only_the_exact_active_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            authorization_payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                decision="authorize",
            )
            authorization_attestation, trust_store = self._external_admission_signature(
                root, authorization_payload
            )
            authorization = lifecycle.ingest_experiment_admission(
                authorization_payload,
                attestation=authorization_attestation,
                trust_store=trust_store,
            )["gate"]
            denial_payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope={"task_id": "t1"},
                decision="deny",
            )
            denial_attestation, trust_store = self._external_admission_signature(
                root, denial_payload
            )
            denial = lifecycle.ingest_experiment_admission(
                denial_payload,
                attestation=denial_attestation,
                trust_store=trust_store,
            )["gate"]
            denial_object = repository.get(denial.object_id)
            self.assertIn(
                {"type": "supersedes", "target": authorization.object_id},
                denial_object["relations"],
            )
            self.assertIs(denial_object["payload"]["experiment_authorized"], False)
            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope={"task_id": "t1"},
                )

    def test_signed_admission_rejects_untrusted_inputs_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                decision="authorize",
            )
            attestation, trust_store = self._external_admission_signature(root, payload)

            with self.assertRaisesRegex(ResearchGitError, "requires an Ed25519"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation={},
                    trust_store=trust_store,
                )

            with self.assertRaisesRegex(ResearchGitError, "requires an Ed25519"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=sign_attestation(
                        payload,
                        purpose=EXPERIMENT_ADMISSION_PROTOCOL,
                        identity="service:shared-secret-reviewer",
                        key_id="key:shared-secret",
                        algorithm="hmac-sha256",
                        key=b"not-an-independent-trust-root",
                    ),
                    trust_store=trust_store,
                )

            tampered_payload = {**payload, "decision": "deny"}
            with self.assertRaisesRegex(ResearchGitError, "payload_hash_mismatch"):
                lifecycle.ingest_experiment_admission(
                    tampered_payload,
                    attestation=attestation,
                    trust_store=trust_store,
                )

            _, other_store = self._external_admission_signature(
                root,
                payload,
                identity="human:other-reviewer",
                key_id="key:other-reviewer",
            )
            with self.assertRaisesRegex(ResearchGitError, "key_untrusted"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=attestation,
                    trust_store=other_store,
                )

            revoked_attestation, revoked_store = self._external_admission_signature(
                root, payload, revoked=True
            )
            with self.assertRaisesRegex(ResearchGitError, "verification failed"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=revoked_attestation,
                    trust_store=revoked_store,
                )

            inside_authority = repository.path / "self-authority"
            shutil.copytree(root / "external-authority", inside_authority)
            with self.assertRaisesRegex(
                ResearchGitError, "trust_store_inside_research_boundary"
            ):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=attestation,
                    trust_store=inside_authority / "trust.json",
                )

            producer_attestation, producer_store = self._external_admission_signature(
                root,
                payload,
                identity="human:xscientist",
                key_id="key:producer-alias",
            )
            with self.assertRaisesRegex(ResearchGitError, "verifier independent"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=producer_attestation,
                    trust_store=producer_store,
                )
            self.assertEqual(repository.objects(kind="gate_decision"), [])

    def test_signed_admission_rejects_head_drift_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                decision="authorize",
            )
            attestation, trust_store = self._external_admission_signature(root, payload)
            repository.record(
                "review",
                {"summary": "Repository moved after the external decision."},
                state="draft",
            )
            repository.commit(stage="review", subject="advance signed admission head")

            with self.assertRaisesRegex(ResearchGitError, "HEAD changed"):
                lifecycle.ingest_experiment_admission(
                    payload,
                    attestation=attestation,
                    trust_store=trust_store,
                )
            self.assertEqual(repository.objects(kind="gate_decision"), [])

    def test_signed_admission_checkpoint_failure_removes_new_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                decision="authorize",
            )
            attestation, trust_store = self._external_admission_signature(root, payload)
            head_before = repository.status()["head"]

            with mock.patch(
                "xscientist.research_lifecycle.create_checkpoint",
                side_effect=ResearchGitError("synthetic checkpoint failure"),
            ):
                with self.assertRaisesRegex(
                    ResearchGitError, "synthetic checkpoint failure"
                ):
                    lifecycle.ingest_experiment_admission(
                        payload,
                        attestation=attestation,
                        trust_store=trust_store,
                    )
            self.assertEqual(repository.status()["head"], head_before)
            self.assertEqual(repository.objects(kind="gate_decision"), [])

    def test_valid_signature_path_collision_is_encoded_without_privacy_bypass(
        self,
    ) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("cryptography is required for Ed25519 admission tests")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            lifecycle = ResearchLifecycle(repository)
            payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                decision="authorize",
            )
            private_key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            authority_root = root / "fixed-external-authority"
            authority_root.mkdir()
            public_key_path = authority_root / "reviewer.pub"
            public_key_path.write_bytes(
                private_key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            trust_store = authority_root / "trust.json"
            trust_store.write_text(
                json.dumps(
                    {
                        "keys": {
                            "key:path-collision-reviewer": {
                                "identity": "human:path-collision-reviewer",
                                "algorithm": "ed25519",
                                "public_key_file": public_key_path.name,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            attestation = None
            colliding_signature = ""
            for index in range(10_000):
                candidate = sign_attestation(
                    payload,
                    purpose=EXPERIMENT_ADMISSION_PROTOCOL,
                    identity="human:path-collision-reviewer",
                    key_id="key:path-collision-reviewer",
                    algorithm="ed25519",
                    key=private_pem,
                    issued_at=f"2026-09-04T00:00:00.{index:06d}+00:00",
                )
                signature = str(candidate["signature"])
                if redact_sensitive_text(signature) != signature:
                    attestation = candidate
                    colliding_signature = signature
                    break
            self.assertIsNotNone(attestation)

            ingested = lifecycle.ingest_experiment_admission(
                payload,
                attestation=attestation,
                trust_store=trust_store,
            )
            gate = repository.get(ingested["gate"].object_id)
            record = gate["payload"]["authority_receipt"]["attestation_record"]
            self.assertEqual(record["signature_encoding"], "hex")
            self.assertNotIn(colliding_signature, json.dumps(gate))

            with self.assertRaisesRegex(ResearchGitError, "privacy gate refused"):
                repository.record(
                    "review",
                    {"summary": colliding_signature},
                    state="draft",
                )

    def test_experiment_admission_cli_prepare_and_ingest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(
                root / "research", admission_mode="required"
            )
            payload_path = root / "admission-payload.json"
            head_before = repository.status()["head"]
            output = io.StringIO()
            with redirect_stdout(output):
                returncode = research_main(
                    [
                        "experiment-admission",
                        "prepare",
                        "--repo",
                        str(repository.path),
                        "--plan",
                        plan["object_id"],
                        "--decision",
                        "authorize",
                        "--scope",
                        'task_id="t1"',
                        "--output",
                        str(payload_path),
                        "--json",
                    ]
                )
            self.assertEqual(returncode, 0)
            prepared = json.loads(output.getvalue())
            self.assertEqual(prepared["status"], "prepared")
            self.assertIs(prepared["authoritative"], False)
            self.assertEqual(repository.status()["head"], head_before)
            self.assertEqual(repository.objects(kind="gate_decision"), [])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            attestation, trust_store = self._external_admission_signature(root, payload)
            attestation_path = root / "admission-attestation.json"
            attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

            output = io.StringIO()
            with redirect_stdout(output):
                returncode = research_main(
                    [
                        "experiment-admission",
                        "ingest",
                        "--repo",
                        str(repository.path),
                        "--payload",
                        str(payload_path),
                        "--attestation",
                        str(attestation_path),
                        "--trust-store",
                        str(trust_store),
                        "--json",
                    ]
                )
            self.assertEqual(returncode, 0)
            ingested = json.loads(output.getvalue())
            self.assertEqual(ingested["status"], "ingested")
            self.assertIs(ingested["experiment_authorized"], True)
            self.assertEqual(len(repository.objects(kind="gate_decision")), 1)

    def test_validated_attempt_contract_is_deeply_immutable_and_detached(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, plan = self._repository(
                Path(td) / "research", admission_mode=None
            )
            lifecycle = ResearchLifecycle(repository)
            attempt = {
                "status": "completed",
                "summary": "sealed legacy attempt",
                "producer_id": "executor:sealed-runner",
                "metrics": {"score": 1.0, "tags": ["baseline"]},
            }
            provenance = {
                "environment_hash": "sha256:" + "a" * 64,
                "dataset_hashes": ["sha256:" + "b" * 64],
            }
            prepared = lifecycle.validate_experiment_attempt(
                attempt,
                plan_id=plan["object_id"],
                provenance=provenance,
            )

            with self.assertRaises(TypeError):
                prepared["state"] = "failed"
            with self.assertRaises(TypeError):
                prepared["actor"]["actor_id"] = "executor:attacker"
            with self.assertRaises(TypeError):
                prepared["provenance"]["dataset_hashes"][0] = "sha256:" + "c" * 64
            with self.assertRaises(TypeError):
                prepared["relations"][0]["target"] = "rso-0000000000000000"
            with self.assertRaises(TypeError):
                prepared["payload"]["metrics"]["score"] = 0.0

            attempt["metrics"]["score"] = 0.0
            attempt["metrics"]["tags"].append("mutated")
            provenance["dataset_hashes"][0] = "sha256:" + "c" * 64
            recorded = lifecycle.record_validated_experiment_attempt(
                prepared,
                commit=False,
            )
            saved = repository.get(recorded["attempt"].object_id)
            self.assertEqual(saved["state"], "completed")
            self.assertEqual(saved["actor"]["actor_id"], "executor:sealed-runner")
            self.assertEqual(
                saved["relations"],
                [
                    {
                        "type": "depends_on",
                        "target": plan["object_id"],
                        "role": "plan",
                    }
                ],
            )
            self.assertEqual(saved["payload"]["metrics"]["score"], 1.0)
            self.assertEqual(saved["payload"]["metrics"]["tags"], ["baseline"])
            self.assertEqual(
                saved["provenance"]["dataset_hashes"],
                ["sha256:" + "b" * 64],
            )

    def test_record_validated_rejects_raw_mapping_in_legacy_repository(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, _, _ = self._repository(
                Path(td) / "research", admission_mode=None
            )

            with self.assertRaisesRegex(
                ResearchGitError, "must be returned by validate_experiment_attempt"
            ):
                ResearchLifecycle(repository).record_validated_experiment_attempt(
                    {
                        "payload": {"status": "completed", "summary": "forged"},
                        "state": "completed",
                        "relations": [],
                        "actor": None,
                        "provenance": None,
                    },
                    commit=False,
                )

            self.assertEqual(repository.objects(kind="experiment_attempt"), [])

    def test_validated_attempt_contract_records_required_admission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository, hypothesis, plan = self._repository(
                Path(td) / "research", admission_mode="required"
            )
            authorization = self._gate(
                repository,
                hypothesis=hypothesis,
                plan=plan,
                scope={"task_id": "sealed"},
            )
            lifecycle = ResearchLifecycle(repository)
            preflight = self._finalized_start(
                lifecycle,
                plan_id=plan["object_id"],
                execution_scope={"task_id": "sealed"},
            )
            with _experiment_admission_execution_lease(repository.path, preflight):
                pass

            prepared = lifecycle.validate_experiment_attempt(
                {"status": "completed", "task_id": "sealed"},
                plan_id=plan["object_id"],
                admission_gate_id=authorization["object_id"],
                admission_preflight=preflight,
            )
            recorded = lifecycle.record_validated_experiment_attempt(
                prepared,
                commit=False,
            )

            saved = repository.get(recorded["attempt"].object_id)
            self.assertEqual(saved["state"], "completed")
            self.assertIs(saved["payload"]["experiment_authorized"], True)
            self.assertIn(
                {
                    "type": "consumes",
                    "target": preflight["start_object_id"],
                    "role": "experiment_admission_start",
                },
                saved["relations"],
            )

    def test_save_experiment_revalidates_after_adding_artifact_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = ResearchRepository.init(root, question="Fixed question")
            artifact = Path(td) / "metrics.json"
            artifact.write_text('{"score": 1}\n', encoding="utf-8")
            validated_artifacts: list[dict] = []
            real_validate = ResearchLifecycle.validate_experiment_attempt

            def capture_validation(lifecycle, attempt, **kwargs):
                validated_artifacts.append(dict(attempt.get("result_artifacts") or {}))
                return real_validate(lifecycle, attempt, **kwargs)

            with mock.patch.object(
                ResearchLifecycle,
                "validate_experiment_attempt",
                new=capture_validation,
            ):
                result = save_experiment(
                    str(root),
                    summary="artifact-bound sealed attempt",
                    status="completed",
                    result_artifact_paths={"metrics": artifact},
                    commit=False,
                )

            self.assertEqual(len(validated_artifacts), 2)
            self.assertEqual(validated_artifacts[0], {})
            self.assertEqual(set(validated_artifacts[1]), {"metrics"})
            saved = repository.get(result["object"].object_id)
            self.assertEqual(
                saved["payload"]["result_artifacts"]["metrics"]["content_hash"],
                saved["payload"]["result_artifact_hashes"]["metrics"],
            )


if __name__ == "__main__":
    unittest.main()
