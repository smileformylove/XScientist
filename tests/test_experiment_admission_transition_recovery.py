from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from ai_scientist.protocol.attestation import sign_attestation
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist.experiment_admission import EXPERIMENT_ADMISSION_PROTOCOL
from xscientist.experiment_admission_state import (
    load_experiment_admission_high_water,
    load_experiment_admission_transition,
)
from xscientist.research_git import ResearchGitError, list_research_objects_at_ref


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ExperimentAdmissionTransitionRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
            )
        except ImportError:
            self.skipTest("cryptography is required for Ed25519 admission tests")
        self.serialization = serialization
        self.private_key = Ed25519PrivateKey.generate()

    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        return completed.stdout.strip()

    @staticmethod
    def _repository(root: Path) -> tuple[ResearchRepository, dict, dict]:
        repository = ResearchRepository.init(
            root,
            question="Should this exact experiment run?",
            git_user_name="Admission Recovery Test",
            git_user_email="admission-recovery@example.invalid",
        )
        config_path = root / "research.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["experiment_admission"] = {"mode": "required"}
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        hypothesis = repository.record(
            "hypothesis",
            {"statement": "The intervention improves the outcome."},
            state="draft",
        )
        plan = repository.record(
            "research_plan",
            {
                "summary": "Run the bounded intervention.",
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
        repository.commit(stage="plan", subject="lock exact test plan")
        return (
            repository,
            repository.get(hypothesis.object_id),
            repository.get(plan.object_id),
        )

    def _authority(self, root: Path) -> tuple[bytes, Path]:
        authority_root = root / "external-authority"
        authority_root.mkdir()
        private_pem = self.private_key.private_bytes(
            encoding=self.serialization.Encoding.PEM,
            format=self.serialization.PrivateFormat.PKCS8,
            encryption_algorithm=self.serialization.NoEncryption(),
        )
        public_key_path = authority_root / "reviewer.pub"
        public_key_path.write_bytes(
            self.private_key.public_key().public_bytes(
                encoding=self.serialization.Encoding.PEM,
                format=self.serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        trust_store = authority_root / "trust.json"
        trust_store.write_text(
            json.dumps(
                {
                    "keys": {
                        "key:independent-reviewer": {
                            "identity": "human:independent-reviewer",
                            "algorithm": "ed25519",
                            "public_key_file": public_key_path.name,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        return private_pem, trust_store

    @staticmethod
    def _attestation(payload: dict, private_pem: bytes) -> dict:
        return sign_attestation(
            payload,
            purpose=EXPERIMENT_ADMISSION_PROTOCOL,
            identity="human:independent-reviewer",
            key_id="key:independent-reviewer",
            algorithm="ed25519",
            key=private_pem,
            issued_at="2026-09-05T00:00:00+00:00",
        )

    def test_committed_denial_publish_failure_recovers_and_blocks_old_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository, _, plan = self._repository(root / "research")
            private_pem, trust_store = self._authority(root)
            lifecycle = ResearchLifecycle(repository, admission_trust_store=trust_store)
            scope = {"task_id": "task-1"}

            authorization_payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope=scope,
                decision="authorize",
            )
            authorization = lifecycle.ingest_experiment_admission(
                authorization_payload,
                attestation=self._attestation(authorization_payload, private_pem),
                trust_store=trust_store,
            )
            authorization_gate = repository.get(authorization["gate"].object_id)
            authorization_head = str(repository.status()["head"])

            linked = root / "old-linked-worktree"
            self._git(
                repository.path,
                "worktree",
                "add",
                "--detach",
                str(linked),
                authorization_head,
            )
            linked_lifecycle = ResearchLifecycle(
                linked, admission_trust_store=trust_store
            )
            baseline = linked_lifecycle.preflight_experiment_admission(
                plan_id=plan["object_id"],
                execution_scope=scope,
            )
            self.assertEqual(
                baseline["admission_gate_id"], authorization_gate["object_id"]
            )

            denial_payload = lifecycle.prepare_experiment_admission_payload(
                plan_id=plan["object_id"],
                execution_scope=scope,
                decision="deny",
            )
            denial_attestation = self._attestation(denial_payload, private_pem)
            real_link = os.link
            publish_attempted = False

            def fail_high_water_publish(
                source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                *args: object,
                **kwargs: object,
            ) -> None:
                nonlocal publish_attempted
                destination_path = Path(destination)
                if (
                    destination_path.parent.name == "events"
                    and "admission-decisions" in destination_path.parts
                ):
                    committed_objects = list_research_objects_at_ref(
                        repository.path, "HEAD"
                    )
                    self.assertTrue(
                        any(
                            item.get("kind") == "gate_decision"
                            and (item.get("payload") or {}).get("decision") == "deny"
                            for item in committed_objects
                        ),
                        "denial gate must be committed before high-water publication",
                    )
                    publish_attempted = True
                    raise OSError("synthetic high-water append failure")
                real_link(source, destination, *args, **kwargs)

            with mock.patch(
                "xscientist.experiment_admission_state.os.link",
                side_effect=fail_high_water_publish,
            ):
                with self.assertRaisesRegex(
                    OSError, "synthetic high-water append failure"
                ):
                    lifecycle.ingest_experiment_admission(
                        denial_payload,
                        attestation=denial_attestation,
                        trust_store=trust_store,
                    )

            self.assertTrue(publish_attempted)
            denial_head = str(repository.status()["head"])
            self.assertNotEqual(denial_head, authorization_head)
            pending = load_experiment_admission_transition(
                repository.path, denial_payload
            )
            self.assertIsNotNone(pending)
            high_water_before_retry = load_experiment_admission_high_water(
                repository.path, denial_payload
            )
            self.assertEqual(
                high_water_before_retry["gate_object_id"],
                authorization_gate["object_id"],
            )

            with self.assertRaisesRegex(ResearchGitError, "transition is pending"):
                linked_lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope=scope,
                )

            recovered = lifecycle.ingest_experiment_admission(
                denial_payload,
                attestation=denial_attestation,
                trust_store=trust_store,
            )
            self.assertTrue(recovered["idempotent"])
            self.assertTrue(recovered["committed"])
            self.assertTrue(recovered["active"])
            self.assertFalse(recovered["usable"])
            self.assertIsNone(recovered["checkpoint"])
            self.assertEqual(str(repository.status()["head"]), denial_head)
            self.assertIsNone(
                load_experiment_admission_transition(repository.path, denial_payload)
            )
            high_water_after_retry = load_experiment_admission_high_water(
                repository.path, denial_payload
            )
            self.assertEqual(high_water_after_retry["decision"], "deny")
            self.assertEqual(high_water_after_retry["checkpoint_commit"], denial_head)

            with self.assertRaisesRegex(ResearchGitError, "is denied"):
                linked_lifecycle.preflight_experiment_admission(
                    plan_id=plan["object_id"],
                    execution_scope=scope,
                )


if __name__ == "__main__":
    unittest.main()
