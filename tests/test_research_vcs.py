from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    build_research_object,
    validate_research_object,
)
from xscientist import ResearchRepository
from xscientist.research_git import ResearchGitError


class ResearchObjectProtocolTests(unittest.TestCase):
    def test_identity_is_deterministic_and_relations_are_canonical(self) -> None:
        first = build_research_object(
            kind="claim",
            state="verified",
            payload={"statement": "H1 improves the registered metric."},
            relations=[
                {"type": "supports", "target": "rso-bbbbbbbbbbbbbbbb"},
                {"type": "depends_on", "target": "rso-aaaaaaaaaaaaaaaa"},
                {"type": "supports", "target": "rso-bbbbbbbbbbbbbbbb"},
            ],
            created_at="2026-01-01T00:00:00+00:00",
        )
        second = build_research_object(
            kind="claim",
            state="verified",
            payload={"statement": "H1 improves the registered metric."},
            relations=list(reversed(first["relations"])),
            created_at="2026-02-01T00:00:00+00:00",
        )

        self.assertEqual(first["object_id"], second["object_id"])
        self.assertEqual(first["content_hash"], second["content_hash"])
        self.assertNotEqual(first["created_at"], second["created_at"])
        self.assertEqual(len(first["relations"]), 2)
        self.assertEqual(first["relations"][0]["type"], "depends_on")

    def test_validation_detects_content_tampering(self) -> None:
        payload = build_research_object(
            kind="hypothesis",
            payload={"statement": "H1"},
        )
        payload["payload"]["statement"] = "H2"

        with self.assertRaisesRegex(ResearchObjectError, "hash mismatch"):
            validate_research_object(payload)


@unittest.skipUnless(shutil.which("git"), "Git is required for repository tests")
class ResearchRepositoryTests(unittest.TestCase):
    def _init(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Does the intervention improve the registered metric?",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def test_record_list_load_commit_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)

            question = repository.record(
                "question",
                {"text": "Does the intervention improve the registered metric?"},
            )
            repeated = repository.record(
                "question",
                {"text": "Does the intervention improve the registered metric?"},
            )
            hypothesis = repository.record(
                "hypothesis",
                {"statement": "The intervention increases the metric."},
                relations=[
                    {"type": "depends_on", "target": question.object_id},
                ],
            )

            self.assertTrue(question.created)
            self.assertFalse(repeated.created)
            self.assertEqual(question.object_id, repeated.object_id)
            self.assertEqual(repository.get(question.object_id)["kind"], "question")
            self.assertEqual(
                [item["kind"] for item in repository.objects()],
                ["hypothesis", "question"],
            )
            self.assertEqual(
                repository.objects(kind="hypothesis")[0]["object_id"],
                hypothesis.object_id,
            )

            checkpoint = repository.commit(
                stage="ideation",
                subject="record question and hypothesis",
            )
            verification = repository.fsck()

            self.assertTrue(checkpoint.committed)
            self.assertTrue(verification["ok"], verification["errors"])
            self.assertIn(
                str(question.path.relative_to(repository.path)),
                checkpoint.staged_paths,
            )

    def test_privacy_gate_rejects_secret_without_persisting_it(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            secret = "sk-" + "q" * 40

            with self.assertRaisesRegex(ResearchGitError, "privacy gate refused") as caught:
                repository.record("evidence", {"credential": secret})

            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(repository.objects(), [])

    def test_damaged_object_is_never_returned_as_valid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            result = repository.record("metric", {"name": "accuracy", "value": 1.0})
            payload = json.loads(result.path.read_text(encoding="utf-8"))
            payload["payload"]["value"] = 0.0
            result.path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "damaged"):
                repository.get(result.object_id)
            with self.assertRaisesRegex(ResearchGitError, "damaged"):
                repository.objects()


if __name__ == "__main__":
    unittest.main()
