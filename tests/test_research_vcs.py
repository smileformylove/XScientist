from __future__ import annotations

import json
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

from ai_scientist.protocol.research_vcs import (
    ResearchObjectError,
    build_research_object,
    validate_research_object,
)
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_git import ResearchGitError
from xscientist.research_git import add_research_object


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

            with self.assertRaisesRegex(
                ResearchGitError, "privacy gate refused"
            ) as caught:
                repository.record("evidence", {"credential": secret})

            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(repository.objects(), [])

    def test_native_stage_commits_only_the_selected_research_change(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            question = repository.record("question", {"text": "Q1"})
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            question_path = question.path.relative_to(repository.path).as_posix()
            hypothesis_path = hypothesis.path.relative_to(repository.path).as_posix()

            staged = repository.stage([question_path])
            status = repository.status()
            checkpoint = repository.commit(
                stage="ideation",
                subject="record only the selected question",
                staged_only=True,
            )

            self.assertEqual(staged.paths, (question_path,))
            self.assertEqual(status["research_stage"]["paths"], [question_path])
            self.assertIn(question_path, checkpoint.staged_paths)
            self.assertNotIn(hypothesis_path, checkpoint.staged_paths)
            self.assertEqual(repository.status()["research_stage"]["paths"], [])
            self.assertIn(hypothesis_path, repository.status()["eligible_changes"])

    def test_native_stage_detects_content_changed_after_selection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            result = repository.record("hypothesis", {"statement": "H1"})
            relative = result.path.relative_to(repository.path).as_posix()
            repository.stage([relative])
            result.path.write_text(result.path.read_text(encoding="utf-8") + " ")

            with self.assertRaisesRegex(ResearchGitError, "changed after selection"):
                repository.commit(
                    stage="ideation",
                    subject="must not commit stale selection",
                    staged_only=True,
                )

    def test_native_stage_binds_only_selected_large_object_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            repository = self._init(root)
            first_source = base / "first.bin"
            second_source = base / "second.bin"
            first_source.write_bytes(b"first evidence")
            second_source.write_bytes(b"second evidence")
            first = add_research_object(
                root, first_source, logical_path="data/first.bin"
            )
            second = add_research_object(
                root, second_source, logical_path="data/second.bin"
            )
            first_path = first.pointer_path.relative_to(repository.path).as_posix()
            second_path = second.pointer_path.relative_to(repository.path).as_posix()

            repository.stage([first_path])
            checkpoint = repository.commit(
                stage="evidence",
                subject="bind selected evidence only",
                staged_only=True,
            )
            verification = repository.fsck()

            self.assertIn(
                first.object_hash, repository.show()["checkpoint"]["object_refs"]
            )
            self.assertNotIn(
                second.object_hash, repository.show()["checkpoint"]["object_refs"]
            )
            self.assertIn(first_path, checkpoint.staged_paths)
            self.assertNotIn(second_path, checkpoint.staged_paths)
            self.assertTrue(verification["ok"], verification["errors"])

    def test_research_branches_and_tags_use_scientific_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)

            forked = repository.fork("hypothesis/h1")
            result = repository.record("hypothesis", {"statement": "H1"})
            repository.commit(stage="ideation", subject="explore H1")
            tag = repository.tag("result/h1-v1")
            branches = repository.branches()
            tags = repository.tags()

            self.assertTrue(forked["current"])
            self.assertEqual(repository.status()["branch"], "hypothesis/h1")
            self.assertEqual(
                [item["name"] for item in branches],
                ["hypothesis/h1", "main"],
            )
            self.assertEqual(
                tag["checkpoint_id"], repository.show()["checkpoint"]["checkpoint_id"]
            )
            self.assertEqual(tags[0]["name"], "result/h1-v1")
            self.assertEqual(tags[0]["checkpoint_id"], tag["checkpoint_id"])
            self.assertTrue(result.path.is_file())
            with self.assertRaisesRegex(ResearchGitError, "already exists"):
                repository.tag("result/h1-v1")

    def test_switch_refuses_uncommitted_research(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            repository = self._init(root)
            repository.fork("alternative", switch=False)
            repository.record("hypothesis", {"statement": "uncommitted"})

            with self.assertRaisesRegex(ResearchGitError, "clean working state"):
                repository.switch("alternative")

    def test_cli_runs_native_record_stage_commit_and_branch_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            commands = [
                [
                    "init",
                    str(root),
                    "--question",
                    "Does H1 improve the metric?",
                    "--git-user-name",
                    "Research Test",
                    "--git-user-email",
                    "research@example.invalid",
                ],
                [
                    "record",
                    "hypothesis",
                    "--repo",
                    str(root),
                    "--data",
                    '{"statement":"H1"}',
                ],
                ["stage", "--repo", str(root), "--all"],
                [
                    "checkpoint",
                    "--repo",
                    str(root),
                    "--stage",
                    "ideation",
                    "--subject",
                    "record H1",
                    "--staged",
                ],
                ["branch", "alternative", "--repo", str(root)],
            ]

            for command in commands:
                with self.subTest(command=command), redirect_stdout(io.StringIO()):
                    self.assertEqual(research_main(command), 0)

            repository = ResearchRepository(root)
            self.assertEqual(
                repository.objects(kind="hypothesis")[0]["payload"], {"statement": "H1"}
            )
            self.assertEqual(
                [item["name"] for item in repository.branches()],
                ["alternative", "main"],
            )

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
