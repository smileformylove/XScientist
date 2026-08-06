from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema
from xscientist.research_cli import main as research_main
from xscientist.research_git import (
    ResearchGitError,
    add_research_object,
    create_checkpoint,
    create_research_bundle,
    init_repository,
    repository_status,
    reproduce_checkpoint,
    research_diff,
    research_log,
    show_checkpoint,
)


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class LocalResearchGitTests(unittest.TestCase):
    def _init(self, root: Path, **kwargs):
        return init_repository(
            root,
            question="# Question\n\nDoes H1 improve the metric?\n",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
            **kwargs,
        )

    def _git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()

    def test_init_creates_serverless_repository_and_initial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"

            result = self._init(root)
            status = repository_status(root)

            self.assertTrue(result.committed)
            self.assertTrue((root / ".git").is_dir())
            self.assertEqual(status["branch"], "main")
            self.assertEqual(status["checkpoint_policy"], "milestone")
            self.assertFalse(status["auto_push"])
            self.assertFalse(self._git(root, "remote"), "init must not create a remote")
            checkpoint = show_checkpoint(root)["checkpoint"]
            self.assertEqual(checkpoint["stage"], "init")
            validate(checkpoint, load_schema("research_checkpoint"))

    def test_checkpoint_stages_only_policy_paths_and_skips_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root, max_file_bytes=2048)
            hypothesis = root / "hypotheses" / "h1.json"
            hypothesis.write_text('{"hypothesis":"H1"}\n', encoding="utf-8")
            (root / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
            (root / "hypotheses" / "large.json").write_text(
                "x" * 4096, encoding="utf-8"
            )

            result = create_checkpoint(
                root,
                stage="preregister",
                subject="lock H1",
                summary="Prospective hypothesis.",
            )
            tree = set(
                self._git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
            )

            self.assertTrue(result.committed)
            self.assertIn("hypotheses/h1.json", tree)
            self.assertNotIn(".env", tree)
            self.assertNotIn("hypotheses/large.json", tree)
            self.assertTrue(any("exceeds" in item for item in result.excluded_paths))

    def test_checkpoint_refuses_a_prepopulated_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            path = root / "hypotheses" / "h1.json"
            path.write_text("{}\n", encoding="utf-8")
            self._git(root, "add", "hypotheses/h1.json")

            with self.assertRaisesRegex(ResearchGitError, "already contains staged"):
                create_checkpoint(root, stage="preregister", subject="lock H1")

    def test_checkpoint_skips_when_only_ignored_or_untracked_noise_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            (root / "scratch.txt").write_text("noise\n", encoding="utf-8")

            result = create_checkpoint(root, stage="experiment", subject="no result")

            self.assertFalse(result.created)
            self.assertIn("no material research change", result.reason or "")

    def test_object_pointer_bundle_and_reproduction_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "dataset.bin"
            source.write_bytes(b"evidence" * 1024)

            pointer = add_research_object(
                root,
                source,
                logical_path="data/dataset.bin",
            )
            checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind dataset",
                reproduce_command="python verify.py",
            )
            bundle_path = base / "research.tar.gz"
            bundle = create_research_bundle(root, bundle_path)
            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "reproduction",
            )
            checkpoint_payload = json.loads(
                checkpoint.checkpoint_path.read_text(encoding="utf-8")
            )

            self.assertIn(pointer.object_hash, checkpoint_payload["object_refs"])
            self.assertTrue(bundle["complete"])
            self.assertTrue(reproduction["objects_complete"])
            self.assertEqual(
                (base / "reproduction" / "data" / "dataset.bin").read_bytes(),
                source.read_bytes(),
            )
            with tarfile.open(bundle_path) as archive:
                names = set(archive.getnames())
            self.assertIn("repository.gitbundle", names)
            self.assertIn("bundle.manifest.json", names)
            self.assertIn(
                f"objects/sha256/{pointer.object_hash.split(':', 1)[1]}", names
            )
            validate(
                json.loads(pointer.pointer_path.read_text(encoding="utf-8")),
                load_schema("research_object_pointer"),
            )
            validate(
                {key: value for key, value in bundle.items() if key != "destination"},
                load_schema("research_bundle"),
            )

    def test_log_show_and_diff_expose_scientific_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            first = self._init(root)
            (root / "claims" / "c1.json").write_text(
                '{"claim":"C1"}\n', encoding="utf-8"
            )
            second = create_checkpoint(
                root,
                stage="evidence",
                subject="bind C1",
                claims=["c1"],
            )

            history = research_log(root)
            shown = show_checkpoint(root, second.commit or "HEAD")
            diff = research_diff(
                root, first.commit or "HEAD~1", second.commit or "HEAD"
            )

            self.assertEqual(history[0]["trailers"]["Research-Stage"], ["evidence"])
            self.assertEqual(shown["checkpoint"]["claims"], ["c1"])
            self.assertTrue(any("claims/c1.json" in line for line in diff["changes"]))

    def test_old_commit_resolves_its_own_pointer_set(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"old evidence")
            pointer = add_research_object(
                root, source, logical_path="data/evidence.bin"
            )
            bound = create_checkpoint(root, stage="evidence", subject="bind object")
            pointer.pointer_path.unlink()
            (root / "claims" / "c2.json").write_text("{}\n", encoding="utf-8")
            create_checkpoint(root, stage="review", subject="supersede object")

            reproduction = reproduce_checkpoint(root, commit=bound.commit or "HEAD")

            self.assertTrue(reproduction["objects_complete"])
            self.assertFalse(reproduction["missing_objects"])

    def test_reproduction_executes_without_a_shell_in_detached_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            script = root / ".xscientist" / "verify.py"
            script.write_text("print('verified')\n", encoding="utf-8")
            checkpoint = create_checkpoint(
                root,
                stage="review",
                subject="record verifier",
                reproduce_command="python .xscientist/verify.py",
            )

            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "execute",
                execute=True,
                timeout_seconds=10,
            )

            self.assertEqual(reproduction["returncode"], 0)
            self.assertFalse(reproduction["timed_out"])
            self.assertEqual(reproduction["stdout"].strip(), "verified")

    def test_reproduction_refuses_a_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            checkpoint_path = sorted((root / "checkpoints").glob("*.json"))[-1]
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["summary"] = "tampered"
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            self._git(root, "add", checkpoint_path.relative_to(root).as_posix())
            self._git(root, "commit", "-m", "tamper checkpoint")

            self.assertFalse(show_checkpoint(root)["checkpoint_hash_valid"])
            with self.assertRaisesRegex(ResearchGitError, "hash verification failed"):
                reproduce_checkpoint(root)

    def test_unified_cli_supports_init_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self.assertEqual(
                research_main(
                    [
                        "init",
                        str(root),
                        "--question",
                        "Does H1 hold?",
                        "--git-user-name",
                        "Research Test",
                        "--git-user-email",
                        "research@example.invalid",
                    ]
                ),
                0,
            )
            self.assertEqual(research_main(["status", "--repo", str(root)]), 0)


if __name__ == "__main__":
    unittest.main()
