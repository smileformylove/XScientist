from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from jsonschema import validate

import xscientist.research_git as research_git_module
from ai_scientist.protocol import content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist.research_cli import main as research_main
from xscientist.research_git import (
    ResearchGitError,
    add_research_object,
    auto_checkpoint,
    create_checkpoint,
    create_research_bundle,
    create_research_tag,
    init_repository,
    repository_status,
    reproduce_checkpoint,
    restore_research_bundle,
    research_diff,
    research_log,
    show_checkpoint,
    verify_research_bundle,
    verify_research_repository,
)
from xscientist.research_interop import export_research_interop


@unittest.skipUnless(shutil.which("git"), "Git is required for research history tests")
class LocalResearchGitTests(unittest.TestCase):
    def test_manifest_delta_reports_added_changed_and_unchanged_paths(self) -> None:
        delta = research_git_module._manifest_delta(
            [
                {"path": "ara/a", "manifest_hash": "sha256:" + "a" * 64},
                {"path": "ara/b", "manifest_hash": "sha256:" + "b" * 64},
                {
                    "path": "ara/graph",
                    "manifest_hash": "sha256:" + "e" * 64,
                    "exploration_graph_hash": "sha256:" + "1" * 64,
                },
            ],
            [
                {"path": "ara/a", "manifest_hash": "sha256:" + "c" * 64},
                {"path": "ara/b", "manifest_hash": "sha256:" + "b" * 64},
                {"path": "ara/c", "manifest_hash": "sha256:" + "d" * 64},
                {
                    "path": "ara/graph",
                    "manifest_hash": "sha256:" + "e" * 64,
                    "exploration_graph_hash": "sha256:" + "2" * 64,
                },
            ],
        )

        self.assertEqual(delta["added"], ["ara/c"])
        self.assertEqual(delta["removed"], [])
        self.assertEqual(delta["unchanged"], ["ara/b"])
        self.assertEqual(delta["changed"][0]["path"], "ara/a")
        graph_change = next(
            item for item in delta["changed"] if item["path"] == "ara/graph"
        )
        self.assertEqual(graph_change["changed_fields"], ["exploration_graph_hash"])

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

    def test_init_preserves_existing_project_files_and_commits_only_managed_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Research Test")
            self._git(root, "config", "user.email", "research@example.invalid")
            (root / ".gitignore").write_text("custom-cache/\n", encoding="utf-8")
            (root / "README.md").write_text("uncommitted project\n", encoding="utf-8")

            self._init(root)

            ignore = (root / ".gitignore").read_text(encoding="utf-8")
            self.assertTrue(ignore.startswith("custom-cache/\n"))
            self.assertIn(".env\n", ignore)
            tree = set(
                self._git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
            )
            self.assertNotIn("README.md", tree)
            self.assertIn("README.md", self._git(root, "status", "--short"))
            checkpoint = show_checkpoint(root)["checkpoint"]
            self.assertEqual(
                set(checkpoint["changed_paths"]),
                {
                    ".gitignore",
                    ".xscientist/README.md",
                    "question.md",
                    "research.yaml",
                },
            )

    def test_init_refuses_to_overwrite_existing_question_without_partial_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            question = root / "question.md"
            question.write_text("user-owned question\n", encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "overwrite.*question"):
                self._init(root)

            self.assertEqual(
                question.read_text(encoding="utf-8"), "user-owned question\n"
            )
            self.assertFalse((root / "research.yaml").exists())
            self.assertFalse((root / ".git").exists())

    def test_init_privacy_failure_removes_only_new_repository_state(self) -> None:
        private_question = "# Question\n\nsk-" + "A" * 40 + "\n"
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            fresh = base / "fresh"

            with self.assertRaisesRegex(ResearchGitError, "privacy gate"):
                init_repository(fresh, question=private_question)

            self.assertFalse(fresh.exists())

            existing = base / "existing"
            hypothesis = existing / "hypotheses" / "user-note.md"
            hypothesis.parent.mkdir(parents=True)
            hypothesis.write_bytes(b"user-owned bytes\r\n")
            hypothesis.chmod(0o640)
            before = (hypothesis.read_bytes(), hypothesis.stat().st_mode & 0o7777)

            with self.assertRaisesRegex(ResearchGitError, "privacy gate"):
                init_repository(existing, question=private_question)

            self.assertEqual(
                (hypothesis.read_bytes(), hypothesis.stat().st_mode & 0o7777),
                before,
            )
            self.assertEqual(
                sorted(
                    path.relative_to(existing).as_posix()
                    for path in existing.rglob("*")
                ),
                ["hypotheses", "hypotheses/user-note.md"],
            )

            tracked = base / "tracked"
            tracked.mkdir()
            self._git(tracked, "init", "-b", "main")
            self._git(tracked, "config", "user.name", "Original Researcher")
            self._git(tracked, "config", "user.email", "original@example.invalid")
            (tracked / "baseline.txt").write_text("baseline\n", encoding="utf-8")
            self._git(tracked, "add", "baseline.txt")
            self._git(tracked, "commit", "-m", "baseline")
            head_before = self._git(tracked, "rev-parse", "HEAD")

            with self.assertRaisesRegex(ResearchGitError, "privacy gate"):
                init_repository(
                    tracked,
                    question=private_question,
                    git_user_name="Changed Researcher",
                    git_user_email="changed@example.invalid",
                )

            self.assertEqual(self._git(tracked, "rev-parse", "HEAD"), head_before)
            self.assertEqual(self._git(tracked, "status", "--porcelain"), "")
            self.assertEqual(
                self._git(tracked, "config", "--local", "user.name"),
                "Original Researcher",
            )
            self.assertEqual(
                self._git(tracked, "config", "--local", "user.email"),
                "original@example.invalid",
            )
            self.assertFalse((tracked / "research.yaml").exists())

    def test_init_refuses_to_populate_an_unknown_git_control_directory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            sentinel = root / ".git" / "sentinel"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_bytes(b"user-owned git control bytes\n")

            with self.assertRaisesRegex(ResearchGitError, "pre-existing .git"):
                self._init(root)

            self.assertEqual(sentinel.read_bytes(), b"user-owned git control bytes\n")
            self.assertEqual(
                sorted(path.relative_to(root).as_posix() for path in root.rglob("*")),
                [".git", ".git/sentinel"],
            )

    def test_init_and_checkpoint_reject_private_free_text_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            private_actor_root = base / "private-actor"
            with self.assertRaisesRegex(ResearchGitError, "privacy gate"):
                init_repository(
                    private_actor_root,
                    question="# Question\n\nDoes H1 improve the metric?\n",
                    actor="private.person@example.com",
                )
            self.assertFalse(private_actor_root.exists())

            root = base / "research"
            self._init(root)
            draft = root / "hypotheses" / "draft.md"
            draft.write_text("candidate hypothesis\n", encoding="utf-8")
            head_before = self._git(root, "rev-parse", "HEAD")
            checkpoints_before = {
                path.name for path in (root / "checkpoints").iterdir()
            }
            cases = (
                {
                    "subject": "analyze /opt/private/data.csv",
                    "summary": "portable summary",
                },
                {
                    "subject": "analyze candidate",
                    "summary": "contact other.person@example.com",
                },
                {
                    "subject": "analyze candidate",
                    "summary": "portable summary",
                    "actor": "private.person@example.com",
                },
            )
            for kwargs in cases:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ResearchGitError, "privacy gate"):
                        create_checkpoint(root, stage="ideation", **kwargs)
                    self.assertEqual(self._git(root, "rev-parse", "HEAD"), head_before)
                    self.assertEqual(
                        {path.name for path in (root / "checkpoints").iterdir()},
                        checkpoints_before,
                    )
                    self.assertTrue(draft.is_file())

    def test_init_rollback_preserves_concurrent_scientific_writes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            concurrent_result = root / "hypotheses" / "concurrent-result.json"
            concurrent_question = b"# Concurrent question\n\nKeep this newer text.\n"

            def fail_after_concurrent_write(*_args, **_kwargs):
                concurrent_result.parent.mkdir(parents=True, exist_ok=True)
                concurrent_result.write_bytes(b'{"owner":"another-agent"}\n')
                (root / "question.md").write_bytes(concurrent_question)
                raise OSError("late checkpoint failure")

            with mock.patch.object(
                research_git_module,
                "create_checkpoint",
                side_effect=fail_after_concurrent_write,
            ):
                with self.assertRaisesRegex(OSError, "late checkpoint failure"):
                    self._init(root)

            self.assertEqual(
                concurrent_result.read_bytes(), b'{"owner":"another-agent"}\n'
            )
            self.assertEqual((root / "question.md").read_bytes(), concurrent_question)
            self.assertFalse((root / "research.yaml").exists())
            self.assertFalse((root / ".xscientist" / "README.md").exists())

    def test_init_rollback_preserves_concurrent_git_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()

            def fail_after_concurrent_config(*_args, **_kwargs):
                self._git(root, "config", "concurrent.marker", "IRREPLACEABLE")
                raise OSError("late checkpoint failure")

            with mock.patch.object(
                research_git_module,
                "create_checkpoint",
                side_effect=fail_after_concurrent_config,
            ):
                with self.assertRaisesRegex(OSError, "late checkpoint failure"):
                    self._init(root)

            self.assertTrue((root / ".git").is_dir())
            self.assertEqual(
                self._git(root, "config", "--local", "concurrent.marker"),
                "IRREPLACEABLE",
            )
            self.assertFalse((root / "research.yaml").exists())

    def test_init_rollback_does_not_overwrite_concurrent_git_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Original Researcher")
            self._git(root, "config", "user.email", "original@example.invalid")
            (root / "baseline.txt").write_text("baseline\n", encoding="utf-8")
            self._git(root, "add", "baseline.txt")
            self._git(root, "commit", "-m", "baseline")

            def fail_after_concurrent_identity(*_args, **_kwargs):
                self._git(root, "config", "user.name", "Concurrent Researcher")
                raise OSError("late checkpoint failure")

            with mock.patch.object(
                research_git_module,
                "create_checkpoint",
                side_effect=fail_after_concurrent_identity,
            ):
                with self.assertRaisesRegex(OSError, "late checkpoint failure"):
                    self._init(root)

            self.assertEqual(
                self._git(root, "config", "--local", "user.name"),
                "Concurrent Researcher",
            )
            self.assertEqual(
                self._git(root, "config", "--local", "user.email"),
                "original@example.invalid",
            )
            self.assertFalse((root / "research.yaml").exists())

    def test_init_refuses_to_overwrite_managed_files_changed_after_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)

            changed_root = base / "changed"
            changed_root.mkdir()
            gitignore = changed_root / ".gitignore"
            gitignore.write_bytes(b"original-rule\n")
            original_merge = research_git_module._merged_research_gitignore

            def change_existing(value: str) -> str:
                gitignore.write_bytes(b"concurrent-rule\n")
                return original_merge(value)

            with mock.patch.object(
                research_git_module,
                "_merged_research_gitignore",
                side_effect=change_existing,
            ):
                with self.assertRaisesRegex(ResearchGitError, "changed concurrently"):
                    self._init(changed_root)
            self.assertEqual(gitignore.read_bytes(), b"concurrent-rule\n")
            self.assertFalse((changed_root / "research.yaml").exists())

            appeared_root = base / "appeared"
            appeared_root.mkdir()
            config_path = appeared_root / "research.yaml"
            original_config_text = research_git_module._config_text

            def create_concurrent_config(**kwargs):
                config_path.write_bytes(b"concurrent: true\n")
                return original_config_text(**kwargs)

            with mock.patch.object(
                research_git_module,
                "_config_text",
                side_effect=create_concurrent_config,
            ):
                with self.assertRaisesRegex(ResearchGitError, "appeared concurrently"):
                    self._init(appeared_root)
            self.assertEqual(config_path.read_bytes(), b"concurrent: true\n")

            unchanged_root = base / "unchanged"
            unchanged_root.mkdir()
            unchanged_question = unchanged_root / "question.md"
            unchanged_question.write_text(
                "# Question\n\nDoes H1 improve the metric?\n",
                encoding="utf-8",
            )
            original_changed_paths = research_git_module._changed_paths
            changed_path_calls = 0

            def change_unwritten_managed_file(repo: Path):
                nonlocal changed_path_calls
                changed_path_calls += 1
                if changed_path_calls == 2:
                    unchanged_question.write_bytes(b"concurrent question\n")
                return original_changed_paths(repo)

            with mock.patch.object(
                research_git_module,
                "_changed_paths",
                side_effect=change_unwritten_managed_file,
            ):
                with self.assertRaisesRegex(ResearchGitError, "changed concurrently"):
                    self._init(unchanged_root)
            self.assertEqual(unchanged_question.read_bytes(), b"concurrent question\n")

    def test_init_refuses_preexisting_git_stage_without_writing_config(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Research Test")
            self._git(root, "config", "user.email", "research@example.invalid")
            (root / "README.md").write_text("staged work\n", encoding="utf-8")
            self._git(root, "add", "README.md")

            with self.assertRaisesRegex(ResearchGitError, "staged work"):
                self._init(root)

            self.assertEqual(
                self._git(root, "diff", "--cached", "--name-only"), "README.md"
            )
            self.assertFalse((root / "research.yaml").exists())

    def test_init_refuses_intent_to_add_without_changing_the_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            pending = root / "pending.txt"
            pending.write_text("user work\n", encoding="utf-8")
            self._git(root, "add", "-N", "pending.txt")
            before = self._git(root, "status", "--porcelain=v1")

            with self.assertRaisesRegex(ResearchGitError, "staged work"):
                self._init(root)

            self.assertEqual(self._git(root, "status", "--porcelain=v1"), before)
            self.assertFalse((root / "research.yaml").exists())

    def test_init_preflight_failure_preserves_local_git_identity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Original Researcher")
            self._git(root, "config", "user.email", "original@example.invalid")
            (root / "README.md").write_text("staged work\n", encoding="utf-8")
            self._git(root, "add", "README.md")

            with self.assertRaisesRegex(ResearchGitError, "staged work"):
                init_repository(
                    root,
                    question="# Question\n\nDoes H1 improve the metric?\n",
                    git_user_name="Changed Researcher",
                    git_user_email="changed@example.invalid",
                )

            self.assertEqual(
                self._git(root, "config", "--local", "user.name"),
                "Original Researcher",
            )
            self.assertEqual(
                self._git(root, "config", "--local", "user.email"),
                "original@example.invalid",
            )
            self.assertEqual(
                self._git(root, "diff", "--cached", "--name-only"), "README.md"
            )
            self.assertFalse((root / "research.yaml").exists())

    def test_init_refuses_to_absorb_tracked_managed_file_edits(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "existing"
            root.mkdir()
            self._git(root, "init", "-b", "main")
            self._git(root, "config", "user.name", "Research Test")
            self._git(root, "config", "user.email", "research@example.invalid")
            ignore = root / ".gitignore"
            ignore.write_text("baseline/\n", encoding="utf-8")
            self._git(root, "add", ".gitignore")
            self._git(root, "commit", "-m", "project baseline")
            ignore.write_text("baseline/\nuser-edit/\n", encoding="utf-8")

            with self.assertRaisesRegex(ResearchGitError, "managed files"):
                self._init(root)

            self.assertEqual(
                ignore.read_text(encoding="utf-8"), "baseline/\nuser-edit/\n"
            )
            self.assertFalse((root / "research.yaml").exists())

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

    def test_checkpoint_privacy_gate_rejects_secret_content_without_echoing_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            token = "sk-" + "q" * 32
            unsafe = root / "hypotheses" / "unsafe.json"
            unsafe.write_text(json.dumps({"credential": token}), encoding="utf-8")

            with self.assertRaises(ResearchGitError) as caught:
                create_checkpoint(root, stage="preregister", subject="unsafe")

            message = str(caught.exception)
            self.assertIn("privacy gate refused", message)
            self.assertIn("hypotheses/unsafe.json", message)
            self.assertNotIn(token, message)
            self.assertFalse(self._git(root, "diff", "--cached", "--name-only"))

    def test_checkpoint_refuses_a_prepopulated_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            path = root / "hypotheses" / "h1.json"
            path.write_text("{}\n", encoding="utf-8")
            self._git(root, "add", "hypotheses/h1.json")

            with self.assertRaisesRegex(ResearchGitError, "already contains staged"):
                create_checkpoint(root, stage="preregister", subject="lock H1")

    def test_checkpoint_commit_failure_restores_index_and_own_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            existing_checkpoints = set((root / "checkpoints").iterdir())
            hypothesis = root / "hypotheses" / "h1.json"
            hypothesis.write_text('{"hypothesis":"H1"}\n', encoding="utf-8")
            real_run_git = research_git_module._run_git

            def fail_commit(repo, args, *, check=True):
                if args and args[0] == "commit":
                    raise ResearchGitError("injected commit failure")
                return real_run_git(repo, args, check=check)

            with mock.patch.object(
                research_git_module,
                "_run_git",
                side_effect=fail_commit,
            ):
                with self.assertRaisesRegex(ResearchGitError, "injected"):
                    create_checkpoint(root, stage="preregister", subject="lock H1")

            self.assertEqual(
                set((root / "checkpoints").iterdir()), existing_checkpoints
            )
            self.assertTrue(hypothesis.is_file())
            self.assertFalse(self._git(root, "diff", "--cached", "--name-only"))
            self.assertIn("hypotheses/", self._git(root, "status", "--porcelain"))

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

    def test_default_milestone_policy_checkpoints_ideation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            hypothesis = root / "hypotheses" / "h1.json"
            hypothesis.write_text('{"hypothesis":"H1"}\n', encoding="utf-8")

            result = auto_checkpoint(
                root,
                stage="ideation",
                subject="record candidate hypothesis",
            )

            self.assertTrue(result.committed)
            self.assertEqual(show_checkpoint(root)["checkpoint"]["stage"], "ideation")

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
            bundle_verification = verify_research_bundle(bundle_path)
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
            self.assertTrue(bundle_verification["ok"], bundle_verification["errors"])
            self.assertTrue(reproduction["objects_complete"])
            validate(reproduction["receipt"], load_schema("reproduction_receipt"))
            self.assertEqual(
                reproduction["receipt"]["schema_version"],
                "xscientist.reproduction-receipt.v2",
            )
            self.assertEqual(
                reproduction["receipt"]["checkpoint_binding"]["commit"],
                checkpoint.commit,
            )
            self.assertEqual(
                reproduction["receipt"]["execution_result"]["result_hash"],
                content_hash(
                    {
                        key: reproduction["receipt"][key]
                        for key in (
                            "command_hash",
                            "reproduction_level",
                            "verdict",
                            "objects_complete",
                            "executed",
                            "returncode",
                            "timed_out",
                            "stdout_hash",
                            "stderr_hash",
                            "stdout_truncated",
                            "stderr_truncated",
                            "output_capture",
                            "max_output_chars",
                        )
                    }
                ),
            )
            self.assertEqual(
                reproduction["receipt"]["reproduction_level"], "artifact_replay"
            )
            self.assertTrue(
                (
                    Path(reproduction["worktree"]) / reproduction["receipt_path"]
                ).is_file()
            )
            persisted_receipt = json.loads(
                (
                    Path(reproduction["worktree"]) / reproduction["receipt_path"]
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted_receipt["execution_isolation"],
                reproduction["execution_isolation"],
            )
            self.assertEqual(persisted_receipt, reproduction["receipt"])
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

    def test_bundle_restore_round_trip_is_clean_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"portable evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind portable evidence",
            )
            bundle_path = base / "research.tar.gz"
            create_research_bundle(root, bundle_path)

            restored = restore_research_bundle(bundle_path, base / "restored")
            inspection = reproduce_checkpoint(
                base / "restored",
                commit=checkpoint.commit or "HEAD",
                destination=base / "restored-run",
            )

            self.assertEqual(restored["commit"], checkpoint.commit)
            self.assertTrue(restored["fsck"]["ok"])
            self.assertEqual(
                restored["fsck"]["repository"],
                str((base / "restored").resolve()),
            )
            self.assertEqual(restored["objects_restored"], 1)
            self.assertFalse(self._git(base / "restored", "remote"))
            self.assertFalse(self._git(base / "restored", "status", "--porcelain"))
            self.assertEqual(
                (Path(inspection["worktree"]) / "data" / "evidence.bin").read_bytes(),
                b"portable evidence",
            )
            restored_object = (
                base
                / "restored"
                / ".ara-store"
                / "objects"
                / "sha256"
                / pointer.object_hash.split(":", 1)[1]
            )
            self.assertEqual(restored_object.read_bytes(), b"portable evidence")

    def test_bundle_restores_cas_for_history_noncurrent_branches_and_tags(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)

            historical_source = base / "historical.bin"
            historical_source.write_bytes(b"historical evidence")
            historical_pointer = add_research_object(
                root,
                historical_source,
                logical_path="data/historical.bin",
            )
            historical_checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind historical evidence",
            )
            create_research_tag(
                root,
                "evidence-v1",
                commit=historical_checkpoint.commit or "HEAD",
            )

            historical_pointer.pointer_path.unlink()
            create_checkpoint(
                root,
                stage="cleanup",
                subject="retire pointer from current main",
            )
            self._git(root, "switch", "-c", "alternative")
            branch_source = base / "branch.bin"
            branch_source.write_bytes(b"alternative evidence")
            branch_pointer = add_research_object(
                root,
                branch_source,
                logical_path="data/branch.bin",
            )
            branch_checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind alternative evidence",
            )
            self._git(root, "switch", "main")

            bundle_path = base / "all-history.tar.gz"
            bundle = create_research_bundle(root, bundle_path)
            verification = verify_research_bundle(bundle_path)
            restored_root = base / "restored"
            restore_research_bundle(bundle_path, restored_root)

            self.assertEqual(
                bundle["closure"]["required_objects"],
                sorted([historical_pointer.object_hash, branch_pointer.object_hash]),
            )
            self.assertTrue(verification["ok"], verification["errors"])
            self.assertEqual(
                self._git(restored_root, "rev-parse", "refs/heads/alternative"),
                branch_checkpoint.commit,
            )
            self.assertEqual(
                self._git(
                    restored_root,
                    "rev-parse",
                    "refs/tags/evidence-v1^{commit}",
                ),
                historical_checkpoint.commit,
            )
            historical_run = reproduce_checkpoint(
                restored_root,
                commit="refs/tags/evidence-v1",
                destination=base / "historical-run",
            )
            branch_run = reproduce_checkpoint(
                restored_root,
                commit="refs/heads/alternative",
                destination=base / "branch-run",
            )
            self.assertEqual(
                (Path(historical_run["worktree"]) / "data/historical.bin").read_bytes(),
                b"historical evidence",
            )
            self.assertEqual(
                (Path(branch_run["worktree"]) / "data/branch.bin").read_bytes(),
                b"alternative evidence",
            )

    def test_bundle_missing_list_covers_historical_cas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "historical.bin"
            source.write_bytes(b"historical evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/historical.bin",
            )
            create_checkpoint(root, stage="evidence", subject="bind historical data")
            pointer.pointer_path.unlink()
            create_checkpoint(root, stage="cleanup", subject="retire current pointer")
            pointer.store_path.unlink()

            refused = base / "refused.tar.gz"
            with self.assertRaisesRegex(
                ResearchGitError,
                pointer.object_hash,
            ):
                create_research_bundle(root, refused)
            self.assertFalse(refused.exists())

            incomplete_path = base / "incomplete.tar.gz"
            incomplete = create_research_bundle(
                root,
                incomplete_path,
                allow_incomplete=True,
            )
            verification = verify_research_bundle(incomplete_path)
            self.assertFalse(incomplete["complete"])
            self.assertEqual(incomplete["missing_objects"], [pointer.object_hash])
            self.assertTrue(verification["ok"], verification["errors"])

    def test_bundle_verifier_recomputes_closure_from_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"closure evidence")
            add_research_object(root, source, logical_path="data/evidence.bin")
            create_checkpoint(root, stage="evidence", subject="bind closure")
            original = base / "original.tar.gz"
            forged = base / "forged.tar.gz"
            create_research_bundle(root, original)

            with tarfile.open(original, "r:gz") as source_archive:
                payloads = {
                    member.name: source_archive.extractfile(member).read()
                    for member in source_archive.getmembers()
                    if member.isfile()
                }
            manifest = json.loads(payloads["bundle.manifest.json"])
            manifest["closure"]["pointers"] = []
            manifest["closure"]["required_objects"] = []
            manifest["entries"] = [
                entry
                for entry in manifest["entries"]
                if entry["path"] == "repository.gitbundle"
            ]
            manifest["missing_objects"] = []
            manifest["complete"] = True
            manifest.pop("content_hash")
            manifest["content_hash"] = research_git_module.content_hash(manifest)
            payloads = {
                "repository.gitbundle": payloads["repository.gitbundle"],
                "bundle.manifest.json": (
                    json.dumps(manifest, sort_keys=True).encode("utf-8")
                ),
            }
            with tarfile.open(forged, "w:gz") as target_archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target_archive.addfile(info, io.BytesIO(payload))

            verification = verify_research_bundle(forged)

            self.assertFalse(verification["ok"])
            self.assertTrue(
                any("reachable history" in error for error in verification["errors"]),
                verification["errors"],
            )

    def test_legacy_bundle_without_closure_still_verifies_and_restores(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"legacy evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(
                root,
                stage="evidence",
                subject="bind legacy evidence",
            )
            modern = base / "modern.tar.gz"
            legacy = base / "legacy.tar.gz"
            create_research_bundle(root, modern)

            with tarfile.open(modern, "r:gz") as source_archive:
                payloads = {
                    member.name: source_archive.extractfile(member).read()
                    for member in source_archive.getmembers()
                    if member.isfile()
                }
            manifest = json.loads(payloads["bundle.manifest.json"])
            manifest.pop("closure")
            manifest["entries"] = [
                entry
                for entry in manifest["entries"]
                if not entry["path"].startswith("pointer-closure/")
            ]
            pointer_name = f"research-objects/{pointer.pointer_path.name}"
            pointer_payload = pointer.pointer_path.read_bytes()
            payloads[pointer_name] = pointer_payload
            manifest["entries"].append(
                {
                    "path": pointer_name,
                    "hash": research_git_module._hash_file(pointer.pointer_path),
                    "size": len(pointer_payload),
                }
            )
            for name in list(payloads):
                if name.startswith("pointer-closure/"):
                    payloads.pop(name)
            manifest.pop("content_hash")
            manifest["content_hash"] = research_git_module.content_hash(manifest)
            payloads["bundle.manifest.json"] = json.dumps(
                manifest,
                sort_keys=True,
            ).encode("utf-8")
            with tarfile.open(legacy, "w:gz") as target_archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target_archive.addfile(info, io.BytesIO(payload))

            verification = verify_research_bundle(legacy)
            restored_root = base / "legacy-restored"
            restored = restore_research_bundle(legacy, restored_root)
            reproduction = reproduce_checkpoint(
                restored_root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "legacy-run",
            )

            self.assertTrue(verification["ok"], verification["errors"])
            self.assertTrue(restored["fsck"]["ok"])
            self.assertEqual(
                (Path(reproduction["worktree"]) / "data/evidence.bin").read_bytes(),
                b"legacy evidence",
            )

    def test_bundle_ignores_only_untracked_generated_views(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            generated = root / "research-dag" / "research-dag.html"
            generated.parent.mkdir(parents=True)
            generated.write_text("<html>regenerable</html>\n", encoding="utf-8")

            bundle = create_research_bundle(root, base / "clean-view.tar.gz")
            self.assertTrue(bundle["complete"])

            eligible = root / "04_logs" / "progress.json"
            eligible.parent.mkdir(parents=True)
            eligible.write_text('{"current_stage":"running"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ResearchGitError, "research-eligible changes"):
                create_research_bundle(root, base / "dirty.tar.gz")

    def test_bundle_verification_rejects_tampered_member(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            create_checkpoint(root, stage="evidence", subject="bind data")
            original = base / "research.tar.gz"
            tampered = base / "tampered.tar.gz"
            create_research_bundle(root, original)

            with tarfile.open(original, "r:gz") as source_archive:
                payloads = {
                    member.name: source_archive.extractfile(member).read()
                    for member in source_archive.getmembers()
                    if member.isfile()
                }
            object_name = f"objects/sha256/{pointer.object_hash.split(':', 1)[1]}"
            payloads[object_name] = b"tampered evidence"
            with tarfile.open(tampered, "w:gz") as target_archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    target_archive.addfile(info, io.BytesIO(payload))

            verification = verify_research_bundle(tampered)

            self.assertFalse(verification["ok"])
            self.assertTrue(
                any("hash mismatch" in error for error in verification["errors"]),
                verification["errors"],
            )
            with self.assertRaisesRegex(ResearchGitError, "verification failed"):
                restore_research_bundle(tampered, base / "restored")

    def test_cas_snapshot_is_independent_from_mutable_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"immutable evidence")

            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            source.write_bytes(b"changed after registration")

            self.assertFalse(pointer.linked)
            self.assertEqual(pointer.store_path.read_bytes(), b"immutable evidence")

    def test_configured_storage_paths_cannot_escape_repository(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            config_path = root / "research.yaml"
            config = config_path.read_text(encoding="utf-8")

            config_path.write_text(
                config.replace("root: .ara-store", "root: ../outside-cas"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ResearchGitError, "CAS root escapes"):
                repository_status(root)

            config_path.write_text(
                config.replace(
                    "pointer_directory: research-objects",
                    "pointer_directory: ../outside-pointers",
                ),
                encoding="utf-8",
            )
            source = base / "evidence.bin"
            source.write_bytes(b"contained evidence")
            with self.assertRaisesRegex(
                ResearchGitError,
                "pointer directory escapes",
            ):
                add_research_object(root, source)

            self.assertFalse((base / "outside-cas").exists())
            self.assertFalse((base / "outside-pointers").exists())

    def test_reproduction_rejects_tampered_cas_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(root, stage="evidence", subject="bind data")
            pointer.store_path.write_bytes(b"tampered evidence")

            inspection = reproduce_checkpoint(root, commit=checkpoint.commit or "HEAD")

            self.assertFalse(inspection["objects_complete"])
            self.assertEqual(inspection["damaged_objects"], [pointer.object_hash])
            with self.assertRaisesRegex(ResearchGitError, "missing or damaged"):
                reproduce_checkpoint(
                    root,
                    commit=checkpoint.commit or "HEAD",
                    destination=base / "reproduction",
                )

    def test_fsck_detects_pointer_and_object_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            create_checkpoint(root, stage="evidence", subject="bind data")

            clean = verify_research_repository(root)
            pointer.store_path.write_bytes(b"damaged")
            damaged = verify_research_repository(root)

            self.assertTrue(clean["ok"])
            self.assertFalse(damaged["ok"])
            self.assertTrue(
                any("CAS object" in error for error in damaged["errors"]),
                damaged["errors"],
            )

    def test_reproduction_refuses_tampered_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            source = base / "evidence.bin"
            source.write_bytes(b"verified evidence")
            pointer = add_research_object(
                root,
                source,
                logical_path="data/evidence.bin",
            )
            checkpoint = create_checkpoint(root, stage="evidence", subject="bind data")
            payload = json.loads(pointer.pointer_path.read_text(encoding="utf-8"))
            payload["size"] += 1
            pointer.pointer_path.write_text(json.dumps(payload), encoding="utf-8")
            self._git(root, "add", f"research-objects/{pointer.pointer_path.name}")
            self._git(root, "commit", "-m", "tamper pointer")

            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                reproduce_checkpoint(root, commit="HEAD")
            self.assertTrue(checkpoint.commit)

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
                root,
                first.commit or "HEAD~1",
                second.commit or "HEAD",
                deep=True,
            )

            self.assertEqual(history[0]["trailers"]["Research-Stage"], ["evidence"])
            self.assertEqual(shown["checkpoint"]["claims"], ["c1"])
            self.assertTrue(any("claims/c1.json" in line for line in diff["changes"]))
            self.assertEqual(diff["semantic"]["claims"]["added"], ["c1"])
            self.assertTrue(
                any(
                    change["file"] == "claims/c1.json"
                    for change in diff["semantic"]["structured_changes"]
                )
            )

    def test_annotated_tag_is_peeled_in_show_reproduction_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            checkpoint = self._init(root)
            create_research_tag(root, "result-v1", commit=checkpoint.commit or "HEAD")
            tag_object = self._git(root, "rev-parse", "result-v1")
            peeled_commit = self._git(root, "rev-parse", "result-v1^{}")

            shown = show_checkpoint(root, "result-v1")
            reproduction = reproduce_checkpoint(root, commit="result-v1")

            self.assertNotEqual(tag_object, peeled_commit)
            self.assertEqual(shown["commit"], peeled_commit)
            self.assertEqual(reproduction["commit"], peeled_commit)
            self.assertEqual(reproduction["receipt"]["commit"], peeled_commit)

    def test_checkpoint_sensitive_entries_reject_an_unbound_raw_commit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            checkpoint = self._init(root)
            (root / "question.md").write_text(
                "# Question\n\nA raw, uncheckpointed revision.\n",
                encoding="utf-8",
            )
            self._git(root, "add", "question.md")
            self._git(root, "commit", "-m", "raw research change")

            # Selecting the actual checkpoint remains valid; only the raw
            # descendant must stop inheriting it from its first-parent history.
            self.assertEqual(
                show_checkpoint(root, checkpoint.commit or "HEAD~1")["checkpoint"][
                    "checkpoint_id"
                ],
                checkpoint.checkpoint_id,
            )
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                show_checkpoint(root)
            verification = verify_research_repository(root)
            self.assertFalse(verification["ok"])
            self.assertTrue(
                any("not bound" in error for error in verification["errors"]),
                verification["errors"],
            )
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                reproduce_checkpoint(root)
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                create_research_tag(root, "raw-head")

            bundle_path = base / "raw-head.tar.gz"
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                create_research_bundle(root, bundle_path)
            self.assertFalse(bundle_path.exists())

            export_path = base / "raw-export"
            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                export_research_interop(root, export_path)
            self.assertFalse(export_path.exists())

    def test_copied_checkpoint_trailers_do_not_bind_a_raw_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            checkpoint = self._init(root)
            payload = show_checkpoint(root)["checkpoint"]
            (root / "question.md").write_text(
                "# Question\n\nCopying trailers must not create a checkpoint.\n",
                encoding="utf-8",
            )
            self._git(root, "add", "question.md")
            self._git(
                root,
                "commit",
                "-m",
                "raw change with copied checkpoint trailers",
                "-m",
                "\n".join(
                    [
                        f"Research-Checkpoint: {checkpoint.checkpoint_id}",
                        f"Research-Stage: {payload['stage']}",
                        f"Research-State: {payload['status']}",
                        f"Research-Event: {checkpoint.content_hash}",
                    ]
                ),
            )

            with self.assertRaisesRegex(ResearchGitError, "parent_commit"):
                show_checkpoint(root)

    def test_environment_receipt_supports_strict_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            (root / "requirements.txt").write_text(
                "jsonschema==4.23.0\n",
                encoding="utf-8",
            )
            checkpoint = create_checkpoint(
                root,
                stage="preregister",
                subject="lock runtime",
            )
            payload = json.loads(checkpoint.checkpoint_path.read_text(encoding="utf-8"))
            receipt = payload["reproduce"]["environment"]

            validate(receipt, load_schema("research_environment"))
            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "strict-reproduction",
                environment_policy="strict",
            )

            self.assertTrue(reproduction["environment"]["matches"])
            self.assertEqual(
                receipt["dependency_locks"][0]["path"],
                "requirements.txt",
            )

    def test_environment_policy_reports_runtime_drift_and_strictly_refuses_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)

            with mock.patch.object(
                research_git_module.platform,
                "python_version",
                return_value="0.0.0",
            ):
                warning = reproduce_checkpoint(root, environment_policy="warn")
                with self.assertRaisesRegex(ResearchGitError, "environment mismatch"):
                    reproduce_checkpoint(root, environment_policy="strict")

            self.assertFalse(warning["environment"]["matches"])
            self.assertEqual(
                warning["environment"]["mismatches"][0]["field"],
                "python.version",
            )

    def test_strict_reproduction_refuses_uncheckpointed_dependency_lock_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            lock = root / "requirements.txt"
            lock.write_text("jsonschema==4.23.0\n", encoding="utf-8")
            create_checkpoint(root, stage="preregister", subject="lock dependencies")
            lock.write_text("jsonschema==4.24.0\n", encoding="utf-8")
            self._git(root, "add", "requirements.txt")
            self._git(root, "commit", "-m", "mutate dependency lock without checkpoint")
            destination = base / "strict-reproduction"

            with self.assertRaisesRegex(ResearchGitError, "not bound"):
                reproduce_checkpoint(
                    root,
                    destination=destination,
                    environment_policy="strict",
                )

            self.assertFalse(destination.exists())

    def test_divergent_branches_converge_with_multiple_scientific_parents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            self._git(root, "switch", "-c", "hypothesis/a")
            (root / "claims" / "a.json").write_text("{}\n", encoding="utf-8")
            branch_a = create_checkpoint(
                root,
                stage="experiment",
                subject="test branch A",
            )
            self._git(root, "switch", "main")
            (root / "claims").mkdir(exist_ok=True)
            (root / "claims" / "b.json").write_text("{}\n", encoding="utf-8")
            branch_b = create_checkpoint(
                root,
                stage="experiment",
                subject="test branch B",
            )
            self._git(
                root,
                "merge",
                "--no-ff",
                "hypothesis/a",
                "-m",
                "merge scientific branches",
            )
            (root / "manuscript" / "merge.md").write_text(
                "# Converged evidence\n",
                encoding="utf-8",
            )

            converged = create_checkpoint(
                root,
                stage="review",
                subject="review converged evidence",
            )
            payload = json.loads(converged.checkpoint_path.read_text(encoding="utf-8"))
            fsck = verify_research_repository(root)

            self.assertEqual(payload["sequence"], 3)
            self.assertEqual(
                set(payload["parent_checkpoint_hashes"]),
                {branch_a.content_hash, branch_b.content_hash},
            )
            self.assertEqual(
                payload["previous_checkpoint_hash"],
                payload["parent_checkpoint_hashes"][0],
            )
            self.assertTrue(fsck["ok"], fsck["errors"])

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

    def test_reproduction_execution_does_not_inherit_host_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            script = root / ".xscientist" / "inspect_env.py"
            script.write_text(
                "import os\n"
                "print(os.environ.get('XSCIENTIST_TEST_SENTINEL', 'absent'))\n",
                encoding="utf-8",
            )
            checkpoint = create_checkpoint(
                root,
                stage="review",
                subject="record environment probe",
                reproduce_command="python .xscientist/inspect_env.py",
            )

            with mock.patch.dict(
                "os.environ",
                {"XSCIENTIST_TEST_SENTINEL": "sensitive-marker"},
                clear=False,
            ):
                reproduction = reproduce_checkpoint(
                    root,
                    commit=checkpoint.commit or "HEAD",
                    destination=base / "sanitized-execute",
                    execute=True,
                    timeout_seconds=10,
                )

            self.assertEqual(reproduction["stdout"].strip(), "absent")
            self.assertNotIn("sensitive-marker", reproduction["stdout"])
            self.assertEqual(
                reproduction["execution_isolation"]["environment"], "sanitized"
            )
            self.assertEqual(
                reproduction["execution_isolation"]["environment_scope"],
                "variables_only",
            )
            self.assertFalse(reproduction["execution_isolation"]["isolated"])
            self.assertFalse(
                reproduction["execution_isolation"][
                    "process_tree_termination_guaranteed"
                ]
            )
            self.assertEqual(
                reproduction["execution_isolation"]["filesystem"], "host_visible"
            )
            self.assertEqual(
                reproduction["execution_isolation"]["network"], "host_unrestricted"
            )
            self.assertEqual(
                reproduction["receipt"]["execution_isolation"],
                reproduction["execution_isolation"],
            )

    def test_reproduction_receipt_discloses_bounded_tail_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            script = root / ".xscientist" / "large_output.py"
            script.write_text("print('x' * 25000)\n", encoding="utf-8")
            checkpoint = create_checkpoint(
                root,
                stage="review",
                subject="record bounded output probe",
                reproduce_command="python .xscientist/large_output.py",
            )

            reproduction = reproduce_checkpoint(
                root,
                commit=checkpoint.commit or "HEAD",
                destination=base / "bounded-output",
                execute=True,
                timeout_seconds=10,
            )

            receipt = reproduction["receipt"]
            self.assertEqual(len(reproduction["stdout"]), 20_000)
            self.assertTrue(reproduction["stdout_truncated"])
            self.assertFalse(reproduction["stderr_truncated"])
            self.assertEqual(receipt["output_capture"], "bounded_tail")
            self.assertEqual(receipt["max_output_chars"], 20_000)
            self.assertTrue(receipt["stdout_truncated"])
            self.assertEqual(
                receipt["stdout_hash"], content_hash(reproduction["stdout"])
            )

    def test_reproduction_isolation_reports_platform_process_limits_honestly(
        self,
    ) -> None:
        posix = research_git_module._reproduction_execution_isolation("posix")
        windows = research_git_module._reproduction_execution_isolation("nt")

        self.assertEqual(posix["process_tree"], "best_effort_process_group")
        self.assertEqual(posix["process_control"], "posix_process_group_best_effort")
        self.assertEqual(windows["process_tree"], "parent_only_no_tree_guarantee")
        self.assertEqual(windows["process_control"], "parent_process_only")
        self.assertFalse(posix["process_tree_termination_guaranteed"])
        self.assertFalse(windows["process_tree_termination_guaranteed"])
        self.assertEqual(posix["filesystem"], "host_visible")
        self.assertEqual(windows["filesystem"], "host_visible")

    def test_reproduction_control_paths_refuse_symlink_parents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            worktree = base / "worktree"
            outside = base / "outside"
            worktree.mkdir()
            outside.mkdir()
            (worktree / ".xscientist").symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ResearchGitError, "contains a symlink"):
                research_git_module._safe_worktree_control_path(
                    worktree,
                    ".xscientist/reproductions/receipt.json",
                )

            self.assertFalse((outside / "reproductions" / "receipt.json").exists())

    def test_reproduction_preflight_failure_leaves_no_registered_worktree(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "research"
            self._init(root)
            checkpoint = create_checkpoint(
                root,
                stage="review",
                subject="checkpoint without an execution command",
            )
            destination = base / "partial-reproduction"

            with self.assertRaisesRegex(
                ResearchGitError, "does not declare a reproduction command"
            ):
                reproduce_checkpoint(
                    root,
                    commit=checkpoint.commit or "HEAD",
                    destination=destination,
                    execute=True,
                )

            self.assertFalse(destination.exists())
            self.assertNotIn(
                str(destination.resolve()),
                self._git(root, "worktree", "list", "--porcelain"),
            )

    def test_reproduction_refuses_a_tampered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            self._init(root)
            checkpoint_path = sorted((root / "checkpoints").glob("*.json"))[-1]
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload["summary"] = "tampered"
            checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
            self._git(root, "add", checkpoint_path.relative_to(root).as_posix())
            self._git(
                root,
                "commit",
                "-m",
                "tamper checkpoint",
                "-m",
                "\n".join(
                    [
                        f"Research-Checkpoint: {payload['checkpoint_id']}",
                        f"Research-Stage: {payload['stage']}",
                        f"Research-State: {payload['status']}",
                        f"Research-Event: {payload['content_hash']}",
                    ]
                ),
            )

            with self.assertRaisesRegex(ResearchGitError, "hash verification failed"):
                show_checkpoint(root)
            with self.assertRaisesRegex(ResearchGitError, "hash verification failed"):
                reproduce_checkpoint(root)

    def test_unified_cli_supports_init_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "research"
            output = io.StringIO()
            with redirect_stdout(output):
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
            self.assertNotIn(str(root), output.getvalue())
            self.assertIn("[REDACTED_PATH]", output.getvalue())


if __name__ == "__main__":
    unittest.main()
