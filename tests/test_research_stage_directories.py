from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import xscientist.research_git as research_git
from xscientist.research_cli import main as research_main


@unittest.skipUnless(shutil.which("git"), "Git is required")
class ResearchStageDirectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="research-stage-directories-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "research"
        research_git.init_repository(
            self.root,
            question="# Question\n\nDirectory staging fixture.\n",
            git_user_name="Stage Test",
            git_user_email="stage@example.invalid",
        )

    def write(self, relative: str, text: str = "{}\n") -> Path:
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return target

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True, text=True, capture_output=True,
        ).stdout.strip()

    def native_bytes(self) -> bytes | None:
        path = research_git._research_stage_path(self.root)
        return path.read_bytes() if path.exists() else None

    def test_directory_expands_changed_files_not_unchanged_or_sibling_prefixes(self) -> None:
        self.write("hypotheses/packet/unchanged.json")
        changed = self.write("hypotheses/packet/changed.json")
        research_git.create_checkpoint(self.root, stage="plan", subject="fixture")
        changed.write_text('{"changed":true}\n', encoding="utf-8")
        nested = self.write("hypotheses/packet/nested/new.json")
        self.write("hypotheses/packet-sibling.json")
        self.write("hypotheses/packet-extra/new.json")
        head = self.git("rev-parse", "HEAD")

        result = research_git.research_stage(self.root, ["hypotheses/packet/"])

        expected = {
            "hypotheses/packet/changed.json", "hypotheses/packet/nested/new.json"
        }
        self.assertEqual(set(result.paths), expected)
        self.assertEqual(set(result.added), expected)
        entries = research_git._load_research_stage(self.root)["entries"]
        self.assertEqual(
            {row["path"]: row["fingerprint"] for row in entries},
            {changed.relative_to(self.root).as_posix(): research_git._hash_file(changed),
             nested.relative_to(self.root).as_posix(): research_git._hash_file(nested)},
        )
        self.assertEqual(self.git("rev-parse", "HEAD"), head)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_directory_keeps_existing_native_selection_and_deduplicates(self) -> None:
        self.write("claims/existing.json")
        research_git.research_stage(self.root, ["claims/existing.json"])
        self.write("hypotheses/packet/new.json")

        result = research_git.research_stage(
            self.root,
            ["hypotheses/packet", "hypotheses/packet/", "hypotheses/packet/new.json"],
        )

        self.assertEqual(result.paths, ("claims/existing.json", "hypotheses/packet/new.json"))
        self.assertEqual(result.added, ("hypotheses/packet/new.json",))

    def test_existing_and_removed_directories_include_tracked_deletion_tombstones(self) -> None:
        removed = self.write("hypotheses/removed/item.json")
        surviving = self.write("hypotheses/surviving/item.json")
        research_git.create_checkpoint(self.root, stage="plan", subject="fixture")
        removed.unlink()
        removed.parent.rmdir()
        surviving.unlink()

        result = research_git.research_stage(
            self.root, ["hypotheses/removed", "hypotheses/surviving"]
        )

        self.assertEqual(set(result.paths), {
            "hypotheses/removed/item.json", "hypotheses/surviving/item.json"
        })
        self.assertTrue(all(row["fingerprint"] == "deleted"
                            for row in research_git._load_research_stage(self.root)["entries"]))

    def test_directory_preserves_deny_size_and_git_ignore_filters(self) -> None:
        self.write("packet/result.txt", "portable result\n")
        self.write("packet/weights.bin", "not a checkpoint\n")
        self.write("packet/large.txt", "x" * (2 * 1024 * 1024 + 1))
        self.write("packet/private.ignored", "not selected\n")
        with (self.root / ".gitignore").open("a", encoding="utf-8") as stream:
            stream.write("\n!packet/weights.bin\n")
        with (self.root / ".git" / "info" / "exclude").open("a", encoding="utf-8") as stream:
            stream.write("\npacket/private.ignored\n")

        result = research_git.research_stage(self.root, ["packet"])

        self.assertEqual(result.paths, ("packet/result.txt",))
        self.assertTrue(any("packet/large.txt" in row and "exceeds" in row
                            for row in result.excluded))
        self.assertTrue(any("packet/weights.bin" in row and "denied pattern" in row
                            for row in result.excluded))
        self.assertNotIn("packet/private.ignored", "\n".join(result.paths))
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_privacy_failure_does_not_mutate_existing_native_or_git_stage(self) -> None:
        self.write("claims/existing.json")
        research_git.research_stage(self.root, ["claims/existing.json"])
        before = self.native_bytes()
        token = "sk-" + "q" * 32
        self.write("hypotheses/packet/result.json")
        self.write("hypotheses/packet/unsafe.json", json.dumps({"credential": token}))

        with self.assertRaisesRegex(research_git.ResearchGitError, "privacy gate refused") as caught:
            research_git.research_stage(self.root, ["hypotheses/packet"])

        self.assertNotIn(token, str(caught.exception))
        self.assertEqual(self.native_bytes(), before)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_symlink_directory_file_and_ancestor_are_rejected_before_scanning(self) -> None:
        self.write("hypotheses/real/result.json")
        alias = self.root / "hypotheses" / "alias"
        try:
            alias.symlink_to("real", target_is_directory=True)
            link = self.root / "hypotheses" / "linked.json"
            link.symlink_to("real/result.json")
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        for request in ("hypotheses/alias", "hypotheses/alias/result.json",
                        "hypotheses/linked.json", "hypotheses"):
            with self.subTest(request=request), mock.patch.object(research_git, "scan_paths") as scanner:
                with self.assertRaisesRegex(research_git.ResearchGitError, "symbolic links"):
                    research_git.research_stage(self.root, [request])
                scanner.assert_not_called()
                self.assertIsNone(self.native_bytes())

    def test_parent_replaced_with_symlink_rejects_changed_descendant(self) -> None:
        self.write("hypotheses/packet/item.json")
        research_git.create_checkpoint(self.root, stage="plan", subject="fixture")
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        (outside / "item.json").write_text("outside\n", encoding="utf-8")
        (self.root / "hypotheses/packet/item.json").unlink()
        (self.root / "hypotheses/packet").rmdir()
        (self.root / "hypotheses/packet").symlink_to(outside, target_is_directory=True)
        with mock.patch.object(research_git, "scan_paths") as scanner:
            with self.assertRaisesRegex(research_git.ResearchGitError, "symbolic links"):
                research_git.research_stage(self.root, ["hypotheses"])
            scanner.assert_not_called()

    def test_absolute_traversing_and_root_selectors_are_rejected(self) -> None:
        self.write("hypotheses/packet/item.json")
        for request in (str(self.root / "hypotheses/packet"), "../hypotheses",
                        "hypotheses/../hypotheses", ".", ""):
            with self.subTest(request=request):
                with self.assertRaisesRegex(research_git.ResearchGitError, "unsafe repository-relative"):
                    research_git.research_stage(self.root, [request])
        self.assertIsNone(self.native_bytes())

    def test_empty_or_missing_selector_fails_without_staging_partial_selection(self) -> None:
        self.write("hypotheses/packet/item.json")
        (self.root / "empty").mkdir()
        for missing in ("empty", "missing"):
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(research_git.ResearchGitError, "no working-state change"):
                    research_git.research_stage(self.root, ["hypotheses/packet", missing])
                self.assertIsNone(self.native_bytes())

    def test_preexisting_git_stage_is_preserved_and_blocks_directory_staging(self) -> None:
        self.write("claims/existing.json")
        research_git.research_stage(self.root, ["claims/existing.json"])
        self.write("hypotheses/packet/item.json")
        self.git("add", "hypotheses/packet/item.json")
        before = self.native_bytes()
        git_before = self.git("diff", "--cached", "--binary")

        with self.assertRaisesRegex(research_git.ResearchGitError, "backend index is not clean"):
            research_git.research_stage(self.root, ["hypotheses/packet"])

        self.assertEqual(self.native_bytes(), before)
        self.assertEqual(self.git("diff", "--cached", "--binary"), git_before)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO requires POSIX")
    def test_nonregular_changed_descendant_is_rejected_before_privacy_scan(self) -> None:
        folder = self.root / "hypotheses/packet"
        folder.mkdir()
        os.mkfifo(folder / "stream")
        with (
            mock.patch.object(research_git, "_changed_paths", return_value=(set(), {"hypotheses/packet/stream"})),
            mock.patch.object(research_git, "scan_paths") as scanner,
        ):
            with self.assertRaisesRegex(research_git.ResearchGitError, "regular file"):
                research_git.research_stage(self.root, ["hypotheses/packet"])
            scanner.assert_not_called()
            self.assertIsNone(self.native_bytes())

    def test_directory_snapshot_does_not_later_absorb_new_files(self) -> None:
        self.write("hypotheses/packet/first.json")
        research_git.research_stage(self.root, ["hypotheses/packet"])
        self.write("hypotheses/packet/later.json")

        result = research_git.commit_research_stage(
            self.root, stage="plan", subject="exact selected fixture"
        )

        self.assertIn("hypotheses/packet/first.json", result.staged_paths)
        self.assertNotIn("hypotheses/packet/later.json", result.staged_paths)
        self.assertIn("hypotheses/packet/later.json", self.git("ls-files", "--others", "--exclude-standard"))

    def test_file_changed_after_directory_selection_still_rejects_stale_fingerprint(self) -> None:
        target = self.write("hypotheses/packet/item.json")
        research_git.research_stage(self.root, ["hypotheses/packet"])
        before = self.native_bytes()
        target.write_text('{"new":true}\n', encoding="utf-8")
        with self.assertRaisesRegex(research_git.ResearchGitError, "changed after selection"):
            research_git.commit_research_stage(self.root, stage="plan", subject="stale fixture")
        self.assertEqual(self.native_bytes(), before)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")

    def test_cli_stage_and_add_accept_directory_and_document_semantics(self) -> None:
        self.write("hypotheses/packet/item.json")
        for command in ("stage", "add"):
            with self.subTest(command=command):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = research_main([command, "hypotheses/packet", "--repo", str(self.root), "--json"])
                self.assertEqual(code, 0)
                self.assertEqual(json.loads(output.getvalue())["paths"], ["hypotheses/packet/item.json"])
                help_output = io.StringIO()
                with redirect_stdout(help_output), self.assertRaises(SystemExit) as caught:
                    research_main([command, "--help"])
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("files or directories", help_output.getvalue())

    def test_directory_with_spaces_is_literal_and_all_remains_mutually_exclusive(self) -> None:
        self.write("packet with spaces/item.txt", "portable fixture\n")
        self.write("packet with spaces sibling/item.txt", "not selected\n")
        result = research_git.research_stage(self.root, ["packet with spaces"])
        self.assertEqual(result.paths, ("packet with spaces/item.txt",))
        before = self.native_bytes()
        with self.assertRaisesRegex(research_git.ResearchGitError, "either explicit"):
            research_git.research_stage(
                self.root, ["packet with spaces"], all_changes=True
            )
        self.assertEqual(self.native_bytes(), before)

    def test_file_to_directory_selector_preserves_existing_stage_on_rejection(self) -> None:
        tracked = self.write("hypotheses/packet", "tracked file\n")
        research_git.create_checkpoint(self.root, stage="plan", subject="fixture")
        self.write("claims/existing.json")
        research_git.research_stage(self.root, ["claims/existing.json"])
        before = self.native_bytes()
        head = self.git("rev-parse", "HEAD")
        tracked.unlink()
        self.write("hypotheses/packet/item.json")
        _, changed = research_git._changed_paths(self.root)
        self.assertIn("hypotheses/packet", changed)
        self.assertIn("hypotheses/packet/item.json", changed)

        with mock.patch.object(
            research_git, "scan_paths", wraps=research_git.scan_paths
        ) as scanner:
            with self.assertRaisesRegex(research_git.ResearchGitError, "file-to-directory"):
                research_git.research_stage(self.root, ["hypotheses/packet"])
            scanner.assert_not_called()
        self.assertEqual(self.native_bytes(), before)
        self.assertEqual(self.git("diff", "--cached", "--name-only"), "")
        self.assertEqual(self.git("rev-parse", "HEAD"), head)


if __name__ == "__main__":
    unittest.main()
