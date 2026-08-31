from __future__ import annotations

import selectors
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist import research_git


@unittest.skipUnless(shutil.which("git"), "Git is required for bounded Git tests")
class BoundedGitCaptureTests(unittest.TestCase):
    def _git(self, root: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _run_python_child(
        self,
        script: str,
        *,
        max_output_bytes: int,
        timeout_seconds: float = 5.0,
    ) -> tuple[subprocess.CompletedProcess[str], subprocess.Popen[bytes]]:
        real_popen = subprocess.Popen
        children: list[subprocess.Popen[bytes]] = []

        def launch(_command, **kwargs):
            child = real_popen([sys.executable, "-c", script], **kwargs)
            children.append(child)
            return child

        with mock.patch.object(research_git.subprocess, "Popen", side_effect=launch):
            completed = research_git._run_git_bounded(
                Path.cwd(),
                ["ignored-by-test"],
                max_output_bytes=max_output_bytes,
                timeout_seconds=timeout_seconds,
            )
        return completed, children[0]

    def test_captures_git_pipe_output_and_enforces_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._git(root, "init")
            payload = ("bounded-git-output\n" * 256).encode("utf-8")
            source = root / "payload.txt"
            source.write_bytes(payload)
            object_id = self._git(root, "hash-object", "-w", "payload.txt")

            completed = research_git._run_git_bounded(
                root,
                ["cat-file", "blob", object_id],
                max_output_bytes=len(payload),
            )

            self.assertEqual(completed.stdout.encode("utf-8"), payload)
            self.assertEqual(completed.stderr, "")
            with self.assertRaisesRegex(
                research_git.ResearchGitError,
                "exceeded output limit",
            ):
                research_git._run_git_bounded(
                    root,
                    ["cat-file", "blob", object_id],
                    max_output_bytes=128,
                )

    def test_nonzero_git_result_retains_stderr_when_check_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._git(root, "init")

            completed = research_git._run_git_bounded(
                root,
                ["rev-parse", "--verify", "missing-ref"],
                check=False,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(completed.stderr.strip())

    def test_capture_does_not_depend_on_socket_only_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._git(root, "init")

            with mock.patch.object(
                selectors,
                "DefaultSelector",
                side_effect=OSError(10038, "not a socket"),
            ):
                completed = research_git._run_git_bounded(
                    root,
                    ["rev-parse", "--git-dir"],
                )

            self.assertEqual(completed.returncode, 0)
            self.assertTrue(completed.stdout.strip())

    def test_dual_pipe_capture_does_not_deadlock_at_exact_combined_limit(
        self,
    ) -> None:
        completed, child = self._run_python_child(
            "import sys; "
            "sys.stderr.buffer.write(b'e' * 100000); sys.stderr.flush(); "
            "sys.stdout.buffer.write(b'o' * 100000); sys.stdout.flush()",
            max_output_bytes=200_000,
        )

        self.assertEqual(completed.stdout, "o" * 100_000)
        self.assertEqual(completed.stderr, "e" * 100_000)
        self.assertIsNotNone(child.poll())

    def test_overflow_and_timeout_terminate_the_child(self) -> None:
        real_popen = subprocess.Popen
        children: list[subprocess.Popen[bytes]] = []

        def launch_overflow(_command, **kwargs):
            child = real_popen(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(b'x' * 300000); "
                    "sys.stdout.flush()",
                ],
                **kwargs,
            )
            children.append(child)
            return child

        with mock.patch.object(
            research_git.subprocess,
            "Popen",
            side_effect=launch_overflow,
        ):
            with self.assertRaisesRegex(
                research_git.ResearchGitError,
                "exceeded output limit",
            ):
                research_git._run_git_bounded(
                    Path.cwd(),
                    ["ignored-by-test"],
                    max_output_bytes=100_000,
                )
        self.assertIsNotNone(children[-1].poll())

        def launch_timeout(_command, **kwargs):
            child = real_popen(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                **kwargs,
            )
            children.append(child)
            return child

        with mock.patch.object(
            research_git.subprocess,
            "Popen",
            side_effect=launch_timeout,
        ):
            with self.assertRaisesRegex(
                research_git.ResearchGitError,
                "exceeded time limit",
            ):
                research_git._run_git_bounded(
                    Path.cwd(),
                    ["ignored-by-test"],
                    timeout_seconds=0.05,
                )
        self.assertIsNotNone(children[-1].poll())

    def test_pipe_read_failure_is_not_treated_as_clean_eof(self) -> None:
        class InvalidStream:
            def fileno(self) -> int:
                return -1

            def close(self) -> None:
                return None

        class CompletedProcess:
            stdout = InvalidStream()
            stderr = InvalidStream()
            returncode = 0

            def poll(self) -> int:
                return self.returncode

            def wait(self, timeout: float | None = None) -> int:
                return self.returncode

            def kill(self) -> None:
                self.returncode = -9

        with mock.patch.object(
            research_git.subprocess,
            "Popen",
            return_value=CompletedProcess(),
        ):
            with self.assertRaisesRegex(
                research_git.ResearchGitError,
                "output capture failed",
            ):
                research_git._run_git_bounded(Path.cwd(), ["status"])


if __name__ == "__main__":
    unittest.main()
