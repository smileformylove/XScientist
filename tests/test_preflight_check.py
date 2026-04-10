from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import preflight_check


def _write_session(path: Path, *, username: str = "smoke-user") -> None:
    payload = {
        "username": username,
        "session_id": "test-session",
        "issued_at": "2026-03-22T00:00:00+00:00",
        "expires_at": "2099-03-22T00:00:00+00:00",
        "last_seen_at": "2026-03-22T00:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PreflightCheckTests(unittest.TestCase):
    def test_preflight_should_respect_auth_file_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "auth" / "session.json"
            _write_session(session_path)
            stdout = io.StringIO()
            with mock.patch.object(preflight_check, "CORE_PACKAGES", {}), mock.patch.object(
                preflight_check, "PIPELINE_PACKAGES", {}
            ), mock.patch.object(preflight_check, "COMMANDS", {}), mock.patch(
                "sys.argv",
                ["preflight_check.py", "--auth-file", str(session_path)],
            ), mock.patch.dict("os.environ", {}, clear=False), redirect_stdout(stdout):
                exit_code = preflight_check.main()

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn(str(session_path), output)
        self.assertIn("[OK] Login session", output)

    def test_preflight_strict_should_fail_for_missing_login_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "auth" / "missing.json"
            stdout = io.StringIO()
            with mock.patch.object(preflight_check, "CORE_PACKAGES", {}), mock.patch.object(
                preflight_check, "PIPELINE_PACKAGES", {}
            ), mock.patch.object(preflight_check, "COMMANDS", {}), mock.patch(
                "sys.argv",
                [
                    "preflight_check.py",
                    "--strict",
                    "--auth-file",
                    str(session_path),
                ],
            ), mock.patch.dict("os.environ", {}, clear=False), redirect_stdout(stdout):
                exit_code = preflight_check.main()

        self.assertEqual(exit_code, 1)
        output = stdout.getvalue()
        self.assertIn(str(session_path), output)
        self.assertIn("[ERROR] Login session", output)
        self.assertIn("未检测到登录会话", output)


if __name__ == "__main__":
    unittest.main()
