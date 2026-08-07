from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xscientist.cli import main as cli_main

from ai_scientist.utils.privacy import (
    REDACTED_EMAIL,
    REDACTED_PATH,
    portable_path,
    privacy_report,
    redact_sensitive_payload,
    redact_sensitive_text,
    relative_path_reference,
)
from ai_scientist.utils.auth_session import require_login
from tools.repository_validation import _run_with_privacy_filter
from tools.privacy_exec import run as privacy_safe_run


class PrivacyTests(unittest.TestCase):
    def test_redaction_removes_secrets_email_and_host_paths(self) -> None:
        token = "sk-" + "x" * 32
        raw = (
            f"api_key={token} contact=private@example.invalid "
            "source=/" + "Users/private-person/research/input.json"
        )
        output = redact_sensitive_text(raw)
        self.assertNotIn(token, output)
        self.assertNotIn("private@example.invalid", output)
        self.assertNotIn("private-person", output)
        self.assertIn("[REDACTED]", output)
        self.assertIn(REDACTED_EMAIL, output)
        self.assertIn(REDACTED_PATH, output)

    def test_payload_redaction_covers_keys_nested_values_and_paths(self) -> None:
        payload = {
            "contact@example.invalid": [
                {"trace": Path("/" + "home/private-person/work/trace.json")}
            ]
        }
        rendered = json.dumps(redact_sensitive_payload(payload))
        self.assertNotIn("contact@example.invalid", rendered)
        self.assertNotIn("private-person", rendered)

    def test_payload_redaction_treats_secret_field_names_as_sensitive(self) -> None:
        value = "provider-specific-value-without-a-known-prefix"
        rendered = json.dumps(redact_sensitive_payload({"api_key": value}))
        self.assertNotIn(value, rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_portable_path_keeps_internal_paths_and_hides_external_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "workspace"
            root.mkdir()
            self.assertEqual(portable_path(root, base=root), ".")
            self.assertEqual(
                portable_path(root / "env" / "local", base=root), "env/local"
            )
            self.assertEqual(
                portable_path(Path(td) / "elsewhere", base=root), REDACTED_PATH
            )
            self.assertEqual(
                relative_path_reference(Path(td) / "elsewhere", base=root),
                "../elsewhere",
            )

    def test_audit_reports_location_but_never_matched_value(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            token = "sk-" + "z" * 32
            (root / "unsafe.txt").write_text(token, encoding="utf-8")
            report = privacy_report(root)
            rendered = json.dumps(report)
            self.assertFalse(report["ok"])
            self.assertEqual(report["finding_count"], 1)
            self.assertEqual(report["findings"][0]["path"], "unsafe.txt")
            self.assertNotIn(token, rendered)
            self.assertFalse(report["matched_values_disclosed"])

    def test_audit_detects_high_entropy_provider_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            value = "account1234567890.signature-abcdef"
            (root / "settings.env").write_text(
                f"ZHIPU_API_KEY={value}\n", encoding="utf-8"
            )
            report = privacy_report(root)
            self.assertFalse(report["ok"])
            self.assertEqual(report["findings"][0]["rule"], "credential_assignment")
            self.assertNotIn(value, json.dumps(report))

    def test_placeholder_examples_are_not_reported_as_personal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "example.md").write_text(
                "/home/user/project and /Users/example/work",
                encoding="utf-8",
            )
            self.assertTrue(privacy_report(root)["ok"])

    def test_cli_audit_never_echoes_a_matched_secret(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            token = "sk-" + "n" * 32
            (root / "unsafe.txt").write_text(token, encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = cli_main(["privacy", "audit", str(root), "--json"])
            rendered = stdout.getvalue() + stderr.getvalue()
            self.assertEqual(exit_code, 1)
            self.assertNotIn(token, rendered)
            self.assertFalse(json.loads(stdout.getvalue())["matched_values_disclosed"])

    def test_login_failure_never_discloses_session_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            session_path = Path(td) / "private-user" / "session.json"
            stdout = io.StringIO()
            with (
                mock.patch.dict(
                    "os.environ",
                    {"AI_SCIENTIST_AUTH_FILE": str(session_path)},
                    clear=False,
                ),
                contextlib.redirect_stdout(stdout),
                self.assertRaises(SystemExit),
            ):
                require_login("privacy test")
            self.assertNotIn(str(session_path), stdout.getvalue())
            self.assertIn(REDACTED_PATH, stdout.getvalue())

    def test_repository_validation_filters_helper_output(self) -> None:
        private_path = "/" + "Users/private-person/work/result.json"
        token = "sk-" + "q" * 32
        stdout = io.StringIO()
        stderr = io.StringIO()

        def emit_private_diagnostics() -> None:
            print(f"result={private_path} api_key={token}")
            print(f"failed at {private_path}", file=sys.stderr)

        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            _run_with_privacy_filter(emit_private_diagnostics)
        rendered = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(private_path, rendered)
        self.assertNotIn(token, rendered)
        self.assertIn(REDACTED_PATH, rendered)

    def test_privacy_exec_redacts_output_and_preserves_exit_code(self) -> None:
        private_path = "/" + "Users/private-person/work/result.json"
        token = "sk-" + "w" * 32
        script = (
            "import sys; "
            f"print({private_path!r}); "
            f"print({token!r}, file=sys.stderr); "
            "raise SystemExit(7)"
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = privacy_safe_run([sys.executable, "-c", script])
        self.assertEqual(exit_code, 7)
        self.assertNotIn(private_path, stdout.getvalue())
        self.assertNotIn(token, stdout.getvalue())
        self.assertIn(REDACTED_PATH, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
