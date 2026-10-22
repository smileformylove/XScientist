"""Environment matches describe the launcher and lock bytes, not child attestation."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from jsonschema import ValidationError, validate

from ai_scientist.protocol.schemas import load_schema
from xscientist import research_git
from xscientist.research_cli import main as research_main

SCOPE = {
    "runtime_identity": "launcher_python_implementation_version_platform_system_machine",
    "dependency_locks": "declared_checkpoint_file_bytes",
    "dependency_locks_checked": False,
    "installed_dependencies_attested": False,
    "child_runtime_attested": False,
}


def initialize(root):
    research_git.init_repository(
        root,
        question="Deterministic replay environment product fixture only.",
        git_user_name="Replay Fixture",
        git_user_email="replay@example.invalid",
    )


class ReproductionEnvironmentScopeTests(unittest.TestCase):
    def test_runtime_comparison_scope_is_explicit_even_without_recorded_environment(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = research_git._environment_receipt(Path(temporary))
        for recorded in (receipt, None):
            with self.subTest(recorded=recorded is not None):
                comparison = research_git._compare_runtime_environment(recorded)
                self.assertEqual(comparison["scope"], SCOPE)
                self.assertIs(comparison["matches"], True if recorded else None)
                comparison["scope"]["child_runtime_attested"] = True
                fresh = research_git._compare_runtime_environment(recorded)
                self.assertFalse(fresh["scope"]["child_runtime_attested"])

    def test_lockfile_scope_distinguishes_inspection_and_materialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            initialize(root)
            (root / "requirements.txt").write_text("example==1.0\n", encoding="utf-8")
            research_git.create_checkpoint(
                root, stage="review", subject="lock bytes fixture"
            )
            inspection = research_git.reproduce_checkpoint(
                root, environment_policy="strict"
            )
            materialized = research_git.reproduce_checkpoint(
                root, destination=base / "replay", environment_policy="strict"
            )
            for result, checked in ((inspection, False), (materialized, True)):
                with self.subTest(checked=checked):
                    expected = {**SCOPE, "dependency_locks_checked": checked}
                    self.assertEqual(result["environment"]["scope"], expected)
                    self.assertEqual(
                        result["receipt"]["environment"]["scope"], expected
                    )
                    self.assertTrue(result["environment"]["matches"])
                    validate(result["receipt"], load_schema("reproduction_receipt"))

    def test_optional_scope_preserves_legacy_v1_and_v2_receipt_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            initialize(root)
            receipt = research_git.reproduce_checkpoint(root)["receipt"]
        schema = load_schema("reproduction_receipt")
        validate(receipt, schema)
        for version in (
            "xscientist.reproduction-receipt.v1",
            "xscientist.reproduction-receipt.v2",
        ):
            with self.subTest(version=version):
                legacy = deepcopy(receipt)
                legacy["schema_version"] = version
                legacy["environment"].pop("scope")
                if version.endswith(".v1"):
                    for field in (
                        "checkpoint_binding",
                        "target_binding",
                        "execution_result",
                        "execution_isolation",
                    ):
                        legacy.pop(field)
                validate(legacy, schema)
        for field in ("installed_dependencies_attested", "child_runtime_attested"):
            with self.subTest(field=field):
                invalid = deepcopy(receipt)
                invalid["environment"]["scope"][field] = True
                with self.assertRaises(ValidationError):
                    validate(invalid, schema)
        invalid = deepcopy(receipt)
        invalid["environment"]["scope"]["executable_path"] = "/private/runtime"
        with self.assertRaises(ValidationError):
            validate(invalid, schema)

    def test_strict_launcher_drift_gate_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            initialize(root)
            destination = Path(temporary) / "replay"
            with mock.patch.object(
                research_git.platform, "python_version", return_value="0.0.0"
            ):
                with self.assertRaisesRegex(
                    research_git.ResearchGitError,
                    "environment mismatch: python.version",
                ):
                    research_git.reproduce_checkpoint(
                        root, destination=destination, environment_policy="strict"
                    )
            self.assertFalse(destination.exists())

    def test_execute_defaults_to_strict_before_worktree_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            initialize(root)
            research_git.create_checkpoint(
                root,
                stage="review",
                subject="default execute policy fixture",
                reproduce_command="python3 probe.py",
            )
            destination = base / "replay"
            with mock.patch.object(
                research_git.platform, "python_version", return_value="0.0.0"
            ):
                with self.assertRaisesRegex(
                    research_git.ResearchGitError,
                    "environment mismatch: python.version",
                ):
                    research_git.reproduce_checkpoint(
                        root, destination=destination, execute=True
                    )
            self.assertFalse(destination.exists())

    def test_inspection_keeps_warning_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "research"
            initialize(root)
            result = research_git.reproduce_checkpoint(root)
            self.assertEqual(result["environment_policy"], "warn")

    def test_accepted_command_forms_never_attest_child_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "research"
            initialize(root)
            commands = (
                ("python3 -B probe.py", os.path.abspath(sys.executable)),
                ("python3.13 -B probe.py", "python3.13"),
                ("env python3 -B probe.py", "env"),
            )
            completed = SimpleNamespace(
                returncode=0,
                stdout="different child runtime\n",
                stderr="",
                stdout_truncated=False,
                stderr_truncated=False,
            )
            for index, (command, executable) in enumerate(commands):
                with self.subTest(command=command):
                    (root / ".xscientist" / "case.json").write_text(
                        json.dumps({"case": index}) + "\n", encoding="utf-8"
                    )
                    research_git.create_checkpoint(
                        root,
                        stage="review",
                        subject=f"command form {index}",
                        reproduce_command=command,
                    )
                    with mock.patch.object(
                        research_git, "run_process_bounded", return_value=completed
                    ) as run:
                        result = research_git.reproduce_checkpoint(
                            root,
                            destination=base / f"replay-{index}",
                            execute=True,
                            environment_policy="strict",
                        )
                    self.assertEqual(run.call_args.args[0][0], executable)
                    self.assertTrue(result["environment"]["matches"])
                    self.assertFalse(
                        result["environment"]["scope"]["child_runtime_attested"]
                    )
                    self.assertFalse(
                        result["receipt"]["environment"]["scope"][
                            "child_runtime_attested"
                        ]
                    )
                    self.assertEqual(result["receipt"]["verdict"], "passed")
                    validate(result["receipt"], load_schema("reproduction_receipt"))


class ReproductionEnvironmentScopeCliTests(unittest.TestCase):
    def invoke(self, scope, *, as_json=False, matches=True):
        environment = {"matches": matches, "mismatches": []}
        if scope is not None:
            environment["scope"] = scope
        payload = {
            "commit": "a" * 40,
            "checkpoint": {"checkpoint_id": "rcp-fixture"},
            "objects_complete": True,
            "environment": environment,
            "command": "python3 probe.py",
            "receipt": {"receipt_id": "rr-fixture"},
        }
        output = io.StringIO()
        with mock.patch(
            "xscientist.research_cli.reproduce_checkpoint", return_value=payload
        ):
            with redirect_stdout(output):
                args = ["reproduce", "HEAD", "--repo", "fixture"]
                if as_json:
                    args.append("--json")
                self.assertEqual(research_main(args), 0)
        return output.getvalue()

    def test_human_output_qualifies_matched_checks(self):
        output = self.invoke(SCOPE)
        self.assertIn("Scoped checks:    matched", output)
        self.assertIn(
            "Identity scope:   launcher Python implementation/version; platform system/machine",
            output,
        )
        self.assertIn("Lockfile bytes:   not checked", output)
        self.assertIn(
            "Not attested:     installed dependencies; child runtimes", output
        )
        self.assertNotIn("Environment:", output)

    def test_checked_lockfiles_and_legacy_scope_are_distinct(self):
        output = self.invoke({**SCOPE, "dependency_locks_checked": True}, matches=False)
        self.assertIn("Scoped checks:    mismatched", output)
        self.assertIn("Lockfile bytes:   checked", output)
        legacy = self.invoke(None, matches=None)
        self.assertIn("Scoped checks:    not recorded", legacy)
        self.assertIn("Check scope:      not recorded (legacy result)", legacy)
        self.assertNotIn("attested: True", legacy)

    def test_json_exposes_scope_without_command_resolution_paths(self):
        result = json.loads(self.invoke(SCOPE, as_json=True))
        self.assertEqual(result["environment"]["scope"], SCOPE)
        self.assertNotIn("argv", result["environment"])
        self.assertNotIn("executable", result["environment"])

    def test_cli_leaves_execute_default_for_api_context_resolution(self):
        environment = {"matches": True, "mismatches": []}
        payload = {
            "commit": "a" * 40,
            "checkpoint": {"checkpoint_id": "rcp-fixture"},
            "objects_complete": True,
            "environment": environment,
            "command": "python3 probe.py",
            "receipt": {"receipt_id": "rr-fixture"},
        }
        with mock.patch(
            "xscientist.research_cli.reproduce_checkpoint", return_value=payload
        ) as reproduce:
            self.assertEqual(
                research_main(
                    ["reproduce", "HEAD", "--repo", "fixture", "--execute"]
                ),
                0,
            )
        self.assertIsNone(reproduce.call_args.kwargs["environment_policy"])


if __name__ == "__main__":
    unittest.main()
