from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from xscientist.benchmark import benchmark_first_run
from xscientist.cli import main
from xscientist.completion import completion_script
from xscientist.conformance import check_conformance, init_conformance_kit
from xscientist.provider_config import CONFIG_RELATIVE_PATH, CONFIG_SCHEMA_VERSION
from xscientist.research_git import REPOSITORY_SCHEMA
from xscientist.upgrade_check import check_upgrade
from xscientist.usage_metrics import (
    export_metrics,
    metrics_status,
    record_event,
    set_metrics_enabled,
)


class _PyPIResponse(io.StringIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ProductivityP2Tests(unittest.TestCase):
    def test_first_run_benchmark_is_zero_cost_and_path_free(self) -> None:
        payload = benchmark_first_run(max_seconds=30)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["network_used"])
        self.assertEqual(payload["model_cost_usd"], 0.0)
        self.assertFalse(payload["workspace_retained"])
        self.assertGreaterEqual(payload["research"]["dag_nodes"], 4)
        self.assertNotIn(str(Path.home()), json.dumps(payload))

    def test_upgrade_check_is_offline_by_default_and_validates_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            provider_path = root / CONFIG_RELATIVE_PATH
            provider_path.parent.mkdir(parents=True)
            provider_path.write_text(
                json.dumps(
                    {
                        "schema_version": CONFIG_SCHEMA_VERSION,
                        "active_provider": None,
                        "env_file": ".env",
                        "providers": {},
                    }
                ),
                encoding="utf-8",
            )
            (root / "research.yaml").write_text(
                yaml.safe_dump({"schema_version": REPOSITORY_SCHEMA}),
                encoding="utf-8",
            )
            with mock.patch("urllib.request.urlopen") as urlopen:
                payload = check_upgrade(root)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["package"]["online_checked"])
            urlopen.assert_not_called()

    def test_upgrade_online_check_reports_a_newer_version_without_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as raw:
            response = _PyPIResponse(json.dumps({"info": {"version": "99.0.0"}}))
            with mock.patch("urllib.request.urlopen", return_value=response):
                payload = check_upgrade(raw, online=True)
            self.assertTrue(payload["package"]["update_available"])
            self.assertEqual(payload["package"]["index_relation"], "update_available")
            self.assertFalse(payload["mutated"])
            self.assertEqual(list(Path(raw).iterdir()), [])

    def test_upgrade_explains_an_unreleased_installed_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            response = _PyPIResponse(json.dumps({"info": {"version": "0.1.2"}}))
            with mock.patch("urllib.request.urlopen", return_value=response):
                payload = check_upgrade(raw, online=True)

        self.assertFalse(payload["package"]["update_available"])
        self.assertEqual(payload["package"]["index_relation"], "newer_than_index")

    def test_upgrade_check_reports_incompatible_schema(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config = root / CONFIG_RELATIVE_PATH
            config.parent.mkdir(parents=True)
            config.write_text('{"schema_version": 999}', encoding="utf-8")
            payload = check_upgrade(root)
            self.assertFalse(payload["compatible"])
            self.assertTrue(payload["remediations"])

    def test_upgrade_check_rejects_a_missing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            payload = check_upgrade(Path(raw) / "missing")

        self.assertFalse(payload["ok"])
        self.assertFalse(payload["checks"]["workspace"]["present"])
        self.assertTrue(payload["remediations"])

    def test_completion_covers_the_practical_command_surface(self) -> None:
        for shell in ("bash", "zsh", "fish"):
            script = completion_script(shell)
            for command in (
                "explore",
                "demo",
                "start",
                "status",
                "audit",
                "history",
                "runs",
                "auth",
                "executor",
                "research",
                "upgrade",
            ):
                self.assertIn(command, script)
            self.assertIn("cancel", script)
            self.assertIn("login", script)
            self.assertIn("guide", script)
            self.assertIn("prepare", script)
            self.assertIn("rollback", script)
            self.assertIn("show", script)
            self.assertIn("diff", script)
            self.assertTrue("--workspace" in script or "-l workspace" in script)

    def test_conformance_kit_expects_the_bad_fixture_to_fail(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "kit"
            created = init_conformance_kit(root)
            report = check_conformance(root)
            self.assertEqual(created["cases"], 2)
            self.assertTrue(report["ok"])
            self.assertEqual(report["passed"], 2)
            self.assertFalse(report["cases"][1]["actual_valid"])
            self.assertNotIn(raw, created["next_command"])

    def test_metrics_are_disabled_by_default_and_store_only_fixed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with mock.patch.dict(
                os.environ,
                {"XSCIENTIST_METRICS_DIR": raw, "XSCIENTIST_USAGE_METRICS": ""},
                clear=False,
            ):
                self.assertFalse(record_event("demo", status="ok"))
                self.assertEqual(metrics_status()["event_count"], 0)
                set_metrics_enabled(True)
                self.assertTrue(record_event("demo", status="ok", duration_seconds=2))
                payload = export_metrics()
                event = payload["events"][0]
                self.assertEqual(
                    set(event),
                    {
                        "schema",
                        "timestamp",
                        "version",
                        "event",
                        "status",
                        "duration_bucket",
                    },
                )
                self.assertNotIn(raw, json.dumps(event))
                set_metrics_enabled(False)
                self.assertFalse(metrics_status()["enabled"])

    def test_cli_completion_and_metrics_are_human_readable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["completion", "bash"]), 0)
            self.assertIn("complete -F", output.getvalue())
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"XSCIENTIST_METRICS_DIR": raw}):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(["metrics", "status"]), 0)
            self.assertIn("Network transmission: disabled", output.getvalue())


if __name__ == "__main__":
    unittest.main()
