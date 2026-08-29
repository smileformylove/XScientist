from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema

from xscientist.benchmark import benchmark_first_run
from xscientist.cli import (
    _DELEGATES,
    _build_capability_parser,
    _build_doctor_parser,
    _build_parser as build_public_parser,
    _build_setup_parser,
    _build_start_parser,
    main,
)
from ai_scientist.apps.auth import build_parser as build_auth_parser
from xscientist.completion import (
    COMMANDS,
    OPTIONS,
    RESEARCH_COMMANDS,
    RESEARCH_SUBCOMMANDS,
    SUBCOMMANDS,
    completion_script,
)
from xscientist.conformance import check_conformance, init_conformance_kit
from xscientist.provider_config import CONFIG_RELATIVE_PATH, CONFIG_SCHEMA_VERSION
from xscientist.research_git import REPOSITORY_SCHEMA, ResearchGitError
from xscientist.research_cli import _build_parser as build_research_parser
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
    @staticmethod
    def _subparser_choices(parser: argparse.ArgumentParser) -> dict[str, object]:
        action = next(
            (
                item
                for item in parser._actions
                if isinstance(item, argparse._SubParsersAction)
            ),
            None,
        )
        return dict(action.choices) if action is not None else {}

    @classmethod
    def _parser_tree_options(cls, parser: argparse.ArgumentParser) -> set[str]:
        options = {
            option for action in parser._actions for option in action.option_strings
        }
        for nested in cls._subparser_choices(parser).values():
            options.update(cls._parser_tree_options(nested))
        return options

    @classmethod
    def _parser_option_tables(
        cls,
        prefix: str,
        parser: argparse.ArgumentParser,
    ) -> dict[str, set[str]]:
        result = {
            prefix: {
                option for action in parser._actions for option in action.option_strings
            }
        }
        for command, nested in cls._subparser_choices(parser).items():
            result.update(cls._parser_option_tables(f"{prefix}::{command}", nested))
        return result

    def test_first_run_profile_is_normalized_and_invalid_profiles_fail_closed(
        self,
    ) -> None:
        payload = benchmark_first_run(profile=" DISCOVERY ", max_seconds=30)
        self.assertEqual(payload["profile"], "discovery")
        with self.assertRaises(ValueError):
            benchmark_first_run(profile="unknown")

    def test_first_run_benchmark_is_zero_cost_and_path_free(self) -> None:
        payload = benchmark_first_run(max_seconds=30)
        validate(payload, load_schema("first_run_benchmark"))
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["network_used"])
        self.assertEqual(payload["model_cost_usd"], 0.0)
        self.assertFalse(payload["workspace_retained"])
        self.assertEqual(
            payload["evidence_index"]["schema"], "xscientist.evidence-index.v1"
        )
        self.assertFalse(payload["evidence_index"]["workspace_mutated"])
        self.assertGreaterEqual(payload["research"]["dag_nodes"], 4)
        self.assertNotIn(str(Path.home()), json.dumps(payload))

    def test_first_run_refuses_to_overwrite_nonempty_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "existing"
            root.mkdir()
            marker = root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(ResearchGitError):
                benchmark_first_run(workspace=root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_first_run_cli_output_is_schema_valid_and_atomic(self) -> None:
        payload = {
            "schema": "xscientist.first-run-benchmark.v1",
            "ok": True,
            "version": "test",
            "runtime": {"python": "3.13", "system": "test"},
            "profile": "balanced",
            "duration_seconds": 0.1,
            "max_seconds": 30.0,
            "threshold_passed": True,
            "network_used": False,
            "provider_used": False,
            "model_cost_usd": 0.0,
            "research": {
                "dag_nodes": 1,
                "dag_relations": 0,
                "closure": "blocked",
                "run_started": True,
                "status_ok": True,
                "budget_available": True,
                "next_step": "review",
            },
            "workspace_retained": False,
            "host_paths_disclosed": False,
            "evidence_index": {
                "schema": "xscientist.evidence-index.v1",
                "available": False,
                "mode": "unavailable",
                "hash_algorithm": "sha256",
                "workspace_root_disclosed": False,
                "paths_disclosed": False,
                "raw_content_included": False,
                "workspace_mutated": False,
                "limits": {"max_files_per_category": 512, "max_bytes": 1},
                "categories": {},
                "ara_contract": {},
                "truncated": False,
                "read_error_count": 0,
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "reports" / "first-run.json"
            output = io.StringIO()
            with (
                mock.patch(
                    "xscientist.benchmark.benchmark_first_run", return_value=payload
                ),
                contextlib.redirect_stdout(output),
            ):
                code = main(
                    [
                        "benchmark",
                        "first-run",
                        "--output",
                        str(destination),
                        "--json",
                    ]
                )
            rendered = json.loads(output.getvalue())
            persisted = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(rendered, persisted)
        validate(rendered, load_schema("first_run_benchmark"))
        self.assertFalse(rendered["report_persistence"]["raw_payloads_included"])

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
                "opportunity",
            ):
                self.assertIn(command, script)
            self.assertIn("cancel", script)
            self.assertIn("login", script)
            self.assertIn("guide", script)
            self.assertIn("prepare", script)
            self.assertIn("rollback", script)
            self.assertIn("show", script)
            self.assertIn("diff", script)
            self.assertIn("provider", script)
            self.assertIn("verify", script)
            self.assertTrue("--report" in script or "-l report" in script)
            self.assertTrue("--live" in script or "-l live" in script)
            self.assertTrue("--workspace" in script or "-l workspace" in script)
            self.assertIn("allocate", script)
            self.assertIn("probability-semantics", script)
            for option in (
                "falsifier",
                "profile-file",
                "preserve-conflicts",
                "include-payloads",
                "ttl-hours",
            ):
                self.assertIn(option, script)
            for command in (
                "discovery",
                "program",
                "literature",
                "estimand",
                "effect",
                "context",
                "decide",
                "adapter",
                "checkpoint",
                "stage",
                "merge",
                "bundle",
                "export",
                "reproduce",
            ):
                self.assertIn(command, script)
            for nested in (
                "template plan assess",
                "portfolio prediction prioritize",
                "plan receipt source update passage",
                "create verify restore",
            ):
                self.assertIn(nested, script)

    def test_completion_command_tables_match_the_real_cli(self) -> None:
        public_choices = self._subparser_choices(build_public_parser())
        self.assertEqual(set(COMMANDS.split()), set(public_choices) | {"help"})

        root_parsers = {
            command: parser
            for command, parser in public_choices.items()
            if command not in _DELEGATES
        }
        root_parsers.update(
            {
                "start": _build_start_parser(),
                "setup": _build_setup_parser(),
                "doctor": _build_doctor_parser(),
                "capability": _build_capability_parser(),
            }
        )
        option_roots = {key.split("::", 1)[0] for key in OPTIONS}
        self.assertEqual(
            option_roots - set(_DELEGATES),
            set(root_parsers) | {"help"},
        )
        for command, parser in root_parsers.items():
            expected = self._parser_option_tables(command, parser)
            actual = {
                key: set(options.split())
                for key, options in OPTIONS.items()
                if key == command or key.startswith(f"{command}::")
            }
            self.assertEqual(actual, expected, command)

        # Delegate stubs accept REMAINDER and intentionally expose no real
        # options. Only delegates with lightweight executable parser contracts
        # are included, and auth is checked against that actual parser here.
        self.assertEqual(option_roots & set(_DELEGATES), {"auth", "batch", "project"})
        expected_auth = self._parser_option_tables("auth", build_auth_parser())
        actual_auth = {
            key: set(options.split())
            for key, options in OPTIONS.items()
            if key == "auth" or key.startswith("auth::")
        }
        self.assertEqual(actual_auth, expected_auth)
        for option in ("--user", "--ttl-hours", "--lang"):
            self.assertIn(option, OPTIONS["auth::login"].split())
        self.assertNotIn("--user", OPTIONS["auth::status"].split())
        self.assertNotIn("--ttl-hours", OPTIONS["auth::status"].split())
        self.assertIn("--live", OPTIONS["provider::check"].split())
        self.assertNotIn("--live", OPTIONS["provider::list"].split())
        self.assertEqual(OPTIONS["help"].split(), ["--all"])
        self.assertEqual(
            set(SUBCOMMANDS["auth"].split()),
            set(self._subparser_choices(build_auth_parser())),
        )

        research_parser = build_research_parser()
        research_choices = self._subparser_choices(research_parser)
        self.assertEqual(set(RESEARCH_COMMANDS.split()), set(research_choices))
        for command, expected in RESEARCH_SUBCOMMANDS.items():
            parser = research_choices[command]
            if command == "bundle":
                action = next(item for item in parser._actions if item.dest == "action")
                actual = set(action.choices)
            else:
                actual = set(self._subparser_choices(parser))
            self.assertEqual(set(expected.split()), actual, command)

    def test_completion_uses_the_research_frontend_for_the_git_alias(self) -> None:
        bash = completion_script("bash")
        zsh = completion_script("zsh")
        fish = completion_script("fish")

        self.assertIn("$command == research || $command == git", bash)
        self.assertIn('"$command" == research || "$command" == git', zsh)
        self.assertIn("__xscientist_research_frontend", fish)
        self.assertNotIn("__fish_seen_subcommand_from", fish)
        self.assertIn("__xscientist_word_is 3 literature", fish)
        self.assertIn("__xscientist_word_is 4 plan", fish)
        self.assertIn("__xscientist_word_is 3 hypothesis", fish)
        literature_plan = [
            line
            for line in fish.splitlines()
            if "__xscientist_word_is 3 literature" in line
            and "__xscientist_word_is 4 plan" in line
        ]
        self.assertTrue(literature_plan)
        self.assertFalse(any("-l test" in line for line in literature_plan))
        for script in (bash, zsh, fish):
            self.assertIn("literature", script)
            self.assertIn("hypothesis", script)
            self.assertTrue("--all" in script or "-l all" in script)

    def test_completion_uses_exact_root_subcommand_option_paths(self) -> None:
        bash = completion_script("bash")
        zsh = completion_script("zsh")
        fish = completion_script("fish")

        for script in (bash, zsh):
            self.assertIn("provider::list", script)
            self.assertIn("provider::check", script)
            self.assertIn("auth::status", script)
            self.assertIn("auth::login", script)

        bash_list = next(
            line
            for line in bash.splitlines()
            if line.strip().startswith("provider::list)")
        )
        zsh_list = next(
            line
            for line in zsh.splitlines()
            if line.strip().startswith("provider::list)")
        )
        self.assertNotIn("--live", bash_list)
        self.assertNotIn("--live", zsh_list)
        bash_auth_status = next(
            line
            for line in bash.splitlines()
            if line.strip().startswith("auth::status)")
        )
        zsh_auth_status = next(
            line
            for line in zsh.splitlines()
            if line.strip().startswith("auth::status)")
        )
        for line in (bash_auth_status, zsh_auth_status):
            self.assertNotIn("--user", line)
            self.assertNotIn("--ttl-hours", line)

        fish_provider_list = [
            line
            for line in fish.splitlines()
            if "__xscientist_word_is 2 provider" in line
            and "__xscientist_word_is 3 list" in line
        ]
        fish_auth_status = [
            line
            for line in fish.splitlines()
            if "__xscientist_word_is 2 auth" in line
            and "__xscientist_word_is 3 status" in line
        ]
        self.assertTrue(fish_provider_list)
        self.assertTrue(fish_auth_status)
        self.assertFalse(any("-l live" in line for line in fish_provider_list))
        self.assertFalse(any("-l user" in line for line in fish_auth_status))
        self.assertFalse(any("-l ttl-hours" in line for line in fish_auth_status))

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
