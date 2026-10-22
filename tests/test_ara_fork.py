"""Tests for ARA fork CLI + re-execution verifier."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.apps import ara as ara_app
from ai_scientist.utils.ara_artifact import export_ara
from ai_scientist.utils.ara_reexec import (
    reexec_ara,
    reexec_enabled,
    reexec_node,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FORK_COMMAND = [sys.executable, "-m", "ai_scientist.apps.ara"]


def _load_fork_module():
    return ara_app


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    stage_dir = logs_dir / run_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _seed_project(tmp: Path, *, metric_value: float = 0.42) -> tuple[Path, Path, str]:
    project = tmp / "project"
    exp = project / "02_experiments" / "20260701_idea"
    (exp / "logs" / "0-run").mkdir(parents=True)
    code = textwrap.dedent(f"""
        import json
        result = {{"name": "acc", "value": {metric_value}}}
        print("ARA_METRIC=" + json.dumps(result))
        """).strip()
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "n1",
                "step": 0,
                "code": code,
                "_term_out": [
                    f'ARA_METRIC={{"name": "acc", "value": {metric_value}}}\n'
                ],
                "metric": {
                    "value": metric_value,
                    "maximize": True,
                    "name": "acc",
                    "description": "",
                },
                "is_buggy": False,
                "parent_id": None,
                "children": [],
            },
        ],
    )
    result = export_ara(
        project_dir=project,
        exp_dir=exp,
        idea={"Name": "idea"},
        timestamp="20260701",
    )
    return project, result.root, "n1"


class RunBundleTest(unittest.TestCase):
    def test_bundle_files_present_and_run_sh_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, ara_root, node_id = _seed_project(Path(tmp))
            node_dir = ara_root / "nodes" / node_id
            self.assertTrue((node_dir / "code.py").exists())
            self.assertTrue((node_dir / "env.json").exists())
            run_sh = node_dir / "run.sh"
            self.assertTrue(run_sh.exists())
            # POSIX permission bit set (best-effort check on unix).
            if os.name == "posix":
                self.assertTrue(bool(run_sh.stat().st_mode & stat.S_IXUSR))
            env = json.loads((node_dir / "env.json").read_text())
            self.assertEqual(env["node_id"], node_id)
            self.assertEqual(env["code_file"], "code.py")


class ForkCLITest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_inspect_prints_key_fields(self) -> None:
        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "inspect",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Node " + self.node_id, completed.stdout)
        self.assertIn("Metric:", completed.stdout)

    def test_node_relationships_ignore_mirrors_under_edges_encoding(self) -> None:
        graph = {
            "topology_encoding": "edges",
            "nodes": [
                {"id": "root", "parent_id": "child", "children": []},
                {"id": "child", "parent_id": None, "children": ["root"]},
            ],
            "edges": [{"parent": "root", "child": "child"}],
        }

        parents, children = ara_app._node_relationships(graph, "child")

        self.assertEqual(parents, ["root"])
        self.assertEqual(children, [])

    def test_exec_re_runs_and_writes_verify_report(self) -> None:
        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "exec",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        verify_dir = self.ara_root / "verify"
        self.assertTrue(verify_dir.exists())
        reports = list(verify_dir.glob(f"{self.node_id}_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        self.assertEqual(payload["node_id"], self.node_id)
        self.assertTrue(payload["comparison"]["within_tolerance"])

    def test_exec_preserves_two_immediate_verify_reports(self) -> None:
        command = [
            *FORK_COMMAND,
            "exec",
            "--ara",
            str(self.ara_root),
            "--node-id",
            self.node_id,
        ]

        first = subprocess.run(command, capture_output=True, text=True)
        second = subprocess.run(command, capture_output=True, text=True)

        self.assertEqual(first.returncode, 0, msg=first.stderr)
        self.assertEqual(second.returncode, 0, msg=second.stderr)
        reports = list((self.ara_root / "verify").glob(f"{self.node_id}_*.json"))
        self.assertEqual(len(reports), 2)
        self.assertNotEqual(reports[0].name, reports[1].name)

    def test_exec_rejects_invalid_resource_options_without_a_report(self) -> None:
        for flag, value in (("--timeout", "0"), ("--tolerance", "inf")):
            with self.subTest(flag=flag, value=value):
                completed = subprocess.run(
                    [
                        *FORK_COMMAND,
                        "exec",
                        "--ara",
                        str(self.ara_root),
                        "--node-id",
                        self.node_id,
                        flag,
                        value,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 9)
                self.assertIn("refused", completed.stderr)
        self.assertFalse((self.ara_root / "verify").exists())

    def test_exec_uses_stdout_metric_and_ignores_stderr_marker(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import sys

                print('ARA_METRIC={"name": "acc", "value": 0.99}')
                print('ARA_METRIC={"name": "acc", "value": 0.42}', file=sys.stderr)
                """).strip(),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "exec",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
            ],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 6, msg=completed.stderr)
        report_path = next((self.ara_root / "verify").glob(f"{self.node_id}_*.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["fresh_metric"]["value"], 0.99)
        self.assertFalse(report["comparison"]["within_tolerance"])

    def test_exec_records_context_compilation_failure_and_still_replays(self) -> None:
        with patch(
            "ai_scientist.utils.ara_context.compile_context_pack",
            side_effect=RuntimeError("context unavailable"),
        ):
            returncode = ara_app.main(
                [
                    "exec",
                    "--ara",
                    str(self.ara_root),
                    "--node-id",
                    self.node_id,
                ]
            )

        self.assertEqual(returncode, 0)
        report_path = next((self.ara_root / "verify").glob(f"{self.node_id}_*.json"))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["context_error"], "context unavailable")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_exec_rejects_node_directory_symlink_escape(self) -> None:
        outside = self.tmp / "outside-node"
        outside.mkdir()
        marker = self.tmp / "executed.txt"
        (outside / "code.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n",
            encoding="utf-8",
        )
        (outside / "metrics.json").write_text("{}", encoding="utf-8")
        (self.ara_root / "nodes" / "escape").symlink_to(
            outside, target_is_directory=True
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "exec",
                "--ara",
                str(self.ara_root),
                "--node-id",
                "escape",
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(marker.exists())

    def test_exec_rejects_unindexed_node_bundle(self) -> None:
        orphan = self.ara_root / "nodes" / "orphan"
        shutil.copytree(self.ara_root / "nodes" / self.node_id, orphan)
        marker = self.tmp / "orphan-executed.txt"
        (orphan / "code.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "exec",
                "--ara",
                str(self.ara_root),
                "--node-id",
                "orphan",
            ],
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(marker.exists())

    def test_fork_copies_node_bundle(self) -> None:
        dest = self.tmp / "forked"
        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "fork",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
                "--dest",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("forked node", completed.stdout)
        # New (O6) layout: fork is itself a conformant ARA, with the node
        # under nodes/<id>/ rather than the legacy top-level `node/`.
        self.assertTrue((dest / "nodes" / self.node_id / "code.py").exists())
        self.assertTrue((dest / "manifest.json").exists())
        self.assertTrue((dest / "exploration_graph.json").exists())
        self.assertTrue((dest / "fork.json").exists())

    def test_forked_dir_validates_as_ara(self) -> None:
        """Regression: forked directories must round-trip through validate_ara."""
        from ai_scientist.protocol import validate_ara

        dest = self.tmp / "forked_conformant"
        subprocess.run(
            [
                *FORK_COMMAND,
                "fork",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
                "--dest",
                str(dest),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        report = validate_ara(dest)
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])

    def test_fork_of_fork_preserves_provenance_chain(self) -> None:
        """A fork of a fork should chain provenance back to the original."""
        import json as _json

        intermediate = self.tmp / "fork_1"
        subprocess.run(
            [
                *FORK_COMMAND,
                "fork",
                "--ara",
                str(self.ara_root),
                "--node-id",
                self.node_id,
                "--dest",
                str(intermediate),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Second fork uses the same node_id (only node in the intermediate ARA).
        grand = self.tmp / "fork_2"
        subprocess.run(
            [
                *FORK_COMMAND,
                "fork",
                "--ara",
                str(intermediate),
                "--node-id",
                self.node_id,
                "--dest",
                str(grand),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        grand_manifest = _json.loads((grand / "manifest.json").read_text())
        # macOS /private path resolution: compare via realpath.
        self.assertEqual(
            os.path.realpath(grand / grand_manifest["provenance"]["parent_ara_root"]),
            os.path.realpath(str(intermediate)),
        )
        # Grand's provenance.parent_content_hash points at the INTERMEDIATE
        # fork's OWN content_hash (which is is_seed-bound), not at the
        # original's hash. This is the correct chain shape after the seed
        # binding fix — each fork has its own distinct hash and its
        # provenance points one step back.
        intermediate_graph = _json.loads(
            (intermediate / "exploration_graph.json").read_text()
        )
        intermediate_own_hash = intermediate_graph["nodes"][0]["content_hash"]
        self.assertEqual(
            grand_manifest["provenance"]["parent_content_hash"],
            intermediate_own_hash,
        )


class VerifyCLITest(unittest.TestCase):
    """The CLI `verify` subcommand — batch re-execution wrapper."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _copy_indexed_node(self, node_id: str) -> Path:
        shutil.copytree(
            self.ara_root / "nodes" / self.node_id,
            self.ara_root / "nodes" / node_id,
        )
        graph_path = self.ara_root / "exploration_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        source = next(node for node in graph["nodes"] if node["id"] == self.node_id)
        graph["nodes"].append({**source, "id": node_id})
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
        return self.ara_root / "nodes" / node_id

    def test_verify_batch_reports_and_zero_exit(self) -> None:
        completed = subprocess.run(
            [*FORK_COMMAND, "verify", "--ara", str(self.ara_root), "--limit", "1"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        # Batch report lands under verify/.
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        self.assertEqual(payload["schema"], "ara.reexec.batch.v1")
        self.assertGreaterEqual(payload["verdict_count"], 1)

    def test_verify_explicit_node_ids(self) -> None:
        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["node_ids"], [self.node_id])

    def test_verify_rejects_invalid_resource_options_without_a_report(self) -> None:
        for flag, value in (
            ("--limit", "-1"),
            ("--timeout", "0"),
            ("--tolerance", "inf"),
            ("--tolerance", "-0.1"),
        ):
            with self.subTest(flag=flag, value=value):
                completed = subprocess.run(
                    [
                        *FORK_COMMAND,
                        "verify",
                        "--ara",
                        str(self.ara_root),
                        flag,
                        value,
                    ],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 9)
                self.assertIn("refused", completed.stderr)
                self.assertNotIn("Traceback", completed.stderr)
        self.assertFalse((self.ara_root / "verify").exists())

    def test_verify_cli_rejects_unindexed_node_without_traceback(self) -> None:
        orphan = self.ara_root / "nodes" / "orphan"
        shutil.copytree(self.ara_root / "nodes" / self.node_id, orphan)

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                "orphan",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 9)
        self.assertIn("not indexed", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_verify_metric_drift_resolves_report_from_ara_root(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            'print(\'ARA_METRIC={"name": "acc", "value": 0.99}\')\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)

    def test_verify_rejects_matching_metric_followed_by_nonzero_exit(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import json

                print("ARA_METRIC=" + json.dumps({"name": "acc", "value": 0.42}))
                raise RuntimeError("failure after metric output")
                """).strip(),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        self.assertEqual(payload["passed"], 0)
        self.assertNotEqual(payload["verdicts"][0]["returncode"], 0)
        self.assertTrue(payload["verdicts"][0]["comparison"]["within_tolerance"])

    def test_verify_allow_drift_still_requires_a_comparable_metric(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text('print("completed without a metric")\n', encoding="utf-8")

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)

    def test_verify_allow_drift_accepts_an_actual_metric_comparison(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            'print(\'ARA_METRIC={"name": "acc", "value": 0.99}\')\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, msg=completed.stderr)

    def test_verify_batch_requires_every_selected_node_to_compare(self) -> None:
        second_id = "n2"
        second_code = self._copy_indexed_node(second_id) / "code.py"
        second_code.write_text(
            'print(\'ARA_METRIC={"name": "acc", "value": 0.99}\')\n',
            encoding="utf-8",
        )

        base_command = [
            *FORK_COMMAND,
            "verify",
            "--ara",
            str(self.ara_root),
            "--node-ids",
            self.node_id,
            second_id,
        ]
        drift = subprocess.run(
            base_command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        allowed_drift = subprocess.run(
            [*base_command, "--allow-drift"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        second_code.write_text('print("no metric")\n', encoding="utf-8")
        missing = subprocess.run(
            [*base_command, "--allow-drift"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(drift.returncode, 10, msg=drift.stderr)
        self.assertEqual(allowed_drift.returncode, 0, msg=allowed_drift.stderr)
        self.assertEqual(missing.returncode, 10, msg=missing.stderr)

    def test_verify_batch_snapshots_every_node_before_execution(self) -> None:
        second_id = "n2"
        second_dir = self._copy_indexed_node(second_id)
        first_code = self.ara_root / "nodes" / self.node_id / "code.py"
        second_metrics = second_dir / "metrics.json"
        first_code.write_text(
            textwrap.dedent(f"""
                import json
                from pathlib import Path

                metrics = Path({str(second_metrics)!r})
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                payload["metric"]["value"] = 0.99
                metrics.write_text(json.dumps(payload), encoding="utf-8")
                print('ARA_METRIC={{"name": "acc", "value": 0.42}}')
                """).strip(),
            encoding="utf-8",
        )
        (second_dir / "code.py").write_text(
            'print(\'ARA_METRIC={"name": "acc", "value": 0.99}\')\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                second_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        first, second = payload["verdicts"]
        self.assertEqual(first["input_integrity"]["status"], "mutated")
        self.assertIn(
            f"nodes/{second_id}/metrics.json",
            first["input_integrity"]["changed_paths"],
        )
        self.assertEqual(second["comparison"]["recorded_value"], 0.42)
        self.assertEqual(second["comparison"]["fresh_value"], 0.99)

    def test_verify_rejects_same_value_under_a_different_metric_name(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            'print(\'ARA_METRIC={"name": "loss", "value": 0.42}\')\n',
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        comparison = payload["verdicts"][0]["comparison"]
        self.assertEqual(comparison["status"], "metric_name_mismatch")

    def test_verify_never_reads_metric_markers_from_stderr(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import sys

                print('ARA_METRIC={"name": "acc", "value": 0.99}')
                print('ARA_METRIC={"name": "acc", "value": 0.42}', file=sys.stderr)
                """).strip(),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        verdict = payload["verdicts"][0]
        self.assertEqual(verdict["fresh_metric"]["value"], 0.99)
        self.assertEqual(verdict["comparison"]["status"], "compared")
        self.assertFalse(verdict["comparison"]["within_tolerance"])

    def test_verify_snapshots_recorded_metric_and_rejects_input_mutation(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import json
                from pathlib import Path

                metrics = Path(__file__).with_name("metrics.json")
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                payload["metric"]["value"] = 0.99
                metrics.write_text(json.dumps(payload), encoding="utf-8")
                print('ARA_METRIC={"name": "acc", "value": 0.99}')
                """).strip(),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                *FORK_COMMAND,
                "verify",
                "--ara",
                str(self.ara_root),
                "--node-ids",
                self.node_id,
                "--allow-drift",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 10, msg=completed.stderr)
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        verdict = payload["verdicts"][0]
        self.assertEqual(verdict["comparison"]["recorded_value"], 0.42)
        self.assertEqual(verdict["comparison"]["fresh_value"], 0.99)
        self.assertEqual(verdict["input_integrity"]["status"], "mutated")
        self.assertEqual(
            verdict["input_integrity"]["changed_paths"],
            [f"nodes/{self.node_id}/metrics.json"],
        )


class ParseMetricTest(unittest.TestCase):
    def test_json_marker_wins(self) -> None:
        mod = _load_fork_module()
        stdout = 'chatter\nARA_METRIC={"name": "acc", "value": 0.9}\nmore output\n'
        result = mod._parse_metric_from_stdout(stdout)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["value"], 0.9)

    def test_text_tail_fallback(self) -> None:
        mod = _load_fork_module()
        stdout = "training log ...\nmetric: 0.5\n"
        result = mod._parse_metric_from_stdout(stdout)
        self.assertTrue(result["available"])
        self.assertAlmostEqual(result["value"], 0.5)

    def test_no_metric_returns_unavailable(self) -> None:
        mod = _load_fork_module()
        self.assertFalse(
            _load_fork_module()._parse_metric_from_stdout("nothing here")["available"]
        )

    def test_non_finite_json_metric_is_unavailable(self) -> None:
        mod = _load_fork_module()
        for token in ("NaN", "Infinity", "-Infinity"):
            result = mod._parse_metric_from_stdout(
                f'ARA_METRIC={{"name":"loss","value":{token}}}'
            )
            self.assertFalse(result["available"])

    def test_metric_identity_and_numeric_domain_must_match(self) -> None:
        from ai_scientist.utils.ara_metric_parser import compare_metrics

        recorded = {"name": "acc", "value": 0.42, "maximize": True}
        direction = compare_metrics(
            recorded,
            {"available": True, "name": "acc", "value": 0.42, "maximize": False},
            1e-3,
        )
        self.assertEqual(direction["status"], "metric_direction_mismatch")
        self.assertEqual(
            compare_metrics(recorded, {"available": True, "value": True}, 1e-3)[
                "status"
            ],
            "non_numeric_metric",
        )
        self.assertEqual(
            compare_metrics(
                recorded,
                {"available": True, "value": 10**10000},
                1e-3,
            )["status"],
            "non_numeric_metric",
        )
        self.assertEqual(
            compare_metrics(
                recorded,
                {"available": True, "value": float("inf")},
                1e-3,
            )["status"],
            "non_finite_metric",
        )

    def test_metric_comparison_preserves_integer_precision_and_finite_delta(
        self,
    ) -> None:
        from ai_scientist.utils.ara_metric_parser import compare_metrics

        for recorded, fresh in (
            (2**53, 2**53 + 1),
            (10**100, 10**100 + 1),
        ):
            with self.subTest(recorded=recorded):
                result = compare_metrics(
                    recorded, {"available": True, "value": fresh}, 0
                )
                self.assertEqual(result["status"], "compared")
                self.assertEqual(result["delta"], 1)
                self.assertFalse(result["within_tolerance"])

        wide = compare_metrics(1e308, {"available": True, "value": -1e308}, 0)
        self.assertEqual(wide["status"], "compared")
        self.assertIsInstance(wide["delta"], int)
        self.assertFalse(wide["within_tolerance"])

    def test_metric_comparison_rejects_invalid_tolerance_and_bool_direction(
        self,
    ) -> None:
        from ai_scientist.utils.ara_metric_parser import compare_metrics

        for tolerance in (-1, float("inf"), float("nan"), True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "finite non-negative"):
                    compare_metrics(1, {"available": True, "value": 1}, tolerance)

        direction = compare_metrics(
            {"value": 1, "maximize": True},
            {"available": True, "value": 1, "maximize": 1},
            0,
        )
        self.assertEqual(direction["status"], "metric_direction_mismatch")


class ReexecHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        _, self.ara_root, self.node_id = _seed_project(self.tmp)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reexec_node_returns_within_tolerance(self) -> None:
        verdict = reexec_node(self.ara_root, self.node_id, timeout=60)
        self.assertEqual(verdict["node_id"], self.node_id)
        self.assertEqual(verdict["returncode"], 0)
        self.assertEqual(verdict["input_integrity"]["status"], "clean")
        self.assertTrue(verdict["comparison"]["within_tolerance"])

    def test_reexec_node_persists_partial_text_when_process_times_out(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import time

                print('ARA_METRIC={"name": "acc", "value": 0.42}', flush=True)
                time.sleep(2)
                """).strip(),
            encoding="utf-8",
        )

        verdict = reexec_node(self.ara_root, self.node_id, timeout=0.5)

        self.assertTrue(verdict["timed_out"])
        self.assertEqual(verdict["returncode"], -1)
        self.assertEqual(verdict["fresh_metric"]["value"], 0.42)
        self.assertIn("ARA_METRIC", verdict["stdout_tail"])

    def test_reexec_node_replaces_invalid_utf8_and_keeps_valid_metric(self) -> None:
        code_path = self.ara_root / "nodes" / self.node_id / "code.py"
        code_path.write_text(
            textwrap.dedent("""
                import os

                os.write(1, b"\\xff\\n")
                os.write(1, b'ARA_METRIC={"name":"acc","value":0.42}\\n')
                """).strip(),
            encoding="utf-8",
        )

        verdict = reexec_node(self.ara_root, self.node_id, timeout=60)

        self.assertEqual(verdict["returncode"], 0)
        self.assertTrue(verdict["comparison"]["within_tolerance"])
        self.assertIn("\ufffd", verdict["stdout_tail"])

    def test_reexec_node_records_missing_interpreter_as_execution_error(self) -> None:
        verdict = reexec_node(
            self.ara_root,
            self.node_id,
            python=str(self.tmp / "missing-python"),
            timeout=60,
        )

        self.assertEqual(verdict["returncode"], -1)
        self.assertEqual(verdict["execution_error"], "FileNotFoundError")
        self.assertEqual(verdict["comparison"]["status"], "no_metric_parsed")

    def test_reexec_node_rejects_explicit_non_directory_cwd(self) -> None:
        with self.assertRaisesRegex(ValueError, "existing directory"):
            reexec_node(
                self.ara_root,
                self.node_id,
                cwd=self.ara_root / "manifest.json",
                timeout=60,
            )

    def test_reexec_node_rejects_absolute_node_id(self) -> None:
        outside = self.tmp / "outside-absolute"
        outside.mkdir()
        (outside / "code.py").write_text("raise AssertionError('executed')\n")

        with self.assertRaisesRegex(ValueError, "unsafe ARA node id"):
            reexec_node(self.ara_root, str(outside), timeout=60)

    def test_reexec_node_rejects_traversal_node_id(self) -> None:
        outside = self.ara_root.parent / "outside-traversal"
        outside.mkdir()
        (outside / "code.py").write_text("raise AssertionError('executed')\n")

        with self.assertRaisesRegex(ValueError, "unsafe ARA node id"):
            reexec_node(self.ara_root, "../../outside-traversal", timeout=60)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_reexec_node_rejects_symlink_escape(self) -> None:
        outside = self.tmp / "outside-symlink"
        outside.mkdir()
        (outside / "code.py").write_text("raise AssertionError('executed')\n")
        (outside / "metrics.json").write_text(
            json.dumps({"metric": {"name": "acc", "value": 0.42}}),
            encoding="utf-8",
        )
        (self.ara_root / "nodes" / "escape").symlink_to(
            outside, target_is_directory=True
        )

        with self.assertRaisesRegex(ValueError, "unsafe ARA node directory"):
            reexec_node(self.ara_root, "escape", timeout=60)

    def test_reexec_node_rejects_unindexed_node_bundle(self) -> None:
        orphan = self.ara_root / "nodes" / "orphan"
        shutil.copytree(self.ara_root / "nodes" / self.node_id, orphan)

        with self.assertRaisesRegex(ValueError, "not indexed"):
            reexec_node(self.ara_root, "orphan", timeout=60)

    def test_reexec_ara_rejects_unsafe_graph_node_before_selection(self) -> None:
        graph_path = self.ara_root / "exploration_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["nodes"].append(
            {
                "id": "../../outside-graph",
                "metric": {"name": "acc", "value": 1.0},
                "is_buggy": False,
            }
        )
        graph_path.write_text(json.dumps(graph), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unsafe ARA node id"):
            reexec_ara(self.ara_root, limit=2)

    def test_reexec_node_does_not_coerce_a_numeric_graph_id(self) -> None:
        graph_path = self.ara_root / "exploration_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["nodes"][0]["id"] = 1
        graph_path.write_text(json.dumps(graph), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid id"):
            reexec_node(self.ara_root, "1", timeout=60)

    def test_reexec_ara_writes_batch_report(self) -> None:
        result = reexec_ara(self.ara_root, limit=2)
        self.assertEqual(result["status"], "ok")
        report_path = self.ara_root / result["report_path"]
        self.assertTrue(report_path.exists())
        payload = json.loads(report_path.read_text())
        self.assertGreaterEqual(payload["verdict_count"], 1)

    def test_reexec_ara_preserves_two_immediate_batch_reports(self) -> None:
        first = reexec_ara(self.ara_root, limit=1)
        second = reexec_ara(self.ara_root, limit=1)

        self.assertNotEqual(first["report_path"], second["report_path"])
        reports = list((self.ara_root / "verify").glob("reexec_batch_*.json"))
        self.assertEqual(len(reports), 2)

    def test_reexec_public_api_rejects_invalid_resource_options(self) -> None:
        for kwargs in (
            {"limit": -1},
            {"limit": 0},
            {"timeout": 0},
            {"timeout": float("inf")},
            {"tolerance": -1},
            {"tolerance": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    reexec_ara(self.ara_root, **kwargs)
        self.assertFalse((self.ara_root / "verify").exists())

    def test_reexec_auto_selection_accepts_huge_exact_metric(self) -> None:
        graph_path = self.ara_root / "exploration_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        graph["nodes"][0]["metric"]["value"] = 10**1000
        graph_path.write_text(json.dumps(graph), encoding="utf-8")

        result = reexec_ara(self.ara_root, limit=1, tolerance=10**1000)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["node_ids"], [self.node_id])

    def test_reexec_enabled_respects_env(self) -> None:
        orig = os.environ.get("AI_SCIENTIST_ARA_REEXEC")
        try:
            os.environ.pop("AI_SCIENTIST_ARA_REEXEC", None)
            self.assertFalse(reexec_enabled())
            os.environ["AI_SCIENTIST_ARA_REEXEC"] = "1"
            self.assertTrue(reexec_enabled())
        finally:
            if orig is None:
                os.environ.pop("AI_SCIENTIST_ARA_REEXEC", None)
            else:
                os.environ["AI_SCIENTIST_ARA_REEXEC"] = orig


if __name__ == "__main__":
    unittest.main()
