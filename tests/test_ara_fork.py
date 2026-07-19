"""Tests for ARA fork CLI + re-execution verifier."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

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
    code = textwrap.dedent(
        f"""
        import json
        result = {{"name": "acc", "value": {metric_value}}}
        print("ARA_METRIC=" + json.dumps(result))
        """
    ).strip()
    _write_journal(
        exp / "logs",
        "0-run",
        [
            {
                "id": "n1",
                "step": 0,
                "code": code,
                "_term_out": [f"ARA_METRIC={{\"name\": \"acc\", \"value\": {metric_value}}}\n"],
                "metric": {"value": metric_value, "maximize": True, "name": "acc", "description": ""},
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
            [*FORK_COMMAND, "inspect", "--ara", str(self.ara_root), "--node-id", self.node_id],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("Node " + self.node_id, completed.stdout)
        self.assertIn("Metric:", completed.stdout)

    def test_exec_re_runs_and_writes_verify_report(self) -> None:
        completed = subprocess.run(
            [*FORK_COMMAND, "exec", "--ara", str(self.ara_root), "--node-id", self.node_id],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        verify_dir = self.ara_root / "verify"
        self.assertTrue(verify_dir.exists())
        reports = list(verify_dir.glob(f"{self.node_id}_*.json"))
        self.assertEqual(len(reports), 1)
        payload = json.loads(reports[0].read_text())
        self.assertEqual(payload["node_id"], self.node_id)
        self.assertTrue(payload["comparison"]["within_tolerance"])

    def test_fork_copies_node_bundle(self) -> None:
        dest = self.tmp / "forked"
        completed = subprocess.run(
            [*FORK_COMMAND, "fork", "--ara", str(self.ara_root),
             "--node-id", self.node_id, "--dest", str(dest)],
            capture_output=True, text=True, check=True,
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
            [*FORK_COMMAND, "fork", "--ara", str(self.ara_root),
             "--node-id", self.node_id, "--dest", str(dest)],
            capture_output=True, text=True, check=True,
        )
        report = validate_ara(dest)
        self.assertTrue(report.ok, msg=[e.__dict__ for e in report.errors])

    def test_fork_of_fork_preserves_provenance_chain(self) -> None:
        """A fork of a fork should chain provenance back to the original."""
        import json as _json

        intermediate = self.tmp / "fork_1"
        subprocess.run(
            [*FORK_COMMAND, "fork", "--ara", str(self.ara_root),
             "--node-id", self.node_id, "--dest", str(intermediate)],
            capture_output=True, text=True, check=True,
        )
        # Second fork uses the same node_id (only node in the intermediate ARA).
        grand = self.tmp / "fork_2"
        subprocess.run(
            [*FORK_COMMAND, "fork", "--ara", str(intermediate),
             "--node-id", self.node_id, "--dest", str(grand)],
            capture_output=True, text=True, check=True,
        )
        grand_manifest = _json.loads((grand / "manifest.json").read_text())
        # macOS /private path resolution: compare via realpath.
        self.assertEqual(
            os.path.realpath(grand_manifest["provenance"]["parent_ara_root"]),
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

    def test_verify_batch_reports_and_zero_exit(self) -> None:
        completed = subprocess.run(
            [*FORK_COMMAND, "verify",
             "--ara", str(self.ara_root), "--limit", "1"],
            capture_output=True, text=True,
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
            [*FORK_COMMAND, "verify",
             "--ara", str(self.ara_root), "--node-ids", self.node_id],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["node_ids"], [self.node_id])


class ParseMetricTest(unittest.TestCase):
    def test_json_marker_wins(self) -> None:
        mod = _load_fork_module()
        stdout = "chatter\nARA_METRIC={\"name\": \"acc\", \"value\": 0.9}\nmore output\n"
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
        self.assertFalse(_load_fork_module()._parse_metric_from_stdout("nothing here")["available"])


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
        self.assertTrue(verdict["comparison"]["within_tolerance"])

    def test_reexec_ara_writes_batch_report(self) -> None:
        result = reexec_ara(self.ara_root, limit=2)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(Path(result["report_path"]).exists())
        payload = json.loads(Path(result["report_path"]).read_text())
        self.assertGreaterEqual(payload["verdict_count"], 1)

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
