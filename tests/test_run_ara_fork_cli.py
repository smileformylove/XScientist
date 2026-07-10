"""Smoke tests for the `describe` verb in run_ara_fork.py.

The engine-level pieces (ara_log, verify_manifest_lock, hash_manifest)
are covered elsewhere. This file exercises the CLI glue: field
selection, top-metric direction, null-safe handling, and ancestry
summary aggregation.
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_run_ara_fork_cli_describe",
    Path(__file__).resolve().parent.parent / "run_ara_fork.py",
)
run_ara_fork = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(run_ara_fork)  # type: ignore[union-attr]

from ai_scientist.utils.ara_artifact import export_ara


def _write_journal(logs_dir: Path, nodes: list[dict]) -> None:
    stage_dir = logs_dir / "0-run"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _node(nid: str, *, value: float | None = 0.5, maximize: bool = True,
          is_buggy: bool = False, code: str | None = None) -> dict:
    metric = None if value is None else {
        "value": value, "maximize": maximize, "name": "acc",
    }
    return {
        "id": nid, "step": 0,
        "code": code if code is not None else f"print({nid!r})",
        "_term_out": [],
        "metric": metric,
        "is_buggy": is_buggy, "parent_id": None, "children": [],
    }


def _make_ara(tmp: Path, sub: str, nodes: list[dict],
              provenance: dict | None = None) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(
        project_dir=project, exp_dir=exp, idea={"Name": sub},
        provenance=provenance,
    )
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class DescribeCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_describe_human_output_contains_key_fields(self) -> None:
        a = _make_ara(self.tmp, "human", [
            _node("nA", value=0.7, maximize=True, is_buggy=False),
            _node("nB", value=0.9, maximize=True, is_buggy=True),
        ])
        rc, out, _ = _run("describe", "--ara", str(a))
        self.assertEqual(rc, 0)
        self.assertIn("Idea", out)
        self.assertIn("Nodes:", out)
        self.assertIn("buggy: 1", out)
        self.assertIn("Top metric", out)
        # The non-buggy node id must show up in the top-metric row.
        self.assertIn("nA", out)

    def test_describe_json_output_shape(self) -> None:
        a = _make_ara(self.tmp, "json", [_node("nA")])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        for key in ("ara_root", "idea", "counts", "top_metric_node",
                    "lock", "verify_state", "ancestors"):
            self.assertIn(key, payload)
        self.assertIn("nodes", payload["counts"])
        self.assertIn("buggy", payload["counts"])
        self.assertIn("state", payload["lock"])

    def test_describe_top_metric_picks_highest_when_maximize(self) -> None:
        a = _make_ara(self.tmp, "hi", [
            _node("nLo", value=0.5, maximize=True),
            _node("nHi", value=0.9, maximize=True),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["top_metric_node"]["id"], "nHi")

    def test_describe_top_metric_picks_lowest_when_minimize(self) -> None:
        a = _make_ara(self.tmp, "lo", [
            _node("nLo", value=0.1, maximize=False),
            _node("nHi", value=0.9, maximize=False),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["top_metric_node"]["id"], "nLo")

    def test_describe_no_scored_nodes_reports_null(self) -> None:
        a = _make_ara(self.tmp, "empty", [
            _node("nBad", value=0.9, is_buggy=True),
            _node("nNone", value=None),
        ])
        rc, out, _ = _run("describe", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsNone(payload["top_metric_node"])

    def test_describe_ancestors_summary_when_parent_reachable(self) -> None:
        parent = _make_ara(self.tmp, "parent", [_node("np", value=0.5)])
        parent_graph = json.loads((parent / "exploration_graph.json").read_text())
        parent_node_hash = parent_graph["nodes"][0]["content_hash"]

        child = _make_ara(
            self.tmp, "child", [_node("nc", value=0.6)],
            provenance={
                "parent_ara_root": str(parent),
                "parent_node_id": "np",
                "parent_content_hash": parent_node_hash,
            },
        )
        rc, out, _ = _run("describe", "--ara", str(child), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload["ancestors"]["count"], 1)
        self.assertTrue(payload["ancestors"]["all_reachable"])


class VerifyLockAllTests(unittest.TestCase):
    """`verify-lock --all --project <path>` sweeps <project>/ara/.

    Aggregate rc rule: tampered > unlocked > ok. Empty projects are ok.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _project_with(self, name: str, count: int) -> Path:
        project = self.tmp / name
        for i in range(count):
            sub = f"{name}_ara{i}"
            exp = project / "02_experiments" / f"20260710_{sub}"
            _write_journal(exp / "logs", [_node(f"n{i}")])
            (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
            export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
        return project

    @staticmethod
    def _tamper(ara: Path) -> None:
        m = ara / "manifest.json"
        p = json.loads(m.read_text(encoding="utf-8"))
        p["__tamper__"] = "edit"
        m.write_text(json.dumps(p, indent=2), encoding="utf-8")

    def test_verify_lock_all_walks_multiple_aras(self) -> None:
        project = self._project_with("multi", 3)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertEqual(sum(1 for line in out.splitlines() if line.strip()), 4)
        self.assertEqual(out.count("clean"), 3)

    def test_verify_lock_all_reports_tampered(self) -> None:
        project = self._project_with("tamp", 2)
        target = sorted((project / "ara").iterdir())[0]
        self._tamper(target)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 2)
        self.assertIn("tampered", out)
        self.assertIn(target.name, out)

    def test_verify_lock_all_reports_unlocked(self) -> None:
        project = self._project_with("unlk", 2)
        (sorted((project / "ara").iterdir())[0] / "manifest.lock").unlink()
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 3)
        self.assertIn("unlocked", out)

    def test_verify_lock_all_tampered_beats_unlocked_in_rc(self) -> None:
        project = self._project_with("mix", 3)
        aras = sorted((project / "ara").iterdir())
        self._tamper(aras[1])
        (aras[2] / "manifest.lock").unlink()
        rc, _, _ = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 2)

    def test_verify_lock_all_empty_project_returns_rc_zero(self) -> None:
        project = self.tmp / "empty_project"
        project.mkdir()
        rc, out, err = _run("verify-lock", "--all", "--project", str(project))
        self.assertEqual(rc, 0)
        self.assertIn("no ARAs", err)
        self.assertEqual(out, "")

    def test_verify_lock_all_json_shape(self) -> None:
        project = self._project_with("json", 3)
        rc, out, _ = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload), 3)
        for entry in payload:
            for key in ("ara_root", "state", "revision_count",
                        "manifest_hash", "detail"):
                self.assertIn(key, entry)
            self.assertEqual(entry["state"], "clean")

    def test_verify_lock_all_json_empty_project_emits_empty_array(self) -> None:
        """`--json` on an empty project must emit `[]` on stdout so CI pipes
        (`... --json | jq`) don't choke on empty input. The human note stays
        on stderr — same invariant every other verb honors."""
        project = self.tmp / "empty_json"
        project.mkdir()
        rc, out, err = _run("verify-lock", "--all", "--project", str(project), "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "[]")
        self.assertEqual(json.loads(out), [])
        self.assertIn("no ARAs", err)

    def test_verify_lock_all_and_ara_mutually_exclusive(self) -> None:
        project = self._project_with("mx", 1)
        ara = next(iter((project / "ara").iterdir()))
        with self.assertRaises(SystemExit) as ctx:
            _run("verify-lock", "--ara", str(ara),
                 "--all", "--project", str(project))
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
