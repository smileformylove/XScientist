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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
