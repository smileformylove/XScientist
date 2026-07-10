"""Smoke tests for the `hash-check` verb in run_ara_fork.py.

The verb recomputes each node's content_hash from disk using the same
binding rule ``ara_artifact._export_nodes_from_journal`` used at write
time, then reports drift against the stored hash. These tests pin exit
codes, state classification, and the JSON shape.
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
    "_run_ara_fork_cli_hash_check",
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


def _make_ara(tmp: Path, sub: str, nodes: list[dict]) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
    return Path(result.root)


def _make_ara_in(project: Path, idea_name: str, nodes: list[dict]) -> Path:
    """Build an ARA under an existing project directory (multi-ARA sweep fixture)."""
    exp = project / "02_experiments" / f"20260710_{idea_name}"
    _write_journal(exp / "logs", nodes)
    (exp / "idea.json").write_text(json.dumps({"Name": idea_name}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": idea_name})
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def _default_node(nid: str, *, code: str = "print('ok')",
                  metric_val: float = 0.5) -> dict:
    return {
        "id": nid, "step": 0, "code": code,
        "_term_out": [],
        "metric": {"value": metric_val, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }


class HashCheckCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_hash_check_clean_ara_returns_zero(self) -> None:
        ara = _make_ara(self.tmp, "clean", [
            _default_node("n1"), _default_node("n2", metric_val=0.7),
        ])
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual({e["state"] for e in payload}, {"clean"})
        for e in payload:
            self.assertEqual(e["stored_hash"], e["computed_hash"])

    def test_hash_check_detects_code_drift(self) -> None:
        ara = _make_ara(self.tmp, "code_drift", [_default_node("n1")])
        # Mutate the code on disk — stored hash won't match anymore.
        (ara / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "drift")
        self.assertIn("code", entry.get("notes", "").lower())

    def test_hash_check_detects_metric_drift(self) -> None:
        ara = _make_ara(self.tmp, "metric_drift", [_default_node("n1")])
        # Edit metrics.json to change the value — stored hash no longer matches.
        mp = ara / "nodes" / "n1" / "metrics.json"
        m = json.loads(mp.read_text(encoding="utf-8"))
        m["metric"] = {"value": 0.999, "maximize": True, "name": "acc"}
        mp.write_text(json.dumps(m), encoding="utf-8")
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "drift")
        self.assertIn("metric", entry.get("notes", "").lower())

    def test_hash_check_missing_code_returns_rc2(self) -> None:
        ara = _make_ara(self.tmp, "no_code", [_default_node("n1")])
        (ara / "nodes" / "n1" / "code.py").unlink()
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 2, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "missing_code")
        self.assertIsNone(entry["computed_hash"])

    def test_hash_check_empty_code_node_reports_clean(self) -> None:
        # A journal node with empty code — export skips writing code.py but
        # still stamps a content_hash computed with code="". hash-check must
        # recognise the empty-code recompute matches the stored hash and
        # report clean, not a false-positive missing_code.
        ara = _make_ara(self.tmp, "empty_code", [_default_node("n1", code="")])
        self.assertFalse((ara / "nodes" / "n1" / "code.py").exists())
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        [entry] = [e for e in payload if e["node_id"] == "n1"]
        self.assertEqual(entry["state"], "clean")
        self.assertEqual(entry["stored_hash"], entry["computed_hash"])

    def test_hash_check_drift_beats_missing_code(self) -> None:
        ara = _make_ara(self.tmp, "drift_wins", [
            _default_node("n1"), _default_node("n2"),
        ])
        # n1 gets code drift; n2 gets its code deleted.
        (ara / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        (ara / "nodes" / "n2" / "code.py").unlink()
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 1, msg=err)  # drift wins over missing_code
        payload = json.loads(out)
        by_id = {e["node_id"]: e for e in payload}
        self.assertEqual(by_id["n1"]["state"], "drift")
        self.assertEqual(by_id["n2"]["state"], "missing_code")

    def test_hash_check_json_shape(self) -> None:
        ara = _make_ara(self.tmp, "shape", [_default_node("n1")])
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        [entry] = payload
        for key in ("node_id", "state", "stored_hash", "computed_hash"):
            self.assertIn(key, entry)

    def test_hash_check_unhashed_node_is_not_a_failure(self) -> None:
        ara = _make_ara(self.tmp, "legacy", [_default_node("n1")])
        # Simulate a legacy graph entry by stripping the stored hash from
        # exploration_graph.json (metrics.json still has it, but the CLI
        # reads content_hash from the graph entry — matching cmd_show).
        gp = ara / "exploration_graph.json"
        g = json.loads(gp.read_text(encoding="utf-8"))
        for n in g["nodes"]:
            n.pop("content_hash", None)
        gp.write_text(json.dumps(g), encoding="utf-8")
        rc, out, err = _run("hash-check", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        [entry] = payload
        self.assertEqual(entry["state"], "unhashed")
        self.assertIsNone(entry["stored_hash"])
        self.assertIsNotNone(entry["computed_hash"])

    def test_hash_check_human_output_prints_table(self) -> None:
        ara = _make_ara(self.tmp, "table", [_default_node("n1")])
        rc, out, err = _run("hash-check", "--ara", str(ara))
        self.assertEqual(rc, 0, msg=err)
        lines = [ln for ln in out.splitlines() if ln.strip()]
        self.assertTrue(lines[0].startswith("NODE"))
        # Truncated hash form uses the ellipsis character.
        self.assertIn("n1", out)


class HashCheckAllCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.project = self.tmp / "proj"
        self.project.mkdir()

    def test_hash_check_all_walks_multiple_aras(self) -> None:
        _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        _make_ara_in(self.project, "idea_b", [_default_node("n1"), _default_node("n2")])
        _make_ara_in(self.project, "idea_c", [_default_node("n1")])
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(len(payload["aras"]), 3)
        self.assertTrue(all(a["state"] == "clean" for a in payload["aras"]))
        self.assertEqual(payload["totals"]["aras"], 3)
        self.assertEqual(payload["totals"]["nodes"], 4)

    def test_hash_check_all_reports_drift(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        bad = _make_ara_in(self.project, "drift_idea", [_default_node("n1")])
        (bad / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 1, msg=err)
        payload = json.loads(out)
        by_name = {Path(a["ara_root"]).name: a for a in payload["aras"]}
        drift_key = next(k for k in by_name if "drift_idea" in k)
        clean_key = next(k for k in by_name if "clean_idea" in k)
        self.assertEqual(by_name[drift_key]["state"], "drift")
        self.assertEqual(by_name[clean_key]["state"], "clean")
        self.assertEqual(payload["totals"]["drift"], 1)

    def test_hash_check_all_reports_missing_code(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        bad = _make_ara_in(self.project, "gone_idea", [_default_node("n1")])
        (bad / "nodes" / "n1" / "code.py").unlink()
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 2, msg=err)
        payload = json.loads(out)
        by_name = {Path(a["ara_root"]).name: a for a in payload["aras"]}
        gone_key = next(k for k in by_name if "gone_idea" in k)
        self.assertEqual(by_name[gone_key]["state"], "missing_code")
        self.assertEqual(payload["totals"]["missing_code"], 1)

    def test_hash_check_all_drift_beats_missing_code_in_rc(self) -> None:
        _make_ara_in(self.project, "clean_idea", [_default_node("n1")])
        drift = _make_ara_in(self.project, "drift_idea", [_default_node("n1")])
        (drift / "nodes" / "n1" / "code.py").write_text(
            "print('tampered')\n", encoding="utf-8"
        )
        gone = _make_ara_in(self.project, "gone_idea", [_default_node("n1")])
        (gone / "nodes" / "n1" / "code.py").unlink()
        rc, _out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 1, msg=err)  # drift wins over missing_code

    def test_hash_check_all_empty_project_returns_rc_zero(self) -> None:
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["aras"], [])
        self.assertIn("no ARAs found", err)

    def test_hash_check_all_json_shape(self) -> None:
        _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        rc, out, err = _run(
            "hash-check", "--all", "--project", str(self.project), "--json"
        )
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIn("aras", payload)
        self.assertIn("totals", payload)
        self.assertIsInstance(payload["aras"], list)
        self.assertIsInstance(payload["totals"], dict)
        [entry] = payload["aras"]
        for key in ("ara_root", "nodes", "counts", "state"):
            self.assertIn(key, entry)

    def test_hash_check_all_and_ara_mutually_exclusive(self) -> None:
        ara = _make_ara_in(self.project, "idea_a", [_default_node("n1")])
        with self.assertRaises(SystemExit):
            _run(
                "hash-check", "--ara", str(ara),
                "--all", "--project", str(self.project),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
