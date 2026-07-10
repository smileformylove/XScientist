"""Smoke tests for the `show` verb in run_ara_fork.py.

The verb is a thin JSON dumper over already-tested primitives
(exploration_graph + metrics.json + plots.json + code.py + term_out.log),
so these tests just pin the output contract: shape, exit codes, and the
term-tail knob.
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
    "_run_ara_fork_cli_show",
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


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class ShowCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_show_returns_json_with_required_keys(self) -> None:
        ara = _make_ara(self.tmp, "ok", [{
            "id": "n1", "step": 0, "code": "print('hi')",
            "_term_out": ["hi\n"],
            "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        for key in ("id", "content_hash", "content_hash_inputs", "is_buggy",
                    "is_seed_node", "step", "parent_id", "children", "metric",
                    "code"):
            self.assertIn(key, payload)
        self.assertEqual(payload["id"], "n1")
        self.assertEqual(payload["code"], "print('hi')")
        self.assertFalse(payload["is_buggy"])
        self.assertIn("code", payload["content_hash_inputs"])

    def test_show_missing_node_returns_rc3(self) -> None:
        ara = _make_ara(self.tmp, "missing", [{
            "id": "n1", "step": 0, "code": "print('hi')",
            "_term_out": [],
            "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        rc, _, err = _run("show", "--ara", str(ara), "--node", "does_not_exist")
        self.assertEqual(rc, 3)
        self.assertIn("does_not_exist", err)

    def test_show_handles_node_without_code(self) -> None:
        # Buggy nodes may end up without a persisted code.py — verify no crash
        # and `code: null` in the output. We simulate the missing-code case by
        # exporting normally, then deleting the code.py off disk.
        ara = _make_ara(self.tmp, "nocode", [{
            "id": "n1", "step": 0, "code": "print('x')",
            "_term_out": [], "metric": None,
            "is_buggy": True, "exc_type": "ValueError",
            "parent_id": None, "children": [],
        }])
        (ara / "nodes" / "n1" / "code.py").unlink()
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIsNone(payload["code"])
        self.assertTrue(payload["is_buggy"])

    def test_show_term_tail_default_and_override(self) -> None:
        ara = _make_ara(self.tmp, "tail", [{
            "id": "n1", "step": 0, "code": "print('hi')",
            "_term_out": ["A" * 6000],
            "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])
        # Default (4000)
        rc, out, _ = _run("show", "--ara", str(ara), "--node", "n1")
        self.assertEqual(rc, 0)
        self.assertEqual(len(json.loads(out)["term_out_tail"]), 4000)
        self.assertEqual(json.loads(out)["term_out_size"], 6000)
        # Explicit 100
        rc, out, _ = _run("show", "--ara", str(ara), "--node", "n1",
                          "--term-tail", "100")
        self.assertEqual(rc, 0)
        self.assertEqual(len(json.loads(out)["term_out_tail"]), 100)
        # 0 → empty
        rc, out, _ = _run("show", "--ara", str(ara), "--node", "n1",
                          "--term-tail", "0")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out)["term_out_tail"], "")

    def test_show_reports_llm_call_refs(self) -> None:
        prompt_hash = "sha256:" + "b" * 64
        ara = _make_ara(self.tmp, "refs", [{
            "id": "n1", "step": 0, "code": "print('hi')",
            "_term_out": [],
            "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
            "llm_call_refs": [prompt_hash],
        }])
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["llm_call_refs"], [prompt_hash])
        self.assertIn("llm_calls", payload["content_hash_inputs"])

    def _terse_ara(self) -> Path:
        return _make_ara(self.tmp, "terse", [{
            "id": "n1", "step": 0, "code": "print('hi')\n" * 100,
            "_term_out": ["A" * 6000],
            "metric": {"value": 0.5, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        }])

    def test_show_terse_omits_code_and_term_out_tail(self) -> None:
        ara = self._terse_ara()
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1", "--terse")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        for key in ("id", "content_hash", "metric", "step", "children"):
            self.assertIn(key, payload)
        self.assertNotIn("code", payload)
        self.assertNotIn("term_out_tail", payload)

    def test_show_terse_preserves_term_out_size(self) -> None:
        ara = self._terse_ara()
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1", "--terse")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(payload["term_out_size"], 6000)

    def test_show_default_still_includes_code(self) -> None:
        ara = self._terse_ara()
        rc, out, err = _run("show", "--ara", str(ara), "--node", "n1")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIn("code", payload)
        self.assertIn("term_out_tail", payload)
        self.assertTrue(payload["code"].startswith("print('hi')"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
