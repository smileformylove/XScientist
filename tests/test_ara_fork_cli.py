"""Smoke tests for the diff/log/refs CLI verbs in run_ara_fork.py.

Verifies that the CLI wraps its engines correctly — engine correctness is
tested in test_ara_diff.py / test_ara_log.py / test_ara_refs.py.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Load run_ara_fork.py as a module. It's a top-level script, so we do this
# once via importlib to keep imports clean across tests.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "_run_ara_fork_cli",
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


def _make_ara(tmp: Path, sub: str, code: str = "print('ok')") -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260709_{sub}"
    _write_journal(exp / "logs", [{
        "id": "n1", "step": 0, "code": code,
        "_term_out": [],
        "metric": {"value": 0.5, "maximize": True, "name": "acc"},
        "is_buggy": False, "parent_id": None, "children": [],
    }])
    (exp / "idea.json").write_text(json.dumps({"Name": sub}), encoding="utf-8")
    result = export_ara(project_dir=project, exp_dir=exp, idea={"Name": sub})
    return Path(result.root)


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class CLIDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_diff_json_output_has_expected_keys(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, out, err = _run("diff", "--ara", str(a), "--other", str(b), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("manifest", payload)
        self.assertIn("nodes_hash_changed", payload)
        self.assertEqual(len(payload["nodes_hash_changed"]), 1)
        cats = payload["nodes_hash_changed"][0]["changed_categories"]
        self.assertIn("code", cats)

    def test_diff_exit_code_on_diff(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, _, _ = _run("diff", "--ara", str(a), "--other", str(b),
                        "--exit-code-on-diff")
        self.assertEqual(rc, 1)

    def test_diff_human_output_has_headers(self) -> None:
        a = _make_ara(self.tmp, "a", code="print(1)")
        b = _make_ara(self.tmp, "b", code="print(2)")
        rc, out, _ = _run("diff", "--ara", str(a), "--other", str(b))
        self.assertEqual(rc, 0)
        self.assertIn("## manifest", out)
        self.assertIn("## nodes", out)


class CLILogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_log_json_reports_revisions_and_ancestors(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("log", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIn("revisions", payload)
        self.assertIn("ancestors", payload)
        # A freshly exported root ARA has no revisions and no ancestors.
        self.assertEqual(payload["revisions"], [])
        self.assertEqual(payload["ancestors"], [])

    def test_log_human_output_shows_lock(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("log", "--ara", str(a))
        self.assertEqual(rc, 0)
        self.assertIn("rev 0 (lock)", out)


class CLIRefsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_refs_set_get_list_delete(self) -> None:
        a = _make_ara(self.tmp, "a")
        target = "sha256:" + "1" * 64

        rc, _, _ = _run("refs", "--ara", str(a), "--set", "HEAD", target)
        self.assertEqual(rc, 0)

        rc, out, _ = _run("refs", "--ara", str(a), "--get", "HEAD")
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), target)

        rc, out, _ = _run("refs", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(payload, [{"name": "HEAD", "target": target}])

        rc, _, _ = _run("refs", "--ara", str(a), "--delete", "HEAD")
        self.assertEqual(rc, 0)

        rc, _, err = _run("refs", "--ara", str(a), "--get", "HEAD")
        self.assertEqual(rc, 3)
        self.assertIn("not set", err)

    def test_refs_bad_name_returns_error(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, _, err = _run(
            "refs", "--ara", str(a),
            "--set", "../escape", "sha256:" + "1" * 64,
        )
        self.assertEqual(rc, 2)
        self.assertIn("refused", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
