"""Smoke tests for the `history` verb in run_ara_fork.py.

The engine (append_manifest_revision / write_manifest_lock / _read_history)
is covered by tests/test_ara_manifest_lock.py — this file just verifies
the CLI wraps it correctly (row 0 synthesis, ordering, --limit, --json,
and the unlocked → rc=3 path).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.apps import ara as run_ara_fork

from ai_scientist.utils.ara_artifact import (
    export_ara, update_manifest_claim_count,
)


def _write_journal(logs_dir: Path, nodes: list[dict]) -> None:
    stage_dir = logs_dir / "0-run"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "journal.json").write_text(
        json.dumps({"nodes": nodes, "node2parent": {}, "__version": "2"}),
        encoding="utf-8",
    )


def _make_ara(tmp: Path, sub: str) -> Path:
    project = tmp / sub
    exp = project / "02_experiments" / f"20260710_{sub}"
    _write_journal(exp / "logs", [{
        "id": "n1", "step": 0, "code": "print('ok')",
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


class HistoryCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_history_shows_base_when_no_revisions(self) -> None:
        a = _make_ara(self.tmp, "a")
        rc, out, _ = _run("history", "--ara", str(a))
        self.assertEqual(rc, 0)
        # Header row + a rev-0 line whose reason marks the initial export.
        self.assertIn("rev", out.splitlines()[0])
        self.assertIn("(initial export)", out)
        # No further data rows past 0.
        data_lines = [ln for ln in out.splitlines()[1:] if ln.strip()]
        self.assertEqual(len(data_lines), 1)
        self.assertTrue(data_lines[0].lstrip().startswith("0"))

    def test_history_shows_multiple_revisions(self) -> None:
        a = _make_ara(self.tmp, "a")
        manifest = a / "manifest.json"
        update_manifest_claim_count(manifest, 3)
        update_manifest_claim_count(manifest, 9)
        rc, out, _ = _run("history", "--ara", str(a))
        self.assertEqual(rc, 0)
        data_lines = [ln for ln in out.splitlines()[1:] if ln.strip()]
        self.assertEqual(len(data_lines), 3)
        # Row 0 first, then rev 1, then rev 2 — order preserved.
        self.assertTrue(data_lines[0].lstrip().startswith("0"))
        self.assertTrue(data_lines[1].lstrip().startswith("1"))
        self.assertTrue(data_lines[2].lstrip().startswith("2"))
        self.assertIn("update_manifest_claim_count", out)

    def test_history_json_output_shape(self) -> None:
        a = _make_ara(self.tmp, "a")
        manifest = a / "manifest.json"
        update_manifest_claim_count(manifest, 4)
        rc, out, _ = _run("history", "--ara", str(a), "--json")
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertGreaterEqual(len(payload), 2)
        for row in payload:
            for key in ("revision", "base_hash", "new_hash"):
                self.assertIn(key, row)
        # Row 0's base_hash is null; later rows carry a full sha256.
        self.assertIsNone(payload[0]["base_hash"])
        self.assertTrue(payload[1]["base_hash"].startswith("sha256:"))
        # Full hashes preserved (>16 hex chars past the prefix).
        self.assertGreater(len(payload[1]["new_hash"].split(":", 1)[1]), 16)

    def test_history_limit_narrows(self) -> None:
        a = _make_ara(self.tmp, "a")
        manifest = a / "manifest.json"
        for n in (1, 2, 3):
            update_manifest_claim_count(manifest, n)
        rc, out, _ = _run("history", "--ara", str(a), "--limit", "2")
        self.assertEqual(rc, 0)
        data_lines = [ln for ln in out.splitlines()[1:] if ln.strip()]
        # Row 0 (always) + last 2 of 3 revisions == 3 rows total.
        self.assertEqual(len(data_lines), 3)
        # rev 1 must be dropped; rev 2 and rev 3 kept.
        starts = [ln.lstrip().split()[0] for ln in data_lines]
        self.assertEqual(starts, ["0", "2", "3"])

    def test_history_unlocked_returns_rc3(self) -> None:
        a = _make_ara(self.tmp, "a")
        (a / "manifest.lock").unlink()
        rc, _, err = _run("history", "--ara", str(a))
        self.assertEqual(rc, 3)
        self.assertIn("unlocked", err)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
