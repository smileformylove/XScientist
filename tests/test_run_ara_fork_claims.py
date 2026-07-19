"""Smoke tests for the `claims` verb in run_ara_fork.py.

The verb is a thin joiner over already-tested primitives
(`claim_registry.write_claims_into_ara` + `exploration_graph.json`),
so these tests just pin the CLI contract: enumeration, --json shape,
--node filter, and the empty / unlinked / no-match edges.

Rather than run the full claim_registry scanner, we hand-write
``<ara>/claims/*.json`` files matching the on-disk shape produced by
``write_claims_into_ara`` — this keeps the test independent from the
tex-parsing pipeline (which has its own tests in test_ara_artifact.py).
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from ai_scientist.apps import ara as run_ara_fork

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


def _write_claim(ara: Path, claim_id: str, node_id: str | None, context: str) -> None:
    """Write a claim JSON matching write_claims_into_ara's shape."""
    claims_dir = ara / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "claim_id": claim_id,
        "node_id": node_id or "",
        "tex_file": "template.tex",
        "line": 1,
        "context": context,
        "options": {},
        "resolved": bool(node_id),
        "node": None,
        "recorded_at": "2026-07-10T00:00:00+00:00",
    }
    (claims_dir / f"{claim_id}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _two_node_ara(tmp: Path, sub: str) -> Path:
    return _make_ara(tmp, sub, [
        {
            "id": "n1", "step": 0, "code": "print('a')",
            "_term_out": [],
            "metric": {"value": 0.874, "maximize": True, "name": "acc"},
            "is_buggy": False, "parent_id": None, "children": [],
        },
        {
            "id": "n2", "step": 1, "code": "print('b')",
            "_term_out": [],
            "metric": {"value": 0.612, "maximize": True, "name": "acc"},
            "is_buggy": True, "parent_id": "n1", "children": [],
        },
    ])


def _run(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = run_ara_fork.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


class ClaimsCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_claims_lists_all(self) -> None:
        ara = _two_node_ara(self.tmp, "list_all")
        _write_claim(ara, "c1", "n1", "Our approach outperforms baselines.")
        _write_claim(ara, "c2", "n2", "Ablation confirms X matters.")
        rc, out, err = _run("claims", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(len(payload), 2)
        ids = {row["claim_id"] for row in payload}
        self.assertEqual(ids, {"c1", "c2"})
        # Each entry links back to its node with the graph-derived hash/metric.
        by_id = {row["claim_id"]: row for row in payload}
        self.assertEqual(by_id["c1"]["node_id"], "n1")
        self.assertEqual(by_id["c2"]["node_id"], "n2")
        self.assertTrue(by_id["c1"]["node_content_hash"].startswith("sha256:"))
        self.assertFalse(by_id["c1"]["node_is_buggy"])
        self.assertTrue(by_id["c2"]["node_is_buggy"])

    def test_claims_no_claims_dir(self) -> None:
        ara = _two_node_ara(self.tmp, "empty")
        # Human view: empty stdout, stderr note, rc=0.
        rc, out, err = _run("claims", "--ara", str(ara))
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), "")
        self.assertIn("no claims recorded", err)
        # JSON view: `[]`, rc=0 (stderr note still emitted — matches iter-13).
        rc, out, err = _run("claims", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), [])

    def test_claims_filter_by_node(self) -> None:
        ara = _two_node_ara(self.tmp, "filter")
        _write_claim(ara, "c1", "n1", "n1 assertion")
        _write_claim(ara, "c2", "n2", "n2 assertion")
        rc, out, err = _run("claims", "--ara", str(ara),
                            "--node", "n1", "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["claim_id"], "c1")
        self.assertEqual(payload[0]["node_id"], "n1")

    def test_claims_filter_by_node_no_match(self) -> None:
        ara = _two_node_ara(self.tmp, "no_match")
        _write_claim(ara, "c1", "n1", "n1 assertion")
        rc, _, err = _run("claims", "--ara", str(ara), "--node", "n999")
        self.assertEqual(rc, 3)
        self.assertIn("n999", err)

    def test_claims_unlinked_shows_dash(self) -> None:
        ara = _two_node_ara(self.tmp, "unlinked")
        # A claim whose node_id doesn't exist in the graph → node fields null.
        _write_claim(ara, "c_ghost", "n_missing", "Unlinked claim; see coverage.json")
        # Human: NODE / HASH_PREFIX / METRIC / BUGGY all render as `-`.
        rc, out, err = _run("claims", "--ara", str(ara))
        self.assertEqual(rc, 0, msg=err)
        # Data row (skip the header) must have four `-` columns after CLAIM_ID.
        data_line = out.splitlines()[1]
        self.assertIn("c_ghost", data_line)
        # Split on 2+ spaces so the header alignment doesn't confuse us.
        import re as _re
        cols = _re.split(r"\s{2,}", data_line.strip())
        # CLAIM_ID, NODE, HASH_PREFIX, METRIC, BUGGY, ASSERTION
        self.assertEqual(cols[1], "-")  # NODE
        self.assertEqual(cols[2], "-")  # HASH_PREFIX
        self.assertEqual(cols[3], "-")  # METRIC
        self.assertEqual(cols[4], "-")  # BUGGY
        # JSON view: node_content_hash / node_metric / node_is_buggy null.
        rc, out, _ = _run("claims", "--ara", str(ara), "--json")
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        row = payload[0]
        self.assertIsNone(row["node_content_hash"])
        self.assertIsNone(row["node_metric"])
        self.assertIsNone(row["node_is_buggy"])

    def test_claims_json_shape(self) -> None:
        ara = _two_node_ara(self.tmp, "shape")
        _write_claim(ara, "c1", "n1", "assertion text")
        rc, out, err = _run("claims", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertTrue(all(isinstance(row, dict) for row in payload))
        expected = {
            "claim_id", "node_id", "node_content_hash", "node_metric",
            "node_is_buggy", "assertion", "tex_file", "line", "severity",
        }
        self.assertLessEqual(expected, set(payload[0].keys()))

    def test_claims_ignores_index_and_coverage_files(self) -> None:
        """`_index.json` / `coverage.json` are registry bookkeeping — not claims."""
        ara = _two_node_ara(self.tmp, "meta")
        _write_claim(ara, "c1", "n1", "real claim")
        (ara / "claims" / "_index.json").write_text(
            json.dumps({"claim_count": 1, "resolved_count": 1}),
            encoding="utf-8",
        )
        (ara / "claims" / "coverage.json").write_text(
            json.dumps({"severity": "ok", "coverage_score": 0.33}),
            encoding="utf-8",
        )
        rc, out, err = _run("claims", "--ara", str(ara), "--json")
        self.assertEqual(rc, 0, msg=err)
        payload = json.loads(out)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["claim_id"], "c1")
        # `coverage.json`'s severity is propagated onto each row.
        self.assertEqual(payload[0]["severity"], "ok")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
