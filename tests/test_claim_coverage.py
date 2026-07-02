"""Tests for the claim-coverage scorer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.claim_coverage import (
    ClaimCoverageReport,
    evaluate_claim_coverage,
    write_coverage_into_ara,
)


def _write_index(ara_dir: Path, payload: dict) -> None:
    claims = ara_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    (claims / "_index.json").write_text(json.dumps(payload), encoding="utf-8")


class CoverageEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ara_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_no_index_returns_unknown(self) -> None:
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "unknown")
        self.assertEqual(report.claim_count, 0)

    def test_zero_claims_flagged_as_none(self) -> None:
        _write_index(self.ara_dir, {"claim_count": 0, "resolved_count": 0, "unresolved_node_ids": []})
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "none")

    def test_full_resolution_hits_ok_at_min_threshold(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 3, "resolved_count": 3, "unresolved_node_ids": []},
        )
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "ok")
        self.assertEqual(report.coverage_score, 1.0)
        self.assertEqual(report.coverage_ratio, 1.0)

    def test_all_resolved_but_below_min_expected_is_sparse(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 1, "resolved_count": 1, "unresolved_node_ids": []},
        )
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "sparse")
        # score normalises against minimum_expected (3), so 1/3 not 1/1.
        self.assertAlmostEqual(report.coverage_score, 1 / 3)

    def test_below_critical_threshold_marked_insufficient(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 5, "resolved_count": 1, "unresolved_node_ids": ["a", "b", "c", "d"]},
        )
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "insufficient")
        self.assertLess(report.coverage_ratio, 0.5)

    def test_partial_resolution_marked_unresolved(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 5, "resolved_count": 4, "unresolved_node_ids": ["ghost"]},
        )
        report = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(report.severity, "unresolved")
        self.assertEqual(report.unresolved_node_ids, ["ghost"])

    def test_write_coverage_into_ara_roundtrip(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 3, "resolved_count": 3, "unresolved_node_ids": []},
        )
        report = evaluate_claim_coverage(self.ara_dir)
        dest = write_coverage_into_ara(self.ara_dir, report)
        self.assertTrue(dest.exists())
        payload = json.loads(dest.read_text())
        self.assertEqual(payload["claim_count"], 3)
        self.assertEqual(payload["severity"], "ok")

    def test_custom_threshold_flips_severity(self) -> None:
        _write_index(
            self.ara_dir,
            {"claim_count": 10, "resolved_count": 8, "unresolved_node_ids": ["a", "b"]},
        )
        # With defaults: ratio 0.8 > critical 0.5 → unresolved (not insufficient).
        base = evaluate_claim_coverage(self.ara_dir)
        self.assertEqual(base.severity, "unresolved")
        # But if we demand 90% coverage, same data → insufficient.
        stricter = evaluate_claim_coverage(self.ara_dir, critical_threshold=0.9)
        self.assertEqual(stricter.severity, "insufficient")


if __name__ == "__main__":
    unittest.main()
