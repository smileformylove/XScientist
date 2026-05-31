from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.pipeline_contracts import initialize_pipeline_contracts
from ai_scientist.utils.repair_attempts import (
    backfill_scores_delta,
    compact_attempts,
    load_all_attempts,
    load_attempts_for_issue,
    record_repair_attempt,
)


class RepairAttemptsTests(unittest.TestCase):
    def test_round_trip_and_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=1,
                job_id="job_a",
                status="success",
                addressed=True,
                coverage_ratio=0.5,
                scores_delta={"Clarity": 0.4},
                notes="first try",
            )
            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=2,
                job_id="job_b",
                status="failed",
                addressed=False,
                coverage_ratio=0.0,
            )
            record_repair_attempt(
                project_root,
                "RVW-2",
                round_index=2,
                job_id="job_b",
                status="success",
                addressed=True,
            )

            rvw1 = load_attempts_for_issue(project_root, "RVW-1")
            self.assertEqual(len(rvw1), 2)
            self.assertEqual([row["round_index"] for row in rvw1], [1, 2])
            self.assertEqual(rvw1[0]["job_id"], "job_a")
            self.assertTrue(rvw1[0]["addressed"])
            self.assertFalse(rvw1[1]["addressed"])

            all_rows = load_all_attempts(project_root)
            self.assertEqual(set(all_rows.keys()), {"RVW-1", "RVW-2"})
            self.assertEqual(len(all_rows["RVW-2"]), 1)

    def test_dedupe_by_latest_generated_at(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            # Same (issue_id, round_index, job_id) triple — later write wins.
            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=1,
                job_id="job_a",
                status="success",
                addressed=True,
                notes="early",
            )
            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=1,
                job_id="job_a",
                status="failed",
                addressed=False,
                notes="later",
            )

            rvw1 = load_attempts_for_issue(project_root, "RVW-1")
            self.assertEqual(len(rvw1), 1)
            self.assertEqual(rvw1[0]["notes"], "later")
            self.assertFalse(rvw1[0]["addressed"])

            kept = compact_attempts(project_root)
            self.assertEqual(kept, 1)

    def test_limit_and_missing_issue(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")

            for idx in range(5):
                record_repair_attempt(
                    project_root,
                    "RVW-1",
                    round_index=idx,
                    job_id=f"job_{idx}",
                    status="success",
                    addressed=True,
                )

            tail = load_attempts_for_issue(project_root, "RVW-1", limit=2)
            self.assertEqual([row["round_index"] for row in tail], [3, 4])
            self.assertEqual(load_attempts_for_issue(project_root, ""), [])
            self.assertEqual(load_attempts_for_issue(project_root, "missing"), [])

    def test_skipped_when_issue_id_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            row = record_repair_attempt(
                project_root,
                "",
                round_index=0,
                status="success",
            )
            self.assertEqual(row.get("status"), "skipped_missing_issue_id")


class BackfillScoresDeltaTests(unittest.TestCase):
    def test_backfill_fills_delta_for_attempts_with_baseline_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=1,
                job_id="job_a",
                status="success",
                addressed=True,
                scores_before={"Clarity": 5.0, "Soundness": 6.0},
            )
            record_repair_attempt(
                project_root,
                "RVW-2",
                round_index=1,
                job_id="job_a",
                status="success",
                addressed=True,
                scores_before={"Clarity": 5.0},
                scores_delta={"Clarity": 1.0},  # already filled, must NOT be overwritten
            )
            updated = backfill_scores_delta(
                project_root,
                current_scores={"Clarity": 7.5, "Soundness": 6.0, "Quality": 8.0},
            )
            self.assertEqual(updated, 1)
            rows = {row["issue_id"]: row for row in load_all_attempts(project_root)["RVW-1"]}
            self.assertEqual(rows["RVW-1"]["scores_delta"], {"Clarity": 2.5, "Soundness": 0.0})
            unchanged = load_attempts_for_issue(project_root, "RVW-2")[0]
            self.assertEqual(unchanged["scores_delta"], {"Clarity": 1.0})

    def test_backfill_no_op_when_empty(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            self.assertEqual(
                backfill_scores_delta(project_root, current_scores={"Clarity": 7.0}),
                0,
            )

    def test_backfill_skips_attempts_without_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root, workflow_mode="review_board")
            record_repair_attempt(
                project_root,
                "RVW-1",
                round_index=1,
                status="success",
                addressed=True,
                # No scores_before
            )
            self.assertEqual(
                backfill_scores_delta(project_root, current_scores={"Clarity": 7.0}),
                0,
            )


if __name__ == "__main__":
    unittest.main()
