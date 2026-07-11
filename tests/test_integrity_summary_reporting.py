from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from continuous_paper_generator import ContinuousPaperGenerator
from run_project import save_project_summary


class IntegritySummaryReportingTests(unittest.TestCase):
    def test_project_summary_reports_integrity_forensics_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_dir = Path(td) / "project"
            report_path = project_dir / "paper0" / "integrity_forensics" / "report.json"
            results = [
                {
                    "idea_idx": 0,
                    "status": "success",
                    "pdf_path": "/tmp/paper0.pdf",
                    "submission_acceptance_passed": True,
                    "submission_priority_score": 91.0,
                    "quality_gate_passed": True,
                    "quality_score": 88.0,
                    "rigor_score": 86.0,
                    "integrity_forensics_enabled": True,
                    "integrity_forensics_status": "completed",
                    "integrity_forensics_verdict": "HARD_FLAGS",
                    "integrity_forensics_findings": 1,
                    "integrity_forensics_report_file": str(report_path),
                },
                {
                    "idea_idx": 1,
                    "status": "success",
                    "pdf_path": "/tmp/paper1.pdf",
                    "submission_acceptance_passed": False,
                    "submission_priority_score": 82.0,
                    "quality_gate_passed": True,
                    "quality_score": 84.0,
                    "rigor_score": 83.0,
                    "integrity_forensics_enabled": True,
                    "integrity_forensics_status": "completed",
                    "integrity_forensics_verdict": "SOFT_FLAGS",
                    "integrity_forensics_findings": 2,
                },
                {
                    "idea_idx": 2,
                    "status": "failed",
                    "stage": "final_submission_bar",
                    "integrity_forensics": {
                        "enabled": True,
                        "status": "completed",
                        "overall_verdict": "HARD_FLAGS",
                        "finding_count": 3,
                        "files": {"report": "/tmp/nested-report.json"},
                    },
                },
            ]

            _, shortlist_file, summary = save_project_summary(str(project_dir), results)

            quality = summary["quality_summary"]
            self.assertEqual(quality["integrity_forensics_enabled"], 3)
            self.assertEqual(quality["integrity_forensics_completed"], 3)
            self.assertEqual(quality["integrity_forensics_hard_flags"], 2)
            self.assertEqual(quality["integrity_forensics_soft_flags"], 1)
            self.assertEqual(
                quality["integrity_forensics_verdict_counts"]["HARD_FLAGS"],
                2,
            )
            shortlist = Path(shortlist_file).read_text(encoding="utf-8")
            self.assertIn("Integrity Forensics: HARD_FLAGS", shortlist)
            self.assertIn(str(report_path), shortlist)

    def test_continuous_summary_reports_integrity_forensics_counts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            generator = ContinuousPaperGenerator.__new__(ContinuousPaperGenerator)
            generator.research_dir = Path(td)
            generator.batch_name = "integrity_summary"
            generator.batch_dir = Path(td) / "batches" / "integrity_summary"
            generator.batch_dir.mkdir(parents=True, exist_ok=True)
            generator.progress = {
                "papers_completed": [
                    {
                        "status": "success",
                        "idea_idx": 0,
                        "idea_name": "soft_flagged",
                        "paper_type": "journal",
                        "target_venue": "nature",
                        "submission_acceptance_passed": True,
                        "quality_gate_passed": True,
                        "submission_priority_score": 92.0,
                        "quality_score": 90.0,
                        "rigor_score": 88.0,
                        "claim_support_score": 87.0,
                        "blocker_count": 0,
                        "integrity_forensics_enabled": True,
                        "integrity_forensics_status": "completed",
                        "integrity_forensics_verdict": "SOFT_FLAGS",
                        "integrity_forensics_findings": 2,
                        "integrity_forensics_report_file": "/tmp/report.json",
                    }
                ],
                "papers_failed": [
                    {
                        "status": "failed",
                        "idea_idx": 1,
                        "idea_name": "errored",
                        "paper_type": "journal",
                        "stage": "review",
                        "integrity_forensics_enabled": True,
                        "integrity_forensics_status": "error",
                    },
                    {
                        "status": "failed",
                        "idea_idx": 2,
                        "idea_name": "nested_hard",
                        "paper_type": "journal",
                        "stage": "final_submission_bar",
                        "integrity_forensics": {
                            "enabled": True,
                            "status": "completed",
                            "overall_verdict": "HARD_FLAGS",
                            "finding_count": 1,
                            "files": {"report": "/tmp/nested-hard.json"},
                        },
                    }
                ],
            }

            with mock.patch("builtins.print"):
                report_file = generator.generate_summary_report()

            report = json.loads(Path(report_file).read_text(encoding="utf-8"))
            quality = report["quality_summary"]
            self.assertEqual(quality["integrity_forensics_enabled"], 3)
            self.assertEqual(quality["integrity_forensics_completed"], 2)
            self.assertEqual(quality["integrity_forensics_hard_flags"], 1)
            self.assertEqual(quality["integrity_forensics_soft_flags"], 1)
            self.assertEqual(quality["integrity_forensics_status_counts"]["error"], 1)
            shortlist = (generator.batch_dir / "submission_shortlist.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("Integrity forensics: SOFT_FLAGS", shortlist)
            self.assertIn("/tmp/report.json", shortlist)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
