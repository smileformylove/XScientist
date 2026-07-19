from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils.integrity_workflow import (
    integrity_forensics_result_fields,
    run_integrity_forensics_for_manuscript,
)
from ai_scientist.apps.batch import ContinuousPaperGenerator
from ai_scientist.apps.project import save_project_summary


class IntegritySummaryReportingTests(unittest.TestCase):
    def test_shared_integrity_workflow_should_return_stable_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            disabled = run_integrity_forensics_for_manuscript(
                root=root,
                paper_id="disabled",
                enabled=False,
            )
            self.assertEqual(disabled, {"enabled": False, "status": "disabled"})

            skipped = run_integrity_forensics_for_manuscript(
                root=root,
                paper_id="missing-source",
                enabled=True,
            )
            self.assertEqual(skipped["status"], "skipped")
            self.assertIn("LaTeX source", skipped["reason"])

            latex_dir = root / "latex"
            latex_dir.mkdir()
            (latex_dir / "template.tex").write_text("content", encoding="utf-8")
            explicit_empty = run_integrity_forensics_for_manuscript(
                root=root,
                paper_id="explicit-empty",
                enabled=True,
                latex_paths=[],
            )
            self.assertEqual(explicit_empty["status"], "skipped")

    def test_shared_integrity_workflow_should_flatten_result_fields(self) -> None:
        fields = integrity_forensics_result_fields(
            {
                "enabled": True,
                "status": "completed",
                "overall_verdict": "SOFT_FLAGS",
                "finding_count": 2,
                "files": {
                    "report": "/tmp/report.json",
                    "markdown": "/tmp/REPORT.md",
                },
            }
        )
        self.assertEqual(fields["integrity_forensics_enabled"], True)
        self.assertEqual(fields["integrity_forensics_status"], "completed")
        self.assertEqual(fields["integrity_forensics_verdict"], "SOFT_FLAGS")
        self.assertEqual(fields["integrity_forensics_findings"], 2)
        self.assertEqual(fields["integrity_forensics_report_file"], "/tmp/report.json")
        self.assertEqual(fields["integrity_forensics_markdown_file"], "/tmp/REPORT.md")

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

    def test_continuous_batch_passes_integrity_forensics_flag_to_worker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ideas_file = Path(td) / "ideas.json"
            ideas_file.write_text(
                json.dumps([{"Name": "flag_case", "Title": "Flag Case"}]),
                encoding="utf-8",
            )
            generator = ContinuousPaperGenerator.__new__(ContinuousPaperGenerator)
            generator.batch_dir = Path(td) / "batch"
            generator.batch_dir.mkdir(parents=True, exist_ok=True)
            generator.research_dir = Path(td) / "research"
            generator.research_dir.mkdir(parents=True, exist_ok=True)
            generator.strict_fallbacks = False
            generator.progress = {
                "papers_completed": [],
                "papers_failed": [],
                "papers_generated": [],
                "source_provenance": {"source": "test"},
            }
            generator._save_progress = lambda: None
            captured: dict[str, object] = {}

            def fake_process(args):
                captured["integrity_flag"] = args[-2]
                captured["requested_workflow_mode"] = args[-1]
                return {
                    "idea_idx": args[2],
                    "status": "success",
                    "pdf_path": "/tmp/paper.pdf",
                }

            with mock.patch(
                "ai_scientist.apps.batch._process_single_paper",
                side_effect=fake_process,
            ), mock.patch("builtins.print"):
                results = generator.generate_paper_batch(
                    str(ideas_file),
                    paper_type="journal",
                    idea_indices=[0],
                    num_workers=1,
                    integrity_forensics_enabled=True,
                )

            self.assertTrue(captured["integrity_flag"])
            self.assertEqual(captured["requested_workflow_mode"], "classic_pipeline")
            self.assertEqual(results[0]["status"], "success")
            self.assertEqual(len(generator.progress["papers_completed"]), 1)

    def test_continuous_batch_defaults_integrity_forensics_for_high_quality(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ideas_file = Path(td) / "ideas.json"
            ideas_file.write_text(
                json.dumps([{"Name": "default_case", "Title": "Default Case"}]),
                encoding="utf-8",
            )
            generator = ContinuousPaperGenerator.__new__(ContinuousPaperGenerator)
            generator.batch_dir = Path(td) / "batch"
            generator.batch_dir.mkdir(parents=True, exist_ok=True)
            generator.research_dir = Path(td) / "research"
            generator.research_dir.mkdir(parents=True, exist_ok=True)
            generator.strict_fallbacks = False
            generator.progress = {
                "papers_completed": [],
                "papers_failed": [],
                "papers_generated": [],
                "source_provenance": {"source": "test"},
            }
            generator._save_progress = lambda: None
            captured: dict[str, object] = {}

            def fake_process(args):
                captured["integrity_flag"] = args[-2]
                captured["requested_workflow_mode"] = args[-1]
                return {
                    "idea_idx": args[2],
                    "status": "success",
                    "pdf_path": "/tmp/paper.pdf",
                }

            with mock.patch(
                "ai_scientist.apps.batch._process_single_paper",
                side_effect=fake_process,
            ), mock.patch("builtins.print"):
                generator.generate_paper_batch(
                    str(ideas_file),
                    paper_type="journal",
                    idea_indices=[0],
                    num_workers=1,
                    high_quality_mode=True,
                )

            self.assertTrue(captured["integrity_flag"])
            self.assertEqual(captured["requested_workflow_mode"], "classic_pipeline")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
