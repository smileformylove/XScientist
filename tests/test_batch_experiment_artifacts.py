from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.batch_experiment_artifacts import (
    _annotate_report_with_experiment_todo,
    _build_batch_experiment_agenda,
    _build_batch_experiment_ledger_rows,
    _build_batch_experiment_todo,
    _build_batch_experiment_todo_markdown,
    _build_paper_experiment_todo_tasks,
    _write_per_paper_experiment_todo_artifacts,
)


class BatchExperimentArtifactTests(unittest.TestCase):
    def test_self_review_gate_builds_prioritized_experiment_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paper_dir = Path(td)
            (paper_dir / "self_review_iteration_summary.json").write_text(
                json.dumps(
                    {
                        "latest_round_gate": {
                            "ready": False,
                            "reasons": ["critical_issues_unresolved"],
                            "next_focus_summaries": ["add ablation evidence"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            tasks = _build_paper_experiment_todo_tasks(
                {
                    "idea_idx": 2,
                    "idea_name": "demo",
                    "paper_dir": str(paper_dir),
                }
            )

        self.assertEqual(tasks[0]["task_id"], "idea2-T01")
        self.assertEqual(tasks[0]["priority"], "P0")
        self.assertEqual(tasks[0]["source_signal"], "critical_issues_unresolved")
        self.assertTrue(
            any("add ablation evidence" in task["action"] for task in tasks)
        )

    def test_batch_todo_writes_and_annotates_per_paper_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            paper_dir = Path(td)
            report = {
                "completed_papers": [
                    {
                        "idea_idx": 1,
                        "idea_name": "paper-a",
                        "paper_dir": str(paper_dir),
                        "unsupported_claims_count": 2,
                    }
                ]
            }
            todo = _build_batch_experiment_todo(report)
            _write_per_paper_experiment_todo_artifacts(todo)
            _annotate_report_with_experiment_todo(report, todo)

            saved = json.loads(
                (paper_dir / "experiment_todo.json").read_text(encoding="utf-8")
            )

        self.assertGreater(saved["counts"]["total_tasks"], 0)
        self.assertGreater(report["completed_papers"][0]["experiment_todo_count"], 0)
        self.assertIn(
            "# Batch Experiment TODO", _build_batch_experiment_todo_markdown(todo)
        )

    def test_ledger_and_agenda_classify_batch_outcomes(self) -> None:
        report = {
            "completed_papers": [
                {
                    "idea_name": "strong",
                    "status": "success",
                    "quality_gate_passed": True,
                    "submission_priority_score": 90,
                    "blocker_count": 1,
                },
                {
                    "idea_name": "weak",
                    "status": "success",
                    "quality_gate_passed": False,
                    "unsupported_claims_count": 4,
                    "blocker_count": 6,
                },
            ],
            "failed_papers": [
                {"idea_name": "broken", "status": "failed", "stage": "writeup"}
            ],
            "quality_summary": {
                "top_papers": [
                    {
                        "idea_name": "strong",
                        "revision_actions": [
                            {
                                "focus": "experiments",
                                "priority": "P0",
                                "action": "run robustness study",
                                "reason": "missing stress test",
                            }
                        ],
                    }
                ]
            },
        }

        rows = _build_batch_experiment_ledger_rows(report)
        agenda = _build_batch_experiment_agenda(report)

        self.assertEqual(
            [row["decision"] for row in rows], ["keep", "discard", "crash"]
        )
        self.assertEqual(agenda["counts"], {"keep": 1, "discard": 1, "crash": 1})
        self.assertEqual(agenda["failed_stages"], {"writeup": 1})
        self.assertEqual(
            agenda["priority_experiments"][0]["action"], "run robustness study"
        )


if __name__ == "__main__":
    unittest.main()
