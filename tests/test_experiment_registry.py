from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.experiment_registry import (
    build_experiment_record,
    load_experiment_records,
    save_experiment_registry,
    summarize_experiment_registry,
)
from ai_scientist.utils.pipeline_contracts import load_pipeline_manifest


class ExperimentRegistryTests(unittest.TestCase):
    def test_save_and_summarize_registry_should_update_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "projects" / "demo_project"
            project_root.mkdir(parents=True, exist_ok=True)
            rows = [
                build_experiment_record(
                    task_id="task_0",
                    dataset="demo-ds",
                    metric="accuracy",
                    baseline_ref="baseline_a",
                    status="completed",
                    result_summary={"metric_mean": 0.82},
                    entered_storyline=True,
                    workflow_mode="program_driven",
                    policy_name="program_driven",
                    acceptance_checks=["Keep budget discipline."],
                ),
                build_experiment_record(
                    task_id="task_1",
                    dataset="demo-ds",
                    metric="f1",
                    baseline_ref="baseline_b",
                    status="failed",
                    error_type="timeout",
                    error_message="budget exhausted",
                    workflow_mode="program_driven",
                    policy_name="program_driven",
                    acceptance_checks=["Record budget exhaustion explicitly."],
                ),
            ]
            save_experiment_registry(project_root, rows)

            loaded = load_experiment_records(project_root)
            summary = summarize_experiment_registry(project_root)
            manifest = load_pipeline_manifest(project_root)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(summary["by_status"]["completed"], 1)
            self.assertEqual(summary["by_status"]["failed"], 1)
            self.assertEqual(summary["by_budget_status"]["within_budget"], 1)
            self.assertEqual(summary["by_budget_status"]["budget_exhausted"], 1)
            self.assertEqual(summary["policy_names"]["program_driven"], 2)
            self.assertEqual(summary["storyline_count"], 1)
            self.assertEqual(loaded[0]["workflow_mode"], "program_driven")
            self.assertTrue(loaded[0]["acceptance_checks"])
            self.assertEqual(
                manifest["artifacts"]["experiment_registry"]["status"],
                "ready",
            )


if __name__ == "__main__":
    unittest.main()
