from __future__ import annotations

from pathlib import Path
from unittest import mock

from ai_scientist.perform_plotting import PlotExecutionPolicyError
from ai_scientist.utils import launcher_workflow
from ai_scientist.utils.run_index import load_workflow_state


def test_plot_isolation_failure_is_auditable_and_resumes_without_rerunning_bfts(
    tmp_path: Path,
) -> None:
    idea_dir = tmp_path / "idea"
    idea_dir.mkdir()
    idea_path = idea_dir / "idea.json"
    idea_path.write_text("{}", encoding="utf-8")
    config = idea_dir / "run-config.yaml"
    config.write_text("exec: {}\n", encoding="utf-8")
    run_dir = idea_dir / "logs" / "0-run"
    results = run_dir / "experiment_results"
    results.mkdir(parents=True)
    (results / "result.npy").write_bytes(b"data")

    with (
        mock.patch.object(
            launcher_workflow,
            "edit_bfts_config_file",
            return_value=str(config),
        ) as edit_config,
        mock.patch.object(launcher_workflow, "is_stage_complete", return_value=False),
        mock.patch.object(
            launcher_workflow,
            "perform_experiments_bfts",
            return_value={"status": "completed"},
        ) as perform_bfts,
        mock.patch.object(
            launcher_workflow,
            "find_latest_bfts_run_dir",
            return_value=run_dir,
        ),
        mock.patch.object(
            launcher_workflow,
            "aggregate_plots",
            side_effect=PlotExecutionPolicyError("Docker isolation unavailable"),
        ),
        mock.patch.object(launcher_workflow, "save_token_tracker"),
    ):
        failed = launcher_workflow.run_experiment_phase(
            idea_dir,
            idea_path,
            "openai/gpt-4.1",
        )

    assert isinstance(failed, dict)
    assert failed["status"] == "failed"
    assert failed["stage"] == "plot_aggregation"
    assert failed["resumable"] is True
    state = load_workflow_state(idea_dir)["stages"]["experiment"]
    assert state["reason"] == "plot_aggregation_failed"
    assert state["metadata"]["experiment_completed"] is True
    assert state["metadata"]["plot_aggregation_pending"] is True
    edit_config.assert_called_once()
    perform_bfts.assert_called_once()

    report_json = idea_dir / "experiment_report.json"
    report_md = idea_dir / "experiment_report.md"
    with (
        mock.patch.object(launcher_workflow, "edit_bfts_config_file") as edit_again,
        mock.patch.object(launcher_workflow, "is_stage_complete", return_value=False),
        mock.patch.object(
            launcher_workflow, "perform_experiments_bfts"
        ) as perform_again,
        mock.patch.object(
            launcher_workflow,
            "find_latest_bfts_run_dir",
            return_value=run_dir,
        ),
        mock.patch.object(launcher_workflow, "aggregate_plots") as aggregate_again,
        mock.patch.object(
            launcher_workflow,
            "write_experiment_report",
            return_value=(report_json, report_md),
        ),
        mock.patch.object(launcher_workflow, "save_token_tracker"),
    ):
        resumed = launcher_workflow.run_experiment_phase(
            idea_dir,
            idea_path,
            "openai/gpt-4.1",
        )

    assert resumed == str(config.resolve())
    edit_again.assert_not_called()
    perform_again.assert_not_called()
    aggregate_again.assert_called_once()
    completed = load_workflow_state(idea_dir)["stages"]["experiment"]
    assert completed["status"] == "completed"
