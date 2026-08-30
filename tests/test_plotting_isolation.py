from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from ai_scientist.perform_plotting import (
    PlotAggregationError,
    PlotExecutionSettings,
    PlotExecutionPolicyError,
    PlotRunResult,
    _audit_input_hashes,
    _publish_plot_bundle,
    _plot_publish_lock,
    _plot_execution_policy,
    _plot_execution_settings,
    _validate_plot_outputs,
    aggregate_plots,
    run_aggregator_script,
)
from ai_scientist.treesearch.interpreter import SandboxPolicy


class _FakeInterpreter:
    execution_backend = "docker"

    def __init__(self, result: SimpleNamespace) -> None:
        self.result = result
        self.cleaned = False

    def run(self, code: str) -> SimpleNamespace:
        self.code = code
        return self.result

    def cleanup_session(self) -> None:
        self.cleaned = True


def _execution_result(*, exc_type: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        term_out=["output"],
        exc_type=exc_type,
        execution_backend="docker",
        isolation={"isolated": True, "network": "none"},
    )


def _png(width: int = 2, height: int = 2) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )


def test_plot_policy_forces_isolation_and_disables_network(tmp_path: Path) -> None:
    config = tmp_path / "bfts.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "exec": {
                    "backend": "auto",
                    "require_isolation": False,
                    "network": "bridge",
                    "timeout": 17,
                }
            }
        ),
        encoding="utf-8",
    )

    policy = _plot_execution_policy(
        config,
        allow_unisolated_local_model_code=False,
    )
    assert policy.require_isolation is True
    assert policy.network == "none"
    settings = _plot_execution_settings(
        config,
        allow_unisolated_local_model_code=False,
    )
    assert settings.timeout == 17

    local_policy = _plot_execution_policy(
        config,
        allow_unisolated_local_model_code=True,
    )
    assert local_policy.require_isolation is False
    assert local_policy.network == "bridge"


def test_local_opt_in_cannot_downgrade_required_isolation(tmp_path: Path) -> None:
    config = tmp_path / "bfts.yaml"
    config.write_text(
        yaml.safe_dump({"exec": {"backend": "docker", "require_isolation": True}}),
        encoding="utf-8",
    )

    with pytest.raises(PlotExecutionPolicyError, match="cannot downgrade"):
        _plot_execution_policy(
            config,
            allow_unisolated_local_model_code=True,
        )


def test_plot_policy_rejects_non_mapping_exec_config(tmp_path: Path) -> None:
    config = tmp_path / "bfts.yaml"
    config.write_text("exec: process\n", encoding="utf-8")

    with pytest.raises(PlotExecutionPolicyError, match="exec must be a mapping"):
        _plot_execution_policy(
            config,
            allow_unisolated_local_model_code=False,
        )


def test_plot_policy_wraps_malformed_yaml(tmp_path: Path) -> None:
    config = tmp_path / "bfts.yaml"
    config.write_text("exec: [\n", encoding="utf-8")

    with pytest.raises(PlotExecutionPolicyError, match="Could not load"):
        _plot_execution_policy(
            config,
            allow_unisolated_local_model_code=False,
        )


def test_strict_plot_run_requires_an_isolated_receipt(tmp_path: Path) -> None:
    script = tmp_path / "auto_plot_aggregator.py"
    fake = _FakeInterpreter(
        SimpleNamespace(
            term_out=[],
            exc_type=None,
            execution_backend="process",
            isolation={"isolated": False},
        )
    )
    with (
        mock.patch("ai_scientist.perform_plotting.Interpreter", return_value=fake),
        pytest.raises(PlotExecutionPolicyError, match="isolated Docker receipt"),
    ):
        run_aggregator_script(
            "print('unsafe')",
            script,
            tmp_path,
            script.name,
            sandbox_policy=SandboxPolicy(backend="auto", require_isolation=True),
        )
    assert fake.cleaned is True


@pytest.mark.parametrize(
    "exc_type", ["SandboxUnavailableError", "TimeoutError", "ResourceLimitError"]
)
def test_execution_boundary_failures_stop_plotting(
    tmp_path: Path,
    exc_type: str,
) -> None:
    script = tmp_path / "auto_plot_aggregator.py"
    fake = _FakeInterpreter(_execution_result(exc_type=exc_type))
    with (
        mock.patch("ai_scientist.perform_plotting.Interpreter", return_value=fake),
        pytest.raises(PlotAggregationError, match=exc_type),
    ):
        run_aggregator_script(
            "print('bounded')",
            script,
            tmp_path,
            script.name,
            sandbox_policy=SandboxPolicy(backend="docker", require_isolation=True),
        )


def test_ordinary_script_failure_can_be_reflected(tmp_path: Path) -> None:
    script = tmp_path / "auto_plot_aggregator.py"
    fake = _FakeInterpreter(_execution_result(exc_type="ProcessExitError"))
    with mock.patch(
        "ai_scientist.perform_plotting.Interpreter", return_value=fake
    ) as interpreter:
        result = run_aggregator_script(
            "raise RuntimeError",
            script,
            tmp_path,
            script.name,
            sandbox_policy=SandboxPolicy(backend="docker", require_isolation=True),
            timeout=23,
        )
    assert result.succeeded is False
    assert result.execution_backend == "docker"
    assert interpreter.call_args.kwargs["timeout"] == 23
    assert interpreter.call_args.kwargs["env_vars"] == {}


def test_isolation_unavailable_fails_before_deleting_existing_figures(
    tmp_path: Path,
) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    sentinel = figures / "previous.png"
    sentinel.write_bytes(b"previous")

    with (
        mock.patch(
            "ai_scientist.perform_plotting.Interpreter",
            side_effect=RuntimeError("Docker unavailable"),
        ),
        mock.patch("ai_scientist.perform_plotting.create_client") as create_client,
        pytest.raises(PlotExecutionPolicyError, match="Docker unavailable"),
    ):
        aggregate_plots(str(tmp_path), n_reflections=0)

    assert sentinel.read_bytes() == b"previous"
    create_client.assert_not_called()


def test_plot_outputs_reject_symlinks_and_non_png_files(tmp_path: Path) -> None:
    script = tmp_path / "auto_plot_aggregator.py"
    code = "print('plot')"
    script.write_text(code, encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(_png())
    (figures / "figure.png").symlink_to(outside)

    with pytest.raises(PlotAggregationError, match="symlink_rejected"):
        _validate_plot_outputs(
            figures_dir=figures,
            aggregator_script_path=script,
            expected_code=code,
        )

    (figures / "figure.png").unlink()
    (figures / "notes.txt").write_text("not a figure", encoding="utf-8")
    with pytest.raises(PlotAggregationError, match="not a PNG"):
        _validate_plot_outputs(
            figures_dir=figures,
            aggregator_script_path=script,
            expected_code=code,
        )


def test_plot_outputs_accept_bounded_regular_png(tmp_path: Path) -> None:
    script = tmp_path / "auto_plot_aggregator.py"
    code = "print('plot')"
    script.write_text(code, encoding="utf-8")
    figures = tmp_path / "figures"
    figures.mkdir()
    payload = _png()
    (figures / "figure.png").write_bytes(payload)

    validated = _validate_plot_outputs(
        figures_dir=figures,
        aggregator_script_path=script,
        expected_code=code,
    )
    assert validated == [
        {
            "path": "figure.png",
            "bytes": len(payload),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
        }
    ]


def test_quality_guidance_is_bound_into_plot_input_hashes(tmp_path: Path) -> None:
    quality = tmp_path / "quality"
    quality.mkdir()
    guidance = quality / "experiment_visualization_brief.md"
    guidance.write_text("use confidence intervals", encoding="utf-8")

    inputs = _audit_input_hashes(tmp_path, {})

    [receipt] = [
        item
        for item in inputs
        if item["path"] == "quality/experiment_visualization_brief.md"
    ]
    assert (
        receipt["sha256"]
        == __import__("hashlib").sha256(guidance.read_bytes()).hexdigest()
    )


def test_quality_guidance_symlink_is_rejected(tmp_path: Path) -> None:
    quality = tmp_path / "quality"
    quality.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("untrusted", encoding="utf-8")
    (quality / "experiment_visualization_brief.md").symlink_to(outside)

    with pytest.raises(PlotAggregationError, match="symlink_rejected"):
        _audit_input_hashes(tmp_path, {})


def test_plot_publish_lock_fails_fast_for_same_experiment(tmp_path: Path) -> None:
    with _plot_publish_lock(tmp_path):
        with pytest.raises(PlotAggregationError, match="already active"):
            with _plot_publish_lock(tmp_path):
                raise AssertionError("second publisher unexpectedly acquired the lock")

    assert not (tmp_path / ".xscientist-plot-publish.lock").exists()


def test_plot_publish_lock_fails_closed_without_platform_lock(
    tmp_path: Path,
) -> None:
    real_import = __import__

    def reject_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("platform lock unavailable")
        return real_import(name, *args, **kwargs)

    with (
        mock.patch("builtins.__import__", side_effect=reject_fcntl),
        pytest.raises(PlotAggregationError, match="POSIX fcntl file-lock"),
    ):
        with _plot_publish_lock(tmp_path):
            raise AssertionError("publication unexpectedly ran without a lock")

    assert not (tmp_path / ".xscientist-plot-publish.lock").exists()


def test_aggregate_fails_before_generated_code_without_platform_lock(
    tmp_path: Path,
) -> None:
    real_import = __import__

    def reject_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("platform lock unavailable")
        return real_import(name, *args, **kwargs)

    with (
        mock.patch("builtins.__import__", side_effect=reject_fcntl),
        mock.patch("ai_scientist.perform_plotting.Interpreter") as interpreter,
        pytest.raises(PlotAggregationError, match="POSIX fcntl file-lock"),
    ):
        aggregate_plots(str(tmp_path), n_reflections=0)

    interpreter.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_aggregate_publishes_verified_staged_bundle_and_receipt(
    tmp_path: Path,
) -> None:
    experiment_results = tmp_path / "experiment_results"
    experiment_results.mkdir()
    data_payload = b"npy-data-v1"
    (experiment_results / "result.npy").write_bytes(data_payload)
    results = tmp_path / "results"
    results.mkdir()
    (results / "nested.npy").write_bytes(b"nested-data")
    (tmp_path / "root.npy").write_bytes(b"root-data")
    old_figures = tmp_path / "figures"
    old_figures.mkdir()
    (old_figures / "old.png").write_bytes(_png())
    (tmp_path / "auto_plot_aggregator.py").write_text("old", encoding="utf-8")

    class StagedInterpreter:
        execution_backend = "docker"

        def __init__(self, working_dir, agent_file_name, **kwargs) -> None:
            self.working_dir = Path(working_dir)
            self.agent_file_name = agent_file_name

        def cleanup_session(self) -> None:
            pass

        def run(self, code: str) -> SimpleNamespace:
            assert self.working_dir != tmp_path
            assert (
                self.working_dir / "experiment_results" / "result.npy"
            ).read_bytes() == data_payload
            assert (
                self.working_dir / "results" / "nested.npy"
            ).read_bytes() == b"nested-data"
            assert (self.working_dir / "root.npy").read_bytes() == b"root-data"
            (self.working_dir / self.agent_file_name).write_text(code, encoding="utf-8")
            figures = self.working_dir / "figures"
            figures.mkdir(exist_ok=True)
            (figures / "new.png").write_bytes(_png())
            return _execution_result()

    code = "print('new')"
    with (
        mock.patch("ai_scientist.perform_plotting.Interpreter", StagedInterpreter),
        mock.patch(
            "ai_scientist.perform_plotting.create_client",
            return_value=(object(), "model"),
        ),
        mock.patch(
            "ai_scientist.perform_plotting.get_response_from_llm",
            return_value=(f"```python\n{code}\n```", []),
        ),
        mock.patch("ai_scientist.perform_plotting.load_idea_text", return_value="idea"),
        mock.patch(
            "ai_scientist.perform_plotting.load_exp_summaries",
            return_value={
                "RESEARCH_SUMMARY": {
                    "best node": {
                        "exp_results_npy_files": [
                            "experiment_results/result.npy",
                            "results/nested.npy",
                            "root.npy",
                        ]
                    }
                }
            },
        ),
    ):
        aggregate_plots(str(tmp_path), n_reflections=0)

    assert (tmp_path / "auto_plot_aggregator.py").read_text(encoding="utf-8") == code
    assert sorted(path.name for path in (tmp_path / "figures").iterdir()) == ["new.png"]
    assert not (tmp_path / ".xscientist-plot-publish.lock").exists()
    receipt = __import__("json").loads(
        (tmp_path / "plot_execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["schema"] == "xscientist.plot-execution-receipt.v1"
    assert receipt["attempt_count"] == 1
    assert receipt["execution_backend"] == "docker"
    assert receipt["actual_execution"]["isolated"] is True
    assert receipt["outputs"][0]["path"] == "new.png"
    data_receipt = next(
        item
        for item in receipt["inputs"]
        if item["path"] == "experiment_results/result.npy"
    )
    assert (
        data_receipt["sha256"] == __import__("hashlib").sha256(data_payload).hexdigest()
    )
    assert receipt["attempts"][0]["script_sha256"] == receipt["script"]["sha256"]


def test_reflection_attempt_cannot_reuse_stale_figures(tmp_path: Path) -> None:
    calls = 0

    class ReflectingInterpreter:
        execution_backend = "docker"

        def __init__(self, working_dir, agent_file_name, **kwargs) -> None:
            self.working_dir = Path(working_dir)
            self.agent_file_name = agent_file_name

        def cleanup_session(self) -> None:
            pass

        def run(self, code: str) -> SimpleNamespace:
            nonlocal calls
            calls += 1
            (self.working_dir / self.agent_file_name).write_text(code, encoding="utf-8")
            figures = self.working_dir / "figures"
            figures.mkdir(exist_ok=True)
            if calls == 1:
                (figures / "old.png").write_bytes(_png())
                return _execution_result(exc_type="ProcessExitError")
            assert not (figures / "old.png").exists()
            (figures / "new.png").write_bytes(_png())
            return _execution_result()

    responses = [
        ("```python\nprint('first')\n```", []),
        ("```python\nprint('second')\n```", []),
    ]
    with (
        mock.patch("ai_scientist.perform_plotting.Interpreter", ReflectingInterpreter),
        mock.patch(
            "ai_scientist.perform_plotting.create_client",
            return_value=(object(), "model"),
        ),
        mock.patch(
            "ai_scientist.perform_plotting.get_response_from_llm",
            side_effect=responses,
        ),
        mock.patch("ai_scientist.perform_plotting.load_idea_text", return_value="idea"),
        mock.patch("ai_scientist.perform_plotting.load_exp_summaries", return_value={}),
    ):
        aggregate_plots(str(tmp_path), n_reflections=1)

    assert sorted(path.name for path in (tmp_path / "figures").iterdir()) == ["new.png"]
    receipt = __import__("json").loads(
        (tmp_path / "plot_execution_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["attempt_count"] == 2
    assert receipt["outputs"][0]["path"] == "new.png"


def test_plot_bundle_publish_rolls_back_all_previous_artifacts(
    tmp_path: Path,
) -> None:
    base = tmp_path / "study"
    base.mkdir()
    old_script = b"old-script"
    old_png = _png(3, 3)
    old_receipt = b'{"old": true}'
    (base / "auto_plot_aggregator.py").write_bytes(old_script)
    (base / "figures").mkdir()
    (base / "figures" / "old.png").write_bytes(old_png)
    (base / "plot_execution_receipt.json").write_bytes(old_receipt)

    execution = tmp_path / "stage"
    execution.mkdir()
    (execution / "figures").mkdir()
    new_png = _png(4, 4)
    (execution / "figures" / "new.png").write_bytes(new_png)
    new_code = "print('new')"
    settings = PlotExecutionSettings(
        policy=SandboxPolicy(backend="docker", require_isolation=True),
        timeout=60,
    )
    run = PlotRunResult(
        output="",
        succeeded=True,
        exc_type=None,
        execution_backend="docker",
        isolation={"isolated": True, "network": "none"},
    )
    output = {
        "path": "new.png",
        "bytes": len(new_png),
        "sha256": __import__("hashlib").sha256(new_png).hexdigest(),
    }

    with (
        mock.patch(
            "ai_scientist.perform_plotting.atomic_write_json",
            side_effect=OSError("receipt disk failure"),
        ),
        pytest.raises(PlotAggregationError, match="transactionally"),
    ):
        _publish_plot_bundle(
            base_path=base,
            execution_path=execution,
            model="openai/gpt-4.1",
            aggregator_code=new_code,
            plot_run=run,
            validated_outputs=[output],
            execution_settings=settings,
            execution_config_sha256=None,
            attempt_count=1,
            attempt_receipts=[],
            input_hashes=[],
        )

    assert (base / "auto_plot_aggregator.py").read_bytes() == old_script
    assert (base / "plot_execution_receipt.json").read_bytes() == old_receipt
    assert sorted(path.name for path in (base / "figures").iterdir()) == ["old.png"]
    assert (base / "figures" / "old.png").read_bytes() == old_png
    assert not list(tmp_path.glob(".*figures-backup-*"))
    assert not list(tmp_path.glob(".xscientist-figures-publish-*"))
    assert not (base / ".xscientist-plot-publish.lock").exists()
