"""Controlled execution for shadow benchmarks and real-research canaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_artifacts import (
    EvolutionArtifactError,
    materialize_evolution_artifact,
    verify_evolution_artifact,
)
from ai_scientist.utils.evolution_gate import (
    build_canary_report,
    validate_evolution_candidate,
)

BENCHMARK_SUITE_SCHEMA = "xscientist.benchmark-suite.v1"
BENCHMARK_RUN_SCHEMA = "xscientist.benchmark-run.v1"
CANARY_SPEC_SCHEMA = "xscientist.canary-suite.v1"
_SECRET_ENV_TOKENS = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
)


class EvolutionRuntimeError(RuntimeError):
    """Raised when controlled execution is unsafe or does not produce evidence."""


class EvolutionExecutionError(EvolutionRuntimeError):
    """Execution failure carrying its content-addressed negative result."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


def _is_hash(value: Any) -> bool:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    try:
        int(text[7:], 16)
    except ValueError:
        return False
    return True


def _bytes_hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _finite_metrics(payload: Any) -> dict[str, float]:
    if not isinstance(payload, Mapping) or not payload:
        raise EvolutionRuntimeError("evaluator output requires a metrics object")
    metrics: dict[str, float] = {}
    for name, value in payload.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise EvolutionRuntimeError("evaluator metrics must be finite numbers")
        metrics[name] = float(value)
    return metrics


def _validate_command(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise EvolutionRuntimeError("commands must be non-empty JSON argv arrays")
    return list(value)


def _safe_environment(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvolutionRuntimeError("benchmark environment must be an object")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name or "").strip()
        if (
            not name
            or not name.replace("_", "").isalnum()
            or any(token in name.upper() for token in _SECRET_ENV_TOKENS)
        ):
            raise EvolutionRuntimeError(
                f"unsafe or secret-like benchmark environment variable: {name}"
            )
        result[name] = str(raw_value)
    return result


def _render_command(
    command: Sequence[str],
    *,
    artifact_root: Path,
    task_id: str,
    variant: str,
    output_path: Path,
) -> list[str]:
    replacements = {
        "artifact_root": str(artifact_root),
        "task_id": task_id,
        "variant": variant,
        "output_path": str(output_path),
    }
    rendered: list[str] = []
    for item in command:
        try:
            rendered.append(item.format_map(replacements))
        except KeyError as exc:
            raise EvolutionRuntimeError(
                f"unsupported command placeholder: {exc.args[0]}"
            ) from exc
    return rendered


def _evaluator_payload(stdout: bytes, output_path: Path) -> dict[str, Any]:
    try:
        raw = output_path.read_bytes() if output_path.is_file() else stdout
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvolutionRuntimeError(
            "evaluator must emit one UTF-8 JSON object to stdout or {output_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise EvolutionRuntimeError("evaluator output must be a JSON object")
    return payload


def _execute_variant(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    artifact_root: Path,
    task_id: str,
    variant: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="xscientist-eval-") as raw_work:
        work = Path(raw_work)
        output_path = work / "result.json"
        argv = _render_command(
            command,
            artifact_root=artifact_root,
            task_id=task_id,
            variant=variant,
            output_path=output_path,
        )
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "XSCIENTIST_ARTIFACT_ROOT": str(artifact_root),
            "XSCIENTIST_TASK_ID": task_id,
            "XSCIENTIST_VARIANT": variant,
            "XSCIENTIST_OUTPUT": str(output_path),
            **dict(environment),
        }
        if os.name == "nt" and os.environ.get("SystemRoot"):
            env["SystemRoot"] = os.environ["SystemRoot"]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            completed = None
            timed_out = True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""
        else:
            stdout = completed.stdout
            stderr = completed.stderr
        duration = round(time.monotonic() - started, 6)
        receipt = {
            "task_id": task_id,
            "variant": variant,
            "argv": argv,
            "environment_names": sorted(env),
            "return_code": completed.returncode if completed is not None else None,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "stdout_hash": _bytes_hash(stdout),
            "stderr_hash": _bytes_hash(stderr),
        }
        if timed_out:
            receipt["status"] = "timed_out"
            receipt["run_hash"] = canonical_content_hash(receipt)
            raise EvolutionExecutionError(
                f"benchmark task timed out: {task_id}/{variant}", receipt
            )
        if completed is None or completed.returncode != 0:
            receipt["status"] = "failed"
            receipt["run_hash"] = canonical_content_hash(receipt)
            raise EvolutionExecutionError(
                f"benchmark task failed: {task_id}/{variant} return_code="
                f"{receipt['return_code']} stderr_hash={receipt['stderr_hash']}",
                receipt,
            )
        try:
            output = _evaluator_payload(stdout, output_path)
            metrics = _finite_metrics(output.get("metrics"))
        except EvolutionRuntimeError as exc:
            receipt["status"] = "invalid_output"
            receipt["error_category"] = type(exc).__name__
            receipt["run_hash"] = canonical_content_hash(receipt)
            raise EvolutionExecutionError(
                f"benchmark task produced invalid evidence: {task_id}/{variant}: {exc}",
                receipt,
            ) from exc
        gates = {
            name: output.get(name) is True
            for name in (
                "safety_pass",
                "integrity_pass",
                "reproducibility_pass",
            )
        }
        receipt.update(
            {
                "status": "completed",
                "output_hash": canonical_content_hash(output),
                "metrics": metrics,
                **gates,
            }
        )
        receipt["run_hash"] = canonical_content_hash(receipt)
        return output, receipt


def validate_benchmark_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if suite.get("schema_version") != BENCHMARK_SUITE_SCHEMA:
        errors.append("schema_version_invalid")
    for field in (
        "benchmark_id",
        "producer_stack_id",
        "evaluator_stack_id",
        "evaluator_id",
    ):
        if not str(suite.get(field) or "").strip():
            errors.append(f"{field}_missing")
    if suite.get("producer_stack_id") == suite.get("evaluator_stack_id"):
        errors.append("evaluator_not_independent")
    if not str(suite.get("evaluator_id") or "").startswith(("service:", "human:")):
        errors.append("evaluator_identity_invalid")
    for field in ("producer_stack_hash", "evaluator_stack_hash"):
        if not _is_hash(suite.get(field)):
            errors.append(f"{field}_invalid")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks_missing")
        tasks = []
    task_ids: set[str] = set()
    for index, task in enumerate(tasks):
        prefix = f"task:{index}:"
        if not isinstance(task, Mapping):
            errors.append(prefix + "invalid")
            continue
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in task_ids:
            errors.append(prefix + "task_id_invalid")
        task_ids.add(task_id)
        if not _is_hash(task.get("task_hash")):
            errors.append(prefix + "task_hash_invalid")
        if task.get("split") not in {"hidden", "public"}:
            errors.append(prefix + "split_invalid")
        if task.get("evaluation_layer") not in {
            "sealed",
            "prospective",
            "public",
        }:
            errors.append(prefix + "evaluation_layer_invalid")
        evaluator_stack_id = task.get(
            "evaluator_stack_id", suite.get("evaluator_stack_id")
        )
        evaluator_stack_hash = task.get(
            "evaluator_stack_hash", suite.get("evaluator_stack_hash")
        )
        if evaluator_stack_id == suite.get("producer_stack_id"):
            errors.append(prefix + "evaluator_not_independent")
        if not _is_hash(evaluator_stack_hash):
            errors.append(prefix + "evaluator_stack_hash_invalid")
        try:
            _validate_command(task.get("command"))
            _safe_environment(task.get("environment"))
        except EvolutionRuntimeError:
            errors.append(prefix + "execution_contract_invalid")
    return {"ok": not errors, "errors": errors, "task_count": len(tasks)}


def run_shadow_benchmark(
    suite: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    artifact_store: str | Path,
    allow_execution: bool = False,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Execute paired baseline/candidate tasks without a shell."""

    if not allow_execution:
        raise EvolutionRuntimeError(
            "benchmark execution is disabled; pass allow_execution=True explicitly"
        )
    suite_check = validate_benchmark_suite(suite)
    if not suite_check["ok"]:
        raise EvolutionRuntimeError(
            "benchmark suite invalid: " + ", ".join(suite_check["errors"])
        )
    candidate_check = validate_evolution_candidate(dict(candidate))
    if not candidate_check["ok"]:
        raise EvolutionRuntimeError(
            "candidate invalid: " + ", ".join(candidate_check["errors"])
        )
    if timeout_seconds < 1:
        raise EvolutionRuntimeError("benchmark timeout must be positive")
    artifacts = {
        "baseline": str(candidate["base_artifact_hash"]),
        "candidate": str(candidate["candidate_artifact_hash"]),
    }
    for artifact_hash in artifacts.values():
        check = verify_evolution_artifact(artifact_store, artifact_hash)
        if not check["ok"]:
            raise EvolutionRuntimeError(
                f"benchmark artifact invalid ({artifact_hash}): "
                + ", ".join(check["errors"])
            )
    samples: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="xscientist-benchmark-") as raw_root:
        root = Path(raw_root)
        materialized: dict[str, Path] = {}
        for variant, artifact_hash in artifacts.items():
            target = root / variant
            materialize_evolution_artifact(
                artifact_store, artifact_hash, target, strip_logical_root=True
            )
            materialized[variant] = target
        for raw_task in suite["tasks"]:
            task = deepcopy(dict(raw_task))
            task_id = str(task["task_id"])
            command = _validate_command(task.pop("command"))
            environment = _safe_environment(task.pop("environment", None))
            evaluator_stack_id = task.pop(
                "evaluator_stack_id", suite["evaluator_stack_id"]
            )
            evaluator_stack_hash = task.pop(
                "evaluator_stack_hash", suite["evaluator_stack_hash"]
            )
            outputs: dict[str, dict[str, Any]] = {}
            run_receipts: dict[str, dict[str, Any]] = {}
            for variant in ("baseline", "candidate"):
                try:
                    outputs[variant], run_receipts[variant] = _execute_variant(
                        command=command,
                        environment=environment,
                        artifact_root=materialized[variant],
                        task_id=task_id,
                        variant=variant,
                        timeout_seconds=timeout_seconds,
                    )
                except EvolutionExecutionError as exc:
                    receipts.append(exc.receipt)
                    execution_errors.append(str(exc))
                    break
                receipts.append(run_receipts[variant])
            if execution_errors:
                break
            sample = {
                **task,
                "frozen_before_candidate": task.get("frozen_before_candidate") is True,
                "producer_stack_id": suite["producer_stack_id"],
                "evaluator_stack_id": evaluator_stack_id,
                "producer_stack_hash": suite["producer_stack_hash"],
                "evaluator_stack_hash": evaluator_stack_hash,
                "baseline": run_receipts["baseline"]["metrics"],
                "candidate": run_receipts["candidate"]["metrics"],
                "safety_pass": all(
                    item["safety_pass"] for item in run_receipts.values()
                ),
                "integrity_pass": all(
                    item["integrity_pass"] for item in run_receipts.values()
                ),
                "reproducibility_pass": all(
                    item["reproducibility_pass"] for item in run_receipts.values()
                ),
                "baseline_run_hash": run_receipts["baseline"]["run_hash"],
                "candidate_run_hash": run_receipts["candidate"]["run_hash"],
            }
            samples.append(sample)
    report = {
        "schema_version": BENCHMARK_RUN_SCHEMA,
        "benchmark_id": suite["benchmark_id"],
        "candidate_hash": candidate["candidate_hash"],
        "evaluator_id": suite["evaluator_id"],
        "suite_hash": canonical_content_hash(suite),
        "samples": samples,
        "run_receipts": receipts,
        "task_count": len(samples),
        "status": "failed" if execution_errors else "completed",
        "errors": execution_errors,
    }
    report["report_hash"] = canonical_content_hash(report)
    return report


def validate_canary_suite(suite: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if suite.get("schema_version") != CANARY_SPEC_SCHEMA:
        errors.append("schema_version_invalid")
    if not str(suite.get("target") or "").strip():
        errors.append("target_missing")
    if not str(suite.get("approval_id") or "").startswith("human:"):
        errors.append("approval_identity_invalid")
    projects = suite.get("projects")
    if not isinstance(projects, list) or not projects:
        errors.append("projects_missing")
        projects = []
    project_ids: set[str] = set()
    for index, project in enumerate(projects):
        prefix = f"project:{index}:"
        if not isinstance(project, Mapping):
            errors.append(prefix + "invalid")
            continue
        project_id = str(project.get("project_id") or "")
        if not project_id or project_id in project_ids:
            errors.append(prefix + "project_id_invalid")
        project_ids.add(project_id)
        baseline = project.get("baseline")
        if not isinstance(baseline, Mapping):
            errors.append(prefix + "baseline_missing")
        else:
            try:
                _finite_metrics(baseline)
            except EvolutionRuntimeError:
                errors.append(prefix + "baseline_invalid")
        try:
            _validate_command(project.get("command"))
            _safe_environment(project.get("environment"))
        except EvolutionRuntimeError:
            errors.append(prefix + "execution_contract_invalid")
    return {"ok": not errors, "errors": errors, "project_count": len(projects)}


def run_canary_suite(
    suite: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    artifact_store: str | Path,
    deployment_root: str | Path,
    executed_by: str,
    allow_execution: bool = False,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Deploy to a bounded target, run real projects, and prove rollback."""

    from ai_scientist.utils.evolution_deployment import LocalEvolutionDeployment

    if not allow_execution:
        raise EvolutionRuntimeError(
            "canary execution is disabled; pass allow_execution=True explicitly"
        )
    suite_check = validate_canary_suite(suite)
    if not suite_check["ok"]:
        raise EvolutionRuntimeError(
            "canary suite invalid: " + ", ".join(suite_check["errors"])
        )
    candidate_check = validate_evolution_candidate(dict(candidate))
    if not candidate_check["ok"]:
        raise EvolutionRuntimeError(
            "candidate invalid: " + ", ".join(candidate_check["errors"])
        )
    adapter = LocalEvolutionDeployment(
        artifact_store=artifact_store,
        deployment_root=deployment_root,
        executed_by=executed_by,
    )
    target = str(suite["target"])
    approval_id = str(suite["approval_id"])
    deployed = adapter.deploy(
        target=target,
        artifact_hash=str(candidate["candidate_artifact_hash"]),
        apply=True,
        approval_id=approval_id,
        production=False,
    )
    project_receipts: list[dict[str, Any]] = []
    errors: list[Exception] = []
    rollback: dict[str, Any] | None = None
    try:
        for project in suite["projects"]:
            project_id = str(project["project_id"])
            output, receipt = _execute_variant(
                command=_validate_command(project["command"]),
                environment=_safe_environment(project.get("environment")),
                artifact_root=Path(deployment_root).expanduser().resolve()
                / Path(target),
                task_id=project_id,
                variant="canary",
                timeout_seconds=timeout_seconds,
            )
            project_receipts.append(receipt)
            baseline = _finite_metrics(project["baseline"])
            metrics = receipt["metrics"]
            for required in ("error_rate", "quality"):
                if required not in baseline or required not in metrics:
                    raise EvolutionRuntimeError(
                        f"canary project {project_id} requires {required} metrics"
                    )
            observations = output.get("observations")
            if (
                not isinstance(observations, int)
                or isinstance(observations, bool)
                or observations < 1
            ):
                raise EvolutionRuntimeError(
                    f"canary project {project_id} requires positive observations"
                )
            incidents = output.get("incidents") or []
            if not isinstance(incidents, list) or any(
                not isinstance(item, str) for item in incidents
            ):
                raise EvolutionRuntimeError(
                    f"canary project {project_id} incidents must be strings"
                )
            receipt["project_id"] = project_id
            receipt["baseline"] = baseline
            receipt["observations"] = observations
            receipt["incidents"] = incidents
            receipt["long_tail_pass"] = output.get("long_tail_pass") is True
            receipt["common_mode_failure_pass"] = (
                output.get("common_mode_failure_pass") is True
            )
            receipt["out_of_distribution_pass"] = (
                output.get("out_of_distribution_pass") is True
            )
            receipt["project_run_hash"] = canonical_content_hash(receipt)
    except EvolutionExecutionError as exc:
        project_receipts.append(exc.receipt)
        errors.append(exc)
    except Exception as exc:
        errors.append(exc)
    finally:
        try:
            rollback = adapter.rollback(
                candidate,
                target=target,
                apply=True,
                approval_id=approval_id,
                production=False,
                trigger="canary_exercise",
            )
        except Exception as exc:
            errors.append(exc)
    if rollback is None:
        messages = "; ".join(str(item) for item in errors)
        raise EvolutionRuntimeError("mandatory canary rollback failed: " + messages)
    completed_receipts = [
        item
        for item in project_receipts
        if item.get("status") == "completed" and "baseline" in item
    ]
    error_deltas = [
        item["metrics"]["error_rate"] - item["baseline"]["error_rate"]
        for item in completed_receipts
    ]
    quality_deltas = [
        item["metrics"]["quality"] - item["baseline"]["quality"]
        for item in completed_receipts
    ]
    canary_report = build_canary_report(
        dict(candidate),
        rollback["rollback_receipt"],
        executed_by=executed_by,
        observation_count=sum(int(item["observations"]) for item in completed_receipts),
        error_rate_delta=mean(error_deltas) if error_deltas else 0.0,
        quality_delta=mean(quality_deltas) if quality_deltas else 0.0,
        real_research_project_ids=[
            str(item["project_id"]) for item in completed_receipts
        ],
        project_run_hashes={
            str(item["project_id"]): str(item["project_run_hash"])
            for item in completed_receipts
        },
        incidents=[
            incident for item in completed_receipts for incident in item["incidents"]
        ],
        long_tail_pass=not errors
        and all(item["long_tail_pass"] for item in completed_receipts),
        common_mode_failure_pass=all(
            item["common_mode_failure_pass"] for item in completed_receipts
        )
        and not errors,
        out_of_distribution_pass=all(
            item["out_of_distribution_pass"] for item in completed_receipts
        )
        and not errors,
    )
    result = {
        "schema_version": "xscientist.canary-run.v1",
        "candidate_hash": candidate["candidate_hash"],
        "canary_report": canary_report,
        "project_receipts": project_receipts,
        "deployment_receipt": deployed,
        "rollback_deployment_receipt": rollback["deployment"],
        "status": canary_report["status"],
        "errors": [str(item) for item in errors],
    }
    result["report_hash"] = canonical_content_hash(result)
    return result


__all__ = [
    "BENCHMARK_RUN_SCHEMA",
    "BENCHMARK_SUITE_SCHEMA",
    "CANARY_SPEC_SCHEMA",
    "EvolutionExecutionError",
    "EvolutionRuntimeError",
    "run_canary_suite",
    "run_shadow_benchmark",
    "validate_canary_suite",
    "validate_benchmark_suite",
]
