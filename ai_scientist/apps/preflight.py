#!/usr/bin/env python3
"""XScientist preflight checks."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_scientist.config.paths import (  # noqa: E402
    LEGACY_OUTPUT_ENV_VAR,
    PRIMARY_OUTPUT_ENV_VAR,
    resolve_output_path,
)
from ai_scientist.resources import resolve_bfts_config_path  # noqa: E402
from ai_scientist.utils.auth_session import (  # noqa: E402
    auth_file_path,
    touch_session,
    validate_session,
)
from ai_scientist.utils.provider_registry import (  # noqa: E402
    describe_model_requirements,
    provider_env_statuses,
    resolve_model_provider,
)
from ai_scientist.utils.privacy import (  # noqa: E402
    portable_path,
    redact_sensitive_text,
)

import yaml  # noqa: E402


@dataclass
class CheckResult:
    label: str
    ok: bool
    severity: str
    detail: str


CORE_PACKAGES = {
    "backoff": "retry support",
    "black": "generated-code formatting",
    "numpy": "numeric utilities",
    "omegaconf": "BFTS config loading",
    "pandas": "data preview",
    "rich": "CLI progress display",
    "sklearn": "adaptive learning similarity",
}

PIPELINE_PACKAGES = {
    "requests": "Semantic Scholar / HTTP fallbacks",
    "yaml": "YAML config editing",
    "psutil": "process cleanup",
    "PIL": "vision review",
    "huggingface_hub": "dataset/model bootstrap",
}

COMMANDS = {
    "pdflatex": "LaTeX PDF compilation",
    "chktex": "LaTeX linting",
}

MIN_PYTHON = (3, 10)


def check_module(name: str, description: str, severity: str) -> CheckResult:
    installed = importlib.util.find_spec(name) is not None
    return CheckResult(
        label=f"Python package `{name}`",
        ok=installed,
        severity=severity,
        detail=description,
    )


def check_command(name: str, description: str) -> CheckResult:
    available = shutil.which(name) is not None
    return CheckResult(
        label=f"Command `{name}`",
        ok=available,
        severity="warning",
        detail=description,
    )


def check_python_version() -> CheckResult:
    minimum = ".".join(str(part) for part in MIN_PYTHON)
    current = sys.version.split()[0]
    supported = sys.version_info >= MIN_PYTHON
    detail = f"current {current}, require >= {minimum}"
    return CheckResult(
        label="Python version",
        ok=supported,
        severity="error",
        detail=detail,
    )


def check_output_dir() -> CheckResult:
    output_path = resolve_output_path()
    display_path = portable_path(output_path, base=Path.cwd())
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        probe = output_path / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult(
            label="Output directory",
            ok=True,
            severity="error",
            detail=f"resolved to {display_path}",
        )
    except OSError as exc:
        return CheckResult(
            label="Output directory",
            ok=False,
            severity="error",
            detail=redact_sensitive_text(f"cannot write to {display_path}: {exc}"),
        )


def check_login_session() -> CheckResult:
    ok, reason, _session = validate_session()
    auth_path = portable_path(auth_file_path(), base=Path.cwd())
    detail = f"{reason}; session file={auth_path}"
    if ok:
        touch_session()
    return CheckResult(
        label="Login session",
        ok=ok,
        severity="error",
        detail=detail,
    )


def check_provider_envs() -> list[CheckResult]:
    results = []
    any_provider_configured = False

    for status in provider_env_statuses(os.environ):
        configured = status.configured
        any_provider_configured |= configured and status.counts_as_configured_provider
        results.append(
            CheckResult(
                label=f"Provider `{status.display_name}`",
                ok=configured,
                severity="warning",
                detail=status.detail,
            )
        )

    if not any_provider_configured:
        results.append(
            CheckResult(
                label="Model credentials",
                ok=False,
                severity="warning",
                detail="no complete provider credential set detected",
            )
        )

    return results


def _configured_models(payload: dict) -> list[str]:
    models: list[str] = []
    report = payload.get("report")
    if isinstance(report, dict):
        models.append(str(report.get("model") or "").strip())
    agent = payload.get("agent")
    if isinstance(agent, dict):
        for key in ("code", "feedback", "vlm_feedback"):
            section = agent.get(key)
            if isinstance(section, dict):
                models.append(str(section.get("model") or "").strip())
    return list(dict.fromkeys(model for model in models if model))


def _model_client_module(model: str) -> str:
    spec = resolve_model_provider(model)
    if spec.provider == "zhipu":
        return "zhipuai"
    if spec.client_family.startswith("anthropic"):
        return "anthropic"
    return "openai"


def check_configured_models(payload: dict) -> list[CheckResult]:
    models = _configured_models(payload)
    if not models:
        return [
            CheckResult(
                label="BFTS models",
                ok=False,
                severity="error",
                detail="no report or agent model IDs found in the configuration",
            )
        ]

    results: list[CheckResult] = []
    for model in models:
        try:
            client_module = _model_client_module(model)
            requirement_rows = describe_model_requirements([model], env=os.environ)
        except ValueError as exc:
            results.append(
                CheckResult(
                    label=f"Configured model `{model}`",
                    ok=False,
                    severity="error",
                    detail=redact_sensitive_text(str(exc)),
                )
            )
            continue
        row = requirement_rows[0] if requirement_rows else {}
        missing_credentials = str(row.get("missing") or "").strip()
        client_installed = importlib.util.find_spec(client_module) is not None
        problems = []
        if missing_credentials:
            problems.append(f"missing credentials: {missing_credentials}")
        if not client_installed:
            problems.append(f"missing Python package: {client_module}")
        results.append(
            CheckResult(
                label=f"Configured model `{model}`",
                ok=not problems,
                severity="error",
                detail=(
                    "; ".join(problems)
                    if problems
                    else f"provider and {client_module} client are ready"
                ),
            )
        )
    return results


def _check_docker_image(image: str) -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker executable not found"
    try:
        daemon = subprocess.run(
            [docker, "info", "--format", "{{json .ServerVersion}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker daemon check failed: {exc}"
    if daemon.returncode != 0:
        return False, "docker daemon is not reachable"
    try:
        inspected = subprocess.run(
            [docker, "image", "inspect", image],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"docker image check failed: {exc}"
    if inspected.returncode != 0:
        return False, f"docker image is not available locally: {image}"
    return True, f"docker image is available locally: {image}"


def check_bfts_config(value: str) -> list[CheckResult]:
    try:
        path = resolve_bfts_config_path(value)
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("configuration root must be a mapping")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [
            CheckResult(
                label="BFTS configuration",
                ok=False,
                severity="error",
                detail=redact_sensitive_text(str(exc)),
            )
        ]

    results = [
        CheckResult(
            label="BFTS configuration",
            ok=True,
            severity="error",
            detail=f"loaded {portable_path(path, base=Path.cwd())}",
        )
    ]
    results.extend(check_configured_models(payload))

    exec_config = payload.get("exec")
    if not isinstance(exec_config, dict):
        results.append(
            CheckResult(
                label="Experiment isolation",
                ok=False,
                severity="error",
                detail="missing exec configuration",
            )
        )
        return results
    backend = str(exec_config.get("backend") or "auto").strip().lower()
    require_isolation = bool(exec_config.get("require_isolation", True))
    if backend not in {"auto", "process", "docker"}:
        results.append(
            CheckResult(
                label="Experiment isolation",
                ok=False,
                severity="error",
                detail=f"unsupported execution backend: {backend}",
            )
        )
        return results
    if backend == "process":
        results.append(
            CheckResult(
                label="Experiment isolation",
                ok=False,
                severity="error" if require_isolation else "warning",
                detail=(
                    "backend=process conflicts with require_isolation=true"
                    if require_isolation
                    else "AI-generated code will execute in the host Python process"
                ),
            )
        )
        return results

    image = str(exec_config.get("docker_image") or "").strip()
    if not image:
        results.append(
            CheckResult(
                label="Experiment isolation",
                ok=False,
                severity="error",
                detail="docker_image is required for the selected backend",
            )
        )
        return results
    docker_ok, detail = _check_docker_image(image)
    results.append(
        CheckResult(
            label="Experiment isolation",
            ok=docker_ok,
            severity=(
                "error" if backend == "docker" or require_isolation else "warning"
            ),
            detail=detail,
        )
    )
    return results


def print_result(result: CheckResult) -> None:
    marker = "OK" if result.ok else result.severity.upper()
    print(f"[{marker}] {result.label}: {result.detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="XScientist preflight checks")
    parser.add_argument(
        "--auth-file",
        help="override auth session file for this preflight run",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return non-zero when critical blockers are found",
    )
    parser.add_argument(
        "--bfts-config",
        help="validate one BFTS config, its models, credentials, and isolation image",
    )
    args = parser.parse_args(argv)
    if args.auth_file:
        os.environ["AI_SCIENTIST_AUTH_FILE"] = str(
            Path(args.auth_file).expanduser().resolve()
        )

    print("XScientist preflight")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Output env: {PRIMARY_OUTPUT_ENV_VAR} (legacy: {LEGACY_OUTPUT_ENV_VAR})")
    print("- Auth session: " f"{portable_path(auth_file_path(), base=Path.cwd())}")

    results = [check_python_version(), check_output_dir(), check_login_session()]
    results.extend(
        check_module(name, desc, "error") for name, desc in CORE_PACKAGES.items()
    )
    results.extend(
        check_module(name, desc, "warning") for name, desc in PIPELINE_PACKAGES.items()
    )
    results.extend(check_command(name, desc) for name, desc in COMMANDS.items())
    results.extend(check_provider_envs())
    if args.bfts_config:
        results.extend(check_bfts_config(args.bfts_config))

    errors = 0
    warnings = 0
    for result in results:
        print_result(result)
        if result.ok:
            continue
        if result.severity == "error":
            errors += 1
        else:
            warnings += 1

    print(
        f"\nSummary: {len(results) - errors - warnings} ok, {warnings} warnings, {errors} errors"
    )
    if errors:
        print("Next actions:")
        if any(
            not result.ok and result.label.startswith("Python package")
            for result in results
        ):
            provider = str(os.environ.get("AI_SCIENTIST_ACTIVE_PROVIDER") or "").strip()
            if provider:
                from xscientist.dependency_profiles import installation_command

                command = installation_command(provider)
            else:
                command = 'python -m pip install "xscientist[research]"'
            print(f"  - Install the selected research runtime: {command}")
        if any(not result.ok and result.label == "Login session" for result in results):
            print("  - Create a login: xscientist auth login --user <your-name>")
        if any(
            not result.ok and result.label == "Experiment isolation"
            for result in results
        ):
            print("  - Build or select the Docker image named in the BFTS config")
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
