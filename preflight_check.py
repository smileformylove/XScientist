#!/usr/bin/env python3
"""XScientist preflight checks."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_scientist.config.paths import (  # noqa: E402
    LEGACY_OUTPUT_ENV_VAR,
    PRIMARY_OUTPUT_ENV_VAR,
    resolve_output_path,
)
from ai_scientist.utils.auth_session import (  # noqa: E402
    auth_file_path,
    touch_session,
    validate_session,
)
from ai_scientist.utils.provider_registry import provider_env_statuses  # noqa: E402


@dataclass
class CheckResult:
    label: str
    ok: bool
    severity: str
    detail: str


CORE_PACKAGES = {
    "openai": "OpenAI client",
    "anthropic": "Anthropic client",
    "backoff": "retry support",
    "numpy": "numeric utilities",
    "omegaconf": "BFTS config loading",
    "rich": "CLI progress display",
}

PIPELINE_PACKAGES = {
    "requests": "Semantic Scholar / HTTP fallbacks",
    "yaml": "YAML config editing",
    "psutil": "process cleanup",
    "pandas": "data preview",
    "sklearn": "adaptive learning similarity",
    "PIL": "vision review",
    "huggingface_hub": "dataset/model bootstrap",
    "zhipuai": "Zhipu backend",
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
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        probe = output_path / ".preflight_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult(
            label="Output directory",
            ok=True,
            severity="error",
            detail=f"resolved to {output_path}",
        )
    except OSError as exc:
        return CheckResult(
            label="Output directory",
            ok=False,
            severity="error",
            detail=f"cannot write to {output_path}: {exc}",
        )


def check_login_session() -> CheckResult:
    ok, reason, _session = validate_session()
    auth_path = auth_file_path()
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
        any_provider_configured |= (
            configured and status.counts_as_configured_provider
        )
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


def print_result(result: CheckResult) -> None:
    marker = "OK" if result.ok else result.severity.upper()
    print(f"[{marker}] {result.label}: {result.detail}")


def main() -> int:
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
    args = parser.parse_args()
    if args.auth_file:
        os.environ["AI_SCIENTIST_AUTH_FILE"] = str(
            Path(args.auth_file).expanduser().resolve()
        )

    print("XScientist preflight")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Output env: {PRIMARY_OUTPUT_ENV_VAR} (legacy: {LEGACY_OUTPUT_ENV_VAR})")
    print(f"- Auth session: {auth_file_path()}")

    results = [check_python_version(), check_output_dir(), check_login_session()]
    results.extend(
        check_module(name, desc, "error") for name, desc in CORE_PACKAGES.items()
    )
    results.extend(
        check_module(name, desc, "warning") for name, desc in PIPELINE_PACKAGES.items()
    )
    results.extend(check_command(name, desc) for name, desc in COMMANDS.items())
    results.extend(check_provider_envs())

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
    if args.strict and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
