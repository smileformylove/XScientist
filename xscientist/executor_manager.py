"""Version-matched Docker executor inspection and cache management."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import yaml

from ._version import __version__

EXECUTOR_SCHEMA = "xscientist.executor-status.v1"


class ExecutorManagerError(RuntimeError):
    """Raised when executor metadata or Docker operations are invalid."""


def _workspace_config(workspace: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    path = root / "bfts_config.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExecutorManagerError("bfts_config.yaml was not found") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ExecutorManagerError(
            f"cannot read bfts_config.yaml: {type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise ExecutorManagerError("bfts_config.yaml must contain a mapping")
    return root, payload


def _image_name(config: dict[str, Any]) -> str:
    exec_config = config.get("exec")
    image = str(
        exec_config.get("docker_image") if isinstance(exec_config, dict) else ""
    )
    if not image:
        image = f"xscientist-exec:{__version__}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]{0,254}", image):
        raise ExecutorManagerError("configured executor image name is invalid")
    return image


def inspect_executor(
    workspace: str | Path,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root, config = _workspace_config(workspace)
    image = _image_name(config)
    docker_available = shutil.which("docker") is not None
    labels: dict[str, str] = {}
    daemon_ready = False
    image_available = False
    error = None
    if docker_available:
        info = run(
            ["docker", "info", "--format", "{{json .ServerVersion}}"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        daemon_ready = info.returncode == 0
        if daemon_ready:
            inspected = run(
                [
                    "docker",
                    "image",
                    "inspect",
                    image,
                    "--format",
                    "{{json .Config.Labels}}",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )
            image_available = inspected.returncode == 0
            if image_available:
                try:
                    parsed = json.loads(inspected.stdout.strip() or "{}")
                    if isinstance(parsed, dict):
                        labels = {str(key): str(value) for key, value in parsed.items()}
                except json.JSONDecodeError:
                    error = "executor image labels are not valid JSON"
    version = labels.get("org.opencontainers.image.version")
    source = labels.get("org.xscientist.install-source")
    version_match = image_available and version == __version__
    ready = docker_available and daemon_ready and image_available and version_match
    if error is None and image_available and not version_match:
        error = f"executor version {version or 'unknown'} does not match {__version__}"
    return {
        "schema": EXECUTOR_SCHEMA,
        "ok": ready,
        "workspace": root.name,
        "image": image,
        "docker_available": docker_available,
        "daemon_ready": daemon_ready,
        "image_available": image_available,
        "version": version,
        "version_match": version_match,
        "revision": labels.get("org.opencontainers.image.revision"),
        "install_source": source,
        "error": error,
        "next_action": (
            None if ready else f"xscientist executor prepare --workspace {root.name}"
        ),
        "host_paths_disclosed": False,
    }


def _source_build_arguments(workspace: Path) -> tuple[Path, list[str]]:
    source_root = Path(__file__).resolve().parents[1]
    local_source = (source_root / "pyproject.toml").is_file() and (
        source_root / "xscientist"
    ).is_dir()
    if not local_source:
        return workspace, []
    revision_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    dirty_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    revision = revision_result.stdout.strip() or "local-source"
    if dirty_result.stdout.strip():
        revision += "-dirty"
    return source_root, [
        "--build-arg",
        "XSCIENTIST_INSTALL_MODE=local",
        "--build-arg",
        f"XSCIENTIST_SOURCE_REVISION={revision}",
        "--build-arg",
        "XSCIENTIST_INSTALL_SOURCE=local-source",
    ]


def build_executor(
    workspace: str | Path,
    *,
    pull_base: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    root, config = _workspace_config(workspace)
    dockerfile = root / "Dockerfile.executor"
    if not dockerfile.is_file():
        raise ExecutorManagerError("Dockerfile.executor was not found")
    if shutil.which("docker") is None:
        raise ExecutorManagerError("Docker CLI is not installed")
    image = _image_name(config)
    context, source_args = _source_build_arguments(root)
    command = ["docker", "build", "-f", str(dockerfile), "-t", image]
    if pull_base:
        command.append("--pull")
    command.extend(source_args)
    command.append(str(context))
    completed = run(command, cwd=context, text=True, check=False)
    if completed.returncode:
        raise ExecutorManagerError(
            f"executor build failed with return code {completed.returncode}"
        )
    return inspect_executor(root, run=run)


def prepare_executor(
    workspace: str | Path,
    *,
    update: bool = False,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    status = inspect_executor(workspace, run=run)
    if status["ok"] and not update:
        return {**status, "cache_hit": True, "built": False}
    built = build_executor(workspace, pull_base=update, run=run)
    return {**built, "cache_hit": False, "built": True}


__all__ = [
    "EXECUTOR_SCHEMA",
    "ExecutorManagerError",
    "build_executor",
    "inspect_executor",
    "prepare_executor",
]
