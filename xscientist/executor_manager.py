"""Version-matched Docker executor inspection and cache management."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import yaml

from ._version import __version__

EXECUTOR_SCHEMA = "xscientist.executor-status.v1"
DOCKER_INSTALL_URL = "https://docs.docker.com/get-started/get-docker/"


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
    if not docker_available:
        error = f"Docker CLI is not installed. Install Docker: {DOCKER_INSTALL_URL}"
    elif not daemon_ready:
        error = "Docker is installed, but its daemon is not running; start Docker"
    elif error is None and image_available and not version_match:
        error = f"executor version {version or 'unknown'} does not match {__version__}"
    if ready:
        next_action = None
    elif not docker_available:
        next_action = DOCKER_INSTALL_URL
    elif not daemon_ready:
        next_action = "Start Docker, then rerun this command"
    else:
        next_action = f"xscientist executor prepare --workspace {root.name}"
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
        "next_action": next_action,
        "host_paths_disclosed": False,
    }


def _source_build_arguments(_workspace: Path) -> tuple[Path | None, list[str]]:
    """Return a source checkout context, or ``None`` for an installed package.

    An installed package must not use the research workspace as its Docker build
    context. Workspaces can contain private datasets and complete Git histories;
    Docker sends the whole context to the daemon before evaluating ``COPY``.
    ``build_executor`` replaces the ``None`` context with a temporary directory
    containing only the explicitly selected Dockerfile.
    """

    source_root = Path(__file__).resolve().parents[1]
    local_source = (source_root / "pyproject.toml").is_file() and (
        source_root / "xscientist"
    ).is_dir()
    if not local_source:
        return None, []
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
        raise ExecutorManagerError(
            f"Docker CLI is not installed. Install Docker: {DOCKER_INSTALL_URL}"
        )
    image = _image_name(config)
    context, source_args = _source_build_arguments(root)

    def run_build(*, build_context: Path, selected_dockerfile: Path) -> None:
        command = [
            "docker",
            "build",
            "-f",
            str(selected_dockerfile),
            "-t",
            image,
        ]
        if pull_base:
            command.append("--pull")
        command.extend(source_args)
        command.append(str(build_context))
        completed = run(command, cwd=build_context, text=True, check=False)
        if completed.returncode:
            raise ExecutorManagerError(
                f"executor build failed with return code {completed.returncode}"
            )

    if context is None:
        # The Dockerfile's PyPI branch does not consume workspace files. Keep it
        # inside an otherwise empty, private context so Docker never receives the
        # research workspace merely because it needs to parse the Dockerfile.
        with tempfile.TemporaryDirectory(prefix="xscientist-executor-build-") as raw:
            safe_context = Path(raw)
            safe_dockerfile = safe_context / "Dockerfile.executor"
            shutil.copyfile(dockerfile, safe_dockerfile)
            run_build(
                build_context=safe_context,
                selected_dockerfile=safe_dockerfile,
            )
    else:
        run_build(build_context=context, selected_dockerfile=dockerfile)
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
    "DOCKER_INSTALL_URL",
    "EXECUTOR_SCHEMA",
    "ExecutorManagerError",
    "build_executor",
    "inspect_executor",
    "prepare_executor",
]
