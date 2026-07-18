"""Cross-process ownership lock for an experiment directory."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOCK_DIR_NAME = ".xscientist-experiment.lock"
OWNER_FILE_NAME = "owner.json"
SENSITIVE_COMMAND_MARKERS = (
    "api-key",
    "api_key",
    "auth",
    "credential",
    "password",
    "secret",
    "token",
)


def _non_root_path(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    return None if resolved == Path(resolved.anchor) else resolved


def experiment_lock_root(config_path: str | Path) -> Path:
    """Resolve the shared idea directory guarded by an experiment config."""

    config_path = Path(config_path).expanduser().resolve()
    config_dir = config_path.parent
    if config_dir.name == "configs" and config_dir.parent.name == ".xscientist":
        layout_root = _non_root_path(config_dir.parent.parent)
        if layout_root is not None:
            return layout_root

    try:
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raw_config = None
    if isinstance(raw_config, dict) and raw_config.get("workspace_dir"):
        workspace_path = Path(str(raw_config["workspace_dir"])).expanduser()
        if not workspace_path.is_absolute():
            workspace_path = config_dir / workspace_path
        workspace_root = _non_root_path(workspace_path)
        if workspace_root is not None:
            return workspace_root

    fallback = _non_root_path(config_dir)
    if fallback is not None:
        return fallback

    cwd = _non_root_path(Path.cwd())
    if cwd is None:
        raise ValueError(
            f"Cannot derive a safe experiment lock root from {config_path}"
        )
    return cwd


def _sanitized_command(argv: list[str] | None = None) -> list[str]:
    command = list(sys.argv if argv is None else argv)
    sanitized: list[str] = []
    redact_next = False
    for arg in command:
        if redact_next:
            sanitized.append("[REDACTED]")
            redact_next = False
            continue
        flag, separator, _value = arg.partition("=")
        sensitive = flag.startswith("-") and any(
            marker in flag.lower() for marker in SENSITIVE_COMMAND_MARKERS
        )
        if sensitive and separator:
            sanitized.append(f"{flag}=[REDACTED]")
        else:
            sanitized.append(arg)
            redact_next = sensitive
    return sanitized


class ExperimentRunLocked(RuntimeError):
    def __init__(self, lock_path: Path, owner: dict[str, Any]):
        self.lock_path = Path(lock_path)
        self.owner = dict(owner)
        pid = self.owner.get("pid", "unknown")
        host = self.owner.get("hostname", "unknown")
        super().__init__(
            f"Experiment directory is already locked by pid {pid} on {host}"
        )


def _pid_is_alive(pid: Any) -> bool:
    try:
        parsed = int(pid)
    except (TypeError, ValueError):
        return False
    if parsed <= 0:
        return False
    try:
        os.kill(parsed, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_owner(lock_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads((lock_dir / OWNER_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_stale(lock_dir: Path, owner: dict[str, Any]) -> bool:
    local_host = socket.gethostname()
    if owner.get("hostname") == local_host and owner.get("pid") is not None:
        return not _pid_is_alive(owner.get("pid"))
    if owner:
        return False
    try:
        return time.time() - lock_dir.stat().st_mtime > 30.0
    except OSError:
        return True


@dataclass
class ExperimentRunLock:
    root: Path
    config_path: Path | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.lock_dir = self.root / LOCK_DIR_NAME
        self.token = uuid.uuid4().hex
        self.acquired = False

    def _owner_payload(self) -> dict[str, Any]:
        return {
            "schema": "xscientist.experiment-lock.v1",
            "token": self.token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),
            "config_path": str(self.config_path) if self.config_path else None,
            "command": _sanitized_command(),
        }

    def acquire(self) -> "ExperimentRunLock":
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                self.lock_dir.mkdir()
            except FileExistsError:
                owner = _read_owner(self.lock_dir)
                if not _is_stale(self.lock_dir, owner):
                    raise ExperimentRunLocked(self.lock_dir, owner)
                stale_dir = self.root / f"{LOCK_DIR_NAME}.stale-{uuid.uuid4().hex}"
                try:
                    self.lock_dir.rename(stale_dir)
                except (FileNotFoundError, FileExistsError):
                    continue
                shutil.rmtree(stale_dir, ignore_errors=True)
                continue

            owner_path = self.lock_dir / OWNER_FILE_NAME
            temp_path = owner_path.with_suffix(".tmp")
            try:
                temp_path.write_text(
                    json.dumps(self._owner_payload(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                temp_path.replace(owner_path)
            except BaseException:
                shutil.rmtree(self.lock_dir, ignore_errors=True)
                raise
            self.acquired = True
            return self
        raise ExperimentRunLocked(self.lock_dir, _read_owner(self.lock_dir))

    def release(self) -> None:
        if not self.acquired:
            return
        owner = _read_owner(self.lock_dir)
        if owner.get("token") == self.token:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
        self.acquired = False

    def __enter__(self) -> "ExperimentRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.release()
        return False
