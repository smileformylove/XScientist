"""Persistent, local control plane for detached XScientist runs."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ai_scientist.utils.privacy import (
    redact_sensitive_payload,
    redact_sensitive_text,
)

RUN_SCHEMA = "xscientist.local-run.v1"
RUN_STATE_ENV = "XSCIENTIST_RUN_STATE"
DETACHED_STARTUP_GRACE_SECONDS = 1.0
_DARWIN_PROC_PIDTBSDINFO = 3
_DARWIN_PROC_BSDINFO_SIZE = 136
_DARWIN_ZOMBIE_STATUS = 5
_MAX_SIGNED_PID = (1 << 31) - 1
_DARWIN_PROCESS_IDENTITY_DOMAIN = b"xscientist.darwin-process-identity.v1\0"
_LINUX_PROCESS_IDENTITY_DOMAIN = b"xscientist.linux-process-identity.v1\0"
_TRUSTED_PS_EXECUTABLES = (Path("/bin/ps"), Path("/usr/bin/ps"))


class _DarwinProcBSDInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class RunControlError(RuntimeError):
    """Raised when a persisted run cannot be controlled safely."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_dir(workspace: str | Path, *, create: bool = False) -> Path:
    root = Path(workspace).expanduser().resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    cursor = root
    for part in ("04_logs", "runs"):
        cursor = cursor / part
        if create:
            cursor.mkdir(mode=0o700, exist_ok=True)
        if cursor.is_symlink():
            raise RunControlError("run control directory must not be a symlink")
        if cursor.exists() and not cursor.is_dir():
            raise RunControlError("run control path is not a directory")
    return cursor


def _require_workspace(workspace: str | Path) -> Path:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise RunControlError("workspace directory does not exist")
    return root


def _state_path(workspace: str | Path, run_id: str) -> Path:
    normalized = str(run_id or "").strip()
    if not normalized or any(char not in "0123456789abcdef" for char in normalized):
        raise RunControlError("run id must contain lowercase hexadecimal characters")
    path = _state_dir(workspace) / f"{normalized}.json"
    if path.is_symlink():
        raise RunControlError("run state must not be a symlink")
    if path.exists() and not path.is_file():
        raise RunControlError("run state is not a regular file")
    return path


def _read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunControlError("run not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RunControlError(f"run state is unreadable: {type(exc).__name__}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != RUN_SCHEMA:
        raise RunControlError("run state has an unsupported schema")
    return payload


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    for parent in (path.parent.parent, path.parent):
        if parent.is_symlink() or not parent.is_dir():
            raise RunControlError("run control directory failed integrity validation")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RunControlError("run state failed integrity validation")
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(raw_temp)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):  # pragma: no cover - platform dependent
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover - platform dependent
            pass
    finally:
        temp.unlink(missing_ok=True)


def _open_private_log(path: Path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise RunControlError("run log failed integrity validation")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return os.fdopen(descriptor, "a", encoding="utf-8")


def _pid_is_alive(pid: Any, *, hostname: str | None = None) -> bool:
    if hostname and hostname != socket.gethostname():
        return True
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
    except OSError:
        return False
    return True


def _darwin_process_metadata(pid: int) -> tuple[int, int, bytes, bytes] | None:
    """Return stable kernel-owned metadata when Darwin cannot execute ``ps``."""

    if (
        sys.platform != "darwin"
        or type(pid) is not int
        or pid <= 0
        or pid > _MAX_SIGNED_PID
        or ctypes.sizeof(_DarwinProcBSDInfo) != _DARWIN_PROC_BSDINFO_SIZE
    ):
        return None
    info = _DarwinProcBSDInfo()
    try:
        proc_pidinfo = ctypes.CDLL(
            "/usr/lib/libproc.dylib", use_errno=True
        ).proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        size = ctypes.sizeof(info)
        received = proc_pidinfo(
            pid,
            _DARWIN_PROC_PIDTBSDINFO,
            0,
            ctypes.byref(info),
            size,
        )
    except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
        return None
    if (
        received != size
        or info.pbi_pid != pid
        or info.pbi_status == _DARWIN_ZOMBIE_STATUS
        or info.pbi_start_tvsec <= 0
        or info.pbi_start_tvusec >= 1_000_000
    ):
        return None
    command = bytes(info.pbi_comm).split(b"\0", 1)[0]
    name = bytes(info.pbi_name).split(b"\0", 1)[0]
    return (
        int(info.pbi_start_tvsec),
        int(info.pbi_start_tvusec),
        command,
        name,
    )


def _ps_process_identity(pid: int) -> str | None:
    for executable in _TRUSTED_PS_EXECUTABLES:
        try:
            completed = subprocess.run(
                [
                    str(executable),
                    "-p",
                    str(pid),
                    "-o",
                    "lstart=",
                    "-o",
                    "command=",
                ],
                text=True,
                capture_output=True,
                check=False,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        value = completed.stdout.strip()
        if completed.returncode == 0 and value:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return None


def _darwin_process_identity(pid: int) -> str | None:
    metadata = _darwin_process_metadata(pid)
    if metadata is None:
        return None
    seconds, microseconds, command, name = metadata
    payload = _DARWIN_PROCESS_IDENTITY_DOMAIN + struct.pack(
        ">IQQHH", pid, seconds, microseconds, len(command), len(name)
    )
    return hashlib.sha256(payload + command + name).hexdigest()


def _linux_process_identity(pid: int) -> str | None:
    """Hash Linux's kernel-owned process start tick to avoid command drift."""

    if sys.platform != "linux":
        return None
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    closing = raw.rfind(")")
    if closing < 0:
        return None
    fields = raw[closing + 2 :].split()
    # After the comm field, field 22 (starttime) is offset 19.
    if len(fields) <= 19 or fields[0] == "Z" or not fields[19].isdigit():
        return None
    payload = _LINUX_PROCESS_IDENTITY_DOMAIN + struct.pack(">IQ", pid, int(fields[19]))
    return hashlib.sha256(payload).hexdigest()


def _process_identity(pid: int) -> str | None:
    """Hash OS-owned process-start metadata to guard against PID reuse."""

    if type(pid) is not int or pid <= 0 or pid > _MAX_SIGNED_PID:
        return None
    if os.name != "posix":  # pragma: no cover - Windows has no portable source
        return None
    if sys.platform == "linux":
        linux_identity = _linux_process_identity(pid)
        if linux_identity is not None:
            return linux_identity
    legacy = _ps_process_identity(pid)
    if legacy is not None or sys.platform != "darwin":
        return legacy
    return _darwin_process_identity(pid)


def _process_identity_matches(pid: int, expected: str) -> bool | None:
    """Compare across every available process-identity provider.

    Darwin can observe the same process through either ``ps`` or libproc.  A
    match from either provider proves ownership, while a mismatch proves PID
    reuse only when both providers are available.  Missing evidence remains
    indeterminate so callers can fail closed.
    """

    if (
        type(pid) is not int
        or pid <= 0
        or pid > _MAX_SIGNED_PID
        or not isinstance(expected, str)
        or not expected
    ):
        return None
    preferred = _process_identity(pid)
    if preferred == expected:
        return True
    if os.name != "posix":  # pragma: no cover - Windows has no portable source
        return None
    if sys.platform != "darwin":
        return None if preferred is None else False

    ps_identity = _ps_process_identity(pid)
    if ps_identity == expected:
        return True
    native_identity = _darwin_process_identity(pid)
    if native_identity == expected:
        return True
    if ps_identity is None or native_identity is None:
        return None
    return False


_PUBLIC_VALUE_FLAGS = {
    "--autopilot",
    "--task",
    "--provider",
    "--model",
    "--judgment-model",
    "--profile",
    "--max-project-tokens",
    "--max-project-hours",
    "--max-cost-usd",
    "--price-input-per-million",
    "--price-output-per-million",
    "--price-cached-input-per-million",
}
_PUBLIC_BOOLEAN_FLAGS = {
    "--allow-synthetic-data",
    "--build-executor",
    "--skip-credentials",
    "--non-interactive",
    "--force",
    "--prepare-only",
    "--json",
}
_PUBLIC_PLACEHOLDER_FLAGS = {
    "--question": "<stored-in-workspace>",
    "--user": "<redacted>",
    "--data-dir": "{data}",
    "--base-url": "<redacted>",
}


def _safe_public_value(flag: str, value: str) -> str:
    if redact_sensitive_text(value) != value:
        return "<redacted>"
    if flag.startswith("--max-") or flag.startswith("--price-"):
        try:
            float(value)
        except ValueError:
            return "<redacted>"
        return value
    if (
        not re.fullmatch(r"[A-Za-z0-9._:+/-]{1,160}", value)
        or ".." in value
        or "://" in value
        or value.startswith(("/", "\\"))
    ):
        return "<redacted>"
    return value


def _public_command(argv: Sequence[str]) -> list[str]:
    """Render a portable allowlisted command; exact argv stays in the manifest."""

    values = [str(item) for item in argv]
    if not values or values[0] != "start":
        return ["start", "{workspace}"]
    rendered = ["start", "{workspace}"]
    index = 2  # argv[1] is the positional workspace and is never public.
    while index < len(values):
        item = values[index]
        inline_flag, separator, inline_value = item.partition("=")
        flag = inline_flag if separator and inline_flag.startswith("--") else item
        if flag in _PUBLIC_BOOLEAN_FLAGS:
            rendered.append(flag)
            index += 1
            continue
        if flag in _PUBLIC_PLACEHOLDER_FLAGS or flag in _PUBLIC_VALUE_FLAGS:
            if separator:
                value = inline_value
                consumed = 1
            elif index + 1 < len(values):
                value = values[index + 1]
                consumed = 2
            else:
                value = ""
                consumed = 1
            public_value = _PUBLIC_PLACEHOLDER_FLAGS.get(flag)
            if public_value is None:
                public_value = _safe_public_value(flag, value)
            rendered.extend([flag, public_value])
            index += consumed
            continue
        if item.startswith("--"):
            # Preserve the presence of a future flag, never its unknown value.
            rendered.append(item.split("=", 1)[0])
            if "=" not in item and index + 1 < len(values):
                if not values[index + 1].startswith("--"):
                    index += 1
            index += 1
            continue
        # Additional positionals are outside the public start contract.
        rendered.append("<redacted>")
        index += 1
    return rendered


def _argv_option(argv: Sequence[str], flag: str) -> str | None:
    values = [str(item) for item in argv]
    inline_prefix = flag + "="
    for item in values:
        if item.startswith(inline_prefix):
            return item[len(inline_prefix) :].strip() or None
    try:
        index = values.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(values):
        return None
    return values[index + 1].strip() or None


def _run_descriptor(workspace: Path, argv: Sequence[str]) -> dict[str, str | None]:
    provider = _argv_option(argv, "--provider")
    model = _argv_option(argv, "--model")
    if not provider or not model:
        try:
            from .provider_config import load_provider_config

            config = load_provider_config(workspace, missing_ok=True)
            provider = provider or str(config.get("active_provider") or "") or None
            entry = (
                (config.get("providers") or {}).get(provider, {}) if provider else {}
            )
            if isinstance(entry, dict):
                model = model or str(entry.get("model") or "") or None
        except (OSError, ValueError):
            pass
    if provider and model:
        try:
            from .provider_config import normalize_provider_model

            model = normalize_provider_model(provider, model)
        except ValueError:
            # Preserve the submitted descriptor when a legacy/custom provider
            # cannot be normalized; start itself remains the authority.
            pass
    return {
        "provider": provider,
        "model": model,
        "judgment_model": _argv_option(argv, "--judgment-model"),
        "profile": _argv_option(argv, "--autopilot") or "balanced",
        "task": _argv_option(argv, "--task") or "research",
    }


def _reconcile(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") not in {"queued", "running", "cancelling"}:
        return payload
    if _pid_is_alive(payload.get("pid"), hostname=payload.get("hostname")):
        stored_identity = payload.get("process_identity")
        if not stored_identity or payload.get("hostname") != socket.gethostname():
            return payload
        identity_match = _process_identity_matches(
            int(payload["pid"]), str(stored_identity)
        )
        if identity_match is not False:
            return payload
    reconciled = dict(payload)
    reconciled["status"] = (
        "cancelled" if reconciled.get("cancel_requested_at") else "interrupted"
    )
    reconciled["finished_at"] = reconciled.get("finished_at") or _now_iso()
    reconciled["returncode"] = reconciled.get("returncode")
    _write_state(path, reconciled)
    return reconciled


def launch_detached_run(
    workspace: str | Path,
    start_argv: Sequence[str],
    *,
    resume_of: str | None = None,
    startup_grace_seconds: float = DETACHED_STARTUP_GRACE_SECONDS,
) -> dict[str, Any]:
    """Launch the exact public ``start`` command in a detached process."""

    root = Path(workspace).expanduser().resolve()
    run_id = uuid.uuid4().hex[:16]
    state_dir = _state_dir(root, create=True)
    state_path = _state_path(root, run_id)
    stdout_path = state_dir / f"{run_id}.out.log"
    stderr_path = state_dir / f"{run_id}.err.log"
    child_argv = [str(item) for item in start_argv]
    if "--detach" in child_argv:
        child_argv.remove("--detach")
    if not child_argv or child_argv[0] != "start":
        raise RunControlError("detached runs must launch an xscientist start command")
    payload: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "id": run_id,
        "status": "queued",
        "workspace": root.name,
        "created_at": _now_iso(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "process_identity": None,
        "hostname": socket.gethostname(),
        "returncode": None,
        "cancel_requested_at": None,
        "resume_of": resume_of,
        "command": _public_command(child_argv),
        # This file is local and forced to mode 0600.  Keeping the exact argv is
        # what makes recovery deterministic; it is never returned in public
        # JSON or human output.
        "resume_argv": child_argv,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        **_run_descriptor(root, child_argv),
    }
    _write_state(state_path, payload)
    env = os.environ.copy()
    env[RUN_STATE_ENV] = str(state_path)
    popen_kwargs: dict[str, Any] = {
        "cwd": str(Path.cwd()),
        "env": env,
        "stdin": subprocess.DEVNULL,
        "start_new_session": os.name == "posix",
    }
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    with (
        _open_private_log(stdout_path) as stdout_handle,
        _open_private_log(stderr_path) as stderr_handle,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", "xscientist", *child_argv],
            stdout=stdout_handle,
            stderr=stderr_handle,
            **popen_kwargs,
        )
    payload["pid"] = process.pid
    payload["process_identity"] = _process_identity(process.pid)
    payload["status"] = "running"
    payload["started_at"] = _now_iso()
    _write_state(state_path, payload)

    # Observe a short startup window so missing dependencies, identity, or
    # runtime prerequisites do not produce a misleading "started" success.
    # Long-running studies still detach after this bounded grace period.
    deadline = time.monotonic() + max(0.0, float(startup_grace_seconds))
    while True:
        returncode = process.poll()
        if returncode is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.05, remaining))
            continue
        if not isinstance(returncode, int):
            # Test doubles and non-standard Popen wrappers may not implement a
            # concrete poll result. In that case retain the running state.
            break
        latest = _read_state(state_path)
        if latest.get("status") in {"queued", "running", "cancelling"}:
            if latest.get("cancel_requested_at"):
                latest["status"] = "cancelled"
            else:
                latest["status"] = "succeeded" if returncode == 0 else "failed"
            latest["returncode"] = returncode
            latest["finished_at"] = latest.get("finished_at") or _now_iso()
            _write_state(state_path, latest)
        return public_run_view(latest)
    return public_run_view(payload)


def begin_active_run() -> Path | None:
    raw = str(os.environ.get(RUN_STATE_ENV) or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    payload = _read_state(path)
    payload.update(
        {
            "status": "running",
            "pid": os.getpid(),
            "process_identity": _process_identity(os.getpid()),
            "hostname": socket.gethostname(),
            "started_at": payload.get("started_at") or _now_iso(),
        }
    )
    _write_state(path, payload)
    return path


def finish_active_run(path: Path | None, returncode: int) -> None:
    if path is None:
        return
    try:
        payload = _read_state(path)
    except RunControlError:
        return
    payload["returncode"] = int(returncode)
    payload["finished_at"] = _now_iso()
    payload["status"] = (
        "cancelled"
        if payload.get("cancel_requested_at")
        else ("succeeded" if returncode == 0 else "failed")
    )
    _write_state(path, payload)


def public_run_view(payload: dict[str, Any]) -> dict[str, Any]:
    private = {"resume_argv", "process_identity", "hostname"}
    view = {key: value for key, value in payload.items() if key not in private}
    view["workspace"] = "."
    command_source = payload.get("resume_argv") or payload.get("command") or []
    view["command"] = _public_command(
        command_source if isinstance(command_source, list) else []
    )
    for key in ("provider", "model", "judgment_model", "profile", "task"):
        value = payload.get(key)
        if value is not None:
            flag = "--judgment-model" if key == "judgment_model" else f"--{key}"
            view[key] = _safe_public_value(flag, str(value))
    started = str(payload.get("started_at") or "")
    finished = str(payload.get("finished_at") or "")
    if started:
        try:
            start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
            end_time = (
                datetime.fromisoformat(finished.replace("Z", "+00:00"))
                if finished
                else datetime.now(timezone.utc)
            )
            view["duration_seconds"] = max(
                0, round((end_time - start_time).total_seconds(), 1)
            )
        except ValueError:
            view["duration_seconds"] = None
    else:
        view["duration_seconds"] = None
    redacted = redact_sensitive_payload(view)
    return dict(redacted) if isinstance(redacted, dict) else {}


def list_runs(workspace: str | Path) -> list[dict[str, Any]]:
    _require_workspace(workspace)
    rows: list[dict[str, Any]] = []
    for path in sorted(_state_dir(workspace).glob("*.json"), reverse=True):
        try:
            payload = _reconcile(path, _read_state(path))
        except RunControlError:
            continue
        rows.append(public_run_view(payload))
    return sorted(rows, key=lambda item: str(item.get("id") or ""), reverse=True)


def get_run(workspace: str | Path, run_id: str) -> dict[str, Any]:
    _require_workspace(workspace)
    path = _state_path(workspace, run_id)
    return public_run_view(_reconcile(path, _read_state(path)))


def read_run_logs(
    workspace: str | Path,
    run_id: str,
    *,
    stream: str = "both",
    tail: int = 200,
) -> dict[str, Any]:
    state = get_run(workspace, run_id)
    if stream not in {"stdout", "stderr", "both"}:
        raise RunControlError("stream must be stdout, stderr, or both")
    if tail < 1 or tail > 10_000:
        raise RunControlError("tail must be between 1 and 10000 lines")
    root = _state_dir(workspace)
    normalized_run_id = str(run_id or "").strip()

    def load(name: str) -> list[str]:
        filename = state.get(name)
        if not filename:
            return []
        suffix = "out.log" if name == "stdout" else "err.log"
        expected = f"{normalized_run_id}.{suffix}"
        if str(filename) != expected:
            raise RunControlError(f"run {name} log path failed integrity validation")
        path = root / expected
        if path.is_symlink():
            raise RunControlError(f"run {name} log must not be a symlink")
        if path.exists() and not path.is_file():
            raise RunControlError(f"run {name} log is not a regular file")
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[
                -tail:
            ]
            return [redact_sensitive_text(line) for line in lines]
        except OSError:
            return []

    return {
        "run": state,
        "stdout": load("stdout") if stream in {"stdout", "both"} else [],
        "stderr": load("stderr") if stream in {"stderr", "both"} else [],
    }


def cancel_run(workspace: str | Path, run_id: str) -> dict[str, Any]:
    path = _state_path(workspace, run_id)
    payload = _reconcile(path, _read_state(path))
    if payload.get("status") in {"succeeded", "failed", "cancelled", "interrupted"}:
        raise RunControlError(f"run is already {payload.get('status')}")
    if payload.get("hostname") != socket.gethostname():
        raise RunControlError("cannot signal a run owned by another host")
    pid = int(payload.get("pid") or 0)
    if os.name == "posix":
        stored_identity = payload.get("process_identity")
        identity_match = (
            _process_identity_matches(pid, str(stored_identity))
            if stored_identity
            else None
        )
        if identity_match is not True:
            raise RunControlError(
                "run process identity cannot be verified; refusing to signal"
            )
        try:
            if os.getpgid(pid) != pid:
                raise RunControlError(
                    "run no longer owns its process group; refusing to signal"
                )
        except ProcessLookupError:
            payload["status"] = "interrupted"
            payload["finished_at"] = _now_iso()
            _write_state(path, payload)
            return public_run_view(payload)
    payload["status"] = "cancelling"
    payload["cancel_requested_at"] = _now_iso()
    _write_state(path, payload)
    try:
        if os.name == "posix":
            os.killpg(pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows CI
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        payload["status"] = "cancelled"
        payload["finished_at"] = _now_iso()
        _write_state(path, payload)
    except OSError as exc:
        raise RunControlError(f"could not signal run: {type(exc).__name__}") from exc
    return public_run_view(payload)


def resume_run(
    workspace: str | Path,
    run_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    path = _state_path(workspace, run_id)
    payload = _reconcile(path, _read_state(path))
    if payload.get("status") not in {"failed", "cancelled", "interrupted"}:
        raise RunControlError("only failed, cancelled, or interrupted runs can resume")
    argv = payload.get("resume_argv")
    if not isinstance(argv, list) or not argv:
        raise RunControlError("run has no resumable command")
    root = _require_workspace(workspace)
    if not force:
        try:
            from .diagnostics import diagnose

            descriptor = _run_descriptor(root, [str(item) for item in argv])
            report = diagnose(
                root,
                task=str(descriptor.get("task") or "research"),
                provider=descriptor.get("provider"),
                deep=True,
            )
        except (OSError, ValueError) as exc:
            raise RunControlError(
                "run prerequisites could not be rechecked; run `xscientist "
                f"doctor --workspace {root}` first, or pass --force to bypass "
                "this safety check"
            ) from exc
        if not report.get("ok"):
            actions = list(report.get("next_actions") or [])
            next_action = str(
                actions[0]
                if actions
                else f"xscientist doctor --workspace {root} --deep"
            )
            next_action = next_action.replace(
                "--workspace .", f"--workspace {shlex.quote(str(root))}"
            )
            raise RunControlError(
                "run prerequisites are still unresolved; next: "
                f"{next_action}. Pass --force only to bypass this safety check"
            )
    return launch_detached_run(
        workspace, [str(item) for item in argv], resume_of=run_id
    )


def watch_run(
    workspace: str | Path,
    run_id: str,
    *,
    interval: float = 2.0,
) -> dict[str, Any]:
    if interval < 0.1 or interval > 60:
        raise RunControlError("watch interval must be between 0.1 and 60 seconds")
    while True:
        state = get_run(workspace, run_id)
        if state.get("status") not in {"queued", "running", "cancelling"}:
            return state
        time.sleep(interval)


__all__ = [
    "RUN_SCHEMA",
    "RunControlError",
    "begin_active_run",
    "cancel_run",
    "finish_active_run",
    "get_run",
    "launch_detached_run",
    "list_runs",
    "read_run_logs",
    "resume_run",
    "watch_run",
]
