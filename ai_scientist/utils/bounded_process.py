"""Bounded subprocess capture and workspace resource guards."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class BoundedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class ProcessResourceLimitExceeded(RuntimeError):
    """Raised after terminating a process that exceeded a workspace quota."""

    def __init__(self, reason: str, *, stdout: str, stderr: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stdout = stdout
        self.stderr = stderr


class ProcessCancelled(RuntimeError):
    """Raised after a caller-requested subprocess cancellation."""

    def __init__(
        self,
        *,
        stdout: str,
        stderr: str,
        stdout_truncated: bool,
        stderr_truncated: bool,
    ) -> None:
        super().__init__("process cancelled")
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated


class BoundedTextBuffer:
    """Thread-safe tail buffer whose retained character count is bounded."""

    def __init__(self, max_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self.max_chars = int(max_chars)
        self._chunks: deque[str] = deque()
        self._chars = 0
        self._truncated = False
        self._lock = threading.Lock()

    def append(self, value: str) -> None:
        text = str(value or "")
        if not text:
            return
        with self._lock:
            self._chunks.append(text)
            self._chars += len(text)
            while self._chars > self.max_chars and self._chunks:
                overflow = self._chars - self.max_chars
                first = self._chunks[0]
                self._truncated = True
                if len(first) <= overflow:
                    self._chunks.popleft()
                    self._chars -= len(first)
                else:
                    self._chunks[0] = first[overflow:]
                    self._chars -= overflow

    def snapshot(self) -> tuple[str, bool]:
        with self._lock:
            return "".join(self._chunks), self._truncated


def workspace_limit_checker(
    root: str | Path,
    *,
    max_bytes: int | None,
    max_files: int | None,
) -> Callable[[], str | None]:
    """Build a symlink-safe checker returning a human-readable violation."""

    workspace = Path(root).expanduser().resolve()
    byte_limit = int(max_bytes) if max_bytes is not None else None
    file_limit = int(max_files) if max_files is not None else None
    if byte_limit is not None and byte_limit < 1:
        raise ValueError("max_bytes must be at least 1")
    if file_limit is not None and file_limit < 1:
        raise ValueError("max_files must be at least 1")

    def check() -> str | None:
        if not workspace.exists():
            return None
        total_bytes = 0
        total_files = 0
        for directory, dirnames, filenames in os.walk(workspace, followlinks=False):
            base = Path(directory)
            dirnames[:] = [name for name in dirnames if not (base / name).is_symlink()]
            for filename in filenames:
                path = base / filename
                total_files += 1
                if file_limit is not None and total_files > file_limit:
                    return (
                        f"workspace file count exceeded {file_limit} files "
                        f"under {workspace.name}"
                    )
                try:
                    total_bytes += path.stat(follow_symlinks=False).st_size
                except (FileNotFoundError, OSError):
                    continue
                if byte_limit is not None and total_bytes > byte_limit:
                    return (
                        f"workspace size exceeded {byte_limit} bytes "
                        f"under {workspace.name}"
                    )
        return None

    return check


def _drain_stream(stream: TextIO, target: BoundedTextBuffer) -> None:
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            target.append(chunk)
    except (OSError, ValueError):
        return
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised by the Windows CI smoke lane
            process.kill()
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            pass


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised by the Windows CI smoke lane
            process.terminate()
    except (OSError, ProcessLookupError):
        return


def run_process_bounded(
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    max_output_chars: int = 200_000,
    limit_check: Callable[[], str | None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    output_callback: Callable[[str, str, bool, bool], None] | None = None,
    poll_interval: float = 0.1,
) -> BoundedProcessResult:
    """Run a subprocess while continuously draining output into bounded tails."""

    argv = [str(item) for item in command]
    stdout_buffer = BoundedTextBuffer(max_output_chars)
    stderr_buffer = BoundedTextBuffer(max_output_chars)
    process = subprocess.Popen(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    threads = (
        threading.Thread(
            target=_drain_stream,
            args=(process.stdout, stdout_buffer),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(process.stderr, stderr_buffer),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()

    started = time.monotonic()
    timed_out = False
    cancelled = False
    limit_reason: str | None = None
    next_limit_check = started
    next_output_callback = started
    try:
        while process.poll() is None:
            now = time.monotonic()
            if timeout is not None and now - started >= timeout:
                timed_out = True
                _kill_process_tree(process)
                break
            if cancel_check is not None and cancel_check():
                cancelled = True
                _terminate_process_tree(process)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    _kill_process_tree(process)
                break
            if limit_check is not None and now >= next_limit_check:
                try:
                    limit_reason = limit_check()
                except Exception as exc:  # fail closed at the resource boundary
                    limit_reason = f"workspace limit check failed: {type(exc).__name__}"
                if limit_reason:
                    _kill_process_tree(process)
                    break
                next_limit_check = now + max(0.1, poll_interval)
            if output_callback is not None and now >= next_output_callback:
                stdout_now, stdout_truncated_now = stdout_buffer.snapshot()
                stderr_now, stderr_truncated_now = stderr_buffer.snapshot()
                try:
                    output_callback(
                        stdout_now,
                        stderr_now,
                        stdout_truncated_now,
                        stderr_truncated_now,
                    )
                except Exception:
                    # Output persistence is observational and must not orphan
                    # or terminate the controlled subprocess.
                    pass
                next_output_callback = now + 0.5
            time.sleep(min(max(0.01, poll_interval), 0.25))
        process.wait()
        if limit_reason is None and limit_check is not None:
            try:
                limit_reason = limit_check()
            except Exception as exc:  # fail closed at the resource boundary
                limit_reason = f"workspace limit check failed: {type(exc).__name__}"
    finally:
        for thread in threads:
            thread.join(timeout=5)

    stdout, stdout_truncated = stdout_buffer.snapshot()
    stderr, stderr_truncated = stderr_buffer.snapshot()
    if output_callback is not None:
        try:
            output_callback(stdout, stderr, stdout_truncated, stderr_truncated)
        except Exception:
            pass
    if timed_out:
        raise subprocess.TimeoutExpired(
            argv,
            timeout,
            output=stdout,
            stderr=stderr,
        )
    if limit_reason:
        raise ProcessResourceLimitExceeded(
            limit_reason,
            stdout=stdout,
            stderr=stderr,
        )
    if cancelled:
        raise ProcessCancelled(
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
    return BoundedProcessResult(
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


__all__ = [
    "BoundedProcessResult",
    "BoundedTextBuffer",
    "ProcessResourceLimitExceeded",
    "run_process_bounded",
    "workspace_limit_checker",
]
