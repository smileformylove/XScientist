"""Persistent background-job execution for the XScientist HTTP service."""

from __future__ import annotations

import json
import ntpath
import os
import re
import sys
import threading
import unicodedata
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.atomic_io import atomic_write_json

from .client import XScientist
from .models import CommandResult, ProjectRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceBusyError(ValueError):
    """Raised when an active in-process job already owns a project workspace."""

    code = "workspace_busy"

    def __init__(self, *, workspace: Path, active_job_id: str) -> None:
        self.workspace = workspace
        self.active_job_id = active_job_id
        super().__init__(
            f"{self.code}: project workspace already has an active job "
            f"({active_job_id})"
        )


@dataclass
class Job:
    id: str
    request: ProjectRequest
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    result: CommandResult | None = None
    error: str | None = None
    resume_of: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "resume_of": self.resume_of,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "request": self.request.to_dict(),
            "result": self.result.to_dict() if self.result is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Job":
        result_payload = payload.get("result")
        return cls(
            id=str(payload["id"]),
            request=ProjectRequest.from_dict(payload["request"]),
            status=str(payload.get("status") or "failed"),
            created_at=str(payload.get("created_at") or _now_iso()),
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            result=(
                CommandResult.from_dict(result_payload)
                if isinstance(result_payload, dict)
                else None
            ),
            error=payload.get("error"),
            resume_of=payload.get("resume_of"),
            stdout_tail=str(payload.get("stdout_tail") or ""),
            stderr_tail=str(payload.get("stderr_tail") or ""),
            stdout_truncated=bool(payload.get("stdout_truncated")),
            stderr_truncated=bool(payload.get("stderr_truncated")),
        )


class JobStore:
    def __init__(
        self,
        client: XScientist,
        *,
        max_workers: int,
        max_output_chars: int,
        max_workspace_bytes: int = 10 * 1024 * 1024 * 1024,
        max_workspace_files: int = 100_000,
        state_dir: str | Path,
        write_json: Callable[[str | Path, Any], None] = atomic_write_json,
    ) -> None:
        self.client = client
        self.max_output_chars = max_output_chars
        self.max_workspace_bytes = max_workspace_bytes
        self.max_workspace_files = max_workspace_files
        self.write_json = write_json
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="xscientist-api"
        )
        self.jobs: dict[str, Job] = {}
        self.futures: dict[str, Future[None]] = {}
        self.cancel_events: dict[str, threading.Event] = {}
        # Deliberately process-local: this prevents concurrent workers in one
        # service instance without pretending to be a cross-process lease.
        self.active_workspaces: dict[str, str] = {}
        self.job_workspaces: dict[str, str] = {}
        self.lock = threading.Lock()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _job_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _workspace_for(self, request: ProjectRequest) -> Path:
        client_output_root = getattr(self.client, "output_root", None)
        if not isinstance(client_output_root, (str, Path)):
            client_output_root = None
        output_root = request.output_root or client_output_root
        if output_root is None:
            client_env = getattr(self.client, "env", None)
            client_env = client_env if isinstance(client_env, dict) else {}
            output_root = client_env.get("RESEARCH_OUTPUT_DIR") or os.environ.get(
                "RESEARCH_OUTPUT_DIR"
            )
            if output_root is None:
                output_root = client_env.get(
                    "AI_SCIENTIST_OUTPUT_DIR"
                ) or os.environ.get("AI_SCIENTIST_OUTPUT_DIR")
        root = Path(output_root or resolve_output_path()).expanduser()
        if not root.is_absolute():
            client_work_dir = getattr(self.client, "work_dir", None)
            if not isinstance(client_work_dir, (str, Path)):
                client_work_dir = None
            root = (client_work_dir or Path.cwd()) / root
        return (root / "projects" / request.project).resolve()

    @staticmethod
    def _workspace_reservation_key(workspace: Path) -> str:
        """Canonicalize aliases that address one workspace on supported hosts."""

        normalized = os.path.normcase(str(workspace))
        if sys.platform == "darwin":
            # Default APFS/HFS+ volumes are case-insensitive and normalize
            # Unicode spellings even though pathlib's POSIX equality does not.
            normalized = unicodedata.normalize("NFC", normalized).casefold()
        elif sys.platform == "win32":
            # Win32 strips trailing dots/spaces from ordinary path components.
            # Normalize those aliases defensively even when JobStore is called
            # directly without the HTTP name validator.
            windows_path = ntpath.normcase(str(workspace))
            drive, tail = ntpath.splitdrive(windows_path)
            components = [
                component.rstrip(" .")
                for component in re.split(r"[\\/]+", tail)
                if component
            ]
            normalized = unicodedata.normalize(
                "NFC", drive + "\\" + "\\".join(components)
            ).casefold()
        return normalized

    def _reserve_workspace(self, job_id: str, workspace: Path) -> None:
        reservation_key = self._workspace_reservation_key(workspace)
        with self.lock:
            active_job_id = self.active_workspaces.get(reservation_key)
            if active_job_id is not None:
                raise WorkspaceBusyError(
                    workspace=workspace,
                    active_job_id=active_job_id,
                )
            self.active_workspaces[reservation_key] = job_id
            self.job_workspaces[job_id] = reservation_key

    def _release_workspace(self, job_id: str) -> None:
        with self.lock:
            reservation_key = self.job_workspaces.pop(job_id, None)
            if (
                reservation_key is not None
                and self.active_workspaces.get(reservation_key) == job_id
            ):
                del self.active_workspaces[reservation_key]

    def _persist(self, job: Job) -> None:
        self.write_json(self._job_path(job.id), job.to_dict())

    def _persist_transition(self, job_id: str, **changes: Any) -> Job:
        with self.lock:
            candidate = replace(self.jobs[job_id], **changes)
        self._persist(candidate)
        with self.lock:
            job = self.jobs[job_id]
            for name, value in changes.items():
                setattr(job, name, value)
            return job

    def _restore(self) -> None:
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                job = Job.from_dict(payload)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            if job.status in {"queued", "running"}:
                job.status = "interrupted"
                job.finished_at = _now_iso()
                job.error = "Service restarted before the job completed"
                self.write_json(path, job.to_dict())
            self.jobs[job.id] = job

    def submit(self, request: ProjectRequest, *, resume_of: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, request=request, resume_of=resume_of)
        workspace = self._workspace_for(request)
        self._reserve_workspace(job.id, workspace)
        try:
            self._persist(job)
            with self.lock:
                self.jobs[job.id] = job
                self.cancel_events[job.id] = threading.Event()
                self.futures[job.id] = self.executor.submit(self._run, job.id)
        except BaseException:
            self._release_workspace(job.id)
            raise
        return job

    def _run(self, job_id: str) -> None:
        final_changes: dict[str, Any]
        try:
            job = self._persist_transition(
                job_id,
                status="running",
                started_at=_now_iso(),
            )
            cancel_event = self.cancel_events[job_id]

            def update_output(
                stdout: str,
                stderr: str,
                stdout_truncated: bool,
                stderr_truncated: bool,
            ) -> None:
                self._persist_transition(
                    job_id,
                    stdout_tail=stdout[-self.max_output_chars :],
                    stderr_tail=stderr[-self.max_output_chars :],
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                )

            result = self.client.run_project(
                job.request,
                max_output_chars=self.max_output_chars,
                max_workspace_bytes=self.max_workspace_bytes,
                max_workspace_files=self.max_workspace_files,
                cancel_check=cancel_event.is_set,
                output_callback=update_output,
            )
            result = CommandResult(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout[-self.max_output_chars :],
                stderr=result.stderr[-self.max_output_chars :],
                started_at=result.started_at,
                finished_at=result.finished_at,
                stdout_truncated=(
                    result.stdout_truncated
                    or len(result.stdout) > self.max_output_chars
                ),
                stderr_truncated=(
                    result.stderr_truncated
                    or len(result.stderr) > self.max_output_chars
                ),
            )
            final_changes = {
                "result": result,
                "status": (
                    "cancelled"
                    if cancel_event.is_set() or result.returncode == 130
                    else ("succeeded" if result.ok else "failed")
                ),
                "stdout_tail": result.stdout,
                "stderr_tail": result.stderr,
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            }
        except BaseException as exc:
            final_changes = {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "failed",
            }
        finally:
            final_changes["finished_at"] = _now_iso()
            try:
                self._persist_transition(job_id, **final_changes)
            except OSError as exc:
                with self.lock:
                    job = self.jobs[job_id]
                    for name, value in final_changes.items():
                        setattr(job, name, value)
                    job.status = "failed"
                    job.error = f"StatePersistenceError: {exc}"
            finally:
                self._release_workspace(job_id)

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(
                self.jobs.values(), key=lambda job: job.created_at, reverse=True
            )

    def cancel(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
            future = self.futures.get(job_id)
            event = self.cancel_events.get(job_id)
        if job is None:
            return None
        if job.status in {"succeeded", "failed", "cancelled", "interrupted"}:
            return job
        if event is not None:
            event.set()
        if future is not None and future.cancel():
            try:
                return self._persist_transition(
                    job_id,
                    status="cancelled",
                    finished_at=_now_iso(),
                    error="Cancelled before execution started",
                )
            finally:
                self._release_workspace(job_id)
        return self._persist_transition(job_id, status="cancelling")

    def resume(self, job_id: str) -> Job | None:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.status not in {"failed", "cancelled", "interrupted"}:
            raise ValueError(
                "only failed, cancelled, or interrupted jobs can be resumed"
            )
        request = replace(job.request, resume=True)
        return self.submit(request, resume_of=job_id)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
