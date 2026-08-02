"""Persistent background-job execution for the XScientist HTTP service."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ai_scientist.utils.atomic_io import atomic_write_json

from .client import XScientist
from .models import CommandResult, ProjectRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
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
        )


class JobStore:
    def __init__(
        self,
        client: XScientist,
        *,
        max_workers: int,
        max_output_chars: int,
        state_dir: str | Path,
        write_json: Callable[[str | Path, Any], None] = atomic_write_json,
    ) -> None:
        self.client = client
        self.max_output_chars = max_output_chars
        self.write_json = write_json
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="xscientist-api"
        )
        self.jobs: dict[str, Job] = {}
        self.futures: dict[str, Future[None]] = {}
        self.lock = threading.Lock()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _job_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

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

    def submit(self, request: ProjectRequest) -> Job:
        job = Job(id=uuid.uuid4().hex, request=request)
        self._persist(job)
        with self.lock:
            self.jobs[job.id] = job
            self.futures[job.id] = self.executor.submit(self._run, job.id)
        return job

    def _run(self, job_id: str) -> None:
        final_changes: dict[str, Any]
        try:
            job = self._persist_transition(
                job_id,
                status="running",
                started_at=_now_iso(),
            )
            result = self.client.run_project(job.request)
            result = CommandResult(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout[-self.max_output_chars :],
                stderr=result.stderr[-self.max_output_chars :],
                started_at=result.started_at,
                finished_at=result.finished_at,
            )
            final_changes = {
                "result": result,
                "status": "succeeded" if result.ok else "failed",
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

    def get(self, job_id: str) -> Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[Job]:
        with self.lock:
            return sorted(
                self.jobs.values(), key=lambda job: job.created_at, reverse=True
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)
