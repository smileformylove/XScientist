from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
import json
import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json
from ai_scientist.config.paths import resolve_output_path

from ._version import __version__
from .client import XScientist
from .models import CommandResult, ProjectRequest, ServiceSettings

try:
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_fastapi():
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HTTP service dependencies are not installed. "
            "Install them with `pip install xscientist[service]`."
        ) from exc
    return FastAPI, HTTPException, Request, JSONResponse


if ConfigDict is not None and Field is not None:

    class ProjectPayload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        project: str = Field(min_length=1)
        topic: str | None = None
        ideas: str | None = None
        output_root: str | None = None
        num_ideas: int = Field(default=3, ge=1)
        parallel: bool = False
        num_workers: int = Field(default=2, ge=1)
        workflow_mode: str = "adaptive"
        target_venue: str | None = None
        submission_mode: bool = False
        breakthrough_mode: bool = False
        high_quality_mode: bool = False
        bfts_config: str | None = None
        extra_args: list[str] = Field(default_factory=list)

        def to_request(self) -> ProjectRequest:
            return ProjectRequest(**self.model_dump())

else:
    ProjectPayload = None  # type: ignore[assignment,misc]


@dataclass
class _Job:
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
    def from_dict(cls, payload: dict[str, Any]) -> "_Job":
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


class _JobStore:
    def __init__(
        self,
        client: XScientist,
        *,
        max_workers: int,
        max_output_chars: int,
        state_dir: str | Path,
    ) -> None:
        self.client = client
        self.max_output_chars = max_output_chars
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="xscientist-api"
        )
        self.jobs: dict[str, _Job] = {}
        self.futures: dict[str, Future[None]] = {}
        self.lock = threading.Lock()
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._restore()

    def _job_path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _persist_locked(self, job: _Job) -> None:
        atomic_write_json(self._job_path(job.id), job.to_dict())

    def _restore(self) -> None:
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                job = _Job.from_dict(payload)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            if job.status in {"queued", "running"}:
                job.status = "interrupted"
                job.finished_at = _now_iso()
                job.error = "Service restarted before the job completed"
                atomic_write_json(path, job.to_dict())
            self.jobs[job.id] = job

    def submit(self, request: ProjectRequest) -> _Job:
        job = _Job(id=uuid.uuid4().hex, request=request)
        with self.lock:
            self._persist_locked(job)
            self.jobs[job.id] = job
            self.futures[job.id] = self.executor.submit(self._run, job.id)
        return job

    def _run(self, job_id: str) -> None:
        job = self.jobs[job_id]
        try:
            with self.lock:
                job.status = "running"
                job.started_at = _now_iso()
                self._persist_locked(job)
            result = self.client.run_project(job.request)
            result = CommandResult(
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout[-self.max_output_chars :],
                stderr=result.stderr[-self.max_output_chars :],
                started_at=result.started_at,
                finished_at=result.finished_at,
            )
            with self.lock:
                job.result = result
                job.status = "succeeded" if result.ok else "failed"
        except BaseException as exc:
            with self.lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
        finally:
            with self.lock:
                job.finished_at = _now_iso()
                try:
                    self._persist_locked(job)
                except OSError as exc:
                    job.status = "failed"
                    job.error = f"StatePersistenceError: {exc}"

    def get(self, job_id: str) -> _Job | None:
        with self.lock:
            return self.jobs.get(job_id)

    def list(self) -> list[_Job]:
        with self.lock:
            return sorted(
                self.jobs.values(), key=lambda job: job.created_at, reverse=True
            )

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=False)


def create_app(settings: ServiceSettings | None = None):
    FastAPI, HTTPException, Request, JSONResponse = _require_fastapi()
    if ProjectPayload is None:
        raise ModuleNotFoundError(
            "Pydantic is not installed. Install `xscientist[service]`."
        )
    resolved = settings or ServiceSettings(api_key=os.environ.get("XSCIENTIST_API_KEY"))
    client = XScientist(
        work_dir=resolved.work_dir,
        output_root=resolved.output_root,
        env=resolved.env,
    )
    output_root = (
        Path(resolved.output_root).expanduser().resolve()
        if resolved.output_root is not None
        else resolve_output_path().resolve()
    )
    state_dir = resolved.state_dir or output_root / ".xscientist" / "api" / "jobs"
    store = _JobStore(
        client,
        max_workers=resolved.max_workers,
        max_output_chars=resolved.max_output_chars,
        state_dir=state_dir,
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        store.shutdown()

    app = FastAPI(
        title="XScientist API",
        version=__version__,
        description="Submit and inspect isolated autonomous research jobs.",
        lifespan=lifespan,
    )
    app.state.job_store = store

    if resolved.api_key:

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            supplied = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(supplied, resolved.api_key or ""):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "missing or invalid X-API-Key"},
                )
            return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"items": [job.to_dict() for job in store.list()]}

    @app.post("/v1/projects", status_code=202)
    def submit_project(payload: ProjectPayload) -> dict[str, Any]:
        try:
            request = payload.to_request()
            request.to_argv()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return store.submit(request).to_dict()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    return app


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    work_dir: str | None = None,
    output_root: str | None = None,
    max_workers: int = 2,
    max_output_chars: int = 200_000,
    api_key: str | None = None,
    state_dir: str | None = None,
    reload: bool = False,
) -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Uvicorn is not installed. Install `xscientist[service]`."
        ) from exc
    if reload and any(value is not None for value in (work_dir, output_root)):
        raise ValueError("reload mode does not support custom work_dir/output_root")
    if reload:
        uvicorn.run(
            "xscientist.service:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    app = create_app(
        ServiceSettings(
            work_dir=work_dir,
            output_root=output_root,
            max_workers=max_workers,
            max_output_chars=max_output_chars,
            api_key=api_key or os.environ.get("XSCIENTIST_API_KEY"),
            state_dir=state_dir,
        )
    )
    uvicorn.run(app, host=host, port=port)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the XScientist HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--max-output-chars", type=int, default=200_000)
    parser.add_argument("--state-dir", default=None)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        work_dir=args.work_dir,
        output_root=args.output_root,
        max_workers=args.max_workers,
        max_output_chars=args.max_output_chars,
        state_dir=args.state_dir,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
