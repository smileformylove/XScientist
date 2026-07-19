from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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
        from fastapi import FastAPI, HTTPException
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "HTTP service dependencies are not installed. "
            "Install them with `pip install xscientist[service]`."
        ) from exc
    return FastAPI, HTTPException


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
            "request": {
                "project": self.request.project,
                "topic": str(self.request.topic) if self.request.topic else None,
                "ideas": str(self.request.ideas) if self.request.ideas else None,
                "output_root": (
                    str(self.request.output_root)
                    if self.request.output_root is not None
                    else None
                ),
                "workflow_mode": self.request.workflow_mode,
            },
            "result": self.result.to_dict() if self.result is not None else None,
        }


class _JobStore:
    def __init__(
        self, client: XScientist, *, max_workers: int, max_output_chars: int
    ) -> None:
        self.client = client
        self.max_output_chars = max_output_chars
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="xscientist-api"
        )
        self.jobs: dict[str, _Job] = {}
        self.futures: dict[str, Future[None]] = {}
        self.lock = threading.Lock()

    def submit(self, request: ProjectRequest) -> _Job:
        job = _Job(id=uuid.uuid4().hex, request=request)
        with self.lock:
            self.jobs[job.id] = job
            self.futures[job.id] = self.executor.submit(self._run, job.id)
        return job

    def _run(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.status = "running"
            job.started_at = _now_iso()
        try:
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
    FastAPI, HTTPException = _require_fastapi()
    if ProjectPayload is None:
        raise ModuleNotFoundError(
            "Pydantic is not installed. Install `xscientist[service]`."
        )
    resolved = settings or ServiceSettings()
    client = XScientist(
        work_dir=resolved.work_dir,
        output_root=resolved.output_root,
        env=resolved.env,
    )
    store = _JobStore(
        client,
        max_workers=resolved.max_workers,
        max_output_chars=resolved.max_output_chars,
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
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        work_dir=args.work_dir,
        output_root=args.output_root,
        max_workers=args.max_workers,
        max_output_chars=args.max_output_chars,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
