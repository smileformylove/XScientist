from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hmac
import os
import re
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json
from ai_scientist.config.paths import resolve_output_path

from ._version import __version__
from .client import XScientist
from .models import ProjectRequest, ServiceSettings
from .service_jobs import Job as _Job
from .service_jobs import JobStore as _JobStore

try:
    from pydantic import BaseModel, ConfigDict, Field
except ModuleNotFoundError:
    BaseModel = object  # type: ignore[assignment,misc]
    ConfigDict = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


_RESERVED_PROJECT_ARGS = {
    "--output-root",
    "--topic",
    "--ideas",
    "--bfts-config",
    "--seed-from-ara",
}
_PACKAGED_CONFIG_ALIASES = {
    "default",
    "deep",
    "bfts_config.yaml",
    "bfts_config_deep.yaml",
}


def _resolve_service_input(
    value: str | None,
    *,
    work_dir: Path,
    label: str,
) -> str | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = work_dir / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(work_dir)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within the service work_dir") from exc
    return str(resolved)


def _validate_project_name(project: str, *, output_root: Path) -> str:
    raw_name = str(project or "")
    name = raw_name.strip()
    if (
        not name
        or name != raw_name
        or name in {".", ".."}
        or name.startswith(("-", "~"))
        or Path(name).is_absolute()
        or "/" in name
        or "\\" in name
        or any(ord(char) < 32 for char in raw_name)
    ):
        raise ValueError("project must be a single directory name")
    projects_root = (output_root / "projects").resolve()
    candidate = (projects_root / name).resolve()
    try:
        candidate.relative_to(projects_root)
    except ValueError as exc:
        raise ValueError("project must stay within the service output_root") from exc
    return name


def _validate_research_ref(ref: str) -> str:
    normalized = str(ref or "").strip()
    if (
        not normalized
        or normalized.startswith("-")
        or ".." in normalized
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", normalized)
    ):
        raise ValueError("ref must be a safe branch, tag, HEAD, or commit name")
    return normalized


def _validate_extra_args(extra_args: list[str]) -> tuple[str, ...]:
    normalized = tuple(str(arg) for arg in extra_args)
    for arg in normalized:
        if any(
            arg == reserved or arg.startswith(f"{reserved}=")
            for reserved in _RESERVED_PROJECT_ARGS
        ):
            raise ValueError(
                f"extra_args cannot override service-controlled option {arg!r}"
            )
    return normalized


def _service_output_view(value: Any, *, output_root: Path) -> Any:
    """Make manager results JSON-safe without exposing host absolute paths."""

    if isinstance(value, dict):
        return {
            str(key): _service_output_view(item, output_root=output_root)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_service_output_view(item, output_root=output_root) for item in value]
    if isinstance(value, Path) or (
        isinstance(value, str) and Path(value).is_absolute()
    ):
        resolved = Path(value).expanduser().resolve()
        try:
            return resolved.relative_to(output_root).as_posix()
        except ValueError:
            return None
    return value


def _service_request(
    payload: "ProjectPayload",
    *,
    work_dir: Path,
    output_root: Path,
) -> ProjectRequest:
    requested_output = payload.output_root
    if requested_output is not None:
        if Path(requested_output).expanduser().resolve() != output_root:
            raise ValueError("output_root is controlled by the API service")

    config = payload.bfts_config
    if (
        config is not None
        and str(config).strip().lower() not in _PACKAGED_CONFIG_ALIASES
    ):
        config = _resolve_service_input(
            config,
            work_dir=work_dir,
            label="bfts_config",
        )

    return ProjectRequest(
        project=_validate_project_name(payload.project, output_root=output_root),
        topic=_resolve_service_input(payload.topic, work_dir=work_dir, label="topic"),
        ideas=_resolve_service_input(payload.ideas, work_dir=work_dir, label="ideas"),
        output_root=output_root,
        num_ideas=payload.num_ideas,
        parallel=payload.parallel,
        num_workers=payload.num_workers,
        workflow_mode=payload.workflow_mode,
        target_venue=payload.target_venue,
        submission_mode=payload.submission_mode,
        breakthrough_mode=payload.breakthrough_mode,
        high_quality_mode=payload.high_quality_mode,
        bfts_config=config,
        extra_args=_validate_extra_args(payload.extra_args),
    )


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


def create_app(settings: ServiceSettings | None = None):
    FastAPI, HTTPException, Request, JSONResponse = _require_fastapi()
    if ProjectPayload is None:
        raise ModuleNotFoundError(
            "Pydantic is not installed. Install `xscientist[service]`."
        )
    resolved = settings or ServiceSettings(api_key=os.environ.get("XSCIENTIST_API_KEY"))
    output_root = (
        Path(resolved.output_root).expanduser().resolve()
        if resolved.output_root is not None
        else resolve_output_path().resolve()
    )
    work_dir = (
        Path(resolved.work_dir).expanduser().resolve()
        if resolved.work_dir is not None
        else Path.cwd().resolve()
    )
    client = XScientist(
        work_dir=work_dir,
        output_root=output_root,
        env=resolved.env,
    )
    state_dir = resolved.state_dir or output_root / ".xscientist" / "api" / "jobs"
    store = _JobStore(
        client,
        max_workers=resolved.max_workers,
        max_output_chars=resolved.max_output_chars,
        state_dir=state_dir,
        write_json=lambda path, payload: atomic_write_json(path, payload),
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        store.shutdown()

    app = FastAPI(
        title="XScientist API",
        version=__version__,
        description=(
            "Submit and inspect isolated autonomous research jobs, papers, "
            "shortlists, and research boards."
        ),
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

    @app.get("/v1/papers")
    def list_papers(
        paper_type: str | None = None,
        sort_by: str = "modified",
        limit: int = 100,
    ) -> dict[str, Any]:
        try:
            items = client.list_papers(
                paper_type=paper_type,
                sort_by=sort_by,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "items": _service_output_view(items, output_root=output_root),
            "count": len(items),
        }

    @app.get("/v1/papers/{folder}")
    def get_paper(folder: str) -> dict[str, Any]:
        try:
            paper = client.get_paper(folder)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if paper is None:
            raise HTTPException(status_code=404, detail="paper not found")
        return {"paper": _service_output_view(paper, output_root=output_root)}

    @app.get("/v1/shortlist")
    def shortlist_papers(
        paper_type: str | None = None,
        target_venue: str | None = None,
        require_gate: bool = False,
        require_ready: bool = False,
        min_breakthrough: float | None = None,
        min_priority: float | None = None,
        max_blockers: int | None = None,
        min_rewrite_gain: float | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        try:
            items = client.shortlist_papers(
                paper_type=paper_type,
                target_venue=target_venue,
                require_gate=require_gate,
                require_ready=require_ready,
                min_breakthrough=min_breakthrough,
                min_priority=min_priority,
                max_blockers=max_blockers,
                min_rewrite_gain=min_rewrite_gain,
                top_n=top_n,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "items": _service_output_view(items, output_root=output_root),
            "count": len(items),
        }

    @app.get("/v1/boards/submission")
    def submission_board(
        top_n_per_venue: int = 3,
        require_gate: bool = False,
        min_breakthrough: float | None = None,
        min_priority: float | None = None,
        max_blockers: int | None = None,
        min_rewrite_gain: float | None = None,
    ) -> dict[str, Any]:
        try:
            venues = client.submission_board(
                top_n_per_venue=top_n_per_venue,
                require_gate=require_gate,
                min_breakthrough=min_breakthrough,
                min_priority=min_priority,
                max_blockers=max_blockers,
                min_rewrite_gain=min_rewrite_gain,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "venues": _service_output_view(venues, output_root=output_root),
            "count": sum(len(items) for items in venues.values()),
        }

    @app.get("/v1/boards/rewrite")
    def rewrite_board(
        top_n: int = 10,
        paper_type: str | None = None,
        target_venue: str | None = None,
        min_priority: float | None = None,
        min_rewrite_gain: float | None = None,
        max_blockers: int | None = None,
        require_gate: bool = False,
        include_ready: bool = False,
    ) -> dict[str, Any]:
        try:
            items = client.rewrite_board(
                top_n=top_n,
                paper_type=paper_type,
                target_venue=target_venue,
                min_priority=min_priority,
                min_rewrite_gain=min_rewrite_gain,
                max_blockers=max_blockers,
                require_gate=require_gate,
                include_ready=include_ready,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "items": _service_output_view(items, output_root=output_root),
            "count": len(items),
        }

    @app.post("/v1/projects", status_code=202)
    def submit_project(payload: ProjectPayload) -> dict[str, Any]:
        try:
            request = _service_request(
                payload,
                work_dir=work_dir,
                output_root=output_root,
            )
            request.to_argv()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return store.submit(request).to_dict()

    @app.get("/v1/projects/{project}/research/status")
    def research_status(project: str) -> dict[str, Any]:
        from .research_git import ResearchGitError

        try:
            name = _validate_project_name(project, output_root=output_root)
            repository = output_root / "projects" / name
            if not repository.is_dir():
                raise HTTPException(status_code=404, detail="project not found")
            from .research_vcs import ResearchRepository

            payload = ResearchRepository(repository).status()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except ResearchGitError as exc:
            raise HTTPException(
                status_code=409, detail="project has no valid Research VCS state"
            ) from exc
        return _service_output_view(payload, output_root=output_root)

    @app.get("/v1/projects/{project}/research/audit")
    def research_audit(
        project: str,
        ref: str = "HEAD",
        level: str = "trace",
        verify_objects: bool = True,
    ) -> dict[str, Any]:
        from .research_git import ResearchGitError

        try:
            name = _validate_project_name(project, output_root=output_root)
            selected_ref = _validate_research_ref(ref)
            if level not in {"trace", "replay", "verify"}:
                raise ValueError("level must be trace, replay, or verify")
            repository = output_root / "projects" / name
            if not repository.is_dir():
                raise HTTPException(status_code=404, detail="project not found")
            from .research_vcs import ResearchRepository

            return ResearchRepository(repository).audit(
                ref=selected_ref,
                level=level,
                verify_objects=verify_objects,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except ResearchGitError as exc:
            raise HTTPException(
                status_code=409, detail="research closure audit failed"
            ) from exc

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
