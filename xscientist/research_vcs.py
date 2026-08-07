"""Public, backend-independent Research Version Control API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_git import (
    CheckpointResult,
    ResearchObjectResult,
    create_checkpoint,
    init_repository,
    list_research_objects,
    load_research_object,
    record_research_object,
    repository_status,
    research_diff,
    research_log,
    show_checkpoint,
    verify_research_repository,
)


class ResearchRepository:
    """One native Research VCS repository.

    The current persistence adapter uses Git for durable commits. Callers use
    scientific operations and do not depend on backend commands or identifiers.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        repository_status(self.path)

    @classmethod
    def init(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        question: str | None = None,
        policy: str = "milestone",
        actor: str = "xscientist",
        git_user_name: str | None = None,
        git_user_email: str | None = None,
    ) -> "ResearchRepository":
        init_repository(
            path,
            name=name,
            question=question,
            policy=policy,
            actor=actor,
            git_user_name=git_user_name,
            git_user_email=git_user_email,
        )
        return cls(path)

    def status(self) -> dict[str, Any]:
        return repository_status(self.path)

    def record(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        state: str = "draft",
        relations: Sequence[Mapping[str, Any]] = (),
        actor: Mapping[str, Any] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ResearchObjectResult:
        return record_research_object(
            self.path,
            kind=kind,
            payload=dict(payload),
            state=state,
            relations=[dict(item) for item in relations],
            actor=dict(actor) if actor is not None else None,
            provenance=dict(provenance) if provenance is not None else None,
        )

    def get(self, object_id: str) -> dict[str, Any]:
        return load_research_object(self.path, object_id)

    def objects(
        self,
        *,
        kind: str | None = None,
        state: str | None = None,
    ) -> list[dict[str, Any]]:
        return list_research_objects(self.path, kind=kind, state=state)

    def commit(
        self,
        *,
        stage: str,
        subject: str,
        summary: str = "",
        status: str = "completed",
        actor: str | None = None,
    ) -> CheckpointResult:
        return create_checkpoint(
            self.path,
            stage=stage,
            subject=subject,
            summary=summary,
            status=status,
            actor=actor,
        )

    checkpoint = commit

    def log(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return research_log(self.path, limit=limit)

    def show(self, commit: str = "HEAD") -> dict[str, Any]:
        return show_checkpoint(self.path, commit)

    def diff(
        self,
        before: str = "HEAD~1",
        after: str = "HEAD",
        *,
        deep: bool = False,
    ) -> dict[str, Any]:
        return research_diff(self.path, before, after, deep=deep)

    def fsck(self, *, commit: str = "HEAD", verify_objects: bool = True) -> dict[str, Any]:
        return verify_research_repository(
            self.path,
            commit=commit,
            verify_objects=verify_objects,
        )


__all__ = ["ResearchRepository"]
