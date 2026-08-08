"""Public, backend-independent Research Version Control API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .research_git import (
    CheckpointResult,
    ResearchObjectResult,
    ResearchMergeResult,
    ResearchStageResult,
    commit_research_stage,
    create_checkpoint,
    create_research_branch,
    create_research_tag,
    init_repository,
    list_research_objects,
    list_research_branches,
    list_research_tags,
    merge_research_branch,
    load_research_object,
    record_research_object,
    repository_status,
    research_diff,
    research_blame,
    research_log,
    research_stage,
    research_unstage,
    preview_research_merge,
    show_checkpoint,
    switch_research_branch,
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

    def stage(
        self,
        paths: Sequence[str] = (),
        *,
        all_changes: bool = False,
    ) -> ResearchStageResult:
        return research_stage(self.path, paths, all_changes=all_changes)

    def unstage(
        self,
        paths: Sequence[str] = (),
        *,
        all_paths: bool = False,
    ) -> ResearchStageResult:
        return research_unstage(self.path, paths, all_paths=all_paths)

    def commit(
        self,
        *,
        stage: str,
        subject: str,
        summary: str = "",
        status: str = "completed",
        actor: str | None = None,
        staged_only: bool = False,
    ) -> CheckpointResult:
        operation = commit_research_stage if staged_only else create_checkpoint
        return operation(
            self.path,
            stage=stage,
            subject=subject,
            summary=summary,
            status=status,
            actor=actor,
        )

    checkpoint = commit

    def branches(self) -> list[dict[str, Any]]:
        return list_research_branches(self.path)

    def fork(
        self,
        name: str,
        *,
        from_ref: str = "HEAD",
        switch: bool = True,
    ) -> dict[str, Any]:
        return create_research_branch(
            self.path,
            name,
            from_ref=from_ref,
            switch=switch,
        )

    def switch(self, name: str) -> dict[str, Any]:
        return switch_research_branch(self.path, name)

    def tag(
        self,
        name: str,
        *,
        commit: str = "HEAD",
        annotation: str = "",
    ) -> dict[str, Any]:
        return create_research_tag(
            self.path,
            name,
            commit=commit,
            annotation=annotation,
        )

    def tags(self) -> list[dict[str, Any]]:
        return list_research_tags(self.path)

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

    def blame(self, object_id: str, *, commit: str = "HEAD") -> dict[str, Any]:
        return research_blame(self.path, object_id, commit=commit)

    def merge_preview(self, source: str) -> dict[str, Any]:
        return preview_research_merge(self.path, source)

    def decide(
        self,
        *,
        event: str,
        name: str = "",
        state: str = "",
        source_branch: str | None = None,
        competing_hypothesis: bool = False,
        contradictory_evidence: bool = False,
        protocol_change: bool = False,
        independent_replication: bool = False,
    ) -> dict[str, Any]:
        from .research_policy import decide_research_transition

        return decide_research_transition(
            self.path,
            event=event,
            name=name,
            state=state,
            source_branch=source_branch,
            competing_hypothesis=competing_hypothesis,
            contradictory_evidence=contradictory_evidence,
            protocol_change=protocol_change,
            independent_replication=independent_replication,
        )

    def technology_tree(self) -> dict[str, Any]:
        from .research_policy import build_research_technology_tree

        return build_research_technology_tree(self.path)

    def merge(
        self,
        source: str,
        *,
        subject: str | None = None,
        summary: str = "",
        actor: str | None = None,
        preserve_conflicts: bool = False,
    ) -> ResearchMergeResult:
        return merge_research_branch(
            self.path,
            source,
            subject=subject,
            summary=summary,
            actor=actor,
            preserve_conflicts=preserve_conflicts,
        )

    def fsck(
        self, *, commit: str = "HEAD", verify_objects: bool = True
    ) -> dict[str, Any]:
        return verify_research_repository(
            self.path,
            commit=commit,
            verify_objects=verify_objects,
        )

    def audit(
        self,
        *,
        ref: str = "HEAD",
        level: str = "trace",
        verify_objects: bool = True,
    ) -> dict[str, Any]:
        """Audit claim-to-evidence-to-reproduction closure at one ref."""

        from .research_closure import audit_research_closure

        return audit_research_closure(
            self.path,
            ref=ref,
            level=level,
            verify_objects=verify_objects,
        )


__all__ = ["ResearchRepository"]
