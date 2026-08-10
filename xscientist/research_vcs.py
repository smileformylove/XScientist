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
    delete_research_branch,
    init_repository,
    list_research_objects,
    list_research_branches,
    list_research_tags,
    merge_research_branch,
    load_research_object,
    record_research_object,
    rename_research_branch,
    repository_status,
    research_diff,
    research_blame,
    research_log,
    research_stage,
    research_unstage,
    preview_research_merge,
    show_checkpoint,
    restore_research_paths,
    resolve_research_object_id,
    revert_research_checkpoint,
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
        semantic_profile: Mapping[str, Any] | None = None,
    ) -> ResearchObjectResult:
        return record_research_object(
            self.path,
            kind=kind,
            payload=dict(payload),
            state=state,
            relations=[dict(item) for item in relations],
            actor=dict(actor) if actor is not None else None,
            provenance=dict(provenance) if provenance is not None else None,
            semantic_profile=(
                dict(semantic_profile) if semantic_profile is not None else None
            ),
        )

    def get(self, object_id: str) -> dict[str, Any]:
        return load_research_object(self.path, object_id)

    def resolve(self, selector: str, *, kind: str | None = None) -> str:
        return resolve_research_object_id(self.path, selector, expected_kind=kind)

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

    def delete_branch(self, name: str, *, force: bool = False) -> dict[str, Any]:
        return delete_research_branch(self.path, name, force=force)

    def rename_branch(self, name: str, new_name: str) -> dict[str, Any]:
        return rename_research_branch(self.path, name, new_name)

    def restore(self, source: str, *paths: str) -> dict[str, Any]:
        return restore_research_paths(self.path, source, paths)

    def revert(self, commit: str, *, subject: str | None = None) -> dict[str, Any]:
        return revert_research_checkpoint(self.path, commit, subject=subject)

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

    def log(self, *, limit: int = 20, ref: str = "HEAD") -> list[dict[str, Any]]:
        return research_log(self.path, limit=limit, ref=ref)

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

    def context(
        self,
        *,
        target_ids: Sequence[str],
        intent: str = "decide",
        decision_kind: str = "research_decision",
        selected: str = "",
        options_considered: Sequence[Mapping[str, Any] | str] = (),
        rationale: Sequence[str] = (),
        constraints: Sequence[str] = (),
        memory_refs: Sequence[str] = (),
        ref: str | None = None,
        budget_tokens: int = 4000,
    ) -> dict[str, Any]:
        """Compile a hash-bound view of the evidence and memory for a decision."""

        from .research_context import build_research_context_snapshot

        return build_research_context_snapshot(
            self,
            target_ids=target_ids,
            intent=intent,
            decision_kind=decision_kind,
            selected=selected,
            options_considered=options_considered,
            rationale=rationale,
            constraints=constraints,
            memory_refs=memory_refs,
            ref=ref,
            budget_tokens=budget_tokens,
        )

    def context_prompt(
        self,
        *,
        target_ids: Sequence[str],
        intent: str = "decide",
        decision_kind: str = "research_decision",
        selected: str = "",
        options_considered: Sequence[Mapping[str, Any] | str] = (),
        rationale: Sequence[str] = (),
        constraints: Sequence[str] = (),
        memory_refs: Sequence[str] = (),
        ref: str | None = None,
        budget_tokens: int = 4000,
    ) -> str:
        """Render the bounded, decision-usable view intended for an agent prompt.

        ``context()`` remains the complete auditable snapshot.  This method
        deliberately returns only the hash-bound working set, preventing an
        agent from accidentally ingesting the entire historical closure.
        """

        from .research_context import render_research_context_for_prompt

        snapshot = self.context(
            target_ids=target_ids,
            intent=intent,
            decision_kind=decision_kind,
            selected=selected,
            options_considered=options_considered,
            rationale=rationale,
            constraints=constraints,
            memory_refs=memory_refs,
            ref=ref,
            budget_tokens=budget_tokens,
        )
        return render_research_context_for_prompt(snapshot)

    def technology_tree(self) -> dict[str, Any]:
        from .research_policy import build_research_technology_tree

        return build_research_technology_tree(self.path)

    def guide(self, *, language: str = "auto") -> dict[str, Any]:
        """Return a beginner-friendly progress explanation and next actions."""

        from .research_journey import build_research_guide

        return build_research_guide(self.path, language=language)

    def dag(
        self,
        *,
        ref: str = "HEAD",
        ara_roots: Sequence[str | Path] = (),
        disclose_summaries: bool = True,
    ) -> dict[str, Any]:
        """Build a unified evidence, verification, and evolution DAG."""

        from .research_dag import build_research_dag

        return build_research_dag(
            self.path,
            ref=ref,
            ara_roots=ara_roots,
            disclose_summaries=disclose_summaries,
        )

    def export_dag(
        self,
        destination: str | Path,
        *,
        ref: str = "HEAD",
        ara_roots: Sequence[str | Path] = (),
        disclose_summaries: bool = True,
    ) -> dict[str, Any]:
        """Write deterministic DAG JSON and a self-contained offline browser."""

        from .research_dag import export_research_dag

        return export_research_dag(
            self.path,
            destination,
            ref=ref,
            ara_roots=ara_roots,
            disclose_summaries=disclose_summaries,
        )

    def sync(
        self,
        *,
        adapter: str,
        destination: str,
        ref: str = "HEAD",
        formats: Sequence[str] = (
            "ro-crate",
            "prov-json",
            "cwl",
            "dvc",
            "mlflow",
        ),
        include_payloads: bool = False,
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Publish a committed exchange package through an explicit adapter."""

        from .research_adapters import sync_research_repository

        return sync_research_repository(
            self.path,
            adapter_name=adapter,
            destination=destination,
            ref=ref,
            formats=formats,
            include_payloads=include_payloads,
            options=options,
        )

    def ingest_tool_evidence(
        self,
        receipt: Mapping[str, Any],
        *,
        attempt_ids: Sequence[str],
        supports: Sequence[str] = (),
        refutes: Sequence[str] = (),
        message: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Import external tool output without granting it verified status."""

        from .research_tools import ingest_tool_evidence

        return ingest_tool_evidence(
            self.path,
            receipt,
            attempt_ids=attempt_ids,
            supports=supports,
            refutes=refutes,
            message=message,
            commit=commit,
        )

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

    def export(
        self,
        destination: str | Path,
        *,
        ref: str = "HEAD",
        formats: Sequence[str] = (
            "ro-crate",
            "prov-json",
            "cwl",
            "dvc",
            "mlflow",
        ),
        include_payloads: bool = False,
    ) -> dict[str, Any]:
        """Export one committed state to standard provenance/tool formats."""

        from .research_interop import export_research_interop

        return export_research_interop(
            self.path,
            destination,
            ref=ref,
            formats=formats,
            include_payloads=include_payloads,
        )


__all__ = ["ResearchRepository"]
