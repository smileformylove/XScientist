"""Scientific lifecycle operations built on native Research VCS objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_scientist.utils.research_integrity import validate_preregistration

from .research_git import CheckpointResult, ResearchGitError, ResearchObjectResult
from .research_vcs import ResearchRepository


class ResearchLifecycle:
    """Record an evidence-gated research lifecycle without backend commands."""

    def __init__(self, repository: ResearchRepository | str | Path) -> None:
        self.repository = (
            repository
            if isinstance(repository, ResearchRepository)
            else ResearchRepository(repository)
        )

    def planning(
        self,
        *,
        hypothesis: Mapping[str, Any],
        plan: Mapping[str, Any],
        preregistration: Mapping[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record hypothesis -> plan -> preregistration provenance."""

        hypothesis_result = self.repository.record(
            "hypothesis",
            hypothesis,
            state="draft",
        )
        plan_result = self.repository.record(
            "research_plan",
            plan,
            state="draft",
            relations=[{"type": "depends_on", "target": hypothesis_result.object_id}],
        )
        preregistration_result: ResearchObjectResult | None = None
        locked = False
        validation: dict[str, Any] | None = None
        if preregistration is not None:
            registration_payload = dict(preregistration)
            locked = registration_payload.get("status") == "locked"
            validation = validate_preregistration(
                registration_payload,
                require_locked=locked,
            )
            if locked and not validation["ok"]:
                raise ResearchGitError(
                    "locked preregistration failed integrity validation: "
                    + ", ".join(validation["errors"])
                )
            preregistration_result = self.repository.record(
                "preregistration",
                registration_payload,
                state="locked" if locked else "draft",
                relations=[{"type": "depends_on", "target": plan_result.object_id}],
                actor={
                    "actor_id": str(
                        registration_payload.get("registered_by") or "research-planner"
                    ),
                    "authority": "research_agent",
                },
            )
        checkpoint: CheckpointResult | None = None
        if commit:
            checkpoint = self.repository.commit(
                stage="preregister" if locked else "ideation",
                subject=(
                    "lock confirmatory research plan"
                    if locked
                    else "record exploratory research plan"
                ),
                status="completed" if locked else "draft",
            )
        return {
            "hypothesis": hypothesis_result,
            "plan": plan_result,
            "preregistration": preregistration_result,
            "preregistration_validation": validation,
            "checkpoint": checkpoint,
        }

    def experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record every attempt, including failure, timeout, and cancellation."""

        payload = dict(attempt)
        raw_status = str(payload.get("status") or "completed").lower()
        state = {
            "success": "completed",
            "completed": "completed",
            "failed": "failed",
            "error": "failed",
            "timeout": "timed_out",
            "timed_out": "timed_out",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "running": "running",
        }.get(raw_status)
        if state is None:
            raise ResearchGitError(
                f"unsupported experiment attempt status: {raw_status}"
            )
        if payload.get("study_phase") == "confirmatory" and not preregistration_id:
            raise ResearchGitError(
                "confirmatory experiment requires a locked preregistration"
            )
        relations = []
        if preregistration_id:
            registration = self.repository.get(preregistration_id)
            if registration["kind"] != "preregistration":
                raise ResearchGitError(
                    "experiment preregistration reference has wrong kind"
                )
            if (
                payload.get("study_phase") == "confirmatory"
                and registration["state"] != "locked"
            ):
                raise ResearchGitError(
                    "confirmatory experiment requires a locked preregistration"
                )
            relations.append(
                {"type": "depends_on", "target": preregistration_id, "role": "protocol"}
            )
        if plan_id:
            plan = self.repository.get(plan_id)
            if plan["kind"] != "research_plan":
                raise ResearchGitError("experiment plan reference has wrong kind")
            relations.append({"type": "depends_on", "target": plan_id, "role": "plan"})
        result = self.repository.record(
            "experiment_attempt",
            payload,
            state=state,
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="experiment" if state == "completed" else "failed",
                subject=f"record {state} experiment attempt",
                status=state,
            )
            if commit
            else None
        )
        return {"attempt": result, "checkpoint": checkpoint}

    def evidence(
        self,
        payload: Mapping[str, Any],
        *,
        attempt_ids: Sequence[str],
        supports: Sequence[str] = (),
        refutes: Sequence[str] = (),
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not attempt_ids:
            raise ResearchGitError("evidence requires at least one experiment attempt")
        relations: list[dict[str, str]] = []
        for object_id in attempt_ids:
            attempt = self.repository.get(object_id)
            if attempt["kind"] != "experiment_attempt":
                raise ResearchGitError("evidence attempt reference has wrong kind")
            relations.append({"type": "derived_from", "target": object_id})
        for object_id in supports:
            self.repository.get(object_id)
            relations.append({"type": "supports", "target": object_id})
        for object_id in refutes:
            self.repository.get(object_id)
            relations.append({"type": "refutes", "target": object_id})
        result = self.repository.record(
            "evidence",
            payload,
            state="verified" if verified else "completed",
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="evidence",
                subject="bind experiment evidence",
                status=result.state,
            )
            if commit
            else None
        )
        return {"evidence": result, "checkpoint": checkpoint}

    def evaluation(
        self,
        report: Mapping[str, Any],
        *,
        evaluates: Sequence[str],
        verifier_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not verifier_id.strip():
            raise ResearchGitError("independent verifier_id is required")
        report_payload = dict(report)
        verified = (
            report_payload.get("status") == "verified"
            and report_payload.get("claim_promotion_allowed") is True
            and not report_payload.get("required_failures")
        )
        relations = []
        for object_id in evaluates:
            self.repository.get(object_id)
            relations.append({"type": "evaluates", "target": object_id})
        review = self.repository.record(
            "review",
            report_payload,
            state="verified" if verified else "rejected",
            relations=relations,
            actor={
                "actor_id": verifier_id,
                "authority": "independent_evaluator",
            },
        )
        gate = self.repository.record(
            "gate_decision",
            {
                "decision": "promote" if verified else "hold",
                "claim_promotion_allowed": verified,
                "required_failures": list(
                    report_payload.get("required_failures") or []
                ),
                "report_hash": report_payload.get("report_hash"),
            },
            state="verified" if verified else "rejected",
            relations=[{"type": "evaluates", "target": review.object_id}],
            actor={
                "actor_id": "research-integrity-gate",
                "authority": "deterministic_gate",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="review",
                subject="record independent evidence gate",
                status=gate.state,
            )
            if commit
            else None
        )
        return {"review": review, "gate": gate, "checkpoint": checkpoint}

    def claim(
        self,
        payload: Mapping[str, Any],
        *,
        evidence_ids: Sequence[str],
        gate_id: str | None = None,
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not evidence_ids:
            raise ResearchGitError("claim requires evidence")
        relations: list[dict[str, str]] = []
        for object_id in evidence_ids:
            evidence = self.repository.get(object_id)
            if evidence["kind"] != "evidence":
                raise ResearchGitError("claim evidence reference has wrong kind")
            relations.append({"type": "depends_on", "target": object_id})
        if verified:
            if not gate_id:
                raise ResearchGitError("verified claim requires a gate decision")
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision" or gate["state"] != "verified":
                raise ResearchGitError(
                    "verified claim requires a passing gate decision"
                )
            relations.append({"type": "depends_on", "target": gate_id, "role": "gate"})
        result = self.repository.record(
            "claim",
            payload,
            state="verified" if verified else "draft",
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="evidence",
                subject="record evidence-bound claim",
                status=result.state,
            )
            if commit
            else None
        )
        return {"claim": result, "checkpoint": checkpoint}

    def manuscript(
        self,
        payload: Mapping[str, Any],
        *,
        claim_ids: Sequence[str],
        gate_id: str | None = None,
        final: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        relations: list[dict[str, str]] = []
        for object_id in claim_ids:
            claim = self.repository.get(object_id)
            if claim["kind"] != "claim":
                raise ResearchGitError("manuscript claim reference has wrong kind")
            if final and claim["state"] != "verified":
                raise ResearchGitError(
                    "final manuscript cannot include an unverified claim"
                )
            relations.append({"type": "depends_on", "target": object_id})
        if final:
            if not gate_id:
                raise ResearchGitError(
                    "final manuscript requires a passing gate decision"
                )
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision" or gate["state"] != "verified":
                raise ResearchGitError(
                    "final manuscript requires a passing gate decision"
                )
            relations.append({"type": "depends_on", "target": gate_id, "role": "gate"})
        result = self.repository.record(
            "manuscript",
            payload,
            state="completed" if final else "draft",
            relations=relations,
        )
        checkpoint = (
            self.repository.commit(
                stage="paper",
                subject=(
                    "freeze evidence-bound manuscript"
                    if final
                    else "record manuscript draft"
                ),
                status=result.state,
            )
            if commit
            else None
        )
        return {"manuscript": result, "checkpoint": checkpoint}


__all__ = ["ResearchLifecycle"]
