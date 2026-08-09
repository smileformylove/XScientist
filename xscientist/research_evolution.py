"""Versioned, evidence-gated self-evolution on top of Research VCS."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ai_scientist.utils.evolution_gate import (
    validate_evolution_candidate,
    validate_evolution_gate,
    validate_production_promotion,
    validate_rollback_receipt,
)
from ai_scientist.utils.evolution_deployment import validate_deployment_receipt

from .research_git import CheckpointResult, ResearchGitError, ResearchObjectResult
from .research_context import record_research_context_snapshot
from .research_vcs import ResearchRepository


class ResearchEvolution:
    """Keep candidate agents shadow-only until independent promotion gates pass."""

    def __init__(self, repository: ResearchRepository | str | Path) -> None:
        self.repository = (
            repository
            if isinstance(repository, ResearchRepository)
            else ResearchRepository(repository)
        )

    def candidate_line(self, name: str, *, switch: bool = True) -> dict[str, Any]:
        normalized = str(name or "").strip().replace(" ", "-")
        if not normalized:
            raise ResearchGitError("evolution candidate line requires a name")
        return self.repository.fork(f"evolve/{normalized}", switch=switch)

    def candidate(
        self,
        payload: Mapping[str, Any],
        *,
        constitution: Mapping[str, Any],
        commit: bool = True,
    ) -> dict[str, Any]:
        candidate_payload = dict(payload)
        branch = self.repository.status()["branch"]
        if not str(branch).startswith("evolve/"):
            raise ResearchGitError(
                "agent candidates must be recorded on an evolve/* research line"
            )
        validation = validate_evolution_candidate(
            candidate_payload,
            constitution=dict(constitution),
        )
        if not validation["ok"]:
            raise ResearchGitError(
                "evolution candidate failed integrity validation: "
                + ", ".join(validation["errors"])
            )
        result = self.repository.record(
            "agent_candidate",
            candidate_payload,
            state="draft",
            actor={
                "actor_id": str(candidate_payload["proposed_by"]),
                "authority": "research_agent",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="evolve",
                subject=f"record shadow candidate {candidate_payload['candidate_id']}",
                status="draft",
            )
            if commit
            else None
        )
        return {"candidate": result, "checkpoint": checkpoint}

    def evaluate(
        self,
        payload: Mapping[str, Any],
        *,
        constitution: Mapping[str, Any],
        candidate_id: str,
        evaluator_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not evaluator_id.strip():
            raise ResearchGitError("independent evolution evaluator is required")
        candidate = self.repository.get(candidate_id)
        if candidate["kind"] != "agent_candidate" or candidate["state"] != "draft":
            raise ResearchGitError("evolution evaluation requires a draft candidate")
        gate_payload = dict(payload)
        validation = validate_evolution_gate(
            gate_payload,
            constitution=dict(constitution),
        )
        if not validation["passed"]:
            raise ResearchGitError(
                "evolution gate failed integrity validation: "
                + ", ".join(validation["errors"])
            )
        if (gate_payload.get("candidate") or {}).get("candidate_hash") != candidate[
            "payload"
        ].get("candidate_hash"):
            raise ResearchGitError("evolution gate targets a different candidate")
        passed = gate_payload.get("decision") == "promote_to_canary"
        context = record_research_context_snapshot(
            self.repository,
            target_ids=[candidate_id],
            decision_kind="agent_candidate_evaluation",
            selected="promote_to_canary" if passed else "hold",
            options_considered=[
                {
                    "option": "promote_to_canary",
                    "rejected_because": (
                        "independent evolution gate did not pass" if not passed else ""
                    ),
                },
                {
                    "option": "hold",
                    "rejected_because": (
                        "independent evolution gate passed" if passed else ""
                    ),
                },
            ],
            rationale=[
                str(gate_payload.get("summary") or gate_payload.get("decision"))
            ],
            constraints=[
                str(value) for value in gate_payload.get("required_failures") or []
            ],
            actor_id="agent-evaluation-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        gate_payload["context_required"] = True
        gate_payload["context_hash"] = context_payload["context_hash"]
        result = self.repository.record(
            "agent_evaluation",
            gate_payload,
            state="verified" if passed else "rejected",
            relations=[
                {"type": "evaluates", "target": candidate_id},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": evaluator_id,
                "authority": "independent_evaluator",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="review",
                subject="record independent agent evaluation",
                status=result.state,
            )
            if commit
            else None
        )
        return {"context": context, "evaluation": result, "checkpoint": checkpoint}

    def promote(
        self,
        payload: Mapping[str, Any],
        *,
        constitution: Mapping[str, Any],
        candidate_id: str,
        evaluation_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        candidate = self.repository.get(candidate_id)
        evaluation = self.repository.get(evaluation_id)
        if candidate["kind"] != "agent_candidate" or candidate["state"] != "draft":
            raise ResearchGitError("promotion requires a draft agent candidate")
        if (
            evaluation["kind"] != "agent_evaluation"
            or evaluation["state"] != "verified"
        ):
            raise ResearchGitError(
                "promotion requires a verified independent evaluation"
            )
        promotion_payload = dict(payload)
        validation = validate_production_promotion(
            promotion_payload,
            constitution=dict(constitution),
        )
        if not validation["passed"]:
            raise ResearchGitError(
                "production promotion failed integrity validation: "
                + ", ".join(validation["errors"])
            )
        if (
            promotion_payload.get("decision") != "approved"
            or promotion_payload.get("production_promotion_allowed") is not True
        ):
            raise ResearchGitError("production promotion remains blocked")
        exact_hash = candidate["payload"].get("candidate_hash")
        if (promotion_payload.get("candidate") or {}).get(
            "candidate_hash"
        ) != exact_hash or (evaluation["payload"].get("candidate") or {}).get(
            "candidate_hash"
        ) != exact_hash:
            raise ResearchGitError(
                "promotion evidence does not bind the exact candidate"
            )
        context = record_research_context_snapshot(
            self.repository,
            target_ids=[candidate_id, evaluation_id],
            decision_kind="agent_production_promotion",
            selected="approved",
            options_considered=[
                {"option": "approved", "rejected_because": ""},
                {
                    "option": "blocked",
                    "rejected_because": "candidate and independent evaluation satisfy the fixed promotion gate",
                },
            ],
            rationale=[
                "Promote only the exact independently evaluated candidate hash."
            ],
            constraints=["rollback reference must remain available"],
            actor_id="agent-promotion-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        promoted = self.repository.record(
            "agent_candidate",
            {
                "candidate": candidate["payload"],
                "promotion": promotion_payload,
            },
            state="promoted",
            relations=[
                {"type": "supersedes", "target": candidate_id},
                {"type": "evaluates", "target": evaluation_id},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": ",".join(promotion_payload.get("approver_ids") or []),
                "authority": "human",
            },
        )
        decision = self.repository.record(
            "gate_decision",
            {
                "decision": "promote",
                "candidate_hash": exact_hash,
                "promotion_hash": promotion_payload.get("promotion_hash"),
                "rollback_ref": promotion_payload.get("rollback_ref"),
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="promoted",
            relations=[
                {"type": "promotes", "target": promoted.object_id},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": "production-promotion-gate",
                "authority": "deterministic_gate",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="promotion",
                subject=f"promote agent candidate {exact_hash}",
                status="promoted",
            )
            if commit
            else None
        )
        return {
            "promoted_candidate": promoted,
            "context": context,
            "decision": decision,
            "checkpoint": checkpoint,
            "execution_mode": "semantic_receipt_only",
            "production_mutated": False,
            "next_action": (
                "apply the exact candidate artifact through an authorized deployment "
                "adapter, then record its independently observed canary receipt"
            ),
        }

    def rollback(
        self,
        receipt: Mapping[str, Any],
        *,
        candidate_id: str,
        promoted_id: str,
        trigger: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        candidate = self.repository.get(candidate_id)
        promoted = self.repository.get(promoted_id)
        if candidate["kind"] != "agent_candidate":
            raise ResearchGitError("rollback candidate reference has wrong kind")
        if promoted["kind"] != "agent_candidate" or promoted["state"] != "promoted":
            raise ResearchGitError("rollback requires a promoted agent candidate")
        receipt_payload = dict(receipt)
        validation = validate_rollback_receipt(
            receipt_payload,
            candidate=candidate["payload"],
        )
        if not validation["passed"]:
            raise ResearchGitError(
                "rollback receipt failed integrity validation: "
                + ", ".join(validation["errors"])
            )
        context = record_research_context_snapshot(
            self.repository,
            target_ids=[candidate_id, promoted_id],
            decision_kind="agent_rollback",
            selected="rollback",
            options_considered=[
                {"option": "rollback", "rejected_because": ""},
                {
                    "option": "continue_deployment",
                    "rejected_because": str(
                        trigger or "verified rollback trigger fired"
                    ),
                },
            ],
            rationale=[str(trigger or "verified rollback trigger fired")],
            constraints=["restore the approved baseline artifact"],
            actor_id="agent-rollback-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        decision = self.repository.record(
            "gate_decision",
            {
                "decision": "rollback",
                "trigger": trigger,
                "receipt": receipt_payload,
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="superseded",
            relations=[
                {"type": "supersedes", "target": promoted_id},
                {"type": "depends_on", "target": candidate_id, "role": "baseline"},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": str(receipt_payload.get("executed_by") or "rollback-gate"),
                "authority": "deterministic_gate",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="rollback",
                subject="record verified agent rollback",
                status="superseded",
            )
            if commit
            else None
        )
        return {
            "decision": decision,
            "context": context,
            "checkpoint": checkpoint,
            "execution_mode": (
                "verified_external_rollback"
                if receipt_payload.get("exercise_only") is False
                else "semantic_receipt_only"
            ),
            "production_mutated": receipt_payload.get("exercise_only") is False,
            "receipt_mode": (
                "production"
                if receipt_payload.get("exercise_only") is False
                else "exercise"
            ),
            "next_action": (
                "rollback receipt recorded; no further production mutation is implied"
                if receipt_payload.get("exercise_only") is False
                else "restore the approved baseline through an authorized rollback adapter "
                "and attach a production rollback receipt"
            ),
        }

    def deployment(
        self,
        receipt: Mapping[str, Any],
        *,
        promoted_id: str,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Attach an observed production deployment to the promoted candidate."""

        promoted = self.repository.get(promoted_id)
        if promoted["kind"] != "agent_candidate" or promoted["state"] != "promoted":
            raise ResearchGitError(
                "deployment receipt requires a promoted agent candidate"
            )
        candidate_payload = promoted["payload"].get("candidate") or {}
        expected_hash = candidate_payload.get("candidate_artifact_hash")
        validation = validate_deployment_receipt(
            receipt, candidate_artifact_hash=expected_hash
        )
        if not validation["ok"]:
            raise ResearchGitError(
                "deployment receipt failed integrity validation: "
                + ", ".join(validation["errors"])
            )
        receipt_payload = validation["receipt"]
        if (
            receipt_payload.get("mode") != "production"
            or receipt_payload.get("production_mutated") is not True
        ):
            raise ResearchGitError(
                "Research VCS production deployment requires an applied production receipt"
            )
        context = record_research_context_snapshot(
            self.repository,
            target_ids=[promoted_id],
            decision_kind="agent_production_deployment",
            selected="deploy",
            options_considered=[
                {"option": "deploy", "rejected_because": ""},
                {
                    "option": "hold",
                    "rejected_because": "verified production receipt confirms the approved artifact was applied",
                },
            ],
            rationale=[
                "Bind the observed production mutation to the promoted artifact."
            ],
            constraints=[
                "deployment receipt must validate and identify the exact artifact"
            ],
            actor_id="agent-deployment-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        decision = self.repository.record(
            "gate_decision",
            {
                "decision": "deployed",
                "candidate_hash": candidate_payload.get("candidate_hash"),
                "candidate_artifact_hash": expected_hash,
                "deployment_receipt": receipt_payload,
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="promoted",
            relations=[
                {"type": "depends_on", "target": promoted_id, "role": "deployment"},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
            actor={
                "actor_id": str(receipt_payload.get("executed_by")),
                "authority": "deterministic_gate",
            },
        )
        checkpoint = (
            self.repository.commit(
                stage="deployment",
                subject="record verified production deployment",
                status="promoted",
            )
            if commit
            else None
        )
        return {"context": context, "decision": decision, "checkpoint": checkpoint}


__all__ = ["ResearchEvolution"]
