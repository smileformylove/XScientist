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

from .research_git import CheckpointResult, ResearchGitError, ResearchObjectResult
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
        result = self.repository.record(
            "agent_evaluation",
            gate_payload,
            state="verified" if passed else "rejected",
            relations=[{"type": "evaluates", "target": candidate_id}],
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
        return {"evaluation": result, "checkpoint": checkpoint}

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
            },
            state="promoted",
            relations=[{"type": "promotes", "target": promoted.object_id}],
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
            "decision": decision,
            "checkpoint": checkpoint,
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
        decision = self.repository.record(
            "gate_decision",
            {"decision": "rollback", "trigger": trigger, "receipt": receipt_payload},
            state="superseded",
            relations=[
                {"type": "supersedes", "target": promoted_id},
                {"type": "depends_on", "target": candidate_id, "role": "baseline"},
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
        return {"decision": decision, "checkpoint": checkpoint}


__all__ = ["ResearchEvolution"]
