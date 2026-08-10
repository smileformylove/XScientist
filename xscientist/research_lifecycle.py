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
        provenance: Mapping[str, Any] | None = None,
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
            provenance=provenance,
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
        verifier_id: str | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not attempt_ids:
            raise ResearchGitError("evidence requires at least one experiment attempt")
        if verified and not str(verifier_id or "").strip():
            raise ResearchGitError("verified evidence requires verifier_id")
        relations: list[dict[str, str]] = []
        producer_ids: set[str] = set()
        for object_id in attempt_ids:
            attempt = self.repository.get(object_id)
            if attempt["kind"] != "experiment_attempt":
                raise ResearchGitError("evidence attempt reference has wrong kind")
            actor_id = str((attempt.get("actor") or {}).get("actor_id") or "")
            if actor_id:
                producer_ids.add(actor_id)
            relations.append({"type": "derived_from", "target": object_id})
        if verified and str(verifier_id) in producer_ids:
            raise ResearchGitError(
                "verified evidence requires a verifier independent of the experiment producer"
            )
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
            actor=(
                {
                    "actor_id": str(verifier_id),
                    "authority": "independent_evaluator",
                }
                if verified
                else None
            ),
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
        producer_ids: set[str] = set()
        for object_id in evaluates:
            evaluated = self.repository.get(object_id)
            actor_id = str((evaluated.get("actor") or {}).get("actor_id") or "")
            if actor_id:
                producer_ids.add(actor_id)
            relations.append({"type": "evaluates", "target": object_id})
        if verifier_id in producer_ids:
            raise ResearchGitError(
                "independent verifier must differ from the evaluated object's producer"
            )
        from .research_context import record_research_context_snapshot

        selected_option = "promote" if verified else "hold"
        context = record_research_context_snapshot(
            self.repository,
            target_ids=list(evaluates),
            decision_kind="independent_evidence_review",
            selected=selected_option,
            options_considered=[
                {
                    "option": "promote",
                    "rejected_because": (
                        "review is not verified or retains required failures"
                        if not verified
                        else ""
                    ),
                },
                {
                    "option": "hold",
                    "rejected_because": (
                        "independent review passed every required gate"
                        if verified
                        else ""
                    ),
                },
            ],
            rationale=[
                str(report_payload.get("summary") or "independent review decision")
            ],
            constraints=[
                str(value) for value in report_payload.get("required_failures") or []
            ],
            actor_id="research-review-context-recorder",
        )
        context_payload = self.repository.get(context.object_id)["payload"]
        report_payload["context_required"] = True
        report_payload["context_hash"] = context_payload["context_hash"]
        review = self.repository.record(
            "review",
            report_payload,
            state="verified" if verified else "rejected",
            relations=[
                *relations,
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
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
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="verified" if verified else "rejected",
            relations=[
                {"type": "evaluates", "target": review.object_id},
                {
                    "type": "depends_on",
                    "target": context.object_id,
                    "role": "decision_context",
                },
            ],
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
        return {
            "context": context,
            "review": review,
            "gate": gate,
            "checkpoint": checkpoint,
        }

    def claim(
        self,
        payload: Mapping[str, Any],
        *,
        evidence_ids: Sequence[str],
        qualification_ids: Sequence[str] = (),
        gate_id: str | None = None,
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        if not evidence_ids:
            raise ResearchGitError("claim requires evidence")
        relations: list[dict[str, str]] = []
        resolved_evidence_ids: list[str] = []
        for object_id in evidence_ids:
            evidence = self.repository.get(object_id)
            if evidence["kind"] not in {
                "evidence",
                "passage_evidence",
                "inference",
                "evidence_synthesis",
            }:
                raise ResearchGitError("claim evidence reference has wrong kind")
            resolved_id = str(evidence["object_id"])
            resolved_evidence_ids.append(resolved_id)
            relations.append({"type": "depends_on", "target": resolved_id})
        qualifications: dict[str, list[dict[str, Any]]] = {
            "mechanism_model": [],
            "evidence_quality": [],
            "transfer_matrix": [],
        }
        role_by_kind = {
            "mechanism_model": "mechanism",
            "evidence_quality": "quality",
            "transfer_matrix": "transfer",
        }
        for object_id in qualification_ids:
            qualification = self.repository.get(object_id)
            kind = str(qualification.get("kind") or "")
            if kind not in qualifications:
                raise ResearchGitError("claim qualification reference has wrong kind")
            qualifications[kind].append(qualification)
            relations.append(
                {
                    "type": "depends_on",
                    "target": str(qualification["object_id"]),
                    "role": role_by_kind[kind],
                }
            )
        depth_level = str(payload.get("depth_level") or "descriptive")
        if depth_level not in {"descriptive", "causal", "transferable"}:
            raise ResearchGitError("claim depth_level is invalid")
        if verified and depth_level in {"causal", "transferable"}:
            valid_mechanisms = [
                item
                for item in qualifications["mechanism_model"]
                if item.get("state") == "verified"
                and (item.get("payload") or {}).get("status") == "validated"
                and set(
                    (item.get("payload") or {}).get("evidence_ids") or []
                ).intersection(resolved_evidence_ids)
            ]
            valid_quality = [
                item
                for item in qualifications["evidence_quality"]
                if item.get("state") == "verified"
                and (item.get("payload") or {}).get("independent") is True
                and (item.get("payload") or {}).get("overall_grade")
                in {"strong", "moderate"}
                and (item.get("payload") or {}).get("evidence_id")
                in resolved_evidence_ids
            ]
            if not valid_mechanisms:
                raise ResearchGitError(
                    "verified causal claim requires a validated intervention-tested "
                    "mechanism bound to its evidence"
                )
            if not valid_quality:
                raise ResearchGitError(
                    "verified causal claim requires an independent strong/moderate "
                    "quality assessment of its evidence"
                )
            if depth_level == "transferable":
                valid_transfer = []
                claim_scope_hash = payload.get("scope_hash")
                claim_statement = " ".join(str(payload.get("statement") or "").split())
                for item in qualifications["transfer_matrix"]:
                    matrix_payload = item.get("payload") or {}
                    if (
                        item.get("state") != "verified"
                        or matrix_payload.get("transfer_ready") is not True
                    ):
                        continue
                    matrix_claim = self.repository.get(
                        str(matrix_payload.get("claim_id") or "")
                    )
                    matrix_claim_payload = matrix_claim.get("payload") or {}
                    matrix_statement = " ".join(
                        str(matrix_claim_payload.get("statement") or "").split()
                    )
                    if (
                        matrix_statement != claim_statement
                        or matrix_claim_payload.get("scope_hash") != claim_scope_hash
                    ):
                        continue
                    valid_transfer.append(item)
                if not valid_transfer:
                    raise ResearchGitError(
                        "verified transferable claim requires a passing transfer matrix "
                        "for the same statement and scope"
                    )
        if payload.get("contribution_level") == "method_discovery" and verified:
            supported = any(
                item["kind"] == "evidence_synthesis"
                and (item.get("payload") or {}).get("protocol_kind")
                == "generalization_assessment"
                and (item.get("payload") or {}).get("verdict")
                == "method_discovery_supported"
                for item in (
                    self.repository.get(object_id)
                    for object_id in resolved_evidence_ids
                )
            )
            if not supported:
                raise ResearchGitError(
                    "verified method-discovery claim requires a passing "
                    "generalization assessment"
                )
        if verified:
            if not gate_id:
                raise ResearchGitError("verified claim requires a gate decision")
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision" or gate["state"] != "verified":
                raise ResearchGitError(
                    "verified claim requires a passing gate decision"
                )
            evaluated = {
                str(item.get("target") or "")
                for item in gate.get("relations") or []
                if item.get("type") == "evaluates"
            }
            for target in list(evaluated):
                try:
                    linked = self.repository.get(target)
                except ResearchGitError:
                    continue
                if linked.get("kind") == "review":
                    evaluated.update(
                        str(item.get("target") or "")
                        for item in linked.get("relations") or []
                        if item.get("type") == "evaluates"
                    )
            if not evaluated.intersection(resolved_evidence_ids):
                raise ResearchGitError(
                    "verified claim gate does not evaluate the selected evidence"
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
        if final and not gate_id:
            raise ResearchGitError("final manuscript requires a passing gate decision")
        if gate_id:
            gate = self.repository.get(gate_id)
            if gate["kind"] != "gate_decision":
                raise ResearchGitError("manuscript gate reference has wrong kind")
            if final and gate["state"] != "verified":
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

    def reproduction(
        self,
        receipt: Mapping[str, Any],
        *,
        reproduces: Sequence[str],
        verifier_id: str | None = None,
        verified: bool = False,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Bind a compact reproduction receipt to the objects it checked."""

        from jsonschema import ValidationError, validate as validate_json

        from ai_scientist.protocol.schemas import load_schema

        if not reproduces:
            raise ResearchGitError("reproduction requires at least one target object")
        receipt_payload = dict(receipt)
        try:
            validate_json(receipt_payload, load_schema("reproduction_receipt"))
        except ValidationError as exc:
            raise ResearchGitError(
                f"invalid reproduction receipt: {exc.message}"
            ) from exc
        from ai_scientist.protocol import content_hash

        receipt_base = {
            key: value
            for key, value in receipt_payload.items()
            if key not in {"receipt_id", "content_hash"}
        }
        expected_hash = content_hash(receipt_base)
        expected_id = f"rr-{expected_hash.split(':', 1)[1][:16]}"
        if receipt_payload.get("content_hash") != expected_hash:
            raise ResearchGitError("reproduction receipt content hash mismatch")
        if receipt_payload.get("receipt_id") != expected_id:
            raise ResearchGitError("reproduction receipt identifier mismatch")
        if verified and not str(verifier_id or "").strip():
            raise ResearchGitError("verified reproduction requires verifier_id")
        if verified and (
            receipt_payload.get("verdict") != "passed"
            or receipt_payload.get("reproduction_level") != "computational_rerun"
            or receipt_payload.get("executed") is not True
            or receipt_payload.get("returncode") != 0
            or receipt_payload.get("timed_out") is True
            or receipt_payload.get("objects_complete") is not True
        ):
            raise ResearchGitError(
                "verified reproduction requires a successful computational rerun"
            )
        relations: list[dict[str, str]] = []
        producer_ids: set[str] = set()
        for object_id in reproduces:
            reproduced = self.repository.get(object_id)
            actor_id = str((reproduced.get("actor") or {}).get("actor_id") or "")
            if actor_id:
                producer_ids.add(actor_id)
            relations.append({"type": "reproduces", "target": object_id})
        if verified and str(verifier_id) in producer_ids:
            raise ResearchGitError(
                "verified reproduction requires a verifier independent of its targets"
            )
        payload = {
            key: receipt_payload[key]
            for key in (
                "receipt_id",
                "content_hash",
                "checkpoint_hash",
                "commit",
                "reproduction_level",
                "verdict",
                "objects_complete",
                "executed",
                "returncode",
                "timed_out",
            )
        }
        payload["receipt_hash"] = payload.pop("content_hash")
        payload["receipt"] = receipt_payload
        result = self.repository.record(
            "reproduction",
            payload,
            state=(
                "verified"
                if verified
                else (
                    "completed"
                    if receipt_payload.get("verdict") != "failed"
                    else "failed"
                )
            ),
            relations=relations,
            actor={
                "actor_id": str(verifier_id or "reproduction-recorder"),
                "authority": "independent_evaluator" if verified else "recorder",
            },
            provenance=(
                {
                    "environment_hash": str(
                        (receipt_payload.get("environment") or {}).get(
                            "recorded_content_hash"
                        )
                        or ""
                    )
                }
                if (receipt_payload.get("environment") or {}).get(
                    "recorded_content_hash"
                )
                else None
            ),
        )
        checkpoint = (
            self.repository.commit(
                stage="review",
                subject="record reproduction receipt",
                status=result.state,
            )
            if commit
            else None
        )
        return {"reproduction": result, "checkpoint": checkpoint}


__all__ = ["ResearchLifecycle"]
