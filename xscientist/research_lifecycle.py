"""Scientific lifecycle operations built on native Research VCS objects."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ai_scientist.protocol import content_hash
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.principal_identity import canonical_principal
from ai_scientist.utils.research_integrity import (
    _protocol_fidelity_hash,
    validate_preregistration,
)

from .research_git import (
    CheckpointResult,
    ResearchGitError,
    ResearchObjectResult,
    _pointer_records_at_commit,
    show_checkpoint,
)
from ai_scientist.utils.trajectory_binding import terminal_negative_contract_errors
from .research_vcs import ResearchRepository
from .research_authority import require_independent_evaluator

_MAX_CONFIRMATORY_LINEAGE_COMMITS = 4096
_MAX_CONFIRMATORY_CAMPAIGN_BYTES = 16 * 1024 * 1024
_RESEARCH_OBJECT_PATH = re.compile(
    r"^\.xscientist/objects/(?P<kind>[a-z][a-z0-9_-]*)/"
    r"(?P<object_id>rso-[0-9a-f]{16})\.json$"
)
_CONTENT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _relation_targets(
    research_object: Mapping[str, Any],
    *,
    relation_type: str,
    role: str | None = None,
) -> set[str]:
    return {
        str(item.get("target") or "").strip()
        for item in research_object.get("relations") or []
        if isinstance(item, Mapping)
        and item.get("type") == relation_type
        and (role is None or item.get("role") == role)
        and str(item.get("target") or "").strip()
    }


def _registered_task_ids(preregistration: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("task_id") or "").strip()
        for item in preregistration.get("outcomes") or []
        if isinstance(item, Mapping) and str(item.get("task_id") or "").strip()
    }


def _safe_campaign_path(value: Any) -> str:
    text = str(value or "").strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ResearchGitError("confirmatory campaign artifact path is unsafe")
    return path.as_posix()


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

        # Validate the whole multi-object transition before the first immutable
        # object is written.  A malformed locked registration must not leave a
        # hypothesis and plan behind for a later checkpoint to absorb.
        registration_payload: dict[str, Any] | None = None
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
        if registration_payload is not None:
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

    def _validate_confirmatory_attempt_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
        preregistration: Mapping[str, Any],
        adaptive_freeze: Mapping[str, Any],
    ) -> str:
        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "experiment_attempt":
            raise ResearchGitError(
                "confirmatory experiment lineage changed a frozen research path: "
                + path
            )
        object_id = match.group("object_id")
        attempt = self.repository.get(object_id)
        if attempt.get("kind") != "experiment_attempt":
            raise ResearchGitError(
                "confirmatory experiment lineage contains a non-attempt object"
            )
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory experiment lineage contains an untraceable attempt"
            ) from exc
        if str(origin.get("commit") or "") != commit:
            raise ResearchGitError(
                "confirmatory experiment lineage modified an existing attempt"
            )
        payload = attempt.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        if str(payload.get("study_phase") or "").strip().lower() != "confirmatory":
            raise ResearchGitError(
                "confirmatory experiment lineage contains an exploratory attempt"
            )
        if _relation_targets(
            attempt,
            relation_type="depends_on",
            role="protocol",
        ) != {preregistration_id}:
            raise ResearchGitError(
                "confirmatory experiment lineage contains an attempt bound to a "
                "different protocol"
            )
        expected_bindings = {
            "adaptive_state_hash": adaptive_freeze.get("state_hash"),
            "research_state_hash": adaptive_freeze.get("research_state_hash"),
            "post_freeze_adaptation": False,
            "preregistration_id": preregistration.get("preregistration_id"),
        }
        task_id = str(payload.get("task_id") or "").strip()
        if task_id and task_id not in _registered_task_ids(preregistration):
            raise ResearchGitError(
                "confirmatory experiment lineage contains an unregistered task"
            )
        data_policy = preregistration.get("data_policy")
        data_policy = data_policy if isinstance(data_policy, Mapping) else {}
        for field in ("data_manifest_hash", "data_snapshot_id"):
            expected_bindings[field] = data_policy.get(field)
        if task_id:
            outcome = next(
                (
                    item
                    for item in preregistration.get("outcomes") or []
                    if isinstance(item, Mapping)
                    and str(item.get("task_id") or "").strip() == task_id
                ),
                {},
            )
            for field in (
                "evidence_role",
                "paired_control_task_id",
                "intervention_variant",
                "stress_condition",
                "transformation_contract",
                "transformation_contract_hash",
            ):
                expected_bindings[field] = outcome.get(field)
            expected_bindings["protocol_fidelity_hash"] = _protocol_fidelity_hash(
                preregistration, task_id
            )
            expected_bindings["dataset_split_hash"] = (
                data_policy.get("split_hashes") or {}
            ).get(task_id)
        if any(
            payload.get(field) != expected
            for field, expected in expected_bindings.items()
        ):
            raise ResearchGitError(
                "confirmatory experiment lineage contains an attempt that changed "
                "the frozen adaptive state, data, or task contract"
            )
        provenance = attempt.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        code_commit = str(provenance.get("code_commit") or "").strip()
        if code_commit != str(adaptive_freeze.get("research_vcs_head") or "").strip():
            raise ResearchGitError(
                "confirmatory experiment lineage is missing or changed the frozen "
                "code state"
            )
        return object_id

    def _validate_confirmatory_gate_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
    ) -> tuple[str, str]:
        """Validate a post-freeze trajectory binding/disposition decision."""

        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "gate_decision":
            raise ResearchGitError(
                "confirmatory lineage contains a non-attempt/non-gate object"
            )
        object_id = match.group("object_id")
        decision = self.repository.get(object_id)
        if decision.get("kind") != "gate_decision" or decision.get("state") not in {
            "completed",
            "verified",
        }:
            raise ResearchGitError(
                "confirmatory lineage contains an invalid gate decision"
            )
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory lineage contains an untraceable gate decision"
            ) from exc
        if str(origin.get("commit") or "") != commit:
            raise ResearchGitError(
                "confirmatory lineage modified an existing gate decision"
            )
        payload = decision.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        protocol_kind = str(payload.get("protocol_kind") or "").strip()
        if protocol_kind not in {
            "xscientist.trajectory-binding.v1",
            "xscientist.attempt-disposition.v1",
        }:
            raise ResearchGitError(
                "confirmatory lineage contains an unrecognized gate decision"
            )
        actor = decision.get("actor")
        actor = actor if isinstance(actor, Mapping) else {}
        expected_authority = (
            "deterministic_gate"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "recorder"
        )
        if (
            actor.get("authority") != expected_authority
            or not str(actor.get("actor_id") or "").strip()
        ):
            raise ResearchGitError(
                "confirmatory gate decision actor authority is invalid"
            )
        hash_field = (
            "binding_hash"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "disposition_hash"
        )
        payload_core = {
            key: value for key, value in payload.items() if key != hash_field
        }
        if payload.get(hash_field) != canonical_content_hash(payload_core):
            raise ResearchGitError(
                "confirmatory gate decision integrity hash is invalid"
            )
        expected_decision = (
            "bind_registry_record_to_research_trajectory"
            if protocol_kind == "xscientist.trajectory-binding.v1"
            else "record_attempt_disposition"
        )
        if payload.get("decision") != expected_decision:
            raise ResearchGitError("confirmatory gate decision action is invalid")
        if _relation_targets(decision, relation_type="depends_on", role="protocol") != {
            preregistration_id
        }:
            raise ResearchGitError(
                "confirmatory gate decision is bound to a different protocol"
            )
        attempt_targets = _relation_targets(decision, relation_type="attests")
        attempt_id = str(payload.get("attempt_id") or "").strip()
        if attempt_targets != {attempt_id}:
            raise ResearchGitError(
                "confirmatory gate decision does not attest its declared attempt"
            )
        attempt = self.repository.get(attempt_id)
        if (
            attempt.get("kind") != "experiment_attempt"
            or (attempt.get("payload") or {}).get("study_phase") != "confirmatory"
            or _relation_targets(attempt, relation_type="depends_on", role="protocol")
            != {preregistration_id}
            or payload.get("attempt_content_hash") != attempt.get("content_hash")
        ):
            raise ResearchGitError(
                "confirmatory gate decision attempt binding is invalid"
            )
        if protocol_kind == "xscientist.trajectory-binding.v1":
            required_text_fields = (
                "record_id",
                "registry_row_hash",
                "attempt_origin_commit",
                "attempt_checkpoint_hash",
            )
            if any(
                not str(payload.get(field) or "").strip()
                for field in required_text_fields
            ) or any(
                _CONTENT_HASH_RE.fullmatch(str(payload.get(field) or "")) is None
                for field in (
                    "registry_row_hash",
                    "attempt_content_hash",
                    "attempt_checkpoint_hash",
                    "binding_hash",
                )
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding payload is incomplete"
                )
            if _relation_targets(
                decision,
                relation_type="depends_on",
                role="retry",
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding has an invalid retry relation"
                )
            attempt_origin = (
                self.repository.blame(attempt_id, commit=current_head).get("origin")
                or {}
            )
            attempt_origin_commit = str(attempt_origin.get("commit") or "").strip()
            try:
                attempt_checkpoint = self.repository.show(attempt_origin_commit)
            except ResearchGitError as exc:
                raise ResearchGitError(
                    "confirmatory trajectory binding attempt checkpoint is invalid"
                ) from exc
            if (
                payload.get("attempt_origin_commit") != attempt_origin_commit
                or attempt_checkpoint.get("checkpoint_hash_valid") is not True
                or payload.get("attempt_checkpoint_hash")
                != (attempt_checkpoint.get("checkpoint") or {}).get("content_hash")
            ):
                raise ResearchGitError(
                    "confirmatory trajectory binding attempt checkpoint is invalid"
                )
        else:
            disposition = str(payload.get("disposition") or "").strip()
            if (
                disposition
                not in {
                    "terminal_negative",
                    "technical_failure_retried",
                    "approved_deviation",
                    "excluded_with_reason",
                }
                or not str(payload.get("attempt_record_id") or "").strip()
                or not str(payload.get("reason") or "").strip()
                or _CONTENT_HASH_RE.fullmatch(
                    str(payload.get("attempt_record_hash") or "")
                )
                is None
                or _CONTENT_HASH_RE.fullmatch(
                    str(payload.get("attempt_content_hash") or "")
                )
                is None
                or not isinstance(payload.get("approved_before_unblinding"), bool)
                or not isinstance(payload.get("negative_result_preserved"), bool)
            ):
                raise ResearchGitError(
                    "confirmatory attempt disposition payload is incomplete"
                )
            retry_targets = _relation_targets(
                decision,
                relation_type="depends_on",
                role="retry",
            )
            retry_attempt_id = str(payload.get("retry_attempt_id") or "").strip()
            retry_record_id = str(payload.get("retry_record_id") or "").strip()
            if disposition == "technical_failure_retried":
                if (
                    not retry_record_id
                    or not retry_attempt_id
                    or retry_targets != {retry_attempt_id}
                ):
                    raise ResearchGitError(
                        "confirmatory attempt disposition retry binding is invalid"
                    )
                retry_attempt = self.repository.get(retry_attempt_id)
                retry_payload = retry_attempt.get("payload") or {}
                attempt_payload = attempt.get("payload") or {}
                if (
                    retry_attempt.get("kind") != "experiment_attempt"
                    or retry_attempt.get("state") != "completed"
                    or retry_payload.get("study_phase") != "confirmatory"
                    or _relation_targets(
                        retry_attempt,
                        relation_type="depends_on",
                        role="protocol",
                    )
                    != {preregistration_id}
                    or retry_payload.get("task_id") != attempt_payload.get("task_id")
                    or retry_payload.get("preregistration_id")
                    != attempt_payload.get("preregistration_id")
                ):
                    raise ResearchGitError(
                        "confirmatory attempt disposition retry attempt is invalid"
                    )
            elif retry_record_id or retry_attempt_id or retry_targets:
                raise ResearchGitError(
                    "confirmatory attempt disposition has an unexpected retry binding"
                )
            if (
                disposition == "approved_deviation"
                and payload.get("approved_before_unblinding") is not True
            ):
                raise ResearchGitError(
                    "confirmatory attempt disposition approval is invalid"
                )
            if disposition == "terminal_negative":
                terminal_errors = terminal_negative_contract_errors(
                    self.repository,
                    decision,
                    attempt,
                )
                if terminal_errors:
                    raise ResearchGitError(
                        "confirmatory attempt disposition negative result is not "
                        "host-verifiable: " + ", ".join(terminal_errors)
                    )
        return object_id, protocol_kind

    def _validate_confirmatory_negative_evidence_path(
        self,
        path: str,
        *,
        commit: str,
        current_head: str,
        preregistration_id: str,
    ) -> str:
        """Admit only a metric-bearing assessment of one scientific failure."""

        match = _RESEARCH_OBJECT_PATH.fullmatch(path)
        if match is None or match.group("kind") != "evidence":
            raise ResearchGitError(
                "confirmatory lineage contains an unsupported evidence object"
            )
        object_id = match.group("object_id")
        evidence = self.repository.get(object_id)
        try:
            origin = (
                self.repository.blame(object_id, commit=current_head).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory lineage contains untraceable negative evidence"
            ) from exc
        payload = evidence.get("payload")
        payload = payload if isinstance(payload, Mapping) else {}
        attempt_ids = _relation_targets(
            evidence,
            relation_type="derived_from",
        )
        if len(attempt_ids) != 1:
            raise ResearchGitError(
                "confirmatory negative evidence must derive from exactly one attempt"
            )
        attempt = self.repository.get(next(iter(attempt_ids)))
        attempt_payload = attempt.get("payload")
        attempt_payload = (
            attempt_payload if isinstance(attempt_payload, Mapping) else {}
        )
        expected_measurement_hash = content_hash(
            {
                "result": payload.get("result"),
                "metrics": payload.get("metrics") or {},
            }
        )
        if (
            evidence.get("kind") != "evidence"
            or evidence.get("state") not in {"completed", "verified"}
            or str(origin.get("commit") or "") != commit
            or attempt.get("kind") != "experiment_attempt"
            or attempt.get("state") != "failed"
            or attempt_payload.get("study_phase") != "confirmatory"
            or str(attempt_payload.get("failure_class") or "").strip().lower()
            != "scientific_negative_result"
            or _relation_targets(
                attempt,
                relation_type="depends_on",
                role="protocol",
            )
            != {preregistration_id}
            or not str(payload.get("result") or "").strip()
            or not isinstance(payload.get("metrics"), Mapping)
            or not payload.get("metrics")
            or payload.get("measurement_hash") != expected_measurement_hash
        ):
            raise ResearchGitError(
                "confirmatory negative evidence is not bound to one host-verifiable "
                "scientific negative attempt"
            )
        return object_id

    def _confirmatory_lineage_attestation(
        self,
        *,
        registration: Mapping[str, Any],
        preregistration_id: str,
        preregistration: Mapping[str, Any],
        adaptive_freeze: Mapping[str, Any],
        allowed_pending_paths: set[str] | None = None,
    ) -> dict[str, Any]:
        """Prove every post-freeze path is a protocol-bound immutable attempt.

        Merely checking that ``HEAD`` descends from the frozen commit is unsafe:
        a descendant could change the hypothesis, method, code, memory, protocol,
        or evaluator.  This verifier walks the first-parent checkpoint chain and
        accepts only the exact preregistration transition followed by newly added
        confirmatory-attempt objects bound to this registration.
        """

        current_state = self.repository.status()
        pending_paths = {
            *list(current_state.get("staged_paths") or []),
            *list(current_state.get("tracked_changes") or []),
            *list(current_state.get("eligible_changes") or []),
        }
        pending_allowed = bool(allowed_pending_paths) and not (
            pending_paths - set(allowed_pending_paths or set())
        )
        if current_state.get("worktree_clean") is not True and not pending_allowed:
            raise ResearchGitError(
                "confirmatory experiment requires the exact clean Research VCS "
                "lineage frozen at preregistration"
            )
        frozen_head = str(adaptive_freeze.get("research_vcs_head") or "").strip()
        current_head = str(current_state.get("head") or "").strip()
        if not frozen_head or not current_head:
            raise ResearchGitError(
                "confirmatory experiment requires a committed Research VCS freeze"
            )

        registration_object_id = str(
            registration.get("object_id") or preregistration_id or ""
        ).strip()
        plan_ids = []
        for target in sorted(
            _relation_targets(registration, relation_type="depends_on")
        ):
            try:
                dependency = self.repository.get(target)
            except ResearchGitError:
                continue
            if dependency.get("kind") == "research_plan":
                plan_ids.append(str(dependency.get("object_id") or ""))
        if not registration_object_id or len(plan_ids) != 1:
            raise ResearchGitError(
                "confirmatory experiment requires one committed locked research plan"
            )
        plan_object_id = plan_ids[0]
        try:
            registration_origin = (
                self.repository.blame(
                    registration_object_id,
                    commit=current_head,
                ).get("origin")
                or {}
            )
            plan_origin = (
                self.repository.blame(
                    plan_object_id,
                    commit=current_head,
                ).get("origin")
                or {}
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "confirmatory experiment requires a committed preregistration "
                "transition"
            ) from exc
        registration_commit = str(registration_origin.get("commit") or "").strip()
        if (
            not registration_commit
            or str(plan_origin.get("commit") or "").strip() != registration_commit
        ):
            raise ResearchGitError(
                "confirmatory experiment requires the plan and preregistration in "
                "one committed transition"
            )

        expected_registration_paths = {
            ".xscientist/objects/preregistration/" f"{registration_object_id}.json",
            ".xscientist/objects/research_plan/" f"{plan_object_id}.json",
        }
        campaign = preregistration.get("confirmatory_campaign")
        campaign = campaign if isinstance(campaign, Mapping) else {}
        expected_campaign_paths: set[str] = set()
        allowed_registry_paths: set[str] = set()
        if campaign:
            if campaign.get("schema") != "xscientist.confirmatory-campaign.v1":
                raise ResearchGitError("confirmatory campaign schema is invalid")
            queue_contract = campaign.get("queue_contract")
            queue_contract = (
                queue_contract if isinstance(queue_contract, Mapping) else {}
            )
            queue_contract_core = {
                key: value
                for key, value in queue_contract.items()
                if key != "queue_contract_hash"
            }
            queue_contract_hash = canonical_content_hash(queue_contract_core)
            if (
                queue_contract.get("schema")
                != "xscientist.confirmatory-queue-contract.v1"
                or queue_contract.get("queue_contract_hash") != queue_contract_hash
                or campaign.get("queue_contract_hash") != queue_contract_hash
            ):
                raise ResearchGitError(
                    "confirmatory campaign queue contract is missing or hash-invalid"
                )
            expected_campaign_paths = {
                _safe_campaign_path(campaign.get("preregistration_path")),
                _safe_campaign_path(campaign.get("queue_path")),
            }
            if len(expected_campaign_paths) != 2:
                raise ResearchGitError(
                    "confirmatory campaign requires distinct registration and queue "
                    "artifacts"
                )
            paper_prefix = PurePosixPath(
                _safe_campaign_path(campaign.get("preregistration_path"))
            ).parent
            allowed_registry_paths = {
                (paper_prefix / filename).as_posix()
                for filename in (
                    "experiment_registry.jsonl",
                    "experiment_registry.integrity.json",
                    "experiment_registry.history.jsonl",
                    "pipeline_manifest.json",
                    "stage_standards.json",
                    "process_alignment.json",
                )
            }
        cursor = current_head
        visited: set[str] = set()
        checked_paths: list[dict[str, str]] = []
        prior_attempt_ids: list[str] = []
        prior_gate_decision_ids: list[str] = []
        registration_seen = False
        campaign_seen = not expected_campaign_paths
        for _ in range(_MAX_CONFIRMATORY_LINEAGE_COMMITS):
            if cursor == frozen_head:
                break
            if not cursor or cursor in visited:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage is cyclic or broken"
                )
            visited.add(cursor)
            try:
                shown = self.repository.show(cursor)
            except ResearchGitError as exc:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage is not fully "
                    "checkpoint-attested"
                ) from exc
            if shown.get("checkpoint_hash_valid") is not True:
                raise ResearchGitError(
                    "confirmatory experiment Research VCS lineage has an invalid "
                    "checkpoint hash"
                )
            checkpoint = shown.get("checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
            changed_paths = {
                str(path) for path in checkpoint.get("changed_paths") or []
            }
            parent = str(checkpoint.get("parent_commit") or "").strip()
            if cursor == registration_commit:
                if (
                    registration_seen
                    or checkpoint.get("stage") != "preregister"
                    or parent != frozen_head
                    or changed_paths != expected_registration_paths
                ):
                    raise ResearchGitError(
                        "confirmatory experiment requires the exact clean Research "
                        "VCS preregistration transition"
                    )
                registration_seen = True
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "frozen_contract",
                    }
                    for path in sorted(changed_paths)
                )
            else:
                if expected_campaign_paths and checkpoint.get("stage") == "preregister":
                    if (
                        campaign_seen
                        or parent != registration_commit
                        or changed_paths != expected_campaign_paths
                    ):
                        raise ResearchGitError(
                            "confirmatory campaign artifacts are not the exact "
                            "post-registration materialization transition"
                        )
                    preregistration_path = self.repository.path / _safe_campaign_path(
                        campaign.get("preregistration_path")
                    )
                    queue_path = self.repository.path / _safe_campaign_path(
                        campaign.get("queue_path")
                    )
                    try:
                        if (
                            preregistration_path.is_symlink()
                            or queue_path.is_symlink()
                            or not preregistration_path.is_file()
                            or not queue_path.is_file()
                            or preregistration_path.stat().st_size
                            > _MAX_CONFIRMATORY_CAMPAIGN_BYTES
                            or queue_path.stat().st_size
                            > _MAX_CONFIRMATORY_CAMPAIGN_BYTES
                        ):
                            raise ValueError("campaign artifact boundary violation")
                        saved_registration = json.loads(
                            preregistration_path.read_text(encoding="utf-8")
                        )
                        saved_queue = json.loads(queue_path.read_text(encoding="utf-8"))
                    except (
                        OSError,
                        UnicodeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise ResearchGitError(
                            "confirmatory campaign artifacts are unreadable or unsafe"
                        ) from exc
                    if saved_registration != preregistration:
                        raise ResearchGitError(
                            "confirmatory campaign registration artifact differs from "
                            "the locked Research VCS object"
                        )
                    if not isinstance(saved_queue, Mapping):
                        raise ResearchGitError(
                            "confirmatory campaign queue must be a JSON object"
                        )
                    queue_core = {
                        key: value
                        for key, value in saved_queue.items()
                        if key != "queue_hash"
                    }
                    queue_valid = bool(
                        saved_queue.get("schema") == "xscientist.confirmatory-queue.v1"
                        and saved_queue.get("preregistration_object_id")
                        == registration_object_id
                        and saved_queue.get("plan_object_id") == plan_object_id
                        and saved_queue.get("registration_hash")
                        == preregistration.get("registration_hash")
                        and saved_queue.get("frozen_state_hash")
                        == adaptive_freeze.get("state_hash")
                        and saved_queue.get("data_manifest_hash")
                        == (preregistration.get("data_policy") or {}).get(
                            "data_manifest_hash"
                        )
                        and saved_queue.get("data_snapshot_id")
                        == (preregistration.get("data_policy") or {}).get(
                            "data_snapshot_id"
                        )
                        and saved_queue.get("queue_hash")
                        == canonical_content_hash(queue_core)
                    )
                    queued_task_ids = {
                        str(item.get("task_id") or "").strip()
                        for item in saved_queue.get("tasks") or []
                        if isinstance(item, Mapping)
                    }
                    queue_tasks = {
                        str(item.get("task_id") or "").strip(): item
                        for item in saved_queue.get("tasks") or []
                        if isinstance(item, Mapping)
                        and str(item.get("task_id") or "").strip()
                    }
                    data_policy = preregistration.get("data_policy")
                    data_policy = (
                        data_policy if isinstance(data_policy, Mapping) else {}
                    )
                    outcome_contracts_valid = True
                    for outcome in preregistration.get("outcomes") or []:
                        if not isinstance(outcome, Mapping):
                            outcome_contracts_valid = False
                            continue
                        task_id = str(outcome.get("task_id") or "").strip()
                        queued = queue_tasks.get(task_id) or {}
                        expected_fields = {
                            "dataset": outcome.get("dataset"),
                            "metric": outcome.get("metric"),
                            "baseline": outcome.get("baseline"),
                            "evidence_role": outcome.get("evidence_role"),
                            "paired_control_task_id": outcome.get(
                                "paired_control_task_id"
                            ),
                            "intervention_variant": outcome.get("intervention_variant"),
                            "stress_condition": outcome.get("stress_condition"),
                            "transformation_contract": outcome.get(
                                "transformation_contract"
                            ),
                            "transformation_contract_hash": outcome.get(
                                "transformation_contract_hash"
                            ),
                            "split_hash": (data_policy.get("split_hashes") or {}).get(
                                task_id
                            ),
                            "data_manifest_hash": data_policy.get("data_manifest_hash"),
                            "data_snapshot_id": data_policy.get("data_snapshot_id"),
                        }
                        if any(
                            queued.get(field) != expected
                            for field, expected in expected_fields.items()
                        ):
                            outcome_contracts_valid = False
                    contract_tasks = {
                        str(item.get("task_id") or "").strip(): item
                        for item in queue_contract.get("tasks") or []
                        if isinstance(item, Mapping)
                        and str(item.get("task_id") or "").strip()
                    }
                    materialized_contract_valid = set(contract_tasks) == set(
                        queue_tasks
                    )
                    for task_id, contract_task in contract_tasks.items():
                        queued = queue_tasks.get(task_id) or {}
                        contract_core = {
                            key: value
                            for key, value in contract_task.items()
                            if key != "task_contract_hash"
                        }
                        expected_prefix = [
                            "xscientist",
                            "research",
                            "experiment",
                            "{RESULT_SUMMARY}",
                            "--study-phase",
                            "confirmatory",
                            "--task",
                            task_id,
                            "--plan",
                            plan_object_id,
                            "--preregistration",
                            registration_object_id,
                            "--repo",
                            ".",
                        ]
                        if (
                            contract_task.get("task_contract_hash")
                            != canonical_content_hash(contract_core)
                            or any(
                                queued.get(field) != value
                                for field, value in contract_task.items()
                            )
                            or queued.get("record_command_prefix") != expected_prefix
                            or queued.get("bound_object_ids")
                            != {
                                "plan": plan_object_id,
                                "preregistration": registration_object_id,
                            }
                            or queued.get("status") != "queued"
                            or queued.get("study_phase") != "confirmatory"
                            or queued.get("adaptive_state_hash")
                            != adaptive_freeze.get("state_hash")
                            or queued.get("research_state_hash")
                            != adaptive_freeze.get("research_state_hash")
                            or queued.get("post_freeze_adaptation") is not False
                        ):
                            materialized_contract_valid = False
                    if (
                        not queue_valid
                        or saved_queue.get("queue_contract_hash") != queue_contract_hash
                        or queued_task_ids != _registered_task_ids(preregistration)
                        or len(queue_tasks)
                        != len(_registered_task_ids(preregistration))
                        or not outcome_contracts_valid
                        or not materialized_contract_valid
                    ):
                        raise ResearchGitError(
                            "confirmatory campaign queue is not bound to the locked "
                            "plan, data, and adaptive state"
                        )
                    campaign_seen = True
                    checked_paths.extend(
                        {
                            "commit": cursor,
                            "path": path,
                            "classification": "derived_campaign_artifact",
                        }
                        for path in sorted(changed_paths)
                    )
                    cursor = parent
                    continue
                if registration_seen or checkpoint.get("stage") not in {
                    "experiment",
                    "failed",
                    "evidence",
                    "review",
                }:
                    raise ResearchGitError(
                        "confirmatory experiment requires the exact clean Research "
                        "VCS lineage; found a non-confirmatory post-freeze transition"
                    )
                if not changed_paths:
                    raise ResearchGitError(
                        "confirmatory experiment lineage contains an empty transition"
                    )
                registry_artifact_paths = changed_paths & allowed_registry_paths
                pointer_paths = {
                    path
                    for path in changed_paths
                    if path.startswith("research-objects/") and path.endswith(".json")
                }
                object_paths = changed_paths - registry_artifact_paths - pointer_paths
                if registry_artifact_paths and checkpoint.get("stage") != "review":
                    raise ResearchGitError(
                        "confirmatory registry artifacts must be admitted with a "
                        "trajectory-binding review transition"
                    )
                if registry_artifact_paths and not object_paths:
                    raise ResearchGitError(
                        "confirmatory registry artifact transition has no binding object"
                    )
                declared_pointer_bindings: dict[str, str] = {}
                for path in sorted(object_paths):
                    match = _RESEARCH_OBJECT_PATH.fullmatch(path)
                    if (
                        match is not None
                        and match.group("kind") == "experiment_attempt"
                    ):
                        object_id = self._validate_confirmatory_attempt_path(
                            path,
                            commit=cursor,
                            current_head=current_head,
                            preregistration_id=registration_object_id,
                            preregistration=preregistration,
                            adaptive_freeze=adaptive_freeze,
                        )
                        prior_attempt_ids.append(object_id)
                        classification = "confirmatory_attempt"
                        attempt = self.repository.get(object_id)
                        attempt_payload = attempt.get("payload") or {}
                        result_artifacts = attempt_payload.get("result_artifacts") or {}
                        if isinstance(result_artifacts, Mapping):
                            for item in result_artifacts.values():
                                if not isinstance(item, Mapping):
                                    continue
                                pointer_path = str(item.get("pointer_path") or "")
                                object_hash = str(item.get("content_hash") or "")
                                if pointer_path and object_hash:
                                    declared_pointer_bindings[pointer_path] = (
                                        object_hash
                                    )
                    elif match is not None and match.group("kind") == "evidence":
                        object_id = self._validate_confirmatory_negative_evidence_path(
                            path,
                            commit=cursor,
                            current_head=current_head,
                            preregistration_id=registration_object_id,
                        )
                        classification = "terminal_negative_evidence"
                    else:
                        object_id, protocol_kind = (
                            self._validate_confirmatory_gate_path(
                                path,
                                commit=cursor,
                                current_head=current_head,
                                preregistration_id=registration_object_id,
                            )
                        )
                        prior_gate_decision_ids.append(object_id)
                        classification = protocol_kind
                    checked_paths.append(
                        {
                            "commit": cursor,
                            "path": path,
                            "classification": classification,
                        }
                    )
                unexpected_pointer_paths = pointer_paths - set(
                    declared_pointer_bindings
                )
                if unexpected_pointer_paths:
                    raise ResearchGitError(
                        "confirmatory attempt transition contains an unbound research "
                        "object pointer: " + ", ".join(sorted(unexpected_pointer_paths))
                    )
                committed_pointers = _pointer_records_at_commit(
                    self.repository.path,
                    cursor,
                    strict=True,
                )
                committed_pointer_hashes = {
                    str(pointer["pointer_path"]): object_hash
                    for object_hash, pointers in committed_pointers.items()
                    for pointer in pointers
                }
                mismatched_pointer_paths = {
                    path
                    for path, declared_hash in declared_pointer_bindings.items()
                    if committed_pointer_hashes.get(path) != declared_hash
                }
                if mismatched_pointer_paths:
                    raise ResearchGitError(
                        "confirmatory result pointer does not match its attempt: "
                        + ", ".join(sorted(mismatched_pointer_paths))
                    )
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "confirmatory_result_pointer",
                    }
                    for path in sorted(pointer_paths)
                )
                checked_paths.extend(
                    {
                        "commit": cursor,
                        "path": path,
                        "classification": "derived_registry_artifact",
                    }
                    for path in sorted(registry_artifact_paths)
                )
            cursor = parent
        else:
            raise ResearchGitError(
                "confirmatory experiment lineage exceeds the bounded audit limit"
            )
        if cursor != frozen_head or not registration_seen or not campaign_seen:
            raise ResearchGitError(
                "confirmatory experiment requires the exact clean Research VCS "
                "lineage frozen at preregistration"
            )

        component_fields = {
            "hypothesis": "hypothesis_hash",
            "method": "method_hash",
            "code": "code_state_hash",
            "memory": "memory_state_hash",
            "protocol": "protocol_hash",
            "evaluator": "evaluator_spec_hash",
        }
        return {
            "schema": "xscientist.confirmatory-lineage-attestation.v1",
            "status": "verified",
            "frozen_head": frozen_head,
            "lineage_head": current_head,
            "registration_commit": registration_commit,
            "registration_object_id": registration_object_id,
            "plan_object_id": plan_object_id,
            "prior_confirmatory_attempt_ids": list(reversed(prior_attempt_ids)),
            "prior_gate_decision_ids": list(reversed(prior_gate_decision_ids)),
            "checked_paths": list(reversed(checked_paths)),
            "path_policy": (
                "exact_preregistration_transition_then_new_protocol_bound_"
                "confirmatory_attempt_objects_only"
            ),
            "frozen_components": {
                component: {
                    "hash": adaptive_freeze.get(field),
                    "unchanged": True,
                    "proof": "no_post_freeze_path_can_mutate_frozen_tree",
                }
                for component, field in component_fields.items()
            },
        }

    def _prepare_experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize an attempt without mutating the repository."""

        payload = dict(attempt)
        study_phase = str(payload.get("study_phase") or "").strip().lower()
        if study_phase:
            if study_phase not in {"exploratory", "confirmatory"}:
                raise ResearchGitError(
                    f"unsupported experiment study phase: {study_phase}"
                )
            payload["study_phase"] = study_phase
        raw_status = str(payload.get("status") or "completed").strip().lower()
        payload["status"] = raw_status
        state = {
            "success": "completed",
            "completed": "completed",
            "failed": "failed",
            "error": "failed",
            "timeout": "timed_out",
            "timed_out": "timed_out",
            "cancelled": "cancelled",
            "canceled": "cancelled",
            "rejected": "rejected",
            "orphan": "failed",
            "orphaned": "failed",
        }.get(raw_status)
        if state is None:
            if raw_status == "running":
                raise ResearchGitError(
                    "running attempts are mutable execution state and cannot be "
                    "persisted as immutable Research VCS objects; record a terminal "
                    "completed, failed, timed_out, cancelled, rejected, or orphaned "
                    "receipt"
                )
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
            preregistration_id = str(registration.get("object_id") or "").strip()
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
            if payload.get("study_phase") == "confirmatory":
                registration_payload = registration.get("payload") or {}
                registration_validation = validate_preregistration(
                    registration_payload,
                    require_locked=True,
                )
                if not registration_validation["ok"]:
                    raise ResearchGitError(
                        "confirmatory experiment preregistration failed integrity "
                        "validation: " + ", ".join(registration_validation["errors"])
                    )
                adaptive_freeze = registration_payload.get("adaptive_state_freeze")
                freeze_validation = registration_validation.get("adaptive_state_freeze")
                if isinstance(adaptive_freeze, dict) and isinstance(
                    freeze_validation, dict
                ):
                    lineage_attestation = self._confirmatory_lineage_attestation(
                        registration=registration,
                        preregistration_id=preregistration_id,
                        preregistration=registration_payload,
                        adaptive_freeze=adaptive_freeze,
                    )
                    task_id = str(payload.get("task_id") or "").strip()
                    registered_task_ids = _registered_task_ids(registration_payload)
                    if len(registered_task_ids) > 1 and not task_id:
                        raise ResearchGitError(
                            "multi-task confirmatory experiment requires a locked "
                            "task id"
                        )
                    if task_id and task_id not in _registered_task_ids(
                        registration_payload
                    ):
                        raise ResearchGitError(
                            "confirmatory experiment task is not in the locked "
                            "preregistration"
                        )
                    expected_bindings = {
                        "adaptive_state_hash": adaptive_freeze.get("state_hash"),
                        "research_state_hash": adaptive_freeze.get(
                            "research_state_hash"
                        ),
                        "post_freeze_adaptation": False,
                        "preregistration_id": registration_payload.get(
                            "preregistration_id"
                        ),
                    }
                    data_policy = registration_payload.get("data_policy")
                    data_policy = (
                        data_policy if isinstance(data_policy, Mapping) else {}
                    )
                    for field in ("data_manifest_hash", "data_snapshot_id"):
                        expected_bindings[field] = data_policy.get(field)
                    if task_id:
                        outcome = next(
                            (
                                item
                                for item in registration_payload.get("outcomes") or []
                                if isinstance(item, Mapping)
                                and str(item.get("task_id") or "").strip() == task_id
                            ),
                            {},
                        )
                        for field in (
                            "evidence_role",
                            "paired_control_task_id",
                            "intervention_variant",
                            "stress_condition",
                            "transformation_contract",
                            "transformation_contract_hash",
                        ):
                            expected_bindings[field] = outcome.get(field)
                        expected_bindings["protocol_fidelity_hash"] = (
                            _protocol_fidelity_hash(registration_payload, task_id)
                        )
                        expected_bindings["dataset_split_hash"] = (
                            data_policy.get("split_hashes") or {}
                        ).get(task_id)
                    campaign = registration_payload.get("confirmatory_campaign")
                    campaign = campaign if isinstance(campaign, Mapping) else {}
                    if campaign.get("schema") == "xscientist.confirmatory-campaign.v1":
                        if not str(payload.get("producer_id") or "").strip():
                            raise ResearchGitError(
                                "campaign experiment requires an explicit producer id"
                            )
                        if state == "completed":
                            if not isinstance(payload.get("configuration"), Mapping):
                                raise ResearchGitError(
                                    "completed campaign experiment requires its exact "
                                    "configuration"
                                )
                            if payload.get(
                                "configuration_hash"
                            ) != canonical_content_hash(dict(payload["configuration"])):
                                raise ResearchGitError(
                                    "completed campaign experiment configuration hash "
                                    "is invalid"
                                )
                            artifact_hashes = payload.get("result_artifact_hashes")
                            if (
                                not isinstance(artifact_hashes, Mapping)
                                or not artifact_hashes
                            ):
                                raise ResearchGitError(
                                    "completed campaign experiment requires at least one "
                                    "content-addressed result artifact"
                                )
                        elif (
                            state in {"failed", "timed_out", "cancelled", "rejected"}
                            and not str(payload.get("failure_class") or "").strip()
                        ):
                            raise ResearchGitError(
                                "unsuccessful campaign experiment requires a failure class"
                            )
                    conflicts = [
                        field
                        for field, expected in expected_bindings.items()
                        if field in payload and payload.get(field) != expected
                    ]
                    if conflicts:
                        raise ResearchGitError(
                            "confirmatory experiment conflicts with the frozen "
                            "research state: " + ", ".join(sorted(conflicts))
                        )
                    payload.update(expected_bindings)
                    payload["frozen_path_attestation"] = lineage_attestation
                    provenance_payload = dict(provenance or {})
                    frozen_code_commit = str(
                        adaptive_freeze.get("research_vcs_head") or ""
                    ).strip()
                    supplied_code_commit = str(
                        provenance_payload.get("code_commit") or ""
                    ).strip()
                    if (
                        supplied_code_commit
                        and supplied_code_commit != frozen_code_commit
                    ):
                        raise ResearchGitError(
                            "confirmatory experiment conflicts with the frozen code "
                            "state: code_commit"
                        )
                    provenance_payload["code_commit"] = frozen_code_commit
                    provenance = provenance_payload
                else:
                    raise ResearchGitError(
                        "confirmatory experiment requires a host-attested adaptive "
                        "state freeze"
                    )
            relations.append(
                {"type": "depends_on", "target": preregistration_id, "role": "protocol"}
            )
        if plan_id:
            plan = self.repository.get(plan_id)
            if plan["kind"] not in {"research_plan", "experiment_design"}:
                raise ResearchGitError("experiment plan reference has wrong kind")
            plan_id = str(plan["object_id"])
            plan_role = "design" if plan["kind"] == "experiment_design" else "plan"
            relations.append(
                {"type": "depends_on", "target": plan_id, "role": plan_role}
            )
            if (
                plan["kind"] == "experiment_design"
                and (plan.get("payload") or {}).get("protocol_kind")
                == "competitive_experiment_candidate"
            ):
                if not priority_id:
                    raise ResearchGitError(
                        "competitive experiment design requires its locked priority"
                    )
                priority = self.repository.get(priority_id)
                priority_payload = priority.get("payload") or {}
                if (
                    priority.get("kind") != "experiment_priority"
                    or priority.get("state") != "locked"
                    or priority_payload.get("selected_design_id") != plan_id
                ):
                    raise ResearchGitError(
                        "experiment priority does not select the supplied design"
                    )
                relations.append(
                    {
                        "type": "consumes",
                        "target": str(priority["object_id"]),
                        "role": "selected_priority",
                    }
                )
                payload["priority_id"] = str(priority["object_id"])
                payload["design_id"] = plan_id
        elif priority_id:
            raise ResearchGitError(
                "experiment priority requires a selected plan/design"
            )
        producer_id = str(payload.get("producer_id") or "").strip()
        producer_actor = None
        if producer_id:
            try:
                canonical_principal(producer_id, label="experiment producer_id")
            except ValueError as exc:
                raise ResearchGitError(
                    "experiment producer_id is not a valid principal"
                ) from exc
            # Bind the immutable object's actor to the producer declared in the
            # scientific payload.  Role labels remain metadata; independence is
            # evaluated on the canonical principal.
            producer_actor = {
                "actor_id": producer_id,
                "authority": "research_agent",
            }
        return {
            "payload": payload,
            "state": state,
            "relations": relations,
            "actor": producer_actor,
            "provenance": provenance,
        }

    def validate_experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the normalized attempt contract without recording any object."""

        return self._prepare_experiment_attempt(
            attempt,
            preregistration_id=preregistration_id,
            plan_id=plan_id,
            priority_id=priority_id,
            provenance=provenance,
        )

    def experiment_attempt(
        self,
        attempt: Mapping[str, Any],
        *,
        preregistration_id: str | None = None,
        plan_id: str | None = None,
        priority_id: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record every attempt, including failure, timeout, and cancellation."""

        prepared = self._prepare_experiment_attempt(
            attempt,
            preregistration_id=preregistration_id,
            plan_id=plan_id,
            priority_id=priority_id,
            provenance=provenance,
        )
        return self.record_validated_experiment_attempt(prepared, commit=commit)

    def record_validated_experiment_attempt(
        self,
        prepared: Mapping[str, Any],
        *,
        commit: bool = True,
    ) -> dict[str, Any]:
        """Record a contract returned by ``validate_experiment_attempt`` once."""

        required = {"payload", "state", "relations", "actor", "provenance"}
        if not required <= set(prepared):
            raise ResearchGitError(
                "validated experiment attempt contract is incomplete"
            )
        state = str(prepared["state"])
        result = self.repository.record(
            "experiment_attempt",
            prepared["payload"],
            state=state,
            relations=prepared["relations"],
            actor=prepared["actor"],
            provenance=prepared["provenance"],
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
        for object_id in attempt_ids:
            attempt = self.repository.get(object_id)
            if attempt["kind"] != "experiment_attempt":
                raise ResearchGitError("evidence attempt reference has wrong kind")
            relations.append(
                {"type": "derived_from", "target": str(attempt["object_id"])}
            )
        independence = None
        if verified:
            independence = require_independent_evaluator(
                self.repository,
                evaluator_id=str(verifier_id or ""),
                target_ids=attempt_ids,
                label="verified evidence",
            )
        for object_id in supports:
            self.repository.get(object_id)
            relations.append({"type": "supports", "target": object_id})
        for object_id in refutes:
            self.repository.get(object_id)
            relations.append({"type": "refutes", "target": object_id})
        evidence_payload = dict(payload)
        if independence is not None:
            evidence_payload["independence"] = independence
        result = self.repository.record(
            "evidence",
            evidence_payload,
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
        requested_verified = (
            report_payload.get("status") == "verified"
            and report_payload.get("claim_promotion_allowed") is True
            and not report_payload.get("required_failures")
        )
        relations = []
        for object_id in evaluates:
            evaluated = self.repository.get(object_id)
            relations.append(
                {"type": "evaluates", "target": str(evaluated["object_id"])}
            )
        independence = require_independent_evaluator(
            self.repository,
            evaluator_id=verifier_id,
            target_ids=evaluates,
            label="independent review",
        )
        from .research_context import record_research_context_snapshot

        # This API records an in-workspace, declared-identity review.  It is
        # useful scientific criticism, but it is not an externally signed
        # authority receipt and therefore cannot authorize claim promotion.
        # Preserve what the reviewer requested for audit while forcing the
        # effective decision to hold.
        report_payload["requested_status"] = report_payload.get("status")
        report_payload["requested_claim_promotion_allowed"] = (
            report_payload.get("claim_promotion_allowed") is True
        )
        report_payload["status"] = "completed"
        report_payload["claim_promotion_allowed"] = False
        report_payload["authority_scope"] = "local_advisory"
        report_payload["external_authority_required"] = True
        advisory_failures = list(report_payload.get("required_failures") or [])
        if requested_verified and "external_authority_missing" not in advisory_failures:
            advisory_failures.append("external_authority_missing")
        report_payload["required_failures"] = advisory_failures
        selected_option = "hold"
        context = record_research_context_snapshot(
            self.repository,
            target_ids=list(evaluates),
            decision_kind="independent_evidence_review",
            selected=selected_option,
            options_considered=[
                {
                    "option": "promote",
                    "rejected_because": (
                        "local declared-identity review has advisory scope only"
                    ),
                },
                {
                    "option": "hold",
                    "rejected_because": (""),
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
        report_payload["independence"] = independence
        review = self.repository.record(
            "review",
            report_payload,
            state="completed",
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
                "authority": "recorder",
            },
        )
        gate = self.repository.record(
            "gate_decision",
            {
                "decision": "hold",
                "claim_promotion_allowed": False,
                "required_failures": list(
                    report_payload.get("required_failures") or []
                ),
                "report_hash": report_payload.get("report_hash"),
                "context_required": True,
                "context_hash": context_payload["context_hash"],
            },
            state="rejected",
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
                and (item.get("payload") or {}).get("validation")
                and set(
                    (item.get("payload") or {}).get("evidence_ids") or []
                ).intersection(resolved_evidence_ids)
            ]
            valid_quality = [
                item
                for item in qualifications["evidence_quality"]
                if item.get("state") == "verified"
                and (item.get("payload") or {}).get("independent") is True
                and (item.get("payload") or {}).get("independence_receipt")
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
            review_target_sets: list[set[str]] = []
            for target in list(evaluated):
                try:
                    linked = self.repository.get(target)
                except ResearchGitError:
                    continue
                if linked.get("kind") == "review":
                    review_targets = {
                        str(item.get("target") or "")
                        for item in linked.get("relations") or []
                        if item.get("type") == "evaluates"
                    }
                    review_target_sets.append(review_targets)
                    evaluated.update(review_targets)
            required_closure_ids = set(resolved_evidence_ids)
            required_closure_ids.update(
                str(item["object_id"])
                for rows in qualifications.values()
                for item in rows
            )
            if not required_closure_ids.issubset(evaluated):
                raise ResearchGitError(
                    "verified claim gate does not evaluate the complete selected "
                    "scientific closure"
                )
            if not any(
                required_closure_ids.issubset(review_targets)
                for review_targets in review_target_sets
            ):
                raise ResearchGitError(
                    "verified claim requires one review covering the complete "
                    "selected scientific closure"
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
        reproduced_objects = [
            self.repository.get(object_id) for object_id in reproduces
        ]
        resolved_target_ids = sorted(
            {str(item["object_id"]) for item in reproduced_objects}
        )
        if (
            verified
            and receipt_payload.get("schema_version")
            == "xscientist.reproduction-receipt.v1"
        ):
            # Existing clients may still hand the lifecycle a locally generated
            # v1 execution receipt. Upgrade it before recording, while resolving
            # the checkpoint independently instead of trusting its copied fields.
            source_record = show_checkpoint(
                self.repository.path, str(receipt_payload.get("commit") or "")
            )
            source_checkpoint = source_record.get("checkpoint") or {}
            checkpoint_core = {
                "commit": str(source_record.get("commit") or ""),
                "checkpoint_id": str(source_checkpoint.get("checkpoint_id") or ""),
                "checkpoint_content_hash": str(
                    source_checkpoint.get("content_hash") or ""
                ),
            }
            if (
                receipt_payload.get("commit") != checkpoint_core["commit"]
                or receipt_payload.get("checkpoint_id")
                != checkpoint_core["checkpoint_id"]
                or receipt_payload.get("checkpoint_hash")
                != checkpoint_core["checkpoint_content_hash"]
            ):
                raise ResearchGitError(
                    "legacy reproduction receipt disagrees with its resolved checkpoint"
                )
            receipt_payload.update(
                {
                    "stdout_truncated": None,
                    "stderr_truncated": None,
                    "output_capture": "legacy_unknown",
                    "max_output_chars": None,
                }
            )
            execution_core = {
                key: receipt_payload.get(key)
                for key in (
                    "command_hash",
                    "reproduction_level",
                    "verdict",
                    "objects_complete",
                    "executed",
                    "returncode",
                    "timed_out",
                    "stdout_hash",
                    "stderr_hash",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                )
            }
            receipt_payload["schema_version"] = "xscientist.reproduction-receipt.v2"
            receipt_payload["checkpoint_binding"] = {
                **checkpoint_core,
                "binding_hash": content_hash(checkpoint_core),
            }
            receipt_payload["execution_result"] = {
                **execution_core,
                "result_hash": content_hash(execution_core),
            }
            # Historical v1 receipts did not persist execution-boundary
            # metadata. Preserve that uncertainty explicitly instead of
            # inventing a platform or claiming that the old run was sanitized.
            receipt_payload["execution_isolation"] = {
                "isolated": False,
                "security_boundary": False,
                "environment": "legacy_unknown",
                "environment_scope": "legacy_unknown",
                "process_tree": "legacy_unknown_no_tree_guarantee",
                "process_control": "legacy_unknown",
                "process_tree_termination_guaranteed": False,
                "filesystem": "host_visible",
                "network": "host_unrestricted",
            }
        if (
            receipt_payload.get("schema_version")
            == "xscientist.reproduction-receipt.v2"
        ):
            from .research_closure import build_reproduction_target_binding

            source_record = show_checkpoint(
                self.repository.path, str(receipt_payload.get("commit") or "")
            )
            source_checkpoint = source_record.get("checkpoint") or {}
            checkpoint_core = {
                "commit": str(source_record.get("commit") or ""),
                "checkpoint_id": str(source_checkpoint.get("checkpoint_id") or ""),
                "checkpoint_content_hash": str(
                    source_checkpoint.get("content_hash") or ""
                ),
            }
            expected_checkpoint_binding = {
                **checkpoint_core,
                "binding_hash": content_hash(checkpoint_core),
            }
            if (
                receipt_payload.get("commit") != checkpoint_core["commit"]
                or receipt_payload.get("checkpoint_id")
                != checkpoint_core["checkpoint_id"]
                or receipt_payload.get("checkpoint_hash")
                != checkpoint_core["checkpoint_content_hash"]
                or receipt_payload.get("checkpoint_binding")
                != expected_checkpoint_binding
            ):
                raise ResearchGitError(
                    "reproduction receipt checkpoint binding is inconsistent"
                )
            execution_core = {
                key: receipt_payload.get(key)
                for key in (
                    "command_hash",
                    "reproduction_level",
                    "verdict",
                    "objects_complete",
                    "executed",
                    "returncode",
                    "timed_out",
                    "stdout_hash",
                    "stderr_hash",
                    "stdout_truncated",
                    "stderr_truncated",
                    "output_capture",
                    "max_output_chars",
                )
            }
            expected_execution_result = {
                **execution_core,
                "result_hash": content_hash(execution_core),
            }
            if receipt_payload.get("execution_result") != expected_execution_result:
                raise ResearchGitError(
                    "reproduction receipt execution-result binding is inconsistent"
                )

            receipt_payload["target_binding"] = build_reproduction_target_binding(
                self.repository.path,
                resolved_target_ids,
                ref=str(receipt_payload.get("commit") or ""),
            )
            receipt_base = {
                key: value
                for key, value in receipt_payload.items()
                if key not in {"receipt_id", "content_hash"}
            }
            expected_hash = content_hash(receipt_base)
            receipt_payload["receipt_id"] = f"rr-{expected_hash.split(':', 1)[1][:16]}"
            receipt_payload["content_hash"] = expected_hash
            try:
                validate_json(receipt_payload, load_schema("reproduction_receipt"))
            except ValidationError as exc:  # pragma: no cover - contract guard
                raise ResearchGitError(
                    f"bound reproduction receipt is invalid: {exc.message}"
                ) from exc
        elif verified:  # pragma: no cover - schema currently permits only v1/v2
            raise ResearchGitError("verified reproduction requires a v2 receipt")
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
        if verified and not any(
            item.get("kind") == "claim" for item in reproduced_objects
        ):
            raise ResearchGitError(
                "verified reproduction requires an exact reproduced claim target"
            )
        relations = [
            {"type": "reproduces", "target": object_id}
            for object_id in resolved_target_ids
        ]
        independence = None
        if verified:
            independence = require_independent_evaluator(
                self.repository,
                evaluator_id=str(verifier_id or ""),
                target_ids=resolved_target_ids,
                label="verified reproduction",
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
        if independence is not None:
            payload["independence"] = independence
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
        return {
            "reproduction": result,
            "checkpoint": checkpoint,
            "receipt": receipt_payload,
        }


__all__ = ["ResearchLifecycle"]
