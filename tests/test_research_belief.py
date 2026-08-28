from __future__ import annotations

import copy
import json
import shutil

import pytest
from jsonschema import validate

from ai_scientist.protocol.hashing import content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist.research_belief import (
    BeliefContextError,
    audit_belief_context_projection,
    belief_context_issues,
    build_belief_context_projection,
)
from xscientist.research_cli import main as research_main
from xscientist.research_vcs import ResearchRepository


def _object(
    index: int,
    *,
    kind: str,
    relations: list[dict[str, str]] | None = None,
    payload: dict | None = None,
    state: str = "completed",
    actor_id: str | None = None,
    authority: str = "recorder",
    created_at: str = "2026-08-28T00:00:00+00:00",
) -> dict:
    object_id = f"rso-{index:016x}"
    core = {"object_id": object_id, "kind": kind, "index": index}
    return {
        "object_id": object_id,
        "kind": kind,
        "state": state,
        "content_hash": content_hash(core),
        "payload": payload or {},
        "relations": relations or [],
        "actor": (
            {
                "actor_id": actor_id,
                "authority": authority,
            }
            if actor_id
            else {"authority": authority}
        ),
        "created_at": created_at,
    }


def test_same_root_source_is_not_double_counted_and_projection_is_stable() -> None:
    hypothesis = _object(1, kind="hypothesis", state="locked")
    first_source = _object(
        2,
        kind="source_snapshot",
        payload={"doi": "10.1000/one", "content_hash": "sha256:" + "1" * 64},
    )
    second_source = _object(
        3,
        kind="source_snapshot",
        payload={"doi": "10.1000/two", "content_hash": "sha256:" + "2" * 64},
    )
    first_passage = _object(
        4,
        kind="passage_evidence",
        payload={"selector_hash": "sha256:" + "4" * 64},
        relations=[
            {"type": "quotes", "target": first_source["object_id"]},
            {"type": "qualified_supports", "target": hypothesis["object_id"]},
        ],
    )
    repeated_passage = _object(
        5,
        kind="passage_evidence",
        payload={"selector_hash": "sha256:" + "5" * 64},
        relations=[
            {"type": "quotes", "target": first_source["object_id"]},
            {"type": "qualified_supports", "target": hypothesis["object_id"]},
        ],
    )
    independent_passage = _object(
        6,
        kind="passage_evidence",
        payload={"selector_hash": "sha256:" + "6" * 64},
        relations=[
            {"type": "quotes", "target": second_source["object_id"]},
            {"type": "qualified_supports", "target": hypothesis["object_id"]},
        ],
    )
    objects = [
        hypothesis,
        first_source,
        second_source,
        first_passage,
        repeated_passage,
        independent_passage,
    ]

    report = build_belief_context_projection(
        objects, target_ids=[hypothesis["object_id"]]
    )
    reordered = build_belief_context_projection(
        list(reversed(objects)), target_ids=[hypothesis["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert assessment["active_support_count"] == 3
    assert assessment["independent_support_source_count"] == 2
    assert assessment["belief_state"] == "corroborated"
    assert assessment["scientific_promotion_allowed"] is False
    assert report["projection_hash"] == reordered["projection_hash"]
    assert not belief_context_issues(report)
    validate(report, load_schema("belief_context"))


def test_same_actor_with_different_authority_is_one_source_family() -> None:
    target = _object(7, kind="claim", state="completed")
    first = _object(
        8,
        kind="evidence",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="same-producer",
        authority="research_agent",
    )
    second = _object(
        9,
        kind="evidence",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="same-producer",
        authority="independent_evaluator",
    )

    report = build_belief_context_projection(
        [target, first, second], target_ids=[target["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert assessment["active_support_count"] == 2
    assert assessment["independent_support_source_count"] == 1
    assert assessment["belief_state"] == "supported"


@pytest.mark.parametrize("state", ["draft", "locked", "running", "unknown"])
def test_nonterminal_signal_states_cannot_change_belief(state: str) -> None:
    target = _object(13, kind="claim", state="completed")
    signal = _object(
        14,
        kind="evidence",
        state=state,
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="producer-a",
    )

    report = build_belief_context_projection(
        [target, signal], target_ids=[target["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert assessment["active_support_count"] == 0
    assert assessment["belief_state"] == "stale"
    assert assessment["supporting_signals"][0]["active"] is False


def test_draft_invalidation_cannot_retract_completed_evidence() -> None:
    target = _object(15, kind="claim", state="completed")
    evidence = _object(
        16,
        kind="evidence",
        state="completed",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="producer-a",
    )
    draft_update = _object(
        17,
        kind="source_update",
        state="draft",
        relations=[{"type": "invalidates", "target": evidence["object_id"]}],
        actor_id="producer-b",
    )

    report = build_belief_context_projection(
        [target, evidence, draft_update], target_ids=[target["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert assessment["active_support_count"] == 1
    assert assessment["supporting_signals"][0]["invalidated"] is False


def test_support_and_challenge_form_a_stable_unresolved_conflict() -> None:
    target = _object(10, kind="claim", state="completed")
    support = _object(
        11,
        kind="evidence",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="producer-a",
        authority="independent_evaluator",
    )
    challenge = _object(
        12,
        kind="evidence",
        relations=[{"type": "refutes", "target": target["object_id"]}],
        actor_id="producer-b",
        authority="independent_evaluator",
    )

    report = build_belief_context_projection(
        [target, challenge, support], target_ids=[target["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert assessment["belief_state"] == "contested"
    assert assessment["decision_posture"] == "investigate_conflict"
    assert report["conflict_sets"][0]["status"] == "unresolved"
    assert report["scientific_promotion_allowed"] is False


def test_expired_or_invalidated_support_cannot_remain_active() -> None:
    target = _object(20, kind="claim", state="completed")
    evidence = _object(
        21,
        kind="evidence",
        payload={"valid_until": "2026-01-01T00:00:00+00:00"},
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="study-a",
    )

    report = build_belief_context_projection(
        [target, evidence],
        target_ids=[target["object_id"]],
        as_of="2026-08-28T00:00:00+00:00",
    )

    assessment = report["target_assessments"][0]
    assert assessment["belief_state"] == "stale"
    assert assessment["active_support_count"] == 0
    assert assessment["supporting_signals"][0]["temporal_status"] == "expired"


def test_malformed_validity_metadata_fails_closed() -> None:
    target = _object(22, kind="claim", state="completed")
    evidence = _object(
        23,
        kind="evidence",
        payload={"valid_until": "not-a-timestamp"},
        relations=[{"type": "supports", "target": target["object_id"]}],
    )

    report = build_belief_context_projection(
        [target, evidence], target_ids=[target["object_id"]]
    )

    signal = report["target_assessments"][0]["supporting_signals"][0]
    assert signal["temporal_status"] == "invalid"
    assert signal["active"] is False
    assert report["target_assessments"][0]["active_support_count"] == 0


def test_unavailable_logical_time_and_failed_signal_are_not_active() -> None:
    target = _object(24, kind="claim", state="completed", created_at="")
    undated = _object(
        25,
        kind="evidence",
        payload={"valid_until": "2027-01-01T00:00:00+00:00"},
        relations=[{"type": "supports", "target": target["object_id"]}],
        created_at="",
    )
    failed = _object(
        26,
        kind="evidence",
        state="failed",
        relations=[{"type": "supports", "target": target["object_id"]}],
        created_at="",
    )

    report = build_belief_context_projection(
        [target, undated, failed], target_ids=[target["object_id"]]
    )

    assessment = report["target_assessments"][0]
    assert report["as_of_source"] == "unavailable"
    assert assessment["active_support_count"] == 0
    assert {row["active"] for row in assessment["supporting_signals"]} == {False}
    assert assessment["belief_state"] == "stale"


def test_explicit_as_of_excludes_future_evidence_and_future_invalidation() -> None:
    target = _object(27, kind="claim", created_at="2025-01-01T00:00:00+00:00")
    evidence = _object(
        28,
        kind="evidence",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="study-a",
        created_at="2025-02-01T00:00:00+00:00",
    )
    future_evidence = _object(
        29,
        kind="evidence",
        relations=[{"type": "supports", "target": target["object_id"]}],
        actor_id="study-b",
        created_at="2026-01-01T00:00:00+00:00",
    )
    future_invalidator = _object(
        34,
        kind="source_update",
        relations=[{"type": "invalidates", "target": evidence["object_id"]}],
        created_at="2026-02-01T00:00:00+00:00",
    )

    historical = build_belief_context_projection(
        [target, evidence, future_evidence, future_invalidator],
        target_ids=[target["object_id"]],
        as_of="2025-06-01T00:00:00+00:00",
    )
    historical_assessment = historical["target_assessments"][0]
    assert historical_assessment["active_support_count"] == 1
    assert historical_assessment["belief_state"] == "supported"
    future_signal = next(
        row
        for row in historical_assessment["supporting_signals"]
        if row["object_id"] == future_evidence["object_id"]
    )
    assert future_signal["temporal_status"] == "not_yet_observed"
    assert future_signal["active"] is False

    current = build_belief_context_projection(
        [target, evidence, future_evidence, future_invalidator],
        target_ids=[target["object_id"]],
        as_of="2027-01-01T00:00:00+00:00",
    )
    current_evidence = next(
        row
        for row in current["target_assessments"][0]["supporting_signals"]
        if row["object_id"] == evidence["object_id"]
    )
    assert current_evidence["invalidated"] is True
    assert current_evidence["active"] is False


def test_lineage_cycle_and_hard_limits_fail_closed() -> None:
    first = _object(30, kind="hypothesis", state="locked")
    second = _object(31, kind="evidence")
    first["relations"] = [{"type": "depends_on", "target": second["object_id"]}]
    second["relations"] = [
        {"type": "depends_on", "target": first["object_id"]},
        {"type": "supports", "target": first["object_id"]},
    ]

    report = build_belief_context_projection(
        [first, second], target_ids=[first["object_id"]]
    )
    assert report["complete"] is False
    assert report["decision_context_usable"] is False
    assert "lineage_cycle_detected" in report["blockers"]
    cycle_audit = audit_belief_context_projection(report)
    assert cycle_audit["verification_allowed"] is False
    assert "projection_incomplete" in cycle_audit["issues"]

    with pytest.raises(BeliefContextError, match="hard maximum"):
        build_belief_context_projection(
            [first], target_ids=[first["object_id"]], max_nodes=1025
        )

    bounded_target = _object(32, kind="claim")
    bounded_source = _object(
        33,
        kind="evidence",
        relations=[
            {"type": "supports", "target": bounded_target["object_id"]},
            {"type": "supports", "target": bounded_target["object_id"]},
        ],
    )
    bounded = build_belief_context_projection(
        [bounded_target, bounded_source],
        target_ids=[bounded_target["object_id"]],
        max_relations=1,
    )
    assert bounded["complete"] is False
    assert bounded["graph_audit"]["observed_relation_count"] == 1
    assert "relation_limit_exceeded" in bounded["blockers"]

    deduplicated = build_belief_context_projection(
        [bounded_target, bounded_source],
        target_ids=[bounded_target["object_id"]],
    )
    assert deduplicated["target_assessments"][0]["active_support_count"] == 1

    invalid_target = copy.deepcopy(bounded_target)
    invalid_target["content_hash"] = "invalid"
    invalid_report = build_belief_context_projection(
        [invalid_target], target_ids=[invalid_target["object_id"]]
    )
    assert invalid_report["complete"] is False
    assert invalid_report["target_assessments"] == []
    assert "invalid_content_hash" in invalid_report["blockers"]
    validate(invalid_report, load_schema("belief_context"))


def test_audit_is_payload_free_and_rejects_tampering() -> None:
    target = _object(40, kind="hypothesis", state="locked")
    report = build_belief_context_projection([target], target_ids=[target["object_id"]])
    audit = audit_belief_context_projection(report)
    assert audit["verification_allowed"] is True
    validate(audit, load_schema("belief_context_audit"))

    tampered = copy.deepcopy(report)
    tampered["scientific_promotion_allowed"] = True
    blocked = audit_belief_context_projection(tampered)
    assert blocked["verification_allowed"] is False
    assert "promotion_authority_escalated" in blocked["issues"]
    assert "hypothesis" not in str(blocked)

    malformed = copy.deepcopy(report)
    malformed["target_assessments"][0]["target_state"] = "x" * 129
    malformed_core = {
        key: value for key, value in malformed.items() if key != "projection_hash"
    }
    malformed["projection_hash"] = content_hash(malformed_core)
    malformed_audit = audit_belief_context_projection(malformed)
    assert malformed_audit["verification_allowed"] is False
    assert "schema_invalid" in malformed_audit["issues"]

    bad_time = copy.deepcopy(report)
    bad_time["as_of"] = "not-a-timestamp"
    bad_time["as_of_source"] = "explicit"
    bad_time_core = {
        key: value for key, value in bad_time.items() if key != "projection_hash"
    }
    bad_time["projection_hash"] = content_hash(bad_time_core)
    bad_time_audit = audit_belief_context_projection(bad_time)
    assert bad_time_audit["verification_allowed"] is False
    assert "as_of_invalid" in bad_time_audit["issues"]

    supported_target = _object(41, kind="claim", state="completed")
    supporting_evidence = _object(
        42,
        kind="evidence",
        state="completed",
        relations=[{"type": "supports", "target": supported_target["object_id"]}],
        actor_id="producer-a",
    )
    impossible = build_belief_context_projection(
        [supported_target, supporting_evidence],
        target_ids=[supported_target["object_id"]],
    )
    signal = impossible["target_assessments"][0]["supporting_signals"][0]
    signal["active"] = False
    signal["invalidated"] = True
    signal["temporal_status"] = "expired"
    signal["valid_until"] = "2026-01-01T00:00:00+00:00"
    impossible_core = {
        key: value for key, value in impossible.items() if key != "projection_hash"
    }
    impossible["projection_hash"] = content_hash(impossible_core)
    impossible_audit = audit_belief_context_projection(impossible)
    assert impossible_audit["verification_allowed"] is False
    assert "assessment_counts_inconsistent" in impossible_audit["issues"]

    malformed_temporal = build_belief_context_projection(
        [supported_target, supporting_evidence],
        target_ids=[supported_target["object_id"]],
    )
    malformed_temporal_signal = malformed_temporal["target_assessments"][0][
        "supporting_signals"
    ][0]
    malformed_temporal_signal["temporal_status"] = "current"
    malformed_temporal_signal["valid_until"] = "garbage"
    malformed_temporal_core = {
        key: value
        for key, value in malformed_temporal.items()
        if key != "projection_hash"
    }
    malformed_temporal["projection_hash"] = content_hash(malformed_temporal_core)
    malformed_temporal_audit = audit_belief_context_projection(malformed_temporal)
    assert malformed_temporal_audit["verification_allowed"] is False
    assert "temporal_metadata_inconsistent" in malformed_temporal_audit["issues"]

    for field, value in (
        ("lineage_cycle_detected", True),
        ("truncated", True),
        ("observed_node_count", 0),
    ):
        impossible_graph = build_belief_context_projection(
            [supported_target, supporting_evidence],
            target_ids=[supported_target["object_id"]],
        )
        impossible_graph["graph_audit"][field] = value
        impossible_graph_core = {
            key: item
            for key, item in impossible_graph.items()
            if key != "projection_hash"
        }
        impossible_graph["projection_hash"] = content_hash(impossible_graph_core)
        impossible_graph_audit = audit_belief_context_projection(impossible_graph)
        assert impossible_graph_audit["verification_allowed"] is False
        assert "graph_audit_inconsistent" in impossible_graph_audit["issues"]

    impossible_limits = build_belief_context_projection(
        [supported_target, supporting_evidence],
        target_ids=[supported_target["object_id"]],
    )
    impossible_limits["limits"]["max_nodes"] = 1
    impossible_limits_core = {
        key: value
        for key, value in impossible_limits.items()
        if key != "projection_hash"
    }
    impossible_limits["projection_hash"] = content_hash(impossible_limits_core)
    impossible_limits_audit = audit_belief_context_projection(impossible_limits)
    assert impossible_limits_audit["verification_allowed"] is False
    assert "graph_audit_inconsistent" in impossible_limits_audit["issues"]

    duplicate_signal = build_belief_context_projection(
        [supported_target, supporting_evidence],
        target_ids=[supported_target["object_id"]],
    )
    duplicate_assessment = duplicate_signal["target_assessments"][0]
    duplicate_assessment["supporting_signals"].append(
        copy.deepcopy(duplicate_assessment["supporting_signals"][0])
    )
    duplicate_assessment["active_support_count"] = 2
    duplicate_signal_core = {
        key: value
        for key, value in duplicate_signal.items()
        if key != "projection_hash"
    }
    duplicate_signal["projection_hash"] = content_hash(duplicate_signal_core)
    duplicate_signal_audit = audit_belief_context_projection(duplicate_signal)
    assert duplicate_signal_audit["verification_allowed"] is False
    assert "duplicate_assessment_signal" in duplicate_signal_audit["issues"]


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")
def test_belief_cli_builds_and_audits_projection(tmp_path, capsys) -> None:
    repository = ResearchRepository.init(
        tmp_path / "research",
        name="belief-cli",
        question="# Belief CLI\n",
        git_user_name="Belief Test",
        git_user_email="belief@example.invalid",
    )
    target = repository.record("hypothesis", {"statement": "H", "falsifier": "not H"})

    status = research_main(
        [
            "belief",
            target.object_id,
            "--repo",
            str(repository.path),
            "--json",
        ]
    )
    assert status == 0
    projection = json.loads(capsys.readouterr().out)
    assert projection["target_ids"] == [target.object_id]
    assert projection["scientific_promotion_allowed"] is False

    report_path = tmp_path / "belief.json"
    report_path.write_text(json.dumps(projection), encoding="utf-8")
    audit_status = research_main(["belief-audit", str(report_path), "--json"])
    assert audit_status == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["verification_allowed"] is True
    assert audit["payloads_disclosed"] is False
