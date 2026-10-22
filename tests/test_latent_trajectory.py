from __future__ import annotations

import copy
import json
import re
import tempfile
import weakref
from pathlib import Path

import pytest
from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema
from ai_scientist.protocol.hashing import content_hash
from ai_scientist.utils.ara_graph import (
    render_exploration_graph_html,
    write_exploration_graph_visualization,
)
from ai_scientist.utils.latent_trajectory import (
    audit_latent_trajectory_projection,
    build_latent_trajectory_projection,
    latent_trajectory_issues,
)

HASH = "sha256:" + "a" * 64


def _event(event_id: str = "event-1", **attributes: object) -> dict:
    return {
        "event_id": event_id,
        "event_type": "latent_step",
        "timestamp": "2026-09-20T00:00:00+00:00",
        "subject": {"type": "node", "id": "node-1"},
        "attributes": attributes,
    }


def test_projection_requires_explicit_components_and_is_fail_closed() -> None:
    projection = build_latent_trajectory_projection(
        [
            _event(attributes_text="latent_state=z0"),
            _event("event-2", status="succeeded"),
        ]
    )
    assert projection["status"] == "unavailable"
    assert projection["steps"] == []
    assert "latent_state_not_declared" in projection["blockers"]
    validate(projection, load_schema("latent_trajectory"))
    assert latent_trajectory_issues(projection) == []
    audit = audit_latent_trajectory_projection(projection)
    assert audit["verification_allowed"] is False
    assert audit["presentation_only"] is True


def test_projection_preserves_declared_latent_belief_and_transition_only() -> None:
    projection = build_latent_trajectory_projection(
        [
            _event(
                latent_state={
                    "state_id": "z0",
                    "representation_ref": HASH,
                    "dimension": 4,
                },
                belief_state={
                    "belief_id": "b0",
                    "semantics": "distribution",
                    "support_ref": "redacted",
                },
                action={"action_id": "move-left", "label": "left"},
                observation={"observation_ref": HASH, "kind": "pixel"},
                transition={
                    "from_state_id": "z0",
                    "to_state_id": "z1",
                    "operator": "f",
                },
            )
        ]
    )
    assert projection["status"] == "observed"
    assert projection["steps"][0]["components"] == [
        "latent_state",
        "belief_state",
        "action",
        "observation",
        "transition",
    ]
    assert projection["steps"][0]["transition"]["to_state_id"] == "z1"
    validate(projection, load_schema("latent_trajectory"))
    assert latent_trajectory_issues(projection) == []


def test_projection_is_bounded_deterministic_and_duplicate_safe() -> None:
    events = [
        _event(f"event-{i}", latent_state={"state_id": f"z{i}"}) for i in range(4)
    ]
    events.append(copy.deepcopy(events[-1]))
    first = build_latent_trajectory_projection(events, max_steps=4)
    second = build_latent_trajectory_projection(list(reversed(events)), max_steps=4)
    assert first["source"]["ledger_window_truncated"] is True
    assert first["integrity"]["duplicate_event_count"] == 1
    assert first["projection_hash"] != second["projection_hash"]
    assert len(first["steps"]) <= 4


def test_action_only_is_partial_and_audit_never_grants_verification() -> None:
    projection = build_latent_trajectory_projection(
        [_event(action={"action_id": "left"})]
    )
    assert projection["status"] == "partial"
    assert "latent_state_not_declared" in projection["blockers"]
    audit = audit_latent_trajectory_projection(projection)
    assert audit["verification_allowed"] is False
    assert audit["presentation_contract_valid"] is True
    validate(projection, load_schema("latent_trajectory"))


def test_hostile_boundaries_are_capped_and_source_window_is_explicit() -> None:
    malformed = [
        _event(
            f"bad-{i}",
            **{
                key: {"invalid": True}
                for key in (
                    "latent_state",
                    "belief_state",
                    "action",
                    "observation",
                    "transition",
                )
            },
        )
        for i in range(600)
    ]
    projection = build_latent_trajectory_projection(malformed, max_steps=512)
    assert projection["source"]["event_count"] == 600
    assert projection["source"]["window_event_count"] == 512
    assert projection["source"]["ledger_window_truncated"] is True
    assert projection["integrity"]["invalid_component_count"] <= 3072
    assert projection["status"] == "unavailable"
    validate(projection, load_schema("latent_trajectory"))
    assert latent_trajectory_issues(projection) == []
    with pytest.raises(ValueError):
        build_latent_trajectory_projection([], max_steps=True)


def test_text_fields_are_bounded_to_schema_limits() -> None:
    projection = build_latent_trajectory_projection(
        [
            _event(
                latent_state={"state_id": "z", "summary": "x" * 1000},
                action={"action_id": "a", "label": "x" * 1000},
            )
        ]
    )
    step = projection["steps"][0]
    assert len(step["latent_state"]["summary"]) <= 240
    assert len(step["action"]["label"]) <= 240
    validate(projection, load_schema("latent_trajectory"))


def test_streaming_source_does_not_retain_full_ledger() -> None:
    class Event(dict):
        pass

    refs = []

    def events():
        for index in range(1000):
            event = Event(_event(f"event-{index}", latent_state={"state_id": "z"}))
            refs.append(weakref.ref(event))
            if index > 20:
                assert refs[index - 20]() is None
            yield event

    projection = build_latent_trajectory_projection(events(), max_steps=8)
    assert projection["source"]["event_count"] == 1000
    assert projection["source"]["window_event_count"] == 8
    assert projection["status"] == "partial"
    assert latent_trajectory_issues(projection) == []


@pytest.mark.parametrize(
    "mutation,issue",
    [
        (
            lambda p: p["steps"][0].update(components=["action"]),
            "component_presence_inconsistent",
        ),
        (lambda p: p["source"].update(event_count=600), "source_counts_inconsistent"),
        (lambda p: p.update(status="unavailable"), "status_inconsistent"),
        (
            lambda p: p.update(scientific_promotion_allowed=True),
            "promotion_authority_escalated",
        ),
    ],
)
def test_rehashed_tampering_fails_semantic_audit(mutation, issue) -> None:
    projection = build_latent_trajectory_projection(
        [_event(latent_state={"state_id": "z"})]
    )
    mutation(projection)
    projection["projection_hash"] = content_hash(
        {key: value for key, value in projection.items() if key != "projection_hash"}
    )
    assert issue in latent_trajectory_issues(projection)
    assert (
        audit_latent_trajectory_projection(projection)["verification_allowed"] is False
    )


def test_file_projection_retains_explicit_fields_and_global_truncation(
    tmp_path,
) -> None:
    graph = {"nodes": [{"id": "root"}], "edges": []}
    (tmp_path / "events").mkdir()
    events = [
        _event(f"event-{i}", latent_state={"state_id": f"z-{i}"}) for i in range(520)
    ]
    (tmp_path / "events" / "research_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    result = write_exploration_graph_visualization(tmp_path, graph)
    page = (tmp_path / result["html"]).read_text(encoding="utf-8")
    embedded = json.loads(
        re.search(
            '<script id="graph-data" type="application/json">(.*?)</script>', page, re.S
        ).group(1)
    )
    projection = embedded["presentation_latent_trajectory"]
    assert projection["steps"][0]["latent_state"]["state_id"] == "z-8"
    assert projection["source"]["event_count"] == 520
    assert projection["source"]["window_event_count"] == 512
    assert projection["source"]["ledger_window_truncated"] is True
    assert embedded["presentation_event_count"] == 520
    assert embedded["presentation_events_truncated"] is True
    assert projection["status"] == "partial"
    assert latent_trajectory_issues(projection) == []


def test_truncated_event_suffix_cannot_rebuild_a_complete_latent_projection() -> None:
    page = render_exploration_graph_html(
        {"nodes": [], "edges": []},
        events=[_event(latent_state={"state_id": "z"})],
        event_count=600,
    )
    assert '"presentation_latent_trajectory": null' in page
    assert '"presentation_events_truncated": true' in page


def test_html_exposes_latent_panel_without_promoting_canonical_dag() -> None:
    graph = {
        "schema_version": "ara.v1",
        "protocol_kind": "exploration_graph",
        "topology_encoding": "edges",
        "nodes": [{"id": "root"}],
        "edges": [],
    }
    html = render_exploration_graph_html(
        graph,
        events=[_event(latent_state={"state_id": "z0"})],
    )
    assert "Latent dynamics / belief trajectory" in html
    assert "presentation_latent_trajectory" in html
    assert "scientific_promotion_allowed" in html
    assert "Explicit producer-declared fields only" in html


def test_write_visualization_keeps_summary_canonical_and_adds_projection_only_to_html() -> (
    None
):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        graph = {
            "schema_version": "ara.v1",
            "protocol_kind": "exploration_graph",
            "topology_encoding": "edges",
            "nodes": [{"id": "root"}],
            "edges": [],
        }
        (root / "exploration_graph.json").write_text(
            json.dumps(graph), encoding="utf-8"
        )
        (root / "events").mkdir()
        (root / "events" / "research_events.jsonl").write_text(
            json.dumps(_event(latent_state={"state_id": "z0"})) + "\n", encoding="utf-8"
        )
        refs = write_exploration_graph_visualization(root)
        html = (root / refs["html"]).read_text(encoding="utf-8")
        summary = json.loads((root / refs["summary"]).read_text(encoding="utf-8"))
        assert "Latent dynamics / belief trajectory" in html
        assert "presentation_latent_trajectory" in html
        assert "presentation_latent_trajectory" not in summary
