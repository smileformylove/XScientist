from __future__ import annotations

import copy
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import pytest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.llm_trace import ENV_ACTIVE_ROOT, ENV_ENABLED, ENV_STRICT
from ai_scientist.protocol.research_vcs import research_payload_issues
from ai_scientist.utils.llm_budget import LLMBudgetManager
from xscientist.research_cli import main
from xscientist.research_git import ResearchGitError
from xscientist.video_evidence import (
    TIME_ORIGIN,
    _blob,
    build_video_evidence_pack,
    record_video_evidence,
)
from xscientist.video_inspection import inspect_video_evidence, video_execution_feedback
from xscientist.video_planning import plan_video_inspection, run_video_inspection
from tests.test_video_retrieval import import_text, study, video_file


@pytest.fixture
def provider(monkeypatch, tmp_path):
    from ai_scientist import vlm

    monkeypatch.setattr(vlm, "llm_budget_manager", LLMBudgetManager())
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-test-key")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://gateway.example/v1")
    (tmp_path / "ara").mkdir()
    monkeypatch.setenv(ENV_ACTIVE_ROOT, str(tmp_path / "ara"))
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_STRICT, "1")
    response = SimpleNamespace(
        model="glm-5.3",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="private-response-canary"),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3, total_tokens=12),
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=mock.Mock(return_value=response))
        )
    )
    return client


def evidence(repo, asset):
    return record_video_evidence(
        repo.path,
        asset,
        start_seconds=0,
        end_seconds=2,
        max_frames=4,
        result="Fixture visual observation",
    )["object"].object_id


def inspect(repo, evidence_id, provider, **kwargs):
    return inspect_video_evidence(
        repo.path,
        [evidence_id],
        message="private-question-canary",
        client=provider,
        model="openai_compat/glm-5.3",
        max_frames=2,
        **kwargs,
    )


def test_planning_is_read_only_and_tracks_decode_budget(study):
    repo, asset = study
    head = repo.status()["head"]
    with mock.patch(
        "xscientist.video_evidence._run_media_tool", side_effect=AssertionError
    ):
        plan = plan_video_inspection(
            repo.path, asset, "unseen event", max_frames=4, max_decodes=1
        )
        blocked = plan_video_inspection(
            repo.path, asset, "unseen event", max_frames=1, max_decodes=0
        )
    assert plan["actions"][0]["tool"] == "sample_interval"
    assert plan["planned_frames"] == 4
    assert plan["planned_decodes"] == 1
    assert not blocked["actions"]
    assert blocked["omitted_windows"] == 1
    assert not blocked["temporal_pair_budget_available"]
    assert plan["previously_submitted_point_count"] == 0
    assert not plan["continuous_coverage_established"]
    assert repo.status()["head"] == head


def test_retrieval_focus_can_be_sent_to_the_provider_without_other_frames(
    study, tmp_path, provider
):
    repo, asset = study
    evidence(repo, asset)
    import_text(tmp_path, repo, asset)
    plan = plan_video_inspection(repo.path, asset, "red marker", max_frames=2)
    [action] = plan["actions"]
    assert action["tool"] == "inspect_existing_evidence"
    assert action["start_seconds"] == 1
    assert [row["timestamp_seconds"] for row in action["frame_refs"]] == [1, 1.5]
    saved = inspect_video_evidence(
        repo.path,
        action["evidence_ids"],
        frame_refs=action["frame_refs"],
        message="red marker",
        client=provider,
        model="openai_compat/glm-5.3",
        max_frames=2,
    )
    [item] = saved["inspection"]["items"]
    assert [row["timestamp_seconds"] for row in item["frames"]] == [1, 1.5]
    assert saved["inspection"]["frame_count"] == 2
    request = provider.chat.completions.create.call_args.kwargs["messages"][-1][
        "content"
    ]
    assert sum(part["type"] == "image_url" for part in request) == 2
    assert repo.fsck()["ok"]


def test_inspection_receipt_is_digest_only_and_binds_provider_trace(study, provider):
    repo, asset = study
    saved = inspect(repo, evidence(repo, asset), provider)
    receipt = repo.get(saved["object"].object_id)
    serialized = json.dumps(receipt)
    assert "private-question-canary" not in serialized
    assert "private-response-canary" not in serialized
    assert "base64" not in serialized
    assert saved["response"] == "private-response-canary"
    assert receipt["payload"]["response_hash"] == canonical_content_hash(
        saved["response"]
    )
    assert receipt["payload"]["scientific_status"] == "unverified_model_response"
    assert receipt["payload"]["trace_status"] == "captured"
    assert len(receipt["payload"]["provider_receipt_refs"]) == 1
    assert not repo.objects(kind="claim")


def test_feedback_counts_real_submissions_and_keeps_unknown_cost_unknown(
    study, provider
):
    repo, asset = study
    evidence_id = evidence(repo, asset)
    for _ in range(2):
        inspect(repo, evidence_id, provider)
    feedback = video_execution_feedback(repo.path, asset_ids=[asset])
    assert feedback["recorded_responses"] == 2
    assert feedback["frames_sent"] == 4
    assert feedback["distinct_source_points"] == 2
    assert feedback["repeated_same_query_points"] == 2
    assert feedback["provider_cost_status"] == "not_available_in_native_receipts"
    assert not feedback["automatic_harness_adoption"]
    plan = plan_video_inspection(
        repo.path, asset, "private-question-canary", max_frames=2
    )
    assert plan["available_point_count"] == 4
    assert plan["previously_submitted_point_count"] == 2
    assert len(plan["prior_response_receipts"]) == 2


def test_provider_failure_never_creates_an_inspection_receipt(study, provider):
    from ai_scientist.llm import LLMResponseContractError

    repo, asset = study
    evidence_id = evidence(repo, asset)
    head = repo.status()["head"]
    provider.chat.completions.create.return_value.choices[0].finish_reason = "length"
    with pytest.raises(LLMResponseContractError):
        inspect(repo, evidence_id, provider)
    assert not repo.objects(kind="context_snapshot")
    assert repo.status()["head"] == head


def test_invalid_exact_reference_and_corrupt_frame_fail_before_provider(
    study, provider
):
    repo, asset = study
    evidence_id = evidence(repo, asset)
    payload = repo.get(evidence_id)["payload"]
    frame = payload["frames"][0]
    with pytest.raises(ResearchGitError, match="exact frame references"):
        inspect(
            repo,
            evidence_id,
            provider,
            frame_refs=[
                {
                    "evidence_id": evidence_id,
                    "timestamp_seconds": 0.125,
                    "artifact_hash": frame["artifact_hash"],
                }
            ],
        )
    provider.chat.completions.create.assert_not_called()
    path = _blob(repo, frame["artifact_hash"])
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ResearchGitError, match="hash mismatch"):
        inspect(repo, evidence_id, provider)
    provider.chat.completions.create.assert_not_called()


def test_receipt_source_binding_is_checked_beyond_its_commitment(study, provider):
    repo, asset = study
    saved = inspect(repo, evidence(repo, asset), provider)
    payload = copy.deepcopy(saved["inspection"])
    payload["items"][0]["frames"][0]["timestamp_seconds"] = 0.125
    payload["context_hash"] = canonical_content_hash(
        {key: value for key, value in payload.items() if key != "context_hash"}
    )
    assert not research_payload_issues("context_snapshot", payload)
    repo.record(
        "context_snapshot",
        payload,
        state="completed",
        relations=[
            {"type": "derived_from", "target": payload["items"][0]["evidence_id"]}
        ],
    )
    repo.commit(stage="evidence", subject="Bad inspection binding fixture")
    with pytest.raises(ResearchGitError, match="evidence binding mismatch"):
        video_execution_feedback(repo.path)


def test_exact_frame_context_does_not_include_unspecified_frames(study):
    repo, asset = study
    evidence_id = evidence(repo, asset)
    frame = repo.get(evidence_id)["payload"]["frames"][2]
    pack = build_video_evidence_pack(
        repo.path, [evidence_id], frame_refs=[{"evidence_id": evidence_id, **frame}]
    )
    assert pack["frame_count"] == 1
    assert pack["items"][0]["frames"] == [frame]
    assert pack["items"][0]["omitted_frames"] == 3


def test_bounded_run_decodes_once_then_reuses_existing_visual_inputs(study, provider):
    repo, asset = study
    first = run_video_inspection(
        repo.path,
        asset,
        "unseen event",
        client=provider,
        model="openai_compat/glm-5.3",
        max_frames=2,
        max_decodes=1,
    )
    assert first["inspection"]["preparation"]["decode_operations"] == 1
    assert first["inspection"]["preparation"]["decoded_frames"] == 2
    with mock.patch(
        "xscientist.video_evidence._extract_frames", side_effect=AssertionError
    ):
        second = run_video_inspection(
            repo.path,
            asset,
            "unseen event",
            client=provider,
            model="openai_compat/glm-5.3",
            max_frames=2,
            max_decodes=0,
        )
    assert second["inspection"]["preparation"]["decode_operations"] == 0
    assert second["inspection"]["preparation"]["existing_frame_refs"] == 2
    assert provider.chat.completions.create.call_count == 2
    feedback = video_execution_feedback(repo.path)
    assert feedback["decoded_frames"] == 2
    assert feedback["preparation_receipts"] == 2
    assert feedback["preparation_accounting_complete"]
    assert not repo.objects(kind="claim")


def test_empty_plan_does_not_call_provider_or_modify_history(study, provider):
    repo, asset = study
    head = repo.status()["head"]
    with pytest.raises(ResearchGitError, match="no video inputs fit"):
        run_video_inspection(
            repo.path,
            asset,
            "unseen event",
            client=provider,
            model="openai_compat/glm-5.3",
            max_decodes=0,
        )
    assert repo.status()["head"] == head
    provider.chat.completions.create.assert_not_called()


def test_planner_keeps_relevance_priority_when_only_one_window_fits(study, tmp_path):
    repo, asset = study
    import_text(
        tmp_path,
        repo,
        asset,
        body={
            "time_origin": TIME_ORIGIN,
            "segments": [
                {"start_seconds": 0, "end_seconds": 0.5, "text": "marker"},
                {"start_seconds": 1, "end_seconds": 1.75, "text": "red marker"},
            ],
        },
    )
    plan = plan_video_inspection(repo.path, asset, "red marker", max_frames=2)
    assert plan["actions"][0]["start_seconds"] == 1
    assert plan["omitted_windows"] == 1


def test_tight_budget_keeps_both_sides_and_passes_source_cues_to_model(study, provider):
    repo, asset = study
    hypothesis = repo.record(
        "hypothesis", {"statement": "Object moves", "falsifier": "No motion"}
    )
    repo.commit(stage="planning", subject="Motion hypothesis")
    for relation, start, end, text in (
        ("supports", 0, 0.5, "Object moves near the beginning"),
        (
            "refutes",
            1,
            2,
            "Object moves only in the annotation, not in the late frames",
        ),
    ):
        record_video_evidence(
            repo.path,
            asset,
            start_seconds=start,
            end_seconds=end,
            max_frames=1,
            result=text,
            **{relation: [hypothesis.object_id]},
        )
    result = run_video_inspection(
        repo.path,
        asset,
        "object moves",
        client=provider,
        model="openai_compat/glm-5.3",
        max_frames=2,
        max_decodes=0,
        target_id=hypothesis.object_id,
    )
    assert len(result["plan"]["actions"]) == 2
    assert not result["plan"]["temporal_pair_budget_available"]
    assert result["inspection"]["frame_count"] == 2
    message = provider.chat.completions.create.call_args.kwargs["messages"][-1][
        "content"
    ][0]["text"]
    assert "supports" in message and "refutes" in message


def test_inspection_without_provider_tracing_marks_it_unavailable(
    study, provider, monkeypatch
):
    repo, asset = study
    monkeypatch.setenv(ENV_ENABLED, "0")
    monkeypatch.setenv(ENV_STRICT, "0")
    saved = inspect(repo, evidence(repo, asset), provider)
    assert saved["inspection"]["trace_status"] == "unavailable"
    assert not saved["inspection"]["provider_receipt_refs"]


def test_cli_plan_inspect_feedback_remain_valid_json(
    study, tmp_path, provider, monkeypatch
):
    from ai_scientist import vlm

    repo, asset = study
    evidence_id = evidence(repo, asset)

    def create_client(model):
        print("Provider setup")
        return provider, model

    monkeypatch.setattr(vlm, "create_client", create_client)

    def run(arguments):
        with redirect_stdout(io.StringIO()) as output:
            assert main(["video", *arguments, "--repo", str(repo.path), "--json"]) == 0
        return json.loads(output.getvalue())

    plan = run(
        ["plan", "private-question-canary", "--asset", asset, "--max-frames", "2"]
    )
    frames = tmp_path / "frames.json"
    frames.write_text(json.dumps(plan["actions"][0]["frame_refs"]), encoding="utf-8")
    result = run(
        [
            "inspect",
            "--evidence",
            evidence_id,
            "--message",
            "private-question-canary",
            "--model",
            "openai_compat/glm-5.3",
            "--frames-file",
            str(frames),
        ]
    )
    assert result["inspection"]["frame_count"] == 2
    assert result["response"] == "private-response-canary"
    feedback = run(["feedback", "--asset", asset])
    assert feedback["recorded_responses"] == 1


@pytest.mark.parametrize(
    "change", ["frame_count", "trace_status", "artifact_hashes", "scientific_status"]
)
def test_inspection_contract_rejects_fabricated_fields(study, provider, change):
    repo, asset = study
    payload = copy.deepcopy(
        inspect(repo, evidence(repo, asset), provider)["inspection"]
    )
    if change == "frame_count":
        payload["frame_count"] = 16
    elif change == "trace_status":
        payload["trace_status"] = "unavailable"
    elif change == "artifact_hashes":
        payload["artifact_hashes"] = payload["artifact_hashes"][:2]
    else:
        payload["scientific_status"] = "verified"
    payload["context_hash"] = canonical_content_hash(
        {key: value for key, value in payload.items() if key != "context_hash"}
    )
    assert research_payload_issues("context_snapshot", payload)
