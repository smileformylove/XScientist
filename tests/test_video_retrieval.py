from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.research_vcs import research_payload_issues
from ai_scientist.protocol.video import video_text_segment_issues
from xscientist.research_cli import main
from xscientist.research_git import ResearchGitError
from xscientist.research_vcs import ResearchRepository
from xscientist.video_evidence import (
    TIME_ORIGIN,
    _blob,
    record_video_evidence,
    register_video_asset,
)
from xscientist.video_retrieval import register_video_text_index, search_video_evidence


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    if not all(shutil.which(tool) for tool in ("git", "ffmpeg", "ffprobe")):
        pytest.skip("Git and FFmpeg are required")
    path = tmp_path_factory.mktemp("video") / "private-footage.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x24:rate=4:duration=2",
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    return path


@pytest.fixture
def study(tmp_path, video_file):
    repo = ResearchRepository.init(
        tmp_path / "study",
        question="Video retrieval",
        git_user_name="Video Test",
        git_user_email="video@example.invalid",
    )
    asset = register_video_asset(repo.path, video_file)["object"].object_id
    return repo, asset


def segments():
    return {
        "time_origin": TIME_ORIGIN,
        "segments": [
            {"start_seconds": 0, "end_seconds": 0.75, "text": "The object moves left"},
            {
                "start_seconds": 1,
                "end_seconds": 1.75,
                "text": "The object stops near a red marker",
            },
        ],
    }


def import_text(tmp_path, repo, asset, *, body=None, kind="transcript"):
    path = tmp_path / "private-transcript.json"
    path.write_text(
        json.dumps(body if body is not None else segments()), encoding="utf-8"
    )
    return register_video_text_index(
        repo.path,
        asset,
        path,
        text_kind=kind,
        producer="fixture-tool",
        producer_version="1",
    )


def test_text_index_is_immutable_deduplicated_and_retrieved_without_decoding(
    study, tmp_path
):
    repo, asset = study
    saved = import_text(tmp_path, repo, asset)
    repeated = import_text(tmp_path, repo, asset)
    assert not repeated["object"].created
    assert repeated["object"].object_id == saved["object"].object_id
    item = repo.get(saved["object"].object_id)
    assert "private-transcript" not in json.dumps(item)
    assert "The object" not in json.dumps(item)
    with mock.patch(
        "xscientist.video_evidence._run_media_tool", side_effect=AssertionError
    ):
        result = search_video_evidence(repo.path, "red marker", asset_ids=[asset])
    [hit] = result["results"]
    assert hit["source_id"] == asset
    assert hit["segment_index"] == 1
    assert hit["selector"]["start_seconds"] == 1
    assert hit["visual_inspection"] == "not_established"
    assert hit["verification_status"] == "unverified_source_text"
    assert hit["source_content_hash"] in hit["locator"]
    assert repo.fsck()["ok"]


@pytest.mark.parametrize(
    "change", ["nan", "reversed", "outside", "blank", "origin", "unordered"]
)
def test_text_contract_rejects_invalid_temporal_data(change):
    body = segments()
    if change == "nan":
        body["segments"][0]["start_seconds"] = float("nan")
    elif change == "reversed":
        body["segments"][0]["end_seconds"] = 0
    elif change == "outside":
        body["segments"][1]["end_seconds"] = 3
    elif change == "blank":
        body["segments"][0]["text"] = " "
    elif change == "origin":
        body["time_origin"] = "container_start"
    else:
        body["segments"].reverse()
    assert video_text_segment_issues(body, duration=2)


def test_temporal_filter_is_half_open_and_chinese_ocr_is_searchable(study, tmp_path):
    repo, asset = study
    body = segments()
    body["segments"][1]["text"] = "\u706b\u8f66\u505c\u8f66\u7ad9"
    import_text(tmp_path, repo, asset, body=body, kind="ocr")
    result = search_video_evidence(
        repo.path, "\u706b\u8f66", start_seconds=0.75, end_seconds=1
    )
    assert result["match_count"] == 0
    result = search_video_evidence(
        repo.path, "\u706b\u8f66", start_seconds=1, end_seconds=2
    )
    assert result["results"][0]["text_kind"] == "ocr"


def test_targeted_retrieval_keeps_support_and_refutation(study, tmp_path):
    repo, asset = study
    hypothesis = repo.record(
        "hypothesis",
        {"statement": "The object moves", "falsifier": "The object remains still"},
    )
    repo.commit(stage="planning", subject="Motion hypothesis")
    ids = []
    for relation, text in (
        ("supports", "object moves"),
        ("refutes", "object moves only in the annotation; frames show no motion"),
    ):
        saved = record_video_evidence(
            repo.path,
            asset,
            start_seconds=0,
            end_seconds=1,
            max_frames=2,
            result=text,
            **{relation: [hypothesis.object_id]},
        )
        ids.append(saved["object"].object_id)
    import_text(tmp_path, repo, asset)
    result = search_video_evidence(
        repo.path, "object moves", target_id=hypothesis.object_id, limit=2
    )
    assert {row["object_id"] for row in result["results"]} == set(ids)
    assert {rel["type"] for row in result["results"] for rel in row["relations"]} == {
        "supports",
        "refutes",
    }
    assert all(
        row["visual_inspection"] == "not_established" for row in result["results"]
    )
    with pytest.raises(ResearchGitError, match="at least two"):
        search_video_evidence(
            repo.path, "motion", target_id=hypothesis.object_id, limit=1
        )


def test_corrupt_text_cas_is_rejected(study, tmp_path):
    repo, asset = study
    saved = import_text(tmp_path, repo, asset)
    payload = repo.get(saved["object"].object_id)["payload"]
    path = _blob(repo, payload["content_hash"])
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ResearchGitError, match="hash mismatch"):
        search_video_evidence(repo.path, "object")


def test_source_mismatch_is_rejected_even_with_a_valid_index_commitment(
    study, tmp_path
):
    repo, asset = study
    saved = import_text(tmp_path, repo, asset)
    payload = copy.deepcopy(repo.get(saved["object"].object_id)["payload"])
    payload["stream_index"] = 99
    payload["source_hash"] = canonical_content_hash(
        {key: value for key, value in payload.items() if key != "source_hash"}
    )
    assert not research_payload_issues("source_snapshot", payload)
    repo.record(
        "source_snapshot",
        payload,
        state="completed",
        relations=[{"type": "derived_from", "target": asset}],
    )
    repo.commit(stage="evidence", subject="Invalid binding fixture")
    with pytest.raises(ResearchGitError, match="source binding mismatch"):
        search_video_evidence(repo.path, "object")


def test_cli_indexes_and_searches_timed_text(study, tmp_path):
    repo, asset = study
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(segments()), encoding="utf-8")

    def run(arguments):
        with redirect_stdout(io.StringIO()) as output:
            assert main(["video", *arguments, "--repo", str(repo.path), "--json"]) == 0
        return json.loads(output.getvalue())

    saved = run(
        [
            "index-text",
            str(path),
            "--asset",
            asset,
            "--kind",
            "transcript",
            "--producer",
            "fixture",
            "--producer-version",
            "1",
        ]
    )
    result = run(["search", "red marker", "--asset", asset])
    assert result["results"][0]["object_id"] == saved["object"]["object_id"]


def test_bad_or_linked_text_input_leaves_history_unchanged(study, tmp_path):
    repo, asset = study
    head = repo.status()["head"]
    path = tmp_path / "segments.json"
    path.write_text(json.dumps(segments()), encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(path)
    with pytest.raises(ResearchGitError, match="bounded regular"):
        register_video_text_index(
            repo.path,
            asset,
            linked,
            text_kind="ocr",
            producer="fixture",
            producer_version="1",
        )
    body = segments()
    body["segments"][0]["end_seconds"] = 9
    with pytest.raises(ResearchGitError, match="source duration"):
        import_text(tmp_path, repo, asset, body=body)
    assert repo.status()["head"] == head
    assert len(repo.objects(kind="source_snapshot")) == 1
