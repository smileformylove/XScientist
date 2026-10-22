"""Record actual VLM video inputs and expose bounded execution feedback."""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.llm_trace import capture_llm_calls
from ai_scientist.protocol.video import video_payload_issues

from .research_commands import _ensure_direct_save_is_safe
from .research_git import ResearchGitError
from .research_vcs import ResearchRepository
from .video_evidence import (
    MAX_FRAME_BYTES,
    _VideoArtifacts,
    _asset,
    _frame_key,
    _frame_limit,
    _save,
    _video_evidence,
    _video_objects,
)

VIDEO_INSPECTION_PROTOCOL = "xscientist.video-inspection.v1"


def _sent_pack(content: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the structured manifest against the image bytes in the actual request."""
    try:
        pack = json.loads(content[1]["text"])
        if pack["protocol_kind"] != "xscientist.video-evidence-pack.v1" or pack[
            "pack_hash"
        ] != canonical_content_hash(
            {key: value for key, value in pack.items() if key != "pack_hash"}
        ):
            raise ValueError("invalid pack commitment")
        unique = {}
        for item in pack["items"]:
            for frame in item["frames"]:
                unique.setdefault(_frame_key(item, frame), frame["artifact_hash"])
        images = [
            part["image_url"]["url"] for part in content if part["type"] == "image_url"
        ]
        if not 1 <= len(images) == len(unique) == pack["frame_count"] <= 16:
            raise ValueError("image count mismatch")
        for url, expected in zip(images, unique.values()):
            prefix = "data:image/png;base64,"
            if not url.startswith(prefix) or len(url) > len(prefix) + 4 * (
                (MAX_FRAME_BYTES + 2) // 3
            ):
                raise ValueError("invalid image input")
            raw = base64.b64decode(url[len(prefix) :], validate=True)
            if "sha256:" + hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("image commitment mismatch")
        return pack
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ResearchGitError(
            "VLM request has no attributable video input manifest"
        ) from exc


def inspect_video_evidence(
    repo: str | Path,
    evidence_ids: Sequence[str],
    *,
    message: str,
    client: Any,
    model: str,
    system_message: str = "Inspect the supplied video frames. Cite evidence IDs and actual timestamps. Separate direct observations from inference and flag missing temporal or audio evidence.",
    max_frames: int = 8,
    frame_refs: Sequence[Mapping[str, Any]] | None = None,
    budget_tokens: int = 4000,
    commit: bool = True,
    _preparation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the existing provider boundary, then record a digest-only input receipt."""
    from ai_scientist.vlm import get_response_from_vlm

    _frame_limit(max_frames)
    if isinstance(evidence_ids, str) or not evidence_ids or len(evidence_ids) > 32:
        raise ResearchGitError("provide between 1 and 32 video evidence IDs")
    if not isinstance(message, str) or not message.strip() or len(message) > 10000:
        raise ResearchGitError("inspection message must contain 1-10000 characters")
    if (
        not isinstance(system_message, str)
        or not system_message.strip()
        or len(system_message) > 10000
    ):
        raise ResearchGitError(
            "inspection system message must contain 1-10000 characters"
        )
    if not isinstance(model, str) or not model.strip() or len(model) > 256:
        raise ResearchGitError("inspection model must contain 1-256 characters")
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    started = time.monotonic()
    with capture_llm_calls() as provider_refs:
        response, history = get_response_from_vlm(
            message,
            [],
            client,
            model,
            system_message,
            max_images=max_frames,
            video_repo=str(repository.path),
            video_evidence_ids=list(evidence_ids),
            video_frame_refs=list(frame_refs) if frame_refs is not None else None,
            video_budget_tokens=budget_tokens,
        )
    latency_ms = int((time.monotonic() - started) * 1000)
    pack = _sent_pack(history[-2]["content"])
    items = []
    for item in pack["items"]:
        if not item["frames"]:
            continue
        evidence = repository.get(item["evidence_id"])
        _video_evidence(repository, evidence)
        items.append(
            {
                "evidence_id": item["evidence_id"],
                "evidence_hash": evidence["payload"]["evidence_hash"],
                "source_id": item["source_id"],
                "source_content_hash": item["source_content_hash"],
                "stream_index": item["selector"]["stream_index"],
                "frames": item["frames"],
            }
        )
    payload = {
        "protocol_kind": VIDEO_INSPECTION_PROTOCOL,
        "inspection_id": uuid.uuid4().hex,
        "query_hash": canonical_content_hash(message),
        "system_message_hash": canonical_content_hash(system_message),
        "request_hash": canonical_content_hash(
            {"system_message": system_message, "messages": history[:-1]}
        ),
        "response_hash": canonical_content_hash(response),
        "model": model,
        "pack_hash": pack["pack_hash"],
        "items": items,
        "frame_count": pack["frame_count"],
        "latency_ms": latency_ms,
        "provider_receipt_refs": sorted(set(provider_refs)),
        "trace_status": "captured" if provider_refs else "unavailable",
        "coverage_kind": "sparse_frames",
        "scientific_status": "unverified_model_response",
        "audio_inspected": False,
        "artifact_hashes": sorted(
            {
                digest
                for item in items
                for digest in [
                    item["source_content_hash"],
                    *(frame["artifact_hash"] for frame in item["frames"]),
                ]
            }
        ),
    }
    if _preparation is not None:
        payload["preparation"] = dict(_preparation)
    payload["context_hash"] = canonical_content_hash(payload)
    saved = _save(
        repository,
        "context_snapshot",
        payload,
        files=[],
        relations=[
            {"type": "derived_from", "target": item["evidence_id"]} for item in items
        ],
        commit=commit,
    )
    saved["response"] = response
    saved["inspection"] = payload
    return saved


def _inspection_key(item: dict[str, Any], frame: dict[str, Any]) -> tuple[Any, ...]:
    return (
        item["source_content_hash"],
        item["stream_index"],
        frame["timestamp_seconds"],
        frame["artifact_hash"],
    )


def _inspection_records(
    repository: ResearchRepository,
    *,
    asset_ids: Sequence[str] = (),
    limit: int = 128,
    artifacts: _VideoArtifacts | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ResearchGitError("video feedback limit must be between 1 and 256")
    if isinstance(asset_ids, str) or len(asset_ids) > 32:
        raise ResearchGitError("provide at most 32 asset IDs")
    selected = {_asset(repository, value)["object_id"] for value in asset_ids}
    artifacts = artifacts or _VideoArtifacts(repository)
    candidates = [
        item
        for item in _video_objects(repository, "context_snapshot")
        if item["payload"].get("protocol_kind") == VIDEO_INSPECTION_PROTOCOL
        and item["state"] == "completed"
        and (
            not selected
            or any(row["source_id"] in selected for row in item["payload"]["items"])
        )
    ]
    candidates.sort(
        key=lambda item: (item["created_at"], item["object_id"]), reverse=True
    )
    records = candidates[:limit]
    for record in records:
        payload = record["payload"]
        issues = video_payload_issues("context_snapshot", payload)
        if issues:
            raise ResearchGitError("invalid video inspection: " + "; ".join(issues))
        for item in payload["items"]:
            evidence = repository.get(item["evidence_id"])
            _video_evidence(repository, evidence)
            source = evidence["payload"]
            if (
                source["evidence_hash"] != item["evidence_hash"]
                or source["source_id"] != item["source_id"]
                or source["source_content_hash"] != item["source_content_hash"]
                or source["selector"]["stream_index"] != item["stream_index"]
                or any(frame not in source["frames"] for frame in item["frames"])
                or {"type": "derived_from", "target": item["evidence_id"]}
                not in record["relations"]
            ):
                raise ResearchGitError("video inspection evidence binding mismatch")
            artifacts.path(item["source_content_hash"])
            for frame in item["frames"]:
                artifacts.path(frame["artifact_hash"])
    return records, len(candidates) - len(records)


def video_execution_feedback(
    repo: str | Path,
    *,
    asset_ids: Sequence[str] = (),
    limit: int = 128,
) -> dict[str, Any]:
    repository = ResearchRepository(repo)
    records, omitted = _inspection_records(repository, asset_ids=asset_ids, limit=limit)
    unique_points: set[tuple[Any, ...]] = set()
    query_points: set[tuple[Any, ...]] = set()
    total = 0
    events = []
    for record in records:
        payload = record["payload"]
        points = {
            _inspection_key(item, frame)
            for item in payload["items"]
            for frame in item["frames"]
        }
        total += len(points)
        unique_points.update(points)
        query_points.update(
            (
                payload["query_hash"],
                payload["system_message_hash"],
                payload["model"],
                *point,
            )
            for point in points
        )
        events.append(
            {
                "event": "provider_response_recorded",
                "inspection_object_id": record["object_id"],
                "inspection_hash": payload["context_hash"],
                "pack_hash": payload["pack_hash"],
                "query_hash": payload["query_hash"],
                "frames_sent": payload["frame_count"],
                "latency_ms": payload["latency_ms"],
                "provider_receipt_refs": payload["provider_receipt_refs"],
            }
        )
    report = {
        "protocol_kind": "xscientist.video-feedback.v1",
        "recorded_responses": len(records),
        "omitted_receipts": omitted,
        "frames_sent": total,
        "distinct_source_points": len(unique_points),
        "repeated_same_query_points": total - len(query_points),
        "observed_latency_ms": sum(
            record["payload"]["latency_ms"] for record in records
        ),
        "preparation_receipts": sum(
            "preparation" in record["payload"] for record in records
        ),
        "decoded_frames": sum(
            record["payload"].get("preparation", {}).get("decoded_frames", 0)
            for record in records
        ),
        "cached_frames": sum(
            record["payload"].get("preparation", {}).get("cached_frames", 0)
            for record in records
        ),
        "preparation_accounting_complete": all(
            "preparation" in record["payload"] for record in records
        ),
        "events": events,
        "provider_cost_status": "not_available_in_native_receipts",
        "automatic_harness_adoption": False,
        "limitations": "Host-recorded input/response events do not establish model comprehension, continuous coverage, scientific correctness or causal improvement. Repeated submissions may be intentional. Provider traces remain in their existing ARA store.",
    }
    report["feedback_hash"] = canonical_content_hash(report)
    return report
