"""Validation shared by video producers, Research VCS and context consumers."""

from __future__ import annotations

from typing import Any, Mapping

from jsonschema import ValidationError, validate

from .canonical_json import canonical_content_hash
from .schemas import load_schema

VIDEO_PROTOCOLS = {
    "xscientist.video-asset.v1": ("source_snapshot", "video_asset", "source_hash"),
    "xscientist.video-evidence.v1": ("evidence", "video_evidence", "evidence_hash"),
    "xscientist.video-text-index.v1": (
        "source_snapshot",
        "video_text_index",
        "source_hash",
    ),
}


def video_payload_issues(kind: str, payload: Mapping[str, Any]) -> list[str]:
    protocol = payload.get("protocol_kind")
    spec = VIDEO_PROTOCOLS.get(protocol) if isinstance(protocol, str) else None
    if spec is None:
        return ["unknown video payload protocol"]
    expected_kind, schema, commitment = spec
    issues = []
    if kind != expected_kind:
        issues.append("video payload has an incompatible research object kind")
    try:
        validate(dict(payload), load_schema(schema))
    except ValidationError as exc:
        return [*issues, f"video payload schema invalid: {exc.message}"]
    try:
        expected = canonical_content_hash(
            {k: v for k, v in payload.items() if k != commitment}
        )
    except (TypeError, ValueError):
        return [*issues, "video payload requires finite canonical JSON values"]
    if payload[commitment] != expected:
        issues.append("video payload commitment mismatch")
    if schema == "video_asset":
        if payload["artifact_hashes"] != [payload["content_hash"]]:
            issues.append("video asset artifact binding mismatch")
        return issues
    if schema == "video_text_index":
        if set(payload["artifact_hashes"]) != {
            payload["content_hash"],
            payload["source_content_hash"],
        }:
            issues.append("video text index artifact binding mismatch")
        return issues
    start, end = (
        payload["selector"]["start_seconds"],
        payload["selector"]["end_seconds"],
    )
    times = [row["timestamp_seconds"] for row in payload["frames"]]
    requested = payload["sampling"]["requested_seconds"]
    if not start < end or any(t < start or t >= end for t in [*times, *requested]):
        issues.append("video evidence frames must be inside an ordered interval")
    if times != sorted(set(times)) or requested != sorted(set(requested)):
        issues.append("video evidence timestamps must be strictly increasing")
    if requested != [
        start + (end - start) * index / len(requested)
        for index in range(len(requested))
    ]:
        issues.append("video evidence sampling policy mismatch")
    if len(times) > len(requested):
        issues.append("video evidence exceeds the requested frame count")
    expected_artifacts = {
        payload["source_content_hash"],
        *(row["artifact_hash"] for row in payload["frames"]),
    }
    if set(payload["artifact_hashes"]) != expected_artifacts:
        issues.append("video evidence artifact binding mismatch")
    return issues


def video_text_segment_issues(body: Any, *, duration: float) -> list[str]:
    try:
        validate(body, load_schema("video_text_segments"))
        canonical_content_hash(body)
    except (ValidationError, TypeError, ValueError) as exc:
        return [
            f"invalid video text segments: {exc.message if isinstance(exc, ValidationError) else str(exc)}"
        ]
    segments = body["segments"]
    starts = [row["start_seconds"] for row in segments]
    issues = []
    if starts != sorted(starts):
        issues.append("video text segments must be ordered by start time")
    if any(
        not row["start_seconds"] < row["end_seconds"] <= duration
        or not row["text"].strip()
        for row in segments
    ):
        issues.append(
            "video text segments require nonempty text inside the source duration"
        )
    return issues
