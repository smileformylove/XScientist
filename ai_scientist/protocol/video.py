"""Validation shared by video producers, Research VCS and context consumers."""

from __future__ import annotations

from typing import Any, Mapping

from jsonschema import ValidationError, validate

from .canonical_json import canonical_content_hash
from .schemas import load_schema


def video_payload_issues(kind: str, payload: Mapping[str, Any]) -> list[str]:
    is_asset = payload.get("protocol_kind") == "xscientist.video-asset.v1"
    issues = []
    if kind != ("source_snapshot" if is_asset else "evidence"):
        issues.append("video payload has an incompatible research object kind")
    try:
        validate(
            dict(payload), load_schema("video_asset" if is_asset else "video_evidence")
        )
    except ValidationError as exc:
        return [*issues, f"video payload schema invalid: {exc.message}"]
    commitment = "source_hash" if is_asset else "evidence_hash"
    try:
        expected = canonical_content_hash(
            {k: v for k, v in payload.items() if k != commitment}
        )
    except (TypeError, ValueError):
        return [*issues, "video payload requires finite canonical JSON values"]
    if payload[commitment] != expected:
        issues.append("video payload commitment mismatch")
    if is_asset:
        if payload["artifact_hashes"] != [payload["content_hash"]]:
            issues.append("video asset artifact binding mismatch")
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
