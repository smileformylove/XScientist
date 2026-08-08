"""Schema-bound ingestion of evidence produced by external research tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema

from .research_commands import _ensure_direct_save_is_safe, _finish
from .research_git import ResearchGitError
from .research_vcs import ResearchRepository

TOOL_EVIDENCE_SCHEMA = "xscientist.tool-evidence.v1"


def load_tool_evidence(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.is_file() or source.is_symlink():
        raise ResearchGitError("tool evidence input must be a regular JSON file")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchGitError(
            "tool evidence input must be one UTF-8 JSON object"
        ) from exc
    if not isinstance(payload, dict):
        raise ResearchGitError("tool evidence input must be a JSON object")
    try:
        validate_json(payload, load_schema("tool_evidence"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"tool evidence input is invalid: {exc.message}"
        ) from exc
    return payload


def ingest_tool_evidence(
    repo: str | Path,
    receipt: Mapping[str, Any],
    *,
    attempt_ids: Sequence[str],
    supports: Sequence[str] = (),
    refutes: Sequence[str] = (),
    message: str | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Record external output as completed—not independently verified—evidence."""

    try:
        validate_json(dict(receipt), load_schema("tool_evidence"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"tool evidence input is invalid: {exc.message}"
        ) from exc
    if not attempt_ids:
        raise ResearchGitError("tool evidence requires at least one experiment attempt")
    repository = ResearchRepository(repo)
    _ensure_direct_save_is_safe(repository, commit=commit)
    relations: list[dict[str, str]] = []
    for selector in attempt_ids:
        attempt_id = repository.resolve(selector, kind="experiment_attempt")
        relations.append({"type": "derived_from", "target": attempt_id})
    for selector in supports:
        target = repository.resolve(selector)
        relations.append({"type": "supports", "target": target})
    for selector in refutes:
        target = repository.resolve(selector)
        relations.append({"type": "refutes", "target": target})
    tool = dict(receipt["tool"])
    try:
        receipt_hash = canonical_content_hash(dict(receipt))
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "tool evidence must contain canonical JSON data"
        ) from exc
    payload = {
        "result": str(receipt["result"]),
        "metrics": dict(receipt.get("metrics") or {}),
        "external_tool": tool,
        "external_run_id": str(receipt["run_id"]),
        "source_receipt_hash": receipt_hash,
        "artifact_hashes": list(receipt.get("artifact_hashes") or []),
    }
    result = repository.record(
        "evidence",
        payload,
        state="completed",
        relations=relations,
        actor={"actor_id": f"tool:{tool['name']}", "authority": "recorder"},
    )
    return _finish(
        repository,
        result,
        stage="evidence",
        subject=message or f"ingest evidence from {tool['name']}",
        status="completed",
        commit=commit,
    )


__all__ = [
    "TOOL_EVIDENCE_SCHEMA",
    "ingest_tool_evidence",
    "load_tool_evidence",
]
