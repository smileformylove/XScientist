"""Content-addressed snapshots for the final scientific evidence chain.

The snapshot intentionally contains relative paths and byte hashes only.  It
can therefore be moved with a project without leaking the local user's home
directory, and a later verifier can recompute it without trusting a cached
quality score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.utils.atomic_io import atomic_write_json

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_FILENAME = "evidence_snapshot.json"
REGISTRY_INTEGRITY_FILENAME = "experiment_registry.integrity.json"
REGISTRY_HISTORY_FILENAME = "experiment_registry.history.jsonl"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative_file(root: Path, candidate: Path) -> str | None:
    try:
        resolved = candidate.resolve()
        relative = resolved.relative_to(root)
    except (OSError, ValueError):
        return None
    return relative.as_posix()


def _iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(resolved)
    return files


def _snapshot_file_entries(root: Path) -> list[dict[str, Any]]:
    """Collect manuscript sources and scientific evidence without temp outputs."""

    patterns = [
        "latex/**/*.tex",
        "latex/**/*.bib",
        "latex/**/*.bbl",
        "latex/**/*.bst",
        "latex/**/*.sty",
        "latex/**/*.cls",
        "latex/**/*.png",
        "latex/**/*.jpg",
        "latex/**/*.jpeg",
        "latex/**/*.pdf",
        "latex/**/*.svg",
        "figures/**/*",
        "figure/**/*",
        "supplementary/**/*",
        "supplement/**/*",
        "experiment/**/*",
        "experiment_results/**/*",
        "artifacts/runs/**/*",
        "claim_evidence_graph.json",
        "experiment_registry.jsonl",
        REGISTRY_INTEGRITY_FILENAME,
        REGISTRY_HISTORY_FILENAME,
        "preregistration.json",
        "manuscript_state.json",
        "research_plan.json",
        "figure_spec.json",
    ]
    excluded_parts = {
        ".git",
        "quality",
        "__pycache__",
        ".pytest_cache",
    }
    entries: list[dict[str, Any]] = []
    for path in _iter_files(root, patterns):
        relative = _relative_file(root, path)
        if not relative or any(part in excluded_parts for part in Path(relative).parts):
            continue
        # Do not include the snapshot itself. Including it would make every
        # verification pass change the bytes it is meant to verify.
        if relative == SNAPSHOT_FILENAME or relative.endswith("/" + SNAPSHOT_FILENAME):
            continue
        try:
            digest = file_hash(path)
            size = path.stat().st_size
        except OSError:
            continue
        if relative.endswith(".tex"):
            category = "manuscript_source"
        elif relative.endswith((".bib", ".bbl", ".bst", ".sty", ".cls")):
            category = "references_and_latex_dependencies"
        elif relative.endswith((".png", ".jpg", ".jpeg", ".pdf", ".svg")):
            category = "figures_and_supplement"
        else:
            category = "scientific_evidence"
        entries.append(
            {
                "path": relative,
                "sha256": digest,
                "size": size,
                "category": category,
            }
        )
    return sorted(entries, key=lambda item: str(item["path"]))


def _manifest_hash(entries: list[dict[str, Any]]) -> str:
    return canonical_hash(entries)


def _manuscript_hash(entries: list[dict[str, Any]]) -> str:
    selected = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
        }
        for item in entries
        if item.get("category")
        in {"manuscript_source", "references_and_latex_dependencies"}
    ]
    return canonical_hash(selected)


def build_evidence_snapshot(
    project_root: str | Path,
    *,
    registry_bytes: bytes | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic snapshot of files used by a publication claim."""

    root = Path(project_root).expanduser().resolve()
    entries = _snapshot_file_entries(root)
    registry_path = root / "experiment_registry.jsonl"
    if registry_bytes is None and registry_path.is_file():
        try:
            registry_bytes = registry_path.read_bytes()
        except OSError:
            registry_bytes = None
    if records is None and registry_path.is_file():
        records = []
        try:
            for line in registry_bytes.decode("utf-8").splitlines() if registry_bytes else []:
                if line.strip():
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        records.append(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            records = None
    record_hashes = [canonical_hash(record) for record in (records or [])]
    registry_hash = (
        "sha256:" + hashlib.sha256(registry_bytes).hexdigest()
        if registry_bytes is not None
        else None
    )
    records_hash = canonical_hash(records) if records is not None else None
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "files": entries,
        "manuscript_hash": _manuscript_hash(entries),
        "registry_hash": registry_hash,
        "records_hash": records_hash,
        "record_hashes": record_hashes,
    }
    payload["snapshot_hash"] = canonical_hash(payload)
    return payload


def save_evidence_snapshot(
    project_root: str | Path,
    snapshot: dict[str, Any] | None = None,
) -> str:
    root = Path(project_root).expanduser().resolve()
    payload = snapshot or build_evidence_snapshot(root)
    atomic_write_json(root / SNAPSHOT_FILENAME, payload, indent=2, ensure_ascii=False)
    return str(root / SNAPSHOT_FILENAME)


def verify_evidence_snapshot(
    project_root: str | Path,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    if expected is None:
        try:
            expected = json.loads((root / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            expected = {}
    current = build_evidence_snapshot(root)
    expected_hash = str(expected.get("snapshot_hash") or "")
    current_hash = str(current.get("snapshot_hash") or "")
    mismatches: list[str] = []
    if expected_hash != current_hash:
        expected_files = {
            str(item.get("path")): item.get("sha256")
            for item in expected.get("files", [])
            if isinstance(item, dict)
        }
        current_files = {
            str(item.get("path")): item.get("sha256")
            for item in current.get("files", [])
            if isinstance(item, dict)
        }
        for path in sorted(set(expected_files) | set(current_files)):
            if expected_files.get(path) != current_files.get(path):
                mismatches.append(path)
        if expected.get("registry_hash") != current.get("registry_hash"):
            mismatches.append("experiment_registry.jsonl")
        if expected.get("records_hash") != current.get("records_hash"):
            mismatches.append("experiment_registry.records")
    return {
        "ok": bool(expected_hash and expected_hash == current_hash),
        "expected_hash": expected_hash or None,
        "current_hash": current_hash,
        "mismatches": sorted(set(mismatches)),
        "current": current,
    }


def build_registry_integrity(
    records: list[dict[str, Any]], *, raw_bytes: bytes | None = None
) -> dict[str, Any]:
    """Return ordered row hashes and a chain that exposes delete/reorder edits."""

    row_hashes = [canonical_hash(row) for row in records]
    previous = "GENESIS"
    chain: list[str] = []
    for row_hash in row_hashes:
        previous = canonical_hash({"previous": previous, "row": row_hash})
        chain.append(previous)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "row_count": len(records),
        "record_ids": [str(row.get("record_id") or "") for row in records],
        "row_hashes": row_hashes,
        "chain_hashes": chain,
        "chain_tip": previous,
        "records_hash": canonical_hash(records),
        "raw_hash": (
            "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
            if raw_bytes is not None
            else None
        ),
    }


def verify_registry_integrity(
    project_root: str | Path,
    records: list[dict[str, Any]],
    *,
    raw_bytes: bytes | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    path = root / REGISTRY_INTEGRITY_FILENAME
    try:
        expected = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        expected = {}
    current = build_registry_integrity(records, raw_bytes=raw_bytes)
    errors: list[str] = []
    if not expected:
        errors.append("registry_integrity_manifest_missing")
    for key in ("row_count", "record_ids", "row_hashes", "chain_hashes", "chain_tip", "records_hash", "raw_hash"):
        if expected.get(key) != current.get(key):
            errors.append(f"registry_{key}_mismatch")
    ids = current["record_ids"]
    if not ids or any(not value for value in ids):
        errors.append("registry_record_id_missing")
    if len(ids) != len(set(ids)):
        errors.append("registry_record_id_duplicate")
    return {
        "ok": not errors,
        "errors": sorted(set(errors)),
        "expected": expected,
        "current": current,
        "manifest_path": str(path),
    }
