"""Deterministic, immutable artifacts for executable self-evolution candidates."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_gate import (
    MUTABLE_COMPONENT_SCOPES,
    build_evolution_candidate,
)

ARTIFACT_SCHEMA = "xscientist.evolution-artifact.v1"
CANDIDATE_BUILD_SCHEMA = "xscientist.evolution-candidate-build.v1"


class EvolutionArtifactError(ValueError):
    """Raised when a candidate artifact is unsafe, incomplete, or has drifted."""


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _safe_relative(value: str | Path) -> str:
    raw = str(value).replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise EvolutionArtifactError(f"unsafe artifact path: {value}")
    return path.as_posix()


def _component_prefix(component_type: str) -> str:
    prefix = MUTABLE_COMPONENT_SCOPES.get(str(component_type or "").strip())
    if prefix is None:
        raise EvolutionArtifactError(f"unsupported mutable component: {component_type}")
    return prefix.rstrip("/")


def _object_dir(store_root: str | Path, artifact_hash: str) -> Path:
    prefix, separator, digest = str(artifact_hash or "").partition(":")
    if prefix != "sha256" or not separator or len(digest) != 64:
        raise EvolutionArtifactError("artifact hash must be sha256:<64 hex>")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise EvolutionArtifactError("artifact hash is not hexadecimal") from exc
    return Path(store_root).expanduser().resolve() / "objects" / digest


def _artifact_core(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "artifact_hash"}


def _scan_component(
    source_root: Path,
    *,
    component_type: str,
    max_files: int,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    prefix = _component_prefix(component_type)
    component_root = source_root / Path(prefix)
    if not component_root.is_dir():
        raise EvolutionArtifactError(
            f"component root does not exist under source: {prefix}"
        )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(component_root.rglob("*")):
        if path.is_symlink():
            raise EvolutionArtifactError(
                f"symbolic links are forbidden in candidate artifacts: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise EvolutionArtifactError(f"special file is forbidden: {path}")
        relative = _safe_relative(path.relative_to(source_root))
        file_hash, size = _hash_file(path)
        total_bytes += size
        entries.append(
            {
                "path": relative,
                "sha256": file_hash,
                "size": size,
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
        if len(entries) > max_files:
            raise EvolutionArtifactError(
                f"artifact exceeds the file limit ({max_files})"
            )
        if total_bytes > max_total_bytes:
            raise EvolutionArtifactError(
                f"artifact exceeds the byte limit ({max_total_bytes})"
            )
    if not entries:
        raise EvolutionArtifactError("candidate component contains no files")
    return entries


def build_evolution_artifact(
    source_root: str | Path,
    *,
    component_type: str,
    store_root: str | Path,
    max_files: int = 10_000,
    max_total_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Capture one component tree in an immutable content-addressed store."""

    source = Path(source_root).expanduser().resolve()
    if source.is_symlink() or not source.is_dir():
        raise EvolutionArtifactError("artifact source root must be a real directory")
    if max_files < 1 or max_total_bytes < 1:
        raise EvolutionArtifactError("artifact limits must be positive")
    normalized_component = str(component_type or "").strip()
    prefix = _component_prefix(normalized_component)
    store = Path(store_root).expanduser().resolve()
    component_root = source / Path(prefix)
    if component_root == store or component_root in store.parents:
        raise EvolutionArtifactError(
            "artifact store may not be nested inside the captured component"
        )
    entries = _scan_component(
        source,
        component_type=normalized_component,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
    )
    manifest = {
        "schema_version": ARTIFACT_SCHEMA,
        "component_type": normalized_component,
        "logical_root": prefix,
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": sum(int(item["size"]) for item in entries),
    }
    manifest["artifact_hash"] = canonical_content_hash(manifest)
    destination = _object_dir(store, manifest["artifact_hash"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        check = verify_evolution_artifact(store, manifest["artifact_hash"])
        if not check["ok"] or check["manifest"] != manifest:
            raise EvolutionArtifactError(
                "content-addressed artifact store contains conflicting data"
            )
        return manifest
    staging = Path(tempfile.mkdtemp(prefix=".artifact-", dir=str(destination.parent)))
    try:
        files_root = staging / "files"
        for entry in entries:
            source_path = source / Path(entry["path"])
            target = files_root / Path(entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            target.chmod(0o755 if entry["executable"] else 0o644)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            os.replace(staging, destination)
        except OSError:
            if not destination.exists():
                raise
            check = verify_evolution_artifact(store, manifest["artifact_hash"])
            if not check["ok"] or check["manifest"] != manifest:
                raise EvolutionArtifactError(
                    "concurrent artifact publication produced conflicting data"
                )
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def verify_evolution_artifact(
    store_root: str | Path, artifact_hash: str
) -> dict[str, Any]:
    errors: list[str] = []
    object_dir = _object_dir(store_root, artifact_hash)
    manifest_path = object_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "ok": False,
            "errors": ["manifest_unreadable"],
            "manifest": {},
            "artifact_hash": artifact_hash,
        }
    if manifest.get("schema_version") != ARTIFACT_SCHEMA:
        errors.append("schema_version_invalid")
    if manifest.get("artifact_hash") != artifact_hash:
        errors.append("artifact_hash_binding_mismatch")
    try:
        if manifest.get("artifact_hash") != canonical_content_hash(
            _artifact_core(manifest)
        ):
            errors.append("manifest_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("manifest_not_canonicalizable")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("entries_invalid")
        entries = []
    seen: set[str] = set()
    total_bytes = 0
    for item in entries:
        try:
            relative = _safe_relative(item.get("path"))
        except (AttributeError, EvolutionArtifactError):
            errors.append("entry_path_invalid")
            continue
        if relative in seen:
            errors.append("entry_path_duplicate")
        seen.add(relative)
        path = object_dir / "files" / Path(relative)
        if path.is_symlink() or not path.is_file():
            errors.append(f"entry_missing:{relative}")
            continue
        file_hash, size = _hash_file(path)
        total_bytes += size
        if item.get("sha256") != file_hash or item.get("size") != size:
            errors.append(f"entry_hash_mismatch:{relative}")
        executable = bool(path.stat().st_mode & 0o111)
        if item.get("executable") is not executable:
            errors.append(f"entry_mode_mismatch:{relative}")
    if manifest.get("file_count") != len(entries):
        errors.append("file_count_mismatch")
    if manifest.get("total_bytes") != total_bytes:
        errors.append("total_bytes_mismatch")
    return {
        "ok": not errors,
        "errors": errors,
        "manifest": manifest,
        "artifact_hash": artifact_hash,
        "store_path": str(object_dir),
    }


def materialize_evolution_artifact(
    store_root: str | Path,
    artifact_hash: str,
    destination: str | Path,
    *,
    strip_logical_root: bool = False,
) -> dict[str, Any]:
    """Materialize a verified artifact into a new directory."""

    check = verify_evolution_artifact(store_root, artifact_hash)
    if not check["ok"]:
        raise EvolutionArtifactError(
            "artifact verification failed: " + ", ".join(check["errors"])
        )
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise EvolutionArtifactError("artifact destination already exists")
    target.mkdir(parents=True)
    manifest = check["manifest"]
    logical_root = str(manifest["logical_root"]).rstrip("/")
    try:
        for item in manifest["entries"]:
            relative = str(item["path"])
            if strip_logical_root:
                prefix = logical_root + "/"
                if not relative.startswith(prefix):
                    raise EvolutionArtifactError("artifact entry escapes logical root")
                relative = relative[len(prefix) :]
            output = target / Path(_safe_relative(relative))
            output.parent.mkdir(parents=True, exist_ok=True)
            source = Path(check["store_path"]) / "files" / Path(item["path"])
            shutil.copyfile(source, output)
            output.chmod(0o755 if item["executable"] else 0o644)
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return {
        "artifact_hash": artifact_hash,
        "destination": str(target),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
    }


def _entry_map(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["path"]): item for item in artifact["entries"]}


def _scope_covers(scope: str, path: str) -> bool:
    return path == scope or path.startswith(scope.rstrip("/") + "/")


def build_evolution_candidate_from_sources(
    *,
    constitution: dict[str, Any],
    base_root: str | Path,
    candidate_root: str | Path,
    store_root: str | Path,
    candidate_id: str,
    component_type: str,
    base_version: str,
    candidate_version: str,
    proposed_by: str,
    change_summary: str,
    change_scope: Iterable[str],
    applicability_domains: Iterable[str],
    failure_taxonomy_refs: Iterable[str],
    ablation_dimensions: Iterable[str],
    provenance_hashes: Iterable[str] = (),
    risk_tier: str = "moderate",
) -> dict[str, Any]:
    """Build both artifacts, prove the declared diff, and create the candidate."""

    base = build_evolution_artifact(
        base_root, component_type=component_type, store_root=store_root
    )
    candidate_artifact = build_evolution_artifact(
        candidate_root, component_type=component_type, store_root=store_root
    )
    if base["artifact_hash"] == candidate_artifact["artifact_hash"]:
        raise EvolutionArtifactError("candidate source is identical to the baseline")
    base_entries = _entry_map(base)
    candidate_entries = _entry_map(candidate_artifact)
    added = sorted(set(candidate_entries) - set(base_entries))
    removed = sorted(set(base_entries) - set(candidate_entries))
    modified = sorted(
        path
        for path in set(base_entries) & set(candidate_entries)
        if (
            base_entries[path]["sha256"] != candidate_entries[path]["sha256"]
            or base_entries[path]["executable"]
            is not candidate_entries[path]["executable"]
        )
    )
    changed = sorted([*added, *removed, *modified])
    prefix = _component_prefix(component_type)
    scopes = sorted({_safe_relative(item) for item in change_scope})
    if not scopes or any(
        scope != prefix and not scope.startswith(prefix + "/") for scope in scopes
    ):
        raise EvolutionArtifactError(
            f"all declared changes must remain under {prefix}/"
        )
    undeclared = [
        path
        for path in changed
        if not any(_scope_covers(scope, path) for scope in scopes)
    ]
    unused = [
        scope
        for scope in scopes
        if not any(_scope_covers(scope, path) for path in changed)
    ]
    if undeclared:
        raise EvolutionArtifactError(
            "candidate changes undeclared paths: " + ", ".join(undeclared)
        )
    if unused:
        raise EvolutionArtifactError(
            "declared change scope has no artifact delta: " + ", ".join(unused)
        )
    change_set = {
        "added": added,
        "modified": modified,
        "removed": removed,
    }
    change_set_hash = canonical_content_hash(change_set)
    candidate = build_evolution_candidate(
        constitution=constitution,
        candidate_id=candidate_id,
        component_type=component_type,
        base_version=base_version,
        candidate_version=candidate_version,
        base_artifact_hash=base["artifact_hash"],
        candidate_artifact_hash=candidate_artifact["artifact_hash"],
        rollback_ref="artifact:" + base["artifact_hash"],
        proposed_by=proposed_by,
        change_summary=change_summary,
        change_scope=scopes,
        applicability_domains=applicability_domains,
        failure_taxonomy_refs=failure_taxonomy_refs,
        ablation_dimensions=ablation_dimensions,
        provenance_hashes=sorted({*provenance_hashes, change_set_hash}),
        risk_tier=risk_tier,
    )
    return {
        "schema_version": CANDIDATE_BUILD_SCHEMA,
        "candidate": candidate,
        "base_artifact": base,
        "candidate_artifact": candidate_artifact,
        "change_set": change_set,
        "change_set_hash": change_set_hash,
    }


__all__ = [
    "ARTIFACT_SCHEMA",
    "CANDIDATE_BUILD_SCHEMA",
    "EvolutionArtifactError",
    "build_evolution_artifact",
    "build_evolution_candidate_from_sources",
    "materialize_evolution_artifact",
    "verify_evolution_artifact",
]
