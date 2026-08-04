"""Storage inventory, reachability, and recoverable garbage collection for ARA.

The ARA protocol is append-friendly, but append-only objects still need a
lifecycle.  This module deliberately separates three concerns:

* metadata remains immutable and is scanned for content-hash references;
* object payloads live under ``objects/<algo>/...`` and may be deduplicated;
* reclamation is a two-step operation: plan first, then quarantine.

Nothing in this module silently deletes data.  ``apply_gc_plan`` only moves
objects into ``gc/quarantine``.  Permanent removal requires a separate,
explicit ``purge_quarantine`` call after a second grace period.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ai_scientist.protocol.hashing import content_hash

_HASH_RE = re.compile(r"^([a-z0-9][a-z0-9_-]*):([0-9a-f]{64})$")
_METADATA_SUFFIXES = {".json", ".jsonl"}
_SCAN_EXCLUDED_DIRS = {"objects", "gc", ".git", "__pycache__"}


class ARAStorageError(RuntimeError):
    """Raised when a storage operation would be unsafe or inconsistent."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _metadata_files(ara_root: Path) -> Iterable[Path]:
    """Yield files that may carry live object references.

    The GC workspace is excluded: a hash copied into an old plan or receipt
    must not make the object live again.  ``refs/`` is included even though
    refs are plain text, because ``refs/pins/*`` is the user-facing pinning
    mechanism.
    """

    for path in sorted(ara_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(ara_root)
        except ValueError:  # pragma: no cover - defensive
            continue
        if any(part in _SCAN_EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix in _METADATA_SUFFIXES or rel.parts[:1] == ("refs",):
            yield path


def _walk_hashes(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if _HASH_RE.fullmatch(value):
            yield value
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_hashes(nested)
        return
    if isinstance(value, list):
        for nested in value:
            yield from _walk_hashes(nested)


def collect_hash_references(ara_root: str | Path) -> dict[str, list[str]]:
    """Return ``content_hash -> metadata paths`` for one ARA.

    The scan is intentionally schema-agnostic so additive protocol fields are
    automatically safe for GC.  Only hashes that correspond to an object in
    the local store matter to collection; node/manifest hashes are harmless
    extra roots when no object exists at that address.
    """

    root = Path(ara_root).expanduser().resolve()
    refs: dict[str, set[str]] = {}
    for path in _metadata_files(root):
        rel = str(path.relative_to(root))
        values: list[Any] = []
        if path.suffix == ".json":
            payload = _load_json(path)
            if payload is not None:
                values.append(payload)
        elif path.suffix == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                lines = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    values.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        else:  # refs/* plain text
            try:
                values.append(path.read_text(encoding="utf-8").strip())
            except OSError:
                continue
        for value in values:
            for ref in _walk_hashes(value):
                refs.setdefault(ref, set()).add(rel)
    return {key: sorted(paths) for key, paths in sorted(refs.items())}


def object_inventory(ara_root: str | Path) -> dict[str, dict[str, Any]]:
    """Inventory well-formed local CAS objects without creating directories."""

    root = Path(ara_root).expanduser().resolve()
    objects_root = root / "objects"
    out: dict[str, dict[str, Any]] = {}
    if not objects_root.is_dir():
        return out
    for algo_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        algo = algo_dir.name
        for shard in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            if len(shard.name) != 2:
                continue
            for path in sorted(p for p in shard.iterdir() if p.is_file()):
                digest = shard.name + path.name
                ref = f"{algo}:{digest}"
                if not _HASH_RE.fullmatch(ref):
                    continue
                stat = path.stat()
                out[ref] = {
                    "hash": ref,
                    "path": str(path.relative_to(root)),
                    "physical_bytes": stat.st_size,
                    "mtime": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
    return out


def _logical_file_fingerprint(path: Path, *, cas_object: bool) -> tuple[str, int]:
    """Hash logical bytes, transparently decoding compressed CAS objects."""

    digest = hashlib.sha256()
    size = 0
    opener = path.open
    if cas_object:
        try:
            with path.open("rb") as raw:
                magic = raw.read(2)
        except OSError:
            magic = b""
        if magic == b"\x1f\x8b":
            opener = gzip.open
    with opener("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def storage_report(ara_root: str | Path) -> dict[str, Any]:
    """Return size, duplication, and object-reachability statistics."""

    root = Path(ara_root).expanduser().resolve()
    if not root.is_dir():
        raise ARAStorageError(f"ARA root is not a directory: {root}")

    groups: dict[str, dict[str, Any]] = {}
    by_category: dict[str, dict[str, int]] = {}
    file_count = 0
    physical_bytes = 0
    allocated_inodes: set[tuple[int, int]] = set()
    allocated_bytes = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        try:
            stat = path.stat()
            fingerprint, logical_size = _logical_file_fingerprint(
                path, cas_object=rel.parts[:1] == ("objects",)
            )
        except OSError:
            continue
        file_count += 1
        physical_bytes += stat.st_size
        inode = (stat.st_dev, stat.st_ino)
        if inode not in allocated_inodes:
            allocated_inodes.add(inode)
            allocated_bytes += stat.st_blocks * 512
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        bucket = by_category.setdefault(
            category, {"files": 0, "physical_bytes": 0, "logical_bytes": 0}
        )
        bucket["files"] += 1
        bucket["physical_bytes"] += stat.st_size
        bucket["logical_bytes"] += logical_size
        group = groups.setdefault(
            fingerprint,
            {"logical_bytes": logical_size, "copies": 0, "paths": []},
        )
        group["copies"] += 1
        group["paths"].append(str(rel))

    logical_bytes = sum(g["logical_bytes"] * g["copies"] for g in groups.values())
    unique_logical_bytes = sum(g["logical_bytes"] for g in groups.values())
    duplicate_groups = [
        {"hash": ref, **group} for ref, group in groups.items() if group["copies"] > 1
    ]
    duplicate_groups.sort(
        key=lambda row: (-(row["copies"] - 1) * row["logical_bytes"], row["hash"])
    )

    inventory = object_inventory(root)
    references = collect_hash_references(root)
    reachable = sorted(set(inventory) & set(references))
    unreachable = sorted(set(inventory) - set(references))
    return {
        "schema": "ara.storage.report.v1",
        "ara_root": str(root),
        "generated_at": _now_iso(),
        "files": file_count,
        "physical_bytes": physical_bytes,
        "allocated_bytes": allocated_bytes,
        "logical_bytes": logical_bytes,
        "unique_logical_bytes": unique_logical_bytes,
        "duplicate_logical_bytes": max(0, logical_bytes - unique_logical_bytes),
        "duplicate_groups": duplicate_groups[:50],
        "by_category": dict(sorted(by_category.items())),
        "objects": {
            "count": len(inventory),
            "physical_bytes": sum(v["physical_bytes"] for v in inventory.values()),
            "reachable": len(reachable),
            "unreachable": len(unreachable),
            "unreachable_bytes": sum(
                inventory[ref]["physical_bytes"] for ref in unreachable
            ),
            "unreachable_hashes": unreachable,
        },
    }


def _add_tree(paths: set[Path], root: Path, relative: str) -> None:
    target = root / relative
    if target.is_file() and target.name != ".ledger.lock":
        paths.add(target)
    elif target.is_dir():
        paths.update(
            path
            for path in target.rglob("*")
            if path.is_file() and path.name != ".ledger.lock"
        )


def _object_refs_in_payload(value: Any) -> Iterable[str]:
    """Yield hashes from explicit ObjectRef-shaped dictionaries."""

    if isinstance(value, dict):
        hash_value = value.get("hash")
        if (
            isinstance(hash_value, str)
            and _HASH_RE.fullmatch(hash_value)
            and "size" in value
        ):
            yield hash_value
        for nested in value.values():
            yield from _object_refs_in_payload(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _object_refs_in_payload(nested)


def _object_refs_in_files(paths: Iterable[Path]) -> set[str]:
    refs: set[str] = set()
    for path in paths:
        if path.suffix == ".json":
            payload = _load_json(path)
            if payload is not None:
                refs.update(_object_refs_in_payload(payload))
        elif path.suffix == ".jsonl":
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                refs.update(_object_refs_in_payload(payload))
    return refs


def _metric_value(node: dict[str, Any]) -> float:
    metric = node.get("metric")
    if isinstance(metric, dict):
        try:
            return float(metric.get("value"))
        except (TypeError, ValueError):
            pass
    return float("-inf")


def bundle_selection(
    ara_root: str | Path,
    *,
    profile: str,
    node_ids: Iterable[str] | None = None,
    claim_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Select a portable, deduplicated file closure for a bundle profile."""

    root = Path(ara_root).expanduser().resolve()
    profile = str(profile or "audit").lower()
    if profile not in {"index", "fork", "reproduce", "audit"}:
        raise ARAStorageError(f"unknown bundle profile: {profile}")

    all_files = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.name.endswith(".tmp")
        and path.name != ".ledger.lock"
        and "gc" not in path.relative_to(root).parts
    }
    if profile == "audit":
        return {
            "schema": "ara.bundle.selection.v1",
            "profile": profile,
            "selected_nodes": [],
            "selected_claims": [],
            "paths": sorted(str(path.relative_to(root)) for path in all_files),
            "object_refs": [],
            "missing_object_refs": [],
        }

    selected_paths: set[Path] = set()
    for name in (
        "manifest.json",
        "manifest.lock",
        "manifest.history.jsonl",
        "exploration_graph.json",
        "README.md",
    ):
        _add_tree(selected_paths, root, name)
    _add_tree(selected_paths, root, "history")
    _add_tree(selected_paths, root, "claims")
    _add_tree(selected_paths, root, "events")
    _add_tree(selected_paths, root, "catalog")

    graph = _load_json(root / "exploration_graph.json") or {}
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    by_id = {str(node.get("id")): node for node in nodes if node.get("id") is not None}
    requested_nodes = {str(value) for value in (node_ids or []) if str(value)}
    requested_claims = {str(value) for value in (claim_ids or []) if str(value)}

    claim_nodes: set[str] = set()
    matched_claims: set[str] = set()
    claims_dir = root / "claims"
    if claims_dir.is_dir():
        for path in claims_dir.glob("*.json"):
            if path.name in {"_index.json", "coverage.json"}:
                continue
            claim = _load_json(path)
            if not isinstance(claim, dict):
                continue
            claim_id = str(claim.get("claim_id") or path.stem)
            if requested_claims and claim_id not in requested_claims:
                continue
            if not requested_claims and profile != "reproduce":
                continue
            if requested_claims:
                matched_claims.add(claim_id)
            node_id = str(claim.get("node_id") or "").strip()
            if node_id:
                claim_nodes.add(node_id)

    if requested_claims - matched_claims:
        missing = sorted(requested_claims - matched_claims)
        raise ARAStorageError(f"claim ids not found: {', '.join(missing)}")
    requested_nodes.update(claim_nodes)
    if requested_nodes - set(by_id):
        missing = sorted(requested_nodes - set(by_id))
        raise ARAStorageError(f"node ids not found: {', '.join(missing)}")

    if profile in {"fork", "reproduce"} and not requested_nodes:
        if profile == "reproduce" and claim_nodes:
            requested_nodes.update(claim_nodes)
        else:
            candidates = [
                node
                for node in nodes
                if not node.get("is_buggy")
                and (root / "nodes" / str(node.get("id")) / "code.py").exists()
            ]
            candidates.sort(key=_metric_value, reverse=True)
            if candidates:
                requested_nodes.add(str(candidates[0].get("id")))

    object_refs: set[str] = set()
    if profile in {"fork", "reproduce"}:
        for node_id in requested_nodes:
            _add_tree(selected_paths, root, f"nodes/{node_id}")
            node = by_id.get(node_id) or {}
            object_refs.update(
                ref
                for ref in node.get("llm_call_refs") or []
                if isinstance(ref, str) and _HASH_RE.fullmatch(ref)
            )
            object_refs.update(
                ref
                for ref in node.get("context_pack_refs") or []
                if isinstance(ref, str) and _HASH_RE.fullmatch(ref)
            )
        _add_tree(selected_paths, root, "env")
        _add_tree(selected_paths, root, "seed")
        _add_tree(selected_paths, root, "context")

    if profile == "reproduce":
        _add_tree(selected_paths, root, "verify")
        manifest = _load_json(root / "manifest.json") or {}
        pipeline_entries = (
            (manifest.get("references") or {}).get("pipeline_artifacts") or []
            if isinstance(manifest, dict)
            else []
        )
        reproduce_kinds = {
            "claim_evidence_graph",
            "experiment_registry",
            "idea_cards",
            "research_plan",
            "stage_standards",
            "process_alignment",
        }
        for entry in pipeline_entries:
            if not isinstance(entry, dict) or entry.get("kind") not in reproduce_kinds:
                continue
            relative = entry.get("path")
            if isinstance(relative, str):
                _add_tree(selected_paths, root, relative)
            ref = entry.get("content_hash")
            if isinstance(ref, str) and _HASH_RE.fullmatch(ref):
                object_refs.add(ref)

    object_refs.update(_object_refs_in_files(selected_paths))
    inventory = object_inventory(root)
    missing_object_refs = sorted(ref for ref in object_refs if ref not in inventory)
    for ref in sorted(object_refs & set(inventory)):
        selected_paths.add(root / inventory[ref]["path"])

    return {
        "schema": "ara.bundle.selection.v1",
        "profile": profile,
        "selected_nodes": sorted(requested_nodes),
        "selected_claims": sorted(matched_claims),
        "paths": sorted(str(path.relative_to(root)) for path in selected_paths),
        "object_refs": sorted(object_refs),
        "missing_object_refs": missing_object_refs,
    }


def hydrate_objects(
    ara_root: str | Path,
    *,
    hashes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Restore missing local CAS views from the project's shared store."""

    root = Path(ara_root).expanduser().resolve()
    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, dict) or not manifest.get("project_dir"):
        raise ARAStorageError("manifest.project_dir is required to locate .ara-store")
    project_dir = Path(str(manifest["project_dir"])).expanduser().resolve()
    shared_root = project_dir / ".ara-store" / "objects"
    requested = {
        str(value)
        for value in (hashes or collect_hash_references(root))
        if isinstance(value, str) and _HASH_RE.fullmatch(value)
    }
    inventory = object_inventory(root)
    restored: list[str] = []
    already_present: list[str] = []
    unavailable: list[str] = []
    for ref in sorted(requested):
        if ref in inventory:
            already_present.append(ref)
            continue
        algo, digest = ref.split(":", 1)
        source = shared_root / algo / digest[:2] / digest[2:]
        if not source.is_file():
            unavailable.append(ref)
            continue
        target = root / "objects" / algo / digest[:2] / digest[2:]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        restored.append(ref)
    return {
        "schema": "ara.hydrate.v1",
        "ara_root": str(root),
        "shared_root": str(shared_root),
        "restored": restored,
        "already_present": already_present,
        "unavailable": unavailable,
        "complete": not unavailable,
    }


def _roots_hash(references: dict[str, list[str]]) -> str:
    return content_hash({"references": sorted(references)})


def create_gc_plan(
    ara_root: str | Path,
    *,
    grace_seconds: int = 0,
    write: bool = True,
) -> dict[str, Any]:
    """Plan reclamation of currently-unreferenced CAS objects.

    ``grace_seconds`` is evaluated against the object's modification time.
    Recent unreferenced objects are reported as deferred, never selected.
    """

    root = Path(ara_root).expanduser().resolve()
    inventory = object_inventory(root)
    references = collect_hash_references(root)
    now = datetime.now(timezone.utc)
    candidates: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for ref in sorted(set(inventory) - set(references)):
        item = dict(inventory[ref])
        try:
            modified = datetime.fromisoformat(item["mtime"])
            age_seconds = max(0, int((now - modified).total_seconds()))
        except (TypeError, ValueError):
            age_seconds = 0
        item["age_seconds"] = age_seconds
        if age_seconds >= max(0, int(grace_seconds)):
            candidates.append(item)
        else:
            deferred.append(item)

    base = {
        "schema": "ara.gc.plan.v1",
        "ara_root": str(root),
        "generated_at": _now_iso(),
        "grace_seconds": max(0, int(grace_seconds)),
        "roots_hash": _roots_hash(references),
        "candidate_count": len(candidates),
        "candidate_bytes": sum(i["physical_bytes"] for i in candidates),
        "candidates": candidates,
        "deferred": deferred,
    }
    plan_id = content_hash(base)
    plan = {**base, "plan_id": plan_id}
    if write:
        digest = plan_id.split(":", 1)[1]
        plan_path = root / "gc" / "plans" / f"{digest}.json"
        _atomic_write_json(plan_path, plan)
        plan["plan_path"] = str(plan_path)
    return plan


def _validated_plan(plan_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(plan_path).expanduser().resolve()
    plan = _load_json(path)
    if not isinstance(plan, dict) or plan.get("schema") != "ara.gc.plan.v1":
        raise ARAStorageError(f"not an ARA GC plan: {path}")
    root = Path(str(plan.get("ara_root") or "")).expanduser().resolve()
    expected_dir = root / "gc" / "plans"
    try:
        path.relative_to(expected_dir)
    except ValueError as exc:
        raise ARAStorageError("GC plan must live under <ara>/gc/plans") from exc
    if not (root / "manifest.json").exists():
        raise ARAStorageError(f"plan ARA root is no longer valid: {root}")
    return root, plan


def apply_gc_plan(plan_path: str | Path) -> dict[str, Any]:
    """Move planned objects to quarantine after revalidating all roots."""

    root, plan = _validated_plan(plan_path)
    references = collect_hash_references(root)
    if _roots_hash(references) != plan.get("roots_hash"):
        raise ARAStorageError("ARA references changed after the GC plan was created")

    inventory = object_inventory(root)
    plan_id = str(plan.get("plan_id") or "")
    digest = plan_id.split(":", 1)[-1]
    quarantine = root / "gc" / "quarantine" / digest
    moved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for candidate in plan.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        ref = str(candidate.get("hash") or "")
        current = inventory.get(ref)
        if current is None or ref in references:
            skipped.append({"hash": ref, "reason": "missing_or_now_referenced"})
            continue
        source = root / current["path"]
        try:
            source.relative_to(root / "objects")
        except ValueError as exc:  # pragma: no cover - defensive
            raise ARAStorageError(f"unsafe object path in inventory: {source}") from exc
        target = quarantine / current["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)
        moved.append(
            {
                "hash": ref,
                "source": current["path"],
                "quarantine": str(target.relative_to(root)),
                "physical_bytes": current["physical_bytes"],
            }
        )

    receipt = {
        "schema": "ara.gc.receipt.v1",
        "plan_id": plan_id,
        "ara_root": str(root),
        "applied_at": _now_iso(),
        "moved_count": len(moved),
        "moved_bytes": sum(item["physical_bytes"] for item in moved),
        "moved": moved,
        "skipped": skipped,
        "recoverable": True,
    }
    receipt_path = quarantine / "receipt.json"
    _atomic_write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def restore_quarantine(receipt_path: str | Path) -> dict[str, Any]:
    """Restore every still-quarantined object named by a GC receipt."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json(path)
    if not isinstance(receipt, dict) or receipt.get("schema") != "ara.gc.receipt.v1":
        raise ARAStorageError(f"not an ARA GC receipt: {path}")
    root = Path(str(receipt.get("ara_root") or "")).expanduser().resolve()
    quarantine_root = root / "gc" / "quarantine"
    try:
        path.relative_to(quarantine_root)
    except ValueError as exc:
        raise ARAStorageError("receipt must live under <ara>/gc/quarantine") from exc

    restored: list[str] = []
    for item in receipt.get("moved") or []:
        if not isinstance(item, dict):
            continue
        source = root / str(item.get("quarantine") or "")
        target = root / str(item.get("source") or "")
        try:
            source.relative_to(quarantine_root)
            target.relative_to(root / "objects")
        except ValueError as exc:
            raise ARAStorageError("unsafe path in GC receipt") from exc
        if not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            source.unlink()
        else:
            os.replace(source, target)
        restored.append(str(item.get("hash") or ""))

    result = {
        "schema": "ara.gc.restore.v1",
        "receipt": str(path),
        "restored_at": _now_iso(),
        "restored_count": len(restored),
        "restored": restored,
    }
    _atomic_write_json(path.parent / "restore.json", result)
    return result


def purge_quarantine(
    receipt_path: str | Path,
    *,
    grace_seconds: int,
) -> dict[str, Any]:
    """Permanently remove a quarantined plan after an explicit grace period."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json(path)
    if not isinstance(receipt, dict) or receipt.get("schema") != "ara.gc.receipt.v1":
        raise ARAStorageError(f"not an ARA GC receipt: {path}")
    root = Path(str(receipt.get("ara_root") or "")).expanduser().resolve()
    quarantine_dir = path.parent
    try:
        quarantine_dir.relative_to(root / "gc" / "quarantine")
    except ValueError as exc:
        raise ARAStorageError("receipt must live under <ara>/gc/quarantine") from exc
    try:
        applied = datetime.fromisoformat(str(receipt.get("applied_at")))
    except (TypeError, ValueError) as exc:
        raise ARAStorageError("GC receipt has no valid applied_at") from exc
    age = (datetime.now(timezone.utc) - applied).total_seconds()
    required = max(0, int(grace_seconds))
    if age < required:
        raise ARAStorageError(
            f"quarantine grace period not met: age={int(age)}s required={required}s"
        )
    moved_count = int(receipt.get("moved_count") or 0)
    moved_bytes = int(receipt.get("moved_bytes") or 0)
    plan_id = str(receipt.get("plan_id") or "")
    digest = plan_id.split(":", 1)[-1]
    receipts_dir = root / "gc" / "receipts"
    _atomic_write_json(receipts_dir / f"{digest}.quarantine.json", receipt)
    result = {
        "schema": "ara.gc.purge.v1",
        "purged_at": _now_iso(),
        "plan_id": plan_id,
        "purged_count": moved_count,
        "purged_bytes": moved_bytes,
        "recoverable": False,
    }
    purge_receipt = receipts_dir / f"{digest}.purge.json"
    _atomic_write_json(purge_receipt, result)
    shutil.rmtree(quarantine_dir)
    result["receipt_path"] = str(purge_receipt)
    return result


__all__ = [
    "ARAStorageError",
    "apply_gc_plan",
    "bundle_selection",
    "collect_hash_references",
    "create_gc_plan",
    "hydrate_objects",
    "object_inventory",
    "purge_quarantine",
    "restore_quarantine",
    "storage_report",
]
