"""Bounded, read-only evidence inventory for benchmark reports.

The index answers *whether* the local workspace retained the major evidence
surfaces without copying or exposing their payloads.  It intentionally scans
only paths owned by the XScientist/ARA storage contract; it is not a general
filesystem crawler.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterator

EVIDENCE_INDEX_SCHEMA = "xscientist.evidence-index.v1"
_DEFAULT_MAX_FILES = 512
_DEFAULT_MAX_BYTES = 32 * 1024 * 1024
_CHUNK_SIZE = 1024 * 1024
_MAX_WALK_ENTRIES = 8192
_HARD_MAX_FILES = _MAX_WALK_ENTRIES
_HARD_MAX_BYTES = 128 * 1024 * 1024

# The labels are deliberately logical rather than relative paths.  They make
# the report useful while preventing accidental disclosure of workspace file
# names or host paths.
_CATEGORY_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "research_vcs": (
        ("research_config", "research.yaml"),
        ("typed_object_store", ".xscientist/objects"),
        ("checkpoints", "checkpoints"),
        ("research_objects", "research-objects"),
    ),
    "ara": (
        ("ara_manifests_and_events", "ara"),
        ("ara_cas_store", ".ara-store"),
    ),
    "generated_views": (
        ("logs", "04_logs"),
        ("outputs", "05_outputs"),
        ("reports", "reports"),
        ("artifacts", "artifacts"),
        ("guides", "guide"),
    ),
}


def _has_symlink_component(root: Path, relative: str) -> bool:
    """Return true when an allowlisted path crosses a symlink boundary."""

    current = root
    try:
        for part in Path(relative).parts:
            current = current / part
            if current.is_symlink():
                return True
    except (OSError, RuntimeError):
        # An inaccessible component is not safe to follow speculatively.
        return True
    return False


def _collect_regular_files(
    root: Path, relative: str, *, max_files: int
) -> tuple[list[tuple[str, Path]], bool, int, int]:
    """Collect a bounded allowlisted file prefix without following symlinks.

    The old generator bounded hashing but still allowed ``os.walk`` to visit an
    arbitrarily large directory tree.  This collector bounds directory-entry
    inspection as well as candidate files.  It returns ``(files, truncated,
    entries_seen, read_errors)``; callers can therefore disclose when counts
    describe only a prefix instead of silently presenting them as totals.
    """

    path = root / relative
    if _has_symlink_component(root, relative):
        # A symlink boundary is intentionally not followed, but it also means
        # we cannot claim that the allowlisted source was completely observed.
        return [], True, 0, 1
    try:
        if path.is_file():
            return [(relative, path)], False, 1, 0
        if not path.is_dir():
            if path.exists():
                return [], True, 1, 1
            return [], False, 0, 0
    except OSError:
        return [], False, 1, 1

    files: list[tuple[str, Path]] = []
    stack = [path]
    entries_seen = 0
    read_errors = 0
    truncated = False
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                entries: list[os.DirEntry[str]] = []
                for entry in iterator:
                    if len(entries) >= _MAX_WALK_ENTRIES:
                        truncated = True
                        break
                    entries.append(entry)
                entries.sort(key=lambda entry: entry.name)
        except OSError:
            read_errors += 1
            continue
        if truncated:
            stack.clear()
        for entry in entries:
            entries_seen += 1
            if entries_seen > _MAX_WALK_ENTRIES:
                truncated = True
                stack.clear()
                break
            try:
                if entry.is_symlink():
                    # Do not follow external files, but do not silently count
                    # a skipped link as a complete empty source either.
                    read_errors += 1
                    truncated = True
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
            except OSError:
                read_errors += 1
                continue
            if len(files) >= max_files:
                truncated = True
                stack.clear()
                break
            candidate = Path(entry.path)
            try:
                relative_name = candidate.relative_to(root).as_posix()
            except ValueError:
                continue
            files.append((relative_name, candidate))
        if truncated:
            break
    files.sort(key=lambda row: row[0])
    return files, truncated, entries_seen, read_errors


def _iter_regular_files(root: Path, relative: str) -> Iterator[tuple[str, Path]]:
    """Compatibility iterator over a bounded regular-file prefix."""

    files, _truncated, _entries_seen, _read_errors = _collect_regular_files(
        root, relative, max_files=_DEFAULT_MAX_FILES
    )
    yield from files


def _digest_file(path: Path, byte_budget: int) -> tuple[str | None, int, bool]:
    """Hash at most ``byte_budget`` bytes and report (digest, bytes, partial)."""

    digest = hashlib.sha256()
    consumed = 0
    try:
        with path.open("rb") as handle:
            while consumed < byte_budget:
                chunk = handle.read(min(_CHUNK_SIZE, byte_budget - consumed))
                if not chunk:
                    break
                digest.update(chunk)
                consumed += len(chunk)
    except (OSError, ValueError):
        return None, consumed, True
    try:
        size = path.stat().st_size
    except OSError:
        return None, consumed, True
    return f"sha256:{digest.hexdigest()}", consumed, consumed < size


def _category_index(
    root: Path,
    sources: tuple[tuple[str, str], ...],
    *,
    max_files: int,
    byte_budget: int,
) -> dict[str, Any]:
    source_labels: list[str] = []
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    traversal_truncated = False
    walk_entries = 0
    read_error_count = 0
    for label, relative in sources:
        marker = root / relative
        try:
            if marker.exists() and not _has_symlink_component(root, relative):
                source_labels.append(label)
        except OSError:
            read_error_count += 1
        remaining_files = max_files + 1 - len(candidates)
        if remaining_files <= 0:
            traversal_truncated = True
            break
        found, scan_truncated, entries_seen, scan_errors = _collect_regular_files(
            root, relative, max_files=remaining_files
        )
        traversal_truncated = traversal_truncated or scan_truncated
        walk_entries += entries_seen
        read_error_count += scan_errors
        # A permission/race error means the source total and aggregate digest
        # are only a partial observation even when the entry cap was not hit.
        if scan_errors:
            traversal_truncated = True
        for relative_name, path in found:
            if relative_name in seen:
                continue
            seen.add(relative_name)
            candidates.append((relative_name, path))
            if len(candidates) > max_files:
                traversal_truncated = True
                break
        if len(candidates) > max_files:
            break

    file_count = 0
    byte_count = 0
    truncated = False
    rows: list[tuple[str, int, str]] = []
    partial_hash = False
    for index, (relative_name, path) in enumerate(candidates):
        if index >= max_files:
            truncated = True
            break
        if byte_count >= byte_budget:
            truncated = True
            break
        file_count += 1
        try:
            size = path.stat().st_size
        except OSError:
            read_error_count += 1
            continue
        digest, consumed, partial = _digest_file(path, byte_budget - byte_count)
        byte_count += consumed
        if digest is None:
            read_error_count += 1
            continue
        rows.append((relative_name, size, digest))
        if partial:
            partial_hash = True
            truncated = True
            break

    aggregate = hashlib.sha256()
    for relative_name, size, digest in rows:
        # Relative names participate only in the digest, never in report
        # output.  This prevents two different file layouts from appearing
        # equivalent while preserving the report's redaction boundary.
        aggregate.update(relative_name.encode("utf-8", errors="replace"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")

    return {
        "present": bool(file_count),
        "file_count": file_count,
        "byte_count": byte_count,
        "digest": f"sha256:{aggregate.hexdigest()}" if rows else None,
        "digest_scope": (
            "bounded_prefix"
            if partial_hash or truncated or traversal_truncated or read_error_count
            else "observed_files"
        ),
        "sources": sorted(set(source_labels)),
        "truncated": truncated or traversal_truncated or bool(read_error_count),
        "read_error_count": read_error_count,
        "walk_entries_observed": walk_entries,
        "walk_truncated": traversal_truncated,
        "source_count_complete": not traversal_truncated
        and not bool(read_error_count)
        and not (len(candidates) > max_files),
    }


def _ara_contract_summary(
    root: Path, *, max_files: int, byte_budget: int
) -> dict[str, Any]:
    """Count ARA control files without asserting that their hashes verify."""

    manifest_count = 0
    lock_count = 0
    graph_count = 0
    verify_count = 0
    scanned = 0
    truncated = False
    digested_control = False
    bytes_read = 0
    partial_hash = False
    aggregate = hashlib.sha256()
    control_files, walk_truncated, walk_entries, walk_errors = _collect_regular_files(
        root, "ara", max_files=max_files
    )
    truncated = truncated or walk_truncated
    read_errors = walk_errors
    for relative_name, path in control_files:
        if scanned >= max_files:
            truncated = True
            break
        scanned += 1
        basename = Path(relative_name).name
        if basename == "manifest.json":
            manifest_count += 1
        elif basename == "manifest.lock":
            lock_count += 1
        elif basename == "exploration_graph.json":
            graph_count += 1
        elif "/verify/" in f"/{relative_name}/" or basename == "verify.json":
            verify_count += 1
        if basename in {"manifest.json", "manifest.lock", "exploration_graph.json"}:
            available = max(0, byte_budget - bytes_read)
            if available <= 0:
                truncated = True
                continue
            digest, consumed, partial = _digest_file(path, min(64 * 1024, available))
            bytes_read += consumed
            if digest:
                digested_control = True
                aggregate.update(relative_name.encode("utf-8", errors="replace"))
                aggregate.update(b"\0")
                aggregate.update(digest.encode("ascii"))
            else:
                read_errors += 1
                truncated = True
            partial_hash = partial_hash or partial
            truncated = truncated or partial
    if manifest_count == 0:
        lock_state = "not_observed"
    elif lock_count >= manifest_count:
        lock_state = "lock_present"
    else:
        lock_state = "lock_missing_or_incomplete"
    if read_errors:
        truncated = True
    return {
        "manifest_count": manifest_count,
        "lock_count": lock_count,
        "graph_count": graph_count,
        "verify_report_count": verify_count,
        "lock_state": lock_state,
        "control_digest": (
            f"sha256:{aggregate.hexdigest()}" if digested_control else None
        ),
        "digest_scope": (
            "bounded_prefix"
            if partial_hash or truncated or read_errors
            else "observed_control_files"
        ),
        "bytes_read": bytes_read,
        "fsck_run": False,
        "bundle_created": False,
        "truncated": truncated,
        "raw_payloads_included": False,
        "walk_entries_observed": walk_entries,
        "walk_truncated": walk_truncated,
        "source_count_complete": not walk_truncated and not bool(read_errors),
        "read_error_count": read_errors,
    }


def build_evidence_index(
    workspace: str | Path,
    *,
    max_files: int = _DEFAULT_MAX_FILES,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Return a privacy-preserving inventory of retained local evidence.

    This function never writes to ``workspace``.  ``file_count`` and
    ``byte_count`` describe the bounded portion actually hashed; ``truncated``
    makes that scope explicit.  Raw ARA/CAS payloads are therefore observed,
    never copied into the benchmark report.
    """

    if (
        isinstance(max_files, bool)
        or not isinstance(max_files, int)
        or max_files <= 0
        or max_files > _HARD_MAX_FILES
    ):
        raise ValueError("max_files must be greater than zero")
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or max_bytes > _HARD_MAX_BYTES
    ):
        raise ValueError("max_bytes must be greater than zero")
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("workspace does not exist or is not a directory")

    categories: dict[str, Any] = {}
    remaining = max_bytes
    truncated = False
    errors = 0
    for name, sources in _CATEGORY_SOURCES.items():
        category = _category_index(
            root,
            sources,
            max_files=max_files,
            byte_budget=remaining,
        )
        categories[name] = category
        remaining = max(0, remaining - int(category["byte_count"]))
        truncated = truncated or bool(category["truncated"])
        errors += int(category["read_error_count"])
        if remaining == 0 and name != "generated_views":
            truncated = True

    ara_contract = _ara_contract_summary(
        root, max_files=max_files, byte_budget=remaining
    )
    truncated = truncated or bool(ara_contract["truncated"])
    errors += int(ara_contract.get("read_error_count") or 0)
    return {
        "schema": EVIDENCE_INDEX_SCHEMA,
        "available": True,
        "mode": "read_only_bounded_hash_index",
        "hash_algorithm": "sha256",
        "workspace_root_disclosed": False,
        "paths_disclosed": False,
        "raw_content_included": False,
        "workspace_mutated": False,
        "limits": {"max_files_per_category": max_files, "max_bytes": max_bytes},
        "categories": categories,
        "ara_contract": ara_contract,
        "truncated": truncated,
        "read_error_count": errors,
    }


__all__ = ["EVIDENCE_INDEX_SCHEMA", "build_evidence_index"]
