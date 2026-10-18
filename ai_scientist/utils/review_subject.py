"""Portable, content-bound review inputs. These are not verification credentials."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.pipeline_contracts import (
    ARTIFACT_FILENAMES,
    load_contract_artifact,
)

REVIEW_SUBJECT_SCHEMA = "xscientist.review-subject.v1"
MAX_SUBJECT_BYTES = 256 * 1024 * 1024
MAX_SOURCE_FILES = 10_000


def _file_binding(root: Path, value: str | Path) -> dict[str, Any]:
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    relative = path.relative_to(root).as_posix()
    if not path.is_file():
        raise ValueError("not_a_regular_file")
    digest = hashlib.sha256()
    size = 0
    with os.fdopen(
        os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)), "rb"
    ) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("not_a_regular_file")
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if size > MAX_SUBJECT_BYTES:
                raise ValueError("subject_too_large")
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    current = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(current):
        raise ValueError("subject_changed_while_hashing")
    return {
        "path": relative,
        "sha256": "sha256:" + digest.hexdigest(),
        "size_bytes": size,
    }


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(root),
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    ).stdout


def _generated_review_path(value: str) -> bool:
    path = Path(value)
    return value in ARTIFACT_FILENAMES.values() or path.parts[0] in {"reviews", ".ara"}


def _source_revision(root: Path) -> dict[str, Any]:
    # A run directory must not accidentally inherit the application repo's HEAD.
    if not (root / ".git").exists():
        return {"status": "not_applicable"}
    top = Path(
        os.fsdecode(_git(root, "rev-parse", "--show-toplevel")).strip()
    ).resolve()
    if top != root:
        raise ValueError("unexpected_repository_root")
    commit = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip()
    paths = sorted(
        set(
            os.fsdecode(value)
            for value in _git(root, "ls-files", "--cached", "-z").split(b"\0")
            if value
        )
    )
    files = []
    total_bytes = 0
    for relative in paths:
        if _generated_review_path(relative):
            continue
        if len(files) >= MAX_SOURCE_FILES:
            raise ValueError("source_file_limit")
        path = root / relative
        if not path.exists() and not path.is_symlink():
            files.append({"path": relative, "status": "deleted"})
            continue
        binding = _file_binding(root, relative)
        total_bytes += binding["size_bytes"]
        if total_bytes > MAX_SUBJECT_BYTES:
            raise ValueError("source_byte_limit")
        files.append(
            {
                "path": relative,
                "content": binding,
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    if _git(root, "rev-parse", "--verify", "HEAD").decode("ascii").strip() != commit:
        raise ValueError("revision_changed_while_hashing")
    return {
        "status": "bound",
        "commit": commit,
        "working_tree_sha256": canonical_content_hash(files),
        "file_count": len(files),
    }


def capture_review_subject(
    project_root: str | Path, pdf_path: str | Path | None
) -> dict[str, Any]:
    """Fingerprint only locally addressable inputs, without persisting host paths."""
    if not pdf_path:
        return {
            "schema": REVIEW_SUBJECT_SCHEMA,
            "status": "unavailable",
            "reason": "missing_pdf",
        }
    root = Path(project_root).expanduser().resolve()
    try:
        pdf = _file_binding(root, pdf_path)
        if pdf["size_bytes"] == 0:
            raise ValueError("empty_pdf")
        manuscript = load_contract_artifact(root, "manuscript_state", default={})
        source_path = (
            manuscript.get("latex_path") if isinstance(manuscript, dict) else None
        )
        source = _file_binding(root, source_path) if source_path else None
        subject = {
            "schema": REVIEW_SUBJECT_SCHEMA,
            "status": "bound",
            "pdf": pdf,
            "manuscript_source": source,
            "source_revision": _source_revision(root),
        }
        return {**subject, "subject_hash": canonical_content_hash(subject)}
    except (OSError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError):
        return {
            "schema": REVIEW_SUBJECT_SCHEMA,
            "status": "unavailable",
            "reason": "unreadable_or_unbound_subject",
        }


def check_review_subject(
    project_root: str | Path, subject: Any, *, pdf_path: str | Path | None = None
) -> tuple[bool, str]:
    """Reject legacy, incomplete, tampered, or stale input bindings."""
    if not isinstance(subject, dict) or subject.get("status") != "bound":
        return False, "missing_review_subject"
    if subject.get("schema") != REVIEW_SUBJECT_SCHEMA:
        return False, "invalid_review_subject"
    try:
        core = {key: value for key, value in subject.items() if key != "subject_hash"}
        if canonical_content_hash(core) != subject.get("subject_hash"):
            return False, "invalid_review_subject"
        path = pdf_path if pdf_path is not None else subject["pdf"]["path"]
        current = capture_review_subject(project_root, path)
        if current.get("status") != "bound":
            return False, "unavailable_review_subject"
        if current["subject_hash"] != subject["subject_hash"]:
            return False, "stale_review_subject"
    except (KeyError, TypeError, ValueError):
        return False, "invalid_review_subject"
    return True, "current_review_subject"
