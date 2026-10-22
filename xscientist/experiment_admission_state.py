"""Checkout-independent high-water marks for experiment-admission decisions.

Research objects live in Git history, so checking out an older commit can hide a
later denial.  This module keeps an append-only decision chain in Git's shared
common directory.  Every linked worktree observes the same latest decision.

The common directory is a cooperative host boundary, like the execution-claim
and terminal-reservation records in :mod:`xscientist.research_git`.  A process
that can arbitrarily rewrite Git control files can still destroy this state;
production deployments must make that directory host-controlled.
"""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_git import (
    ResearchGitError,
    _REPOSITORY_LOCK_STATE,
    _fsync_directory,
    _repository_common_git_dir,
    _run_git,
)

_STATE_SCHEMA = "xscientist.experiment-admission-high-water.v1"
_PENDING_SCHEMA = "xscientist.experiment-admission-transition.v1"
_MAX_EVENTS_PER_SUBJECT = 4096
_MAX_DECISION_SUBJECTS = 4096
_MAX_SUBJECT_ENTRIES = 16
_MAX_EVENT_BYTES = 1 << 20
_OBJECT_ID_RE = re.compile(r"^rso-[0-9a-f]{16}$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_EVENT_NAME_RE = re.compile(r"^(?P<sequence>[0-9]{20})-(?P<digest>[0-9a-f]{16})\.json$")
_SUBJECT_DIRECTORY_RE = re.compile(r"^[0-9a-f]{64}$")
_PENDING_STAGING_NAME_RE = re.compile(r"^\.admission-pending-[A-Za-z0-9_-]+\.tmp$")
_EVENT_STAGING_NAME_RE = re.compile(r"^\.admission-event-[A-Za-z0-9_-]+\.tmp$")
_CHECKPOINT_REF_PREFIX = "refs/xscientist/admission"
_EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "subject_hash",
        "sequence",
        "previous_event_hash",
        "gate_object_id",
        "gate_object_hash",
        "decision",
        "admission_request_hash",
        "authority_receipt_hash",
        "checkpoint_commit",
        "event_hash",
    }
)
_PENDING_FIELDS = frozenset(
    {
        "schema_version",
        "subject_hash",
        "gate_object_id",
        "gate_object_hash",
        "decision",
        "admission_request_hash",
        "authority_receipt_hash",
        "transition_hash",
    }
)


def experiment_admission_subject_hash(payload: Mapping[str, Any]) -> str:
    """Return the exact repository/input/scope identity shared by all verdicts."""

    if not isinstance(payload, Mapping):
        raise ResearchGitError("experiment admission decision payload is invalid")
    repository_id = payload.get("repository_id")
    bindings = payload.get("input_bindings")
    scope = payload.get("scope")
    if (
        not isinstance(repository_id, str)
        or not repository_id
        or not isinstance(bindings, Mapping)
        or not isinstance(scope, Mapping)
    ):
        raise ResearchGitError("experiment admission decision subject is invalid")
    try:
        return canonical_content_hash(
            {
                "repository_id": repository_id,
                "input_bindings": dict(bindings),
                "scope": dict(scope),
            }
        )
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "experiment admission decision subject is not canonical JSON"
        ) from exc


def _checkpoint_ref_name(subject_hash: str) -> str:
    if not _HASH_RE.fullmatch(subject_hash):
        raise ResearchGitError("experiment admission decision subject hash is invalid")
    return f"{_CHECKPOINT_REF_PREFIX}/{subject_hash.removeprefix('sha256:')}"


def _read_checkpoint_ref(
    repo: str | Path,
    subject_hash: str,
) -> str | None:
    """Resolve one direct hidden retention ref without accepting symbolic refs."""

    root = Path(repo).expanduser().resolve()
    ref_name = _checkpoint_ref_name(subject_hash)
    symbolic = _run_git(
        root,
        ["symbolic-ref", "--quiet", ref_name],
        check=False,
    )
    if symbolic.returncode == 0:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref is unsafe"
        )
    if symbolic.returncode != 1:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref is unreadable"
        )
    shown = _run_git(
        root,
        ["show-ref", "--verify", "--quiet", ref_name],
        check=False,
    )
    if shown.returncode == 1:
        return None
    if shown.returncode != 0:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref is unreadable"
        )
    resolved = _run_git(
        root,
        ["rev-parse", "--verify", f"{ref_name}^{{commit}}"],
        check=False,
    )
    checkpoint = resolved.stdout.strip()
    if resolved.returncode != 0 or not _COMMIT_RE.fullmatch(checkpoint):
        raise ResearchGitError(
            "experiment admission checkpoint retention ref is invalid"
        )
    return checkpoint


def _validate_checkpoint_ref(
    repo: str | Path,
    subject_hash: str,
    latest_event: Mapping[str, Any] | None,
) -> None:
    expected = (
        str(latest_event.get("checkpoint_commit") or "")
        if latest_event is not None
        else None
    )
    if _read_checkpoint_ref(repo, subject_hash) != expected:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref does not match the "
            "latest high-water event"
        )


def _reconcile_checkpoint_ref(
    repo: str | Path,
    subject_hash: str,
    events: list[dict[str, Any]],
) -> None:
    """Create or advance the hidden ref for an already durable latest event."""

    if not events:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref has no high-water event"
        )
    latest_checkpoint = str(events[-1]["checkpoint_commit"])
    current_checkpoint = _read_checkpoint_ref(repo, subject_hash)
    if current_checkpoint == latest_checkpoint:
        return
    previous_checkpoint = (
        str(events[-2]["checkpoint_commit"]) if len(events) > 1 else None
    )
    if current_checkpoint is not None and current_checkpoint != previous_checkpoint:
        raise ResearchGitError(
            "experiment admission checkpoint retention ref conflicts with the "
            "high-water chain"
        )
    expected_old = current_checkpoint or ("0" * len(latest_checkpoint))
    _run_git(
        Path(repo).expanduser().resolve(),
        [
            "update-ref",
            "--no-deref",
            _checkpoint_ref_name(subject_hash),
            latest_checkpoint,
            expected_old,
        ],
    )
    _validate_checkpoint_ref(repo, subject_hash, events[-1])


def _safe_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink():
        raise ResearchGitError(f"experiment admission {label} directory is unsafe")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ResearchGitError(f"experiment admission {label} directory is unsafe")
    return path


def _directory_entries(
    path: Path,
    *,
    label: str,
    maximum: int | None = None,
) -> list[Path]:
    """List one validated directory without silently following a replacement."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ResearchGitError(
            f"experiment admission {label} directory is unreadable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ResearchGitError(f"experiment admission {label} directory is unsafe")
    try:
        entries = []
        for entry in path.iterdir():
            entries.append(entry)
            if maximum is not None and len(entries) > maximum:
                raise ResearchGitError(
                    f"experiment admission {label} directory has too many entries"
                )
        entries.sort(key=lambda item: item.name)
        current = path.lstat()
    except OSError as exc:
        raise ResearchGitError(
            f"experiment admission {label} directory is unreadable"
        ) from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino, current.st_mode)
        != (metadata.st_dev, metadata.st_ino, metadata.st_mode)
    ):
        raise ResearchGitError(
            f"experiment admission {label} directory changed during enumeration"
        )
    return entries


def _decision_directories_root(repo: str | Path, *, create: bool) -> Path:
    common = _repository_common_git_dir(Path(repo).expanduser().resolve())
    state_root = common / "xscientist"
    decisions_root = state_root / "admission-decisions"
    if create:
        _safe_directory(state_root, label="state")
        return _safe_directory(decisions_root, label="decision-state")
    for path, label in (
        (state_root, "state"),
        (decisions_root, "decision-state"),
    ):
        if path.is_symlink():
            raise ResearchGitError(f"experiment admission {label} directory is unsafe")
        if not path.exists():
            return decisions_root
        if not path.is_dir():
            raise ResearchGitError(f"experiment admission {label} directory is unsafe")
    return decisions_root


def _validate_subject_layout(subject_root: Path) -> None:
    """Reject unrecognized subject state and crash staging residue fail closed."""

    saw_events = False
    for entry in _directory_entries(
        subject_root,
        label="decision-subject",
        maximum=_MAX_SUBJECT_ENTRIES,
    ):
        if entry.name == "events":
            try:
                metadata = entry.lstat()
            except OSError as exc:
                raise ResearchGitError(
                    "experiment admission decision-events directory is unreadable"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ResearchGitError(
                    "experiment admission decision-events directory is unsafe"
                )
            saw_events = True
            continue
        if entry.name == "pending.json":
            # The bounded, no-follow reader validates the file itself.
            continue
        if _PENDING_STAGING_NAME_RE.fullmatch(
            entry.name
        ) or _EVENT_STAGING_NAME_RE.fullmatch(entry.name):
            try:
                metadata = entry.lstat()
            except OSError as exc:
                raise ResearchGitError(
                    "experiment admission staging residue is unreadable"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ResearchGitError("experiment admission staging residue is unsafe")
            raise ResearchGitError(
                "experiment admission staging residue requires exact transition retry"
            )
        raise ResearchGitError("experiment admission decision-subject entry is invalid")
    if not saw_events:
        raise ResearchGitError(
            "experiment admission decision-subject events directory is missing"
        )


def _cleanup_subject_staging(
    subject_root: Path,
    *,
    pending: bool,
    events: bool,
) -> None:
    """Remove only this module's non-authoritative temp files on a write retry."""

    removed = False
    for entry in _directory_entries(
        subject_root,
        label="decision-subject",
        maximum=_MAX_SUBJECT_ENTRIES,
    ):
        recognized = bool(
            (pending and _PENDING_STAGING_NAME_RE.fullmatch(entry.name))
            or (events and _EVENT_STAGING_NAME_RE.fullmatch(entry.name))
        )
        if not recognized:
            continue
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise ResearchGitError(
                "experiment admission staging residue is unreadable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ResearchGitError("experiment admission staging residue is unsafe")
        entry.unlink()
        removed = True
    if removed:
        _fsync_directory(subject_root)


def recover_experiment_admission_staging(
    repo: str | Path,
    decision_payload: Mapping[str, Any],
) -> None:
    """Clean exact-subject temp residue while the exclusive repository lock is held."""

    root = Path(repo).expanduser().resolve()
    common = _repository_common_git_dir(root)
    lock_key = str((common / "xscientist-research.lock").resolve())
    held = getattr(_REPOSITORY_LOCK_STATE, "held", set())
    if lock_key not in held:
        raise ResearchGitError(
            "experiment admission staging recovery requires the exclusive "
            "repository lock"
        )
    subject_hash = experiment_admission_subject_hash(decision_payload)
    events_dir = _events_directory(root, subject_hash, create=False)
    subject_root = events_dir.parent
    if not subject_root.exists() and not subject_root.is_symlink():
        return
    if subject_root.is_symlink() or not subject_root.is_dir():
        raise ResearchGitError(
            "experiment admission decision-subject directory is unsafe"
        )
    # Creating a missing events directory repairs only an interrupted local
    # directory setup; no authoritative event or pending marker is synthesized.
    events_dir = _events_directory(root, subject_hash, create=True)
    _cleanup_subject_staging(subject_root, pending=True, events=True)
    _validate_subject_layout(events_dir.parent)


def _events_directory(repo: str | Path, subject_hash: str, *, create: bool) -> Path:
    if not _HASH_RE.fullmatch(subject_hash):
        raise ResearchGitError("experiment admission decision subject hash is invalid")
    decisions_root = _decision_directories_root(repo, create=create)
    subject_root = decisions_root / subject_hash.removeprefix("sha256:")
    events = subject_root / "events"
    if create:
        if not subject_root.exists() and not subject_root.is_symlink():
            existing_subjects = _directory_entries(
                decisions_root,
                label="decision-state",
                maximum=_MAX_DECISION_SUBJECTS,
            )
            if len(existing_subjects) >= _MAX_DECISION_SUBJECTS:
                raise ResearchGitError(
                    "experiment admission decision subject capacity is exhausted"
                )
        _safe_directory(subject_root, label="decision-subject")
        return _safe_directory(events, label="decision-events")
    for path, label in (
        (subject_root, "decision-subject"),
        (events, "decision-events"),
    ):
        if path.is_symlink():
            raise ResearchGitError(f"experiment admission {label} directory is unsafe")
        if not path.exists():
            return events
        if not path.is_dir():
            raise ResearchGitError(f"experiment admission {label} directory is unsafe")
    return events


def _read_event(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ResearchGitError(
            "experiment admission high-water event is unreadable"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResearchGitError("experiment admission high-water event is unsafe")
    if metadata.st_size > _MAX_EVENT_BYTES:
        raise ResearchGitError("experiment admission high-water event is too large")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResearchGitError(
            "experiment admission high-water event is unreadable"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or opened.st_size > _MAX_EVENT_BYTES
        ):
            raise ResearchGitError(
                "experiment admission high-water event changed during read"
            )
        chunks: list[bytes] = []
        remaining = _MAX_EVENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(encoded) > _MAX_EVENT_BYTES:
        raise ResearchGitError("experiment admission high-water event is too large")
    try:
        decoded = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResearchGitError(
            "experiment admission high-water event is invalid JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise ResearchGitError("experiment admission high-water event is invalid")
    return decoded


def _validate_event(
    event: Mapping[str, Any],
    *,
    subject_hash: str,
    expected_sequence: int,
    previous_event_hash: str | None,
) -> dict[str, Any]:
    row = dict(event)
    event_hash = row.pop("event_hash", None)
    if (
        set(event) != _EVENT_FIELDS
        or row.get("schema_version") != _STATE_SCHEMA
        or row.get("subject_hash") != subject_hash
        or row.get("sequence") != expected_sequence
        or isinstance(row.get("sequence"), bool)
        or row.get("previous_event_hash") != previous_event_hash
        or not _OBJECT_ID_RE.fullmatch(str(row.get("gate_object_id") or ""))
        or not _HASH_RE.fullmatch(str(row.get("gate_object_hash") or ""))
        or row.get("decision") not in {"authorize", "deny"}
        or not _HASH_RE.fullmatch(str(row.get("admission_request_hash") or ""))
        or not _HASH_RE.fullmatch(str(row.get("authority_receipt_hash") or ""))
        or not _COMMIT_RE.fullmatch(str(row.get("checkpoint_commit") or ""))
    ):
        raise ResearchGitError("experiment admission high-water event is invalid")
    try:
        expected_hash = canonical_content_hash(row)
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(
            "experiment admission high-water event is invalid"
        ) from exc
    if event_hash != expected_hash:
        raise ResearchGitError(
            "experiment admission high-water event hash does not match"
        )
    return {**row, "event_hash": event_hash}


def _load_experiment_admission_events_by_subject(
    repo: str | Path,
    subject_hash: str,
) -> list[dict[str, Any]]:
    """Validate and return one complete event chain by its trusted directory key."""

    events_dir = _events_directory(repo, subject_hash, create=False)
    if events_dir.parent.exists() or events_dir.parent.is_symlink():
        _validate_subject_layout(events_dir.parent)
    if not events_dir.exists():
        return []
    try:
        entries = _directory_entries(
            events_dir,
            label="decision-events",
            maximum=_MAX_EVENTS_PER_SUBJECT,
        )
    except ResearchGitError as exc:
        if "too many entries" in str(exc):
            raise ResearchGitError(
                "experiment admission high-water chain is too long"
            ) from exc
        raise
    if not entries:
        return []
    previous_hash: str | None = None
    events: list[dict[str, Any]] = []
    for sequence, path in enumerate(entries, start=1):
        match = _EVENT_NAME_RE.fullmatch(path.name)
        if match is None or int(match.group("sequence")) != sequence:
            raise ResearchGitError(
                "experiment admission high-water chain is not contiguous"
            )
        current = _validate_event(
            _read_event(path),
            subject_hash=subject_hash,
            expected_sequence=sequence,
            previous_event_hash=previous_hash,
        )
        if current["event_hash"].removeprefix("sha256:")[:16] != match.group("digest"):
            raise ResearchGitError(
                "experiment admission high-water filename is invalid"
            )
        previous_hash = str(current["event_hash"])
        events.append(current)
    return events


def _load_experiment_admission_events(
    repo: str | Path,
    decision_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate and return the complete per-subject event chain."""

    return _load_experiment_admission_events_by_subject(
        repo,
        experiment_admission_subject_hash(decision_payload),
    )


def load_experiment_admission_high_water(
    repo: str | Path,
    decision_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Validate the complete per-subject event chain and return its latest event."""

    events = _load_experiment_admission_events(repo, decision_payload)
    return events[-1] if events else None


def _gate_transition_values(gate: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = gate.get("payload") if isinstance(gate, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ResearchGitError("experiment admission gate payload is invalid")
    subject_hash = experiment_admission_subject_hash(payload)
    authority = payload.get("authority_receipt")
    authority = authority if isinstance(authority, Mapping) else {}
    values = {
        "gate_object_id": str(gate.get("object_id") or ""),
        "gate_object_hash": str(gate.get("content_hash") or ""),
        "decision": payload.get("decision"),
        "admission_request_hash": str(payload.get("admission_request_hash") or ""),
        "authority_receipt_hash": str(authority.get("receipt_hash") or ""),
    }
    if (
        not _OBJECT_ID_RE.fullmatch(values["gate_object_id"])
        or not _HASH_RE.fullmatch(values["gate_object_hash"])
        or values["decision"] not in {"authorize", "deny"}
        or not _HASH_RE.fullmatch(values["admission_request_hash"])
        or not _HASH_RE.fullmatch(values["authority_receipt_hash"])
    ):
        raise ResearchGitError(
            "experiment admission gate cannot enter high-water state"
        )
    return subject_hash, values


def _pending_transition_path_by_subject(
    repo: str | Path,
    subject_hash: str,
    *,
    create: bool,
) -> Path:
    return _events_directory(repo, subject_hash, create=create).parent / "pending.json"


def _pending_transition_path(
    repo: str | Path,
    decision_payload: Mapping[str, Any],
    *,
    create: bool,
) -> Path:
    return _pending_transition_path_by_subject(
        repo,
        experiment_admission_subject_hash(decision_payload),
        create=create,
    )


def _validate_pending_transition(
    transition: Mapping[str, Any],
    *,
    subject_hash: str,
) -> dict[str, Any] | None:
    core = dict(transition)
    transition_hash = core.pop("transition_hash", None)
    if (
        set(transition) != _PENDING_FIELDS
        or core.get("schema_version") != _PENDING_SCHEMA
        or core.get("subject_hash") != subject_hash
        or not _OBJECT_ID_RE.fullmatch(str(core.get("gate_object_id") or ""))
        or not _HASH_RE.fullmatch(str(core.get("gate_object_hash") or ""))
        or core.get("decision") not in {"authorize", "deny"}
        or not _HASH_RE.fullmatch(str(core.get("admission_request_hash") or ""))
        or not _HASH_RE.fullmatch(str(core.get("authority_receipt_hash") or ""))
    ):
        raise ResearchGitError(
            "experiment admission pending high-water transition is invalid"
        )
    if transition_hash != canonical_content_hash(core):
        raise ResearchGitError(
            "experiment admission pending high-water transition hash does not match"
        )
    return {**core, "transition_hash": transition_hash}


def _load_experiment_admission_transition_by_subject(
    repo: str | Path,
    subject_hash: str,
    *,
    validate_layout: bool = True,
) -> dict[str, Any] | None:
    path = _pending_transition_path_by_subject(repo, subject_hash, create=False)
    if path.parent.exists() or path.parent.is_symlink():
        if validate_layout:
            _validate_subject_layout(path.parent)
    if not path.exists() and not path.is_symlink():
        return None
    return _validate_pending_transition(
        _read_event(path),
        subject_hash=subject_hash,
    )


def load_experiment_admission_transition(
    repo: str | Path,
    decision_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated verdict transition that has not fully published."""

    return _load_experiment_admission_transition_by_subject(
        repo,
        experiment_admission_subject_hash(decision_payload),
    )


def _repository_decision_subjects(
    repo: str | Path,
) -> list[tuple[str, Path]]:
    """Return every validated per-subject state directory in stable order."""

    decisions_root = _decision_directories_root(repo, create=False)
    if not decisions_root.exists() and not decisions_root.is_symlink():
        return []
    entries = _directory_entries(
        decisions_root,
        label="decision-state",
        maximum=_MAX_DECISION_SUBJECTS,
    )
    subjects: list[tuple[str, Path]] = []
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise ResearchGitError(
                "experiment admission decision subject is unreadable"
            ) from exc
        if (
            _SUBJECT_DIRECTORY_RE.fullmatch(entry.name) is None
            or stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
        ):
            raise ResearchGitError("experiment admission decision subject is unsafe")
        _validate_subject_layout(entry)
        subjects.append((f"sha256:{entry.name}", entry))
    return subjects


def _load_repository_experiment_admission_state(
    repo: str | Path,
    *,
    validate_checkpoint_refs: bool = True,
) -> list[tuple[list[dict[str, Any]], dict[str, Any] | None]]:
    """Validate every subject before exposing any repository-wide state."""

    rows: list[tuple[list[dict[str, Any]], dict[str, Any] | None]] = []
    for subject_hash, _subject_root in _repository_decision_subjects(repo):
        events = _load_experiment_admission_events_by_subject(repo, subject_hash)
        pending = _load_experiment_admission_transition_by_subject(repo, subject_hash)
        if validate_checkpoint_refs:
            _validate_checkpoint_ref(
                repo,
                subject_hash,
                events[-1] if events else None,
            )
        rows.append((events, pending))
    return rows


def load_all_experiment_admission_high_water(
    repo: str | Path,
) -> list[dict[str, Any]]:
    """Return the validated latest event for every repository decision subject."""

    return [
        events[-1]
        for events, _pending in _load_repository_experiment_admission_state(repo)
        if events
    ]


def load_all_experiment_admission_transitions(
    repo: str | Path,
) -> list[dict[str, Any]]:
    """Return every validated pending transition across all decision subjects."""

    return [
        pending
        for _events, pending in _load_repository_experiment_admission_state(repo)
        if pending is not None
    ]


def begin_experiment_admission_transition(
    repo: str | Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish a fail-closed marker before committing a new signed verdict."""

    subject_hash, values = _gate_transition_values(gate)
    core = {
        "schema_version": _PENDING_SCHEMA,
        "subject_hash": subject_hash,
        **values,
    }
    transition = {
        **core,
        "transition_hash": canonical_content_hash(core),
    }
    events_dir = _events_directory(repo, subject_hash, create=True)
    path = events_dir.parent / "pending.json"
    _cleanup_subject_staging(path.parent, pending=True, events=True)
    events = _load_experiment_admission_events_by_subject(repo, subject_hash)
    already_published = bool(
        events
        and events[-1].get("gate_object_id") == values["gate_object_id"]
        and events[-1].get("gate_object_hash") == values["gate_object_hash"]
        and events[-1].get("admission_request_hash") == values["admission_request_hash"]
        and events[-1].get("authority_receipt_hash") == values["authority_receipt_hash"]
    )
    if not already_published and len(events) >= _MAX_EVENTS_PER_SUBJECT:
        raise ResearchGitError("experiment admission high-water chain is too long")
    pending = [
        transition
        for _events, transition in _load_repository_experiment_admission_state(
            repo,
            validate_checkpoint_refs=False,
        )
        if transition is not None
    ]
    if pending:
        if len(pending) != 1 or pending[0] != transition:
            raise ResearchGitError(
                "another experiment admission high-water transition is pending"
            )
        if already_published:
            _fsync_directory(events_dir)
        # A prior link may have become visible before its directory fsync failed.
        # Exact retry cannot acknowledge it until that durability barrier succeeds.
        _fsync_directory(path.parent)
        return pending[0]
    if already_published:
        # The lifecycle's exact-retry branch deliberately skips another append.
        # Re-establish durability here before it is allowed to clear pending state.
        _fsync_directory(events_dir)

    encoded = (
        json.dumps(
            transition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary_descriptor = -1
    temporary_name = ""
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".admission-pending-",
            suffix=".tmp",
            dir=path.parent,
        )
        os.chmod(temporary_name, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(temporary_descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("experiment admission transition write made no progress")
            offset += written
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.link(temporary_name, path)
        _fsync_directory(path.parent)
    except FileExistsError:
        existing = _load_experiment_admission_transition_by_subject(
            repo,
            subject_hash,
            validate_layout=False,
        )
        if existing != transition:
            raise ResearchGitError(
                "another experiment admission high-water transition is pending"
            )
        _fsync_directory(path.parent)
        return transition
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
            _fsync_directory(path.parent)
    return transition


def clear_experiment_admission_transition(
    repo: str | Path,
    gate: Mapping[str, Any],
) -> None:
    """Clear exactly the pending transition represented by ``gate``."""

    subject_hash, values = _gate_transition_values(gate)
    core = {
        "schema_version": _PENDING_SCHEMA,
        "subject_hash": subject_hash,
        **values,
    }
    expected = {**core, "transition_hash": canonical_content_hash(core)}
    payload = gate.get("payload") or {}
    existing = load_experiment_admission_transition(repo, payload)
    if existing is None:
        return
    if existing != expected:
        raise ResearchGitError(
            "experiment admission pending high-water transition does not match"
        )
    path = _pending_transition_path(repo, payload, create=False)
    path.unlink()
    _fsync_directory(path.parent)


def record_experiment_admission_high_water(
    repo: str | Path,
    gate: Mapping[str, Any],
    *,
    checkpoint_commit: str,
) -> dict[str, Any]:
    """Append one committed signed verdict to the checkout-independent chain.

    The caller must hold the repository's exclusive lock and must first verify
    the gate, its dedicated checkpoint, and the current external trust policy.
    """

    payload = gate.get("payload") if isinstance(gate, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ResearchGitError("experiment admission gate payload is invalid")
    subject_hash, values = _gate_transition_values(gate)
    checkpoint = str(checkpoint_commit or "")
    if not _COMMIT_RE.fullmatch(checkpoint):
        raise ResearchGitError(
            "experiment admission gate cannot enter high-water state"
        )
    values["checkpoint_commit"] = checkpoint
    events_dir = _events_directory(repo, subject_hash, create=True)
    _cleanup_subject_staging(events_dir.parent, pending=False, events=True)
    events = _load_experiment_admission_events_by_subject(repo, subject_hash)
    current = events[-1] if events else None
    if current is not None and all(
        current.get(key) == value for key, value in values.items()
    ):
        _fsync_directory(_events_directory(repo, subject_hash, create=False))
        _reconcile_checkpoint_ref(repo, subject_hash, events)
        return current
    _validate_checkpoint_ref(repo, subject_hash, current)
    if (
        current is not None
        and current.get("gate_object_id") == values["gate_object_id"]
    ):
        raise ResearchGitError(
            "experiment admission high-water gate identity conflicts"
        )
    if any(
        event.get("gate_object_id") == values["gate_object_id"]
        or (
            event.get("admission_request_hash") == values["admission_request_hash"]
            and event.get("authority_receipt_hash") == values["authority_receipt_hash"]
        )
        for event in events
    ):
        raise ResearchGitError(
            "experiment admission high-water decision replay is older than the "
            "current decision"
        )
    sequence = int(current["sequence"]) + 1 if current is not None else 1
    if sequence > _MAX_EVENTS_PER_SUBJECT:
        raise ResearchGitError("experiment admission high-water chain is too long")
    core = {
        "schema_version": _STATE_SCHEMA,
        "subject_hash": subject_hash,
        "sequence": sequence,
        "previous_event_hash": (
            current.get("event_hash") if current is not None else None
        ),
        **values,
    }
    event = {**core, "event_hash": canonical_content_hash(core)}
    path = events_dir / (
        f"{sequence:020d}-" f"{event['event_hash'].removeprefix('sha256:')[:16]}.json"
    )
    encoded = (
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary_descriptor = -1
    temporary_name = ""
    try:
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".admission-event-",
            suffix=".tmp",
            dir=events_dir.parent,
        )
        os.chmod(temporary_name, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(temporary_descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("experiment admission high-water write made no progress")
            offset += written
        os.fsync(temporary_descriptor)
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ResearchGitError(
                "experiment admission high-water sequence already exists"
            ) from exc
        _fsync_directory(events_dir)
        _reconcile_checkpoint_ref(repo, subject_hash, [*events, event])
    except BaseException:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        raise
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)
            _fsync_directory(events_dir.parent)
    return event


def require_current_experiment_admission_decision(
    repo: str | Path,
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Require ``gate`` to be the latest decision across every checkout."""

    payload = gate.get("payload") if isinstance(gate, Mapping) else None
    if not isinstance(payload, Mapping):
        raise ResearchGitError("experiment admission gate payload is invalid")
    if load_experiment_admission_transition(repo, payload) is not None:
        raise ResearchGitError(
            "experiment admission high-water transition is pending; retry the signed "
            "verdict ingestion before executing"
        )
    current = load_experiment_admission_high_water(repo, payload)
    if current is None:
        raise ResearchGitError(
            "experiment admission gate has no host high-water record; re-ingest it "
            "against the current external trust policy"
        )
    _validate_checkpoint_ref(
        repo,
        experiment_admission_subject_hash(payload),
        current,
    )
    if (
        current.get("gate_object_id") != gate.get("object_id")
        or current.get("gate_object_hash") != gate.get("content_hash")
        or current.get("decision") != payload.get("decision")
        or current.get("admission_request_hash")
        != payload.get("admission_request_hash")
        or current.get("authority_receipt_hash")
        != (payload.get("authority_receipt") or {}).get("receipt_hash")
    ):
        raise ResearchGitError(
            "experiment admission gate is older than the host high-water decision"
        )
    return current


__all__ = [
    "begin_experiment_admission_transition",
    "clear_experiment_admission_transition",
    "experiment_admission_subject_hash",
    "load_all_experiment_admission_high_water",
    "load_all_experiment_admission_transitions",
    "load_experiment_admission_high_water",
    "load_experiment_admission_transition",
    "record_experiment_admission_high_water",
    "recover_experiment_admission_staging",
    "require_current_experiment_admission_decision",
]
