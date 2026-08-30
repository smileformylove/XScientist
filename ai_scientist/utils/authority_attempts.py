"""Append-only receipts for scientific-authority decisions and execution attempts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, TypeVar

from ai_scientist.utils.safe_files import BoundedFileError, read_bounded_regular_file

ATTEMPT_ID_RE = re.compile(r"^attempt-[0-9a-f]{32}$")
OBJECT_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OBJECT_REF_RE = re.compile(
    r"^authority_objects/[a-z][a-z0-9-]{0,63}/([0-9a-f]{64})\.json$"
)
RESULT_REF_RE = re.compile(r"^authority_objects/attempt-result/([0-9a-f]{64})\.json$")
EVENT_FILE_RE = re.compile(r"^(0|[1-9][0-9]{0,8})\.json$")
TERMINAL_ATTEMPT_STATUSES = frozenset({"accepted", "rejected", "failed", "timeout"})
MAX_ATTEMPT_EVENT_BYTES = 1024 * 1024
MAX_ATTEMPT_EVENTS = 4096
_T = TypeVar("_T")
_RESULT_UNSET = object()


class AuthorityAttemptError(RuntimeError):
    """Raised when an authority attempt cannot be persisted or verified safely."""


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuthorityAttemptError("authority payload is not strict JSON") from exc


def _strict_json_load(payload: bytes, *, label: str) -> Any:
    def reject_duplicate_pairs(pairs):
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AuthorityAttemptError(f"{label} has duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise AuthorityAttemptError(f"{label} contains non-finite JSON")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except AuthorityAttemptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorityAttemptError(f"{label} is invalid JSON") from exc
    if _canonical_json_bytes(parsed) != payload:
        raise AuthorityAttemptError(f"{label} is not canonical JSON")
    return parsed


def canonical_authority_hash(payload: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_uid() -> int | None:
    getter = getattr(os, "getuid", None)
    return int(getter()) if callable(getter) else None


def _normalized_log_dir(log_dir: str | os.PathLike[str]) -> Path:
    selected = Path(log_dir).expanduser()
    if not selected.is_absolute():
        selected = Path.cwd() / selected
    # abspath removes dot segments without following symlinks.
    return Path(os.path.abspath(selected))


def _bounded_text(
    value: object,
    *,
    label: str,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise AuthorityAttemptError(f"{label} must be text")
    selected = value.strip()
    if (not selected and not allow_empty) or len(selected) > maximum:
        raise AuthorityAttemptError(f"{label} has an invalid length")
    if any(ord(character) < 32 or ord(character) == 127 for character in selected):
        raise AuthorityAttemptError(f"{label} contains control characters")
    return selected


def _directory_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _validate_directory_metadata(
    path_metadata: os.stat_result,
    descriptor_metadata: os.stat_result,
    *,
    private: bool,
) -> None:
    uid = _current_uid()
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or not stat.S_ISDIR(descriptor_metadata.st_mode)
        or path_metadata.st_dev != descriptor_metadata.st_dev
        or path_metadata.st_ino != descriptor_metadata.st_ino
        or (uid is not None and descriptor_metadata.st_uid != uid)
        or (private and stat.S_IMODE(descriptor_metadata.st_mode) & 0o077)
    ):
        raise AuthorityAttemptError("authority attempt directory is unsafe")


def _open_directory(path: Path, *, private: bool) -> int:
    try:
        path_metadata = path.lstat()
        descriptor = os.open(path, _directory_flags())
        descriptor_metadata = os.fstat(descriptor)
        _validate_directory_metadata(
            path_metadata,
            descriptor_metadata,
            private=private,
        )
        return descriptor
    except AuthorityAttemptError:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass
        raise
    except OSError as exc:
        raise AuthorityAttemptError(
            "authority attempt directory is unavailable"
        ) from exc


def _open_private_directory(path: Path) -> int:
    descriptor = _open_directory(path, private=False)
    try:
        os.fchmod(descriptor, 0o700)
        metadata = os.fstat(descriptor)
        uid = _current_uid()
        if stat.S_IMODE(metadata.st_mode) & 0o077 or (
            uid is not None and metadata.st_uid != uid
        ):
            raise AuthorityAttemptError("authority directory is not private")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _attempt_root(
    log_dir: str | os.PathLike[str],
    *,
    create: bool,
) -> Path | None:
    root_log = _normalized_log_dir(log_dir)
    if root_log.is_symlink() or (root_log.exists() and not root_log.is_dir()):
        raise AuthorityAttemptError("authority log directory is unsafe")
    if not root_log.exists():
        if not create:
            return None
        try:
            root_log.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise AuthorityAttemptError(
                "authority log directory is unavailable"
            ) from exc
    root = root_log / "authority_attempts"
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise AuthorityAttemptError("authority attempt root is unsafe")
    if not root.exists():
        if not create:
            return None
        try:
            root.mkdir(mode=0o700, parents=False, exist_ok=False)
        except OSError as exc:
            raise AuthorityAttemptError(
                "authority attempt root is unavailable"
            ) from exc
    descriptor = _open_directory(root, private=True)
    os.close(descriptor)
    return root


def persist_authority_object(
    log_dir: str | os.PathLike[str],
    *,
    category: str,
    payload: dict[str, Any],
) -> tuple[str, str | None]:
    """Persist one immutable content-addressed decision/specification object."""

    safe_category = str(category).strip().replace("_", "-")
    if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", safe_category) is None:
        raise AuthorityAttemptError("invalid authority object category")
    object_hash = canonical_authority_hash(payload)
    root_log = _normalized_log_dir(log_dir)
    if root_log.is_symlink() or (root_log.exists() and not root_log.is_dir()):
        raise AuthorityAttemptError("authority log directory is unsafe")
    try:
        root_log.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AuthorityAttemptError("authority log directory is unavailable") from exc
    objects = root_log / "authority_objects"
    category_dir = objects / safe_category
    for directory, mode in ((objects, 0o700), (category_dir, 0o700)):
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise AuthorityAttemptError("authority object directory is unsafe")
        try:
            directory.mkdir(mode=mode, parents=False, exist_ok=True)
        except OSError as exc:
            raise AuthorityAttemptError(
                "authority object directory is unavailable"
            ) from exc
        descriptor = _open_private_directory(directory)
        os.close(descriptor)
    relative = (
        Path("authority_objects")
        / safe_category
        / f"{object_hash.removeprefix('sha256:')}.json"
    )
    target_name = relative.name
    rendered = _canonical_json_bytes(payload)
    if len(rendered) > MAX_ATTEMPT_EVENT_BYTES:
        raise AuthorityAttemptError("authority object is too large")
    category_descriptor = _open_private_directory(category_dir)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(
            target_name,
            flags,
            0o600,
            dir_fd=category_descriptor,
        )
    except FileExistsError:
        target = category_dir / target_name
        try:
            existing = read_bounded_regular_file(
                target,
                maximum=MAX_ATTEMPT_EVENT_BYTES,
                label="authority_object",
            )
        except BoundedFileError as exc:
            raise AuthorityAttemptError(
                "existing authority object failed safety validation"
            ) from exc
        if existing != rendered:
            raise AuthorityAttemptError("authority object hash collision")
    except OSError as exc:
        raise AuthorityAttemptError("authority object could not be created") from exc
    else:
        try:
            view = memoryview(rendered)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short authority object write")
                view = view[written:]
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(target_name, dir_fd=category_descriptor)
            except OSError:
                pass
            raise
        else:
            os.close(descriptor)
            try:
                os.fsync(category_descriptor)
            except OSError:
                pass
    finally:
        os.close(category_descriptor)
    return object_hash, relative.as_posix()


def _open_attempt_directory(root_descriptor: int, attempt_id: str) -> int:
    if ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise AuthorityAttemptError("invalid authority attempt id")
    try:
        path_metadata = os.stat(
            attempt_id,
            dir_fd=root_descriptor,
            follow_symlinks=False,
        )
        descriptor = os.open(
            attempt_id,
            _directory_flags(),
            dir_fd=root_descriptor,
        )
        descriptor_metadata = os.fstat(descriptor)
        _validate_directory_metadata(
            path_metadata,
            descriptor_metadata,
            private=True,
        )
        return descriptor
    except AuthorityAttemptError:
        try:
            os.close(descriptor)
        except (NameError, OSError):
            pass
        raise
    except OSError as exc:
        raise AuthorityAttemptError(
            "authority attempt directory is unavailable"
        ) from exc


def _read_event_file(attempt_descriptor: int, name: str) -> dict[str, Any]:
    try:
        metadata = os.stat(name, dir_fd=attempt_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise AuthorityAttemptError("authority attempt event is unreadable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > MAX_ATTEMPT_EVENT_BYTES
    ):
        raise AuthorityAttemptError("authority attempt event is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=attempt_descriptor)
    except OSError as exc:
        raise AuthorityAttemptError("authority attempt event is unreadable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            before.st_dev != metadata.st_dev
            or before.st_ino != metadata.st_ino
            or before.st_nlink != 1
            or not stat.S_ISREG(before.st_mode)
        ):
            raise AuthorityAttemptError("authority attempt event changed")
        chunks: list[bytes] = []
        remaining = MAX_ATTEMPT_EVENT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            len(payload) > MAX_ATTEMPT_EVENT_BYTES
            or after.st_size != len(payload)
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise AuthorityAttemptError("authority attempt event changed")
    finally:
        os.close(descriptor)
    parsed = _strict_json_load(payload, label="authority attempt event")
    if not isinstance(parsed, dict):
        raise AuthorityAttemptError("authority attempt event must be an object")
    return parsed


def _event_hash(event: dict[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "event_hash"}
    return canonical_authority_hash(body)


def _load_events_from_descriptor(
    attempt_descriptor: int,
    *,
    expected_attempt_id: str,
) -> list[dict[str, Any]]:
    try:
        names = os.listdir(attempt_descriptor)
    except OSError as exc:
        raise AuthorityAttemptError(
            "authority attempt directory is unreadable"
        ) from exc
    if len(names) > MAX_ATTEMPT_EVENTS:
        raise AuthorityAttemptError("authority attempt has too many events")
    indexed: list[tuple[int, str]] = []
    for name in names:
        match = EVENT_FILE_RE.fullmatch(name)
        if match is None:
            raise AuthorityAttemptError("authority attempt has an unexpected artifact")
        indexed.append((int(match.group(1)), name))
    indexed.sort()
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    terminal_seen = False
    for expected_sequence, (sequence, name) in enumerate(indexed):
        if sequence != expected_sequence:
            raise AuthorityAttemptError("authority attempt event sequence has a gap")
        event = _read_event_file(attempt_descriptor, name)
        planned_keys = {
            "schema",
            "attempt_id",
            "sequence",
            "previous_event_hash",
            "recorded_at",
            "status",
            "spec_hash",
            "spec_ref",
            "parent_node_id",
            "role",
            "model",
            "task_kind",
            "event_hash",
        }
        terminal_keys = {
            "schema",
            "attempt_id",
            "sequence",
            "previous_event_hash",
            "recorded_at",
            "status",
            "result_hash",
            "result_ref",
            "error_type",
            "event_hash",
        }
        if set(event) != (planned_keys if sequence == 0 else terminal_keys):
            raise AuthorityAttemptError("authority attempt event fields are invalid")
        recorded_at = _bounded_text(
            event.get("recorded_at"),
            label="authority attempt timestamp",
            maximum=64,
        )
        try:
            parsed_timestamp = datetime.fromisoformat(recorded_at)
        except ValueError as exc:
            raise AuthorityAttemptError(
                "authority attempt timestamp is invalid"
            ) from exc
        if parsed_timestamp.tzinfo is None:
            raise AuthorityAttemptError("authority attempt timestamp lacks timezone")
        if (
            event.get("schema") != "xscientist.authority-attempt-event.v1"
            or event.get("attempt_id") != expected_attempt_id
            or event.get("sequence") != sequence
            or event.get("previous_event_hash") != previous_hash
            or event.get("event_hash") != _event_hash(event)
        ):
            raise AuthorityAttemptError("authority attempt hash chain is invalid")
        status = event.get("status")
        if sequence == 0 and status != "planned":
            raise AuthorityAttemptError("authority attempt does not start as planned")
        if sequence > 0 and status not in TERMINAL_ATTEMPT_STATUSES:
            raise AuthorityAttemptError("authority attempt result status is invalid")
        if sequence == 0:
            _bounded_text(event.get("role"), label="authority role", maximum=128)
            _bounded_text(event.get("model"), label="authority model", maximum=256)
            _bounded_text(
                event.get("task_kind"),
                label="authority task kind",
                maximum=128,
            )
            parent_node_id = event.get("parent_node_id")
            if parent_node_id is not None:
                _bounded_text(
                    parent_node_id,
                    label="authority parent node id",
                    maximum=128,
                )
        else:
            result_hash = event.get("result_hash")
            result_ref = event.get("result_ref")
            if (result_hash is None) != (result_ref is None):
                raise AuthorityAttemptError(
                    "authority attempt result object binding is incomplete"
                )
            if result_hash is not None:
                reference_match = RESULT_REF_RE.fullmatch(str(result_ref))
                if (
                    OBJECT_HASH_RE.fullmatch(str(result_hash)) is None
                    or reference_match is None
                    or reference_match.group(1)
                    != str(result_hash).removeprefix("sha256:")
                ):
                    raise AuthorityAttemptError(
                        "authority attempt result object binding is invalid"
                    )
            error_type = event.get("error_type")
            if error_type is not None:
                _bounded_text(
                    error_type,
                    label="authority attempt error type",
                    maximum=128,
                )
        if terminal_seen:
            raise AuthorityAttemptError("authority attempt has events after completion")
        terminal_seen = status in TERMINAL_ATTEMPT_STATUSES
        previous_hash = str(event["event_hash"])
        events.append(event)
    return events


def begin_authority_attempt(
    log_dir: str | os.PathLike[str],
    *,
    spec_hash: str,
    spec_ref: str | None,
    parent_node_id: str | None,
    role: str,
    model: str,
    task_kind: str,
) -> str | None:
    """Durably create a planned attempt before any execution-model submission."""

    if OBJECT_HASH_RE.fullmatch(str(spec_hash)) is None:
        raise AuthorityAttemptError("authority attempt spec hash is invalid")
    role_value = _bounded_text(role, label="authority role", maximum=128)
    model_value = _bounded_text(model, label="authority model", maximum=256)
    task_kind_value = _bounded_text(
        task_kind,
        label="authority task kind",
        maximum=128,
    )
    parent_value = None
    if parent_node_id is not None:
        parent_value = _bounded_text(
            parent_node_id,
            label="authority parent node id",
            maximum=128,
        )
    root = _attempt_root(log_dir, create=True)
    assert root is not None
    reference_match = OBJECT_REF_RE.fullmatch(str(spec_ref or ""))
    if reference_match is None or reference_match.group(1) != spec_hash.removeprefix(
        "sha256:"
    ):
        raise AuthorityAttemptError("authority attempt spec reference is invalid")
    root_descriptor = _open_directory(root, private=True)
    attempt_id: str | None = None
    attempt_descriptor: int | None = None
    try:
        for _ in range(32):
            candidate = f"attempt-{uuid.uuid4().hex}"
            try:
                os.mkdir(candidate, 0o700, dir_fd=root_descriptor)
            except FileExistsError:
                continue
            try:
                os.fsync(root_descriptor)
            except OSError:
                pass
            attempt_id = candidate
            attempt_descriptor = _open_attempt_directory(root_descriptor, candidate)
            break
        if attempt_id is None or attempt_descriptor is None:
            raise AuthorityAttemptError("could not allocate an authority attempt id")
        event: dict[str, Any] = {
            "schema": "xscientist.authority-attempt-event.v1",
            "attempt_id": attempt_id,
            "sequence": 0,
            "previous_event_hash": None,
            "recorded_at": _now(),
            "status": "planned",
            "spec_hash": spec_hash,
            "spec_ref": str(spec_ref) if spec_ref else None,
            "parent_node_id": parent_value,
            "role": role_value,
            "model": model_value,
            "task_kind": task_kind_value,
        }
        event["event_hash"] = _event_hash(event)
        _write_event_exclusive(attempt_descriptor, event)
        return attempt_id
    finally:
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        os.close(root_descriptor)


def _write_event_exclusive(
    attempt_descriptor: int,
    event: dict[str, Any],
) -> None:
    sequence = event.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise AuthorityAttemptError("authority attempt event sequence is invalid")
    name = f"{sequence}.json"
    rendered = _canonical_json_bytes(event)
    if len(rendered) > MAX_ATTEMPT_EVENT_BYTES:
        raise AuthorityAttemptError("authority attempt event is too large")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=attempt_descriptor)
    except FileExistsError:
        raise
    except OSError as exc:
        raise AuthorityAttemptError(
            "authority attempt event could not be created"
        ) from exc
    try:
        view = memoryview(rendered)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short authority attempt event write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            os.unlink(name, dir_fd=attempt_descriptor)
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
        try:
            os.fsync(attempt_descriptor)
        except OSError:
            pass


def record_authority_attempt_result(
    log_dir: str | os.PathLike[str],
    attempt_id: str | None,
    *,
    status: str,
    result_ref: str | None = None,
    result_payload: Any = _RESULT_UNSET,
    error_type: str | None = None,
) -> str | None:
    """Append exactly one terminal result without replacing the planned event."""

    if attempt_id is None:
        return None
    if status not in TERMINAL_ATTEMPT_STATUSES:
        raise AuthorityAttemptError("authority attempt terminal status is invalid")
    if result_ref is not None and OBJECT_HASH_RE.fullmatch(str(result_ref)) is None:
        raise AuthorityAttemptError("authority attempt result reference is invalid")
    if result_ref is not None and result_payload is not _RESULT_UNSET:
        raise AuthorityAttemptError(
            "authority attempt result must use one payload source"
        )
    result_hash: str | None = None
    result_object_ref: str | None = None
    if result_payload is not _RESULT_UNSET or result_ref is not None:
        result_object = {
            "schema": "xscientist.authority-attempt-result.v1",
            "result": None if result_payload is _RESULT_UNSET else result_payload,
            "declared_content_hash": str(result_ref) if result_ref else None,
        }
        result_hash, result_object_ref = persist_authority_object(
            log_dir,
            category="attempt-result",
            payload=result_object,
        )
    error_type_value = None
    if error_type is not None:
        error_type_value = _bounded_text(
            error_type,
            label="authority attempt error type",
            maximum=128,
        )
    root = _attempt_root(log_dir, create=False)
    if root is None:
        raise AuthorityAttemptError("authority attempt root is missing")
    root_descriptor = _open_directory(root, private=True)
    attempt_descriptor = _open_attempt_directory(root_descriptor, attempt_id)
    try:
        for _ in range(32):
            events = _load_events_from_descriptor(
                attempt_descriptor,
                expected_attempt_id=attempt_id,
            )
            if not events:
                raise AuthorityAttemptError("authority attempt is orphaned")
            if events[-1]["status"] in TERMINAL_ATTEMPT_STATUSES:
                if (
                    events[-1]["status"] == status
                    and events[-1].get("result_hash") == result_hash
                    and events[-1].get("result_ref") == result_object_ref
                    and events[-1].get("error_type") == error_type_value
                ):
                    return str(events[-1]["event_hash"])
                raise AuthorityAttemptError("authority attempt is already complete")
            event: dict[str, Any] = {
                "schema": "xscientist.authority-attempt-event.v1",
                "attempt_id": attempt_id,
                "sequence": len(events),
                "previous_event_hash": events[-1]["event_hash"],
                "recorded_at": _now(),
                "status": status,
                "result_hash": result_hash,
                "result_ref": result_object_ref,
                "error_type": error_type_value,
            }
            event["event_hash"] = _event_hash(event)
            try:
                _write_event_exclusive(attempt_descriptor, event)
            except FileExistsError:
                continue
            return str(event["event_hash"])
        raise AuthorityAttemptError("authority attempt event append did not converge")
    finally:
        os.close(attempt_descriptor)
        os.close(root_descriptor)


def run_authority_call(
    log_dir: str | os.PathLike[str],
    *,
    category: str,
    specification: dict[str, Any],
    parent_node_id: str | None,
    role: str,
    model: str,
    task_kind: str,
    operation: Callable[[], _T],
) -> tuple[_T, str | None, str | None]:
    """Run one judgment call with a durable planned/terminal receipt pair."""

    spec_hash, spec_ref = persist_authority_object(
        log_dir,
        category=category,
        payload=specification,
    )
    attempt_id = begin_authority_attempt(
        log_dir,
        spec_hash=spec_hash,
        spec_ref=spec_ref,
        parent_node_id=parent_node_id,
        role=role,
        model=model,
        task_kind=task_kind,
    )
    try:
        result = operation()
    except BaseException as exc:
        record_authority_attempt_result(
            log_dir,
            attempt_id,
            status="timeout" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__,
        )
        raise
    terminal_hash = record_authority_attempt_result(
        log_dir,
        attempt_id,
        status="accepted",
        result_payload=result,
    )
    return result, attempt_id, terminal_hash


def inspect_authority_attempts(
    log_dir: str | os.PathLike[str],
    *,
    expected_attempt_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Enumerate valid, incomplete, orphaned, and tampered attempt chains."""

    expected = sorted(set(expected_attempt_ids))
    if any(ATTEMPT_ID_RE.fullmatch(item) is None for item in expected):
        raise AuthorityAttemptError("expected authority attempt id is invalid")
    root = _attempt_root(log_dir, create=False)
    if root is None:
        return {
            "schema": "xscientist.authority-attempt-audit.v1",
            "attempts": [],
            "valid": False,
            "incomplete_attempt_ids": [],
            "orphan_attempt_ids": [],
            "invalid_attempt_ids": [],
            "missing_expected_attempt_ids": expected,
            "terminal_event_hashes": {},
            "errors": ["attempt_root_missing"],
            "expected_valid": False,
        }
    root_descriptor = _open_directory(root, private=True)
    rows: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir(root_descriptor))
        if len(names) > 100_000:
            raise AuthorityAttemptError("authority attempt root is too large")
        for name in names:
            row: dict[str, Any] = {
                "attempt_id": name,
                "valid": False,
                "incomplete": False,
                "orphan": False,
                "status": None,
                "event_count": 0,
                "errors": [],
            }
            if ATTEMPT_ID_RE.fullmatch(name) is None:
                row["errors"] = ["unexpected_attempt_artifact"]
                rows.append(row)
                continue
            try:
                attempt_descriptor = _open_attempt_directory(root_descriptor, name)
                try:
                    events = _load_events_from_descriptor(
                        attempt_descriptor,
                        expected_attempt_id=name,
                    )
                finally:
                    os.close(attempt_descriptor)
                row["event_count"] = len(events)
                if not events:
                    row["orphan"] = True
                    row["errors"] = ["planned_event_missing"]
                else:
                    spec_hash = events[0].get("spec_hash")
                    spec_ref = events[0].get("spec_ref")
                    reference_match = OBJECT_REF_RE.fullmatch(str(spec_ref or ""))
                    if (
                        OBJECT_HASH_RE.fullmatch(str(spec_hash or "")) is None
                        or reference_match is None
                        or reference_match.group(1)
                        != str(spec_hash).removeprefix("sha256:")
                    ):
                        raise AuthorityAttemptError(
                            "authority attempt spec reference is invalid"
                        )
                    spec_path = _normalized_log_dir(log_dir) / str(spec_ref)
                    try:
                        spec_payload = read_bounded_regular_file(
                            spec_path,
                            maximum=MAX_ATTEMPT_EVENT_BYTES,
                            label="authority_attempt_spec",
                        )
                        parsed_spec = _strict_json_load(
                            spec_payload,
                            label="authority attempt spec",
                        )
                    except (
                        BoundedFileError,
                        AuthorityAttemptError,
                    ) as exc:
                        raise AuthorityAttemptError(
                            "authority attempt spec is unavailable"
                        ) from exc
                    if canonical_authority_hash(parsed_spec) != spec_hash:
                        raise AuthorityAttemptError(
                            "authority attempt spec hash is invalid"
                        )
                    for terminal_event in events[1:]:
                        result_hash = terminal_event.get("result_hash")
                        result_ref = terminal_event.get("result_ref")
                        if result_hash is None:
                            continue
                        result_path = _normalized_log_dir(log_dir) / str(result_ref)
                        try:
                            result_payload = read_bounded_regular_file(
                                result_path,
                                maximum=MAX_ATTEMPT_EVENT_BYTES,
                                label="authority_attempt_result",
                            )
                            parsed_result = _strict_json_load(
                                result_payload,
                                label="authority attempt result",
                            )
                        except (BoundedFileError, AuthorityAttemptError) as exc:
                            raise AuthorityAttemptError(
                                "authority attempt result is unavailable"
                            ) from exc
                        if canonical_authority_hash(parsed_result) != result_hash:
                            raise AuthorityAttemptError(
                                "authority attempt result hash is invalid"
                            )
                        if (
                            not isinstance(parsed_result, dict)
                            or set(parsed_result)
                            != {"schema", "result", "declared_content_hash"}
                            or parsed_result.get("schema")
                            != "xscientist.authority-attempt-result.v1"
                            or (
                                parsed_result.get("declared_content_hash") is not None
                                and OBJECT_HASH_RE.fullmatch(
                                    str(parsed_result.get("declared_content_hash"))
                                )
                                is None
                            )
                        ):
                            raise AuthorityAttemptError(
                                "authority attempt result object is invalid"
                            )
                    row["valid"] = True
                    row["status"] = events[-1]["status"]
                    row["incomplete"] = events[-1]["status"] == "planned"
                    row["spec_hash"] = spec_hash
                    row["spec_ref"] = spec_ref
                    row["parent_node_id"] = events[0].get("parent_node_id")
                    row["role"] = events[0].get("role")
                    row["model"] = events[0].get("model")
                    row["task_kind"] = events[0].get("task_kind")
                    row["event_hashes"] = [event["event_hash"] for event in events]
                    row["terminal_event_hash"] = (
                        events[-1]["event_hash"]
                        if events[-1]["status"] in TERMINAL_ATTEMPT_STATUSES
                        else None
                    )
            except AuthorityAttemptError as exc:
                row["errors"] = [str(exc)]
            rows.append(row)
    finally:
        os.close(root_descriptor)
    observed_ids = {
        str(row["attempt_id"])
        for row in rows
        if ATTEMPT_ID_RE.fullmatch(str(row["attempt_id"])) is not None
    }
    missing_expected = sorted(set(expected) - observed_ids)
    incomplete = [row["attempt_id"] for row in rows if row["incomplete"]]
    orphaned = [row["attempt_id"] for row in rows if row["orphan"]]
    invalid = [
        row["attempt_id"] for row in rows if not row["valid"] and not row["orphan"]
    ]
    errors: list[str] = []
    if not rows:
        errors.append("attempt_ledger_empty")
    if missing_expected:
        errors.append("expected_attempt_missing")
    if incomplete:
        errors.append("attempt_incomplete")
    if orphaned:
        errors.append("attempt_orphaned")
    if invalid:
        errors.append("attempt_invalid")
    expected_rows = [row for row in rows if row["attempt_id"] in set(expected)]
    expected_valid = (
        not missing_expected
        and len(expected_rows) == len(expected)
        and all(
            bool(row["valid"])
            and not bool(row["incomplete"])
            and not bool(row["orphan"])
            for row in expected_rows
        )
    )
    return {
        "schema": "xscientist.authority-attempt-audit.v1",
        "attempts": rows,
        "valid": not errors,
        "incomplete_attempt_ids": incomplete,
        "orphan_attempt_ids": orphaned,
        "invalid_attempt_ids": invalid,
        "missing_expected_attempt_ids": missing_expected,
        "terminal_event_hashes": {
            str(row["attempt_id"]): str(row["terminal_event_hash"])
            for row in rows
            if row.get("terminal_event_hash")
        },
        "errors": errors,
        "expected_valid": expected_valid,
    }


__all__ = [
    "ATTEMPT_ID_RE",
    "AuthorityAttemptError",
    "begin_authority_attempt",
    "canonical_authority_hash",
    "inspect_authority_attempts",
    "persist_authority_object",
    "record_authority_attempt_result",
    "run_authority_call",
]
