"""Crash-durable runtime truth for required experiment admission.

Research VCS commits are the final scientific authority.  This module covers
the interval between consuming an admitted start and publishing that terminal
commit.  Claims and terminal outboxes live in Git's common directory so linked
worktrees observe one immutable runtime state.
"""

from __future__ import annotations

import errno
import hashlib
import json
import mimetypes
import os
import re
import socket
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ai_scientist.protocol.canonical_json import (
    canonical_content_hash,
    canonical_json_bytes,
)

from .research_git import (
    ResearchGitError,
    _repository_common_git_dir,
    _repository_root,
)
from .run_control import _process_identity, _process_identity_matches

CLAIM_SCHEMA = "xscientist.experiment-admission-execution-claim.v2"
BFTS_TERMINAL_SCHEMA = "xscientist.experiment-admission-bfts-terminal.v1"
WORKER_TERMINAL_SCHEMA = "xscientist.experiment-admission-worker-terminal.v1"

_CLAIM_DIRECTORY = "admission-claims"
_BFTS_DIRECTORY = "admission-bfts-terminals"
_WORKER_DIRECTORY = "admission-worker-terminals"
_MAX_MARKER_BYTES = 1 << 20
_MAX_MARKERS = 4096
_MAX_ARTIFACTS = 128
_MAX_ARTIFACT_BYTES = 1 << 30
_LINUX_BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
_TRUSTED_SYSCTL_EXECUTABLES = ("/usr/sbin/sysctl", "/sbin/sysctl")
_START_ID_RE = re.compile(r"rso-[0-9a-f]{16}\Z")
_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MARKER_RE = re.compile(r"(?P<start_id>rso-[0-9a-f]{16})\.json\Z")
_TERMINAL_STATUSES = frozenset(
    {"completed", "failed", "budget_exhausted", "interrupted"}
)
_PIPELINE_STATUSES = frozenset({"success", "failed"})
_EXPERIMENT_STATUSES = frozenset(
    {"completed", "failed", "timed_out", "cancelled", "rejected"}
)
_RESULT_SCALAR_FIELDS = frozenset(
    {
        "idea_idx",
        "status",
        "runtime_status",
        "registry_status",
        "terminal_receipt_hash",
        "stage",
        "failure_stage",
        "resumable",
        "quality_score",
        "rigor_score",
        "claim_support_score",
        "quality_gate_passed",
        "submission_priority_score",
        "submission_priority_tier",
        "submission_acceptance_passed",
        "guardrail_blocking",
        "ara_node_count",
        "ara_claim_coverage_score",
        "ara_claim_coverage_severity",
        "workflow_mode",
        "template_profile",
        "template_capability",
        "execution_policy",
        "admission_post_execution_status",
        "research_status",
        "pause_reason",
    }
)
_RESULT_PATH_FIELDS = frozenset(
    {
        "exp_dir",
        "checkpoint_path",
        "run_status_path",
        "manager_state_path",
        "initialization_status_path",
        "pdf_path",
        "ara_dir",
        "ara_manifest",
        "pipeline_manifest",
        "research_plan_file",
        "claim_evidence_graph_file",
        "experiment_registry_file",
        "figure_spec_file",
        "plot_execution_receipt_file",
        "manuscript_state_file",
        "review_state_file",
        "repair_plan_file",
        "self_evolution_file",
        "stage_standards_file",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checked_directory(
    repo: str | Path,
    directory_name: str,
    *,
    create: bool,
) -> Path:
    root = _repository_root(repo)
    common = _repository_common_git_dir(root)
    state = common / "xscientist"
    destination = state / directory_name
    for path in (state, destination):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if not create:
                return destination
            path.mkdir(mode=0o700)
            _strict_fsync_directory(path.parent)
            metadata = path.lstat()
        except OSError as exc:
            raise ResearchGitError(
                "experiment admission runtime directory is unreadable"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ResearchGitError("experiment admission runtime directory is unsafe")
    return destination


def _marker_path(
    repo: str | Path,
    directory_name: str,
    start_object_id: str,
    *,
    create: bool,
) -> Path:
    if not isinstance(start_object_id, str) or not _START_ID_RE.fullmatch(
        start_object_id
    ):
        raise ResearchGitError("experiment admission runtime start id is invalid")
    return _checked_directory(repo, directory_name, create=create) / (
        f"{start_object_id}.json"
    )


def _decode_json_marker(path: Path, *, label: str) -> dict[str, Any] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ResearchGitError(f"experiment admission {label} is unreadable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > _MAX_MARKER_BYTES
    ):
        raise ResearchGitError(f"experiment admission {label} is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResearchGitError(f"experiment admission {label} is unreadable") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or opened.st_size > _MAX_MARKER_BYTES
        ):
            raise ResearchGitError(f"experiment admission {label} changed during read")
        chunks: list[bytes] = []
        remaining = _MAX_MARKER_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or getattr(after, "st_mtime_ns", None)
            != getattr(opened, "st_mtime_ns", None)
            or getattr(after, "st_ctime_ns", None)
            != getattr(opened, "st_ctime_ns", None)
        ):
            raise ResearchGitError(f"experiment admission {label} changed during read")
    finally:
        os.close(descriptor)
    if len(encoded) > _MAX_MARKER_BYTES:
        raise ResearchGitError(f"experiment admission {label} is too large")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in pairs:
            if key in decoded:
                raise ValueError("duplicate key")
            decoded[key] = value
        return decoded

    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResearchGitError(f"experiment admission {label} is invalid JSON") from exc
    if not isinstance(value, dict) or encoded != canonical_json_bytes(value) + b"\n":
        raise ResearchGitError(f"experiment admission {label} is not canonical JSON")
    return value


def _publish_immutable(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    normalized = dict(payload)
    encoded = canonical_json_bytes(normalized) + b"\n"
    if len(encoded) > _MAX_MARKER_BYTES:
        raise ResearchGitError(f"experiment admission {label} is too large")
    existing = _decode_json_marker(path, label=label)
    if existing is not None:
        if existing != normalized:
            raise ResearchGitError(f"experiment admission {label} conflicts")
        return existing
    descriptor = -1
    temporary_name = ""
    published = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
        )
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("admission runtime marker write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary_name, path)
            published = True
        except FileExistsError:
            raced = _decode_json_marker(path, label=label)
            if raced != normalized:
                raise ResearchGitError(f"experiment admission {label} conflicts")
        _strict_fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            else:
                _strict_fsync_directory(path.parent)
    read_back = _decode_json_marker(path, label=label)
    if read_back != normalized:
        qualifier = "published" if published else "existing"
        raise ResearchGitError(
            f"experiment admission {qualifier} {label} failed read-back validation"
        )
    return read_back


def _validate_timestamp(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ResearchGitError(f"experiment admission {label} timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchGitError(
            f"experiment admission {label} timestamp is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResearchGitError(f"experiment admission {label} timestamp is invalid")


def _boot_id() -> str | None:
    if os.name != "posix":  # pragma: no cover - no portable Windows boot identity
        return None
    try:
        value = _LINUX_BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    if value:
        return hashlib.sha256(value.encode("ascii")).hexdigest()
    for executable in _TRUSTED_SYSCTL_EXECUTABLES:
        try:
            completed = subprocess.run(
                [executable, "-n", "kern.boottime"],
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError:
            continue
        value = completed.stdout.strip()
        if completed.returncode == 0 and value:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return None


def _owner_identity(*, owner_nonce: str | None = None) -> dict[str, Any]:
    nonce = owner_nonce or os.urandom(16).hex()
    boot_id = _boot_id()
    process_identity = _process_identity(os.getpid())
    if not _NONCE_RE.fullmatch(nonce):
        raise ResearchGitError("experiment admission owner nonce is invalid")
    if not boot_id or not re.fullmatch(r"[0-9a-f]{64}", boot_id):
        raise ResearchGitError("experiment admission host boot identity is unavailable")
    if not process_identity or not re.fullmatch(r"[0-9a-f]{64}", process_identity):
        raise ResearchGitError("experiment admission process identity is unavailable")
    return {
        "host_id": socket.gethostname(),
        "boot_id": boot_id,
        "pid": os.getpid(),
        "process_identity": process_identity,
        "owner_nonce": nonce,
        "started_at": _now_iso(),
    }


def _validate_relative_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ResearchGitError(f"experiment admission {label} path is invalid")
    path = Path(value)
    if path.is_absolute() or value != path.as_posix() or ".." in path.parts:
        raise ResearchGitError(f"experiment admission {label} path is invalid")
    return value


def _project_relative_path(
    project_root: Path,
    value: str | Path,
    *,
    label: str,
    require_exists: bool,
) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(project_root)
    except ValueError as exc:
        raise ResearchGitError(
            f"experiment admission {label} leaves the project"
        ) from exc
    cursor = project_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ResearchGitError(f"experiment admission {label} contains a symlink")
    if require_exists and not lexical.exists():
        raise ResearchGitError(f"experiment admission {label} does not exist")
    return _validate_relative_path(relative.as_posix(), label=label)


def _validate_owner(owner: Any) -> dict[str, Any]:
    if not isinstance(owner, Mapping) or set(owner) != {
        "host_id",
        "boot_id",
        "pid",
        "process_identity",
        "owner_nonce",
        "started_at",
    }:
        raise ResearchGitError("experiment admission claim owner is invalid")
    normalized = dict(owner)
    if (
        not isinstance(normalized["host_id"], str)
        or not normalized["host_id"]
        or len(normalized["host_id"]) > 255
        or not re.fullmatch(r"[0-9a-f]{64}", str(normalized["boot_id"]))
        or type(normalized["pid"]) is not int
        or normalized["pid"] <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", str(normalized["process_identity"]))
        or not _NONCE_RE.fullmatch(str(normalized["owner_nonce"]))
    ):
        raise ResearchGitError("experiment admission claim owner is invalid")
    _validate_timestamp(normalized["started_at"], label="claim owner")
    return normalized


def validate_execution_claim_v2(
    claim: Mapping[str, Any],
    *,
    start_object_id: str | None = None,
    receipt_hash: str | None = None,
) -> dict[str, Any]:
    if not isinstance(claim, Mapping):
        raise ResearchGitError("experiment admission execution claim v2 is invalid")
    normalized = dict(claim)
    expected = {
        "schema",
        "repository_id",
        "start_object_id",
        "start_object_hash",
        "receipt_hash",
        "start_checkpoint_commit",
        "start_checkpoint_id",
        "start_checkpoint_hash",
        "execution_scope_hash",
        "ideas_hash",
        "configuration_hash",
        "budget_hash",
        "idea_index",
        "idea_key",
        "project_workspace",
        "experiment_dir",
        "batch_id",
        "run_nonce",
        "worker_nonce",
        "bfts_run_nonce",
        "outbox_slots",
        "owner",
        "claimed_at",
        "claim_hash",
    }
    core = {key: value for key, value in normalized.items() if key != "claim_hash"}
    scope_hashes = (
        normalized.get("ideas_hash"),
        normalized.get("configuration_hash"),
        normalized.get("budget_hash"),
    )
    if (
        set(normalized) != expected
        or normalized.get("schema") != CLAIM_SCHEMA
        or not isinstance(normalized.get("repository_id"), str)
        or not normalized["repository_id"]
        or not _START_ID_RE.fullmatch(str(normalized.get("start_object_id") or ""))
        or not _HASH_RE.fullmatch(str(normalized.get("start_object_hash") or ""))
        or not _HASH_RE.fullmatch(str(normalized.get("receipt_hash") or ""))
        or not _COMMIT_RE.fullmatch(
            str(normalized.get("start_checkpoint_commit") or "")
        )
        or not _SAFE_NAME_RE.fullmatch(str(normalized.get("start_checkpoint_id") or ""))
        or not _HASH_RE.fullmatch(str(normalized.get("start_checkpoint_hash") or ""))
        or not _HASH_RE.fullmatch(str(normalized.get("execution_scope_hash") or ""))
        or any(
            value is not None and not _HASH_RE.fullmatch(str(value))
            for value in scope_hashes
        )
        or type(normalized.get("idea_index")) is not int
        or normalized["idea_index"] < 0
        or not _SAFE_NAME_RE.fullmatch(str(normalized.get("idea_key") or ""))
        or normalized.get("project_workspace") != "."
        or not _HASH_RE.fullmatch(str(normalized.get("batch_id") or ""))
        or any(
            not _NONCE_RE.fullmatch(str(normalized.get(field) or ""))
            for field in ("run_nonce", "worker_nonce", "bfts_run_nonce")
        )
        or normalized.get("claim_hash") != canonical_content_hash(core)
    ):
        raise ResearchGitError("experiment admission execution claim v2 is invalid")
    _validate_relative_path(normalized["experiment_dir"], label="experiment directory")
    slots = normalized.get("outbox_slots")
    expected_slots = {
        "bfts": f"{_BFTS_DIRECTORY}/{normalized['start_object_id']}.json",
        "worker": f"{_WORKER_DIRECTORY}/{normalized['start_object_id']}.json",
    }
    if slots != expected_slots:
        raise ResearchGitError(
            "experiment admission execution claim outbox slots are invalid"
        )
    _validate_owner(normalized["owner"])
    _validate_timestamp(normalized["claimed_at"], label="execution claim")
    if start_object_id is not None and normalized["start_object_id"] != start_object_id:
        raise ResearchGitError(
            "experiment admission execution claim start id does not match"
        )
    if receipt_hash is not None and normalized["receipt_hash"] != receipt_hash:
        raise ResearchGitError(
            "experiment admission execution claim receipt hash does not match"
        )
    return normalized


def load_execution_claim_v2(
    repo: str | Path,
    *,
    start_object_id: str,
    receipt_hash: str | None = None,
) -> dict[str, Any] | None:
    path = _marker_path(repo, _CLAIM_DIRECTORY, start_object_id, create=False)
    marker = _decode_json_marker(path, label="execution claim v2")
    if marker is None:
        return None
    return validate_execution_claim_v2(
        marker, start_object_id=start_object_id, receipt_hash=receipt_hash
    )


def claim_experiment_admission_start_v2(
    repo: str | Path,
    receipt: Mapping[str, Any],
    *,
    experiment_dir: str | Path,
    idea_key: str,
    run_nonce: str | None = None,
    worker_nonce: str | None = None,
    bfts_run_nonce: str | None = None,
    owner_nonce: str | None = None,
) -> dict[str, Any]:
    root = _repository_root(repo)
    # The runtime marker is not an alternate admission path.  Re-prove the
    # committed start and its signed policy snapshot before consuming it.
    from .research_lifecycle import ResearchLifecycle

    finalized = ResearchLifecycle(root).validate_finalized_experiment_admission_start(
        receipt
    )
    finalized = dict(finalized)
    receipt_hash = str(finalized.pop("receipt_hash", ""))
    scope = finalized.get("execution_scope")
    if (
        finalized.get("protocol_kind") != "xscientist.experiment-admission-preflight.v2"
        or finalized.get("status") != "finalized"
        or not _HASH_RE.fullmatch(receipt_hash)
        or canonical_content_hash(finalized) != receipt_hash
        or not isinstance(scope, Mapping)
        or canonical_content_hash(dict(scope)) != finalized.get("execution_scope_hash")
    ):
        raise ResearchGitError("experiment admission finalized receipt is invalid")
    start_object_id = str(finalized.get("start_object_id") or "")
    existing = load_execution_claim_v2(
        root, start_object_id=start_object_id, receipt_hash=receipt_hash
    )
    if existing is not None:
        raise ResearchGitError(
            "experiment admission start has already been claimed for execution"
        )
    idea_index = scope.get("idea_index")
    if type(idea_index) is not int or idea_index < 0:
        raise ResearchGitError("experiment admission claim idea index is invalid")
    relative_experiment_dir = _project_relative_path(
        root,
        experiment_dir,
        label="experiment directory",
        require_exists=True,
    )
    nonces = {
        "run_nonce": run_nonce or os.urandom(16).hex(),
        "worker_nonce": worker_nonce or os.urandom(16).hex(),
        "bfts_run_nonce": bfts_run_nonce or os.urandom(16).hex(),
    }
    owner = _owner_identity(owner_nonce=owner_nonce)
    core = {
        "schema": CLAIM_SCHEMA,
        "repository_id": str(finalized.get("repository_id") or ""),
        "start_object_id": start_object_id,
        "start_object_hash": str(finalized.get("start_object_hash") or ""),
        "receipt_hash": receipt_hash,
        "start_checkpoint_commit": str(finalized.get("start_checkpoint") or ""),
        "start_checkpoint_id": str(finalized.get("start_checkpoint_id") or ""),
        "start_checkpoint_hash": str(finalized.get("start_checkpoint_hash") or ""),
        "execution_scope_hash": str(finalized.get("execution_scope_hash") or ""),
        "ideas_hash": scope.get("ideas_hash"),
        "configuration_hash": scope.get("configuration_hash"),
        "budget_hash": scope.get("budget_hash"),
        "idea_index": idea_index,
        "idea_key": str(idea_key),
        "project_workspace": ".",
        "experiment_dir": relative_experiment_dir,
        "batch_id": canonical_content_hash(
            {
                "repository_id": finalized.get("repository_id"),
                "start_checkpoint": finalized.get("start_checkpoint"),
            }
        ),
        **nonces,
        "outbox_slots": {
            "bfts": f"{_BFTS_DIRECTORY}/{start_object_id}.json",
            "worker": f"{_WORKER_DIRECTORY}/{start_object_id}.json",
        },
        "owner": owner,
        "claimed_at": owner["started_at"],
    }
    payload = {**core, "claim_hash": canonical_content_hash(core)}
    validate_execution_claim_v2(payload)
    path = _marker_path(root, _CLAIM_DIRECTORY, start_object_id, create=True)
    try:
        return _publish_immutable(path, payload, label="execution claim v2")
    except ResearchGitError:
        existing = load_execution_claim_v2(
            root, start_object_id=start_object_id, receipt_hash=receipt_hash
        )
        if existing == payload:
            return existing
        raise


def execution_claim_owner_state(claim: Mapping[str, Any]) -> str:
    normalized = validate_execution_claim_v2(claim)
    owner = normalized["owner"]
    if owner["host_id"] != socket.gethostname():
        return "unknown"
    current_boot = _boot_id()
    if current_boot is None:
        return "unknown"
    if current_boot != owner["boot_id"]:
        return "dead"
    pid = int(owner["pid"])
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return "dead"
    except PermissionError:
        return "unknown"
    except OSError as exc:
        return "dead" if exc.errno == errno.ESRCH else "unknown"
    identity_match = _process_identity_matches(pid, owner["process_identity"])
    if identity_match is None:
        return "unknown"
    return "alive" if identity_match else "dead"


def _artifact_manifest(
    project_root: Path,
    paths: Iterable[str | Path | None],
) -> list[dict[str, Any]]:
    unique: dict[str, Path] = {}
    for raw in paths:
        if raw is None or not str(raw).strip():
            continue
        relative = _project_relative_path(
            project_root, raw, label="artifact", require_exists=True
        )
        candidate = project_root / relative
        if candidate.is_dir():
            continue
        unique[relative] = candidate
        if len(unique) > _MAX_ARTIFACTS:
            raise ResearchGitError(
                "experiment admission terminal has too many artifacts"
            )
    rows: list[dict[str, Any]] = []
    for relative, path in sorted(unique.items()):
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise ResearchGitError("experiment admission terminal artifact is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) != (metadata.st_dev, metadata.st_ino, metadata.st_size):
                raise ResearchGitError(
                    "experiment admission terminal artifact changed during hashing"
                )
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (after.st_dev, after.st_ino, after.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ) or getattr(after, "st_mtime_ns", None) != getattr(
                opened, "st_mtime_ns", None
            ):
                raise ResearchGitError(
                    "experiment admission terminal artifact changed during hashing"
                )
        finally:
            os.close(descriptor)
        rows.append(
            {
                "path": relative,
                "sha256": f"sha256:{digest.hexdigest()}",
                "size": metadata.st_size,
                "media_type": mimetypes.guess_type(path.name)[0]
                or "application/octet-stream",
            }
        )
    return rows


def _validate_artifacts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > _MAX_ARTIFACTS:
        raise ResearchGitError("experiment admission terminal artifacts are invalid")
    paths: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "size",
            "media_type",
        }:
            raise ResearchGitError("experiment admission terminal artifact is invalid")
        item = dict(row)
        path = _validate_relative_path(item["path"], label="artifact")
        if (
            path in paths
            or not _HASH_RE.fullmatch(str(item["sha256"]))
            or type(item["size"]) is not int
            or item["size"] < 0
            or item["size"] > _MAX_ARTIFACT_BYTES
            or not isinstance(item["media_type"], str)
            or not item["media_type"]
            or len(item["media_type"]) > 255
        ):
            raise ResearchGitError("experiment admission terminal artifact is invalid")
        paths.add(path)
        normalized.append(item)
    if [row["path"] for row in normalized] != sorted(paths):
        raise ResearchGitError(
            "experiment admission terminal artifacts are not ordered"
        )
    return normalized


def verify_terminal_artifacts(
    repo: str | Path,
    terminal: Mapping[str, Any],
) -> None:
    root = _repository_root(repo)
    artifacts = _validate_artifacts(terminal.get("artifacts"))
    current = _artifact_manifest(root, [row["path"] for row in artifacts])
    if current != artifacts:
        raise ResearchGitError("experiment admission terminal artifact changed")


def _terminal_identity(claim: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_execution_claim_v2(claim)
    return {
        "start_object_id": normalized["start_object_id"],
        "receipt_hash": normalized["receipt_hash"],
        "claim_hash": normalized["claim_hash"],
        "run_nonce": normalized["run_nonce"],
        "worker_nonce": normalized["worker_nonce"],
        "bfts_run_nonce": normalized["bfts_run_nonce"],
        "idea_index": normalized["idea_index"],
        "experiment_dir": normalized["experiment_dir"],
    }


def _validate_failure(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"code", "exception_type"}:
        raise ResearchGitError("experiment admission terminal failure is invalid")
    code = str(value.get("code") or "")
    exception_type = str(value.get("exception_type") or "")
    if not _SAFE_NAME_RE.fullmatch(code) or not _SAFE_NAME_RE.fullmatch(exception_type):
        raise ResearchGitError("experiment admission terminal failure is invalid")
    return {"code": code, "exception_type": exception_type}


def _failure_projection(value: Any, *, fallback_code: str) -> dict[str, str] | None:
    if value is None:
        return None
    exception_type = "UnknownError"
    code = fallback_code
    if isinstance(value, BaseException):
        exception_type = type(value).__name__
    elif isinstance(value, Mapping):
        exception_type = str(
            value.get("type") or value.get("exception_type") or exception_type
        )
        code = str(value.get("code") or value.get("error_code") or code)
    code = re.sub(r"[^A-Za-z0-9._-]+", "_", code)[:128] or fallback_code
    exception_type = (
        re.sub(r"[^A-Za-z0-9._-]+", "_", exception_type)[:128] or "UnknownError"
    )
    return {"code": code, "exception_type": exception_type}


def validate_bfts_terminal(
    terminal: Mapping[str, Any],
    *,
    claim: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(terminal, Mapping):
        raise ResearchGitError("experiment admission BFTS terminal is invalid")
    normalized = dict(terminal)
    expected = {
        "schema",
        "start_object_id",
        "receipt_hash",
        "claim_hash",
        "run_nonce",
        "worker_nonce",
        "bfts_run_nonce",
        "idea_index",
        "experiment_dir",
        "status",
        "stage",
        "resumable",
        "failure",
        "artifacts",
        "finished_at",
        "terminal_hash",
    }
    core = {key: value for key, value in normalized.items() if key != "terminal_hash"}
    if (
        set(normalized) != expected
        or normalized.get("schema") != BFTS_TERMINAL_SCHEMA
        or normalized.get("status") not in _TERMINAL_STATUSES
        or not _SAFE_NAME_RE.fullmatch(str(normalized.get("stage") or ""))
        or type(normalized.get("resumable")) is not bool
        or normalized.get("terminal_hash") != canonical_content_hash(core)
    ):
        raise ResearchGitError("experiment admission BFTS terminal is invalid")
    _validate_failure(normalized["failure"])
    _validate_artifacts(normalized["artifacts"])
    _validate_timestamp(normalized["finished_at"], label="BFTS terminal")
    _validate_relative_path(normalized["experiment_dir"], label="experiment directory")
    if claim is not None:
        identity = _terminal_identity(claim)
        if any(normalized.get(key) != value for key, value in identity.items()):
            raise ResearchGitError(
                "experiment admission BFTS terminal identity conflicts"
            )
    return normalized


def publish_bfts_terminal(
    repo: str | Path,
    claim: Mapping[str, Any],
    *,
    status: str,
    stage: str,
    resumable: bool,
    failure: Any = None,
    artifact_paths: Iterable[str | Path | None] = (),
) -> dict[str, Any]:
    root = _repository_root(repo)
    identity = _terminal_identity(claim)
    existing = load_bfts_terminal(
        root, start_object_id=identity["start_object_id"], claim=claim
    )
    normalized_status = str(status or "").strip().lower()
    if normalized_status not in _TERMINAL_STATUSES:
        raise ResearchGitError("experiment admission BFTS status is invalid")
    core = {
        "schema": BFTS_TERMINAL_SCHEMA,
        **identity,
        "status": normalized_status,
        "stage": str(stage or "bfts"),
        "resumable": bool(resumable),
        "failure": _failure_projection(failure, fallback_code="bfts_failed"),
        "artifacts": _artifact_manifest(root, artifact_paths),
        "finished_at": existing["finished_at"] if existing is not None else _now_iso(),
    }
    payload = {**core, "terminal_hash": canonical_content_hash(core)}
    validate_bfts_terminal(payload, claim=claim)
    path = _marker_path(root, _BFTS_DIRECTORY, identity["start_object_id"], create=True)
    return _publish_immutable(path, payload, label="BFTS terminal")


def load_bfts_terminal(
    repo: str | Path,
    *,
    start_object_id: str,
    claim: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _marker_path(repo, _BFTS_DIRECTORY, start_object_id, create=False)
    marker = _decode_json_marker(path, label="BFTS terminal")
    return validate_bfts_terminal(marker, claim=claim) if marker is not None else None


def _portable_result(
    project_root: Path,
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    portable: dict[str, Any] = {}
    artifact_paths: list[str] = []
    for field in sorted(_RESULT_SCALAR_FIELDS):
        value = result.get(field)
        if value is None or isinstance(value, (str, int, float, bool)):
            portable[field] = value
    for field in sorted(_RESULT_PATH_FIELDS):
        value = result.get(field)
        if value is None or not str(value).strip():
            continue
        relative = _project_relative_path(
            project_root, value, label=f"result {field}", require_exists=False
        )
        if not (project_root / relative).exists():
            continue
        portable[field] = relative
        artifact_paths.append(relative)
    return portable, artifact_paths


def validate_worker_terminal(
    terminal: Mapping[str, Any],
    *,
    claim: Mapping[str, Any] | None = None,
    bfts_terminal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(terminal, Mapping):
        raise ResearchGitError("experiment admission worker terminal is invalid")
    normalized = dict(terminal)
    expected = {
        "schema",
        "start_object_id",
        "receipt_hash",
        "claim_hash",
        "run_nonce",
        "worker_nonce",
        "bfts_run_nonce",
        "idea_index",
        "experiment_dir",
        "bfts_terminal_hash",
        "pipeline_status",
        "experiment_status",
        "research_status",
        "stage",
        "resumable",
        "failure",
        "result",
        "provenance",
        "artifacts",
        "finished_at",
        "terminal_hash",
    }
    core = {key: value for key, value in normalized.items() if key != "terminal_hash"}
    provenance = normalized.get("provenance")
    if (
        set(normalized) != expected
        or normalized.get("schema") != WORKER_TERMINAL_SCHEMA
        or normalized.get("bfts_terminal_hash") is not None
        and not _HASH_RE.fullmatch(str(normalized["bfts_terminal_hash"]))
        or normalized.get("pipeline_status") not in _PIPELINE_STATUSES
        or normalized.get("experiment_status") not in _EXPERIMENT_STATUSES
        or not isinstance(normalized.get("research_status"), str)
        or len(normalized["research_status"]) > 128
        or not _SAFE_NAME_RE.fullmatch(str(normalized.get("stage") or ""))
        or type(normalized.get("resumable")) is not bool
        or not isinstance(normalized.get("result"), Mapping)
        or set(normalized["result"]) - (_RESULT_SCALAR_FIELDS | _RESULT_PATH_FIELDS)
        or not isinstance(provenance, Mapping)
        or any(not isinstance(key, str) for key in provenance)
        or normalized.get("terminal_hash") != canonical_content_hash(core)
    ):
        raise ResearchGitError("experiment admission worker terminal is invalid")
    _validate_failure(normalized["failure"])
    _validate_artifacts(normalized["artifacts"])
    _validate_timestamp(normalized["finished_at"], label="worker terminal")
    _validate_relative_path(normalized["experiment_dir"], label="experiment directory")
    if claim is not None:
        identity = _terminal_identity(claim)
        if any(normalized.get(key) != value for key, value in identity.items()):
            raise ResearchGitError(
                "experiment admission worker terminal identity conflicts"
            )
    if bfts_terminal is not None:
        validated_bfts = validate_bfts_terminal(bfts_terminal, claim=claim)
        if normalized["bfts_terminal_hash"] != validated_bfts["terminal_hash"]:
            raise ResearchGitError(
                "experiment admission worker terminal BFTS hash conflicts"
            )
    return normalized


def publish_worker_terminal(
    repo: str | Path,
    claim: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
    failure: Any = None,
) -> dict[str, Any]:
    root = _repository_root(repo)
    identity = _terminal_identity(claim)
    existing = load_worker_terminal(
        root, start_object_id=identity["start_object_id"], claim=claim
    )
    portable, artifact_paths = _portable_result(root, result)
    bfts = load_bfts_terminal(
        root, start_object_id=identity["start_object_id"], claim=claim
    )
    pipeline_status = "success" if result.get("status") == "success" else "failed"
    runtime_status = str(
        result.get("registry_status")
        or ("completed" if result.get("runtime_status") == "completed" else "failed")
    ).lower()
    experiment_status = (
        runtime_status if runtime_status in _EXPERIMENT_STATUSES else "failed"
    )
    stage = re.sub(r"[^A-Za-z0-9._-]+", "_", str(result.get("stage") or "pipeline"))
    stage = stage[:128] or "pipeline"
    research_status = str(result.get("research_status") or "unknown")[:128]
    projected_provenance = dict(provenance or {})
    canonical_content_hash(projected_provenance)
    core = {
        "schema": WORKER_TERMINAL_SCHEMA,
        **identity,
        "bfts_terminal_hash": bfts.get("terminal_hash") if bfts else None,
        "pipeline_status": pipeline_status,
        "experiment_status": experiment_status,
        "research_status": research_status,
        "stage": stage,
        "resumable": bool(result.get("resumable")),
        "failure": _failure_projection(failure, fallback_code="worker_failed"),
        "result": portable,
        "provenance": projected_provenance,
        "artifacts": _artifact_manifest(root, artifact_paths),
        "finished_at": existing["finished_at"] if existing is not None else _now_iso(),
    }
    payload = {**core, "terminal_hash": canonical_content_hash(core)}
    validate_worker_terminal(payload, claim=claim, bfts_terminal=bfts)
    path = _marker_path(
        root, _WORKER_DIRECTORY, identity["start_object_id"], create=True
    )
    return _publish_immutable(path, payload, label="worker terminal")


def load_worker_terminal(
    repo: str | Path,
    *,
    start_object_id: str,
    claim: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _marker_path(repo, _WORKER_DIRECTORY, start_object_id, create=False)
    marker = _decode_json_marker(path, label="worker terminal")
    if marker is None:
        return None
    bfts = load_bfts_terminal(repo, start_object_id=start_object_id, claim=claim)
    return validate_worker_terminal(marker, claim=claim, bfts_terminal=bfts)


def _enumerate_ids(repo: str | Path, directory_name: str) -> list[str]:
    directory = _checked_directory(repo, directory_name, create=False)
    if not directory.exists():
        return []
    metadata = directory.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ResearchGitError("experiment admission runtime directory is unsafe")
    ids: list[str] = []
    for entry in directory.iterdir():
        match = _MARKER_RE.fullmatch(entry.name)
        if match is None:
            raise ResearchGitError(
                "experiment admission runtime marker name is invalid"
            )
        ids.append(match.group("start_id"))
        if len(ids) > _MAX_MARKERS:
            raise ResearchGitError("experiment admission runtime has too many markers")
    current = directory.lstat()
    if (current.st_dev, current.st_ino, current.st_mode) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
    ):
        raise ResearchGitError("experiment admission runtime directory changed")
    return sorted(ids)


def admission_runtime_records(repo: str | Path) -> list[dict[str, Any]]:
    from .research_git import _load_experiment_admission_execution_claim

    claim_ids = set(_enumerate_ids(repo, _CLAIM_DIRECTORY))
    bfts_ids = set(_enumerate_ids(repo, _BFTS_DIRECTORY))
    worker_ids = set(_enumerate_ids(repo, _WORKER_DIRECTORY))
    ids = claim_ids | bfts_ids | worker_ids
    records: list[dict[str, Any]] = []
    for start_object_id in sorted(ids):
        claim = _load_experiment_admission_execution_claim(
            repo, start_object_id=start_object_id
        )
        if claim is None:
            raise ResearchGitError(
                "experiment admission terminal outbox has no v2 claim"
            )
        if claim.get("schema") != CLAIM_SCHEMA:
            if start_object_id in bfts_ids or start_object_id in worker_ids:
                raise ResearchGitError(
                    "experiment admission terminal outbox is bound to a legacy claim"
                )
            records.append(
                {
                    "start_object_id": start_object_id,
                    "claim": claim,
                    "bfts_terminal": None,
                    "worker_terminal": None,
                    "owner_state": "legacy_in_doubt",
                }
            )
            continue
        claim = validate_execution_claim_v2(claim, start_object_id=start_object_id)
        bfts = load_bfts_terminal(repo, start_object_id=start_object_id, claim=claim)
        worker = load_worker_terminal(
            repo, start_object_id=start_object_id, claim=claim
        )
        records.append(
            {
                "start_object_id": start_object_id,
                "claim": claim,
                "bfts_terminal": bfts,
                "worker_terminal": worker,
                "owner_state": execution_claim_owner_state(claim),
            }
        )
    return records


def result_from_runtime_record(
    repo: str | Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    root = _repository_root(repo)
    raw_claim = record.get("claim")
    if isinstance(raw_claim, Mapping) and raw_claim.get("schema") != CLAIM_SCHEMA:
        if (
            record.get("bfts_terminal") is not None
            or record.get("worker_terminal") is not None
        ):
            raise ResearchGitError(
                "experiment admission terminal outbox is bound to a legacy claim"
            )
        if record.get("owner_state") != "legacy_in_doubt":
            raise ResearchGitError(
                "experiment admission legacy claim recovery state is invalid"
            )
        return None
    claim = validate_execution_claim_v2(raw_claim)
    worker = record.get("worker_terminal")
    if worker is not None:
        terminal = validate_worker_terminal(worker, claim=claim)
        verify_terminal_artifacts(root, terminal)
        result = dict(terminal["result"])
        for field in _RESULT_PATH_FIELDS & set(result):
            result[field] = str(
                root / _validate_relative_path(result[field], label=field)
            )
        result["idea_idx"] = claim["idea_index"]
        result["status"] = terminal["pipeline_status"]
        result["registry_status"] = terminal["experiment_status"]
        result["research_status"] = terminal["research_status"]
        result["stage"] = terminal["stage"]
        result["resumable"] = terminal["resumable"]
        return result, dict(terminal["provenance"])
    bfts = record.get("bfts_terminal")
    if bfts is not None:
        terminal = validate_bfts_terminal(bfts, claim=claim)
        verify_terminal_artifacts(root, terminal)
        completed = terminal["status"] == "completed"
        checkpoint = next(
            (
                str(root / row["path"])
                for row in terminal["artifacts"]
                if "checkpoint" in Path(row["path"]).name.lower()
            ),
            None,
        )
        return (
            {
                "idea_idx": claim["idea_index"],
                "exp_dir": str(root / claim["experiment_dir"]),
                "status": "failed",
                "runtime_status": terminal["status"],
                "registry_status": "completed" if completed else "failed",
                "research_status": "failed",
                "stage": "worker_lost_after_bfts",
                "failure_stage": "post_bfts_pipeline",
                "resumable": bool(terminal["resumable"]),
                "checkpoint_path": checkpoint,
            },
            {},
        )
    if record.get("owner_state") == "dead":
        return (
            {
                "idea_idx": claim["idea_index"],
                "exp_dir": str(root / claim["experiment_dir"]),
                "status": "failed",
                "runtime_status": "interrupted",
                "registry_status": "failed",
                "research_status": "failed",
                "stage": "worker_lost",
                "failure_stage": "pre_bfts_terminal",
                "resumable": False,
            },
            {},
        )
    return None


def validate_result_against_runtime_record(
    repo: str | Path,
    record: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the durable outcome only when ``result`` matches its outbox.

    The parent process may carry a richer in-memory result, but the immutable
    worker projection is the sole replayable input to Research VCS. BFTS-only
    and pre-BFTS recovery results are compared against the conservative result
    synthesized from the durable claim state.
    """

    recovered = result_from_runtime_record(repo, record)
    if recovered is None:
        raise ResearchGitError(
            "experiment admission runtime outcome is not recoverable"
        )
    durable_result, provenance = recovered
    claim = validate_execution_claim_v2(record.get("claim"))
    worker = record.get("worker_terminal")
    if worker is not None:
        terminal = validate_worker_terminal(worker, claim=claim)
        candidate, _paths = _portable_result(_repository_root(repo), result)
        if candidate != dict(terminal["result"]):
            raise ResearchGitError(
                "experiment admission worker result conflicts with its terminal outbox"
            )
    else:
        candidate, _paths = _portable_result(_repository_root(repo), result)
        expected, _expected_paths = _portable_result(
            _repository_root(repo), durable_result
        )
        if candidate != expected:
            raise ResearchGitError(
                "experiment admission recovered result conflicts with its runtime state"
            )
    return durable_result, provenance


__all__ = [
    "BFTS_TERMINAL_SCHEMA",
    "CLAIM_SCHEMA",
    "WORKER_TERMINAL_SCHEMA",
    "admission_runtime_records",
    "claim_experiment_admission_start_v2",
    "execution_claim_owner_state",
    "load_bfts_terminal",
    "load_execution_claim_v2",
    "load_worker_terminal",
    "publish_bfts_terminal",
    "publish_worker_terminal",
    "result_from_runtime_record",
    "validate_result_against_runtime_record",
    "validate_bfts_terminal",
    "validate_execution_claim_v2",
    "validate_worker_terminal",
    "verify_terminal_artifacts",
]
