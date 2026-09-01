"""Local-first Git history for scientific projects.

Git is used as the small, reviewable control plane. Large immutable payloads
stay in the local ARA content-addressed store and enter commits as pointer
records. Nothing in this module creates a remote or pushes data.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import mimetypes
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - platform-specific import
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - platform-specific import
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

import yaml
from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.hashing import content_hash, hash_manifest
from ai_scientist.protocol.research_vcs import (
    RESEARCH_RELATION_TARGET_KINDS,
    RESEARCH_RELATION_TYPES,
    ResearchObjectError,
    build_research_object,
    validate_research_object,
)
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.bounded_process import run_process_bounded
from ai_scientist.utils.privacy import (
    format_privacy_findings,
    redact_sensitive_payload,
    redact_sensitive_text,
    scan_file,
    scan_paths,
)

REPOSITORY_SCHEMA = "xscientist.research-repository.v1"
CHECKPOINT_SCHEMA = "xscientist.research-checkpoint.v1"
OBJECT_POINTER_SCHEMA = "xscientist.research-object-pointer.v1"
BUNDLE_SCHEMA = "xscientist.research-bundle.v1"
ENVIRONMENT_SCHEMA = "xscientist.research-environment.v1"
REPRODUCTION_RECEIPT_SCHEMA = "xscientist.reproduction-receipt.v2"
_MAX_REPRODUCTION_OUTPUT_CHARS = 20_000
_HARD_MAX_STORE_ENTRIES = 32768
_HARD_MAX_STORE_FILES = 8192
_HARD_MAX_STORE_BYTES = 128 * 1024 * 1024
_HARD_MAX_RESEARCH_OBJECTS = 8192
_HARD_MAX_BOUNDED_LOG_OUTPUT = 1 * 1024 * 1024
_BOUNDED_LOG_TIMEOUT_SECONDS = 10.0
_MAX_TRAJECTORY_CHECKPOINTS = 127
_MAX_TRAJECTORY_CHECKPOINT_BYTES = 512 * 1024
_MAX_TRAJECTORY_CHECKPOINT_CANDIDATES = 16
_MAX_TRAJECTORY_OBJECT_BYTES = 512 * 1024
_MAX_RESEARCH_STAGE_BYTES = 1 * 1024 * 1024
_MAX_RESEARCH_STAGE_ENTRIES = 4096
_MAX_BUNDLE_POINTER_BYTES = 1 * 1024 * 1024
_MAX_PRIVACY_SCANNABLE_OBJECT_BYTES = 4_000_000

DEFAULT_TRACK_PATTERNS = (
    ".gitignore",
    ".dockerignore",
    ".env.example",
    "README.md",
    "topic.md",
    "bfts_config.yaml",
    "research.yaml",
    "question.md",
    ".xscientist/**",
    "hypotheses/**",
    "claims/**",
    "manuscript/**",
    "checkpoints/**",
    "research-objects/**",
    "ara/**/manifest.json",
    "ara/**/manifest.lock",
    "ara/**/manifest.history.jsonl",
    "ara/**/history/*.json",
    "ara/**/exploration_graph.json",
    "ara/**/claims/*.json",
    "ara/**/events/*.jsonl",
    "ara/**/context/receipts.jsonl",
    "ara/**/env/*.json",
    "ara/**/env/*.yaml",
    "ara/**/env/*.yml",
    "ara/**/verify/*.json",
    "00_config/*.json",
    "00_config/*.yaml",
    "00_config/*.yml",
    "00_config/*.toml",
    "01_ideas/*.json",
    "01_ideas/*.md",
    "02_experiments/**/code.py",
    "02_experiments/**/run.sh",
    "02_experiments/**/idea.json",
    "02_experiments/**/idea.md",
    "02_experiments/**/metrics.json",
    "02_experiments/**/env.json",
    "02_experiments/**/pipeline_manifest.json",
    "02_experiments/**/pipeline/*.json",
    "02_experiments/**/claim_evidence_graph.json",
    "02_experiments/**/experiment_registry.jsonl",
    "02_experiments/**/research_plan.json",
    "02_experiments/**/figure_spec.json",
    "02_experiments/**/plot_execution_receipt.json",
    "02_experiments/**/manuscript_state.json",
    "02_experiments/**/manuscript_candidate_pool.json",
    "02_experiments/**/review_state.json",
    "02_experiments/**/critic_findings.json",
    "02_experiments/**/repair_plan.json",
    "02_experiments/**/repair_attempts.jsonl",
    "02_experiments/**/stage_standards.json",
    "02_experiments/**/process_alignment.json",
    "02_experiments/**/arft_coverage.json",
    "02_experiments/**/truth_contract.json",
    "02_experiments/**/hallucination_checks.json",
    "02_experiments/**/hallucination_review.json",
    "02_experiments/**/sample_gate.json",
    "02_experiments/**/decision_log.jsonl",
    "02_experiments/**/verification_report.json",
    "02_experiments/**/evaluation_charter.json",
    "02_experiments/**/evaluation_benchmarks.jsonl",
    "02_experiments/**/evaluation_report.json",
    "02_experiments/**/final_review*.json",
    "02_experiments/**/self_review*.json",
    "02_experiments/**/reviews*/*.json",
    "02_experiments/**/reviews*/*.jsonl",
    "02_experiments/**/reviews*/*.md",
    "02_experiments/**/reviews*/*.tex",
    "02_experiments/**/reviews*/*.bib",
    "02_experiments/**/hostile_critic/*.json",
    "02_experiments/**/hostile_critic/*.jsonl",
    "02_experiments/**/hostile_critic/*.md",
    "02_experiments/**/hostile_critic/*.tex",
    "02_experiments/**/hostile_critic/*.bib",
    "02_experiments/**/*.tex",
    "02_experiments/**/*.bib",
    "03_papers/*.tex",
    "03_papers/*.bib",
    "03_papers/*.md",
    "04_logs/project_summary.json",
    "04_logs/project_summary.md",
    "04_logs/submission_shortlist.md",
    "04_logs/progress.json",
    "04_logs/autopilot_run.json",
    "04_logs/llm_budget.json",
    "04_logs/insight_report.json",
    "04_logs/insight_report.md",
    "04_logs/autopilot_fixture_receipt.json",
    "pipeline_manifest.json",
    "figure_spec.json",
    "plot_execution_receipt.json",
    "manuscript_state.json",
    "manuscript_candidate_pool.json",
    "review_state.json",
    "critic_findings.json",
    "repair_plan.json",
    "repair_attempts.jsonl",
    "stage_standards.json",
    "process_alignment.json",
    "arft_coverage.json",
    "truth_contract.json",
    "hallucination_checks.json",
    "hallucination_review.json",
    "sample_gate.json",
    "decision_log.jsonl",
    "verification_report.json",
    "evaluation_charter.json",
    "evaluation_benchmarks.jsonl",
    "evaluation_report.json",
    "final_review*.json",
    "self_review*.json",
    "reviews*/*.json",
    "reviews*/*.jsonl",
    "reviews*/*.md",
    "reviews*/*.tex",
    "reviews*/*.bib",
    "hostile_critic/*.json",
    "hostile_critic/*.jsonl",
    "hostile_critic/*.md",
    "hostile_critic/*.tex",
    "hostile_critic/*.bib",
    "science_constitution.json",
    "epistemic_graph.json",
    "research_plan.json",
    "pyproject.toml",
    "requirements*.txt",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "environment*.yml",
    "environment*.yaml",
    "conda-lock*.yml",
    "conda-lock*.yaml",
    "Dockerfile",
    "Dockerfile.*",
)

DEFAULT_DENY_PATTERNS = (
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    ".ara-store/**",
    "**/__pycache__/**",
    "**/*.log",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_ed25519",
    "**/credentials.json",
    "**/secrets.json",
    "**/*token.json",
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/*.bin",
    "**/*.npy",
    "**/*.npz",
    "**/*.parquet",
    "**/*.arrow",
    "**/*.zip",
    "**/*.tar",
    "**/*.tar.gz",
    "**/*.pdf",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
)

SECRET_DENY_PATTERNS = (
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    ".secrets",
    ".secrets/**",
    "**/.secrets",
    "**/.secrets/**",
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_ed25519",
    "**/credentials.json",
    "**/secrets.json",
    "**/*token.json",
)

MILESTONE_STAGES = frozenset(
    {
        "init",
        "ideation",
        "planning",
        "preregister",
        "experiment",
        "evidence",
        "review",
        "claim",
        "paper",
        "evolve",
        "merge",
        "release",
    }
)

_RESEARCH_GITIGNORE = """# XScientist local research repository
# Secrets and local credentials
.env
.env.*
!.env.example
*.pem
*.key
credentials.json
secrets.json

# Local content-addressed payloads (tracked through research-objects pointers)
.ara-store/

# Generated caches and raw logs
.xscientist/readiness.json
__pycache__/
*.py[cod]
*.log
.pytest_cache/
.mypy_cache/
cache/

# Large binary/scientific payloads: register with `xscientist research object add`
*.pt
*.pth
*.ckpt
*.bin
*.npy
*.npz
*.parquet
*.arrow
*.zip
*.tar
*.tar.gz
*.pdf
*.png
*.jpg
*.jpeg
"""

_REPOSITORY_NOTE = """# XScientist local research history

This directory is a local-first scientific Git repository. Git tracks compact
research state and checkpoints. Large immutable evidence is stored under
`.ara-store/` and represented by committed files in `research-objects/`.

No remote is required and XScientist never pushes automatically.
"""


def _merged_research_gitignore(existing: str) -> str:
    """Append an ordered safety block without replacing project-owned rules."""

    required_rules = [
        line
        for line in _RESEARCH_GITIGNORE.splitlines()
        if line and not line.startswith("#")
    ]
    existing_rules = set(existing.splitlines())
    if all(rule in existing_rules for rule in required_rules):
        return existing
    prefix = existing
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    if prefix and not prefix.endswith("\n\n"):
        prefix += "\n"
    # Repeat the complete ordered block. Negations such as ``!.env.example``
    # must occur after the broad secret patterns to remain effective.
    return prefix + _RESEARCH_GITIGNORE


class ResearchGitError(RuntimeError):
    """A safe local-repository operation could not be completed."""


_PROCESS_REPOSITORY_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_REPOSITORY_LOCKS_GUARD = threading.Lock()
_REPOSITORY_LOCK_STATE = threading.local()


def _reset_repository_locks_after_fork() -> None:
    global _PROCESS_REPOSITORY_LOCKS
    global _PROCESS_REPOSITORY_LOCKS_GUARD
    global _REPOSITORY_LOCK_STATE
    _PROCESS_REPOSITORY_LOCKS = {}
    _PROCESS_REPOSITORY_LOCKS_GUARD = threading.Lock()
    _REPOSITORY_LOCK_STATE = threading.local()


if hasattr(os, "register_at_fork"):  # pragma: no branch - POSIX capability
    os.register_at_fork(after_in_child=_reset_repository_locks_after_fork)


@contextmanager
def _repository_lock(repo: Path, *, timeout_seconds: float = 30.0):
    """Serialize scientific state transitions without leaving stale locks."""

    raw_path = _run_git(
        repo, ["rev-parse", "--git-path", "xscientist-research.lock"]
    ).stdout.strip()
    lock_path = Path(raw_path)
    if not lock_path.is_absolute():
        lock_path = repo / lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(lock_path.resolve())
    deadline = time.monotonic() + timeout_seconds
    remaining = max(0.0, deadline - time.monotonic())
    if not _PROCESS_REPOSITORY_LOCKS_GUARD.acquire(timeout=remaining):
        raise ResearchGitError(
            f"timed out waiting for research repository lock: {lock_path}"
        )
    try:
        process_lock = _PROCESS_REPOSITORY_LOCKS.setdefault(lock_key, threading.RLock())
    finally:
        _PROCESS_REPOSITORY_LOCKS_GUARD.release()
    remaining = max(0.0, deadline - time.monotonic())
    if not process_lock.acquire(timeout=remaining):
        raise ResearchGitError(
            f"timed out waiting for research repository lock: {lock_path}"
        )
    try:
        held = getattr(_REPOSITORY_LOCK_STATE, "held", set())
        if lock_key in held:
            yield
            return
        _REPOSITORY_LOCK_STATE.held = {*held, lock_key}
        try:
            with lock_path.open("a+b") as stream:
                _acquire_repository_file_lock(
                    stream,
                    lock_path=lock_path,
                    timeout_seconds=max(0.0, deadline - time.monotonic()),
                )
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                    elif msvcrt is not None:
                        stream.seek(0)
                        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            _REPOSITORY_LOCK_STATE.held = held
    finally:
        process_lock.release()


def _acquire_repository_file_lock(
    stream: Any,
    *,
    lock_path: Path,
    timeout_seconds: float,
) -> None:
    """Acquire the cross-process half of a repository lock."""

    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                raise ResearchGitError(
                    f"timed out waiting for research repository lock: {lock_path}"
                )
            time.sleep(0.05)


@dataclass(frozen=True)
class CheckpointResult:
    created: bool
    committed: bool
    checkpoint_path: Path | None = None
    commit: str | None = None
    checkpoint_id: str | None = None
    content_hash: str | None = None
    staged_paths: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "committed": self.committed,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "commit": self.commit,
            "checkpoint_id": self.checkpoint_id,
            "content_hash": self.content_hash,
            "staged_paths": list(self.staged_paths),
            "excluded_paths": list(self.excluded_paths),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ObjectPointerResult:
    pointer_path: Path
    store_path: Path
    object_hash: str
    size: int
    linked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer_path": str(self.pointer_path),
            "store_path": str(self.store_path),
            "object_hash": self.object_hash,
            "size": self.size,
            "linked": self.linked,
        }


@dataclass(frozen=True)
class ResearchObjectResult:
    """Result of recording one immutable, typed Research VCS object."""

    created: bool
    path: Path
    object_id: str
    qualified_id: str | None
    object_hash: str
    kind: str
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "path": str(self.path),
            "object_id": self.object_id,
            "qualified_id": self.qualified_id,
            "object_hash": self.object_hash,
            "kind": self.kind,
            "state": self.state,
        }


@dataclass(frozen=True)
class ResearchStageResult:
    """The native Research VCS staging selection."""

    paths: tuple[str, ...]
    added: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": list(self.paths),
            "added": list(self.added),
            "removed": list(self.removed),
            "excluded": list(self.excluded),
        }


@dataclass(frozen=True)
class ResearchMergeResult:
    """A completed scientific merge with its multi-parent checkpoint."""

    source: str
    target: str
    source_commit: str
    commit: str
    checkpoint_id: str
    content_hash: str
    resolution_objects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "source_commit": self.source_commit,
            "commit": self.commit,
            "checkpoint_id": self.checkpoint_id,
            "content_hash": self.content_hash,
            "resolution_objects": list(self.resolution_objects),
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomically write canonical UTF-8 bytes without platform newline rewriting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry update where the platform supports it."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync. The file itself was
        # already flushed, so keep the portable best-effort behaviour.
        pass
    finally:
        os.close(descriptor)


def _run_git(
    repo: Path,
    args: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except FileNotFoundError as exc:
        raise ResearchGitError("Git is required for local research history") from exc
    except OSError as exc:
        raise ResearchGitError("cannot start Git command") from exc
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise ResearchGitError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _run_git_bounded(
    repo: Path,
    args: Sequence[str],
    *,
    max_output_bytes: int = _HARD_MAX_BOUNDED_LOG_OUTPUT,
    timeout_seconds: float = _BOUNDED_LOG_TIMEOUT_SECONDS,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a read-only Git command with a hard stdout/stderr budget.

    ``subprocess.run(capture_output=True)`` allocates the complete stream
    before a caller can truncate it.  Audit paths must not make that promise
    against an untrusted repository, so this helper drains both pipes while
    enforcing a combined byte cap and a wall-clock deadline.  It is kept
    separate from the historical ``_run_git`` API so existing integrations and
    test doubles retain their call signature.
    """

    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes <= 0
        or max_output_bytes > _HARD_MAX_BOUNDED_LOG_OUTPUT
    ):
        raise ResearchGitError("bounded Git output limit is invalid")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
        or timeout_seconds > 60.0
    ):
        raise ResearchGitError("bounded Git timeout is invalid")

    command = ["git", "-C", str(repo), *args]
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except FileNotFoundError as exc:
        raise ResearchGitError("Git is required for local research history") from exc
    except OSError as exc:
        raise ResearchGitError("cannot start bounded Git audit command") from exc

    streams = [stream for stream in (process.stdout, process.stderr) if stream]
    buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    capture_lock = threading.Lock()
    capture_state = {"total": 0}
    overflow_event = threading.Event()
    capture_error_event = threading.Event()

    def stop_process() -> None:
        """Wake the bounded wait when a pipe reader rejects the command."""

        try:
            if process.poll() is None:
                process.kill()
        except OSError:
            pass

    def drain(name: str, stream: Any) -> None:
        """Drain one pipe without relying on Windows-incompatible selectors."""

        try:
            while not overflow_event.is_set():
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    return
                overflowed = False
                with capture_lock:
                    room = max_output_bytes - capture_state["total"]
                    if len(chunk) > room:
                        overflow_event.set()
                        overflowed = True
                    else:
                        buffers[name].extend(chunk)
                        capture_state["total"] += len(chunk)
                if overflowed:
                    stop_process()
                    return
        except (OSError, ValueError):
            capture_error_event.set()
            stop_process()

    drain_threads = [
        threading.Thread(
            target=drain,
            args=(name, stream),
            daemon=True,
            name=f"xscientist-git-{name}",
        )
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))
        if stream is not None
    ]
    deadline = time.monotonic() + float(timeout_seconds)
    timed_out = False
    capture_setup_failed = False
    reap_failed = False
    started_threads: list[threading.Thread] = []
    try:
        for thread in drain_threads:
            try:
                thread.start()
            except Exception:  # fail closed if the capture boundary cannot start
                capture_setup_failed = True
                break
            started_threads.append(thread)

        if not capture_setup_failed and process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
            else:
                try:
                    process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    timed_out = True
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                reap_failed = True
        for thread in started_threads:
            thread.join(timeout=1.0)
        for stream in streams:
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        for thread in started_threads:
            if thread.is_alive():
                thread.join(timeout=0.1)

    if timed_out:
        raise ResearchGitError("bounded Git command exceeded time limit")
    if overflow_event.is_set():
        raise ResearchGitError("bounded Git command exceeded output limit")
    if reap_failed:
        raise ResearchGitError("bounded Git command did not terminate")
    if capture_setup_failed or capture_error_event.is_set():
        raise ResearchGitError("bounded Git command output capture failed")
    if any(thread.is_alive() for thread in started_threads):
        raise ResearchGitError("bounded Git command output capture did not finish")
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=bytes(buffers["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(buffers["stderr"]).decode("utf-8", errors="replace"),
    )
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[:512]
        raise ResearchGitError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _normalise_relative(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ResearchGitError(f"unsafe repository-relative path: {path}")
    return candidate.as_posix()


def validate_research_logical_component(value: str | Path) -> str:
    """Validate one portable filename component used for CAS hydration."""

    rendered = str(value)
    if (
        not rendered
        or rendered in {".", ".."}
        or rendered[-1:] in {" ", "."}
        or any(ord(character) < 32 for character in rendered)
        or any(character in '<>:"/\\|?*' for character in rendered)
    ):
        raise ResearchGitError(
            f"unsafe cross-platform research object filename: {rendered!r}"
        )
    device_stem = rendered.split(".", 1)[0].casefold()
    if device_stem in {"con", "prn", "aux", "nul"} or re.fullmatch(
        r"(?:com|lpt)[1-9]", device_stem
    ):
        raise ResearchGitError(
            f"unsafe cross-platform research object filename: {rendered!r}"
        )
    return rendered


def _validate_research_logical_path(value: str | Path) -> str:
    logical = _normalise_relative(value)
    for component in PurePosixPath(logical).parts:
        validate_research_logical_component(component)
    return logical


def _research_logical_alias_key(value: str | Path) -> tuple[str, ...]:
    logical = _validate_research_logical_path(value)
    return tuple(
        unicodedata.normalize("NFC", component).casefold()
        for component in PurePosixPath(logical).parts
    )


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _path_is_explicit(path: str, explicit: Iterable[str]) -> bool:
    for raw in explicit:
        prefix = _normalise_relative(raw).rstrip("/")
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


_DEPENDENCY_LOCK_PATTERNS = (
    "pyproject.toml",
    "requirements*.txt",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "environment*.yml",
    "environment*.yaml",
    "conda-lock*.yml",
    "conda-lock*.yaml",
    "Dockerfile",
    "Dockerfile.*",
)


def _environment_receipt(repo: Path) -> dict[str, Any]:
    dependency_paths = sorted(
        {
            path
            for pattern in _DEPENDENCY_LOCK_PATTERNS
            for path in repo.glob(pattern)
            if path.is_file() and not path.is_symlink()
        },
        key=lambda path: path.relative_to(repo).as_posix(),
    )
    base = {
        "schema_version": ENVIRONMENT_SCHEMA,
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependency_locks": [
            {
                "path": path.relative_to(repo).as_posix(),
                "hash": _hash_file(path),
                "size": path.stat().st_size,
            }
            for path in dependency_paths
        ],
    }
    receipt = {**base, "content_hash": content_hash(base)}
    try:
        validate_json(receipt, load_schema("research_environment"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"generated research environment receipt is invalid: {exc.message}"
        ) from exc
    return receipt


def capture_environment_receipt(repo: str | Path) -> dict[str, Any]:
    """Capture the compact, secret-free environment identity for a repository."""

    return _environment_receipt(_repository_root(repo))


def _environment_receipt_hash_valid(receipt: Any) -> bool:
    if not isinstance(receipt, dict):
        return False
    expected = str(receipt.get("content_hash") or "")
    scrubbed = {key: value for key, value in receipt.items() if key != "content_hash"}
    return bool(expected and content_hash(scrubbed) == expected)


def _compare_runtime_environment(receipt: Any) -> dict[str, Any]:
    if not isinstance(receipt, dict) or not _environment_receipt_hash_valid(receipt):
        return {
            "recorded": False,
            "matches": None,
            "mismatches": [
                {
                    "field": "reproduce.environment",
                    "expected": "valid environment receipt",
                    "actual": "missing or invalid",
                }
            ],
        }
    expected_python = receipt.get("python") or {}
    expected_platform = receipt.get("platform") or {}
    actual = {
        "python.implementation": sys.implementation.name,
        "python.version": platform.python_version(),
        "platform.system": platform.system(),
        "platform.machine": platform.machine(),
    }
    expected = {
        "python.implementation": expected_python.get("implementation"),
        "python.version": expected_python.get("version"),
        "platform.system": expected_platform.get("system"),
        "platform.machine": expected_platform.get("machine"),
    }
    mismatches = [
        {"field": field, "expected": expected[field], "actual": value}
        for field, value in actual.items()
        if expected[field] != value
    ]
    return {
        "recorded": True,
        "matches": not mismatches,
        "mismatches": mismatches,
        "recorded_content_hash": receipt.get("content_hash"),
    }


def _compare_dependency_locks(worktree: Path, receipt: Any) -> list[dict[str, Any]]:
    if not isinstance(receipt, dict):
        return []
    mismatches: list[dict[str, Any]] = []
    for item in receipt.get("dependency_locks") or []:
        relative = _normalise_relative(str(item.get("path") or ""))
        target = worktree / relative
        actual_hash = (
            _hash_file(target) if target.is_file() and not target.is_symlink() else None
        )
        if actual_hash != item.get("hash"):
            mismatches.append(
                {
                    "field": f"dependency_locks.{relative}",
                    "expected": item.get("hash"),
                    "actual": actual_hash,
                }
            )
    return mismatches


def _copy_into_store(
    source_fd: int,
    store_root: Path,
    *,
    privacy_root: Path,
) -> tuple[str, int, Path]:
    """Copy one immutable snapshot into CAS and return hash, size, and path.

    A hard link is deliberately not used: mutating the caller's source after
    registration must never mutate the content-addressed object.
    """

    incoming = store_root / "incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix="object-", dir=incoming)
    temp = Path(raw_temp)
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            os.fdopen(source_fd, "rb") as source_stream,
            os.fdopen(handle, "wb") as target,
        ):
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if size <= _MAX_PRIVACY_SCANNABLE_OBJECT_BYTES:
            try:
                snapshot = temp.read_bytes()
            except OSError as exc:
                raise ResearchGitError(
                    "privacy gate could not scan the research object snapshot"
                ) from exc
            if len(snapshot) != size:
                raise ResearchGitError(
                    "privacy gate could not verify the research object snapshot"
                )
            if b"\0" not in snapshot[:8192]:
                findings = scan_file(temp, root=privacy_root, scope="research_object")
                if findings:
                    raise ResearchGitError(
                        "privacy gate refused the research object:\n"
                        + format_privacy_findings(findings)
                    )
        object_hash = f"sha256:{digest.hexdigest()}"
        store_path = store_root / "objects" / "sha256" / digest.hexdigest()
        store_path.parent.mkdir(parents=True, exist_ok=True)
        if store_path.exists():
            if not store_path.is_file() or _hash_file(store_path) != object_hash:
                raise ResearchGitError(
                    f"CAS collision or damaged object at {store_path}"
                )
        else:
            temp.replace(store_path)
            _fsync_directory(store_path.parent)
        return object_hash, size, store_path
    finally:
        temp.unlink(missing_ok=True)


def _repository_root(path: str | Path, *, require_config: bool = True) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise ResearchGitError(f"research repository does not exist: {candidate}")
    completed = _run_git(candidate, ["rev-parse", "--show-toplevel"])
    root = Path(completed.stdout.strip()).resolve()
    config_path = root / "research.yaml"
    if require_config and config_path.is_symlink():
        raise ResearchGitError("research.yaml must not be a symlink")
    if require_config and not config_path.is_file():
        raise ResearchGitError(
            f"{root} is a Git repository but not an XScientist research repository; "
            "run `xscientist research init <path>` first"
        )
    return root


def _resolve_configured_path(root: Path, raw_path: Any, *, label: str) -> Path:
    try:
        relative = _normalise_relative(str(raw_path))
    except ResearchGitError as exc:
        raise ResearchGitError(
            f"configured {label} escapes research repository: {raw_path}"
        ) from exc
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ResearchGitError(
            f"configured {label} escapes research repository: {relative}"
        ) from exc
    return target


def _has_symlink_component(root: Path, raw_path: Any) -> bool:
    """Return whether an allowlisted relative path crosses a symlink."""

    try:
        relative = PurePosixPath(str(raw_path).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            return True
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return True
    except (OSError, RuntimeError, ValueError):
        return True
    return False


def load_repository_config(repo: str | Path) -> dict[str, Any]:
    root = _repository_root(repo)
    try:
        payload = yaml.safe_load((root / "research.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ResearchGitError(f"cannot read research.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchGitError("research.yaml must contain a YAML mapping")
    try:
        validate_json(payload, load_schema("research_repository"))
    except ValidationError as exc:
        raise ResearchGitError(f"invalid research.yaml: {exc.message}") from exc
    git_config = payload["git"]
    if git_config.get("auto_push") is not False:
        raise ResearchGitError(
            "local research repositories must keep git.auto_push false"
        )
    storage = payload["storage"]
    _resolve_configured_path(root, storage["root"], label="CAS root")
    _resolve_configured_path(
        root,
        storage["pointer_directory"],
        label="pointer directory",
    )
    return payload


def _config_text(
    *,
    name: str,
    policy: str,
    actor: str,
    max_file_bytes: int,
) -> str:
    patterns = "\n".join(f"    - {json.dumps(item)}" for item in DEFAULT_TRACK_PATTERNS)
    denied = "\n".join(f"    - {json.dumps(item)}" for item in DEFAULT_DENY_PATTERNS)
    repository_id = f"ws-{uuid.uuid4().hex[:16]}"
    return f"""schema_version: {REPOSITORY_SCHEMA}
repository_id: {repository_id}
name: {json.dumps(name, ensure_ascii=False)}
question: question.md
actor: {json.dumps(actor, ensure_ascii=False)}
git:
  mode: local
  checkpoint_policy: {policy}
  auto_commit: true
  auto_push: false
  max_file_bytes: {max_file_bytes}
  track_patterns:
{patterns}
  deny_patterns:
{denied}
storage:
  mode: local-cas
  root: .ara-store
  pointer_directory: research-objects
"""


def _ensure_git_identity(repo: Path, name: str | None, email: str | None) -> None:
    current_name = _run_git(repo, ["config", "user.name"], check=False).stdout.strip()
    current_email = _run_git(repo, ["config", "user.email"], check=False).stdout.strip()
    if name:
        _run_git(repo, ["config", "user.name", name])
    elif not current_name:
        _run_git(repo, ["config", "user.name", "XScientist"])
    if email:
        _run_git(repo, ["config", "user.email", email])
    elif not current_email:
        _run_git(repo, ["config", "user.email", "xscientist@localhost"])


def _local_git_identity(repo: Path) -> dict[str, str | None]:
    identity: dict[str, str | None] = {}
    for key in ("user.name", "user.email"):
        result = _run_git(repo, ["config", "--local", "--get", key], check=False)
        identity[key] = result.stdout.rstrip("\n") if result.returncode == 0 else None
    return identity


def _restore_local_git_identity(repo: Path, identity: Mapping[str, str | None]) -> None:
    for key in ("user.name", "user.email"):
        value = identity.get(key)
        if value is None:
            _run_git(
                repo,
                ["config", "--local", "--unset-all", key],
                check=False,
            )
        else:
            _run_git(repo, ["config", "--local", key, value])


def _local_git_config_snapshot(repo: Path) -> str:
    """Return an exact local-config snapshot without exposing its values."""

    result = _run_git(
        repo,
        ["config", "--local", "--null", "--list"],
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def _refuse_private_research_metadata(
    payload: Mapping[str, Any], *, operation: str
) -> None:
    """Reject private free text before it reaches Git or protocol metadata."""

    candidate = dict(payload)
    if redact_sensitive_payload(candidate) != candidate:
        raise ResearchGitError(
            f"privacy gate refused {operation} metadata; matched values were not displayed"
        )


def init_repository(
    path: str | Path,
    *,
    name: str | None = None,
    question: str | None = None,
    policy: str = "milestone",
    actor: str = "xscientist",
    git_user_name: str | None = None,
    git_user_email: str | None = None,
    max_file_bytes: int = 2 * 1024 * 1024,
    commit: bool = True,
) -> CheckpointResult:
    if policy not in {"manual", "stage", "milestone"}:
        raise ResearchGitError("checkpoint policy must be manual, stage, or milestone")
    if max_file_bytes < 1024:
        raise ResearchGitError("max_file_bytes must be at least 1024")
    _refuse_private_research_metadata(
        {
            "name": str(name or ""),
            "question": str(question or ""),
            "actor": str(actor or ""),
        },
        operation="research initialization",
    )
    root = Path(path).expanduser().resolve()
    root_existed = root.exists()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "research.yaml"
    question_path = root / "question.md"
    note_path = root / ".xscientist" / "README.md"
    gitignore_path = root / ".gitignore"
    if config_path.exists() or config_path.is_symlink():
        raise ResearchGitError(f"research repository is already initialized: {root}")

    managed_directories = (
        root / ".xscientist",
        root / "hypotheses",
        root / "claims",
        root / "manuscript",
        root / "checkpoints",
        root / "research-objects",
        root / "ara",
    )
    managed_directories_existed = {
        directory: directory.is_dir() and not directory.is_symlink()
        for directory in managed_directories
    }
    for directory in managed_directories:
        if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
            raise ResearchGitError(
                "research initialization refused an unsafe managed directory: "
                f"{directory.relative_to(root)}"
            )
    for managed_file in (gitignore_path, question_path, note_path):
        if managed_file.is_symlink() or (
            managed_file.exists() and not managed_file.is_file()
        ):
            raise ResearchGitError(
                "research initialization refused an unsafe managed file: "
                f"{managed_file.relative_to(root)}"
            )

    try:
        existing_gitignore = (
            gitignore_path.read_text(encoding="utf-8")
            if gitignore_path.exists()
            else ""
        )
        existing_question = (
            question_path.read_text(encoding="utf-8")
            if question_path.exists()
            else None
        )
        existing_note = (
            note_path.read_text(encoding="utf-8") if note_path.exists() else None
        )
    except (OSError, UnicodeError) as exc:
        raise ResearchGitError(
            "research initialization could not safely read an existing managed file"
        ) from exc

    requested_question = (
        question or "# Research question\n\nDescribe the research question.\n"
    ).rstrip()
    if (
        existing_question is not None
        and question
        and existing_question.rstrip() != requested_question
    ):
        raise ResearchGitError(
            "research initialization refused to overwrite existing question.md"
        )
    if existing_note is not None and existing_note != _REPOSITORY_NOTE:
        raise ResearchGitError(
            "research initialization refused to overwrite existing .xscientist/README.md"
        )

    git_control_path = root / ".git"
    git_control_existed = git_control_path.exists() or git_control_path.is_symlink()
    existing_git_probe = _run_git(root, ["rev-parse", "--show-toplevel"], check=False)
    had_existing_git_repository = bool(
        existing_git_probe.returncode == 0
        and existing_git_probe.stdout.strip()
        and Path(existing_git_probe.stdout.strip()).resolve() == root
    )
    if git_control_existed and not had_existing_git_repository:
        raise ResearchGitError(
            "research initialization refused a pre-existing .git control path "
            "that is not an exact-root Git repository"
        )
    init = _run_git(root, ["init", "-b", "main"], check=False)
    if init.returncode:
        _run_git(root, ["init"])
        _run_git(root, ["symbolic-ref", "HEAD", "refs/heads/main"])
    identity_before = _local_git_identity(root)
    identity_after_transaction = dict(identity_before)
    git_config_after_transaction = _local_git_config_snapshot(root)

    staged_before, _changed_before = _changed_paths(root)
    if staged_before:
        raise ResearchGitError(
            "research initialization refused because the Git index already "
            "contains staged work: " + ", ".join(sorted(staged_before))
        )
    if had_existing_git_repository:
        tracked_managed_dirt = sorted(
            _git_paths(
                root,
                ["diff", "--no-renames", "--name-only", "-z"],
            )
            & {".gitignore", "question.md", ".xscientist/README.md"}
        )
        if tracked_managed_dirt:
            raise ResearchGitError(
                "research initialization refused to absorb existing edits to "
                "managed files: " + ", ".join(tracked_managed_dirt)
            )

    project_name = (name or root.name).strip() or "research-project"
    originals: dict[Path, dict[str, Any] | None] = {
        gitignore_path: (
            {
                "content": gitignore_path.read_bytes(),
                "mode": gitignore_path.stat().st_mode & 0o7777,
            }
            if gitignore_path.exists()
            else None
        ),
        config_path: None,
        question_path: (
            {
                "content": question_path.read_bytes(),
                "mode": question_path.stat().st_mode & 0o7777,
            }
            if question_path.exists()
            else None
        ),
        note_path: (
            {
                "content": note_path.read_bytes(),
                "mode": note_path.stat().st_mode & 0o7777,
            }
            if note_path.exists()
            else None
        ),
    }
    head_before = _head(root)
    transaction_outputs: dict[Path, dict[str, Any]] = {}

    def managed_file_matches(path: Path, expected: dict[str, Any] | None) -> bool:
        if expected is None:
            return not path.exists() and not path.is_symlink()
        if path.is_symlink() or not path.is_file():
            return False
        try:
            return path.read_bytes() == expected.get(
                "content"
            ) and path.stat().st_mode & 0o7777 == expected.get("mode")
        except OSError:
            return False

    def write_transaction_file(path: Path, content: str) -> None:
        """Record the exact write so rollback never erases a concurrent edit."""

        original = originals.get(path)
        if not managed_file_matches(path, original):
            action = "appeared" if original is None else "changed"
            raise ResearchGitError(
                "research initialization stopped because a managed file "
                f"{action} concurrently: {path.relative_to(root)}"
            )

        _atomic_write_text(path, content)
        transaction_outputs[path] = {
            "content": content.encode("utf-8"),
            "mode": path.stat().st_mode & 0o7777,
        }

    def revalidate_managed_files() -> None:
        for managed_file, original in originals.items():
            expected = transaction_outputs.get(managed_file, original)
            if not managed_file_matches(managed_file, expected):
                raise ResearchGitError(
                    "research initialization stopped because a managed file "
                    f"changed concurrently: {managed_file.relative_to(root)}"
                )

    try:
        merged_gitignore = _merged_research_gitignore(existing_gitignore)
        if not gitignore_path.exists() or merged_gitignore != existing_gitignore:
            write_transaction_file(gitignore_path, merged_gitignore)
        write_transaction_file(
            config_path,
            _config_text(
                name=project_name,
                policy=policy,
                actor=actor,
                max_file_bytes=max_file_bytes,
            ),
        )
        if existing_question is None:
            write_transaction_file(question_path, requested_question + "\n")
        if existing_note is None:
            write_transaction_file(note_path, _REPOSITORY_NOTE)
        for directory in managed_directories:
            directory.mkdir(parents=True, exist_ok=True)

        if not commit:
            # Batch callers record typed objects before their first checkpoint.
            # Establish the requested local author only after every destructive
            # preflight has passed; the exception path below restores it if this
            # initialization transaction fails.
            _ensure_git_identity(root, git_user_name, git_user_email)
            identity_after_transaction = _local_git_identity(root)
            git_config_after_transaction = _local_git_config_snapshot(root)
            revalidate_managed_files()
            return CheckpointResult(
                created=False,
                committed=False,
                reason="repository initialized without a commit",
            )
        _staged, changed = _changed_paths(root)
        managed_changes = sorted(
            changed
            & {
                ".gitignore",
                "research.yaml",
                "question.md",
                ".xscientist/README.md",
            }
        )
        _ensure_git_identity(root, git_user_name, git_user_email)
        identity_after_transaction = _local_git_identity(root)
        git_config_after_transaction = _local_git_config_snapshot(root)
        revalidate_managed_files()
        return create_checkpoint(
            root,
            stage="init",
            subject=f"initialize {project_name}",
            summary="Initialize the local scientific repository and research question.",
            status="completed",
            actor=actor,
            # Initializing inside an existing Git repository must not absorb
            # unrelated dirty work. A new repository intentionally captures its
            # policy-eligible workspace as the baseline scientific checkpoint.
            only_paths=managed_changes if had_existing_git_repository else None,
            allow_checkpoint_only=True,
        )
    except BaseException:
        # Keep initialization non-destructive when a privacy/checkpoint gate
        # fails before a commit is created.
        if _head(root) == head_before:
            if had_existing_git_repository:
                # ``create_checkpoint`` restores only its own staged paths on
                # failure.  Do not reset the whole index here: another process
                # may have staged unrelated work while initialization was
                # running.
                current_identity = _local_git_identity(root)
                selective_identity = dict(current_identity)
                for key in ("user.name", "user.email"):
                    if current_identity.get(key) == identity_after_transaction.get(key):
                        selective_identity[key] = identity_before.get(key)
                _restore_local_git_identity(root, selective_identity)
            for managed_file, original in originals.items():
                expected = transaction_outputs.get(managed_file)
                if expected is None or not managed_file.is_file():
                    continue
                try:
                    current_content = managed_file.read_bytes()
                    current_mode = managed_file.stat().st_mode & 0o7777
                except OSError:
                    continue
                if current_content != expected.get(
                    "content"
                ) or current_mode != expected.get("mode"):
                    # A concurrent writer owns the newer state.  Preserving it
                    # is safer than claiming a complete rollback.
                    continue
                if original is None:
                    managed_file.unlink(missing_ok=True)
                else:
                    content = original.get("content")
                    if not isinstance(content, bytes):  # pragma: no cover - invariant
                        raise ResearchGitError(
                            "research initialization rollback snapshot is invalid"
                        )
                    managed_file.parent.mkdir(parents=True, exist_ok=True)
                    managed_file.write_bytes(content)
                    original_mode = original.get("mode")
                    managed_file.chmod(
                        int(original_mode) if isinstance(original_mode, int) else 0o600
                    )

            for directory in reversed(managed_directories):
                if managed_directories_existed[directory] or not directory.exists():
                    continue
                if directory.is_symlink() or not directory.is_dir():
                    continue
                try:
                    directory.rmdir()
                except OSError:
                    # Unknown content may have arrived concurrently.  Keep it.
                    pass

            if not git_control_existed and (
                git_control_path.exists() or git_control_path.is_symlink()
            ):
                config_unchanged = (
                    _local_git_config_snapshot(root) == git_config_after_transaction
                )
                staged_probe = _run_git(
                    root,
                    [
                        "diff",
                        "--cached",
                        "--ita-visible-in-index",
                        "--name-only",
                        "-z",
                    ],
                    check=False,
                )
                staged_now = {
                    item
                    for item in staged_probe.stdout.split("\0")
                    if item and not item.endswith("/")
                }
                transaction_paths = {
                    path.relative_to(root).as_posix() for path in transaction_outputs
                }
                has_external_stage = staged_probe.returncode != 0 or bool(
                    staged_now - transaction_paths
                )
                if (
                    config_unchanged
                    and not has_external_stage
                    and git_control_path.is_dir()
                    and not git_control_path.is_symlink()
                ):
                    shutil.rmtree(git_control_path)
            if not root_existed and root.is_dir() and not any(root.iterdir()):
                root.rmdir()
        raise


def _research_object_path(root: Path, payload: dict[str, Any]) -> Path:
    kind = str(payload["kind"])
    object_id = str(payload["object_id"])
    return root / ".xscientist" / "objects" / kind / f"{object_id}.json"


def _refuse_private_research_object(payload: dict[str, Any]) -> None:
    if redact_sensitive_payload(payload) != payload:
        raise ResearchGitError(
            "privacy gate refused the research object; matched values were not displayed"
        )


def _record_research_object_locked(
    root: Path,
    *,
    kind: str,
    payload: dict[str, Any],
    state: str = "draft",
    relations: Sequence[dict[str, Any]] = (),
    actor: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    semantic_profile: dict[str, Any] | None = None,
) -> ResearchObjectResult:
    normalized_relations: list[dict[str, Any]] = []
    for relation in relations:
        normalized_relation = dict(relation)
        target = str(normalized_relation.get("target") or "")
        if (
            target.startswith("@")
            or target.startswith("urn:xscientist:research-object:sha256:")
            or re.fullmatch(r"rso-[0-9a-f]{6,16}", target)
        ):
            normalized_relation["target"] = resolve_research_object_id(root, target)
        normalized_relations.append(normalized_relation)
    _refuse_private_research_object(
        {
            "kind": kind,
            "payload": payload,
            "relations": normalized_relations,
            "actor": actor or {},
            "provenance": provenance or {},
            "semantic_profile": semantic_profile or {},
        }
    )
    try:
        research_object = build_research_object(
            kind=kind,
            payload=payload,
            state=state,
            relations=normalized_relations,
            actor=actor,
            provenance=provenance,
            semantic_profile=semantic_profile,
        )
    except ResearchObjectError as exc:
        raise ResearchGitError(str(exc)) from exc
    _refuse_private_research_object(research_object)
    for relation in research_object.get("relations") or []:
        relation_type = str(relation.get("type") or "")
        if relation_type not in RESEARCH_RELATION_TYPES:
            continue
        relation_target = str(relation.get("target") or "")
        try:
            target_object = load_research_object(root, relation_target)
        except ResearchGitError as exc:
            raise ResearchGitError(
                f"built-in research relation {relation_type} references missing "
                f"object: {relation_target}"
            ) from exc
        expected_kinds = RESEARCH_RELATION_TARGET_KINDS.get(relation_type, ())
        actual_kind = str(target_object.get("kind") or "")
        if expected_kinds and actual_kind not in expected_kinds:
            raise ResearchGitError(
                f"built-in research relation {relation_type} requires target kind "
                f"{' or '.join(expected_kinds)}; got {actual_kind}: {relation_target}"
            )
    target = _research_object_path(root, research_object)
    if target.exists():
        try:
            existing = validate_research_object(
                json.loads(target.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ResearchObjectError) as exc:
            raise ResearchGitError(
                f"existing research object is damaged: {target.relative_to(root)}"
            ) from exc
        if existing["content_hash"] != research_object["content_hash"]:
            raise ResearchGitError(
                f"research object identifier collision: {research_object['object_id']}"
            )
        persisted = existing
        created = False
    else:
        _atomic_write_json(target, research_object)
        findings = scan_paths(root, [target.relative_to(root)])
        if findings:
            target.unlink(missing_ok=True)
            _fsync_directory(target.parent)
            raise ResearchGitError(
                "privacy gate refused the research object; matched values were not displayed:\n"
                + format_privacy_findings(findings)
            )
        persisted = research_object
        created = True
    return ResearchObjectResult(
        created=created,
        path=target,
        object_id=str(persisted["object_id"]),
        qualified_id=(
            str(persisted["qualified_id"]) if persisted.get("qualified_id") else None
        ),
        object_hash=str(persisted["content_hash"]),
        kind=str(persisted["kind"]),
        state=str(persisted["state"]),
    )


def record_research_object(
    repo: str | Path,
    *,
    kind: str,
    payload: dict[str, Any],
    state: str = "draft",
    relations: Sequence[dict[str, Any]] = (),
    actor: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    semantic_profile: dict[str, Any] | None = None,
) -> ResearchObjectResult:
    """Record one immutable scientific object in the repository working state.

    Recording is idempotent: the same semantic object resolves to the same
    object ID and path. The caller creates a research commit separately so one
    atomic scientific transition may bind several related objects.
    """

    root = _repository_root(repo)
    with _repository_lock(root):
        return _record_research_object_locked(
            root,
            kind=kind,
            payload=payload,
            state=state,
            relations=relations,
            actor=actor,
            provenance=provenance,
            semantic_profile=semantic_profile,
        )


def resolve_research_object_id(
    repo: str | Path,
    selector: str,
    *,
    expected_kind: str | None = None,
) -> str:
    """Resolve a full ID, unique prefix, or ``@latest:<kind>`` selector."""

    root = _repository_root(repo)
    normalized = str(selector or "").strip()
    object_root = root / ".xscientist" / "objects"
    if normalized.startswith("@latest"):
        prefix, separator, selected_kind = normalized.partition(":")
        if prefix != "@latest" or (separator and not selected_kind):
            raise ResearchGitError("object selector must use @latest:<kind>")
        kind = selected_kind or str(expected_kind or "")
        if not kind:
            raise ResearchGitError("@latest requires an object kind")
        if expected_kind and kind != expected_kind:
            raise ResearchGitError(
                f"object selector kind mismatch: expected {expected_kind}, got {kind}"
            )
        candidates = list_research_objects(root, kind=kind)
        if not candidates:
            raise ResearchGitError(
                f"no research objects found for selector: {normalized}"
            )
        commit_order = _research_object_commit_order(root, ref="HEAD", kind=kind)

        candidates_by_path = {
            (
                f".xscientist/objects/{kind}/"
                f"{str(item.get('object_id') or '')}.json"
            ): item
            for item in candidates
        }
        uncommitted = [
            item
            for path, item in candidates_by_path.items()
            if path not in commit_order
        ]
        if len(uncommitted) > 1:
            raise ResearchGitError(
                f"@latest:{kind} is ambiguous: multiple uncommitted objects have "
                "no Git introduction order; commit them separately or use an ID"
            )
        if uncommitted:
            return str(uncommitted[0]["object_id"])
        latest_sequence = max(commit_order[path] for path in candidates_by_path)
        latest = [
            item
            for path, item in candidates_by_path.items()
            if commit_order[path] == latest_sequence
        ]
        if len(latest) != 1:
            raise ResearchGitError(
                f"@latest:{kind} is ambiguous: multiple objects share the latest "
                "Git checkpoint; use an explicit ID"
            )
        return str(latest[0]["object_id"])
    if normalized.startswith("urn:xscientist:research-object:sha256:"):
        matches = [
            item
            for item in list_research_objects(root, kind=expected_kind)
            if item.get("qualified_id") == normalized
        ]
        if not matches:
            raise ResearchGitError(f"research object not found: {normalized}")
        if len(matches) != 1:
            raise ResearchGitError(f"duplicate qualified research object: {normalized}")
        resolved = str(matches[0]["object_id"])
    elif re.fullmatch(r"rso-[0-9a-f]{16}", normalized):
        resolved = normalized
    elif re.fullmatch(r"rso-[0-9a-f]{6,15}", normalized):
        matches = sorted(object_root.glob(f"*/{normalized}*.json"))
        if not matches:
            raise ResearchGitError(f"research object not found: {normalized}")
        if len(matches) != 1:
            raise ResearchGitError(f"ambiguous research object prefix: {normalized}")
        resolved = matches[0].stem
    else:
        raise ResearchGitError("invalid research object identifier or selector")
    if expected_kind:
        matches = sorted(object_root.glob(f"*/{resolved}.json"))
        if len(matches) != 1 or matches[0].parent.name != expected_kind:
            raise ResearchGitError(
                f"research object has wrong kind; expected {expected_kind}: {resolved}"
            )
    return resolved


def _research_object_commit_order(
    repo: Path,
    *,
    ref: str,
    kind: str | None,
) -> dict[str, int]:
    """Map object paths to their latest first-parent introduction sequence."""

    if kind is not None and re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", kind) is None:
        raise ResearchGitError("research object kind is invalid")
    exists = _run_git(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
    if exists.returncode:
        return {}
    raw = _run_git(
        repo,
        [
            "log",
            "--first-parent",
            "--reverse",
            "--no-renames",
            "--diff-merges=first-parent",
            "--diff-filter=AD",
            "--format=%x1e%H",
            "--name-status",
            exists.stdout.strip(),
            "--",
            (
                f".xscientist/objects/{kind}"
                if kind is not None
                else ".xscientist/objects"
            ),
        ],
    ).stdout
    order: dict[str, int] = {}
    sequence = 0
    for block in raw.split("\x1e"):
        lines = [line for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        commit_hash = lines[0].strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", commit_hash) is None:
            raise ResearchGitError("cannot parse research object commit order")
        sequence += 1
        for line in lines[1:]:
            status, separator, raw_path = line.partition("\t")
            if not separator or status not in {"A", "D"}:
                raise ResearchGitError(
                    "cannot parse research object history transition"
                )
            path = _normalise_relative(raw_path)
            if status == "A":
                order[path] = sequence
            else:
                order.pop(path, None)
    return order


def research_object_introduction_order(
    repo: str | Path,
    *,
    ref: str = "HEAD",
) -> dict[str, int]:
    """Return Git-topology introduction order for objects visible at ``ref``.

    Objects introduced together intentionally share a sequence. Missing IDs are
    uncommitted working-state objects and therefore have no internal chronology.
    """

    root = _repository_root(repo)
    path_order = _research_object_commit_order(root, ref=ref, kind=None)
    result: dict[str, int] = {}
    for path, sequence in path_order.items():
        parts = PurePosixPath(path).parts
        if (
            len(parts) == 4
            and parts[:2] == (".xscientist", "objects")
            and re.fullmatch(r"rso-[0-9a-f]{16}\.json", parts[3])
        ):
            result[PurePosixPath(parts[3]).stem] = sequence
    return result


def load_research_object(repo: str | Path, object_id: str) -> dict[str, Any]:
    """Load and validate one typed Research VCS object from working state."""

    root = _repository_root(repo)
    normalized = resolve_research_object_id(root, object_id)
    matches = sorted((root / ".xscientist" / "objects").glob(f"*/{normalized}.json"))
    if not matches:
        raise ResearchGitError(f"research object not found: {normalized}")
    if len(matches) != 1:
        raise ResearchGitError(f"duplicate research object identifier: {normalized}")
    try:
        return validate_research_object(
            json.loads(matches[0].read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ResearchObjectError) as exc:
        raise ResearchGitError(f"research object is damaged: {normalized}") from exc


def list_research_objects(
    repo: str | Path,
    *,
    kind: str | None = None,
    state: str | None = None,
    max_objects: int | None = None,
    max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    """List validated typed objects in deterministic order.

    ``max_objects``/``max_bytes`` are optional read-only safety caps used by
    audit callers.  The default remains the historical complete listing API;
    capped callers stop before opening an unbounded object store.
    """

    root = _repository_root(repo)
    object_root = root / ".xscientist" / "objects"
    if max_objects is not None and (
        isinstance(max_objects, bool)
        or not isinstance(max_objects, int)
        or max_objects <= 0
        or max_objects > _HARD_MAX_RESEARCH_OBJECTS
    ):
        raise ValueError("max_objects must be a positive integer")
    if max_bytes is not None and (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or max_bytes > _HARD_MAX_STORE_BYTES
    ):
        raise ValueError("max_bytes must be a positive integer")
    if object_root.is_symlink() or not object_root.is_dir():
        return []

    if max_objects is None and max_bytes is None:
        pattern = f"{kind}/*.json" if kind else "*/*.json"
        paths = [
            path
            for path in sorted(object_root.glob(pattern))
            if path.is_file() and not path.is_symlink()
        ]
    else:
        # The capped path intentionally avoids ``glob`` and materialising an
        # unbounded directory listing.  It is used by audit/report code where
        # a hostile or simply very large object store must not become an I/O
        # or memory denial of service.
        paths = []
        consumed_bytes = 0
        entries_seen = 0
        entry_cap = min(
            _HARD_MAX_STORE_ENTRIES,
            max(1024, (max_objects or 1) * 8),
        )
        kind_paths: list[Path] = []
        try:
            with os.scandir(object_root) as kind_entries:
                for entry in kind_entries:
                    entries_seen += 1
                    if entries_seen > entry_cap:
                        break
                    if (
                        entry.is_symlink()
                        or not entry.is_dir(follow_symlinks=False)
                        or (kind is not None and entry.name != kind)
                    ):
                        continue
                    kind_paths.append(Path(entry.path))
        except OSError:
            return []
        for kind_path in sorted(kind_paths):
            if max_objects is not None and len(paths) >= max_objects:
                break
            try:
                with os.scandir(kind_path) as object_entries:
                    for entry in object_entries:
                        entries_seen += 1
                        if entries_seen > entry_cap:
                            break
                        if (
                            entry.is_symlink()
                            or not entry.is_file(follow_symlinks=False)
                            or not entry.name.endswith(".json")
                        ):
                            continue
                        try:
                            current_size = int(
                                entry.stat(follow_symlinks=False).st_size
                            )
                        except (OSError, ValueError):
                            continue
                        if (
                            max_bytes is not None
                            and consumed_bytes + current_size > max_bytes
                        ):
                            break
                        paths.append(Path(entry.path))
                        consumed_bytes += current_size
                        if max_objects is not None and len(paths) >= max_objects:
                            break
            except OSError:
                continue
        paths.sort()
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = validate_research_object(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ResearchObjectError) as exc:
            raise ResearchGitError(
                f"research object is damaged: {path.relative_to(root)}"
            ) from exc
        if state is None or payload["state"] == state:
            rows.append(payload)
    return sorted(rows, key=lambda row: (row["kind"], row["object_id"]))


def _git_paths(repo: Path, args: Sequence[str]) -> set[str]:
    completed = _run_git(repo, list(args))
    return {
        item for item in completed.stdout.split("\0") if item and not item.endswith("/")
    }


def _changed_paths(repo: Path) -> tuple[set[str], set[str]]:
    staged = _git_paths(
        repo,
        [
            "diff",
            "--no-renames",
            "--cached",
            "--ita-visible-in-index",
            "--name-only",
            "-z",
        ],
    )
    unstaged = _git_paths(repo, ["diff", "--no-renames", "--name-only", "-z"])
    untracked = _git_paths(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    return staged, unstaged | untracked


def _worktree_change_summary(
    repo: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify changes by whether a scientific transition may overwrite them.

    Generated views and other policy-excluded untracked files are preserved by
    Git operations and therefore do not make the scientific worktree dirty.
    Tracked, staged, research-eligible, and explicitly selected changes do.
    """

    repository_config = config or load_repository_config(repo)
    backend_staged, changed = _changed_paths(repo)
    tracked = (
        _git_paths(repo, ["ls-files", "-z", "--", *sorted(changed)])
        if changed
        else set()
    )
    eligible, excluded = _select_paths(repo, repository_config, changed)
    research_stage_payload = _load_research_stage(repo)
    research_stage = [
        str(item["path"]) for item in research_stage_payload.get("entries") or []
    ]
    blocking = set(backend_staged) | tracked | set(eligible) | set(research_stage)
    return {
        "clean": not blocking,
        "backend_staged": sorted(backend_staged),
        "tracked": sorted(tracked),
        "eligible": eligible,
        "excluded": excluded,
        "research_stage_head": research_stage_payload.get("head"),
        "research_stage": sorted(research_stage),
        "blocking": sorted(blocking),
    }


def _research_stage_path(repo: Path) -> Path:
    raw = _run_git(repo, ["rev-parse", "--git-path", "xscientist/stage.json"]).stdout
    path = Path(raw.strip())
    return path if path.is_absolute() else repo / path


def _worktree_fingerprint(repo: Path, relative: str) -> str:
    target = repo / _normalise_relative(relative)
    if target.is_symlink():
        raise ResearchGitError(f"research staging refuses symbolic links: {relative}")
    if not target.exists():
        return "deleted"
    if not target.is_file():
        raise ResearchGitError(f"research staging requires a regular file: {relative}")
    return _hash_file(target)


def _stage_payload(repo: Path, entries: dict[str, str]) -> dict[str, Any]:
    base = {
        "schema_version": "xscientist.research-stage.v1",
        "head": _head(repo),
        "entries": [
            {"path": path, "fingerprint": entries[path]} for path in sorted(entries)
        ],
    }
    return {**base, "content_hash": content_hash(base)}


def _load_research_stage(repo: Path) -> dict[str, Any]:
    path = _research_stage_path(repo)
    # The stage file is a small control-plane receipt.  Refuse symlinks and
    # oversized/deep JSON before parsing so status/audit commands cannot be
    # turned into an unbounded read of an externally-owned file.
    if path.is_symlink():
        raise ResearchGitError("native research stage is outside the repository")
    if not path.is_file():
        return _stage_payload(repo, {})
    try:
        if path.stat().st_size > _MAX_RESEARCH_STAGE_BYTES:
            raise ResearchGitError("native research stage exceeds safety limit")
        raw = path.read_bytes().decode("utf-8")

        def reject_constant(token: str) -> None:
            raise ValueError(f"non-finite JSON constant: {token}")

        payload = json.loads(raw, parse_constant=reject_constant)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        MemoryError,
    ) as exc:
        raise ResearchGitError("native research stage is damaged") from exc
    if not isinstance(payload, dict):
        raise ResearchGitError("native research stage is damaged")
    base = {key: value for key, value in payload.items() if key != "content_hash"}
    if (
        payload.get("schema_version") != "xscientist.research-stage.v1"
        or payload.get("content_hash") != content_hash(base)
        or not isinstance(payload.get("entries"), list)
    ):
        raise ResearchGitError("native research stage failed integrity validation")
    if len(payload["entries"]) > _MAX_RESEARCH_STAGE_ENTRIES:
        raise ResearchGitError("native research stage exceeds entry limit")
    seen: set[str] = set()
    for item in payload["entries"]:
        if not isinstance(item, dict) or set(item) != {"path", "fingerprint"}:
            raise ResearchGitError("native research stage is damaged")
        relative = _normalise_relative(str(item["path"]))
        if relative in seen or not isinstance(item["fingerprint"], str):
            raise ResearchGitError("native research stage is damaged")
        seen.add(relative)
    return payload


def _write_research_stage(repo: Path, entries: dict[str, str]) -> None:
    path = _research_stage_path(repo)
    if entries:
        _atomic_write_json(path, _stage_payload(repo, entries))
        return
    path.unlink(missing_ok=True)
    if path.parent.exists():
        _fsync_directory(path.parent)


def research_stage(
    repo: str | Path,
    paths: Sequence[str] = (),
    *,
    all_changes: bool = False,
) -> ResearchStageResult:
    """Select exact scientific changes for the next research commit."""

    root = _repository_root(repo)
    with _repository_lock(root):
        if all_changes and paths:
            raise ResearchGitError(
                "use either explicit research paths or all_changes=True"
            )
        config = load_repository_config(root)
        git_staged, changed = _changed_paths(root)
        if git_staged:
            raise ResearchGitError(
                "research staging refused because the backend index is not clean"
            )
        requested = (
            set(changed)
            if all_changes
            else {_normalise_relative(path) for path in paths}
        )
        if not requested:
            raise ResearchGitError("specify research paths or use all_changes=True")
        missing = sorted(requested - changed)
        if missing:
            raise ResearchGitError(
                "research paths have no working-state change: " + ", ".join(missing)
            )
        selected, excluded = _select_paths(root, config, requested, explicit=paths)
        if not selected:
            return ResearchStageResult(paths=(), excluded=tuple(excluded))
        findings = scan_paths(root, selected)
        if findings:
            raise ResearchGitError(
                "privacy gate refused research staging; matched values were not displayed:\n"
                + format_privacy_findings(findings)
            )
        current = _load_research_stage(root)
        if current.get("head") != _head(root) and current.get("entries"):
            raise ResearchGitError(
                "native research stage belongs to another repository state; unstage it first"
            )
        entries = {
            str(item["path"]): str(item["fingerprint"])
            for item in current.get("entries") or []
        }
        before = set(entries)
        for relative in selected:
            entries[relative] = _worktree_fingerprint(root, relative)
        _write_research_stage(root, entries)
        return ResearchStageResult(
            paths=tuple(sorted(entries)),
            added=tuple(sorted(set(entries) - before)),
            excluded=tuple(excluded),
        )


def research_unstage(
    repo: str | Path,
    paths: Sequence[str] = (),
    *,
    all_paths: bool = False,
) -> ResearchStageResult:
    """Remove paths from the native stage without modifying research files."""

    root = _repository_root(repo)
    with _repository_lock(root):
        current = _load_research_stage(root)
        entries = {
            str(item["path"]): str(item["fingerprint"])
            for item in current.get("entries") or []
        }
        requested = (
            set(entries) if all_paths else {_normalise_relative(path) for path in paths}
        )
        if not requested:
            raise ResearchGitError("specify staged paths or use all_paths=True")
        unknown = sorted(requested - set(entries))
        if unknown:
            raise ResearchGitError(
                "research paths are not staged: " + ", ".join(unknown)
            )
        removed = tuple(sorted(requested))
        for relative in requested:
            entries.pop(relative, None)
        _write_research_stage(root, entries)
        return ResearchStageResult(paths=tuple(sorted(entries)), removed=removed)


def _select_paths(
    repo: Path,
    config: dict[str, Any],
    paths: Iterable[str],
    *,
    explicit: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    git_config = config["git"]
    allowed_patterns = tuple(git_config.get("track_patterns") or DEFAULT_TRACK_PATTERNS)
    denied_patterns = tuple(git_config.get("deny_patterns") or DEFAULT_DENY_PATTERNS)
    max_bytes = int(git_config.get("max_file_bytes") or 2 * 1024 * 1024)
    selected: list[str] = []
    excluded: list[str] = []
    for raw in sorted(set(paths)):
        try:
            relative = _normalise_relative(raw)
        except ResearchGitError:
            excluded.append(f"{raw} (unsafe path)")
            continue
        # A generated example contains placeholders only and still passes the
        # secret scanner below; real .env variants remain denied.
        if relative != ".env.example" and _matches(relative, denied_patterns):
            excluded.append(f"{relative} (denied pattern)")
            continue
        if not (
            _matches(relative, allowed_patterns)
            or _path_is_explicit(relative, explicit)
        ):
            excluded.append(f"{relative} (not tracked by policy)")
            continue
        target = repo / relative
        if target.is_symlink():
            excluded.append(f"{relative} (symbolic links are not checkpoint-safe)")
            continue
        if target.exists() and target.is_file() and target.stat().st_size > max_bytes:
            excluded.append(
                f"{relative} ({target.stat().st_size} bytes exceeds {max_bytes}; add as research object)"
            )
            continue
        selected.append(relative)
    return selected, excluded


def _head(repo: Path) -> str | None:
    completed = _run_git(repo, ["rev-parse", "HEAD"], check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _branch(repo: Path) -> str:
    completed = _run_git(repo, ["symbolic-ref", "--short", "HEAD"], check=False)
    return completed.stdout.strip() or "detached"


def _checkpoint_id_from_commit(repo: Path, commit: str) -> str | None:
    body = _run_git(repo, ["show", "-s", "--format=%B", commit]).stdout
    values = [
        line.split(":", 1)[1].strip()
        for line in body.splitlines()
        if line.startswith("Research-Checkpoint:")
    ]
    return values[-1] if values else None


def _checkpoint_by_id_at_commit(
    repo: Path,
    commit: str,
    checkpoint_id: str,
    *,
    candidate_paths: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    tree = (
        list(candidate_paths)
        if candidate_paths is not None
        else _run_git(
            repo, ["ls-tree", "-r", "--name-only", commit]
        ).stdout.splitlines()
    )
    checkpoint_paths = sorted(
        path
        for path in tree
        if path.startswith("checkpoints/") and path.endswith(".json")
    )
    if candidate_paths is not None and len(checkpoint_paths) > 32:
        raise ResearchGitError(
            f"too many checkpoint candidates for {checkpoint_id} at {commit}"
        )
    for checkpoint_path in checkpoint_paths:
        try:
            payload = json.loads(
                _run_git(repo, ["show", f"{commit}:{checkpoint_path}"]).stdout
            )
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("checkpoint_id") == checkpoint_id:
            return checkpoint_path, payload
    return None


def _latest_checkpoint_record(
    repo: Path,
    start: str = "HEAD",
) -> tuple[str, str, dict[str, Any]] | None:
    selected_commit = _run_git(repo, ["rev-parse", start], check=False)
    if selected_commit.returncode:
        return None
    selected = selected_commit.stdout.strip()
    history = _run_git(
        repo,
        ["log", "--first-parent", "--format=%H", start],
        check=False,
    )
    if history.returncode:
        return None
    for commit in history.stdout.splitlines():
        checkpoint_id = _checkpoint_id_from_commit(repo, commit)
        if not checkpoint_id:
            continue
        resolved = _checkpoint_by_id_at_commit(repo, commit, checkpoint_id)
        if resolved is not None:
            path, _payload_at_origin = resolved
            selected_blob = _run_git(
                repo,
                ["show", f"{selected}:{path}"],
                check=False,
            )
            if selected_blob.returncode:
                raise ResearchGitError(
                    f"selected commit removed its latest research checkpoint: "
                    f"{selected}:{path}"
                )
            try:
                payload = json.loads(selected_blob.stdout)
            except json.JSONDecodeError as exc:
                raise ResearchGitError(
                    f"invalid checkpoint at {selected}:{path}"
                ) from exc
            return commit, path, payload
    return None


def _checkpoint_parent_records(
    repo: Path,
    *,
    stage: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    """Resolve exact scientific parents without laundering a raw Git tip.

    An older implementation walked past an uncheckpointed ``HEAD`` and reused
    the nearest Research VCS ancestor. The newly created checkpoint then made
    the raw transition look like an ordinary scientific parent. Only an
    explicit migration may bridge that gap; the first ``init`` has no reachable
    Research VCS ancestor and remains valid on top of an existing Git project.
    """

    head = _head(repo)
    if head is None:
        return []
    checkpoint_at_head = _checkpoint_id_from_commit(repo, head)
    if checkpoint_at_head:
        try:
            path, payload = _checkpoint_at_commit(repo, head)
        except ResearchGitError as exc:
            if stage != "migration":
                raise ResearchGitError(
                    "checkpoint refused because HEAD is not an exact, hash-valid "
                    "Research VCS checkpoint; use an explicit migration"
                ) from exc
        else:
            return [(head, path, payload)]
    ancestry = _run_git(repo, ["rev-list", "--parents", "-n", "1", head]).stdout.split()
    # Start at the backend parents, never at the uncheckpointed tip itself.
    # This still finds the nearest scientific parent on every side of a raw
    # merge, but lets the caller distinguish that legacy ancestry from an exact
    # checkpoint at HEAD.
    starts = ancestry[1:]
    records: list[tuple[str, str, dict[str, Any]]] = []
    seen_hashes: set[str] = set()
    for start in starts:
        record = _latest_checkpoint_record(repo, start)
        if record is None:
            continue
        _validate_checkpoint_payload(
            record[2],
            checkpoint_path=f"{record[0]}:{record[1]}",
        )
        checkpoint_hash = str(record[2].get("content_hash") or "")
        if checkpoint_hash and checkpoint_hash not in seen_hashes:
            records.append(record)
            seen_hashes.add(checkpoint_hash)
    if records and stage != "migration":
        raise ResearchGitError(
            "checkpoint refused because HEAD is an uncheckpointed raw Git commit "
            "above Research VCS history; use an explicit migration"
        )
    return records


def _append_checkpoint_parent_records(
    repo: Path,
    records: list[tuple[str, str, dict[str, Any]]],
    refs: Sequence[str],
) -> list[tuple[str, str, dict[str, Any]]]:
    """Add explicit scientific parents while preserving content-hash uniqueness."""

    output = list(records)
    seen = {str(record[2].get("content_hash") or "") for record in records}
    for ref in refs:
        resolved = _run_git(
            repo,
            ["rev-parse", "--verify", f"{ref}^{{commit}}"],
        ).stdout.strip()
        checkpoint_path, payload = _checkpoint_at_commit(repo, resolved)
        record = (resolved, checkpoint_path, payload)
        checkpoint_hash = str(payload.get("content_hash") or "")
        if checkpoint_hash and checkpoint_hash not in seen:
            output.append(record)
            seen.add(checkpoint_hash)
    return output


def _previous_checkpoint(repo: Path) -> dict[str, Any] | None:
    record = _latest_checkpoint_record(repo)
    if record is not None:
        return record[2]
    # A repository initialized with --no-commit can still have an intentional
    # uncommitted checkpoint. Fall back to the worktree only when no Git-bound
    # checkpoint exists.
    candidates = sorted((repo / "checkpoints").glob("*.json"))
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("protocol_kind") == "research_checkpoint"
        ):
            return payload
    return None


def _manifest_refs(repo: Path, paths: Iterable[str]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for relative in sorted(set(paths)):
        path = repo / relative
        if path.name != "manifest.json" or not path.is_file():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchGitError(f"ARA manifest is invalid: {relative}") from exc
        if not isinstance(manifest, dict):
            raise ResearchGitError(f"ARA manifest is not an object: {relative}")
        reference = {"path": relative, "manifest_hash": hash_manifest(manifest)}
        graph_path = path.parent / "exploration_graph.json"
        if graph_path.is_file():
            try:
                graph = json.loads(graph_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResearchGitError(
                    f"ARA exploration graph is invalid: "
                    f"{graph_path.relative_to(repo).as_posix()}"
                ) from exc
            if not isinstance(graph, dict):
                raise ResearchGitError(
                    f"ARA exploration graph is not an object: "
                    f"{graph_path.relative_to(repo).as_posix()}"
                )
            reference["exploration_graph_hash"] = content_hash(graph)
        refs.append(reference)
    return refs


def _touched_ara_manifest_paths(repo: Path, paths: Iterable[str]) -> set[str]:
    """Resolve changed ARA companion files back to their owning manifest."""

    ara_root = (repo / "ara").resolve()
    manifests: set[str] = set()
    for relative in paths:
        normalized = _normalise_relative(relative)
        pure = PurePosixPath(normalized)
        if not pure.parts or pure.parts[0] != "ara":
            continue
        if pure.name == "manifest.json":
            manifests.add(normalized)
            continue
        candidate = (repo / normalized).resolve()
        current = candidate if candidate.is_dir() else candidate.parent
        while current != ara_root and ara_root in current.parents:
            manifest = current / "manifest.json"
            if manifest.is_file():
                manifests.add(manifest.relative_to(repo).as_posix())
                break
            current = current.parent
    return manifests


def _pointer_hash_valid(payload: dict[str, Any]) -> bool:
    expected = str(payload.get("pointer_hash") or "")
    scrubbed = {key: value for key, value in payload.items() if key != "pointer_hash"}
    return bool(expected and content_hash(scrubbed) == expected)


def _validate_pointer_payload(
    payload: Any,
    *,
    pointer_path: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResearchGitError(f"invalid research object pointer: {pointer_path}")
    try:
        validate_json(payload, load_schema("research_object_pointer"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"invalid research object pointer {pointer_path}: {exc.message}"
        ) from exc
    if not _pointer_hash_valid(payload):
        raise ResearchGitError(
            f"research object pointer hash verification failed: {pointer_path}"
        )
    logical_path = _validate_research_logical_path(
        str(payload.get("logical_path") or "")
    )
    store_relpath = _normalise_relative(str(payload.get("store_relpath") or ""))
    digest = str(payload["object_hash"]).split(":", 1)[1]
    if PurePosixPath(store_relpath).parts[-3:] != ("objects", "sha256", digest):
        raise ResearchGitError(
            f"research object pointer has inconsistent CAS path: {pointer_path}"
        )
    pointer_name = PurePosixPath(pointer_path).name
    logical_digest = hashlib.sha256(logical_path.encode("utf-8")).hexdigest()[:16]
    valid_names = {
        f"sha256-{digest}.json",  # Legacy one-pointer-per-object layout.
        f"sha256-{digest}-{logical_digest}.json",
    }
    if pointer_name.startswith("sha256-") and pointer_name not in valid_names:
        raise ResearchGitError(
            "research object pointer filename does not match its hash and logical "
            f"path: {pointer_path}"
        )
    return payload


def _record_pointer(
    records: dict[str, list[dict[str, Any]]],
    logical_bindings: dict[tuple[str, ...], tuple[str, str]],
    *,
    object_hash: str,
    record: dict[str, Any],
) -> None:
    logical_path = str(record.get("logical_path") or "")
    alias_key = _research_logical_alias_key(logical_path)
    previous = logical_bindings.get(alias_key)
    if previous is not None and previous != (logical_path, object_hash):
        raise ResearchGitError(
            "research object logical paths collide on a portable filesystem: "
            f"{previous[0]} and {logical_path}"
        )
    logical_bindings[alias_key] = (logical_path, object_hash)
    records.setdefault(object_hash, []).append(record)


def _pointer_records(
    repo: Path,
    *,
    strict: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    logical_bindings: dict[tuple[str, ...], tuple[str, str]] = {}
    pointer_root = repo / "research-objects"
    if not pointer_root.exists():
        return records
    for path in sorted(pointer_root.glob("*.json")):
        relative_path = path.relative_to(repo).as_posix()
        if _has_symlink_component(repo, relative_path):
            if strict:
                raise ResearchGitError(
                    "research object pointer path contains a symbolic link: "
                    f"{relative_path}"
                )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = _validate_pointer_payload(
                payload,
                pointer_path=relative_path,
            )
        except (OSError, json.JSONDecodeError, ResearchGitError) as exc:
            if strict:
                raise ResearchGitError(
                    f"cannot validate research object pointer: "
                    f"{relative_path}: {exc}"
                ) from exc
            continue
        object_hash = str(payload.get("object_hash") or "")
        if object_hash.startswith("sha256:"):
            try:
                _record_pointer(
                    records,
                    logical_bindings,
                    object_hash=object_hash,
                    record={
                        **payload,
                        "pointer_path": relative_path,
                    },
                )
            except ResearchGitError:
                if strict:
                    raise
    return records


def _pointer_records_at_commit(
    repo: Path,
    commit: str,
    *,
    strict: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    tree = _run_git(repo, ["ls-tree", "-r", "--name-only", commit]).stdout.splitlines()
    records: dict[str, list[dict[str, Any]]] = {}
    logical_bindings: dict[tuple[str, ...], tuple[str, str]] = {}
    for pointer_path in sorted(
        path
        for path in tree
        if path.startswith("research-objects/") and path.endswith(".json")
    ):
        raw = _run_git(repo, ["show", f"{commit}:{pointer_path}"]).stdout
        try:
            payload = json.loads(raw)
            payload = _validate_pointer_payload(payload, pointer_path=pointer_path)
        except (json.JSONDecodeError, ResearchGitError) as exc:
            if strict:
                raise ResearchGitError(
                    f"cannot validate research object pointer at "
                    f"{commit}:{pointer_path}"
                ) from exc
            continue
        object_hash = str(payload.get("object_hash") or "")
        if object_hash.startswith("sha256:"):
            try:
                _record_pointer(
                    records,
                    logical_bindings,
                    object_hash=object_hash,
                    record={**payload, "pointer_path": pointer_path},
                )
            except ResearchGitError:
                if strict:
                    raise
    return records


def _configured_store_root(repo: Path) -> Path:
    config = load_repository_config(repo)
    return _resolve_configured_path(
        repo,
        (config.get("storage") or {}).get("root") or ".ara-store",
        label="CAS root",
    )


def _pointer_store_path(
    repo: Path,
    pointer: dict[str, Any],
    *,
    store_root: Path | None = None,
) -> Path:
    relative = _normalise_relative(str(pointer.get("store_relpath") or ""))
    target = (repo / relative).resolve()
    allowed_root = (store_root or _configured_store_root(repo)).resolve()
    try:
        target.relative_to(allowed_root)
    except ValueError as exc:
        raise ResearchGitError(
            f"research object store path escapes configured CAS: {relative}"
        ) from exc
    return target


def _verify_pointer_object(
    repo: Path,
    object_hash: str,
    pointer: dict[str, Any],
    *,
    store_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    try:
        store_path = _pointer_store_path(repo, pointer, store_root=store_root)
    except ResearchGitError as exc:
        return None, str(exc)
    if not store_path.is_file():
        return None, f"missing CAS object: {object_hash}"
    expected_size = int(pointer.get("size") or 0)
    actual_size = store_path.stat().st_size
    if actual_size != expected_size:
        return store_path, (
            f"CAS object size mismatch for {object_hash}: "
            f"expected {expected_size}, got {actual_size}"
        )
    actual_hash = _hash_file(store_path)
    if actual_hash != object_hash:
        return store_path, (
            f"CAS object hash mismatch for {object_hash}: got {actual_hash}"
        )
    return store_path, None


def _checkpoint_hash_valid(payload: dict[str, Any]) -> bool:
    expected = str(payload.get("content_hash") or "")
    scrubbed = {
        key: value
        for key, value in payload.items()
        if key not in {"checkpoint_id", "content_hash"}
    }
    return bool(expected and content_hash(scrubbed) == expected)


def _validate_checkpoint_payload(
    payload: Any,
    *,
    checkpoint_path: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ResearchGitError(f"invalid research checkpoint: {checkpoint_path}")
    try:
        validate_json(payload, load_schema("research_checkpoint"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"invalid research checkpoint {checkpoint_path}: {exc.message}"
        ) from exc
    if not _checkpoint_hash_valid(payload):
        raise ResearchGitError(
            f"checkpoint hash verification failed: {checkpoint_path}"
        )
    is_revert = str(payload.get("stage") or "") == "revert"
    has_reverts_commit = bool(str(payload.get("reverts_commit") or ""))
    has_reverts_checkpoint_hash = bool(
        str(payload.get("reverts_checkpoint_hash") or "")
    )
    if is_revert and not (has_reverts_commit and has_reverts_checkpoint_hash):
        raise ResearchGitError(
            f"revert checkpoint lacks a typed rollback edge: {checkpoint_path}"
        )
    if not is_revert and (has_reverts_commit or has_reverts_checkpoint_hash):
        raise ResearchGitError(
            f"non-revert checkpoint contains rollback metadata: {checkpoint_path}"
        )
    return payload


def _checkpoint_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['subject']}",
        "",
        f"- Stage: `{payload['stage']}`",
        f"- State: `{payload['status']}`",
        f"- Actor: `{payload['actor']}`",
        f"- Created: `{payload['created_at']}`",
        f"- Checkpoint: `{payload['checkpoint_id']}`",
        f"- Content hash: `{payload['content_hash']}`",
        "",
        payload.get("summary") or "No summary supplied.",
        "",
        "## Scientific changes",
        "",
    ]
    if payload.get("stage") == "revert":
        lines[8:8] = [
            f"- Reverts commit: `{payload.get('reverts_commit')}`",
            f"- Reverts checkpoint: `{payload.get('reverts_checkpoint_hash')}`",
        ]
    lines.extend(f"- `{path}`" for path in payload.get("changed_paths") or [])
    if not payload.get("changed_paths"):
        lines.append("- Semantic checkpoint only; no additional tracked file changed.")
    if payload.get("ara_manifests"):
        lines.extend(["", "## ARA manifests", ""])
        lines.extend(
            (
                f"- `{item['path']}` — manifest `{item['manifest_hash']}`"
                + (
                    f", graph `{item['exploration_graph_hash']}`"
                    if item.get("exploration_graph_hash")
                    else ""
                )
            )
            for item in payload["ara_manifests"]
        )
    command = (payload.get("reproduce") or {}).get("command")
    if command:
        lines.extend(["", "## Reproduce", "", "```bash", command, "```"])
    return "\n".join(lines).rstrip() + "\n"


def _commit_subject(stage: str, subject: str) -> str:
    category = {
        "init": "research",
        "ideation": "research",
        "preregister": "research",
        "experiment": "experiment",
        "evidence": "evidence",
        "review": "review",
        "paper": "paper",
        "release": "paper",
        "failed": "experiment",
    }.get(stage, "research")
    clean_subject = " ".join(subject.strip().split())
    if not clean_subject:
        clean_subject = f"record {stage} checkpoint"
    return f"{category}({stage}): {clean_subject}"[:200]


def _tree_entry(
    repo: Path,
    revision: str,
    path: str,
    *,
    bounded: bool = False,
) -> tuple[str, str] | None:
    runner = _run_git_bounded if bounded else _run_git
    result = runner(
        repo,
        ["ls-tree", revision, "--", path],
        **({"max_output_bytes": 4096, "check": False} if bounded else {"check": False}),
    )
    if result.returncode or not result.stdout.strip():
        return None
    metadata, separator, returned_path = result.stdout.rstrip("\n").partition("\t")
    fields = metadata.split()
    if (
        not separator
        or returned_path != path
        or len(fields) != 3
        or fields[1] not in {"blob", "commit"}
        or re.fullmatch(r"[0-9a-f]{40,64}", fields[2]) is None
    ):
        raise ResearchGitError(f"cannot parse Git tree entry for {path}")
    return fields[0], fields[2]


def _index_entry(repo: Path, path: str) -> tuple[str, str] | None:
    result = _run_git(repo, ["ls-files", "--stage", "--", path])
    if not result.stdout.strip():
        return None
    rows = result.stdout.rstrip("\n").splitlines()
    if len(rows) != 1:
        raise ResearchGitError(f"checkpoint index contains conflicted path: {path}")
    metadata, separator, returned_path = rows[0].partition("\t")
    fields = metadata.split()
    if (
        not separator
        or returned_path != path
        or len(fields) != 3
        or fields[2] != "0"
        or re.fullmatch(r"[0-9a-f]{40,64}", fields[1]) is None
    ):
        raise ResearchGitError(f"cannot parse checkpoint index entry for {path}")
    return fields[0], fields[1]


def _material_changes_between(
    repo: Path,
    before: str | None,
    after: str,
    *,
    bounded: bool = False,
) -> dict[str, str]:
    if before is None:
        args = [
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--no-renames",
            "--name-status",
            "-r",
            "-z",
            after,
        ]
    else:
        args = [
            "diff",
            "--no-renames",
            "--name-status",
            "-z",
            before,
            after,
        ]
    runner = _run_git_bounded if bounded else _run_git
    result = runner(
        repo,
        args,
        **({"max_output_bytes": _HARD_MAX_BOUNDED_LOG_OUTPUT} if bounded else {}),
    )
    tokens = result.stdout.split("\0")
    if tokens and tokens[-1] == "":
        tokens.pop()
    if len(tokens) % 2:
        raise ResearchGitError("cannot parse Git material transition")
    changes: dict[str, str] = {}
    for index in range(0, len(tokens), 2):
        status, raw_path = tokens[index : index + 2]
        if status not in {"A", "M", "D"}:
            raise ResearchGitError("research revert contains a rename or complex diff")
        path = _normalise_relative(raw_path)
        if path.startswith("checkpoints/"):
            continue
        if path in changes:
            raise ResearchGitError("research revert contains duplicate path changes")
        changes[path] = status
    return changes


def _commit_first_parent(
    repo: Path,
    commit: str,
    *,
    bounded: bool = False,
) -> str | None:
    runner = _run_git_bounded if bounded else _run_git
    result = runner(
        repo,
        ["rev-list", "--parents", "-n", "1", commit],
        **({"max_output_bytes": 4096} if bounded else {}),
    )
    fields = result.stdout.split()
    if not fields or fields[0] != commit:
        raise ResearchGitError("cannot resolve research revert target ancestry")
    return fields[1] if len(fields) > 1 else None


def _validate_committed_inverse_revert(
    repo: Path,
    *,
    revert_commit: str,
    revert_parent: str,
    target_commit: str,
    bounded: bool = False,
) -> None:
    target_parent = _commit_first_parent(repo, target_commit, bounded=bounded)
    target_changes = _material_changes_between(
        repo, target_parent, target_commit, bounded=bounded
    )
    revert_changes = _material_changes_between(
        repo, revert_parent, revert_commit, bounded=bounded
    )
    inverse_status = {"A": "D", "D": "A", "M": "M"}
    expected_revert_changes = {
        path: inverse_status[status] for path, status in target_changes.items()
    }
    if expected_revert_changes != revert_changes:
        raise ResearchGitError(
            "research revert material paths/statuses are not the exact target inverse"
        )
    for path in target_changes:
        target_before = (
            _tree_entry(repo, target_parent, path, bounded=bounded)
            if target_parent
            else None
        )
        target_after = _tree_entry(repo, target_commit, path, bounded=bounded)
        revert_before = _tree_entry(repo, revert_parent, path, bounded=bounded)
        revert_after = _tree_entry(repo, revert_commit, path, bounded=bounded)
        if revert_before != target_after or revert_after != target_before:
            raise ResearchGitError(
                f"research revert blob transition is not inverse for {path}"
            )


def _validate_index_inverse_revert(
    repo: Path,
    *,
    target_commit: str,
    material: Sequence[str],
) -> None:
    head = _head(repo)
    if head is None:
        raise ResearchGitError("research revert requires an existing HEAD")
    target_parent = _commit_first_parent(repo, target_commit)
    target_changes = _material_changes_between(repo, target_parent, target_commit)
    if set(target_changes) != set(material):
        raise ResearchGitError(
            "research revert index contains paths outside the exact target inverse"
        )
    for path in target_changes:
        target_before = (
            _tree_entry(repo, target_parent, path) if target_parent else None
        )
        target_after = _tree_entry(repo, target_commit, path)
        current_before = _tree_entry(repo, head, path)
        current_after = _index_entry(repo, path)
        if current_before != target_after or current_after != target_before:
            raise ResearchGitError(
                f"research revert index is not the exact inverse for {path}"
            )


def _create_checkpoint_locked(
    root: Path,
    *,
    stage: str,
    subject: str,
    summary: str = "",
    status: str = "completed",
    actor: str | None = None,
    nodes: Sequence[str] = (),
    claims: Sequence[str] = (),
    ara_paths: Sequence[str] = (),
    object_refs: Sequence[str] = (),
    reproduce_command: str | None = None,
    include: Sequence[str] = (),
    only_paths: Sequence[str] | None = None,
    additional_parent_refs: Sequence[str] = (),
    allow_backend_stage: bool = False,
    reverts_commit: str | None = None,
    reverts_checkpoint_hash: str | None = None,
    commit: bool = True,
    allow_checkpoint_only: bool = False,
) -> CheckpointResult:
    config = load_repository_config(root)
    checkpoint_actor = actor or str(config.get("actor") or "xscientist")
    _refuse_private_research_metadata(
        {
            "subject": subject,
            "summary": summary,
            "actor": checkpoint_actor,
            "nodes": [str(item) for item in nodes],
            "claims": [str(item) for item in claims],
            "reproduce_command": str(reproduce_command or ""),
        },
        operation="checkpoint",
    )
    if not stage or not stage.replace("-", "_").replace("_", "a").isalnum():
        raise ResearchGitError(
            "stage must contain only letters, numbers, hyphens, or underscores"
        )
    if reproduce_command and any(char in reproduce_command for char in "\r\n"):
        raise ResearchGitError("reproduction command must be a single line")
    # Validate the scientific parent before inspecting or materializing a new
    # checkpoint. In particular, never let an ordinary checkpoint bridge a raw
    # Git commit above an existing Research VCS ancestor.
    checkpoint_parent_records = _checkpoint_parent_records(root, stage=stage)
    staged_before, changed = _changed_paths(root)
    if staged_before and not allow_backend_stage:
        raise ResearchGitError(
            "checkpoint refused because the Git index already contains staged changes: "
            + ", ".join(sorted(staged_before))
        )
    if allow_backend_stage:
        changed |= staged_before
    candidates = changed
    if only_paths is not None:
        normalized_only = {_normalise_relative(path) for path in only_paths}
        absent = sorted(normalized_only - changed)
        if absent:
            raise ResearchGitError(
                "staged research paths no longer contain a change: " + ", ".join(absent)
            )
        candidates = changed & normalized_only
    explicit_paths = [*include, *(only_paths or ())]
    selected, excluded = _select_paths(
        root,
        config,
        candidates,
        explicit=explicit_paths,
    )
    if allow_backend_stage:
        excluded_backend_paths = sorted(set(staged_before) - set(selected))
        if excluded_backend_paths:
            raise ResearchGitError(
                "checkpoint refused because the backend index contains paths outside "
                "the research safety policy: " + ", ".join(excluded_backend_paths)
            )
    material = [path for path in selected if not path.startswith("checkpoints/")]
    rollback_fields = (reverts_commit, reverts_checkpoint_hash)
    if stage == "revert":
        if not all(rollback_fields):
            raise ResearchGitError(
                "revert checkpoints require an exact target commit and checkpoint hash"
            )
        if re.fullmatch(r"[0-9a-f]{40,64}", str(reverts_commit)) is None:
            raise ResearchGitError("revert target commit is invalid")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", str(reverts_checkpoint_hash)) is None:
            raise ResearchGitError("revert target checkpoint hash is invalid")
    elif any(rollback_fields):
        raise ResearchGitError("rollback metadata is only valid for revert checkpoints")
    deleted_immutable_objects = sorted(
        path
        for path in material
        if path.startswith(".xscientist/objects/")
        and path.endswith(".json")
        and not (root / path).exists()
    )
    if deleted_immutable_objects and stage != "revert":
        raise ResearchGitError(
            "immutable research objects can only be removed by a typed research "
            "revert: " + ", ".join(deleted_immutable_objects)
        )
    immutable_object_paths = sorted(
        path
        for path in material
        if path.startswith(".xscientist/objects/") and path.endswith(".json")
    )
    if stage != "revert":
        head = _head(root)
        for path in immutable_object_paths:
            if head and _tree_entry(root, head, path) is not None:
                raise ResearchGitError(
                    "immutable research objects cannot be modified or deleted by "
                    f"an ordinary checkpoint: {path}"
                )
            target = root / path
            if not target.is_file() or target.is_symlink():
                raise ResearchGitError(f"new research object path is invalid: {path}")
            try:
                research_object = validate_research_object(
                    json.loads(target.read_text(encoding="utf-8"))
                )
            except (OSError, json.JSONDecodeError, ResearchObjectError) as exc:
                raise ResearchGitError(
                    f"new research object is invalid: {path}"
                ) from exc
            expected_path = _research_object_path(root, research_object)
            if expected_path.relative_to(root).as_posix() != path:
                raise ResearchGitError(
                    f"new research object identity disagrees with its path: {path}"
                )
    else:
        _target_path, target_checkpoint = _checkpoint_at_commit(
            root, str(reverts_commit)
        )
        if target_checkpoint.get("content_hash") != reverts_checkpoint_hash:
            raise ResearchGitError(
                "revert target checkpoint hash does not match the target commit"
            )
        head = _head(root)
        if (
            head is None
            or _run_git(
                root,
                ["merge-base", "--is-ancestor", str(reverts_commit), head],
                check=False,
            ).returncode
        ):
            raise ResearchGitError(
                "research revert target must be an ancestor of the current HEAD"
            )
        _validate_index_inverse_revert(
            root,
            target_commit=str(reverts_commit),
            material=material,
        )
    if not material and not allow_checkpoint_only:
        return CheckpointResult(
            created=False,
            committed=False,
            excluded_paths=tuple(excluded),
            reason="no material research change matched the checkpoint policy",
        )

    parent_records = _append_checkpoint_parent_records(
        root,
        checkpoint_parent_records,
        additional_parent_refs,
    )
    previous = parent_records[0][2] if parent_records else None
    sequence = (
        max(
            (int(record[2].get("sequence") or 0) for record in parent_records),
            default=0,
        )
        + 1
    )
    parent_checkpoint_hashes = [
        str(record[2]["content_hash"])
        for record in parent_records
        if record[2].get("content_hash")
    ]
    explicit_ara: list[str] = []
    for raw in ara_paths:
        relative = _normalise_relative(raw)
        target = root / relative
        if target.is_dir():
            target = target / "manifest.json"
            relative = target.relative_to(root).as_posix()
        if not target.is_file() or target.name != "manifest.json":
            raise ResearchGitError(
                f"ARA path is not a manifest or ARA directory: {raw}"
            )
        explicit_ara.append(relative)
    touched_ara = _touched_ara_manifest_paths(root, selected)
    touched_ara.update(explicit_ara)
    previous_manifests = {
        str(item.get("path")): dict(item)
        for item in (previous or {}).get("ara_manifests") or []
        if isinstance(item, dict) and item.get("path")
    }
    for manifest_path in touched_ara:
        previous_manifests.pop(manifest_path, None)
    previous_manifests.update(
        {item["path"]: item for item in _manifest_refs(root, touched_ara)}
    )
    manifests = [previous_manifests[path] for path in sorted(previous_manifests)]

    pointers = _pointer_records(root, strict=True)
    if only_paths is not None:
        selected_pointer_paths = {
            path for path in selected if path.startswith("research-objects/")
        }
        head = _head(root)
        committed_pointers = (
            _pointer_records_at_commit(root, head, strict=True) if head else {}
        )
        merged_pointers: dict[str, list[dict[str, Any]]] = {}
        logical_bindings: dict[tuple[str, ...], tuple[str, str]] = {}
        for object_hash, items in committed_pointers.items():
            for item in items:
                if item.get("pointer_path") not in selected_pointer_paths:
                    _record_pointer(
                        merged_pointers,
                        logical_bindings,
                        object_hash=object_hash,
                        record=item,
                    )
        for object_hash, items in _pointer_records(root, strict=True).items():
            for item in items:
                if item.get("pointer_path") in selected_pointer_paths:
                    _record_pointer(
                        merged_pointers,
                        logical_bindings,
                        object_hash=object_hash,
                        record=item,
                    )
        pointers = merged_pointers
    resolved_object_refs = sorted(
        {
            *(str(item) for item in object_refs if str(item).startswith("sha256:")),
            *pointers.keys(),
        }
    )
    environment = _environment_receipt(root)
    created_at = _now_iso()
    base_payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA,
        "protocol_kind": "research_checkpoint",
        "sequence": sequence,
        "created_at": created_at,
        "stage": stage,
        "status": status,
        "subject": " ".join(subject.strip().split()),
        "summary": summary.strip(),
        "actor": checkpoint_actor,
        "branch": _branch(root),
        "parent_commit": _head(root),
        "previous_checkpoint_hash": (previous or {}).get("content_hash"),
        "parent_checkpoint_hashes": parent_checkpoint_hashes,
        "changed_paths": sorted(material),
        "nodes": sorted({str(item) for item in nodes if str(item)}),
        "claims": sorted({str(item) for item in claims if str(item)}),
        "ara_manifests": manifests,
        "object_refs": resolved_object_refs,
        "reproduce": {
            "command": reproduce_command or "",
            "working_directory": ".",
            "environment": environment,
        },
    }
    if stage == "revert":
        base_payload["reverts_commit"] = str(reverts_commit)
        base_payload["reverts_checkpoint_hash"] = str(reverts_checkpoint_hash)
    checkpoint_hash = content_hash(base_payload)
    checkpoint_id = f"rcp-{checkpoint_hash.split(':', 1)[1][:16]}"
    payload = {
        **base_payload,
        "checkpoint_id": checkpoint_id,
        "content_hash": checkpoint_hash,
    }
    try:
        validate_json(payload, load_schema("research_checkpoint"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"generated checkpoint is invalid: {exc.message}"
        ) from exc
    slug = "".join(char if char.isalnum() or char in "-_" else "-" for char in stage)
    basename = f"{sequence:04d}-{slug}-{checkpoint_id[4:12]}"
    checkpoint_path = root / "checkpoints" / f"{basename}.json"
    markdown_path = root / "checkpoints" / f"{basename}.md"
    created_paths = [checkpoint_path, markdown_path]
    try:
        _atomic_write_json(checkpoint_path, payload)
        _atomic_write_text(markdown_path, _checkpoint_markdown(payload))
    except BaseException:
        for created_path in created_paths:
            created_path.unlink(missing_ok=True)
        _fsync_directory(checkpoint_path.parent)
        raise
    stage_paths = sorted(
        {
            *selected,
            checkpoint_path.relative_to(root).as_posix(),
            markdown_path.relative_to(root).as_posix(),
        }
    )

    privacy_findings = scan_paths(root, stage_paths)
    if privacy_findings:
        for created_path in created_paths:
            created_path.unlink(missing_ok=True)
        _fsync_directory(checkpoint_path.parent)
        raise ResearchGitError(
            "privacy gate refused the checkpoint; matched values were not displayed:\n"
            + format_privacy_findings(privacy_findings)
        )

    if not commit:
        return CheckpointResult(
            created=True,
            committed=False,
            checkpoint_path=checkpoint_path,
            checkpoint_id=checkpoint_id,
            content_hash=checkpoint_hash,
            staged_paths=tuple(stage_paths),
            excluded_paths=tuple(excluded),
            reason="checkpoint written without creating a Git commit",
        )

    try:
        # ``-A`` is required when a semantic transition removes files, as a
        # rollback of a checkpoint that originally introduced them will do.
        # A deletion created by ``git revert --no-commit`` is already staged
        # and no longer matches a pathspec, so only add paths not already in
        # the backend index.
        paths_to_add = sorted(set(stage_paths) - set(staged_before))
        if paths_to_add:
            _run_git(root, ["add", "-A", "--", *paths_to_add])
        staged_for_commit = _git_paths(
            root,
            ["diff", "--no-renames", "--cached", "--name-only", "-z"],
        )
        expected_stage = set(stage_paths)
        if staged_for_commit != expected_stage:
            unexpected = sorted(staged_for_commit - expected_stage)
            absent = sorted(expected_stage - staged_for_commit)
            details: list[str] = []
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            if absent:
                details.append("absent: " + ", ".join(absent))
            raise ResearchGitError(
                "checkpoint index does not exactly match the declared paths"
                + (" (" + "; ".join(details) + ")" if details else "")
            )
        # Inspect the complete index-bound path set. Refuse a split
        # index/worktree state so the scanner cannot inspect benign working
        # bytes while Git commits a different staged blob.
        index_drift = _git_paths(
            root,
            [
                "diff",
                "--no-renames",
                "--name-only",
                "-z",
                "--",
                *sorted(staged_for_commit),
            ],
        )
        if index_drift:
            raise ResearchGitError(
                "checkpoint refused because staged content differs from the "
                "working copy: " + ", ".join(sorted(index_drift))
            )
        privacy_findings = scan_paths(root, staged_for_commit)
        if privacy_findings:
            raise ResearchGitError(
                "privacy gate refused the checkpoint; matched values were not displayed:\n"
                + format_privacy_findings(privacy_findings)
            )
        index_drift = _git_paths(
            root,
            [
                "diff",
                "--no-renames",
                "--name-only",
                "-z",
                "--",
                *sorted(staged_for_commit),
            ],
        )
        if index_drift:
            raise ResearchGitError(
                "checkpoint refused because staged content changed during privacy "
                "validation: " + ", ".join(sorted(index_drift))
            )
        if _run_git(root, ["diff", "--cached", "--quiet"], check=False).returncode == 0:
            raise ResearchGitError("checkpoint produced no staged Git change")
        trailers = [
            f"Research-Checkpoint: {checkpoint_id}",
            f"Research-Stage: {stage}",
            f"Research-State: {status}",
            f"Research-Event: {checkpoint_hash}",
        ]
        if stage == "revert":
            trailers.extend(
                [
                    f"Research-Reverts: {reverts_commit}",
                    f"Research-Reverts-Checkpoint: {reverts_checkpoint_hash}",
                ]
            )
        trailers.extend(f"Research-Parent: {item}" for item in parent_checkpoint_hashes)
        trailers.extend(f"ARA-Manifest: {item['manifest_hash']}" for item in manifests)
        if reproduce_command:
            trailers.append(f"Reproduce: {reproduce_command}")
        _run_git(
            root,
            [
                "commit",
                "-m",
                _commit_subject(stage, subject),
                "-m",
                "\n".join(trailers),
            ],
        )
    except BaseException:
        if _head(root) is None:
            _run_git(
                root,
                ["rm", "--cached", "-q", "--ignore-unmatch", "--", *stage_paths],
                check=False,
            )
        else:
            _run_git(root, ["reset", "-q", "HEAD", "--", *stage_paths], check=False)
        for created_path in created_paths:
            created_path.unlink(missing_ok=True)
        _fsync_directory(checkpoint_path.parent)
        raise
    commit_hash = _head(root)
    return CheckpointResult(
        created=True,
        committed=True,
        checkpoint_path=checkpoint_path,
        commit=commit_hash,
        checkpoint_id=checkpoint_id,
        content_hash=checkpoint_hash,
        staged_paths=tuple(stage_paths),
        excluded_paths=tuple(excluded),
    )


def create_checkpoint(
    repo: str | Path,
    *,
    stage: str,
    subject: str,
    summary: str = "",
    status: str = "completed",
    actor: str | None = None,
    nodes: Sequence[str] = (),
    claims: Sequence[str] = (),
    ara_paths: Sequence[str] = (),
    object_refs: Sequence[str] = (),
    reproduce_command: str | None = None,
    include: Sequence[str] = (),
    only_paths: Sequence[str] | None = None,
    commit: bool = True,
    allow_checkpoint_only: bool = False,
) -> CheckpointResult:
    root = _repository_root(repo)
    with _repository_lock(root):
        return _create_checkpoint_locked(
            root,
            stage=stage,
            subject=subject,
            summary=summary,
            status=status,
            actor=actor,
            nodes=nodes,
            claims=claims,
            ara_paths=ara_paths,
            object_refs=object_refs,
            reproduce_command=reproduce_command,
            include=include,
            only_paths=only_paths,
            commit=commit,
            allow_checkpoint_only=allow_checkpoint_only,
        )


def commit_research_stage(
    repo: str | Path,
    *,
    stage: str,
    subject: str,
    summary: str = "",
    status: str = "completed",
    actor: str | None = None,
    nodes: Sequence[str] = (),
    claims: Sequence[str] = (),
    ara_paths: Sequence[str] = (),
    object_refs: Sequence[str] = (),
    reproduce_command: str | None = None,
) -> CheckpointResult:
    """Commit exactly the native-stage selection as one scientific transition."""

    root = _repository_root(repo)
    with _repository_lock(root):
        staged = _load_research_stage(root)
        entries = {
            str(item["path"]): str(item["fingerprint"])
            for item in staged.get("entries") or []
        }
        if not entries:
            raise ResearchGitError("native research stage is empty")
        if staged.get("head") != _head(root):
            raise ResearchGitError(
                "native research stage belongs to another repository state; unstage it first"
            )
        stale = [
            path
            for path, expected in entries.items()
            if _worktree_fingerprint(root, path) != expected
        ]
        if stale:
            raise ResearchGitError(
                "staged research content changed after selection: "
                + ", ".join(sorted(stale))
                + "; stage it again"
            )
        result = _create_checkpoint_locked(
            root,
            stage=stage,
            subject=subject,
            summary=summary,
            status=status,
            actor=actor,
            nodes=nodes,
            claims=claims,
            ara_paths=ara_paths,
            object_refs=object_refs,
            reproduce_command=reproduce_command,
            only_paths=tuple(entries),
        )
        if result.committed:
            _write_research_stage(root, {})
        return result


def auto_checkpoint(
    repo: str | Path,
    *,
    stage: str,
    subject: str,
    summary: str = "",
    status: str = "completed",
    nodes: Sequence[str] = (),
    claims: Sequence[str] = (),
    ara_paths: Sequence[str] = (),
    reproduce_command: str | None = None,
) -> CheckpointResult:
    root = _repository_root(repo)
    config = load_repository_config(root)
    git_config = config["git"]
    if not bool(git_config.get("auto_commit", True)):
        return CheckpointResult(False, False, reason="auto_commit is disabled")
    policy = str(git_config.get("checkpoint_policy") or "milestone")
    if policy == "manual":
        return CheckpointResult(False, False, reason="manual checkpoint policy")
    if policy == "milestone" and stage not in MILESTONE_STAGES:
        return CheckpointResult(
            False, False, reason=f"{stage} is not a milestone stage"
        )
    return create_checkpoint(
        root,
        stage=stage,
        subject=subject,
        summary=summary,
        status=status,
        nodes=nodes,
        claims=claims,
        ara_paths=ara_paths,
        reproduce_command=reproduce_command,
    )


def _validate_optional_positive_cap(
    value: int | None, *, label: str, maximum: int | None = None
) -> None:
    """Validate an optional resource cap without accepting bool/int coercions."""

    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{label} must be a positive integer")


def _bounded_store_summary(
    store: Path,
    *,
    max_entries: int,
    max_files: int,
    max_bytes: int,
) -> dict[str, Any]:
    """Summarise a CAS tree without recursively materialising unbounded paths.

    The ordinary repository APIs historically returned an exact count.  Audit
    callers use this explicitly bounded variant so a large or hostile local
    CAS cannot turn a benchmark report into an unbounded filesystem walk.
    """

    result: dict[str, Any] = {
        "path": str(store),
        "objects": 0,
        "bytes": 0,
        "truncated": False,
        "scan_entries": 0,
        "read_error_count": 0,
        "scan_complete": True,
        "scan_scope": "all_store_files",
    }
    files_seen = 0
    if not store.exists():
        return result
    if store.is_symlink() or not store.is_dir():
        result.update(
            {
                "truncated": True,
                "scan_complete": False,
                "scan_scope": "bounded_store_prefix",
                "read_error_count": 1,
            }
        )
        return result

    pending = [store]
    while pending:
        current = pending.pop()
        try:
            iterator = os.scandir(current)
        except OSError:
            result["read_error_count"] += 1
            result["scan_complete"] = False
            result["scan_scope"] = "bounded_store_prefix"
            continue
        try:
            while True:
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                except OSError:
                    result["read_error_count"] += 1
                    result["scan_complete"] = False
                    result["scan_scope"] = "bounded_store_prefix"
                    break
                result["scan_entries"] += 1
                if result["scan_entries"] > max_entries:
                    result["truncated"] = True
                    result["scan_complete"] = False
                    result["scan_scope"] = "bounded_store_prefix"
                    break
                try:
                    if entry.is_symlink():
                        # A symlink is deliberately not followed; its presence
                        # still means the complete store cannot be claimed.
                        result["truncated"] = True
                        result["scan_complete"] = False
                        result["scan_scope"] = "bounded_store_prefix"
                        result["read_error_count"] += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    size = int(entry.stat(follow_symlinks=False).st_size)
                except (OSError, ValueError):
                    result["read_error_count"] += 1
                    result["scan_complete"] = False
                    result["scan_scope"] = "bounded_store_prefix"
                    continue
                if result["bytes"] + size > max_bytes:
                    result["truncated"] = True
                    result["scan_complete"] = False
                    result["scan_scope"] = "bounded_store_prefix"
                    break
                files_seen += 1
                if files_seen > max_files:
                    result["truncated"] = True
                    result["scan_complete"] = False
                    result["scan_scope"] = "bounded_store_prefix"
                    break
                result["objects"] += 1
                result["bytes"] += size
            if result["truncated"]:
                break
        finally:
            iterator.close()
    return result


def repository_status(
    repo: str | Path,
    *,
    max_store_entries: int | None = None,
    max_store_files: int | None = None,
    max_store_bytes: int | None = None,
    skip_worktree_scan: bool = False,
    skip_checkpoint_scan: bool = False,
) -> dict[str, Any]:
    if not isinstance(skip_worktree_scan, bool):
        raise ValueError("skip_worktree_scan must be boolean")
    if not isinstance(skip_checkpoint_scan, bool):
        raise ValueError("skip_checkpoint_scan must be boolean")
    root = _repository_root(repo)
    config = load_repository_config(root)
    if isinstance(skip_worktree_scan, bool) and skip_worktree_scan:
        # Audit reports only need the repository identity and topology.  Git's
        # untracked/staged path listing is intentionally not streamed here;
        # callers that need the full worktree view can use the default status
        # API explicitly.
        worktree = {
            "clean": None,
            "backend_staged": [],
            "tracked": [],
            "eligible": [],
            "excluded": [],
            "research_stage_head": None,
            "research_stage": [],
        }
    else:
        worktree = _worktree_change_summary(root, config)
    # A shareable process audit must not walk an unbounded first-parent history
    # merely to recover the latest checkpoint.  Keep the historical default
    # for callers that explicitly request the full repository status, while
    # allowing bounded/read-only callers to opt out.
    previous = None if skip_checkpoint_scan else _previous_checkpoint(root)
    store = _resolve_configured_path(
        root,
        (config.get("storage") or {}).get("root") or ".ara-store",
        label="CAS root",
    )
    store_relative = (config.get("storage") or {}).get("root") or ".ara-store"
    _validate_optional_positive_cap(
        max_store_entries,
        label="max_store_entries",
        maximum=_HARD_MAX_STORE_ENTRIES,
    )
    _validate_optional_positive_cap(
        max_store_files,
        label="max_store_files",
        maximum=_HARD_MAX_STORE_FILES,
    )
    _validate_optional_positive_cap(
        max_store_bytes,
        label="max_store_bytes",
        maximum=_HARD_MAX_STORE_BYTES,
    )
    if (
        max_store_entries is None
        and max_store_files is None
        and max_store_bytes is None
    ):
        object_files = (
            [path for path in store.rglob("*") if path.is_file()]
            if store.exists()
            else []
        )
        object_store = {
            "path": str(store),
            "objects": len(object_files),
            "bytes": sum(path.stat().st_size for path in object_files),
        }
    else:
        if _has_symlink_component(root, store_relative):
            object_store = {
                "path": str(store),
                "objects": 0,
                "bytes": 0,
                "truncated": True,
                "scan_entries": 0,
                "read_error_count": 1,
                "scan_complete": False,
                "scan_scope": "symlink_boundary",
            }
        else:
            object_store = _bounded_store_summary(
                store,
                max_entries=max_store_entries or 8192,
                max_files=max_store_files or 4096,
                max_bytes=max_store_bytes or 64 * 1024 * 1024,
            )
    return {
        "repository": str(root),
        "name": config.get("name"),
        "branch": _branch(root),
        "head": _head(root),
        "checkpoint_policy": config["git"].get("checkpoint_policy"),
        "auto_commit": bool(config["git"].get("auto_commit", True)),
        "auto_push": False,
        "worktree_clean": worktree["clean"],
        "staged_paths": worktree["backend_staged"],
        "tracked_changes": worktree["tracked"],
        "research_stage": {
            "head": worktree["research_stage_head"],
            "paths": worktree["research_stage"],
        },
        "eligible_changes": worktree["eligible"],
        "excluded_changes": worktree["excluded"],
        "last_checkpoint": previous,
        "object_store": object_store,
    }


def _validate_research_ref_name(name: str, *, kind: str) -> str:
    normalized = str(name or "").strip()
    if not normalized or normalized.startswith("-"):
        raise ResearchGitError(f"invalid research {kind} name")
    candidate = normalized if kind == "branch" else f"refs/tags/{normalized}"
    arguments = ["check-ref-format"]
    if kind == "branch":
        arguments.append("--branch")
    arguments.append(candidate)
    completed = _run_git(Path.cwd(), arguments, check=False)
    if completed.returncode:
        raise ResearchGitError(f"invalid research {kind} name: {normalized}")
    return normalized


def list_research_branches(
    repo: str | Path, *, max_branches: int | None = None
) -> list[dict[str, Any]]:
    """List research lines with their latest scientific checkpoint.

    ``max_branches`` is an optional metadata-scan cap for shareable audits;
    the historical uncapped listing remains the default for repository APIs.
    """

    root = _repository_root(repo)
    if max_branches is not None and (
        isinstance(max_branches, bool)
        or not isinstance(max_branches, int)
        or max_branches <= 0
        or max_branches > 64
    ):
        raise ValueError("max_branches must be a positive integer")
    current = _branch(root)
    format_spec = (
        "%(refname:short)%00%(objectname)%00%(subject)"
        if max_branches is None
        else "%(refname:short)%00%(objectname)%00"
    )

    def read_refs(ref_glob: str, *, count: int | None = None) -> str:
        args = ["for-each-ref"]
        if count is not None:
            args.append(f"--count={count}")
        args.extend(["--sort=refname", f"--format={format_spec}", ref_glob])
        return _run_git(root, args).stdout

    raw = read_refs("refs/heads", count=max_branches)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = line.split("\0", 2)
        if len(fields) != 3:
            continue
        name, commit, subject = fields
        # The capped audit path deliberately does not walk each branch's
        # complete first-parent history or scan every checkpoint tree.  The
        # process report obtains stage/status from its separately bounded log;
        # a branch listing itself remains a cheap topology probe.
        checkpoint = (
            _latest_checkpoint_record(root, name) if max_branches is None else None
        )
        checkpoint_payload = checkpoint[2] if checkpoint is not None else {}
        rows.append(
            {
                "name": name,
                "current": name == current,
                "commit": commit,
                "subject": subject,
                "checkpoint_id": checkpoint_payload.get("checkpoint_id"),
                "stage": checkpoint_payload.get("stage"),
                "status": checkpoint_payload.get("status"),
            }
        )
    # A cap can hide the current line when refname sorting puts it after the
    # prefix.  Read that single ref explicitly so process views always retain
    # the current checkout, while the caller can still mark the source list as
    # truncated.
    if (
        max_branches is not None
        and current
        and not any(str(row.get("name") or "") == current for row in rows)
    ):
        current_raw = read_refs(f"refs/heads/{current}", count=1)
        for line in current_raw.splitlines():
            fields = line.split("\0", 2)
            if len(fields) != 3:
                continue
            name, commit, subject = fields
            checkpoint = None
            checkpoint_payload = checkpoint[2] if checkpoint is not None else {}
            rows.append(
                {
                    "name": name,
                    "current": True,
                    "commit": commit,
                    "subject": subject,
                    "checkpoint_id": checkpoint_payload.get("checkpoint_id"),
                    "stage": checkpoint_payload.get("stage"),
                    "status": checkpoint_payload.get("status"),
                }
            )
            break
    return rows


def create_research_branch(
    repo: str | Path,
    name: str,
    *,
    from_ref: str = "HEAD",
    switch: bool = False,
) -> dict[str, Any]:
    """Fork a named research line without requiring backend commands."""

    root = _repository_root(repo)
    normalized = _validate_research_ref_name(name, kind="branch")
    with _repository_lock(root):
        _run_git(root, ["rev-parse", "--verify", f"{from_ref}^{{commit}}"])
        existing = _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/heads/{normalized}"],
            check=False,
        )
        if existing.returncode == 0:
            raise ResearchGitError(f"research branch already exists: {normalized}")
        if switch:
            _require_clean_research_switch(root)
        _run_git(root, ["branch", "--", normalized, from_ref])
        if switch:
            _run_git(root, ["switch", "--", normalized])
        return {
            "name": normalized,
            "commit": _run_git(root, ["rev-parse", normalized]).stdout.strip(),
            "current": switch,
            "from": _run_git(root, ["rev-parse", from_ref]).stdout.strip(),
        }


def _require_clean_research_switch(root: Path) -> None:
    worktree = _worktree_change_summary(root)
    if not worktree["clean"]:
        raise ResearchGitError(
            "research transition requires a clean working state: no staged, "
            "tracked, research-eligible, or selected changes; policy-excluded "
            "generated views are preserved"
        )


def switch_research_branch(repo: str | Path, name: str) -> dict[str, Any]:
    """Switch research lines only when no working evidence can be overwritten."""

    root = _repository_root(repo)
    normalized = _validate_research_ref_name(name, kind="branch")
    with _repository_lock(root):
        _require_clean_research_switch(root)
        exists = _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/heads/{normalized}"],
            check=False,
        )
        if exists.returncode:
            raise ResearchGitError(f"research branch not found: {normalized}")
        _run_git(root, ["switch", "--", normalized])
        return {
            "name": normalized,
            "commit": _head(root),
            "current": True,
        }


def delete_research_branch(
    repo: str | Path,
    name: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Delete a non-current research line, protecting unmerged work by default."""

    root = _repository_root(repo)
    normalized = _validate_research_ref_name(name, kind="branch")
    with _repository_lock(root):
        if normalized == _branch(root):
            raise ResearchGitError("cannot delete the current research branch")
        exists = _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/heads/{normalized}"],
            check=False,
        )
        if exists.returncode:
            raise ResearchGitError(f"research branch not found: {normalized}")
        commit = _run_git(root, ["rev-parse", normalized]).stdout.strip()
        deleted = _run_git(
            root,
            ["branch", "-D" if force else "-d", "--", normalized],
            check=False,
        )
        if deleted.returncode:
            raise ResearchGitError(
                f"research branch is not fully merged: {normalized}; use -D only after review"
            )
        return {"name": normalized, "commit": commit, "deleted": True, "force": force}


def rename_research_branch(
    repo: str | Path,
    name: str,
    new_name: str,
) -> dict[str, Any]:
    """Rename a research line without changing its scientific history."""

    root = _repository_root(repo)
    old = _validate_research_ref_name(name, kind="branch")
    new = _validate_research_ref_name(new_name, kind="branch")
    with _repository_lock(root):
        if _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/heads/{old}"],
            check=False,
        ).returncode:
            raise ResearchGitError(f"research branch not found: {old}")
        if not _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/heads/{new}"],
            check=False,
        ).returncode:
            raise ResearchGitError(f"research branch already exists: {new}")
        _run_git(root, ["branch", "-m", old, new])
        return {
            "old_name": old,
            "name": new,
            "commit": _run_git(root, ["rev-parse", new]).stdout.strip(),
            "current": _branch(root) == new,
        }


def restore_research_paths(
    repo: str | Path,
    source: str,
    paths: Sequence[str],
) -> dict[str, Any]:
    """Restore explicit research paths from a checkpoint into working state."""

    root = _repository_root(repo)
    if not paths:
        raise ResearchGitError("restore requires at least one explicit research path")
    with _repository_lock(root):
        if _load_research_stage(root).get("entries"):
            raise ResearchGitError("restore requires an empty research stage")
        resolved = _run_git(root, ["rev-parse", "--verify", f"{source}^{{commit}}"])
        normalized = sorted({_normalise_relative(path) for path in paths})
        config = load_repository_config(root)
        selected, excluded = _select_paths(
            root, config, normalized, explicit=normalized
        )
        if excluded or selected != normalized:
            raise ResearchGitError(
                "restore refused paths outside the research safety policy: "
                + ", ".join(excluded or sorted(set(normalized) - set(selected)))
            )
        restored = _run_git(
            root,
            [
                "restore",
                f"--source={resolved.stdout.strip()}",
                "--worktree",
                "--",
                *normalized,
            ],
            check=False,
        )
        if restored.returncode:
            raise ResearchGitError(
                "one or more research paths do not exist at the source ref"
            )
        return {
            "source": resolved.stdout.strip(),
            "paths": normalized,
            "changed_paths": sorted(_changed_paths(root)[1]),
        }


def revert_research_checkpoint(
    repo: str | Path,
    commit: str,
    *,
    subject: str | None = None,
) -> dict[str, Any]:
    """Revert one Git commit and immediately document it as a research checkpoint."""

    root = _repository_root(repo)
    with _repository_lock(root):
        _require_clean_research_switch(root)
        resolved = _run_git(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
        resolved_commit = resolved.stdout.strip()
        _target_path, target_checkpoint = _checkpoint_at_commit(root, resolved_commit)
        head = _head(root)
        if (
            head is None
            or _run_git(
                root,
                ["merge-base", "--is-ancestor", resolved_commit, head],
                check=False,
            ).returncode
        ):
            raise ResearchGitError(
                "research revert target must be an ancestor of the current HEAD"
            )
        checkpoint_paths = [
            path
            for path in _run_git(
                root,
                ["diff-tree", "--no-commit-id", "--name-only", "-r", resolved_commit],
            ).stdout.splitlines()
            if path.startswith("checkpoints/")
        ]
        reverted = _run_git(
            root,
            ["revert", "--no-commit", resolved_commit],
            check=False,
        )
        if reverted.returncode:
            _run_git(root, ["revert", "--abort"], check=False)
            raise ResearchGitError(
                "research revert has conflicts; resolve on a dedicated branch"
            )
        if checkpoint_paths:
            _run_git(
                root,
                [
                    "restore",
                    "--source=HEAD",
                    "--staged",
                    "--worktree",
                    "--",
                    *checkpoint_paths,
                ],
            )
        try:
            checkpoint = _create_checkpoint_locked(
                root,
                stage="revert",
                subject=subject or f"revert research checkpoint {resolved_commit[:12]}",
                summary=f"Semantic revert of Git commit {resolved_commit}.",
                status="completed",
                allow_backend_stage=True,
                allow_checkpoint_only=True,
                reverts_commit=resolved_commit,
                reverts_checkpoint_hash=str(target_checkpoint["content_hash"]),
            )
        except Exception:
            _run_git(root, ["revert", "--abort"], check=False)
            # ``git revert --no-commit`` does not always leave sequencer state
            # for ``--abort`` (notably for a single commit). The transition
            # started from a verified clean state, so restoring tracked files
            # to HEAD is lossless and preserves policy-excluded local views.
            _run_git(
                root,
                ["restore", "--source=HEAD", "--staged", "--worktree", "--", "."],
                check=False,
            )
            raise
        return {
            "reverted": resolved_commit,
            "revert_commit": checkpoint.commit,
            "checkpoint": checkpoint.to_dict(),
        }


def create_research_tag(
    repo: str | Path,
    name: str,
    *,
    commit: str = "HEAD",
    annotation: str = "",
) -> dict[str, Any]:
    """Create an immutable, annotated name for a scientific checkpoint."""

    root = _repository_root(repo)
    normalized = _validate_research_ref_name(name, kind="tag")
    with _repository_lock(root):
        resolved = _run_git(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
        exists = _run_git(
            root,
            ["show-ref", "--verify", "--quiet", f"refs/tags/{normalized}"],
            check=False,
        )
        if exists.returncode == 0:
            raise ResearchGitError(f"research tag already exists: {normalized}")
        _path, checkpoint = _checkpoint_at_commit(root, resolved.stdout.strip())
        message = annotation.strip() or (
            f"Research checkpoint {checkpoint['checkpoint_id']}\n\n"
            f"Research-Checkpoint: {checkpoint['checkpoint_id']}\n"
            f"Research-Event: {checkpoint['content_hash']}"
        )
        if redact_sensitive_text(message) != message:
            raise ResearchGitError(
                "privacy gate refused the research tag annotation; matched values were not displayed"
            )
        _run_git(
            root,
            ["tag", "-a", normalized, resolved.stdout.strip(), "-m", message],
        )
        return {
            "name": normalized,
            "commit": resolved.stdout.strip(),
            "checkpoint_id": checkpoint["checkpoint_id"],
            "content_hash": checkpoint["content_hash"],
        }


def list_research_tags(repo: str | Path) -> list[dict[str, Any]]:
    root = _repository_root(repo)
    raw = _run_git(
        root,
        [
            "for-each-ref",
            "--sort=refname",
            "--format=%(refname:short)%00%(*objectname)%00%(objectname)",
            "refs/tags",
        ],
    ).stdout
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = line.split("\0")
        if len(fields) != 3:
            continue
        name, peeled, tag_object = fields
        commit = peeled or tag_object
        checkpoint = _latest_checkpoint_record(root, commit)
        rows.append(
            {
                "name": name,
                "commit": commit,
                "checkpoint_id": (
                    checkpoint[2].get("checkpoint_id") if checkpoint else None
                ),
            }
        )
    return rows


def _resolve_research_object_source(source: str | Path) -> Path:
    """Resolve a regular source while rejecting user-spelled symlink components."""

    candidate = Path(source).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = Path(os.path.abspath(candidate))
    if sys.platform == "darwin" and candidate.parts[:2] in {
        ("/", "etc"),
        ("/", "tmp"),
        ("/", "var"),
    }:
        candidate = Path("/private") / candidate.relative_to("/")
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        try:
            metadata = cursor.lstat()
        except OSError as exc:
            raise ResearchGitError(
                f"research object is not a regular file: {candidate}"
            ) from exc
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(metadata.st_mode) or bool(
            getattr(metadata, "st_file_attributes", 0) & reparse_flag
        ):
            raise ResearchGitError(
                f"research object source contains a symbolic link: {candidate}"
            )
    if not stat.S_ISREG(candidate.lstat().st_mode):
        raise ResearchGitError(f"research object is not a regular file: {candidate}")
    return candidate


def _open_research_object_source(source: str | Path) -> tuple[Path, int]:
    """Open a source once without following symlinks or reparse points."""

    candidate = _resolve_research_object_source(source)
    read_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name == "posix" and hasattr(os, "O_NOFOLLOW"):
        directory_flags = (
            getattr(os, "O_PATH", os.O_RDONLY)
            | getattr(os, "O_DIRECTORY", 0)
            | os.O_NOFOLLOW
        )
        directory_fd = os.open(candidate.anchor, directory_flags)
        try:
            for component in candidate.parts[1:-1]:
                next_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fd,
                )
                os.close(directory_fd)
                directory_fd = next_fd
            source_fd = os.open(
                candidate.name,
                read_flags | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            raise ResearchGitError(
                f"research object source changed or contains a symbolic link: {candidate}"
            ) from exc
        finally:
            os.close(directory_fd)
    else:
        try:
            source_fd = os.open(candidate, read_flags)
        except OSError as exc:
            raise ResearchGitError(
                f"research object source could not be opened safely: {candidate}"
            ) from exc
        try:
            # Windows has no portable openat/O_NOFOLLOW.  Fail closed if any
            # component is or becomes a reparse point, and require the opened
            # handle to still identify the same regular leaf.
            checked = _resolve_research_object_source(candidate)
            if not os.path.samestat(os.fstat(source_fd), checked.lstat()):
                raise ResearchGitError(
                    f"research object source changed while opening: {candidate}"
                )
        except BaseException:
            os.close(source_fd)
            raise
    if not stat.S_ISREG(os.fstat(source_fd).st_mode):
        os.close(source_fd)
        raise ResearchGitError(f"research object is not a regular file: {candidate}")
    return candidate, source_fd


def _add_research_object_locked(
    root: Path,
    source: str | Path,
    *,
    logical_path: str | None = None,
    media_type: str | None = None,
) -> ObjectPointerResult:
    config = load_repository_config(root)
    source_path = _resolve_research_object_source(source)
    if logical_path:
        logical = _validate_research_logical_path(logical_path)
    else:
        try:
            logical = source_path.relative_to(root).as_posix()
        except ValueError:
            logical = _validate_research_logical_path(f"external/{source_path.name}")
    logical = _validate_research_logical_path(logical)
    if _matches(logical, SECRET_DENY_PATTERNS):
        raise ResearchGitError(
            f"refusing to register a denied secret/binary logical path: {logical}"
        )
    store_root = _configured_store_root(root)
    source_path, source_fd = _open_research_object_source(source_path)
    try:
        object_hash, object_size, store_path = _copy_into_store(
            source_fd,
            store_root,
            privacy_root=root,
        )
    except BaseException:
        try:
            os.close(source_fd)
        except OSError:
            pass
        raise
    digest = object_hash.split(":", 1)[1]
    pointer_dir = _resolve_configured_path(
        root,
        (config.get("storage") or {}).get("pointer_directory") or "research-objects",
        label="pointer directory",
    )
    logical_digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()[:16]
    pointer_path = pointer_dir / f"sha256-{digest}-{logical_digest}.json"
    pointer_payload = {
        "schema_version": OBJECT_POINTER_SCHEMA,
        "protocol_kind": "research_object_pointer",
        "object_hash": object_hash,
        "size": object_size,
        "media_type": media_type
        or mimetypes.guess_type(source_path.name)[0]
        or "application/octet-stream",
        "logical_path": logical,
        "store_relpath": store_path.relative_to(root).as_posix(),
        "created_at": _now_iso(),
    }
    logical_alias_key = _research_logical_alias_key(logical)
    for existing_hash, existing_pointers in _pointer_records(root, strict=True).items():
        for existing_pointer in existing_pointers:
            existing_logical = str(existing_pointer["logical_path"])
            if _research_logical_alias_key(existing_logical) == logical_alias_key and (
                existing_logical,
                existing_hash,
            ) != (logical, object_hash):
                raise ResearchGitError(
                    "research object logical paths collide on a portable filesystem: "
                    f"{existing_logical} and {logical}"
                )
    linked = False
    if pointer_path.is_symlink():
        raise ResearchGitError(
            f"research object pointer path contains a symbolic link: {pointer_path}"
        )
    if pointer_path.is_file():
        try:
            existing = _validate_pointer_payload(
                json.loads(pointer_path.read_text(encoding="utf-8")),
                pointer_path=pointer_path.relative_to(root).as_posix(),
            )
        except (OSError, json.JSONDecodeError, ResearchGitError) as exc:
            raise ResearchGitError(
                f"existing research object pointer is damaged: {pointer_path}"
            ) from exc
        stable_fields = (
            "object_hash",
            "size",
            "media_type",
            "logical_path",
            "store_relpath",
        )
        if any(
            existing.get(field) != pointer_payload.get(field) for field in stable_fields
        ):
            raise ResearchGitError(
                f"research object pointer identity collision: {pointer_path}"
            )
        pointer_payload = existing
        linked = True
    pointer_payload["pointer_hash"] = content_hash(
        {key: value for key, value in pointer_payload.items() if key != "pointer_hash"}
    )
    try:
        validate_json(pointer_payload, load_schema("research_object_pointer"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"generated object pointer is invalid: {exc.message}"
        ) from exc
    if not linked:
        _atomic_write_json(pointer_path, pointer_payload)
    return ObjectPointerResult(
        pointer_path=pointer_path,
        store_path=store_path,
        object_hash=object_hash,
        size=object_size,
        linked=linked,
    )


def add_research_object(
    repo: str | Path,
    source: str | Path,
    *,
    logical_path: str | None = None,
    media_type: str | None = None,
) -> ObjectPointerResult:
    root = _repository_root(repo)
    with _repository_lock(root):
        return _add_research_object_locked(
            root,
            source,
            logical_path=logical_path,
            media_type=media_type,
        )


def _add_research_objects_atomically(
    repo: str | Path,
    requests: Sequence[tuple[str | Path, str, str | None, str | None]],
) -> list[ObjectPointerResult]:
    """Publish several pointers as one operation, rolling back new pointers."""

    root = _repository_root(repo)
    results: list[ObjectPointerResult] = []
    with _repository_lock(root):
        try:
            for source, logical_path, media_type, expected_hash in requests:
                result = _add_research_object_locked(
                    root,
                    source,
                    logical_path=logical_path,
                    media_type=media_type,
                )
                results.append(result)
                if expected_hash is not None and result.object_hash != expected_hash:
                    raise ResearchGitError(
                        "research object changed after preflight hashing: "
                        + logical_path
                    )
        except BaseException:
            _rollback_new_research_object_pointers_locked(results)
            raise
    return results


def _rollback_new_research_object_pointers_locked(
    results: Sequence[ObjectPointerResult],
) -> None:
    for result in reversed(results):
        if result.linked:
            continue
        result.pointer_path.unlink(missing_ok=True)
        _fsync_directory(result.pointer_path.parent)


def _checkpoint_at_commit(
    repo: Path,
    commit: str,
    *,
    validate_rollback_edge: bool = True,
    checkpoint_candidates: Sequence[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    resolved = _run_git(
        repo,
        ["rev-parse", "--verify", f"{commit}^{{commit}}"],
    ).stdout.strip()
    trailer_body = _run_git(
        repo,
        ["show", "-s", "--format=%(trailers:only,unfold)", resolved],
    ).stdout
    trailers: dict[str, list[str]] = {}
    for line in trailer_body.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        trailers.setdefault(key, []).append(value.strip())

    required_trailers = (
        "Research-Checkpoint",
        "Research-Stage",
        "Research-State",
        "Research-Event",
    )
    for key in required_trailers:
        values = trailers.get(key) or []
        if len(values) != 1 or not values[0]:
            detail = "missing" if not values else "ambiguous"
            raise ResearchGitError(
                f"commit {resolved} is not bound to a research checkpoint: "
                f"{detail} {key} trailer"
            )

    checkpoint_id = trailers["Research-Checkpoint"][0]
    record = _checkpoint_by_id_at_commit(
        repo,
        resolved,
        checkpoint_id,
        candidate_paths=checkpoint_candidates,
    )
    if record is None:
        raise ResearchGitError(
            f"commit {resolved} is not bound to research checkpoint "
            f"{checkpoint_id}: matching checkpoint JSON is absent"
        )
    checkpoint_path, raw_payload = record
    payload = _validate_checkpoint_payload(
        raw_payload,
        checkpoint_path=f"{resolved}:{checkpoint_path}",
    )

    event_hash = str(payload["content_hash"])
    expected_checkpoint_id = f"rcp-{event_hash.split(':', 1)[1][:16]}"
    binding_fields = {
        "Research-Checkpoint": (checkpoint_id, expected_checkpoint_id),
        "Research-Event": (trailers["Research-Event"][0], event_hash),
        "Research-Stage": (trailers["Research-Stage"][0], str(payload["stage"])),
        "Research-State": (trailers["Research-State"][0], str(payload["status"])),
    }
    for key, (actual, expected) in binding_fields.items():
        if actual != expected:
            raise ResearchGitError(
                f"commit {resolved} research checkpoint binding mismatch: "
                f"{key} does not match {checkpoint_path}"
            )

    ancestry = _run_git(
        repo,
        ["rev-list", "--parents", "-n", "1", resolved],
    ).stdout.split()
    first_parent = ancestry[1] if len(ancestry) > 1 else None
    if payload.get("parent_commit") != first_parent:
        raise ResearchGitError(
            f"commit {resolved} is not bound to research checkpoint "
            f"{checkpoint_id}: parent_commit does not match its first Git parent"
        )

    if first_parent is None:
        commit_paths = set(
            _run_git(repo, ["ls-tree", "-r", "--name-only", resolved])
            .stdout.strip()
            .splitlines()
        )
    else:
        commit_paths = set(
            _run_git(
                repo,
                ["diff", "--no-renames", "--name-only", first_parent, resolved],
            )
            .stdout.strip()
            .splitlines()
        )
    if checkpoint_path not in commit_paths:
        raise ResearchGitError(
            f"commit {resolved} is not bound to research checkpoint "
            f"{checkpoint_id}: checkpoint JSON was not changed by this commit"
        )
    declared_paths = {str(item) for item in payload.get("changed_paths") or []}
    actual_material_paths = {
        path for path in commit_paths if not path.startswith("checkpoints/")
    }
    if declared_paths != actual_material_paths:
        raise ResearchGitError(
            f"commit {resolved} is not bound to research checkpoint "
            f"{checkpoint_id}: changed_paths does not match the committed material"
        )
    is_revert = str(payload.get("stage") or "") == "revert"
    reverts_commit = str(payload.get("reverts_commit") or "")
    reverts_checkpoint_hash = str(payload.get("reverts_checkpoint_hash") or "")
    revert_trailer = trailers.get("Research-Reverts") or []
    revert_checkpoint_trailer = trailers.get("Research-Reverts-Checkpoint") or []
    if is_revert:
        if (
            len(revert_trailer) != 1
            or len(revert_checkpoint_trailer) != 1
            or revert_trailer[0] != reverts_commit
            or revert_checkpoint_trailer[0] != reverts_checkpoint_hash
        ):
            raise ResearchGitError(
                f"commit {resolved} rollback trailers do not match its checkpoint"
            )
        if first_parent is None:
            raise ResearchGitError("a revert checkpoint must have a parent commit")
        if validate_rollback_edge:
            ancestry_check = _run_git(
                repo,
                ["merge-base", "--is-ancestor", reverts_commit, first_parent],
                check=False,
            )
            if ancestry_check.returncode:
                raise ResearchGitError(
                    f"commit {resolved} rollback target is not in first-parent history"
                )
            _target_path, target_checkpoint = _checkpoint_at_commit(
                repo,
                reverts_commit,
                validate_rollback_edge=False,
            )
            if target_checkpoint.get("content_hash") != reverts_checkpoint_hash:
                raise ResearchGitError(
                    f"commit {resolved} rollback checkpoint hash does not match target"
                )
            _validate_committed_inverse_revert(
                repo,
                revert_commit=resolved,
                revert_parent=first_parent,
                target_commit=reverts_commit,
            )
    elif (
        reverts_commit
        or reverts_checkpoint_hash
        or revert_trailer
        or revert_checkpoint_trailer
    ):
        raise ResearchGitError(
            f"commit {resolved} has rollback metadata outside a revert checkpoint"
        )
    return checkpoint_path, payload


def show_checkpoint(repo: str | Path, commit: str = "HEAD") -> dict[str, Any]:
    root = _repository_root(repo)
    resolved = _run_git(
        root, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).stdout.strip()
    path, payload = _checkpoint_at_commit(root, resolved)
    return {
        "commit": resolved,
        "path": path,
        "checkpoint_hash_valid": _checkpoint_hash_valid(payload),
        "checkpoint": payload,
    }


def _checkpoint_json_paths_changed_in_commit(
    repo: Path,
    commit: str,
) -> list[str]:
    """Return only checkpoint JSON paths changed by one exact first-parent edge."""

    ancestry = _run_git(
        repo,
        ["rev-list", "--parents", "-n", "1", commit],
    ).stdout.split()
    first_parent = ancestry[1] if len(ancestry) > 1 else None
    if first_parent is None:
        raw_paths = _run_git(
            repo,
            ["ls-tree", "-r", "--name-only", commit, "--", "checkpoints"],
        ).stdout.splitlines()
    else:
        raw_paths = _run_git(
            repo,
            [
                "diff",
                "--no-renames",
                "--name-only",
                first_parent,
                commit,
                "--",
                "checkpoints",
            ],
        ).stdout.splitlines()
    candidates = sorted(
        path
        for path in raw_paths
        if path.startswith("checkpoints/") and path.endswith(".json")
    )
    if not candidates or len(candidates) > 32:
        raise ResearchGitError(
            f"cannot resolve a bounded checkpoint set at commit {commit}"
        )
    return candidates


def research_object_origin_checkpoint(
    repo: str | Path,
    object_id: str,
    *,
    kind: str,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Resolve one exact object and its introducing checkpoint in bounded work.

    Unlike ``research_blame``, this narrow recovery helper does not construct a
    repository-wide reverse-relation view. It is intended for delayed evidence
    binding, where the caller already holds the full immutable object ID and
    only needs reproduction metadata from the introducing checkpoint.
    """

    selector = str(object_id or "").strip()
    if re.fullmatch(r"rso-[0-9a-f]{16}", selector) is None:
        raise ResearchGitError(
            "origin checkpoint lookup requires a full research object ID"
        )
    selected_kind = str(kind or "").strip()
    if re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", selected_kind) is None:
        raise ResearchGitError("origin checkpoint lookup requires a valid object kind")
    root = _repository_root(repo)
    resolved_commit = _run_git(
        root, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).stdout.strip()
    path = f".xscientist/objects/{selected_kind}/{selector}.json"
    object_blob = _run_git(
        root,
        ["show", f"{resolved_commit}:{path}"],
        check=False,
    )
    if object_blob.returncode:
        raise ResearchGitError(f"research object not found at {commit}: {selector}")
    try:
        payload = validate_research_object(json.loads(object_blob.stdout))
    except (json.JSONDecodeError, ResearchObjectError) as exc:
        raise ResearchGitError(
            f"invalid typed research object at {resolved_commit}:{path}"
        ) from exc
    if str(payload.get("object_id") or "") != selector:
        raise ResearchGitError(
            f"research object identity disagrees with its path at {commit}: {selector}"
        )
    if str(payload.get("kind") or "") != selected_kind:
        raise ResearchGitError(
            f"research object kind disagrees with its path at {commit}: {selector}"
        )

    raw_origin = _run_git(
        root,
        [
            "log",
            "--full-history",
            "--topo-order",
            "--max-count=2",
            "--diff-filter=A",
            "--format=%H%x00%aI%x00%an%x00%s",
            resolved_commit,
            "--",
            path,
        ],
    ).stdout.splitlines()
    if not raw_origin:
        raise ResearchGitError(f"cannot locate research object origin: {selector}")
    if len(raw_origin) != 1:
        raise ResearchGitError(
            "research object has multiple reachable origins at the selected ref; "
            "select a branch or parent ref with a unique origin: " + selector
        )
    fields = raw_origin[0].split("\0", 3)
    if len(fields) != 4:
        raise ResearchGitError(f"cannot parse research object origin: {selector}")
    origin_commit, authored_at, author, subject = fields
    checkpoint_path, checkpoint = _checkpoint_at_commit(
        root,
        origin_commit,
        checkpoint_candidates=_checkpoint_json_paths_changed_in_commit(
            root, origin_commit
        ),
    )
    if path not in {str(item) for item in checkpoint.get("changed_paths") or []}:
        raise ResearchGitError(
            f"object origin is not bound to its research checkpoint: {selector}"
        )
    return {
        "resolved_object_id": selector,
        "object": {
            "object_id": selector,
            "kind": payload["kind"],
            "state": payload["state"],
            "content_hash": payload["content_hash"],
            "path": path,
        },
        "origin": {
            "commit": origin_commit,
            "authored_at": authored_at,
            "author": author,
            "subject": subject,
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "checkpoint_hash": checkpoint.get("content_hash"),
        },
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
    }


def _research_objects_at_commit(
    repo: Path,
    commit: str,
) -> dict[str, dict[str, Any]]:
    tree = _run_git(repo, ["ls-tree", "-r", "--name-only", commit]).stdout.splitlines()
    objects: dict[str, dict[str, Any]] = {}
    for path in sorted(
        item
        for item in tree
        if item.startswith(".xscientist/objects/") and item.endswith(".json")
    ):
        try:
            payload = validate_research_object(
                json.loads(_run_git(repo, ["show", f"{commit}:{path}"]).stdout)
            )
        except (json.JSONDecodeError, ResearchObjectError) as exc:
            raise ResearchGitError(
                f"invalid typed research object at {commit}:{path}"
            ) from exc
        object_id = str(payload["object_id"])
        if object_id in objects:
            raise ResearchGitError(
                f"duplicate typed research object at {commit}: {object_id}"
            )
        objects[object_id] = {**payload, "repository_path": path}
    return objects


def list_research_objects_at_ref(
    repo: str | Path,
    ref: str = "HEAD",
) -> list[dict[str, Any]]:
    """List immutable scientific objects at one committed research ref."""

    root = _repository_root(repo)
    resolved = _run_git(root, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    objects = _research_objects_at_commit(root, resolved.stdout.strip())
    return [
        {
            key: value
            for key, value in objects[object_id].items()
            if key != "repository_path"
        }
        for object_id in sorted(objects)
    ]


def verify_research_repository(
    repo: str | Path,
    *,
    commit: str = "HEAD",
    verify_objects: bool = True,
) -> dict[str, Any]:
    """Verify the scientific closure visible from one Git revision."""

    root = _repository_root(repo)
    store_root = _configured_store_root(root)
    resolved = _run_git(root, ["rev-parse", commit]).stdout.strip()
    tree = set(
        _run_git(root, ["ls-tree", "-r", "--name-only", resolved]).stdout.splitlines()
    )
    errors: list[str] = []
    warnings: list[str] = []
    checkpoints: list[tuple[str, dict[str, Any]]] = []
    pointer_records: dict[str, list[dict[str, Any]]] = {}
    pointer_logical_bindings: dict[tuple[str, ...], tuple[str, str]] = {}
    checked = {
        "checkpoints": 0,
        "pointers": 0,
        "objects": 0,
        "research_objects": 0,
        "ara_manifests": 0,
        "ara_graphs": 0,
    }

    typed_objects: dict[str, dict[str, Any]] = {}
    try:
        typed_objects = _research_objects_at_commit(root, resolved)
        checked["research_objects"] = len(typed_objects)
    except ResearchGitError as exc:
        errors.append(str(exc))
    for object_id, payload in typed_objects.items():
        for relation in payload.get("relations") or []:
            relation_type = str(relation.get("type") or "")
            if relation_type not in RESEARCH_RELATION_TYPES:
                continue
            target = str(relation.get("target") or "")
            target_object = typed_objects.get(target)
            if target_object is None:
                errors.append(
                    f"typed research object {object_id} references missing object: {target}"
                )
                continue
            expected_kinds = RESEARCH_RELATION_TARGET_KINDS.get(relation_type, ())
            actual_kind = str(target_object.get("kind") or "")
            if expected_kinds and actual_kind not in expected_kinds:
                errors.append(
                    f"typed research object {object_id} relation {relation_type} "
                    f"requires target kind {' or '.join(expected_kinds)}; got "
                    f"{actual_kind}: {target}"
                )
    legacy_identity_ids = sorted(
        object_id
        for object_id, payload in typed_objects.items()
        if str(payload.get("identity_profile") or "")
        != "xscientist.research-object-identity.v2"
    )
    if legacy_identity_ids:
        warnings.append(
            f"{len(legacy_identity_ids)} legacy research object(s) have no "
            "authenticated v2 envelope; created_at is ignored and latest/recency "
            "uses Git introduction order or immutable ID fallback. Record a new "
            "superseding v2 object to migrate without rewriting history."
        )

    for checkpoint_path in sorted(
        path
        for path in tree
        if path.startswith("checkpoints/") and path.endswith(".json")
    ):
        checked["checkpoints"] += 1
        try:
            payload = json.loads(
                _run_git(root, ["show", f"{resolved}:{checkpoint_path}"]).stdout
            )
            payload = _validate_checkpoint_payload(
                payload,
                checkpoint_path=f"{resolved}:{checkpoint_path}",
            )
            checkpoints.append((checkpoint_path, payload))
        except (json.JSONDecodeError, ResearchGitError) as exc:
            errors.append(str(exc))

    checkpoints_by_hash: dict[str, tuple[str, dict[str, Any]]] = {}
    parent_graph: dict[str, list[str]] = {}
    for checkpoint_path, payload in checkpoints:
        checkpoint_hash = str(payload.get("content_hash") or "")
        if checkpoint_hash in checkpoints_by_hash:
            errors.append(f"duplicate checkpoint content hash: {checkpoint_hash}")
        checkpoints_by_hash[checkpoint_hash] = (checkpoint_path, payload)
    for checkpoint_hash, (checkpoint_path, payload) in checkpoints_by_hash.items():
        if "parent_checkpoint_hashes" in payload:
            parents = [
                str(item) for item in payload.get("parent_checkpoint_hashes") or []
            ]
        else:
            previous = payload.get("previous_checkpoint_hash")
            parents = [str(previous)] if previous else []
        parent_graph[checkpoint_hash] = parents
        compatibility_parent = parents[0] if parents else None
        if payload.get("previous_checkpoint_hash") != compatibility_parent:
            errors.append(
                f"checkpoint compatibility parent mismatch at {checkpoint_path}"
            )
        parent_sequences: list[int] = []
        for parent_hash in parents:
            parent = checkpoints_by_hash.get(parent_hash)
            if parent is None:
                errors.append(
                    f"checkpoint {checkpoint_path} references unknown parent: {parent_hash}"
                )
                continue
            parent_sequences.append(int(parent[1].get("sequence") or 0))
        expected_sequence = max(parent_sequences, default=0) + 1
        if int(payload.get("sequence") or 0) != expected_sequence:
            errors.append(
                f"checkpoint sequence mismatch at {checkpoint_path}: "
                f"expected {expected_sequence}, got {payload.get('sequence')}"
            )
        if payload.get("stage") == "revert":
            target_commit = str(payload.get("reverts_commit") or "")
            try:
                _target_path, target_checkpoint = _checkpoint_at_commit(
                    root, target_commit
                )
            except ResearchGitError as exc:
                errors.append(
                    f"checkpoint {checkpoint_path} has invalid rollback target: {exc}"
                )
            else:
                if target_checkpoint.get("content_hash") != payload.get(
                    "reverts_checkpoint_hash"
                ):
                    errors.append(
                        f"checkpoint {checkpoint_path} rollback target hash mismatch"
                    )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(checkpoint_hash: str) -> None:
        if checkpoint_hash in visited:
            return
        if checkpoint_hash in visiting:
            errors.append(f"checkpoint ancestry contains a cycle at {checkpoint_hash}")
            return
        visiting.add(checkpoint_hash)
        for parent_hash in parent_graph.get(checkpoint_hash, []):
            if parent_hash in parent_graph:
                visit(parent_hash)
        visiting.remove(checkpoint_hash)
        visited.add(checkpoint_hash)

    for checkpoint_hash in parent_graph:
        visit(checkpoint_hash)

    for pointer_path in sorted(
        path
        for path in tree
        if path.startswith("research-objects/") and path.endswith(".json")
    ):
        checked["pointers"] += 1
        try:
            payload = json.loads(
                _run_git(root, ["show", f"{resolved}:{pointer_path}"]).stdout
            )
            payload = _validate_pointer_payload(payload, pointer_path=pointer_path)
            object_hash = str(payload["object_hash"])
            _record_pointer(
                pointer_records,
                pointer_logical_bindings,
                object_hash=object_hash,
                record={**payload, "pointer_path": pointer_path},
            )
        except (json.JSONDecodeError, ResearchGitError) as exc:
            errors.append(str(exc))

    latest: dict[str, Any] | None = None
    try:
        _latest_path, latest_payload = _checkpoint_at_commit(root, resolved)
        latest = _validate_checkpoint_payload(
            latest_payload,
            checkpoint_path=f"{resolved}:{_latest_path}",
        )
    except ResearchGitError as exc:
        errors.append(str(exc))
    if latest is None:
        errors.append(f"commit {resolved} contains no valid research checkpoint")
    else:
        if latest.get("branch") in {"main", "stable"}:
            for conflict in _stable_evolution_conflicts({}, typed_objects):
                errors.append(
                    "stable research contains an ungated agent candidate: "
                    + str(conflict.get("candidate") or "unknown")
                )
        for object_hash in latest.get("object_refs") or []:
            object_pointers = pointer_records.get(str(object_hash))
            if not object_pointers:
                errors.append(f"checkpoint references missing pointer: {object_hash}")
                continue
            if verify_objects:
                checked["objects"] += 1
                for pointer in object_pointers:
                    _store_path, object_error = _verify_pointer_object(
                        root,
                        str(object_hash),
                        pointer,
                        store_root=store_root,
                    )
                    if object_error:
                        errors.append(object_error)
        for manifest_ref in latest.get("ara_manifests") or []:
            checked["ara_manifests"] += 1
            manifest_path = str(manifest_ref.get("path") or "")
            if manifest_path not in tree:
                errors.append(
                    f"checkpoint references missing ARA manifest: {manifest_path}"
                )
                continue
            try:
                manifest = json.loads(
                    _run_git(root, ["show", f"{resolved}:{manifest_path}"]).stdout
                )
                actual_hash = hash_manifest(manifest)
            except (
                json.JSONDecodeError,
                ResearchGitError,
                TypeError,
                ValueError,
            ) as exc:
                errors.append(f"cannot validate ARA manifest {manifest_path}: {exc}")
                continue
            expected_hash = str(manifest_ref.get("manifest_hash") or "")
            if actual_hash != expected_hash:
                errors.append(
                    f"ARA manifest hash mismatch for {manifest_path}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            expected_graph_hash = str(manifest_ref.get("exploration_graph_hash") or "")
            if expected_graph_hash:
                checked["ara_graphs"] += 1
                graph_path = (
                    PurePosixPath(manifest_path).parent / "exploration_graph.json"
                ).as_posix()
                if graph_path not in tree:
                    errors.append(
                        f"ARA manifest references missing exploration graph: {graph_path}"
                    )
                    continue
                try:
                    graph = json.loads(
                        _run_git(root, ["show", f"{resolved}:{graph_path}"]).stdout
                    )
                except json.JSONDecodeError as exc:
                    errors.append(f"cannot validate ARA graph {graph_path}: {exc}")
                    continue
                actual_graph_hash = content_hash(graph)
                if actual_graph_hash != expected_graph_hash:
                    errors.append(
                        f"ARA exploration graph hash mismatch for {graph_path}: "
                        f"expected {expected_graph_hash}, got {actual_graph_hash}"
                    )

    if not verify_objects:
        warnings.append("CAS payload verification was skipped by request")
    return {
        "schema_version": "xscientist.research-fsck.v1",
        "repository": str(root),
        "commit": resolved,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": checked,
    }


def research_log(
    repo: str | Path,
    *,
    limit: int = 20,
    ref: str = "HEAD",
    bounded: bool = False,
) -> list[dict[str, Any]]:
    root = _repository_root(repo)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ResearchGitError("log limit must be at least 1")
    if not isinstance(bounded, bool):
        raise ResearchGitError("bounded must be boolean")
    if bounded and limit > 128:
        raise ResearchGitError("bounded log limit exceeds safety cap")
    if bounded and (
        not isinstance(ref, str)
        or not ref.strip()
        or len(ref) > 256
        or "\x00" in ref
        or ref.startswith("-")
    ):
        raise ResearchGitError("bounded log ref is invalid")
    separator = "%x1f"
    record = "%x1e"
    if bounded:
        # A process audit needs commit identity, subject, and trailers only.
        # ``%B`` can contain an arbitrarily large prompt/transcript-like body;
        # trailers are sufficient for parent/checkpoint metadata and keep the
        # Git subprocess output bounded by the number of rows.
        subject = "%<(160,trunc)%s"
        body = "%(trailers:unfold,only)"
    else:
        subject = "%s"
        body = "%B"
    author = "%<(160,trunc)%an" if bounded else "%an"
    fmt = f"%H{separator}%h{separator}%aI{separator}{author}{separator}{subject}{separator}{body}{record}"
    resolved_ref = _run_git(
        root, ["rev-parse", "--verify", f"{ref}^{{commit}}"]
    ).stdout.strip()
    if bounded:
        raw = _run_git_bounded(
            root,
            [
                "log",
                "--topo-order",
                f"--max-count={limit}",
                f"--format={fmt}",
                resolved_ref,
            ],
            max_output_bytes=min(
                _HARD_MAX_BOUNDED_LOG_OUTPUT, max(64 * 1024, limit * 8192)
            ),
        ).stdout
    else:
        raw = _run_git(
            root,
            [
                "log",
                "--topo-order",
                f"--max-count={limit}",
                f"--format={fmt}",
                resolved_ref,
            ],
        ).stdout
    entries: list[dict[str, Any]] = []
    for item in raw.split("\x1e"):
        fields = item.strip("\n").split("\x1f", 5)
        if len(fields) != 6:
            continue
        full, short, authored_at, author, subject, body = fields
        trailers: dict[str, list[str]] = {}
        for line in body.splitlines():
            if ": " not in line:
                continue
            key, value = line.split(": ", 1)
            if key.startswith(("Research-", "ARA-", "Reproduce")):
                # Trailer values are metadata hints, not a transcript.  Keep
                # each value bounded before it reaches the redacted process
                # report while preserving the prefix needed for stage/state
                # recovery.
                trailers.setdefault(key, []).append(value[:512])
        entries.append(
            {
                "commit": full,
                "short_commit": short,
                "authored_at": authored_at,
                "author": author,
                "subject": subject,
                "trailers": trailers,
            }
        )
    return entries


def _trajectory_checkpoint_at_commit(
    repo: Path,
    *,
    commit: str,
    validate_rollback_edge: bool = True,
) -> dict[str, Any]:
    """Load one exact checkpoint using only byte- and time-bounded Git reads."""

    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise ResearchGitError("structured trajectory commit identity is invalid")
    trailer_body = _run_git_bounded(
        repo,
        ["show", "-s", "--format=%(trailers:only,unfold)", commit],
        max_output_bytes=64 * 1024,
    ).stdout
    trailers: dict[str, list[str]] = {}
    for line in trailer_body.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key.startswith("Research-"):
            trailers.setdefault(key, []).append(value.strip())
    required_trailers = (
        "Research-Checkpoint",
        "Research-Stage",
        "Research-State",
        "Research-Event",
    )
    for key in required_trailers:
        values = trailers.get(key) or []
        if len(values) != 1 or not values[0]:
            detail = "missing" if not values else "ambiguous"
            raise ResearchGitError(
                "structured trajectory commit is not bound to exactly one "
                f"checkpoint: {detail} {key} trailer"
            )
    checkpoint_id = trailers["Research-Checkpoint"][0]
    if re.fullmatch(r"rcp-[0-9a-f]{16}", checkpoint_id) is None:
        raise ResearchGitError(
            "structured trajectory checkpoint trailer identity is invalid"
        )

    ancestry = _run_git_bounded(
        repo,
        ["rev-list", "--parents", "-n", "1", commit],
        max_output_bytes=64 * 1024,
    ).stdout.split()
    if not ancestry or ancestry[0] != commit:
        raise ResearchGitError("structured trajectory commit ancestry is invalid")
    parent_commits = ancestry[1:]
    first_parent = parent_commits[0] if parent_commits else None
    if first_parent is None:
        changed_args = [
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--no-renames",
            "--name-only",
            "-r",
            "-z",
            commit,
        ]
    else:
        changed_args = [
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            first_parent,
            commit,
        ]
    raw_changed_paths = _run_git_bounded(
        repo,
        changed_args,
        max_output_bytes=_HARD_MAX_BOUNDED_LOG_OUTPUT,
    ).stdout
    commit_paths = {
        _normalise_relative(path) for path in raw_changed_paths.split("\0") if path
    }
    checkpoint_paths = sorted(
        path
        for path in commit_paths
        if path.startswith("checkpoints/") and path.endswith(".json")
    )
    if (
        not checkpoint_paths
        or len(checkpoint_paths) > _MAX_TRAJECTORY_CHECKPOINT_CANDIDATES
    ):
        raise ResearchGitError(
            "structured trajectory checkpoint candidate set is missing or unbounded"
        )
    preferred_suffix = f"-{checkpoint_id.removeprefix('rcp-')[:8]}.json"
    candidates = sorted(
        checkpoint_paths,
        key=lambda path: (not path.endswith(preferred_suffix), path),
    )

    matches: list[tuple[str, dict[str, Any]]] = []
    for checkpoint_path in candidates:
        revision = f"{commit}:{checkpoint_path}"
        size_text = _run_git_bounded(
            repo,
            ["cat-file", "-s", revision],
            max_output_bytes=4096,
        ).stdout.strip()
        try:
            size = int(size_text)
        except ValueError as exc:
            raise ResearchGitError(
                "structured trajectory checkpoint size is invalid"
            ) from exc
        if size < 2 or size > _MAX_TRAJECTORY_CHECKPOINT_BYTES:
            raise ResearchGitError(
                "structured trajectory checkpoint exceeds the bounded view"
            )
        raw = _run_git_bounded(
            repo,
            ["show", revision],
            max_output_bytes=size + 1024,
        ).stdout
        try:
            candidate = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("checkpoint_id") == checkpoint_id
        ):
            matches.append((checkpoint_path, candidate))
    if len(matches) != 1:
        raise ResearchGitError(
            "structured trajectory commit does not contain exactly one matching "
            "checkpoint"
        )
    checkpoint_path, raw_payload = matches[0]
    payload = _validate_checkpoint_payload(
        raw_payload,
        checkpoint_path=f"{commit}:{checkpoint_path}",
    )
    event_hash = str(payload.get("content_hash") or "")
    expected_checkpoint_id = f"rcp-{event_hash.split(':', 1)[-1][:16]}"
    binding_fields = {
        "Research-Checkpoint": (checkpoint_id, expected_checkpoint_id),
        "Research-Event": (trailers["Research-Event"][0], event_hash),
        "Research-Stage": (
            trailers["Research-Stage"][0],
            str(payload.get("stage") or ""),
        ),
        "Research-State": (
            trailers["Research-State"][0],
            str(payload.get("status") or ""),
        ),
    }
    if any(actual != expected for actual, expected in binding_fields.values()):
        raise ResearchGitError(
            "structured trajectory commit/checkpoint trailer binding is invalid"
        )
    if payload.get("parent_commit") != first_parent:
        raise ResearchGitError(
            "structured trajectory checkpoint first-parent binding is invalid"
        )
    parent_checkpoint_hashes = [
        str(item) for item in payload.get("parent_checkpoint_hashes") or []
    ]
    if (
        len(parent_checkpoint_hashes) > len(parent_commits)
        or payload.get("previous_checkpoint_hash")
        != (parent_checkpoint_hashes[0] if parent_checkpoint_hashes else None)
        or (trailers.get("Research-Parent") or []) != parent_checkpoint_hashes
    ):
        raise ResearchGitError(
            "structured trajectory checkpoint-parent metadata is invalid"
        )
    declared_paths = {str(item) for item in payload.get("changed_paths") or []}
    actual_material_paths = {
        path for path in commit_paths if not path.startswith("checkpoints/")
    }
    if checkpoint_path not in commit_paths or declared_paths != actual_material_paths:
        raise ResearchGitError(
            "structured trajectory checkpoint paths disagree with the commit"
        )
    is_revert = str(payload.get("stage") or "") == "revert"
    reverts_commit = str(payload.get("reverts_commit") or "")
    reverts_checkpoint_hash = str(payload.get("reverts_checkpoint_hash") or "")
    revert_trailer = trailers.get("Research-Reverts") or []
    revert_checkpoint_trailer = trailers.get("Research-Reverts-Checkpoint") or []
    if is_revert:
        if (
            len(revert_trailer) != 1
            or len(revert_checkpoint_trailer) != 1
            or revert_trailer[0] != reverts_commit
            or revert_checkpoint_trailer[0] != reverts_checkpoint_hash
            or first_parent is None
        ):
            raise ResearchGitError("structured trajectory rollback metadata is invalid")
        if validate_rollback_edge:
            ancestry_check = _run_git_bounded(
                repo,
                ["merge-base", "--is-ancestor", reverts_commit, first_parent],
                max_output_bytes=4096,
                check=False,
            )
            if ancestry_check.returncode:
                raise ResearchGitError(
                    "structured trajectory rollback target is not an ancestor"
                )
            target = _trajectory_checkpoint_at_commit(
                repo,
                commit=reverts_commit,
                validate_rollback_edge=False,
            )
            target_checkpoint = target.get("checkpoint") or {}
            if target_checkpoint.get("content_hash") != reverts_checkpoint_hash:
                raise ResearchGitError(
                    "structured trajectory rollback target hash is invalid"
                )
            _validate_committed_inverse_revert(
                repo,
                revert_commit=commit,
                revert_parent=first_parent,
                target_commit=reverts_commit,
                bounded=True,
            )
    elif (
        reverts_commit
        or reverts_checkpoint_hash
        or revert_trailer
        or revert_checkpoint_trailer
    ):
        raise ResearchGitError(
            "structured trajectory contains rollback metadata outside a revert"
        )
    return {
        "commit": commit,
        "path": checkpoint_path,
        "checkpoint_hash_valid": _checkpoint_hash_valid(payload),
        "checkpoint": payload,
        "backend_parent_commits": parent_commits,
    }


def _trajectory_object_at_commit(
    repo: Path,
    *,
    commit: str,
    path: str,
) -> dict[str, Any]:
    """Load one checkpoint-declared object without exposing its semantic payload."""

    normalized = _normalise_relative(path)
    parts = PurePosixPath(normalized).parts
    if (
        len(parts) != 4
        or parts[:2] != (".xscientist", "objects")
        or re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", parts[2]) is None
        or re.fullmatch(r"rso-[0-9a-f]{16}\.json", parts[3]) is None
    ):
        raise ResearchGitError(
            f"structured trajectory contains an invalid object path: {normalized}"
        )
    revision = f"{commit}:{normalized}"
    size_text = _run_git_bounded(
        repo,
        ["cat-file", "-s", revision],
        max_output_bytes=4096,
    ).stdout.strip()
    try:
        size = int(size_text)
    except ValueError as exc:
        raise ResearchGitError(
            f"structured trajectory object size is invalid: {normalized}"
        ) from exc
    if size < 2 or size > _MAX_TRAJECTORY_OBJECT_BYTES:
        raise ResearchGitError(
            f"structured trajectory object exceeds the bounded view: {normalized}"
        )
    raw = _run_git_bounded(
        repo,
        ["show", revision],
        max_output_bytes=size + 1024,
    ).stdout
    try:
        research_object = validate_research_object(json.loads(raw))
    except (json.JSONDecodeError, ResearchObjectError) as exc:
        raise ResearchGitError(
            f"structured trajectory object is invalid: {normalized}"
        ) from exc
    expected_id = PurePosixPath(parts[3]).stem
    if (
        research_object.get("object_id") != expected_id
        or research_object.get("kind") != parts[2]
    ):
        raise ResearchGitError(
            f"structured trajectory object identity disagrees with its path: {normalized}"
        )
    actor = research_object.get("actor")
    actor = actor if isinstance(actor, Mapping) else {}
    profile = research_object.get("semantic_profile")
    profile = profile if isinstance(profile, Mapping) else {}
    return {
        "object_id": research_object["object_id"],
        "kind": research_object["kind"],
        "state": research_object["state"],
        "content_hash": research_object["content_hash"],
        "actor": {
            "actor_id": str(actor.get("actor_id") or ""),
            "authority": str(actor.get("authority") or ""),
        },
        "provenance_hash": content_hash(research_object.get("provenance") or {}),
        "semantic_profile": {
            "uri": str(profile.get("uri") or ""),
            "version": str(profile.get("version") or ""),
            "schema_digest": str(profile.get("schema_digest") or ""),
        },
        "relations": [dict(item) for item in research_object.get("relations") or []],
    }


def _trajectory_object_changes(
    repo: Path,
    *,
    commit: str,
    parent_commit: str | None,
    object_paths: Sequence[str],
    allow_removals: bool = False,
) -> dict[str, str]:
    """Return the exact add/delete state of immutable objects in one checkpoint."""

    if not object_paths:
        return {}
    if parent_commit:
        args = [
            "diff",
            "--no-renames",
            "--name-status",
            parent_commit,
            commit,
            "--",
            ".xscientist/objects",
        ]
    else:
        args = [
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--no-renames",
            "--name-status",
            "-r",
            commit,
            "--",
            ".xscientist/objects",
        ]
    raw = _run_git_bounded(
        repo,
        args,
        max_output_bytes=min(
            _HARD_MAX_BOUNDED_LOG_OUTPUT,
            max(4096, len(object_paths) * 256),
        ),
    ).stdout
    changes: dict[str, str] = {}
    for line in raw.splitlines():
        status, separator, path = line.partition("\t")
        normalized = _normalise_relative(path) if separator else ""
        if status not in {"A", "D"} or normalized not in object_paths:
            raise ResearchGitError(
                "structured trajectory attempted to mutate or rename an immutable "
                "research object"
            )
        if status == "D" and not allow_removals:
            raise ResearchGitError(
                "structured trajectory contains immutable object deletion outside "
                "a validated research revert"
            )
        if normalized in changes:
            raise ResearchGitError(
                "structured trajectory contains duplicate object path changes"
            )
        changes[normalized] = "added" if status == "A" else "removed"
    if set(changes) != set(object_paths):
        raise ResearchGitError(
            "structured trajectory object changes disagree with the checkpoint"
        )
    return changes


def research_trajectory(
    repo: str | Path,
    *,
    limit: int = 50,
    ref: str = "HEAD",
) -> dict[str, Any]:
    """Project typed objects and hash-valid checkpoints as scientific history.

    This is a bounded, payload-free inspection surface, not a second log and
    not a publication attestation.  The projection contains the exact object
    identities, relations, actors, checkpoint hashes and checkpoint-parent
    edges needed to navigate scientific history.  Semantic payloads remain in
    their immutable objects and can be inspected explicitly with ``show``.
    """

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < 1
        or limit > _MAX_TRAJECTORY_CHECKPOINTS
    ):
        raise ResearchGitError(
            f"trajectory limit must be between 1 and {_MAX_TRAJECTORY_CHECKPOINTS}"
        )
    root = _repository_root(repo)
    log_entries = research_log(
        root,
        limit=limit + 1,
        ref=ref,
        bounded=True,
    )
    if not log_entries:
        raise ResearchGitError("structured trajectory has no committed checkpoint")

    newest_first: list[dict[str, Any]] = []
    reached_initial_checkpoint = False
    for log_entry in log_entries[:limit]:
        try:
            shown = _trajectory_checkpoint_at_commit(
                root,
                commit=str(log_entry["commit"]),
            )
        except ResearchGitError as exc:
            raise ResearchGitError(
                "structured trajectory contains a commit without an exact, "
                "hash-valid Research VCS checkpoint"
            ) from exc
        if shown.get("checkpoint_hash_valid") is not True:
            raise ResearchGitError(
                "structured trajectory contains a hash-invalid checkpoint"
            )
        checkpoint = shown.get("checkpoint") or {}
        changed_paths = [str(item) for item in checkpoint.get("changed_paths") or []]
        object_paths = sorted(
            path for path in changed_paths if path.startswith(".xscientist/objects/")
        )
        parent_commit = str(checkpoint.get("parent_commit") or "").strip() or None
        object_changes = _trajectory_object_changes(
            root,
            commit=str(log_entry["commit"]),
            parent_commit=parent_commit,
            object_paths=object_paths,
            allow_removals=str(checkpoint.get("stage") or "") == "revert",
        )
        objects = []
        for path in object_paths:
            change = object_changes[path]
            source_commit = (
                str(log_entry["commit"]) if change == "added" else parent_commit
            )
            if source_commit is None:
                raise ResearchGitError(
                    "structured trajectory removed an object without a parent commit"
                )
            objects.append(
                {
                    **_trajectory_object_at_commit(
                        root,
                        commit=source_commit,
                        path=path,
                    ),
                    "change": change,
                }
            )
        backend_parent_commits = [
            str(item) for item in shown.get("backend_parent_commits") or []
        ]
        parent_checkpoint_hashes = [
            str(item) for item in checkpoint.get("parent_checkpoint_hashes") or []
        ]
        event = {
            "commit": str(log_entry["commit"]),
            "backend_parent_commits": backend_parent_commits,
            "parent_commits": backend_parent_commits[: len(parent_checkpoint_hashes)],
            "checkpoint_id": str(checkpoint.get("checkpoint_id") or ""),
            "checkpoint_hash": str(checkpoint.get("content_hash") or ""),
            "parent_checkpoint_hashes": parent_checkpoint_hashes,
            "stage": str(checkpoint.get("stage") or ""),
            "status": str(checkpoint.get("status") or ""),
            "actor": str(checkpoint.get("actor") or ""),
            "subject": str(checkpoint.get("subject") or "")[:512],
            "objects": objects,
            "artifact_paths": sorted(set(changed_paths) - set(object_paths)),
        }
        if str(checkpoint.get("stage") or "") == "revert":
            event["reverts_commit"] = str(checkpoint.get("reverts_commit") or "")
            event["reverts_checkpoint_hash"] = str(
                checkpoint.get("reverts_checkpoint_hash") or ""
            )
        newest_first.append(event)
        if (
            checkpoint.get("previous_checkpoint_hash") in {None, ""}
            and not event["parent_checkpoint_hashes"]
        ):
            reached_initial_checkpoint = True
            break

    if not newest_first:
        raise ResearchGitError("structured trajectory has no Research VCS checkpoint")
    if not reached_initial_checkpoint and len(log_entries) <= limit:
        raise ResearchGitError(
            "structured trajectory ended before its initial Research VCS checkpoint"
        )

    entries_by_commit = {str(entry["commit"]): entry for entry in newest_first}
    if len(entries_by_commit) != len(newest_first):
        raise ResearchGitError("structured trajectory contains duplicate commits")
    children: dict[str, set[str]] = {commit: set() for commit in entries_by_commit}
    in_degree: dict[str, int] = {commit: 0 for commit in entries_by_commit}
    for child_commit, entry in entries_by_commit.items():
        for parent_commit in entry.get("parent_commits") or []:
            parent = str(parent_commit)
            if parent not in entries_by_commit:
                continue
            children[parent].add(child_commit)
            in_degree[child_commit] += 1
    ready = sorted(
        (commit for commit, degree in in_degree.items() if degree == 0),
        key=lambda commit: (
            str(entries_by_commit[commit].get("checkpoint_hash") or ""),
            commit,
        ),
    )
    entries: list[dict[str, Any]] = []
    while ready:
        commit = ready.pop(0)
        entries.append(entries_by_commit[commit])
        for child in sorted(
            children[commit],
            key=lambda candidate: (
                str(entries_by_commit[candidate].get("checkpoint_hash") or ""),
                candidate,
            ),
        ):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
        ready.sort(
            key=lambda candidate: (
                str(entries_by_commit[candidate].get("checkpoint_hash") or ""),
                candidate,
            )
        )
    if len(entries) != len(entries_by_commit):
        raise ResearchGitError(
            "structured trajectory checkpoint graph contains a cycle"
        )
    for index, entry in enumerate(entries):
        entry["sequence"] = index
    boundary_parent_edges: list[dict[str, str]] = []
    for entry in entries:
        backend_parent_commits = [
            str(item) for item in entry.get("backend_parent_commits") or []
        ]
        parent_commits = [str(item) for item in entry.get("parent_commits") or []]
        parent_hashes = [
            str(item) for item in entry.get("parent_checkpoint_hashes") or []
        ]
        if len(parent_commits) != len(parent_hashes):
            raise ResearchGitError(
                "structured trajectory checkpoint-parent arity is invalid"
            )
        undeclared_research_parents = sorted(
            parent
            for parent in backend_parent_commits
            if parent in entries_by_commit and parent not in parent_commits
        )
        if undeclared_research_parents:
            raise ResearchGitError(
                "structured trajectory omits a reachable research parent edge"
            )
        for parent_commit, parent_hash in zip(parent_commits, parent_hashes):
            parent_entry = entries_by_commit.get(parent_commit)
            if parent_entry is None:
                boundary_parent_edges.append(
                    {
                        "child_commit": str(entry["commit"]),
                        "parent_commit": parent_commit,
                        "parent_checkpoint_hash": parent_hash,
                    }
                )
                continue
            if parent_entry.get("checkpoint_hash") != parent_hash or int(
                parent_entry.get("sequence") or 0
            ) >= int(entry.get("sequence") or 0):
                raise ResearchGitError(
                    "structured trajectory checkpoint-parent edge is invalid"
                )
    if reached_initial_checkpoint and boundary_parent_edges:
        raise ResearchGitError(
            "structured trajectory checkpoint-parent closure is incomplete"
        )
    rollback_edges: list[dict[str, str]] = []
    boundary_rollback_edges: list[dict[str, str]] = []
    for entry in entries:
        if entry.get("stage") != "revert":
            continue
        edge = {
            "revert_commit": str(entry["commit"]),
            "revert_checkpoint_hash": str(entry["checkpoint_hash"]),
            "target_commit": str(entry.get("reverts_commit") or ""),
            "target_checkpoint_hash": str(entry.get("reverts_checkpoint_hash") or ""),
        }
        target_entry = entries_by_commit.get(edge["target_commit"])
        if target_entry is None:
            boundary_rollback_edges.append(edge)
        elif target_entry.get("checkpoint_hash") != edge[
            "target_checkpoint_hash"
        ] or int(target_entry.get("sequence") or 0) >= int(entry.get("sequence") or 0):
            raise ResearchGitError("structured trajectory rollback edge is invalid")
        rollback_edges.append(edge)
    if reached_initial_checkpoint and boundary_rollback_edges:
        raise ResearchGitError(
            "structured trajectory rollback-edge closure is incomplete"
        )
    object_transition_count = sum(len(entry["objects"]) for entry in entries)
    object_count = len(
        {
            str(research_object.get("object_id") or "")
            for entry in entries
            for research_object in entry["objects"]
        }
    )
    core = {
        "schema_version": "xscientist.structured-trajectory-projection.v1",
        "resolved_head": str(newest_first[0]["commit"]),
        "complete": reached_initial_checkpoint,
        "truncated": not reached_initial_checkpoint,
        "payloads_disclosed": False,
        "checkpoint_count": len(entries),
        "object_count": object_count,
        "object_transition_count": object_transition_count,
        "boundary_parent_edges": sorted(
            boundary_parent_edges,
            key=lambda item: (
                item["child_commit"],
                item["parent_commit"],
                item["parent_checkpoint_hash"],
            ),
        ),
        "rollback_edges": sorted(
            rollback_edges,
            key=lambda item: (item["revert_commit"], item["target_commit"]),
        ),
        "boundary_rollback_edges": sorted(
            boundary_rollback_edges,
            key=lambda item: (item["revert_commit"], item["target_commit"]),
        ),
        "entries": entries,
    }
    return {
        **core,
        "selected_ref": str(ref),
        "projection_hash": content_hash(core),
    }


def validate_complete_research_trajectory(
    repo: str | Path,
    *,
    ref: str = "HEAD",
) -> dict[str, Any]:
    """Validate and return the complete canonical trajectory at one ref.

    Publication and closure audits must not validate only a recent slice. The
    canonical projection is therefore evaluated at the hard checkpoint cap and
    accepted only when it reaches the initial Research VCS checkpoint with no
    unresolved parent or rollback boundary. Histories beyond the bounded cap
    fail closed instead of being silently truncated.
    """

    projection = research_trajectory(
        repo,
        limit=_MAX_TRAJECTORY_CHECKPOINTS,
        ref=ref,
    )
    projection_core = {
        key: value
        for key, value in projection.items()
        if key not in {"selected_ref", "projection_hash"}
    }
    resolved = _run_git(
        _repository_root(repo),
        ["rev-parse", "--verify", f"{ref}^{{commit}}"],
    ).stdout.strip()
    if (
        projection.get("schema_version")
        != "xscientist.structured-trajectory-projection.v1"
        or projection.get("projection_hash") != content_hash(projection_core)
        or projection.get("resolved_head") != resolved
        or projection.get("complete") is not True
        or projection.get("truncated") is not False
        or projection.get("payloads_disclosed") is not False
        or projection.get("boundary_parent_edges") != []
        or projection.get("boundary_rollback_edges") != []
        or projection.get("checkpoint_count") != len(projection.get("entries") or [])
    ):
        raise ResearchGitError(
            "structured trajectory is incomplete, truncated, or hash-invalid"
        )
    return projection


def _set_delta(before: Iterable[Any], after: Iterable[Any]) -> dict[str, list[str]]:
    before_set = {str(item) for item in before}
    after_set = {str(item) for item in after}
    return {
        "added": sorted(after_set - before_set),
        "removed": sorted(before_set - after_set),
        "unchanged": sorted(before_set & after_set),
    }


def _manifest_delta(before: Iterable[Any], after: Iterable[Any]) -> dict[str, Any]:
    before_map = {
        str(item.get("path")): {
            "manifest_hash": str(item.get("manifest_hash") or ""),
            "exploration_graph_hash": str(item.get("exploration_graph_hash") or ""),
        }
        for item in before
        if isinstance(item, dict)
    }
    after_map = {
        str(item.get("path")): {
            "manifest_hash": str(item.get("manifest_hash") or ""),
            "exploration_graph_hash": str(item.get("exploration_graph_hash") or ""),
        }
        for item in after
        if isinstance(item, dict)
    }
    common = set(before_map) & set(after_map)
    return {
        "added": sorted(set(after_map) - set(before_map)),
        "removed": sorted(set(before_map) - set(after_map)),
        "changed": [
            {
                "path": path,
                "before": before_map[path]["manifest_hash"],
                "after": after_map[path]["manifest_hash"],
                "graph_before": before_map[path]["exploration_graph_hash"] or None,
                "graph_after": after_map[path]["exploration_graph_hash"] or None,
                "changed_fields": [
                    field
                    for field in ("manifest_hash", "exploration_graph_hash")
                    if before_map[path][field] != after_map[path][field]
                ],
            }
            for path in sorted(common)
            if before_map[path] != after_map[path]
        ],
        "unchanged": sorted(
            path for path in common if before_map[path] == after_map[path]
        ),
    }


def _typed_object_delta(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before_ids = set(before)
    after_ids = set(after)
    added_ids = sorted(after_ids - before_ids)
    removed_ids = sorted(before_ids - after_ids)
    common_ids = sorted(before_ids & after_ids)
    changed = [
        object_id
        for object_id in common_ids
        if before[object_id].get("content_hash") != after[object_id].get("content_hash")
    ]

    def summaries(
        source: dict[str, dict[str, Any]], identifiers: Sequence[str]
    ) -> list[dict[str, str]]:
        return [
            {
                "object_id": object_id,
                "kind": str(source[object_id].get("kind") or ""),
                "state": str(source[object_id].get("state") or ""),
                "content_hash": str(source[object_id].get("content_hash") or ""),
            }
            for object_id in identifiers
        ]

    before_relations = {
        (
            object_id,
            str(relation.get("type") or ""),
            str(relation.get("target") or ""),
            str(relation.get("role") or ""),
        )
        for object_id, payload in before.items()
        for relation in payload.get("relations") or []
    }
    after_relations = {
        (
            object_id,
            str(relation.get("type") or ""),
            str(relation.get("target") or ""),
            str(relation.get("role") or ""),
        )
        for object_id, payload in after.items()
        for relation in payload.get("relations") or []
    }

    def relation_rows(values: set[tuple[str, str, str, str]]) -> list[dict[str, str]]:
        return [
            {"source": source, "type": kind, "target": target, "role": role}
            for source, kind, target, role in sorted(values)
        ]

    by_kind: dict[str, dict[str, int]] = {}
    for object_id in added_ids:
        kind = str(after[object_id].get("kind") or "unknown")
        by_kind.setdefault(kind, {"added": 0, "removed": 0})["added"] += 1
    for object_id in removed_ids:
        kind = str(before[object_id].get("kind") or "unknown")
        by_kind.setdefault(kind, {"added": 0, "removed": 0})["removed"] += 1
    return {
        "added": summaries(after, added_ids),
        "removed": summaries(before, removed_ids),
        "unchanged": common_ids,
        "integrity_conflicts": changed,
        "by_kind": {key: by_kind[key] for key in sorted(by_kind)},
        "relations": {
            "added": relation_rows(after_relations - before_relations),
            "removed": relation_rows(before_relations - after_relations),
        },
    }


def _structured_value_diff(
    before: Any,
    after: Any,
    *,
    path: str = "$",
    changes: list[dict[str, Any]] | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    output = changes if changes is not None else []
    if len(output) >= limit or before == after:
        return output
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=str):
            if len(output) >= limit:
                break
            child = f"{path}.{key}"
            if key not in before:
                output.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                output.append({"path": child, "before": before[key], "after": None})
            else:
                _structured_value_diff(
                    before[key],
                    after[key],
                    path=child,
                    changes=output,
                    limit=limit,
                )
        return output
    output.append({"path": path, "before": before, "after": after})
    return output


def _json_at_revision(repo: Path, commit: str, path: str) -> tuple[Any, str | None]:
    blob = _run_git(repo, ["show", f"{commit}:{path}"], check=False)
    if blob.returncode:
        return None, None
    encoded = blob.stdout.encode("utf-8")
    if len(encoded) > 2 * 1024 * 1024:
        return None, "exceeds 2 MiB semantic diff limit"
    try:
        return json.loads(blob.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"


def research_diff(
    repo: str | Path,
    before: str = "HEAD~1",
    after: str = "HEAD",
    *,
    deep: bool = False,
) -> dict[str, Any]:
    root = _repository_root(repo)
    resolved_before = _run_git(root, ["rev-parse", before]).stdout.strip()
    resolved_after = _run_git(root, ["rev-parse", after]).stdout.strip()
    name_status = _run_git(
        root,
        [
            "-c",
            "core.quotePath=false",
            "diff",
            "--name-status",
            resolved_before,
            resolved_after,
        ],
    ).stdout.splitlines()
    stat = _run_git(
        root, ["diff", "--stat", resolved_before, resolved_after]
    ).stdout.rstrip()
    before_path, before_checkpoint = _checkpoint_at_commit(root, resolved_before)
    after_path, after_checkpoint = _checkpoint_at_commit(root, resolved_after)
    before_typed_objects = _research_objects_at_commit(root, resolved_before)
    after_typed_objects = _research_objects_at_commit(root, resolved_after)
    fields = {
        field: {
            "before": before_checkpoint.get(field),
            "after": after_checkpoint.get(field),
            "changed": before_checkpoint.get(field) != after_checkpoint.get(field),
        }
        for field in ("stage", "status", "subject", "summary", "actor", "branch")
    }
    semantic: dict[str, Any] = {
        "before_checkpoint": {
            "path": before_path,
            "checkpoint_id": before_checkpoint.get("checkpoint_id"),
            "content_hash": before_checkpoint.get("content_hash"),
            "hash_valid": _checkpoint_hash_valid(before_checkpoint),
        },
        "after_checkpoint": {
            "path": after_path,
            "checkpoint_id": after_checkpoint.get("checkpoint_id"),
            "content_hash": after_checkpoint.get("content_hash"),
            "hash_valid": _checkpoint_hash_valid(after_checkpoint),
        },
        "fields": fields,
        "nodes": _set_delta(
            before_checkpoint.get("nodes") or [],
            after_checkpoint.get("nodes") or [],
        ),
        "claims": _set_delta(
            before_checkpoint.get("claims") or [],
            after_checkpoint.get("claims") or [],
        ),
        "objects": _set_delta(
            before_checkpoint.get("object_refs") or [],
            after_checkpoint.get("object_refs") or [],
        ),
        "research_objects": _typed_object_delta(
            before_typed_objects,
            after_typed_objects,
        ),
        "ara_manifests": _manifest_delta(
            before_checkpoint.get("ara_manifests") or [],
            after_checkpoint.get("ara_manifests") or [],
        ),
        "environment_changed": (
            ((before_checkpoint.get("reproduce") or {}).get("environment") or {}).get(
                "content_hash"
            )
            != ((after_checkpoint.get("reproduce") or {}).get("environment") or {}).get(
                "content_hash"
            )
        ),
    }
    structured_changes: list[dict[str, Any]] = []
    structured_warnings: list[str] = []
    if deep:
        candidates: set[str] = set()
        for line in name_status:
            parts = line.split("\t")
            for candidate in parts[1:]:
                if candidate.endswith(".json") and (
                    candidate.startswith(("hypotheses/", "claims/", "ara/"))
                    or PurePosixPath(candidate).name
                    in {
                        "preregistration.json",
                        "research_plan.json",
                        "metrics.json",
                        "verification_report.json",
                        "exploration_graph.json",
                    }
                ):
                    candidates.add(candidate)
        for candidate in sorted(candidates):
            before_value, before_warning = _json_at_revision(
                root, resolved_before, candidate
            )
            after_value, after_warning = _json_at_revision(
                root, resolved_after, candidate
            )
            if before_warning:
                structured_warnings.append(f"{candidate} before: {before_warning}")
            if after_warning:
                structured_warnings.append(f"{candidate} after: {after_warning}")
            if before_warning or after_warning:
                continue
            for change in _structured_value_diff(before_value, after_value):
                structured_changes.append({"file": candidate, **change})
    semantic["structured_changes"] = structured_changes
    semantic["warnings"] = structured_warnings
    return {
        "before": resolved_before,
        "after": resolved_after,
        "changes": name_status,
        "stat": stat,
        "semantic": semantic,
    }


def research_blame(
    repo: str | Path,
    object_id: str,
    *,
    commit: str = "HEAD",
) -> dict[str, Any]:
    """Trace one immutable scientific object to its originating transition."""

    root = _repository_root(repo)
    resolved = _run_git(root, ["rev-parse", "--verify", f"{commit}^{{commit}}"])
    resolved_commit = resolved.stdout.strip()
    objects = _research_objects_at_commit(root, resolved_commit)
    selector = str(object_id or "").strip()

    # Resolve against the requested commit, not the mutable working tree.  The
    # previous implementation delegated selectors to the current object store,
    # which made ``blame @latest:hypothesis --commit OLD`` either fail or point
    # at a newer object than the one being audited.
    if selector.startswith("@latest"):
        prefix, separator, selected_kind = selector.partition(":")
        if prefix != "@latest" or not separator or not selected_kind:
            raise ResearchGitError("object selector must use @latest:<kind>")
        candidates = [
            (candidate_id, candidate)
            for candidate_id, candidate in objects.items()
            if str(candidate.get("kind") or "") == selected_kind
        ]
        if not candidates:
            raise ResearchGitError(
                f"no research objects found for selector at {commit}: {selector}"
            )
        commit_order = _research_object_commit_order(
            root,
            ref=resolved_commit,
            kind=selected_kind,
        )

        sequenced = [
            (
                candidate_id,
                commit_order.get(str(candidate.get("repository_path") or "")),
            )
            for candidate_id, candidate in candidates
        ]
        if any(sequence is None for _candidate_id, sequence in sequenced):
            raise ResearchGitError(
                "historical @latest cannot verify every object introduction"
            )
        latest_sequence = max(
            int(sequence or 0) for _candidate_id, sequence in sequenced
        )
        latest_ids = [
            candidate_id
            for candidate_id, sequence in sequenced
            if sequence == latest_sequence
        ]
        if len(latest_ids) != 1:
            raise ResearchGitError(
                f"historical @latest:{selected_kind} is ambiguous at {commit}; "
                "use an explicit object ID"
            )
        resolved_object_id = latest_ids[0]
    elif selector.startswith("urn:xscientist:research-object:sha256:"):
        matches = [
            candidate_id
            for candidate_id, candidate in objects.items()
            if candidate.get("qualified_id") == selector
        ]
        if len(matches) != 1:
            raise ResearchGitError(f"research object not found at {commit}: {selector}")
        resolved_object_id = matches[0]
    elif selector in objects:
        resolved_object_id = selector
    elif re.fullmatch(r"rso-[0-9a-f]{6,15}", selector):
        matches = sorted(
            candidate_id
            for candidate_id in objects
            if candidate_id.startswith(selector)
        )
        if len(matches) != 1:
            detail = "not found" if not matches else "ambiguous"
            raise ResearchGitError(
                f"{detail} research object prefix at {commit}: {selector}"
            )
        resolved_object_id = matches[0]
    else:
        resolved_object_id = selector

    payload = objects.get(resolved_object_id)
    if payload is None:
        raise ResearchGitError(
            f"research object not found at {commit}: {resolved_object_id}"
        )
    path = str(payload["repository_path"])
    raw = _run_git(
        root,
        [
            "log",
            "--full-history",
            "--topo-order",
            "--max-count=2",
            "--diff-filter=A",
            "--format=%H%x00%aI%x00%an%x00%s",
            resolved_commit,
            "--",
            path,
        ],
    ).stdout.splitlines()
    if not raw:
        raise ResearchGitError(
            f"cannot locate research object origin: {resolved_object_id}"
        )
    if len(raw) != 1:
        raise ResearchGitError(
            "research object has multiple reachable origins at the selected ref; "
            "select a branch or parent ref with a unique origin: "
            f"{resolved_object_id}"
        )
    fields = raw[0].split("\0", 3)
    if len(fields) != 4:
        raise ResearchGitError(
            f"cannot parse research object origin: {resolved_object_id}"
        )
    origin_commit, authored_at, author, subject = fields
    checkpoint = _latest_checkpoint_record(root, origin_commit)
    related_by = [
        {
            "object_id": source_id,
            "type": str(relation.get("type") or ""),
            "role": str(relation.get("role") or ""),
        }
        for source_id, source in sorted(objects.items())
        for relation in source.get("relations") or []
        if relation.get("target") == resolved_object_id
    ]
    return {
        "selector": selector,
        "resolved_object_id": resolved_object_id,
        "object": {
            "object_id": resolved_object_id,
            "kind": payload["kind"],
            "state": payload["state"],
            "content_hash": payload["content_hash"],
            "path": path,
        },
        "origin": {
            "commit": origin_commit,
            "authored_at": authored_at,
            "author": author,
            "subject": subject,
            "checkpoint_id": checkpoint[2].get("checkpoint_id") if checkpoint else None,
            "checkpoint_hash": (
                checkpoint[2].get("content_hash") if checkpoint else None
            ),
        },
        "relations": list(payload.get("relations") or []),
        "related_by": related_by,
    }


def _new_objects_since(
    base: dict[str, dict[str, Any]],
    side: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [side[object_id] for object_id in sorted(set(side) - set(base))]


def _semantic_merge_conflicts(
    base: dict[str, dict[str, Any]],
    ours: dict[str, dict[str, Any]],
    theirs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    theirs_new = _new_objects_since(base, theirs)
    conflicts: list[dict[str, Any]] = []
    combined_objects = {**base, **ours, **theirs}

    def relation_objects(
        values: Sequence[dict[str, Any]],
    ) -> dict[str, dict[str, set[str]]]:
        result: dict[str, dict[str, set[str]]] = {}
        for payload in values:
            for relation in payload.get("relations") or []:
                relation_type = str(relation.get("type") or "")
                normalized_relation = {
                    "qualified_supports": "supports",
                    "qualified_refutes": "refutes",
                }.get(relation_type, relation_type)
                if normalized_relation in {"supports", "refutes"}:
                    result.setdefault(str(relation.get("target") or ""), {}).setdefault(
                        normalized_relation, set()
                    ).add(str(payload["object_id"]))
        return result

    # Compare the effective target frontier with the frontier produced by the
    # merge.  Looking only at objects newly created on *both* branches misses a
    # common scientific conflict: support already exists at the merge base and
    # the source branch introduces a refutation (or vice versa).  A conflict is
    # reported only when the merge introduces at least one new opposed pair;
    # an already-contested target does not make every unrelated merge fail.
    ours_relations = relation_objects(list(ours.values()))
    merged_relations = relation_objects([*ours.values(), *theirs.values()])

    from .research_semantics import scopes_compatible

    def opposed_pairs(
        target: str,
        relations: dict[str, dict[str, set[str]]],
    ) -> set[tuple[str, str]]:
        supporting = relations.get(target, {}).get("supports", set())
        refuting = relations.get(target, {}).get("refutes", set())
        return {
            (support_id, refute_id)
            for support_id in supporting
            for refute_id in refuting
            if scopes_compatible(
                (combined_objects.get(support_id, {}).get("payload") or {}).get("scope")
                or {},
                (combined_objects.get(refute_id, {}).get("payload") or {}).get("scope")
                or {},
            )
        }

    for target in sorted(merged_relations):
        merged_pairs = opposed_pairs(target, merged_relations)
        introduced_pairs = merged_pairs - opposed_pairs(target, ours_relations)
        if not introduced_pairs:
            continue
        supporting = sorted({item[0] for item in merged_pairs})
        refuting = sorted({item[1] for item in merged_pairs})
        conflicts.append(
            {
                "type": "opposed_evidence",
                "target": target,
                "supporting_evidence": supporting,
                "refuting_evidence": refuting,
                "overlapping_scope_pairs": [
                    list(item) for item in sorted(introduced_pairs)
                ],
                "message": "the merge introduces support and refutation for the same research object",
            }
        )

    def preregistration_hypotheses(item: dict[str, Any]) -> set[str]:
        """Resolve the scientific hypothesis a registration governs."""

        payload = item.get("payload") or {}
        hypothesis_tokens: set[str] = set()

        def add_token(target: set[str], prefix: str, value: Any) -> None:
            normalized = " ".join(str(value or "").split()).casefold()
            if normalized:
                target.add(f"{prefix}:{normalized}")

        add_token(hypothesis_tokens, "hypothesis", payload.get("hypothesis_id"))
        add_token(hypothesis_tokens, "hypothesis", payload.get("idea_id"))
        hypotheses = payload.get("hypotheses") or {}
        if isinstance(hypotheses, dict):
            add_token(
                hypothesis_tokens,
                "statement",
                hypotheses.get("alternative"),
            )

        for relation in item.get("relations") or []:
            if relation.get("type") != "depends_on":
                continue
            related = combined_objects.get(str(relation.get("target") or "")) or {}
            if related.get("kind") == "hypothesis":
                add_token(hypothesis_tokens, "hypothesis", related.get("object_id"))
                continue
            if related.get("kind") != "research_plan":
                continue
            plan_payload = related.get("payload") or {}
            add_token(
                hypothesis_tokens,
                "hypothesis",
                plan_payload.get("hypothesis_id"),
            )
            for plan_relation in related.get("relations") or []:
                plan_target = (
                    combined_objects.get(str(plan_relation.get("target") or "")) or {}
                )
                if (
                    plan_relation.get("type") == "depends_on"
                    and plan_target.get("kind") == "hypothesis"
                ):
                    add_token(
                        hypothesis_tokens,
                        "hypothesis",
                        plan_target.get("object_id"),
                    )
        return hypothesis_tokens

    def preregistration_plan_identity(item: dict[str, Any]) -> set[str]:
        """Return immutable registration/plan identifiers, not broad topics."""

        payload = item.get("payload") or {}
        identities: set[str] = set()

        def add(prefix: str, value: Any) -> None:
            normalized = " ".join(str(value or "").split()).casefold()
            if normalized:
                identities.add(f"{prefix}:{normalized}")

        add("registration", payload.get("preregistration_id"))
        add("plan", payload.get("plan_id"))
        for relation in item.get("relations") or []:
            if relation.get("type") != "depends_on":
                continue
            related = combined_objects.get(str(relation.get("target") or "")) or {}
            if related.get("kind") != "research_plan":
                continue
            add("plan_object", related.get("object_id"))
            add("plan", (related.get("payload") or {}).get("plan_id"))
        return identities

    def preregistration_scopes(item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload") or {}
        structured = payload.get("scope")
        outcome_scopes: set[tuple[str, str]] = set()
        for outcome in payload.get("outcomes") or []:
            if not isinstance(outcome, dict):
                continue
            dataset = " ".join(str(outcome.get("dataset") or "").split()).casefold()
            metric = " ".join(str(outcome.get("metric") or "").split()).casefold()
            if dataset or metric:
                outcome_scopes.add((dataset, metric))
        return {
            "structured": structured if isinstance(structured, dict) else None,
            "outcomes": outcome_scopes,
        }

    def preregistration_scopes_overlap(
        left: dict[str, Any], right: dict[str, Any]
    ) -> bool:
        left_scope = preregistration_scopes(left)
        right_scope = preregistration_scopes(right)
        if (
            left_scope["structured"] is not None
            and right_scope["structured"] is not None
        ):
            return scopes_compatible(
                left_scope["structured"], right_scope["structured"]
            )
        left_outcomes = left_scope["outcomes"]
        right_outcomes = right_scope["outcomes"]
        if left_outcomes and right_outcomes:
            return any(
                (not left_dataset or not right_dataset or left_dataset == right_dataset)
                and (not left_metric or not right_metric or left_metric == right_metric)
                for left_dataset, left_metric in left_outcomes
                for right_dataset, right_metric in right_outcomes
            )
        # Missing scope is uncertainty, not evidence of independence.
        return True

    def preregistration_protocol(item: dict[str, Any]) -> dict[str, Any]:
        payload = item.get("payload") or {}
        volatile = {
            "created_at",
            "locked_at",
            "registered_by",
            "registration_hash",
            "status",
            "deviations",
        }
        return {key: value for key, value in payload.items() if key not in volatile}

    ours_locked = [
        item
        for item in ours.values()
        if item.get("kind") == "preregistration" and item.get("state") == "locked"
    ]
    theirs_locked = [
        item
        for item in theirs_new
        if item.get("kind") == "preregistration" and item.get("state") == "locked"
    ]
    for ours_item in ours_locked:
        for theirs_item in theirs_locked:
            ours_identity = preregistration_plan_identity(ours_item)
            theirs_identity = preregistration_plan_identity(theirs_item)
            if not ours_identity or not theirs_identity:
                continue
            if ours_identity.isdisjoint(theirs_identity):
                continue
            ours_hypotheses = preregistration_hypotheses(ours_item)
            theirs_hypotheses = preregistration_hypotheses(theirs_item)
            if not ours_hypotheses or not theirs_hypotheses:
                continue
            if ours_hypotheses.isdisjoint(theirs_hypotheses):
                continue
            if not preregistration_scopes_overlap(ours_item, theirs_item):
                continue
            if preregistration_protocol(ours_item) != preregistration_protocol(
                theirs_item
            ):
                conflicts.append(
                    {
                        "type": "locked_preregistration",
                        "ours": ours_item["object_id"],
                        "theirs": theirs_item["object_id"],
                        "message": (
                            "branches contain incompatible locked preregistrations "
                            "for the same hypothesis and overlapping scope"
                        ),
                    }
                )

    def metrics(
        values: Sequence[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for item in values:
            if item.get("kind") != "metric":
                continue
            payload = item.get("payload") or {}
            key = str(
                payload.get("metric_id")
                or payload.get("name")
                or payload.get("metric")
                or ""
            )
            if key:
                result.setdefault(key, []).append(item)
        return result

    ours_metrics = metrics(list(ours.values()))
    theirs_metrics = metrics(theirs_new)
    for key in sorted(set(ours_metrics) & set(theirs_metrics)):
        incompatible_pairs = [
            (ours_item, theirs_item)
            for ours_item in ours_metrics[key]
            for theirs_item in theirs_metrics[key]
            if ours_item.get("object_id") != theirs_item.get("object_id")
            and ours_item.get("payload") != theirs_item.get("payload")
        ]
        if incompatible_pairs:
            ours_item, theirs_item = sorted(
                incompatible_pairs,
                key=lambda pair: (
                    str(pair[0].get("object_id") or ""),
                    str(pair[1].get("object_id") or ""),
                ),
            )[0]
            conflicts.append(
                {
                    "type": "metric_definition",
                    "metric": key,
                    "ours": ours_item["object_id"],
                    "theirs": theirs_item["object_id"],
                    "message": "primary metric definitions differ",
                }
            )
    return conflicts


def _stable_evolution_conflicts(
    base: dict[str, dict[str, Any]],
    source: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    new_objects = {
        object_id: source[object_id] for object_id in sorted(set(source) - set(base))
    }
    draft_candidates = {
        object_id
        for object_id, payload in new_objects.items()
        if payload.get("kind") == "agent_candidate" and payload.get("state") == "draft"
    }
    authorized: set[str] = set()
    for promoted_id, payload in new_objects.items():
        if (
            payload.get("kind") != "agent_candidate"
            or payload.get("state") != "promoted"
        ):
            continue
        superseded = {
            str(item.get("target") or "")
            for item in payload.get("relations") or []
            if item.get("type") == "supersedes"
        }
        evaluation_ids = {
            str(item.get("target") or "")
            for item in payload.get("relations") or []
            if item.get("type") == "evaluates"
        }
        evaluations_valid = bool(evaluation_ids) and all(
            source.get(object_id, {}).get("kind") == "agent_evaluation"
            and source.get(object_id, {}).get("state") == "verified"
            and source.get(object_id, {}).get("payload", {}).get("decision")
            == "promote_to_canary"
            and not source.get(object_id, {})
            .get("payload", {})
            .get("required_failures")
            for object_id in evaluation_ids
        )
        promoted_payload = payload.get("payload") or {}
        candidate_payload = promoted_payload.get("candidate") or {}
        promotion_payload = promoted_payload.get("promotion") or {}
        exact_hash = candidate_payload.get("candidate_hash")
        promotion_valid = (
            bool(exact_hash)
            and promotion_payload.get("decision") == "approved"
            and promotion_payload.get("production_promotion_allowed") is True
            and (promotion_payload.get("candidate") or {}).get("candidate_hash")
            == exact_hash
            and bool(promotion_payload.get("approver_ids"))
            and all(
                str(item).startswith("human:")
                for item in promotion_payload.get("approver_ids") or []
            )
        )
        decision_valid = any(
            decision.get("kind") == "gate_decision"
            and decision.get("state") == "promoted"
            and any(
                relation.get("type") == "promotes"
                and relation.get("target") == promoted_id
                for relation in decision.get("relations") or []
            )
            for decision in new_objects.values()
        )
        if evaluations_valid and promotion_valid and decision_valid:
            authorized.update(superseded)
    return [
        {
            "type": "ungated_agent_candidate",
            "candidate": object_id,
            "message": "stable research cannot admit a shadow-only agent candidate",
        }
        for object_id in sorted(draft_candidates - authorized)
    ]


def _decorate_merge_conflicts(
    conflicts: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    guidance = {
        "opposed_evidence": [
            "preserve both evidence objects on a reconciliation research line",
            "record an independent review and explicit gate decision",
        ],
        "locked_preregistration": [
            "keep incompatible preregistrations as separate confirmatory studies",
            "record a superseding preregistration before any new experiment runs",
        ],
        "metric_definition": [
            "assign distinct metric identifiers or preregister one common definition",
        ],
        "working_state": [
            "resolve file conflicts on a dedicated reconciliation research line",
            "checkpoint the resolution before retrying merge preflight",
        ],
        "ungated_agent_candidate": [
            "complete independent evaluation, canary approval, and promotion gate",
        ],
    }
    decorated: list[dict[str, Any]] = []
    for conflict in conflicts:
        base = dict(conflict)
        conflict_type = str(base.get("type") or "unknown")
        identity = {
            key: value
            for key, value in sorted(base.items())
            if key not in {"message", "resolution", "conflict_id"}
        }
        decorated.append(
            {
                **base,
                "conflict_id": "rvc-" + content_hash(identity).split(":", 1)[-1][:16],
                "severity": "blocking",
                "resolution": guidance.get(
                    conflict_type,
                    ["record an explicit scientific resolution before merging"],
                ),
            }
        )
    return decorated


def preview_research_merge(repo: str | Path, source: str) -> dict[str, Any]:
    """Perform a read-only backend and scientific conflict preflight."""

    root = _repository_root(repo)
    normalized = _validate_research_ref_name(source, kind="branch")
    source_ref = f"refs/heads/{normalized}"
    source_commit = _run_git(
        root, ["rev-parse", "--verify", f"{source_ref}^{{commit}}"]
    ).stdout.strip()
    target_commit = _head(root)
    if target_commit is None:
        raise ResearchGitError("research merge requires an existing checkpoint")
    # A Research VCS merge may only converge exact scientific events. Falling
    # back to an ancestor checkpoint would let an ordinary Git commit smuggle
    # unreviewed paths into the merge index and then inherit trusted trailers.
    _checkpoint_at_commit(root, target_commit)
    _checkpoint_at_commit(root, source_commit)
    if source_commit == target_commit:
        raise ResearchGitError("research branch is already at the current checkpoint")
    base_commit = _run_git(
        root, ["merge-base", target_commit, source_commit]
    ).stdout.strip()
    backend = _run_git(
        root,
        [
            "merge-tree",
            "--write-tree",
            "--messages",
            "--name-only",
            target_commit,
            source_commit,
        ],
        check=False,
    )
    ours_changed = set(
        _run_git(root, ["diff", "--name-only", base_commit, target_commit])
        .stdout.strip()
        .splitlines()
    )
    theirs_changed = set(
        _run_git(root, ["diff", "--name-only", base_commit, source_commit])
        .stdout.strip()
        .splitlines()
    )
    base_objects = _research_objects_at_commit(root, base_commit)
    target_objects = _research_objects_at_commit(root, target_commit)
    source_objects = _research_objects_at_commit(root, source_commit)
    conflicts = _semantic_merge_conflicts(
        base_objects,
        target_objects,
        source_objects,
    )
    if _branch(root) in {"main", "stable"}:
        conflicts.extend(_stable_evolution_conflicts(base_objects, source_objects))
    if backend.returncode:
        paths = sorted(ours_changed & theirs_changed)
        conflicts.append(
            {
                "type": "working_state",
                "paths": paths,
                "message": "backend merge has unresolved file conflicts",
            }
        )
    conflicts = _decorate_merge_conflicts(conflicts)
    return {
        "source": normalized,
        "target": _branch(root),
        "source_commit": source_commit,
        "target_commit": target_commit,
        "base_commit": base_commit,
        "clean": not conflicts,
        "conflicts": conflicts,
        "changed_paths": {
            "ours": sorted(ours_changed),
            "theirs": sorted(theirs_changed),
        },
    }


def merge_research_branch(
    repo: str | Path,
    source: str,
    *,
    subject: str | None = None,
    summary: str = "",
    actor: str | None = None,
    preserve_conflicts: bool = False,
) -> ResearchMergeResult:
    """Merge a clean line, or explicitly retain opposed evidence under a hold gate."""

    root = _repository_root(repo)
    with _repository_lock(root):
        _require_clean_research_switch(root)
        preview = preview_research_merge(root, source)
        preservable = all(
            item.get("type") == "opposed_evidence" for item in preview["conflicts"]
        )
        if not preview["clean"] and not (preserve_conflicts and preservable):
            conflict_types = ", ".join(
                sorted({str(item["type"]) for item in preview["conflicts"]})
            )
            raise ResearchGitError(
                f"research merge requires explicit conflict resolution: {conflict_types}"
            )
        merge = _run_git(
            root,
            ["merge", "--no-ff", "--no-commit", "--no-edit", preview["source_commit"]],
            check=False,
        )
        if merge.returncode:
            _run_git(root, ["merge", "--abort"], check=False)
            raise ResearchGitError(
                "research merge backend failed after clean preflight"
            )
        resolution_objects: list[str] = []
        resolution_paths: list[Path] = []
        try:
            if preview["conflicts"]:
                for conflict in preview["conflicts"]:
                    evaluated = list(
                        dict.fromkeys(
                            [
                                str(conflict["target"]),
                                *conflict.get("supporting_evidence", []),
                                *conflict.get("refuting_evidence", []),
                            ]
                        )
                    )
                    resolution = _record_research_object_locked(
                        root,
                        kind="gate_decision",
                        payload={
                            "decision": "hold",
                            "claim_promotion_allowed": False,
                            "required_failures": [
                                "opposed evidence requires independent reconciliation"
                            ],
                            "merge_conflict_id": conflict["conflict_id"],
                            "resolution": "preserve_as_contested",
                        },
                        state="rejected",
                        relations=[
                            {"type": "evaluates", "target": object_id}
                            for object_id in evaluated
                        ],
                        actor={
                            "actor_id": "research-conflict-gate",
                            "authority": "deterministic_gate",
                        },
                    )
                    resolution_objects.append(resolution.object_id)
                    if resolution.created:
                        resolution_paths.append(resolution.path)
            checkpoint = _create_checkpoint_locked(
                root,
                stage="merge",
                subject=subject or f"merge research line {preview['source']}",
                summary=summary,
                actor=actor,
                additional_parent_refs=(preview["source_commit"],),
                allow_backend_stage=True,
                allow_checkpoint_only=True,
            )
        except BaseException:
            _run_git(root, ["merge", "--abort"], check=False)
            for path in resolution_paths:
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
            raise
        if not checkpoint.committed or not checkpoint.commit:
            _run_git(root, ["merge", "--abort"], check=False)
            raise ResearchGitError("research merge did not create a durable checkpoint")
        return ResearchMergeResult(
            source=str(preview["source"]),
            target=str(preview["target"]),
            source_commit=str(preview["source_commit"]),
            commit=checkpoint.commit,
            checkpoint_id=str(checkpoint.checkpoint_id),
            content_hash=str(checkpoint.content_hash),
            resolution_objects=tuple(resolution_objects),
        )


def _bundle_advertised_refs(repo: Path, bundle: Path) -> list[dict[str, str]]:
    listed = _run_git(
        repo,
        ["bundle", "list-heads", str(bundle)],
        check=False,
    )
    if listed.returncode:
        detail = (listed.stderr or listed.stdout).strip()
        raise ResearchGitError(f"cannot list Git bundle refs: {detail}")
    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in listed.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ResearchGitError("cannot parse Git bundle advertised refs")
        object_id, name = fields
        if name in seen:
            raise ResearchGitError(f"Git bundle advertises a duplicate ref: {name}")
        seen.add(name)
        refs.append({"name": name, "object": object_id})
    return sorted(refs, key=lambda item: (item["name"], item["object"]))


def _bundle_pointer_blob(repo: Path, object_id: str) -> bytes:
    size_result = _run_git(
        repo,
        ["cat-file", "-s", object_id],
        check=False,
    )
    try:
        size = int(size_result.stdout.strip())
    except (TypeError, ValueError) as exc:
        raise ResearchGitError(f"cannot inspect pointer blob: {object_id}") from exc
    if size_result.returncode or size < 0 or size > _MAX_BUNDLE_POINTER_BYTES:
        raise ResearchGitError(
            f"reachable research object pointer exceeds safety limit: {object_id}"
        )
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "blob", object_id],
            capture_output=True,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (FileNotFoundError, OSError) as exc:
        raise ResearchGitError(f"cannot read pointer blob: {object_id}") from exc
    if completed.returncode or len(completed.stdout) != size:
        raise ResearchGitError(f"cannot read pointer blob: {object_id}")
    return completed.stdout


def _bundle_reachable_closure(
    repo: Path,
    advertised_refs: Sequence[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    commit_tips: set[str] = set()
    for item in advertised_refs:
        object_id = str(item.get("object") or "")
        resolved = _run_git(
            repo,
            ["rev-parse", "--verify", f"{object_id}^{{commit}}"],
            check=False,
        )
        if resolved.returncode == 0 and resolved.stdout.strip():
            commit_tips.add(resolved.stdout.strip())
    if not commit_tips:
        raise ResearchGitError("Git bundle advertises no commit-reachable refs")

    commits = sorted(
        set(_run_git(repo, ["rev-list", *sorted(commit_tips)]).stdout.splitlines())
    )
    reachable_objects = _run_git(
        repo,
        [
            "rev-list",
            "--objects",
            *sorted(commit_tips),
            "--",
            "research-objects/",
        ],
    ).stdout.splitlines()
    pointer_records: dict[tuple[str, str], dict[str, Any]] = {}
    pointer_members: dict[str, bytes] = {}
    for line in reachable_objects:
        object_id, separator, path = line.partition(" ")
        if (
            not separator
            or not path.startswith("research-objects/")
            or not path.endswith(".json")
        ):
            continue
        object_type = _run_git(
            repo,
            ["cat-file", "-t", object_id],
            check=False,
        )
        if object_type.returncode or object_type.stdout.strip() != "blob":
            continue
        raw = _bundle_pointer_blob(repo, object_id)
        try:
            pointer = _validate_pointer_payload(
                json.loads(raw.decode("utf-8")),
                pointer_path=path,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ResearchGitError) as exc:
            raise ResearchGitError(
                f"cannot validate reachable research object pointer: {object_id}:{path}"
            ) from exc
        pointer_digest = hashlib.sha256(raw).hexdigest()
        member = f"pointer-closure/sha256-{pointer_digest}.json"
        pointer_members.setdefault(member, raw)
        pointer_records[(path, object_id)] = {
            "path": path,
            "git_blob": object_id,
            "member": member,
            "blob_hash": f"sha256:{pointer_digest}",
            "pointer_hash": str(pointer["pointer_hash"]),
            "object_hash": str(pointer["object_hash"]),
            "size": int(pointer["size"]),
        }

    pointers = [pointer_records[key] for key in sorted(pointer_records)]
    required_objects = sorted({str(item["object_hash"]) for item in pointers})
    closure = {
        "version": 1,
        "scope": "all-advertised-refs",
        "refs": [dict(item) for item in advertised_refs],
        "reachable_commits": {
            "count": len(commits),
            "content_hash": content_hash(commits),
        },
        "pointers": pointers,
        "required_objects": required_objects,
    }
    return closure, pointer_members


def _create_research_bundle_locked(
    root: Path,
    destination: str | Path,
    *,
    profile: str = "reproduce",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    if profile not in {"index", "reproduce", "audit"}:
        raise ResearchGitError("bundle profile must be index, reproduce, or audit")
    store_root = _configured_store_root(root)
    worktree = _worktree_change_summary(root)
    if not worktree["clean"]:
        raise ResearchGitError(
            "bundle refused because the research repository has uncommitted "
            "tracked or research-eligible changes, a backend stage, or selected "
            "changes; generated views excluded by policy do not block bundling"
        )
    head = _head(root)
    if head is None:
        raise ResearchGitError("bundle refused because the repository has no commit")
    _checkpoint_at_commit(root, head)
    branch = _branch(root)
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ResearchGitError(f"bundle destination already exists: {dest}")

    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="xscientist_research_bundle_") as td:
        temp = Path(td)
        git_bundle = temp / "repository.gitbundle"
        _run_git(root, ["bundle", "create", str(git_bundle), "--all"])
        advertised_refs = _bundle_advertised_refs(root, git_bundle)
        if head not in {item["object"] for item in advertised_refs}:
            raise ResearchGitError(
                "bundle refused because repository refs changed during capture"
            )
        closure, pointer_payloads = _bundle_reachable_closure(root, advertised_refs)
        entries: list[dict[str, Any]] = [
            {
                "path": "repository.gitbundle",
                "hash": _hash_file(git_bundle),
                "size": git_bundle.stat().st_size,
            }
        ]
        object_files: list[tuple[Path, str]] = []
        pointer_files: list[tuple[Path, str]] = []
        for pointer_arcname, raw in sorted(pointer_payloads.items()):
            pointer_source = temp / pointer_arcname
            pointer_source.parent.mkdir(parents=True, exist_ok=True)
            pointer_source.write_bytes(raw)
            pointer_files.append((pointer_source, pointer_arcname))
        pointer_sizes: dict[str, set[int]] = {}
        for pointer in closure["pointers"]:
            pointer_sizes.setdefault(str(pointer["object_hash"]), set()).add(
                int(pointer["size"])
            )
        for object_hash in closure["required_objects"]:
            if profile == "index":
                continue
            expected_sizes = pointer_sizes.get(object_hash) or set()
            if len(expected_sizes) != 1:
                raise ResearchGitError(
                    "historical pointers disagree about CAS object size: "
                    f"{object_hash}"
                )
            digest = object_hash.split(":", 1)[1]
            store_path = store_root / "objects" / "sha256" / digest
            if (
                not store_path.is_file()
                or store_path.stat().st_size != next(iter(expected_sizes))
                or _hash_file(store_path) != object_hash
            ):
                missing.append(object_hash)
                continue
            object_arcname = f"objects/sha256/{digest}"
            object_files.append((store_path, object_arcname))
        missing.sort()
        complete = not missing
        if missing and not allow_incomplete:
            raise ResearchGitError(
                "bundle is missing required CAS objects: " + ", ".join(missing)
            )
        for source, arcname in [*pointer_files, *object_files]:
            entries.append(
                {
                    "path": arcname,
                    "hash": _hash_file(source),
                    "size": source.stat().st_size,
                }
            )
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "protocol_kind": "research_bundle",
            "created_at": _now_iso(),
            "profile": profile,
            "repository_head": head,
            "repository_branch": branch,
            "closure": closure,
            "complete": complete,
            "missing_objects": missing,
            "entries": entries,
        }
        manifest["content_hash"] = content_hash(manifest)
        try:
            validate_json(manifest, load_schema("research_bundle"))
        except ValidationError as exc:
            raise ResearchGitError(
                f"generated research bundle is invalid: {exc.message}"
            ) from exc
        manifest_path = temp / "bundle.manifest.json"
        _atomic_write_json(manifest_path, manifest)
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(git_bundle, arcname="repository.gitbundle", recursive=False)
            archive.add(manifest_path, arcname="bundle.manifest.json", recursive=False)
            for source, arcname in [*pointer_files, *object_files]:
                archive.add(source, arcname=arcname, recursive=False)
    return {**manifest, "destination": str(dest)}


def create_research_bundle(
    repo: str | Path,
    destination: str | Path,
    *,
    profile: str = "reproduce",
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    root = _repository_root(repo)
    with _repository_lock(root):
        return _create_research_bundle_locked(
            root,
            destination,
            profile=profile,
            allow_incomplete=allow_incomplete,
        )


def _bundle_manifest_hash_valid(payload: dict[str, Any]) -> bool:
    expected = str(payload.get("content_hash") or "")
    scrubbed = {key: value for key, value in payload.items() if key != "content_hash"}
    return bool(expected and content_hash(scrubbed) == expected)


def _hash_stream(stream: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _safe_bundle_member_name(name: str) -> str:
    if "\\" in name:
        raise ResearchGitError(f"unsafe bundle member path: {name}")
    return _normalise_relative(name)


def verify_research_bundle(bundle: str | Path) -> dict[str, Any]:
    """Verify an offline research bundle without extracting its payload."""

    bundle_path = Path(bundle).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked = {"entries": 0, "bytes": 0, "pointers": 0, "objects": 0}
    manifest: dict[str, Any] | None = None
    if not bundle_path.is_file():
        raise ResearchGitError(f"research bundle does not exist: {bundle_path}")

    try:
        with tarfile.open(bundle_path, "r:*") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                errors.append("bundle contains duplicate member names")
            for member in members:
                try:
                    _safe_bundle_member_name(member.name)
                except ResearchGitError as exc:
                    errors.append(str(exc))
                if not member.isfile():
                    errors.append(f"bundle member is not a regular file: {member.name}")
            manifest_members = [
                member for member in members if member.name == "bundle.manifest.json"
            ]
            if len(manifest_members) != 1:
                errors.append("bundle must contain exactly one bundle.manifest.json")
            elif manifest_members[0].size > 8 * 1024 * 1024:
                errors.append("bundle manifest exceeds the 8 MiB safety limit")
            else:
                stream = archive.extractfile(manifest_members[0])
                if stream is None:
                    errors.append("cannot read bundle.manifest.json")
                else:
                    try:
                        manifest = json.loads(stream.read().decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        errors.append(f"invalid bundle manifest JSON: {exc}")
            if manifest is not None:
                try:
                    validate_json(manifest, load_schema("research_bundle"))
                except ValidationError as exc:
                    errors.append(f"invalid bundle manifest: {exc.message}")
                if not _bundle_manifest_hash_valid(manifest):
                    errors.append("bundle manifest content hash verification failed")

                entries = manifest.get("entries") or []
                entry_by_path: dict[str, dict[str, Any]] = {}
                for entry in entries:
                    try:
                        entry_path = _safe_bundle_member_name(
                            str(entry.get("path") or "")
                        )
                    except (AttributeError, ResearchGitError) as exc:
                        errors.append(f"invalid bundle entry path: {exc}")
                        continue
                    if entry_path in entry_by_path:
                        errors.append(f"duplicate bundle manifest entry: {entry_path}")
                    entry_by_path[entry_path] = entry
                archive_paths = set(names) - {"bundle.manifest.json"}
                expected_paths = set(entry_by_path)
                for unexpected in sorted(archive_paths - expected_paths):
                    errors.append(f"unlisted bundle member: {unexpected}")
                for absent in sorted(expected_paths - archive_paths):
                    errors.append(f"bundle manifest entry is absent: {absent}")

                pointer_payloads: dict[str, dict[str, Any]] = {}
                closure_pointer_payloads: dict[str, bytes] = {}
                object_entries: dict[str, dict[str, Any]] = {}
                computed_closure: dict[str, Any] | None = None
                computed_pointer_payloads: dict[str, bytes] = {}
                with tempfile.TemporaryDirectory(
                    prefix="xscientist_bundle_verify_"
                ) as td:
                    verification_root = Path(td)
                    repository_bundle = verification_root / "repository.gitbundle"
                    for entry_path, entry in sorted(entry_by_path.items()):
                        matches = [
                            member for member in members if member.name == entry_path
                        ]
                        if len(matches) != 1 or not matches[0].isfile():
                            continue
                        stream = archive.extractfile(matches[0])
                        if stream is None:
                            errors.append(f"cannot read bundle member: {entry_path}")
                            continue
                        actual_hash, actual_size = _hash_stream(stream)
                        checked["entries"] += 1
                        checked["bytes"] += actual_size
                        if actual_hash != entry.get("hash"):
                            errors.append(
                                f"bundle entry hash mismatch for {entry_path}: "
                                f"expected {entry.get('hash')}, got {actual_hash}"
                            )
                        if actual_size != entry.get("size"):
                            errors.append(
                                f"bundle entry size mismatch for {entry_path}: "
                                f"expected {entry.get('size')}, got {actual_size}"
                            )
                        if entry_path == "repository.gitbundle":
                            source = archive.extractfile(matches[0])
                            if source is not None:
                                with repository_bundle.open("wb") as target:
                                    shutil.copyfileobj(source, target, 1024 * 1024)
                        elif entry_path.startswith("research-objects/"):
                            checked["pointers"] += 1
                            source = archive.extractfile(matches[0])
                            if source is None:
                                continue
                            try:
                                pointer = json.loads(source.read().decode("utf-8"))
                                pointer = _validate_pointer_payload(
                                    pointer,
                                    pointer_path=entry_path,
                                )
                                object_hash = str(pointer["object_hash"])
                                if object_hash in pointer_payloads:
                                    errors.append(
                                        "duplicate bundle pointer for object: "
                                        f"{object_hash}"
                                    )
                                pointer_payloads[object_hash] = pointer
                            except (
                                UnicodeDecodeError,
                                json.JSONDecodeError,
                                ResearchGitError,
                            ) as exc:
                                errors.append(str(exc))
                        elif entry_path.startswith("pointer-closure/"):
                            checked["pointers"] += 1
                            source = archive.extractfile(matches[0])
                            if source is not None:
                                closure_pointer_payloads[entry_path] = source.read()
                        elif entry_path.startswith("objects/sha256/"):
                            checked["objects"] += 1
                            digest = PurePosixPath(entry_path).name
                            declared_hash = f"sha256:{digest}"
                            if actual_hash != declared_hash:
                                errors.append(
                                    f"CAS member path/hash mismatch for {entry_path}"
                                )
                            object_entries[declared_hash] = entry

                    if not repository_bundle.is_file():
                        errors.append("bundle lacks repository.gitbundle")
                    else:
                        _run_git(verification_root, ["init", "-q"])
                        verified = _run_git(
                            verification_root,
                            ["bundle", "verify", str(repository_bundle)],
                            check=False,
                        )
                        if verified.returncode:
                            detail = (verified.stderr or verified.stdout).strip()
                            errors.append(f"invalid Git bundle: {detail}")
                        advertised_refs: list[dict[str, str]] = []
                        try:
                            advertised_refs = _bundle_advertised_refs(
                                verification_root,
                                repository_bundle,
                            )
                        except ResearchGitError as exc:
                            errors.append(str(exc))
                        advertised = {item["object"] for item in advertised_refs}
                        repository_head = str(manifest.get("repository_head") or "")
                        if repository_head not in advertised:
                            errors.append(
                                "bundle repository_head is not advertised by the Git bundle"
                            )
                        declared_closure = manifest.get("closure")
                        if declared_closure is not None:
                            if not isinstance(declared_closure, dict):
                                errors.append("bundle closure declaration is invalid")
                            elif verified.returncode == 0 and advertised_refs:
                                imported = _run_git(
                                    verification_root,
                                    [
                                        "bundle",
                                        "unbundle",
                                        str(repository_bundle),
                                    ],
                                    check=False,
                                )
                                if imported.returncode:
                                    detail = (
                                        imported.stderr or imported.stdout
                                    ).strip()
                                    errors.append(
                                        f"cannot independently inspect Git bundle: {detail}"
                                    )
                                else:
                                    try:
                                        (
                                            computed_closure,
                                            computed_pointer_payloads,
                                        ) = _bundle_reachable_closure(
                                            verification_root,
                                            advertised_refs,
                                        )
                                    except ResearchGitError as exc:
                                        errors.append(str(exc))
                                    if (
                                        computed_closure is not None
                                        and declared_closure != computed_closure
                                    ):
                                        errors.append(
                                            "bundle closure declaration does not match "
                                            "the Git bundle reachable history"
                                        )
                                    expected_pointer_members = set(
                                        computed_pointer_payloads
                                    )
                                    actual_pointer_members = set(
                                        closure_pointer_payloads
                                    )
                                    for absent in sorted(
                                        expected_pointer_members
                                        - actual_pointer_members
                                    ):
                                        errors.append(
                                            "bundle pointer closure member is absent: "
                                            f"{absent}"
                                        )
                                    for unexpected in sorted(
                                        actual_pointer_members
                                        - expected_pointer_members
                                    ):
                                        errors.append(
                                            "unexpected bundle pointer closure member: "
                                            f"{unexpected}"
                                        )
                                    for member_name in sorted(
                                        expected_pointer_members
                                        & actual_pointer_members
                                    ):
                                        if (
                                            closure_pointer_payloads[member_name]
                                            != computed_pointer_payloads[member_name]
                                        ):
                                            errors.append(
                                                "bundle pointer closure differs from Git "
                                                f"history: {member_name}"
                                            )

                profile = str(manifest.get("profile") or "")
                declared_closure = manifest.get("closure")
                if isinstance(declared_closure, dict):
                    closure_for_completeness = computed_closure or declared_closure
                    required_objects = {
                        str(item)
                        for item in closure_for_completeness.get("required_objects")
                        or []
                    }
                    pointer_sizes: dict[str, set[int]] = {}
                    for pointer in closure_for_completeness.get("pointers") or []:
                        if not isinstance(pointer, dict):
                            continue
                        object_hash = str(pointer.get("object_hash") or "")
                        try:
                            size = int(pointer.get("size"))
                        except (TypeError, ValueError):
                            continue
                        pointer_sizes.setdefault(object_hash, set()).add(size)
                    unexpected_objects = sorted(set(object_entries) - required_objects)
                    for object_hash in unexpected_objects:
                        errors.append(
                            f"bundle contains CAS outside its Git closure: {object_hash}"
                        )
                else:
                    required_objects = set(pointer_payloads)
                    pointer_sizes = {
                        object_hash: {int(pointer.get("size") or 0)}
                        for object_hash, pointer in pointer_payloads.items()
                    }
                if profile == "index":
                    if object_entries:
                        errors.append("index bundle unexpectedly contains CAS payloads")
                    computed_missing: list[str] = []
                else:
                    computed_missing = sorted(required_objects - set(object_entries))
                    for object_hash, expected_sizes in pointer_sizes.items():
                        if len(expected_sizes) != 1:
                            errors.append(
                                "historical pointers disagree about CAS object size: "
                                f"{object_hash}"
                            )
                            continue
                        entry = object_entries.get(object_hash)
                        if (
                            entry is not None
                            and entry.get("size") not in expected_sizes
                        ):
                            errors.append(
                                f"CAS member size disagrees with pointer: {object_hash}"
                            )
                declared_missing = sorted(
                    str(item) for item in manifest.get("missing_objects") or []
                )
                if declared_missing != computed_missing:
                    errors.append(
                        "bundle completeness mismatch: declared missing objects do not "
                        "match archive contents"
                    )
                if bool(manifest.get("complete")) != (not computed_missing):
                    errors.append("bundle completeness verdict is inconsistent")
    except (OSError, tarfile.TarError) as exc:
        errors.append(f"cannot read research bundle: {exc}")

    return {
        "schema_version": "xscientist.research-bundle-verification.v1",
        "bundle": str(bundle_path),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked": checked,
        "manifest": manifest,
    }


def _copy_verified_bundle_member(
    archive: tarfile.TarFile,
    member_name: str,
    destination: Path,
    *,
    expected_hash: str,
) -> None:
    matches = [member for member in archive.getmembers() if member.name == member_name]
    if len(matches) != 1 or not matches[0].isfile():
        raise ResearchGitError(f"bundle member is unavailable: {member_name}")
    source = archive.extractfile(matches[0])
    if source is None:
        raise ResearchGitError(f"cannot read bundle member: {member_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temp = Path(raw_temp)
    digest = hashlib.sha256()
    try:
        with os.fdopen(handle, "wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
            target.flush()
            os.fsync(target.fileno())
        actual_hash = f"sha256:{digest.hexdigest()}"
        if actual_hash != expected_hash:
            raise ResearchGitError(
                f"bundle changed during restore for {member_name}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        temp.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temp.unlink(missing_ok=True)


def restore_research_bundle(
    bundle: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Restore a verified Git and CAS research closure into a new directory."""

    verification = verify_research_bundle(bundle)
    if not verification["ok"]:
        raise ResearchGitError(
            "research bundle verification failed: " + "; ".join(verification["errors"])
        )
    manifest = verification.get("manifest") or {}
    bundle_path = Path(bundle).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise ResearchGitError(
            f"bundle restore destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    entry_by_path = {
        str(entry["path"]): entry for entry in manifest.get("entries") or []
    }

    with tempfile.TemporaryDirectory(
        prefix=".xscientist_restore_",
        dir=destination_path.parent,
    ) as td:
        staging = Path(td)
        git_bundle = staging / "repository.gitbundle"
        worktree = staging / "repository"
        with tarfile.open(bundle_path, "r:*") as archive:
            repository_entry = entry_by_path["repository.gitbundle"]
            _copy_verified_bundle_member(
                archive,
                "repository.gitbundle",
                git_bundle,
                expected_hash=str(repository_entry["hash"]),
            )
            _run_git(staging, ["clone", "-q", str(git_bundle), str(worktree)])
            head = str(manifest["repository_head"])
            branch = str(manifest.get("repository_branch") or "detached")
            if branch == "detached":
                _run_git(worktree, ["checkout", "-q", "--detach", head])
            else:
                _run_git(worktree, ["checkout", "-q", "-B", branch, head])
            _run_git(worktree, ["remote", "remove", "origin"], check=False)
            closure = manifest.get("closure")
            if isinstance(closure, dict):
                for item in closure.get("refs") or []:
                    if not isinstance(item, dict):
                        continue
                    ref_name = str(item.get("name") or "")
                    object_id = str(item.get("object") or "")
                    if not ref_name.startswith("refs/"):
                        continue
                    _run_git(
                        worktree,
                        ["update-ref", ref_name, object_id],
                    )
            if _head(worktree) != head:
                raise ResearchGitError(
                    f"restored Git HEAD does not match bundle manifest: {head}"
                )

            store_root = _configured_store_root(worktree)
            for entry_path, entry in sorted(entry_by_path.items()):
                if entry_path.startswith("objects/sha256/"):
                    digest = PurePosixPath(entry_path).name
                    target = store_root / "objects" / "sha256" / digest
                    _copy_verified_bundle_member(
                        archive,
                        entry_path,
                        target,
                        expected_hash=str(entry["hash"]),
                    )
                elif entry_path.startswith("research-objects/"):
                    tracked_pointer = worktree / _normalise_relative(entry_path)
                    if not tracked_pointer.is_file() or _hash_file(
                        tracked_pointer
                    ) != entry.get("hash"):
                        raise ResearchGitError(
                            f"bundle pointer differs from restored Git tree: {entry_path}"
                        )

        verify_objects = str(manifest.get("profile")) != "index"
        fsck = verify_research_repository(
            worktree,
            commit=head,
            verify_objects=verify_objects,
        )
        if not fsck["ok"]:
            raise ResearchGitError(
                "restored research repository failed fsck: " + "; ".join(fsck["errors"])
            )
        worktree.replace(destination_path)
        _fsync_directory(destination_path.parent)

    fsck["repository"] = str(destination_path)

    return {
        "schema_version": "xscientist.research-bundle-restore.v1",
        "bundle": str(bundle_path),
        "repository": str(destination_path),
        "commit": str(manifest["repository_head"]),
        "branch": str(manifest.get("repository_branch") or "detached"),
        "profile": str(manifest.get("profile") or ""),
        "objects_restored": verification["checked"]["objects"],
        "fsck": fsck,
    }


def _safe_hydration_target(worktree: Path, logical_path: str) -> Path:
    if str(logical_path).strip() in {"", "."}:
        return worktree.resolve()
    relative = PurePosixPath(_validate_research_logical_path(logical_path))
    cursor = worktree
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ResearchGitError(
                "reproduction object path contains a symlink: " + relative.as_posix()
            )
    target = cursor.resolve()
    try:
        target.relative_to(worktree.resolve())
    except ValueError as exc:
        raise ResearchGitError(
            f"object logical path escapes worktree: {logical_path}"
        ) from exc
    return target


def _safe_worktree_control_path(worktree: Path, logical_path: str) -> Path:
    """Resolve a private reproduction path without following symlink parents."""

    relative = PurePosixPath(_normalise_relative(logical_path))
    cursor = worktree
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ResearchGitError(
                "reproduction control path contains a symlink: " + relative.as_posix()
            )
    return _safe_hydration_target(worktree, relative.as_posix())


def _sanitized_reproduction_environment(worktree: Path) -> dict[str, str]:
    """Return a minimal host environment without inherited credentials.

    Reproduction commands are repository-controlled input.  Passing the full
    parent environment would expose provider keys, cloud credentials, agent
    sockets, and user configuration to that code. The replacement home prevents
    default home-directory lookup only; it does not block absolute host paths
    and is not a filesystem sandbox.
    """

    environment = {
        key: value
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "VIRTUAL_ENV",
        )
        if (value := os.environ.get(key))
    }
    private_home = _safe_worktree_control_path(
        worktree, ".xscientist/reproduction-home"
    )
    cache_home = _safe_worktree_control_path(worktree, ".xscientist/reproduction-cache")
    private_home.mkdir(parents=True, exist_ok=True)
    cache_home.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HOME": str(private_home),
            "USERPROFILE": str(private_home),
            "XDG_CONFIG_HOME": str(private_home / "config"),
            "XDG_CACHE_HOME": str(cache_home),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _reproduction_execution_isolation(
    platform_name: str | None = None,
) -> dict[str, Any]:
    """Describe actual execution controls without claiming a sandbox.

    A POSIX process group lets the timeout path signal descendants that remain
    in that group, but a child may create another session. On Windows the
    bounded-process helper terminates only the parent process. Neither mode
    restricts filesystem or network access, so this receipt must not be read as
    an operating-system security boundary.
    """

    selected_platform = platform_name or os.name
    posix_process_group = selected_platform == "posix"
    return {
        "isolated": False,
        "security_boundary": False,
        "environment": "sanitized",
        "environment_scope": "variables_only",
        "process_tree": (
            "best_effort_process_group"
            if posix_process_group
            else "parent_only_no_tree_guarantee"
        ),
        "process_control": (
            "posix_process_group_best_effort"
            if posix_process_group
            else "parent_process_only"
        ),
        "process_tree_termination_guaranteed": False,
        "filesystem": "host_visible",
        "network": "host_unrestricted",
    }


def reproduce_checkpoint(
    repo: str | Path,
    *,
    commit: str = "HEAD",
    destination: str | Path | None = None,
    execute: bool = False,
    timeout_seconds: int = 600,
    environment_policy: str = "warn",
) -> dict[str, Any]:
    if environment_policy not in {"ignore", "warn", "strict"}:
        raise ResearchGitError("environment policy must be ignore, warn, or strict")
    root = _repository_root(repo)
    store_root = _configured_store_root(root)
    resolved = _run_git(
        root, ["rev-parse", "--verify", f"{commit}^{{commit}}"]
    ).stdout.strip()
    checkpoint_path, checkpoint = _checkpoint_at_commit(root, resolved)
    checkpoint = _validate_checkpoint_payload(
        checkpoint,
        checkpoint_path=f"{resolved}:{checkpoint_path}",
    )
    pointers = _pointer_records_at_commit(root, resolved)
    required = list(checkpoint.get("object_refs") or [])
    missing: list[str] = []
    damaged: list[str] = []
    object_errors: dict[str, str] = {}
    verified_sources: dict[str, Path] = {}
    for object_hash in required:
        object_pointers = pointers.get(object_hash)
        if not object_pointers:
            missing.append(object_hash)
            object_errors[object_hash] = "missing object pointer at selected commit"
            continue
        pointer_errors: list[str] = []
        object_missing = False
        verified_path: Path | None = None
        for pointer in object_pointers:
            store_path, object_error = _verify_pointer_object(
                root,
                object_hash,
                pointer,
                store_root=store_root,
            )
            if object_error:
                pointer_errors.append(object_error)
                object_missing = object_missing or (
                    store_path is None or not store_path.is_file()
                )
            elif store_path is not None:
                verified_path = store_path
        if pointer_errors:
            (missing if object_missing else damaged).append(object_hash)
            object_errors[object_hash] = "; ".join(dict.fromkeys(pointer_errors))
        elif verified_path is not None:
            verified_sources[object_hash] = verified_path
    command = str((checkpoint.get("reproduce") or {}).get("command") or "").strip()
    environment_receipt = (checkpoint.get("reproduce") or {}).get("environment")
    environment = _compare_runtime_environment(environment_receipt)
    result: dict[str, Any] = {
        "commit": resolved,
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "command": command,
        "objects_complete": not missing and not damaged,
        "missing_objects": missing,
        "damaged_objects": damaged,
        "object_errors": object_errors,
        "environment_policy": environment_policy,
        "environment": environment,
        "worktree": None,
        "executed": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "execution_isolation": _reproduction_execution_isolation(),
    }

    def finalize() -> dict[str, Any]:
        mismatch_fields = sorted(
            {
                str(item.get("field") or "unknown")
                for item in result["environment"].get("mismatches") or []
            }
        )
        if result.get("executed"):
            reproduction_level = "computational_rerun"
            verdict = (
                "passed"
                if result.get("returncode") == 0 and not result.get("timed_out")
                else "failed"
            )
        elif result.get("worktree"):
            reproduction_level = "artifact_replay"
            verdict = "materialized"
        else:
            reproduction_level = "inspection"
            if not result["objects_complete"]:
                verdict = "failed"
            elif result["command"]:
                verdict = "ready"
            else:
                verdict = "warning"
                result["limitation"] = (
                    "No reproduction command was recorded; this inspection only "
                    "checks the checkpoint and bound objects."
                )
                result["next_action"] = (
                    "Create a checkpoint with a shell-free reproduction command, "
                    "then run `xscientist research reproduce HEAD --dest PATH --execute`."
                )
        if mismatch_fields and verdict in {"ready", "materialized", "passed"}:
            verdict = "warning"
        receipt_base: dict[str, Any] = {
            "schema_version": REPRODUCTION_RECEIPT_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "commit": result["commit"],
            "checkpoint_id": str(result["checkpoint"].get("checkpoint_id") or ""),
            "checkpoint_hash": str(result["checkpoint"].get("content_hash") or ""),
            "reproduction_level": reproduction_level,
            "verdict": verdict,
            "objects_complete": bool(result["objects_complete"]),
            "environment": {
                "policy": result["environment_policy"],
                "recorded": bool(result["environment"].get("recorded")),
                "matches": result["environment"].get("matches"),
                "recorded_content_hash": result["environment"].get(
                    "recorded_content_hash"
                ),
                "mismatch_fields": mismatch_fields,
            },
            "command_hash": (
                content_hash(result["command"]) if result["command"] else None
            ),
            "executed": bool(result.get("executed")),
            "returncode": result.get("returncode"),
            "timed_out": bool(result.get("timed_out", False)),
            "stdout_hash": (
                content_hash(result.get("stdout") or "")
                if result.get("executed")
                else None
            ),
            "stderr_hash": (
                content_hash(result.get("stderr") or "")
                if result.get("executed")
                else None
            ),
            "stdout_truncated": (
                bool(result.get("stdout_truncated"))
                if result.get("executed")
                else False
            ),
            "stderr_truncated": (
                bool(result.get("stderr_truncated"))
                if result.get("executed")
                else False
            ),
            "output_capture": "bounded_tail",
            "max_output_chars": _MAX_REPRODUCTION_OUTPUT_CHARS,
            "execution_isolation": dict(result["execution_isolation"]),
        }
        checkpoint_binding_core = {
            "commit": result["commit"],
            "checkpoint_id": str(result["checkpoint"].get("checkpoint_id") or ""),
            "checkpoint_content_hash": str(
                result["checkpoint"].get("content_hash") or ""
            ),
        }
        target_binding_core = {
            "policy": "xscientist.reproduction-target-binding.v2",
            "audit_commit": result["commit"],
            "audit_checkpoint_id": str(result["checkpoint"].get("checkpoint_id") or ""),
            "audit_checkpoint_hash": str(
                result["checkpoint"].get("content_hash") or ""
            ),
            # Generic inspection/materialization receipts have no scientific
            # target yet. ResearchLifecycle replaces this empty binding with the
            # exact immutable target and claim-closure snapshot before a receipt
            # may be recorded as verified.
            "target_objects": [],
            "claim_closures": [],
        }
        execution_result_core = {
            key: receipt_base[key]
            for key in (
                "command_hash",
                "reproduction_level",
                "verdict",
                "objects_complete",
                "executed",
                "returncode",
                "timed_out",
                "stdout_hash",
                "stderr_hash",
                "stdout_truncated",
                "stderr_truncated",
                "output_capture",
                "max_output_chars",
            )
        }
        receipt_base["checkpoint_binding"] = {
            **checkpoint_binding_core,
            "binding_hash": content_hash(checkpoint_binding_core),
        }
        receipt_base["target_binding"] = {
            **target_binding_core,
            "binding_hash": content_hash(target_binding_core),
        }
        receipt_base["execution_result"] = {
            **execution_result_core,
            "result_hash": content_hash(execution_result_core),
        }
        receipt_hash = content_hash(receipt_base)
        receipt = {
            **receipt_base,
            "receipt_id": f"rr-{receipt_hash.split(':', 1)[1][:16]}",
            "content_hash": receipt_hash,
        }
        try:
            validate_json(receipt, load_schema("reproduction_receipt"))
        except ValidationError as exc:  # pragma: no cover - implementation contract
            raise ResearchGitError(
                f"generated reproduction receipt is invalid: {exc.message}"
            ) from exc
        receipt_path: str | None = None
        if result.get("worktree"):
            worktree_path = Path(str(result["worktree"]))
            target = _safe_worktree_control_path(
                worktree_path,
                f".xscientist/reproductions/{receipt['receipt_id']}.json",
            )
            _atomic_write_json(target, receipt)
            receipt_path = target.relative_to(worktree_path).as_posix()
        result["receipt"] = receipt
        result["receipt_path"] = receipt_path
        return result

    if environment_policy == "strict" and environment["mismatches"]:
        raise ResearchGitError(
            "reproduction environment mismatch: "
            + ", ".join(item["field"] for item in environment["mismatches"])
        )
    if destination is None:
        if execute:
            raise ResearchGitError(
                "--execute requires an explicit reproduction destination"
            )
        return finalize()
    if timeout_seconds < 1:
        raise ResearchGitError("reproduction timeout must be at least one second")
    if missing or damaged:
        detail = [*missing, *damaged]
        raise ResearchGitError(
            "cannot materialize reproduction; missing or damaged objects: "
            + ", ".join(detail)
        )
    worktree = Path(destination).expanduser().resolve()
    if worktree.exists():
        raise ResearchGitError(f"reproduction destination already exists: {worktree}")
    argv: list[str] = []
    if execute:
        if not command:
            raise ResearchGitError("checkpoint does not declare a reproduction command")
        argv = shlex.split(command)
        if not argv:
            raise ResearchGitError("checkpoint reproduction command is empty")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    worktree_attempted = False
    materialized = False
    try:
        worktree_attempted = True
        _run_git(root, ["worktree", "add", "--detach", str(worktree), resolved])
        for object_hash in required:
            source = verified_sources[object_hash]
            hydrated_logical_paths: set[str] = set()
            for pointer in pointers[object_hash]:
                logical_path = str(pointer["logical_path"])
                if logical_path in hydrated_logical_paths:
                    continue
                hydrated_logical_paths.add(logical_path)
                target = _safe_hydration_target(worktree, logical_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.is_file() and _hash_file(target) == object_hash:
                        continue
                    raise ResearchGitError(
                        f"refusing to overwrite reproduction target: {target}"
                    )
                shutil.copy2(source, target)
                if _hash_file(target) != object_hash:
                    target.unlink(missing_ok=True)
                    raise ResearchGitError(
                        f"reproduction object changed during hydration: {object_hash}"
                    )
        dependency_mismatches = _compare_dependency_locks(worktree, environment_receipt)
        environment["mismatches"].extend(dependency_mismatches)
        environment["matches"] = (
            environment["recorded"] and not environment["mismatches"]
        )
        if environment_policy == "strict" and dependency_mismatches:
            raise ResearchGitError(
                "reproduction dependency lock mismatch: "
                + ", ".join(item["field"] for item in dependency_mismatches)
            )
        result["worktree"] = str(worktree)
        if execute:
            working_directory = str(
                (checkpoint.get("reproduce") or {}).get("working_directory") or "."
            )
            cwd = _safe_hydration_target(worktree, working_directory)
            safe_environment = _sanitized_reproduction_environment(worktree)
            try:
                completed = run_process_bounded(
                    argv,
                    cwd=cwd,
                    env=safe_environment,
                    timeout=timeout_seconds,
                    max_output_chars=_MAX_REPRODUCTION_OUTPUT_CHARS,
                )
            except subprocess.TimeoutExpired as exc:
                result["executed"] = True
                result["returncode"] = 124
                result["timed_out"] = True
                result["stdout"] = str(exc.stdout or "")[
                    -_MAX_REPRODUCTION_OUTPUT_CHARS:
                ]
                result["stderr"] = str(exc.stderr or "")[
                    -_MAX_REPRODUCTION_OUTPUT_CHARS:
                ]
                result["stdout_truncated"] = (
                    len(str(exc.stdout or "")) >= _MAX_REPRODUCTION_OUTPUT_CHARS
                )
                result["stderr_truncated"] = (
                    len(str(exc.stderr or "")) >= _MAX_REPRODUCTION_OUTPUT_CHARS
                )
            else:
                result["executed"] = True
                result["timed_out"] = False
                result["returncode"] = completed.returncode
                result["stdout"] = completed.stdout
                result["stderr"] = completed.stderr
                result["stdout_truncated"] = completed.stdout_truncated
                result["stderr_truncated"] = completed.stderr_truncated
        finalized = finalize()
        materialized = True
        return finalized
    finally:
        if worktree_attempted and not materialized:
            _run_git(
                root,
                ["worktree", "remove", "--force", str(worktree)],
                check=False,
            )
            _run_git(root, ["worktree", "prune"], check=False)


__all__ = [
    "BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "REPRODUCTION_RECEIPT_SCHEMA",
    "OBJECT_POINTER_SCHEMA",
    "REPOSITORY_SCHEMA",
    "CheckpointResult",
    "capture_environment_receipt",
    "ObjectPointerResult",
    "ResearchObjectResult",
    "ResearchMergeResult",
    "ResearchStageResult",
    "ResearchGitError",
    "add_research_object",
    "auto_checkpoint",
    "commit_research_stage",
    "create_checkpoint",
    "create_research_branch",
    "create_research_bundle",
    "create_research_tag",
    "delete_research_branch",
    "init_repository",
    "load_repository_config",
    "load_research_object",
    "list_research_objects",
    "list_research_objects_at_ref",
    "list_research_branches",
    "list_research_tags",
    "merge_research_branch",
    "record_research_object",
    "rename_research_branch",
    "repository_status",
    "reproduce_checkpoint",
    "restore_research_bundle",
    "restore_research_paths",
    "resolve_research_object_id",
    "revert_research_checkpoint",
    "research_diff",
    "research_blame",
    "research_object_origin_checkpoint",
    "research_log",
    "research_trajectory",
    "research_stage",
    "research_unstage",
    "preview_research_merge",
    "show_checkpoint",
    "switch_research_branch",
    "verify_research_bundle",
    "verify_research_repository",
    "validate_complete_research_trajectory",
]
