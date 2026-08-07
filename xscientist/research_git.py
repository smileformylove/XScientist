"""Local-first Git history for scientific projects.

Git is used as the small, reviewable control plane. Large immutable payloads
stay in the local ARA content-addressed store and enter commits as pointer
records. Nothing in this module creates a remote or pushes data.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import mimetypes
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

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
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.privacy import format_privacy_findings, scan_paths

REPOSITORY_SCHEMA = "xscientist.research-repository.v1"
CHECKPOINT_SCHEMA = "xscientist.research-checkpoint.v1"
OBJECT_POINTER_SCHEMA = "xscientist.research-object-pointer.v1"
BUNDLE_SCHEMA = "xscientist.research-bundle.v1"
ENVIRONMENT_SCHEMA = "xscientist.research-environment.v1"

DEFAULT_TRACK_PATTERNS = (
    ".gitignore",
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
    "ara/**/exploration_graph.json",
    "ara/**/claims/*.json",
    "ara/**/events/*.jsonl",
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
    "02_experiments/**/pipeline/*.json",
    "02_experiments/**/claim_evidence_graph.json",
    "02_experiments/**/experiment_registry.jsonl",
    "02_experiments/**/research_plan.json",
    "02_experiments/**/*.tex",
    "02_experiments/**/*.bib",
    "03_papers/*.tex",
    "03_papers/*.bib",
    "03_papers/*.md",
    "04_logs/project_summary.json",
    "04_logs/project_summary.md",
    "04_logs/submission_shortlist.md",
    "pipeline_manifest.json",
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
    "**/*.pem",
    "**/*.key",
    "**/id_rsa",
    "**/id_ed25519",
    "**/credentials.json",
    "**/secrets.json",
    "**/*token.json",
)

MILESTONE_STAGES = frozenset(
    {"init", "preregister", "experiment", "evidence", "review", "paper", "release"}
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


class ResearchGitError(RuntimeError):
    """A safe local-repository operation could not be completed."""


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
    with lock_path.open("a+b") as stream:
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
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise ResearchGitError(
                        f"timed out waiting for research repository lock: {lock_path}"
                    )
                time.sleep(0.05)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
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
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise ResearchGitError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _normalise_relative(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ResearchGitError(f"unsafe repository-relative path: {path}")
    return candidate.as_posix()


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


def _copy_into_store(source: Path, store_root: Path) -> tuple[str, int, Path]:
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
        with source.open("rb") as source_stream, os.fdopen(handle, "wb") as target:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target.flush()
            os.fsync(target.fileno())
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
    if require_config and not (root / "research.yaml").is_file():
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
    return f"""schema_version: {REPOSITORY_SCHEMA}
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
    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "research.yaml").exists():
        raise ResearchGitError(f"research repository is already initialized: {root}")

    init = _run_git(root, ["init", "-b", "main"], check=False)
    if init.returncode:
        _run_git(root, ["init"])
        _run_git(root, ["symbolic-ref", "HEAD", "refs/heads/main"])
    _ensure_git_identity(root, git_user_name, git_user_email)

    project_name = (name or root.name).strip() or "research-project"
    _atomic_write_text(root / ".gitignore", _RESEARCH_GITIGNORE)
    _atomic_write_text(
        root / "research.yaml",
        _config_text(
            name=project_name,
            policy=policy,
            actor=actor,
            max_file_bytes=max_file_bytes,
        ),
    )
    question_text = (
        question or "# Research question\n\nDescribe the research question.\n"
    ).rstrip()
    _atomic_write_text(root / "question.md", question_text + "\n")
    _atomic_write_text(root / ".xscientist" / "README.md", _REPOSITORY_NOTE)
    for directory in (
        "hypotheses",
        "claims",
        "manuscript",
        "checkpoints",
        "research-objects",
        "ara",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)

    if not commit:
        return CheckpointResult(
            created=False,
            committed=False,
            reason="repository initialized without a commit",
        )
    return create_checkpoint(
        root,
        stage="init",
        subject=f"initialize {project_name}",
        summary="Initialize the local scientific repository and research question.",
        status="completed",
        actor=actor,
        allow_checkpoint_only=True,
    )


def _git_paths(repo: Path, args: Sequence[str]) -> set[str]:
    completed = _run_git(repo, list(args))
    return {
        item for item in completed.stdout.split("\0") if item and not item.endswith("/")
    }


def _changed_paths(repo: Path) -> tuple[set[str], set[str]]:
    staged = _git_paths(repo, ["diff", "--cached", "--name-only", "-z"])
    unstaged = _git_paths(repo, ["diff", "--name-only", "-z"])
    untracked = _git_paths(repo, ["ls-files", "--others", "--exclude-standard", "-z"])
    return staged, unstaged | untracked


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
        if _matches(relative, denied_patterns):
            excluded.append(f"{relative} (denied pattern)")
            continue
        if not (
            _matches(relative, allowed_patterns)
            or _path_is_explicit(relative, explicit)
        ):
            excluded.append(f"{relative} (not tracked by policy)")
            continue
        target = repo / relative
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
) -> tuple[str, dict[str, Any]] | None:
    tree = _run_git(repo, ["ls-tree", "-r", "--name-only", commit]).stdout.splitlines()
    for checkpoint_path in sorted(
        path
        for path in tree
        if path.startswith("checkpoints/") and path.endswith(".json")
    ):
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
) -> list[tuple[str, str, dict[str, Any]]]:
    head = _head(repo)
    if head is None:
        return []
    checkpoint_at_head = _checkpoint_id_from_commit(repo, head)
    if checkpoint_at_head:
        resolved = _checkpoint_by_id_at_commit(repo, head, checkpoint_at_head)
        if resolved is not None:
            path, payload = resolved
            return [(head, path, payload)]
    ancestry = _run_git(repo, ["rev-list", "--parents", "-n", "1", head]).stdout.split()
    starts = ancestry[1:] if len(ancestry) > 2 else [head]
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
    return records


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
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        refs.append({"path": relative, "manifest_hash": hash_manifest(manifest)})
    return refs


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
    _normalise_relative(str(payload.get("logical_path") or ""))
    store_relpath = _normalise_relative(str(payload.get("store_relpath") or ""))
    digest = str(payload["object_hash"]).split(":", 1)[1]
    if PurePosixPath(store_relpath).parts[-3:] != ("objects", "sha256", digest):
        raise ResearchGitError(
            f"research object pointer has inconsistent CAS path: {pointer_path}"
        )
    pointer_name = PurePosixPath(pointer_path).name
    if pointer_name.startswith("sha256-") and pointer_name != f"sha256-{digest}.json":
        raise ResearchGitError(
            f"research object pointer filename does not match its hash: {pointer_path}"
        )
    return payload


def _pointer_records(
    repo: Path,
    *,
    strict: bool = False,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    pointer_root = repo / "research-objects"
    if not pointer_root.exists():
        return records
    for path in sorted(pointer_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload = _validate_pointer_payload(
                payload,
                pointer_path=path.relative_to(repo).as_posix(),
            )
        except (OSError, json.JSONDecodeError, ResearchGitError) as exc:
            if strict:
                raise ResearchGitError(
                    f"cannot validate research object pointer: "
                    f"{path.relative_to(repo).as_posix()}: {exc}"
                ) from exc
            continue
        object_hash = str(payload.get("object_hash") or "")
        if object_hash.startswith("sha256:"):
            records[object_hash] = {
                **payload,
                "pointer_path": path.relative_to(repo).as_posix(),
            }
    return records


def _pointer_records_at_commit(
    repo: Path,
    commit: str,
    *,
    strict: bool = True,
) -> dict[str, dict[str, Any]]:
    tree = _run_git(repo, ["ls-tree", "-r", "--name-only", commit]).stdout.splitlines()
    records: dict[str, dict[str, Any]] = {}
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
            records[object_hash] = {**payload, "pointer_path": pointer_path}
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
    lines.extend(f"- `{path}`" for path in payload.get("changed_paths") or [])
    if not payload.get("changed_paths"):
        lines.append("- Semantic checkpoint only; no additional tracked file changed.")
    if payload.get("ara_manifests"):
        lines.extend(["", "## ARA manifests", ""])
        lines.extend(
            f"- `{item['path']}` — `{item['manifest_hash']}`"
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
    commit: bool = True,
    allow_checkpoint_only: bool = False,
) -> CheckpointResult:
    config = load_repository_config(root)
    if not stage or not stage.replace("-", "_").replace("_", "a").isalnum():
        raise ResearchGitError(
            "stage must contain only letters, numbers, hyphens, or underscores"
        )
    if reproduce_command and any(char in reproduce_command for char in "\r\n"):
        raise ResearchGitError("reproduction command must be a single line")
    staged_before, changed = _changed_paths(root)
    if staged_before:
        raise ResearchGitError(
            "checkpoint refused because the Git index already contains staged changes: "
            + ", ".join(sorted(staged_before))
        )
    selected, excluded = _select_paths(root, config, changed, explicit=include)
    material = [path for path in selected if not path.startswith("checkpoints/")]
    if not material and not allow_checkpoint_only:
        return CheckpointResult(
            created=False,
            committed=False,
            excluded_paths=tuple(excluded),
            reason="no material research change matched the checkpoint policy",
        )

    parent_records = _checkpoint_parent_records(root)
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
    manifests = _manifest_refs(root, [*selected, *explicit_ara])

    pointers = _pointer_records(root, strict=True)
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
        "actor": actor or str(config.get("actor") or "xscientist"),
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
        _run_git(root, ["add", "--", *stage_paths])
        if _run_git(root, ["diff", "--cached", "--quiet"], check=False).returncode == 0:
            raise ResearchGitError("checkpoint produced no staged Git change")
        trailers = [
            f"Research-Checkpoint: {checkpoint_id}",
            f"Research-Stage: {stage}",
            f"Research-State: {status}",
            f"Research-Event: {checkpoint_hash}",
        ]
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
            commit=commit,
            allow_checkpoint_only=allow_checkpoint_only,
        )


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


def repository_status(repo: str | Path) -> dict[str, Any]:
    root = _repository_root(repo)
    config = load_repository_config(root)
    staged, changed = _changed_paths(root)
    selected, excluded = _select_paths(root, config, changed)
    previous = _previous_checkpoint(root)
    store = _resolve_configured_path(
        root,
        (config.get("storage") or {}).get("root") or ".ara-store",
        label="CAS root",
    )
    object_files = (
        [path for path in store.rglob("*") if path.is_file()] if store.exists() else []
    )
    return {
        "repository": str(root),
        "name": config.get("name"),
        "branch": _branch(root),
        "head": _head(root),
        "checkpoint_policy": config["git"].get("checkpoint_policy"),
        "auto_commit": bool(config["git"].get("auto_commit", True)),
        "auto_push": False,
        "staged_paths": sorted(staged),
        "eligible_changes": selected,
        "excluded_changes": excluded,
        "last_checkpoint": previous,
        "object_store": {
            "path": str(store),
            "objects": len(object_files),
            "bytes": sum(path.stat().st_size for path in object_files),
        },
    }


def _add_research_object_locked(
    root: Path,
    source: str | Path,
    *,
    logical_path: str | None = None,
    media_type: str | None = None,
) -> ObjectPointerResult:
    config = load_repository_config(root)
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise ResearchGitError(f"research object is not a regular file: {source_path}")
    if logical_path:
        logical = _normalise_relative(logical_path)
    else:
        try:
            logical = source_path.relative_to(root).as_posix()
        except ValueError:
            logical = f"external/{source_path.name}"
    if _matches(logical, SECRET_DENY_PATTERNS):
        raise ResearchGitError(
            f"refusing to register a denied secret/binary logical path: {logical}"
        )
    store_root = _configured_store_root(root)
    object_hash, object_size, store_path = _copy_into_store(source_path, store_root)
    digest = object_hash.split(":", 1)[1]
    pointer_dir = _resolve_configured_path(
        root,
        (config.get("storage") or {}).get("pointer_directory") or "research-objects",
        label="pointer directory",
    )
    pointer_path = pointer_dir / f"sha256-{digest}.json"
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
    pointer_payload["pointer_hash"] = content_hash(pointer_payload)
    try:
        validate_json(pointer_payload, load_schema("research_object_pointer"))
    except ValidationError as exc:
        raise ResearchGitError(
            f"generated object pointer is invalid: {exc.message}"
        ) from exc
    _atomic_write_json(pointer_path, pointer_payload)
    return ObjectPointerResult(
        pointer_path=pointer_path,
        store_path=store_path,
        object_hash=object_hash,
        size=object_size,
        linked=False,
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


def _checkpoint_at_commit(repo: Path, commit: str) -> tuple[str, dict[str, Any]]:
    record = _latest_checkpoint_record(repo, commit)
    if record is None:
        raise ResearchGitError(f"commit {commit!r} contains no research checkpoint")
    _checkpoint_commit, checkpoint_path, payload = record
    return checkpoint_path, payload


def show_checkpoint(repo: str | Path, commit: str = "HEAD") -> dict[str, Any]:
    root = _repository_root(repo)
    path, payload = _checkpoint_at_commit(root, commit)
    return {
        "commit": _run_git(root, ["rev-parse", commit]).stdout.strip(),
        "path": path,
        "checkpoint_hash_valid": _checkpoint_hash_valid(payload),
        "checkpoint": payload,
    }


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
    pointer_records: dict[str, dict[str, Any]] = {}
    checked = {
        "checkpoints": 0,
        "pointers": 0,
        "objects": 0,
        "ara_manifests": 0,
    }

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
            if object_hash in pointer_records:
                errors.append(f"duplicate pointer for object {object_hash}")
            pointer_records[object_hash] = {**payload, "pointer_path": pointer_path}
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
        for object_hash in latest.get("object_refs") or []:
            pointer = pointer_records.get(str(object_hash))
            if pointer is None:
                errors.append(f"checkpoint references missing pointer: {object_hash}")
                continue
            if verify_objects:
                checked["objects"] += 1
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


def research_log(repo: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    root = _repository_root(repo)
    if limit < 1:
        raise ResearchGitError("log limit must be at least 1")
    separator = "%x1f"
    record = "%x1e"
    fmt = (
        f"%H{separator}%h{separator}%aI{separator}%an{separator}%s{separator}%B{record}"
    )
    raw = _run_git(root, ["log", f"--max-count={limit}", f"--format={fmt}"]).stdout
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
                trailers.setdefault(key, []).append(value)
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
        str(item.get("path")): str(item.get("manifest_hash"))
        for item in before
        if isinstance(item, dict)
    }
    after_map = {
        str(item.get("path")): str(item.get("manifest_hash"))
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
                "before": before_map[path],
                "after": after_map[path],
            }
            for path in sorted(common)
            if before_map[path] != after_map[path]
        ],
        "unchanged": sorted(
            path for path in common if before_map[path] == after_map[path]
        ),
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
    dirty = _run_git(root, ["status", "--porcelain"]).stdout.strip()
    if dirty:
        raise ResearchGitError(
            "bundle refused because the research repository has uncommitted changes"
        )
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise ResearchGitError(f"bundle destination already exists: {dest}")

    pointers = _pointer_records(root)
    missing: list[str] = []
    with tempfile.TemporaryDirectory(prefix="xscientist_research_bundle_") as td:
        temp = Path(td)
        git_bundle = temp / "repository.gitbundle"
        _run_git(root, ["bundle", "create", str(git_bundle), "--all"])
        entries: list[dict[str, Any]] = [
            {
                "path": "repository.gitbundle",
                "hash": _hash_file(git_bundle),
                "size": git_bundle.stat().st_size,
            }
        ]
        object_files: list[tuple[Path, str]] = []
        pointer_files: list[tuple[Path, str]] = []
        for object_hash, pointer in sorted(pointers.items()):
            pointer_source = root / str(pointer["pointer_path"])
            pointer_arcname = str(pointer["pointer_path"])
            pointer_files.append((pointer_source, pointer_arcname))
            if profile == "index":
                continue
            store_path, object_error = _verify_pointer_object(
                root,
                object_hash,
                pointer,
                store_root=store_root,
            )
            if object_error or store_path is None:
                missing.append(object_hash)
                continue
            object_arcname = f"objects/sha256/{object_hash.split(':', 1)[1]}"
            object_files.append((store_path, object_arcname))
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
            "repository_head": _head(root),
            "repository_branch": _branch(root),
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
                object_entries: dict[str, dict[str, Any]] = {}
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
                        heads = _run_git(
                            verification_root,
                            ["bundle", "list-heads", str(repository_bundle)],
                            check=False,
                        )
                        advertised = {
                            line.split()[0]
                            for line in heads.stdout.splitlines()
                            if line.split()
                        }
                        repository_head = str(manifest.get("repository_head") or "")
                        if heads.returncode or repository_head not in advertised:
                            errors.append(
                                "bundle repository_head is not advertised by the Git bundle"
                            )

                profile = str(manifest.get("profile") or "")
                if profile == "index":
                    if object_entries:
                        errors.append("index bundle unexpectedly contains CAS payloads")
                    computed_missing: list[str] = []
                else:
                    computed_missing = sorted(
                        set(pointer_payloads) - set(object_entries)
                    )
                    for object_hash, pointer in pointer_payloads.items():
                        entry = object_entries.get(object_hash)
                        if entry is not None and entry.get("size") != pointer.get(
                            "size"
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
    relative = _normalise_relative(logical_path)
    target = (worktree / relative).resolve()
    try:
        target.relative_to(worktree.resolve())
    except ValueError as exc:
        raise ResearchGitError(
            f"object logical path escapes worktree: {logical_path}"
        ) from exc
    return target


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
    resolved = _run_git(root, ["rev-parse", commit]).stdout.strip()
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
        pointer = pointers.get(object_hash)
        if pointer is None:
            missing.append(object_hash)
            object_errors[object_hash] = "missing object pointer at selected commit"
            continue
        store_path, object_error = _verify_pointer_object(
            root,
            object_hash,
            pointer,
            store_root=store_root,
        )
        if object_error:
            if store_path is None or not store_path.is_file():
                missing.append(object_hash)
            else:
                damaged.append(object_hash)
            object_errors[object_hash] = object_error
            continue
        if store_path is not None:
            verified_sources[object_hash] = store_path
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
    }
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
        return result
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
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _run_git(root, ["worktree", "add", "--detach", str(worktree), resolved])
    for object_hash in required:
        pointer = pointers[object_hash]
        source = verified_sources[object_hash]
        target = _safe_hydration_target(worktree, str(pointer["logical_path"]))
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
    environment["matches"] = environment["recorded"] and not environment["mismatches"]
    if environment_policy == "strict" and dependency_mismatches:
        _run_git(
            root,
            ["worktree", "remove", "--force", str(worktree)],
            check=False,
        )
        raise ResearchGitError(
            "reproduction dependency lock mismatch: "
            + ", ".join(item["field"] for item in dependency_mismatches)
        )
    result["worktree"] = str(worktree)
    if execute:
        if not command:
            raise ResearchGitError("checkpoint does not declare a reproduction command")
        argv = shlex.split(command)
        if not argv:
            raise ResearchGitError("checkpoint reproduction command is empty")
        working_directory = str(
            (checkpoint.get("reproduce") or {}).get("working_directory") or "."
        )
        cwd = _safe_hydration_target(worktree, working_directory)
        try:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            result["executed"] = True
            result["returncode"] = 124
            result["timed_out"] = True
            result["stdout"] = str(exc.stdout or "")[-20000:]
            result["stderr"] = str(exc.stderr or "")[-20000:]
            return result
        result["executed"] = True
        result["timed_out"] = False
        result["returncode"] = completed.returncode
        result["stdout"] = (completed.stdout or "")[-20000:]
        result["stderr"] = (completed.stderr or "")[-20000:]
    return result


__all__ = [
    "BUNDLE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "OBJECT_POINTER_SCHEMA",
    "REPOSITORY_SCHEMA",
    "CheckpointResult",
    "ObjectPointerResult",
    "ResearchGitError",
    "add_research_object",
    "auto_checkpoint",
    "create_checkpoint",
    "create_research_bundle",
    "init_repository",
    "load_repository_config",
    "repository_status",
    "reproduce_checkpoint",
    "restore_research_bundle",
    "research_diff",
    "research_log",
    "show_checkpoint",
    "verify_research_bundle",
    "verify_research_repository",
]
