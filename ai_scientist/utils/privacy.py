"""Fail-closed privacy helpers for logs, traces, and repository publishing.

The module deliberately reports *where* a risky value was found without ever
returning the matched value.  It is dependency-free so both the installed CLI
and repository engineering checks can use the same rules.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REDACTED_API_KEY = "[REDACTED_API_KEY]"
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PATH = "[REDACTED_PATH]"
REDACTED_PRIVATE = "[REDACTED_PRIVATE]"

_MAX_SCAN_BYTES = 4_000_000
_SKIP_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)

# High-confidence credential signatures.  Keep these conservative: a false
# positive blocks a research checkpoint or release.
_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "openai_api_key",
        re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}"),
    ),
    ("aws_access_key", re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    ("github_token", re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{30,}")),
    (
        "google_api_key",
        re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{35}(?![A-Za-z0-9_-])"),
    ),
    ("slack_token", re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("stripe_live_key", re.compile(r"(?<![A-Za-z0-9])sk_live_[A-Za-z0-9]{20,}")),
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
)

_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|password|secret|session[_-]?token|access[_-]?token)\b"
    r"\s*[:=]\s*([^\s\"']+)"
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z][A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?KEY|CLIENT[_-]?SECRET|"
    r"PASSWORD|CREDENTIAL|TOKEN)[A-Z0-9_]*)\b\s*[\"']?\s*[:=]\s*[\"']?"
    r"([^\s\"',}]+)"
)
_LOOSE_API_KEY_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_FILE_URI_RE = re.compile(r"(?i)\bfile:///(?:[^\s\"'<>]+)")
_WINDOWS_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[A-Z]:\\|\\\\[^\\\s]+\\)[^\r\n\"'<>]+"
)
_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_:/.])/(?:[^/\s\"'<>]+/)+[^/\s\"'<>]*")
_USER_PATH_RE = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9_])"
    r"(?:/(?:Users|home)/([^/\s\"'<>]+)(?:/[^\s\"'<>]+)+"
    r"|/root(?:/[^\s\"'<>]+)+"
    r"|[A-Z]:\\Users\\([^\\\s\"'<>]+)(?:\\[^\s\"'<>]+)+)"
)
_PLACEHOLDER_USERS = frozenset(
    {"example", "name", "runner", "scientist", "user", "username", "your-name"}
)
_PLACEHOLDER_SECRET_MARKERS = (
    "changeme",
    "dummy",
    "example",
    "fake",
    "keep-this",
    "must-not",
    "process-only",
    "redacted",
    "replace",
    "secret-value",
    "test",
    "top-secret",
    "your",
    "xxx",
)


def _is_sensitive_field_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if normalized in {
        "api_key",
        "access_key",
        "access_token",
        "auth_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "session_token",
        "token",
    }:
        return True
    return normalized.endswith(
        (
            "_api_key",
            "_access_key",
            "_client_secret",
            "_credential",
            "_credentials",
            "_password",
            "_secret",
            "_session_token",
        )
    )


@dataclass(frozen=True)
class PrivacyFinding:
    """A location-only finding.  Matched content is intentionally absent."""

    scope: str
    rule: str
    path: str
    object_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _runtime_private_literals() -> tuple[str, ...]:
    values = {str(Path.home()), socket.gethostname()}
    generic = {"localhost", "localhost.localdomain"}
    return tuple(
        sorted(
            value
            for value in values
            if len(value) >= 5 and value.lower() not in generic
        )
    )


def redact_sensitive_text(text: str) -> str:
    """Remove credentials, personal identifiers, and host-local paths."""

    output = str(text)
    for literal in _runtime_private_literals():
        output = output.replace(literal, REDACTED_PRIVATE)
    output = _LOOSE_API_KEY_RE.sub(REDACTED_API_KEY, output)
    for _name, pattern in _SECRET_RULES:
        output = pattern.sub(REDACTED_API_KEY, output)
    output = _BEARER_RE.sub("Bearer [REDACTED]", output)
    output = _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", output)
    output = _CREDENTIAL_ASSIGNMENT_RE.sub(
        lambda match: (
            f"{match.group(1)}=[REDACTED]"
            if _is_sensitive_field_name(match.group(1))
            else match.group(0)
        ),
        output,
    )
    output = _EMAIL_RE.sub(REDACTED_EMAIL, output)
    output = _FILE_URI_RE.sub(REDACTED_PATH, output)
    output = _WINDOWS_PATH_RE.sub(REDACTED_PATH, output)
    output = _POSIX_PATH_RE.sub(REDACTED_PATH, output)
    return output


def redact_sensitive_payload(payload: Any) -> Any:
    """Recursively redact a JSON-like value before it reaches persistent storage."""

    if isinstance(payload, str):
        return redact_sensitive_text(payload)
    if isinstance(payload, Path):
        return redact_sensitive_text(str(payload))
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            safe_key = redact_sensitive_text(str(key))
            secret_field = _is_sensitive_field_name(str(key))
            if secret_field and isinstance(value, str) and value:
                result[safe_key] = "[REDACTED]"
            else:
                result[safe_key] = redact_sensitive_payload(value)
        return result
    if isinstance(payload, tuple):
        return tuple(redact_sensitive_payload(value) for value in payload)
    if isinstance(payload, list):
        return [redact_sensitive_payload(value) for value in payload]
    return payload


def portable_path(path: str | os.PathLike[str], *, base: str | os.PathLike[str]) -> str:
    """Return a relative path inside ``base`` and hide every external path."""

    resolved = Path(path).expanduser().resolve()
    anchor = Path(base).expanduser().resolve()
    try:
        relative = resolved.relative_to(anchor)
    except ValueError:
        return REDACTED_PATH
    rendered = relative.as_posix()
    return rendered if rendered else "."


def relative_path_reference(
    path: str | os.PathLike[str], *, base: str | os.PathLike[str]
) -> str:
    """Return a portable relative reference, including safe ``..`` segments."""

    resolved = Path(path).expanduser().resolve()
    anchor = Path(base).expanduser().resolve()
    try:
        rendered = os.path.relpath(resolved, anchor)
    except ValueError:
        return REDACTED_PATH
    return Path(rendered).as_posix()


def resolve_portable_path(
    value: str | os.PathLike[str] | None,
    *,
    base: str | os.PathLike[str],
) -> Path | None:
    """Resolve new relative references while remaining compatible with legacy ARAs."""

    if value is None or str(value).strip() in {"", REDACTED_PATH, REDACTED_PRIVATE}:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path(base).expanduser().resolve() / candidate
    return candidate.resolve()


def relativize_path_fields(
    payload: Any,
    *,
    base: str | os.PathLike[str],
    field_names: Iterable[str] = (
        "ara_root",
        "config_path",
        "cwd",
        "parent_ara_root",
        "project_dir",
        "project_root",
        "source_ara",
        "source_exp_dir",
        "source_fork_dir",
        "workspace_dir",
    ),
) -> Any:
    """Convert known absolute path fields to portable relative references."""

    selected = frozenset(field_names)

    def transform(value: Any) -> Any:
        if isinstance(value, Mapping):
            result: dict[Any, Any] = {}
            for key, nested in value.items():
                if (
                    str(key) in selected
                    and isinstance(nested, (str, os.PathLike))
                    and Path(nested).expanduser().is_absolute()
                ):
                    result[key] = relative_path_reference(nested, base=base)
                else:
                    result[key] = transform(nested)
            return result
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, tuple):
            return tuple(transform(item) for item in value)
        return value

    return transform(payload)


def _scan_text(text: str) -> set[str]:
    rules = {name for name, pattern in _SECRET_RULES if pattern.search(text)}
    if _FILE_URI_RE.search(text):
        rules.add("file_uri")
    for match in _USER_PATH_RE.finditer(text):
        user = (match.group(1) or match.group(2) or "root").lower()
        if user not in _PLACEHOLDER_USERS:
            rules.add("local_user_path")
            break
    for literal in _runtime_private_literals():
        if literal in text:
            rules.add("local_machine_identifier")
    for match in _CREDENTIAL_ASSIGNMENT_RE.finditer(text):
        if not _is_sensitive_field_name(match.group(1)):
            continue
        value = match.group(2).strip().lower()
        if len(value) < 20 or any(
            marker in value for marker in _PLACEHOLDER_SECRET_MARKERS
        ):
            continue
        character_classes = sum(
            bool(pattern.search(value))
            for pattern in (
                re.compile(r"[a-z]"),
                re.compile(r"[0-9]"),
                re.compile(r"[^a-z0-9]"),
            )
        )
        if character_classes >= 2:
            rules.add("credential_assignment")
            break
    return rules


def scan_file(
    path: Path, *, root: Path, scope: str = "working_tree"
) -> list[PrivacyFinding]:
    """Scan one small text file without retaining or returning matched content."""

    try:
        data = path.read_bytes()
    except OSError:
        return []
    if len(data) > _MAX_SCAN_BYTES or b"\0" in data[:8192]:
        return []
    text = data.decode("utf-8", errors="ignore")
    try:
        label = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        label = REDACTED_PATH
    rules = _scan_text(text)
    try:
        workspace_literals = {
            str(root.expanduser().absolute()),
            str(root.expanduser().resolve()),
        }
    except OSError:
        workspace_literals = set()
    for literal in tuple(workspace_literals):
        if literal.startswith("/private/"):
            workspace_literals.add(literal.removeprefix("/private"))
        elif literal.startswith(("/tmp/", "/var/")):
            workspace_literals.add("/private" + literal)
    if any(
        literal and literal != "/" and literal in text for literal in workspace_literals
    ):
        rules.add("workspace_absolute_path")
    semantic_path = label in {"question.md", "topic.md"} or label.startswith(
        (
            ".xscientist/objects/",
            "claims/",
            "hypotheses/",
            "research-objects/",
        )
    )
    if semantic_path and _POSIX_PATH_RE.search(text):
        rules.add("absolute_path")
    return [
        PrivacyFinding(scope=scope, rule=rule, path=label) for rule in sorted(rules)
    ]


def scan_paths(root: str | Path, paths: Iterable[str | Path]) -> list[PrivacyFinding]:
    anchor = Path(root).expanduser().resolve()
    findings: list[PrivacyFinding] = []
    for raw_path in sorted({str(path) for path in paths}):
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = anchor / candidate
        if candidate.is_file():
            findings.extend(scan_file(candidate, root=anchor))
    return findings


def _tracked_paths(root: Path, *, include_untracked: bool) -> list[Path] | None:
    if not (root / ".git").exists():
        return None
    command = ["git", "ls-files", "-z"]
    if include_untracked:
        command[2:2] = ["--cached", "--others", "--exclude-standard"]
    completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
    if completed.returncode:
        return None
    return [
        root / item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    ]


def _filesystem_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in _SKIP_DIRECTORY_NAMES for part in path.relative_to(root).parts
        )
    )


def _history_findings(root: Path) -> list[PrivacyFinding]:
    if not (root / ".git").exists():
        return []
    listed = subprocess.run(
        ["git", "rev-list", "--objects", "--all"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if listed.returncode:
        return []
    labels: dict[str, str] = {}
    for raw_line in listed.stdout.splitlines():
        raw_oid, _separator, raw_path = raw_line.partition(b" ")
        oid = raw_oid.decode("ascii", errors="ignore")
        labels.setdefault(
            oid,
            raw_path.decode("utf-8", errors="surrogateescape") or "<historical-blob>",
        )
    checks = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=root,
        input=("\n".join(labels) + "\n").encode(),
        capture_output=True,
        check=False,
    )
    blob_oids: list[str] = []
    for raw_line in checks.stdout.splitlines():
        parts = raw_line.decode("ascii", errors="ignore").split()
        if len(parts) == 3 and parts[1] == "blob" and int(parts[2]) <= _MAX_SCAN_BYTES:
            blob_oids.append(parts[0])
    if not blob_oids:
        return []
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=("\n".join(blob_oids) + "\n").encode(),
        capture_output=True,
        check=False,
    ).stdout
    offset = 0
    findings: list[PrivacyFinding] = []
    for expected_oid in blob_oids:
        newline = batch.find(b"\n", offset)
        if newline < 0:
            break
        header = batch[offset:newline].decode("ascii", errors="ignore").split()
        offset = newline + 1
        if len(header) != 3 or header[1] != "blob":
            break
        size = int(header[2])
        data = batch[offset : offset + size]
        offset += size + 1
        if b"\0" in data[:8192]:
            continue
        for rule in sorted(_scan_text(data.decode("utf-8", errors="ignore"))):
            findings.append(
                PrivacyFinding(
                    scope="history",
                    rule=rule,
                    path=labels.get(expected_oid, "<historical-blob>"),
                    object_id=expected_oid,
                )
            )
    return findings


def scan_repository(
    root: str | Path,
    *,
    include_untracked: bool = False,
    history: bool = False,
) -> list[PrivacyFinding]:
    """Scan publishable files and optionally all reachable Git blobs."""

    anchor = Path(root).expanduser().resolve()
    candidates = _tracked_paths(anchor, include_untracked=include_untracked)
    if candidates is None:
        candidates = _filesystem_paths(anchor)
    findings: list[PrivacyFinding] = []
    for path in candidates:
        findings.extend(scan_file(path, root=anchor))
    if history:
        findings.extend(_history_findings(anchor))
    return sorted(
        set(findings),
        key=lambda item: (item.scope, item.path, item.rule, item.object_id or ""),
    )


def privacy_report(
    root: str | Path,
    *,
    include_untracked: bool = False,
    history: bool = False,
) -> dict[str, Any]:
    findings = scan_repository(
        root,
        include_untracked=include_untracked,
        history=history,
    )
    return {
        "schema": "xscientist.privacy-audit.v1",
        "ok": not findings,
        "scopes": ["working_tree", *(["history"] if history else [])],
        "matched_values_disclosed": False,
        "finding_count": len(findings),
        "findings": [finding.to_dict() for finding in findings],
    }


def format_privacy_findings(findings: Iterable[PrivacyFinding]) -> str:
    rows = [f"{item.scope}: {item.rule}: {item.path}" for item in findings]
    return "\n".join(rows)


__all__ = [
    "PrivacyFinding",
    "REDACTED_API_KEY",
    "REDACTED_EMAIL",
    "REDACTED_PATH",
    "REDACTED_PRIVATE",
    "format_privacy_findings",
    "portable_path",
    "privacy_report",
    "relative_path_reference",
    "redact_sensitive_payload",
    "redact_sensitive_text",
    "relativize_path_fields",
    "resolve_portable_path",
    "scan_file",
    "scan_paths",
    "scan_repository",
]
