"""Read-only compatibility and update checks for XScientist installations."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from ._version import __version__
from .provider_config import CONFIG_RELATIVE_PATH, CONFIG_SCHEMA_VERSION
from .research_git import REPOSITORY_SCHEMA

PYPI_JSON_URL = "https://pypi.org/pypi/xscientist/json"


def _version_key(value: str) -> tuple[int, ...]:
    """Return a conservative numeric key for the project's release versions."""

    release = str(value or "").strip().split("+", 1)[0]
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.?(dev|a|b|rc)(\d+))?",
        release,
    )
    if match is None:
        return ()
    major, minor, patch = (int(match.group(index)) for index in range(1, 4))
    qualifier = match.group(4)
    stage = {"dev": 0, "a": 1, "b": 2, "rc": 3, None: 4}[qualifier]
    serial = int(match.group(5) or 0)
    return major, minor, patch, stage, serial


def _read_mapping(
    path: Path, *, yaml_format: bool = False
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if yaml_format else json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "configuration must be an object"
    return payload, None


def _pypi_version(*, timeout: float) -> tuple[str | None, str | None]:
    request = urllib.request.Request(
        PYPI_JSON_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": f"xscientist/{__version__}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    version = (
        (payload.get("info") or {}).get("version")
        if isinstance(payload, dict)
        else None
    )
    return (str(version).strip() or None), None


def check_upgrade(
    workspace: str | Path = ".",
    *,
    online: bool = False,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """Inspect package/workspace compatibility without changing any files."""

    root = Path(workspace).expanduser().resolve()
    provider_path = root / CONFIG_RELATIVE_PATH
    repository_path = root / "research.yaml"
    provider, provider_error = _read_mapping(provider_path)
    repository, repository_error = _read_mapping(repository_path, yaml_format=True)

    provider_actual = provider.get("schema_version") if provider else None
    repository_actual = repository.get("schema_version") if repository else None
    checks = {
        "workspace": {
            "present": root.is_dir(),
            "compatible": root.is_dir(),
            "expected_schema": None,
            "actual_schema": None,
            "error": (
                None
                if root.is_dir()
                else "the requested workspace directory does not exist"
            ),
        },
        "provider_config": {
            "present": provider_path.is_file(),
            "compatible": provider_error is None
            and (provider is None or provider_actual == CONFIG_SCHEMA_VERSION),
            "expected_schema": CONFIG_SCHEMA_VERSION,
            "actual_schema": provider_actual,
            "error": provider_error,
        },
        "research_repository": {
            "present": repository_path.is_file(),
            "compatible": repository_error is None
            and (repository is None or repository_actual == REPOSITORY_SCHEMA),
            "expected_schema": REPOSITORY_SCHEMA,
            "actual_schema": repository_actual,
            "error": repository_error,
        },
    }
    remediations: list[str] = []
    if not checks["workspace"]["compatible"]:
        remediations.append("Check the workspace path and run the command again.")
    if not checks["provider_config"]["compatible"]:
        remediations.append(
            "Back up .xscientist/providers.json, then run `xscientist setup --force`."
        )
    if not checks["research_repository"]["compatible"]:
        remediations.append(
            "Back up the workspace and run the documented Research VCS migration before writing new objects."
        )

    latest = None
    online_error = None
    if online:
        latest, online_error = _pypi_version(timeout=timeout)
    update_available = bool(
        latest
        and _version_key(latest)
        and _version_key(latest) > _version_key(__version__)
    )
    installed_key = _version_key(__version__)
    latest_key = _version_key(latest or "")
    if not latest_key:
        index_relation = "unknown"
    elif installed_key > latest_key:
        index_relation = "newer_than_index"
    elif installed_key < latest_key:
        index_relation = "update_available"
    else:
        index_relation = "current"
    if update_available:
        remediations.append(
            f"Review the changelog, then run `python -m pip install --upgrade xscientist=={latest}`."
        )
    compatible = all(bool(item["compatible"]) for item in checks.values())
    return {
        "schema": "xscientist.upgrade-check.v1",
        "ok": compatible and online_error is None,
        "compatible": compatible,
        "workspace": root.name,
        "package": {
            "installed_version": __version__,
            "latest_version": latest,
            "update_available": update_available,
            "online_checked": online,
            "online_error": online_error,
            "index_relation": index_relation,
        },
        "checks": checks,
        "remediations": remediations,
        "mutated": False,
    }


__all__ = ["PYPI_JSON_URL", "check_upgrade"]
