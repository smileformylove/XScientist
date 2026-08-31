#!/usr/bin/env python3
"""Fast, dependency-light checks for repository engineering invariants."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.parse import unquote

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised in the 3.10 CI lane
    import tomli as tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai_scientist.utils.privacy import scan_repository

REQUIRED_REPOSITORY_FILES = (
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "docs/ENGINEERING.md",
    ".github/CODE_OF_CONDUCT.md",
    ".github/CONTRIBUTING.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/SECURITY.md",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/research_reproduction.yml",
)
SUPPORTED_PYTHON = ("3.10", "3.11", "3.12", "3.13")
_REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9_.-]+)")
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_PINNED_ACTION = re.compile(r"^\s*-?\s*uses:\s*[^\s@]+@([0-9a-f]{40})(?:\s+#.*)?$")


def _normalise_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(lines: Iterable[str]) -> set[str]:
    names: set[str] = set()
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", ".")):
            continue
        match = _REQUIREMENT_NAME.match(line)
        if match:
            names.add(_normalise_name(match.group(1)))
    return names


def _read_pyproject(root: Path) -> dict:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _project_version(root: Path) -> str:
    text = (root / "xscientist" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError("xscientist/_version.py has no literal __version__")
    return match.group(1)


def check_required_files(root: Path) -> list[str]:
    return [
        f"missing required repository file: {relative}"
        for relative in REQUIRED_REPOSITORY_FILES
        if not (root / relative).is_file()
    ]


def check_version_metadata(root: Path) -> list[str]:
    errors: list[str] = []
    version = _project_version(root)
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    if not re.search(rf"^## \[{re.escape(version)}\](?:\s|$)", changelog, re.MULTILINE):
        errors.append(f"CHANGELOG.md has no section for version {version}")
    citation_match = re.search(
        r'^version:\s*["\']?([^"\'\s#]+)', citation, re.MULTILINE
    )
    if not citation_match or citation_match.group(1) != version:
        errors.append(f"CITATION.cff version must equal {version}")
    return errors


def check_dependency_policy(root: Path) -> list[str]:
    errors: list[str] = []
    constraints = _requirement_names(
        (root / "requirements" / "constraints-ci.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    for relative in ("requirements.txt", "requirements/smoke.txt"):
        required = _requirement_names(
            (root / relative).read_text(encoding="utf-8").splitlines()
        )
        missing = sorted(required - constraints)
        if missing:
            errors.append(
                f"{relative} requirements missing CI constraints: {', '.join(missing)}"
            )

    project = _read_pyproject(root).get("project", {})
    project_requirements = list(project.get("dependencies", []))
    for values in (project.get("optional-dependencies", {}) or {}).values():
        project_requirements.extend(values)
    missing = sorted(_requirement_names(project_requirements) - constraints)
    if missing:
        errors.append(
            "pyproject dependencies missing CI constraints: " + ", ".join(missing)
        )
    return errors


def check_protocol_registry(root: Path) -> list[str]:
    errors: list[str] = []
    schema_dir = root / "ai_scientist" / "protocol" / "schemas"
    schema_names: set[str] = set()
    for path in sorted(schema_dir.glob("*.schema.json")):
        schema_names.add(path.name.removesuffix(".schema.json"))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid JSON Schema {path.relative_to(root)}: {exc}")
            continue
        if payload.get("type") != "object" or not payload.get("$id"):
            errors.append(f"schema lacks object type or $id: {path.relative_to(root)}")

    constants = (root / "ai_scientist" / "protocol" / "constants.py").read_text(
        encoding="utf-8"
    )
    registered = set(
        re.findall(r'^\s+[A-Z][A-Z0-9_]*\s*=\s*"([a-z0-9_]+)"', constants, re.MULTILINE)
    )
    missing = sorted(registered - schema_names)
    if missing:
        errors.append(
            "registered protocol kinds without schemas: " + ", ".join(missing)
        )

    for relative in ("README.md", "docs/README.zh.md"):
        text = (root / relative).read_text(encoding="utf-8")
        if re.search(r"\bsix JSON Schemas?\b|\b6\s*份 JSON Schema", text):
            errors.append(f"{relative} contains a hand-maintained schema count")
    return errors


def check_python_compatibility(root: Path) -> list[str]:
    errors: list[str] = []
    project = _read_pyproject(root).get("project", {})
    classifiers = "\n".join(project.get("classifiers", []))
    workflow = (root / ".github" / "workflows" / "smoke.yml").read_text(
        encoding="utf-8"
    )
    for version in SUPPORTED_PYTHON:
        if f"Python :: {version}" not in classifiers:
            errors.append(f"pyproject classifier missing Python {version}")
        if version not in workflow:
            errors.append(f"CI compatibility matrix missing Python {version}")
    for runner in ("macos-latest", "windows-latest"):
        if runner not in workflow:
            errors.append(f"CI portability matrix missing {runner}")
    return errors


def check_markdown_links(root: Path) -> list[str]:
    errors: list[str] = []
    for relative in ("README.md", "docs/README.zh.md"):
        path = root / relative
        for raw_target in _MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>")
            if target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                errors.append(f"broken local link in {relative}: {raw_target}")
    return errors


def check_action_pinning(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((root / ".github" / "workflows").glob("*.yml")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "uses:" not in line:
                continue
            if not _PINNED_ACTION.match(line):
                errors.append(
                    f"GitHub Action is not pinned to a commit: "
                    f"{path.relative_to(root)}:{line_number}"
                )
    release = (root / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    if "startsWith(github.ref, 'refs/tags/v')" not in release:
        errors.append("release publishing must be restricted to v* tag refs")
    if "fetch-depth: 0" not in release:
        errors.append("release checkout must fetch complete Git history")
    if (
        "git merge-base --is-ancestor" not in release
        or "refs/remotes/origin/main" not in release
    ):
        errors.append("release tags must be verified as reachable from origin/main")
    if "python -m pytest" not in release or "requirements/smoke.txt" not in release:
        errors.append(
            "release build must install test dependencies and run full pytest"
        )
    if ".[service,dev]" not in release or "import fastapi, httpx" not in release:
        errors.append(
            "release tests must install and import the HTTP service dependencies"
        )
    return errors


def check_repository_privacy(root: Path) -> list[str]:
    """Block publishable credentials and machine-local paths without echoing them."""

    return [
        f"privacy finding ({finding.rule}) in {finding.path}; matched value omitted"
        for finding in scan_repository(root)
    ]


CHECKS: tuple[tuple[str, Callable[[Path], list[str]]], ...] = (
    ("required-files", check_required_files),
    ("version-metadata", check_version_metadata),
    ("dependency-policy", check_dependency_policy),
    ("protocol-registry", check_protocol_registry),
    ("python-compatibility", check_python_compatibility),
    ("markdown-links", check_markdown_links),
    ("action-pinning", check_action_pinning),
    ("repository-privacy", check_repository_privacy),
)


def run_checks(root: Path = REPOSITORY_ROOT) -> dict[str, list[str]]:
    return {name: check(root) for name, check in CHECKS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    results = run_checks(root)
    errors = [error for group in results.values() for error in group]
    if args.as_json:
        print(
            json.dumps(
                {"ok": not errors, "checks": results, "error_count": len(errors)},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for name, check_errors in results.items():
            if check_errors:
                for error in check_errors:
                    print(f"[FAIL] {name}: {error}", file=sys.stderr)
            else:
                print(f"[OK] {name}")
        print(f"Engineering checks: {len(CHECKS)} checks, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
