"""Capability probing for the local Research VCS persistence backend."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence


def _install_hint() -> str:
    system = platform.system().lower()
    if system == "darwin":
        return "Install the macOS command-line tools with `xcode-select --install`."
    if system == "windows":
        return "Install Git with `winget install --id Git.Git -e`."
    return (
        "Install Git with your system package manager, for example `apt install git`."
    )


def _run(git: str, args: Sequence[str], *, cwd: Path | None = None):
    return subprocess.run(
        [git, *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )


def _probe_capabilities(git: str) -> tuple[dict[str, bool], list[str]]:
    capabilities = {
        "default_branch_init": False,
        "switch": False,
        "merge_tree_write": False,
    }
    errors: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="xscientist_git_doctor_") as td:
            root = Path(td) / "probe"
            initialized = _run(git, ["init", "-q", "-b", "main", str(root)])
            if initialized.returncode:
                errors.append(
                    "Git cannot initialize a repository with an explicit default branch."
                )
                return capabilities, errors
            capabilities["default_branch_init"] = True
            _run(git, ["config", "user.name", "XScientist"], cwd=root)
            _run(git, ["config", "user.email", "xscientist@localhost"], cwd=root)
            (root / "base.txt").write_text("base\n", encoding="utf-8")
            if (
                _run(git, ["add", "base.txt"], cwd=root).returncode
                or _run(git, ["commit", "-q", "-m", "base"], cwd=root).returncode
            ):
                errors.append("Git cannot create the local capability-probe commit.")
                return capabilities, errors
            switched = _run(git, ["switch", "-q", "-c", "candidate"], cwd=root)
            if switched.returncode:
                errors.append("Git does not support the required `switch` command.")
                return capabilities, errors
            capabilities["switch"] = True
            (root / "candidate.txt").write_text("candidate\n", encoding="utf-8")
            _run(git, ["add", "candidate.txt"], cwd=root)
            if _run(git, ["commit", "-q", "-m", "candidate"], cwd=root).returncode:
                errors.append("Git cannot create a branch-local probe commit.")
                return capabilities, errors
            candidate = _run(git, ["rev-parse", "HEAD"], cwd=root).stdout.strip()
            _run(git, ["switch", "-q", "main"], cwd=root)
            (root / "main.txt").write_text("main\n", encoding="utf-8")
            _run(git, ["add", "main.txt"], cwd=root)
            if _run(git, ["commit", "-q", "-m", "main"], cwd=root).returncode:
                errors.append("Git cannot create the second merge-probe commit.")
                return capabilities, errors
            current = _run(git, ["rev-parse", "HEAD"], cwd=root).stdout.strip()
            merge_tree = _run(
                git,
                ["merge-tree", "--write-tree", current, candidate],
                cwd=root,
            )
            capabilities["merge_tree_write"] = merge_tree.returncode == 0
            if not capabilities["merge_tree_write"]:
                errors.append(
                    "Git lacks the `merge-tree --write-tree` capability required "
                    "for read-only scientific merge preflight."
                )
    except OSError:
        errors.append(
            "Git capability probing could not use the system temporary directory."
        )
    return capabilities, errors


def inspect_git_backend() -> dict[str, Any]:
    """Return a path-safe readiness report for the Research VCS Git adapter."""

    git = shutil.which("git")
    if not git:
        return {
            "schema": "xscientist.git-doctor.v1",
            "ok": False,
            "backend": "git",
            "available": False,
            "version": None,
            "capabilities": {
                "default_branch_init": False,
                "switch": False,
                "merge_tree_write": False,
            },
            "errors": ["Git is required by the current local Research VCS adapter."],
            "install_hint": _install_hint(),
            "host_paths_disclosed": False,
        }
    version_result = _run(git, ["--version"])
    version_match = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", version_result.stdout)
    version = version_match.group(1) if version_match else None
    errors: list[str] = []
    if version_result.returncode or version is None:
        errors.append("The Git executable did not return a parseable version.")
        capabilities = {
            "default_branch_init": False,
            "switch": False,
            "merge_tree_write": False,
        }
    else:
        capabilities, probe_errors = _probe_capabilities(git)
        errors.extend(probe_errors)
    return {
        "schema": "xscientist.git-doctor.v1",
        "ok": not errors and all(capabilities.values()),
        "backend": "git",
        "available": True,
        "version": version,
        "capabilities": capabilities,
        "errors": errors,
        "install_hint": None if not errors else _install_hint(),
        "host_paths_disclosed": False,
    }


__all__ = ["inspect_git_backend"]
