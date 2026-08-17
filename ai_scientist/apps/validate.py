#!/usr/bin/env python3
"""Validate either a source distribution or an installed XScientist package."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import py_compile
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_PYTHON = (3, 10)
_IGNORED_PATH_PARTS = {"__pycache__", "tests"}


def is_source_distribution() -> bool:
    return (PROJECT_ROOT / "pyproject.toml").is_file() and (
        PROJECT_ROOT / "tools" / "repository_validation.py"
    ).is_file()


def ensure_supported_python() -> None:
    if sys.version_info >= MIN_PYTHON:
        return
    minimum = ".".join(str(part) for part in MIN_PYTHON)
    raise SystemExit(
        f"xscientist validate requires Python >= {minimum}; "
        f"current interpreter is {sys.version.split()[0]}"
    )


def iter_installed_python_files() -> list[Path]:
    roots = [PROJECT_ROOT / "ai_scientist", PROJECT_ROOT / "xscientist"]
    files = [
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*.py")
        if not any(part in _IGNORED_PATH_PARTS for part in path.relative_to(root).parts)
    ]
    for module_name in (
        "auth_cli.py",
        "continuous_paper_generator.py",
        "continuous_research_daemon.py",
        "feedback_cli.py",
        "launch_scientist_bfts.py",
        "launch_scientist_zhipu.py",
        "preflight_check.py",
        "research_manager.py",
        "run_ara_fork.py",
        "run_project.py",
        "validate_repo.py",
    ):
        path = PROJECT_ROOT / module_name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def run_installed_py_compile() -> None:
    files = iter_installed_python_files()
    with tempfile.TemporaryDirectory(prefix="xscientist_py_compile_") as td:
        cache_root = Path(td)
        for index, path in enumerate(files):
            py_compile.compile(
                str(path),
                cfile=str(cache_root / f"{index}.pyc"),
                doraise=True,
            )
    print(f"[OK] py_compile passed for {len(files)} Python files")


def run_installed_package_smoke() -> None:
    import xscientist
    from ai_scientist.protocol.schemas import load_schema
    from ai_scientist.resources import bfts_config_path, latex_template_dir

    required = [
        bfts_config_path("default"),
        bfts_config_path("deep"),
        latex_template_dir("icbinb") / "template.tex",
        latex_template_dir("icml") / "template.tex",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Installed package resources are missing: {missing}")
    manifest_schema = load_schema("manifest")
    required_fields = {
        "schema_version",
        "created_at",
        "source_exp_dir",
        "idea",
        "counts",
    }
    if not required_fields.issubset(manifest_schema.get("required") or []):
        raise RuntimeError("Installed ARA manifest schema is invalid")
    if not xscientist.__version__:
        raise RuntimeError("Installed xscientist package has no version")
    print(
        "[OK] installed package smoke passed:",
        json.dumps(
            {
                "version": xscientist.__version__,
                "resources": len(required) + 1,
                "project_root_disclosed": False,
            },
            ensure_ascii=False,
        ),
    )


def run_import_smoke() -> None:
    modules = [
        "xscientist",
        "xscientist.cli",
        "xscientist.client",
        "xscientist.service",
        "ai_scientist.apps.project",
        "ai_scientist.apps.batch",
        "ai_scientist.apps.daemon",
        "ai_scientist.apps.manager",
        "ai_scientist.apps.ara",
        "ai_scientist.apps.bfts",
        "ai_scientist.apps.zhipu",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
    print(f"[OK] import smoke passed for {len(modules)} modules")


def main(argv: list[str] | None = None) -> int:
    if is_source_distribution():
        from tools.repository_validation import main as repository_main

        return repository_main(argv)

    parser = argparse.ArgumentParser(description="Installed XScientist validation")
    parser.add_argument(
        "--full-import-smoke",
        action="store_true",
        help="also import all installed application entrypoints",
    )
    args = parser.parse_args(argv)
    ensure_supported_python()
    os.environ.setdefault(
        "RESEARCH_OUTPUT_DIR", tempfile.mkdtemp(prefix="xscientist_validate_")
    )
    run_installed_py_compile()
    run_installed_package_smoke()
    if args.full_import_smoke:
        run_import_smoke()
    else:
        print("[SKIP] import smoke not requested")
    print("Validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
