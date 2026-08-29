#!/usr/bin/env python3
"""Validate either a source distribution or an installed XScientist package."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
_GLM53_MODEL = "openai_compat/glm-5.3"
_GLM53_MAX_TOTAL_TOKENS = 500_000
_GLM53_MAX_WALL_TIME_SECONDS = 21_600
_GLM53_FORBIDDEN_CONFIG_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "base_url",
    "endpoint",
    "headers",
)


def _nested_config_keys(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key).lower() for key in value] + [
            nested for child in value.values() for nested in _nested_config_keys(child)
        ]
    if isinstance(value, list):
        return [nested for child in value for nested in _nested_config_keys(child)]
    return []


def _is_positive_limit(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _validate_glm53_profile(profile: object) -> None:
    if not isinstance(profile, dict):
        raise RuntimeError("Installed GLM-5.3 BFTS profile is invalid")
    agent = profile.get("agent")
    if not isinstance(agent, dict):
        raise RuntimeError("Installed GLM-5.3 BFTS agent profile is invalid")

    models = [
        ((profile.get("report") or {}).get("model")),
        *(
            (agent.get(role) or {}).get("model")
            for role in ("code", "feedback", "vlm_feedback", "summary")
        ),
    ]
    if models != [_GLM53_MODEL] * 5 or "select_node" in agent:
        raise RuntimeError("Installed GLM-5.3 BFTS model routing is invalid")

    forbidden_keys = {
        key
        for key in _nested_config_keys(profile)
        if any(fragment in key for fragment in _GLM53_FORBIDDEN_CONFIG_KEY_FRAGMENTS)
    }
    serialized = json.dumps(profile, ensure_ascii=True, sort_keys=True).lower()
    if forbidden_keys or "sk-" in serialized or "://" in serialized:
        raise RuntimeError("Installed GLM-5.3 BFTS profile contains transport data")

    execution = profile.get("exec") or {}
    if (
        execution.get("require_isolation") is not True
        or execution.get("network") != "none"
        or execution.get("allow_experiment_network") is not False
        or execution.get("read_only_root") is not True
        or not _is_positive_limit(execution.get("timeout"))
    ):
        raise RuntimeError("Installed GLM-5.3 BFTS isolation policy is invalid")

    multi_seed = agent.get("multi_seed_eval") or {}
    seeds = multi_seed.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or not all(
            isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds
        )
        or len(seeds) != len(set(seeds))
        or multi_seed.get("num_seeds") != len(seeds)
        or 42 in seeds
    ):
        raise RuntimeError("Installed GLM-5.3 confirmation seeds are not held out")

    budget = profile.get("llm_budget") or {}
    if (
        budget.get("max_total_tokens") != _GLM53_MAX_TOTAL_TOKENS
        or budget.get("max_wall_time_seconds") != _GLM53_MAX_WALL_TIME_SECONDS
    ):
        raise RuntimeError("Installed GLM-5.3 BFTS budget contract is invalid")


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
    from omegaconf import OmegaConf

    from ai_scientist.protocol.schemas import load_schema
    from ai_scientist.resources import bfts_config_path, latex_template_dir

    required = [
        bfts_config_path("default"),
        bfts_config_path("deep"),
        bfts_config_path("glm53"),
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
    glm53 = OmegaConf.to_container(
        OmegaConf.load(bfts_config_path("glm53")), resolve=True
    )
    _validate_glm53_profile(glm53)
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
