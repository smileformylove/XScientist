#!/usr/bin/env python3
"""Inspect built wheel/sdist contents and smoke-import the wheel in isolation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.engineering_checks import _project_version  # noqa: E402


def _find_one(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(dist_dir.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {pattern} in {dist_dir}, found {len(matches)}"
        )
    return matches[0]


def inspect_distribution(dist_dir: Path) -> tuple[Path, Path]:
    version = _project_version(REPOSITORY_ROOT)
    wheel = _find_one(dist_dir, f"xscientist-{version}-*.whl")
    sdist = _find_one(dist_dir, f"xscientist-{version}.tar.gz")
    errors: list[str] = []

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        metadata_name = next(
            (name for name in wheel_names if name.endswith(".dist-info/METADATA")),
            None,
        )
        if metadata_name is None:
            errors.append("wheel has no METADATA")
        else:
            metadata = BytesParser().parsebytes(archive.read(metadata_name))
            if metadata.get("Version") != version:
                errors.append("wheel version does not match xscientist/_version.py")
            if metadata.get("Requires-Python") != ">=3.10":
                errors.append("wheel Requires-Python must remain >=3.10")

        required_wheel = {
            "xscientist/__init__.py",
            "xscientist/cli.py",
            "ai_scientist/protocol/SPEC.md",
            "ai_scientist/protocol/schemas/manifest.schema.json",
            "ai_scientist/protocol/schemas/context_pack.schema.json",
            "ai_scientist/protocol/schemas/research_checkpoint.schema.json",
            "xscientist/research_git.py",
            "xscientist/research_cli.py",
            "xscientist/research_vcs.py",
            "xscientist/research_lifecycle.py",
            "xscientist/research_closure.py",
            "xscientist/research_evolution.py",
            "xscientist/git_support.py",
            "xscientist/research_commands.py",
            "ai_scientist/resources/configs/bfts_default.yaml",
            "ai_scientist/protocol/schemas/research_closure.schema.json",
            "ai_scientist/protocol/schemas/reproduction_receipt.schema.json",
        }
        missing = sorted(required_wheel - wheel_names)
        if missing:
            errors.append("wheel missing runtime assets: " + ", ".join(missing))
        leaked = sorted(
            name
            for name in wheel_names
            if name.startswith(("tests/", "tools/", "docs/"))
            or name.endswith(".pyc")
            or "/__pycache__/" in name
        )
        if leaked:
            errors.append(
                "wheel contains source-only/generated files: " + ", ".join(leaked[:5])
            )

    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())
    root_prefix = f"xscientist-{version}/"
    required_sdist = {
        root_prefix + relative
        for relative in (
            "CHANGELOG.md",
            "CITATION.cff",
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "tools/check_distribution.py",
            "tools/engineering_checks.py",
            "docs/ENGINEERING.md",
        )
    }
    missing = sorted(required_sdist - sdist_names)
    if missing:
        errors.append("sdist missing maintenance assets: " + ", ".join(missing))

    if errors:
        raise RuntimeError("\n".join(errors))
    return wheel, sdist


def smoke_install(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="xscientist_wheel_check_") as td:
        root = Path(td)
        target = root / "site"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if install.returncode:
            raise RuntimeError(f"isolated wheel install failed:\n{install.stderr}")

        script = """
import json
import sys
sys.path.insert(0, sys.argv[1])
import xscientist
from xscientist import ResearchEvolution, ResearchLifecycle, ResearchRepository
from ai_scientist.protocol.schemas import available_schemas, load_schema
from ai_scientist.resources import bfts_config_path, latex_template_dir
assert xscientist.__version__
assert all((ResearchRepository, ResearchLifecycle, ResearchEvolution))
assert load_schema("manifest")["type"] == "object"
assert "context_pack" in available_schemas()
assert bfts_config_path("default").is_file()
assert (latex_template_dir("icml") / "template.tex").is_file()
print(json.dumps({"version": xscientist.__version__, "schemas": len(available_schemas())}))
"""
        smoke = subprocess.run(
            [sys.executable, "-c", script, str(target)],
            cwd=root,
            env={
                **os.environ,
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        if smoke.returncode:
            raise RuntimeError(f"isolated wheel import failed:\n{smoke.stderr}")
        print(f"[OK] isolated wheel smoke: {smoke.stdout.strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=REPOSITORY_ROOT / "dist")
    parser.add_argument("--no-install-smoke", action="store_true")
    args = parser.parse_args(argv)
    dist_dir = args.dist_dir.expanduser().resolve()
    wheel, sdist = inspect_distribution(dist_dir)
    print(f"[OK] wheel inventory: {wheel.name}")
    print(f"[OK] sdist inventory: {sdist.name}")
    if not args.no_install_smoke:
        smoke_install(wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
