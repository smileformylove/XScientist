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

ISOLATED_WHEEL_SMOKE_TIMEOUT_SECONDS = 120


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
            "xscientist/research_rollout.py",
            "xscientist/research_belief.py",
            "xscientist/research_authority.py",
            "xscientist/research_evolution.py",
            "xscientist/git_support.py",
            "xscientist/research_commands.py",
            "xscientist/demo.py",
            "xscientist/benchmark.py",
            "xscientist/evidence_index.py",
            "xscientist/completion.py",
            "xscientist/conformance.py",
            "xscientist/executor_manager.py",
            "xscientist/run_control.py",
            "xscientist/upgrade_check.py",
            "xscientist/usage_metrics.py",
            "xscientist/workspace_history.py",
            "xscientist/workspace_status.py",
            "ai_scientist/resources/configs/bfts_default.yaml",
            "ai_scientist/protocol/schemas/research_closure.schema.json",
            "ai_scientist/protocol/schemas/reproduction_receipt.schema.json",
            "ai_scientist/protocol/schemas/arft_coverage.schema.json",
            "ai_scientist/protocol/schemas/process_audit.schema.json",
            "ai_scientist/protocol/schemas/evidence_index.schema.json",
            "ai_scientist/protocol/schemas/benchmark_report_verification.schema.json",
            "ai_scientist/protocol/schemas/exploration_audit.schema.json",
            "ai_scientist/protocol/schemas/research_rollout.schema.json",
            "ai_scientist/protocol/schemas/research_rollout_audit.schema.json",
            "ai_scientist/protocol/schemas/belief_context.schema.json",
            "ai_scientist/protocol/schemas/belief_context_audit.schema.json",
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
            "mkdocs.yml",
            "tools/check_distribution.py",
            "tools/benchmark_first_run.py",
            "tools/engineering_checks.py",
            "docs/ENGINEERING.md",
            "docs/GETTING_STARTED.md",
        )
    }
    missing = sorted(required_sdist - sdist_names)
    if missing:
        errors.append("sdist missing maintenance assets: " + ", ".join(missing))

    if errors:
        raise RuntimeError("\n".join(errors))
    return wheel, sdist


def smoke_install(wheel: Path, *, install_deps: bool = True) -> None:
    with tempfile.TemporaryDirectory(prefix="xscientist_wheel_check_") as td:
        root = Path(td)
        target = root / "site"
        install_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--target",
            str(target),
            str(wheel),
        ]
        if not install_deps:
            install_command.insert(-1, "--no-deps")
        try:
            install = subprocess.run(
                install_command,
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "isolated wheel install timed out after 300 seconds"
            ) from exc
        if install.returncode:
            raise RuntimeError(f"isolated wheel install failed:\n{install.stderr}")

        script = """
import contextlib
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import xscientist
from xscientist import (
    ResearchEvolution,
    ResearchLifecycle,
    ResearchRepository,
    inspect_workspace_history,
    preview_workspace_rollback,
    rollback_workspace_checkpoint,
    save_workspace_checkpoint,
)
from xscientist.cli import main as cli_main
from xscientist.research_authority import require_independent_evaluator
from ai_scientist.protocol.schemas import available_schemas, load_schema
from ai_scientist.resources import bfts_config_path, latex_template_dir
assert xscientist.__version__
assert all((ResearchRepository, ResearchLifecycle, ResearchEvolution))
assert all(callable(item) for item in (
    inspect_workspace_history,
    preview_workspace_rollback,
    rollback_workspace_checkpoint,
    save_workspace_checkpoint,
))
assert callable(require_independent_evaluator)
assert load_schema("manifest")["type"] == "object"
assert "context_pack" in available_schemas()
assert bfts_config_path("default").is_file()
assert (latex_template_dir("icml") / "template.tex").is_file()
demo_root = Path.cwd() / "demo"
demo_output = io.StringIO()
with contextlib.redirect_stdout(demo_output):
    demo_exit = cli_main([
        "demo", str(demo_root),
        "--autopilot",
        "--git-user-name", "XScientist CI",
        "--git-user-email", "ci@example.invalid",
        "--json",
    ])
demo = json.loads(demo_output.getvalue())
assert demo_exit == 0
assert demo["provider_used"] is False and demo["network_used"] is False
assert demo["cost_usd"] == 0.0 and demo["dag"]["integrity_ok"] is True
assert demo["autopilot_fixture"]["resumable"] is True
status_output = io.StringIO()
with contextlib.redirect_stdout(status_output):
    status_exit = cli_main(["status", str(demo_root), "--json"])
status = json.loads(status_output.getvalue())
assert status_exit == 0 and status["research"]["initialized"] is True
assert status["run"]["started"] is True
assert status["run"]["current_stage"] == "complete"
assert status["budget"]["used"]["cost_usd"] == 0.0
assert status["result"]["epistemic_status"] == "machine_synthesized_unverified"
human_status_output = io.StringIO()
with contextlib.redirect_stdout(human_status_output):
    human_status_exit = cli_main(["status", str(demo_root), "--lang", "en"])
assert human_status_exit == 0
assert "Scientific progress:" in human_status_output.getvalue()
assert "Resolve or narrow the contested claim" in human_status_output.getvalue()
history_output = io.StringIO()
with contextlib.redirect_stdout(history_output):
    history_exit = cli_main(["history", "list", str(demo_root), "--json"])
history = json.loads(history_output.getvalue())
assert history_exit == 0 and history["entries"]
assert history["auto_push"] is False
assert history["pending"]["eligible"] == []
audit_output = io.StringIO()
with contextlib.redirect_stdout(audit_output):
    audit_exit = cli_main(["audit", str(demo_root), "--level", "trace", "--json"])
audit = json.loads(audit_output.getvalue())
assert audit_exit == 0 and audit["target_level"] == "trace"
assert audit["integrity"]["ok"] is True and audit["payloads_disclosed"] is False
completion_output = io.StringIO()
with contextlib.redirect_stdout(completion_output):
    assert cli_main(["completion", "bash"]) == 0
assert "complete -F" in completion_output.getvalue()
upgrade_output = io.StringIO()
with contextlib.redirect_stdout(upgrade_output):
    assert cli_main(["upgrade", "check", "--workspace", str(demo_root), "--json"]) == 0
upgrade = json.loads(upgrade_output.getvalue())
assert upgrade["compatible"] is True and upgrade["mutated"] is False
kit_root = Path.cwd() / "protocol-kit"
conformance_init_output = io.StringIO()
with contextlib.redirect_stdout(conformance_init_output):
    assert cli_main(["conformance", "init", str(kit_root), "--json"]) == 0
conformance_output = io.StringIO()
with contextlib.redirect_stdout(conformance_output):
    assert cli_main(["conformance", "check", str(kit_root), "--json"]) == 0
conformance = json.loads(conformance_output.getvalue())
assert conformance["passed"] == conformance["total"] == 2
doctor_output = io.StringIO()
with contextlib.redirect_stdout(doctor_output):
    doctor_exit = cli_main([
        "doctor", "--workspace", str(demo_root), "--task", "research"
    ])
assert doctor_exit == 1
assert "Dependencies" in doctor_output.getvalue()
assert "Next actions:" in doctor_output.getvalue()
print(json.dumps({
    "version": xscientist.__version__,
    "schemas": len(available_schemas()),
    "demo_nodes": demo["dag"]["nodes"],
    "demo_closure": demo["dag"]["closure"],
    "human_cli_smoke": True,
    "history_cli_smoke": True,
}))
"""
        try:
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
                timeout=ISOLATED_WHEEL_SMOKE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "isolated wheel demo smoke timed out after "
                f"{ISOLATED_WHEEL_SMOKE_TIMEOUT_SECONDS} seconds"
            ) from exc
        if smoke.returncode:
            raise RuntimeError(f"isolated wheel import failed:\n{smoke.stderr}")
        print(f"[OK] isolated wheel smoke: {smoke.stdout.strip()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=REPOSITORY_ROOT / "dist")
    parser.add_argument("--no-install-smoke", action="store_true")
    parser.add_argument(
        "--no-install-deps",
        action="store_true",
        help="reuse already installed core dependencies while testing wheel code",
    )
    args = parser.parse_args(argv)
    dist_dir = args.dist_dir.expanduser().resolve()
    wheel, sdist = inspect_distribution(dist_dir)
    print(f"[OK] wheel inventory: {wheel.name}")
    print(f"[OK] sdist inventory: {sdist.name}")
    if not args.no_install_smoke:
        smoke_install(wheel, install_deps=not args.no_install_deps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
