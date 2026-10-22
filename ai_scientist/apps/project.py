#!/usr/bin/env python3
"""
XScientist project runner
在一个独立文件夹中运行完整的科研流程：想法生成 -> 实验 -> 论文
支持并行处理多个论文和自动反思改进
不影响 ai_scientist 原有代码结构
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Mapping
import traceback
from contextlib import contextmanager

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

_MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_COMMITS = 4096
_MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_BYTES = 1024 * 1024

from ai_scientist.utils.deferred_imports import load_module_attr
from ai_scientist.utils.high_quality_pipeline import run_high_quality_pass
from ai_scientist.utils.workflow_cli import (
    apply_project_autopilot_profile,
    normalize_project_workflow_args,
)
from ai_scientist.utils.fallback_audit import (
    format_strict_fallback_error,
    record_quality_fallback_if_needed,
    record_ranking_fallbacks,
    should_enforce_strict_fallbacks,
)
from ai_scientist.utils.pipeline_helpers import (
    find_latest_bfts_run_dir,
    find_latest_pdf_path,
)
from ai_scientist.utils.quality_workflow import (
    evaluate_final_submission_readiness,
    execute_quality_workflow_with_followups,
)
from ai_scientist.utils.review_workflow import build_review_execution_plan
from ai_scientist.utils.writeup_workflow import build_writeup_execution_plan
from ai_scientist.utils.workflow_modes import (
    build_workflow_manifest_metadata,
    list_workflow_modes,
    resolve_template_mode_for_workflow,
    resolve_workflow_mode,
)
from ai_scientist.utils.workflow_selection import (
    resolve_paper_type_for_venue,
    select_ranked_idea_candidates,
)
from ai_scientist.utils.experiment_report import write_experiment_report
from ai_scientist.utils.experiment_registry import (
    build_experiment_record,
    record_terminal_experiment_failure,
    save_experiment_registry,
)
from ai_scientist.utils.ara_pipeline import (
    activate_llm_tracing,
    deactivate_llm_tracing,
    finalize_ara_for_idea,
    stage_seed_from_cli,
    summarise_finalize,
    summarise_seed_stage,
)
from ai_scientist.utils.guardrail_artifacts import (
    load_guardrail_artifacts,
    result_passed_writeup_guardrails,
)
from ai_scientist.utils.integrity_workflow import (
    integrity_forensics_result_fields,
    integrity_report_payload,
    run_integrity_forensics_for_manuscript,
    summarize_integrity_forensics_result,
)
from ai_scientist.utils.manuscript_state import (
    build_manuscript_state,
    save_manuscript_state,
)
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    render_research_program_markdown,
    save_contract_artifact,
    update_pipeline_artifact,
)
from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.research_planning import (
    build_claim_evidence_graph,
    build_idea_cards,
    build_research_plan,
)
from ai_scientist.utils.research_integrity import (
    build_preregistration,
    save_preregistration,
)
from ai_scientist.utils.hypothesis_archive import (
    build_hypothesis_archive,
    save_hypothesis_archive,
)
from ai_scientist.utils.epistemic_graph import seed_scientific_foundation
from ai_scientist.utils.self_review_optimizer import (
    apply_issue_driven_rewrite,
    assess_self_review_gate,
    build_issue_ledger,
    evaluate_issue_progress,
    save_self_review_artifacts,
)
from ai_scientist.utils.experiment_todo_progress import (
    bootstrap_todo_tasks_from_round_gate,
    build_todo_progress_payload,
    evaluate_todo_progress_snapshot,
    load_experiment_todo_payload,
    save_todo_progress_artifacts,
)
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    list_writing_profiles,
)
from ai_scientist.writing_skill_pack import list_writing_skills
from ai_scientist.utils.runtime_bootstrap import (
    format_project_relative_path,
    initialize_runtime,
    require_model_credentials,
    resolve_writing_profile_env,
)
from ai_scientist.utils.workflow_runtime import (
    build_workflow_runtime_plan,
    execute_review_suite,
)
from ai_scientist.utils.critic_workflow import run_independent_critic_pass
from ai_scientist.utils.stage_standards import save_stage_standards
from ai_scientist.utils.truth_contracts import save_truth_contract_bundle
from ai_scientist.utils.sample_gate import (
    evaluate_and_save_sample_gate,
    save_sample_gate_plan,
)
from ai_scientist.utils.decision_log import (
    build_sample_gate_decision_options,
    build_workflow_strategy_decision_options,
    record_decision,
    record_model_provider_decisions,
)
from ai_scientist.utils.gate_preconditions import (
    GatePreconditionContext,
    UnsatisfiableGateError,
    assert_gate_preconditions_satisfiable,
)

# 导入路径配置
from ai_scientist.config.paths import (
    confined_path,
    get_project_dir,
    idea_storage_key,
    resolve_output_path,
)

_HOSTILE_CRITIC_ABLATION_ENV = "AI_SCIENTIST_ABLATE_HOSTILE_CRITIC"
_OWNER_AWARE_REPAIR_ABLATION_ENV = "AI_SCIENTIST_DISABLE_OWNER_AWARE_REPAIR"


def create_client(*args, **kwargs):
    return load_module_attr("ai_scientist.llm", "create_client")(*args, **kwargs)


def generate_temp_free_idea(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_ideation_temp_free",
        "generate_temp_free_idea",
    )(*args, **kwargs)


def configure_benchmark_ablation_env(
    *,
    disable_hostile_critic: bool = False,
    disable_owner_aware_repair: bool = False,
) -> dict[str, bool]:
    """Make benchmark ablation flags explicit and reproducible for this run."""

    os.environ.pop(_HOSTILE_CRITIC_ABLATION_ENV, None)
    os.environ.pop(_OWNER_AWARE_REPAIR_ABLATION_ENV, None)
    if disable_hostile_critic:
        os.environ[_HOSTILE_CRITIC_ABLATION_ENV] = "1"
    if disable_owner_aware_repair:
        os.environ[_OWNER_AWARE_REPAIR_ABLATION_ENV] = "1"
    return {
        "disable_hostile_critic": bool(disable_hostile_critic),
        "disable_owner_aware_repair": bool(disable_owner_aware_repair),
    }


def perform_experiments_bfts(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager",
        "perform_experiments_bfts",
    )(*args, **kwargs)


def idea_to_markdown(*args, **kwargs):
    return load_module_attr("ai_scientist.treesearch.bfts_utils", "idea_to_markdown")(
        *args, **kwargs
    )


def edit_bfts_config_file(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.bfts_utils",
        "edit_bfts_config_file",
    )(*args, **kwargs)


def write_bfts_task_descriptor(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.treesearch.bfts_utils",
        "write_bfts_task_descriptor",
    )(*args, **kwargs)


def aggregate_plots(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_plotting", "aggregate_plots")(
        *args, **kwargs
    )


def perform_icbinb_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "perform_writeup")(
        *args, **kwargs
    )


def gather_citations(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_icbinb_writeup", "gather_citations")(
        *args, **kwargs
    )


def perform_normal_writeup(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_writeup", "perform_writeup")(
        *args, **kwargs
    )


def perform_review(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "perform_review")(
        *args, **kwargs
    )


def load_paper(*args, **kwargs):
    return load_module_attr("ai_scientist.perform_llm_review", "load_paper")(
        *args, **kwargs
    )


def perform_imgs_cap_ref_review(*args, **kwargs):
    return load_module_attr(
        "ai_scientist.perform_vlm_review",
        "perform_imgs_cap_ref_review",
    )(*args, **kwargs)


def resolve_project_path(
    project_dir: str,
    *,
    output_root: str | Path | None = None,
) -> Path:
    """解析项目目录路径（相对路径会落到统一输出根目录下）。"""
    project_path = Path(project_dir).expanduser()
    if project_path.is_absolute():
        return project_path
    return get_project_dir(project_dir, output_root=output_root)


_MANAGED_PROJECT_DIRECTORY_NAMES = (
    "00_config",
    "01_ideas",
    "02_experiments",
    "03_papers",
    "04_logs",
)
_PROJECT_PROGRESS_FILENAME = "progress.json"


def _assert_managed_project_layout(project_dir: str | Path) -> Path:
    """Reject managed project directories that escape through a symlink."""

    root = Path(os.path.abspath(Path(project_dir).expanduser()))
    resolved_root = root.resolve(strict=False)
    for name in _MANAGED_PROJECT_DIRECTORY_NAMES:
        candidate = root / name
        if candidate.is_symlink():
            raise ValueError(f"managed project directory {name} must not be a symlink")
        try:
            candidate.resolve(strict=False).relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"managed project directory {name} escapes the project root"
            ) from exc
        if candidate.exists() and not candidate.is_dir():
            raise ValueError(f"managed project directory {name} is not a directory")
    return root


def _managed_project_file(
    project_dir: str | Path,
    directory_name: str,
    filename: str,
) -> Path:
    """Return one regular managed leaf without following a leaf symlink."""

    if directory_name not in _MANAGED_PROJECT_DIRECTORY_NAMES:
        raise ValueError("unsupported managed project directory")
    if Path(filename).name != filename:
        raise ValueError("managed project filename must be a single path component")
    root = _assert_managed_project_layout(project_dir)
    candidate = root / directory_name / filename
    if candidate.is_symlink():
        raise ValueError(
            f"managed project file {directory_name}/{filename} must not be a symlink"
        )
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"managed project file {directory_name}/{filename} escapes the project root"
        ) from exc
    if candidate.exists() and not candidate.is_file():
        raise ValueError(
            f"managed project file {directory_name}/{filename} is not a regular file"
        )
    return candidate


def _load_project_ideas_list(path: Path, *, label: str) -> list[Any]:
    """Load a managed or explicitly supplied ideas list for identity checks."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not validate {label} ideas") from exc
    if not isinstance(payload, list):
        raise ValueError(f"{label} ideas must contain a JSON list")
    return payload


def _managed_project_ideas_hash(project_dir: str | Path) -> str | None:
    """Return the one canonical persisted ideas identity, failing on ambiguity."""

    generated = _managed_project_file(project_dir, "01_ideas", "generated_ideas.json")
    stable = _managed_project_file(project_dir, "01_ideas", "ideas.json")
    existing = [path for path in (generated, stable) if path.is_file()]
    if not existing:
        return None
    identities = {
        canonical_content_hash(_load_project_ideas_list(path, label="persisted"))
        for path in existing
    }
    if len(identities) != 1:
        raise ValueError(
            "--resume refused conflicting generated_ideas.json and ideas.json"
        )
    return next(iter(identities))


def create_project_structure(
    project_dir: str,
    *,
    output_root: str | Path | None = None,
) -> dict:
    """创建项目目录结构。"""
    project_path = _assert_managed_project_layout(
        resolve_project_path(project_dir, output_root=output_root)
    )

    # 创建子目录
    dirs = {
        "root": project_path,
        "ideas": project_path / "01_ideas",
        "experiments": project_path / "02_experiments",
        "papers": project_path / "03_papers",
        "logs": project_path / "04_logs",
    }

    project_path.mkdir(parents=True, exist_ok=True)
    for name in _MANAGED_PROJECT_DIRECTORY_NAMES:
        (project_path / name).mkdir(parents=True, exist_ok=True)
    _assert_managed_project_layout(project_path)

    print(f"✅ 创建项目目录结构: {project_path}")
    print(
        "   (相对于项目根目录: "
        f"{format_project_relative_path(project_path, project_root=PROJECT_ROOT)})"
    )
    return dirs


def _prepare_project_input(args: argparse.Namespace, dirs: dict[str, Path]) -> None:
    supplied = [bool(args.question), bool(args.topic), bool(args.ideas)]
    if sum(supplied) > 1:
        raise ValueError("--question, --topic, and --ideas are mutually exclusive")

    project_root = _assert_managed_project_layout(dirs["root"])
    topic_path = _managed_project_file(project_root, "00_config", "topic.md")
    generated = _managed_project_file(project_root, "01_ideas", "generated_ideas.json")
    stable = _managed_project_file(project_root, "01_ideas", "ideas.json")
    existing_idea_paths = [path for path in (generated, stable) if path.is_file()]

    persisted_idea_hash: str | None = None
    if args.resume and existing_idea_paths:
        persisted_idea_hash = _managed_project_ideas_hash(project_root)

    progress_path = _managed_project_file(
        project_root, "04_logs", _PROJECT_PROGRESS_FILENAME
    )
    prior_input_identity = bool(
        existing_idea_paths or topic_path.is_file() or progress_path.is_file()
    )
    if args.resume and args.ideas:
        supplied_ideas = _load_project_ideas_list(
            Path(args.ideas).expanduser(), label="supplied"
        )
        if persisted_idea_hash is None:
            if prior_input_identity:
                raise ValueError(
                    "--resume refused supplied ideas because the existing project "
                    "has no persisted ideas identity"
                )
        elif canonical_content_hash(supplied_ideas) != persisted_idea_hash:
            raise ValueError(
                "--resume refused because the supplied ideas differ from the "
                "existing project ideas"
            )

    if args.question:
        question = str(args.question).strip()
        if not question:
            raise ValueError("--question cannot be empty")
        content = f"# Research question\n\n{question}\n"
        if args.resume and topic_path.exists():
            existing = topic_path.read_text(encoding="utf-8")
            if existing != content:
                raise ValueError(
                    "--resume refused because the supplied question differs from the "
                    "existing project question"
                )
        elif args.resume and prior_input_identity:
            raise ValueError(
                "--resume refused a question because the existing project has no "
                "persisted topic identity"
            )
        atomic_write_text(topic_path, content)
        args.topic = str(topic_path)
    elif args.topic:
        try:
            supplied_topic = Path(args.topic).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError("could not read the supplied topic") from exc
        if args.resume and topic_path.is_file():
            if topic_path.read_text(encoding="utf-8") != supplied_topic:
                raise ValueError(
                    "--resume refused because the supplied topic differs from the "
                    "existing project topic"
                )
        elif args.resume and prior_input_identity:
            raise ValueError(
                "--resume refused an explicit topic because this older project has "
                "no persisted topic identity; omit --topic or start a new project"
            )
        atomic_write_text(topic_path, supplied_topic)
        args.topic = str(topic_path)
    elif args.resume and not args.topic and not args.ideas and topic_path.is_file():
        args.topic = str(topic_path)

    if args.resume and (generated.is_file() or stable.is_file()):
        args.skip_ideation = True


def _materialize_project_ideas_input(
    project_dir: str | Path,
    source: str | Path,
) -> str:
    """Copy user-supplied ideas with an atomic, project-confined replacement."""

    source_path = Path(source).expanduser()
    try:
        content = source_path.read_text(encoding="utf-8")
        payload = json.loads(content)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("could not read the supplied ideas JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("supplied ideas must contain a JSON list")
    destination = _managed_project_file(project_dir, "01_ideas", "ideas.json")
    atomic_write_text(destination, content)
    return str(destination)


def _run_autopilot_preflight(args: argparse.Namespace) -> list[dict[str, object]]:
    if not args.autopilot:
        return []
    if bool(getattr(args, "skip_experiment", False)):
        return [
            {
                "label": "Experiment isolation",
                "ok": True,
                "severity": "info",
                "detail": (
                    "not required because --skip-experiment disables generated-code "
                    "execution; isolation is checked before any later experiment run"
                ),
            }
        ]
    from ai_scientist.apps.preflight import check_bfts_config

    checks = check_bfts_config(
        args.bfts_config,
        workspace=getattr(args, "project_dir", None),
    )
    rows = [
        {
            "label": item.label,
            "ok": bool(item.ok),
            "severity": item.severity,
            "detail": item.detail,
        }
        for item in checks
    ]
    blockers = [row for row in rows if not row["ok"] and row["severity"] == "error"]
    if blockers:
        details = "; ".join(f"{row['label']}: {row['detail']}" for row in blockers)
        raise RuntimeError(
            "Autopilot preflight failed before any research-model call: "
            f"{details}. Run `xscientist doctor --deep --task research` and ensure "
            "the configured isolated experiment runtime is available."
        )
    return rows


def _prepare_autopilot_bfts_config(
    args: argparse.Namespace,
    *,
    reuse_existing: bool = False,
) -> str | None:
    """Create a finite, isolated BFTS config without mutating the user's source."""

    from ai_scientist.resources import resolve_bfts_config_path

    source = resolve_bfts_config_path(args.bfts_config)
    explicit_executor_workspace = str(
        os.environ.get("XSCIENTIST_WORKSPACE") or ""
    ).strip()
    bootstrap_config_override = False
    try:
        from xscientist.entrypoints import (
            _executor_workspace_was_bootstrapped,
            _project_bfts_config_was_explicit,
        )

        bootstrap_config_override = bool(
            _project_bfts_config_was_explicit()
            and _executor_workspace_was_bootstrapped()
        )
    except ImportError:
        pass
    if not explicit_executor_workspace or bootstrap_config_override:
        from xscientist.executor_manager import resolve_executor_workspace

        trusted_executor_workspace = resolve_executor_workspace(source)
        if trusted_executor_workspace is not None:
            resolved_workspace = str(trusted_executor_workspace.resolve())
            if bootstrap_config_override:
                from xscientist.entrypoints import (
                    _rebind_bootstrapped_executor_workspace,
                )

                _rebind_bootstrapped_executor_workspace(resolved_workspace)
            else:
                os.environ["XSCIENTIST_WORKSPACE"] = resolved_workspace
        elif bootstrap_config_override:
            from xscientist.entrypoints import (
                _rebind_bootstrapped_executor_workspace,
            )

            _rebind_bootstrapped_executor_workspace(None)
    if not args.autopilot:
        return None
    if reuse_existing:
        args.bfts_config = str(source)
        return str(source)

    import yaml

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Autopilot BFTS configuration root must be a mapping")
    source_config_hash = canonical_content_hash(payload)

    limits = {
        "balanced": {
            "max_total_tokens": 800_000,
            "max_wall_time_seconds": 14_400,
            "stages": (8, 6, 6, 8),
            "steps": 3,
        },
        "discovery": {
            "max_total_tokens": 1_500_000,
            "max_wall_time_seconds": 28_800,
            "stages": (12, 10, 10, 12),
            "steps": 4,
        },
        "publication": {
            "max_total_tokens": 1_200_000,
            "max_wall_time_seconds": 28_800,
            "stages": (10, 8, 8, 10),
            "steps": 3,
        },
    }[str(args.autopilot)]
    max_project_tokens = getattr(args, "max_project_tokens", None)
    max_project_hours = getattr(args, "max_project_hours", None)
    max_cost_usd = getattr(args, "max_cost_usd", None)
    if max_project_tokens is not None:
        if int(max_project_tokens) <= 0:
            raise ValueError("--max-project-tokens must be greater than zero")
        limits["max_total_tokens"] = min(
            int(limits["max_total_tokens"]), int(max_project_tokens)
        )
    if max_project_hours is not None:
        if float(max_project_hours) <= 0:
            raise ValueError("--max-project-hours must be greater than zero")
        limits["max_wall_time_seconds"] = min(
            int(limits["max_wall_time_seconds"]),
            int(float(max_project_hours) * 3600),
        )
    if max_cost_usd is not None and float(max_cost_usd) <= 0:
        raise ValueError("--max-cost-usd must be greater than zero")

    execution = payload.setdefault("exec", {})
    if not isinstance(execution, dict):
        raise ValueError("Autopilot BFTS exec configuration must be a mapping")
    execution["require_isolation"] = True
    execution["network"] = "none"
    execution["allow_experiment_network"] = False

    budget = payload.setdefault("llm_budget", {})
    if not isinstance(budget, dict):
        raise ValueError("Autopilot BFTS llm_budget configuration must be a mapping")
    for key in ("max_total_tokens", "max_wall_time_seconds"):
        configured = budget.get(key)
        ceiling = int(limits[key])
        try:
            budget[key] = min(int(configured), ceiling) if configured else ceiling
        except (TypeError, ValueError):
            budget[key] = ceiling
    if max_cost_usd is not None:
        configured_cost = budget.get("max_cost_usd")
        ceiling_cost = float(max_cost_usd)
        try:
            budget["max_cost_usd"] = (
                min(float(configured_cost), ceiling_cost)
                if configured_cost is not None
                else ceiling_cost
            )
        except (TypeError, ValueError):
            budget["max_cost_usd"] = ceiling_cost

    agent = payload.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise ValueError("Autopilot BFTS agent configuration must be a mapping")
    configured_steps = agent.get("steps")
    try:
        agent["steps"] = min(int(configured_steps), int(limits["steps"]))
    except (TypeError, ValueError):
        agent["steps"] = int(limits["steps"])
    stages = agent.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("Autopilot BFTS agent.stages configuration must be a mapping")
    stage_keys = (
        "stage1_max_iters",
        "stage2_max_iters",
        "stage3_max_iters",
        "stage4_max_iters",
    )
    for key, ceiling in zip(stage_keys, limits["stages"]):
        configured = stages.get(key)
        try:
            stages[key] = min(int(configured), int(ceiling))
        except (TypeError, ValueError):
            stages[key] = int(ceiling)

    destination = _managed_project_file(
        args.project_dir, "00_config", "autopilot_bfts.yaml"
    )
    header = (
        "# XScientist finite autopilot derivative\n"
        f"# profile: {args.autopilot}\n"
        f"# source_config_hash: {source_config_hash}\n"
        "# isolation_required: true; experiment_network: none\n"
    )
    atomic_write_text(
        destination,
        header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    args.bfts_config = str(destination)
    return str(destination)


def _configure_autopilot_project_budget(args: argparse.Namespace) -> dict[str, object]:
    """Bind every model call and worker to one concurrency-safe budget ledger."""

    if not args.autopilot:
        return {"enabled": False}
    import yaml

    from ai_scientist.utils.llm_budget import configure_llm_budget

    payload = yaml.safe_load(Path(args.bfts_config).read_text(encoding="utf-8"))
    budget = dict((payload or {}).get("llm_budget") or {})
    state_path = _managed_project_file(args.project_dir, "04_logs", "llm_budget.json")
    manager = configure_llm_budget(
        max_total_tokens=budget.get("max_total_tokens"),
        max_cost_usd=budget.get("max_cost_usd"),
        max_wall_time_seconds=budget.get("max_wall_time_seconds"),
        prices_per_million=budget.get("prices_per_million") or {},
        state_path=state_path,
        reset=False,
    )
    snapshot = manager.snapshot()
    return {
        "enabled": True,
        "shared_across_project": True,
        "state": "04_logs/llm_budget.json",
        "limits": snapshot.get("limits", {}),
    }


def _write_autopilot_receipt(
    args: argparse.Namespace,
    *,
    checks: list[dict[str, object]],
) -> str | None:
    if not args.autopilot:
        return None
    topic_text = ""
    if args.topic:
        try:
            topic_text = Path(args.topic).read_text(encoding="utf-8")
        except OSError:
            topic_text = str(args.question or "")
    receipt: dict[str, object] = {
        "schema_version": "xscientist.autopilot-run.v1",
        "profile": args.autopilot,
        "topic_hash": canonical_content_hash(topic_text),
        "resolved_controls": {
            "target_venue": args.target_venue,
            "workflow_mode": args.workflow_mode,
            "num_ideas": args.num_ideas,
            "top_k_ideas": args.top_k_ideas,
            "parallel": args.parallel,
            "num_workers": args.num_workers,
            "improvement_rounds": args.improvement_rounds,
            "quality_preset": args.quality_preset,
            "integrity_forensics": args.integrity_forensics,
            "research_vcs": args.research_git,
            "checkpoint_policy": args.git_checkpoint_policy,
            "resume": args.resume,
            "data_contract": getattr(args, "_data_contract", None),
            "project_budget": getattr(args, "_project_budget", None),
            "bfts_config_hash": canonical_content_hash(
                Path(args.bfts_config).read_text(encoding="utf-8")
            ),
        },
        "scientific_boundaries": {
            "independent_verification_required_for_verified_claims": True,
            "internal_review_is_independent_verification": False,
            "retain_negative_and_failed_results": True,
            "respect_bfts_budget_and_stopping_conditions": True,
        },
        "preflight": checks,
    }
    receipt["receipt_hash"] = canonical_content_hash(receipt)
    destination = _managed_project_file(
        args.project_dir, "04_logs", "autopilot_run.json"
    )
    atomic_write_json(destination, receipt, ensure_ascii=False)
    return str(destination)


def _load_project_progress(project_dir: str | Path) -> dict[str, Any]:
    progress_path = _managed_project_file(
        project_dir, "04_logs", _PROJECT_PROGRESS_FILENAME
    )
    if not progress_path.is_file():
        return {}
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "--resume refused because saved project progress is damaged; start a "
            "new project or rerun without --resume"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "--resume refused because saved project progress must be a JSON object; "
            "start a new project or rerun without --resume"
        )
    has_resume_state = bool(payload.get("results") or payload.get("selected_indices"))
    if has_resume_state:
        recorded_ideas_hash = payload.get("ideas_hash")
        if not isinstance(recorded_ideas_hash, str) or not recorded_ideas_hash.strip():
            raise ValueError(
                "--resume refused legacy progress without an ideas hash; start a "
                "new project or rerun without --resume"
            )
        current_ideas_hash = _managed_project_ideas_hash(project_dir)
        if current_ideas_hash is None:
            raise ValueError(
                "--resume refused because saved progress cannot be matched to "
                "persisted ideas; start a new project or rerun without --resume"
            )
        if recorded_ideas_hash != current_ideas_hash:
            raise ValueError(
                "--resume refused because current ideas differ from the ideas bound "
                "to saved progress; start a new project or rerun without --resume"
            )
    return _resolve_project_progress_paths(payload, project_dir=project_dir)


_PROJECT_PROGRESS_PATH_FIELDS = frozenset(
    {
        "ara_manifest",
        "pipeline_manifest",
    }
)


def _is_project_progress_path_field(name: object) -> bool:
    rendered = str(name)
    return rendered in _PROJECT_PROGRESS_PATH_FIELDS or rendered.endswith(
        ("_dir", "_file", "_path")
    )


def _portable_project_progress_payload(
    payload: Any,
    *,
    project_dir: str | Path,
) -> Any:
    """Persist project-local paths without retaining host filesystem identity."""

    from ai_scientist.utils.privacy import portable_path, redact_sensitive_payload

    root = Path(project_dir).expanduser().resolve()

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[object, Any] = {}
            for key, nested in value.items():
                if (
                    _is_project_progress_path_field(key)
                    and isinstance(nested, (str, os.PathLike))
                    and str(nested).strip()
                    and "://" not in str(nested)
                ):
                    candidate = Path(nested).expanduser()
                    result[key] = (
                        portable_path(candidate, base=root)
                        if candidate.is_absolute()
                        else candidate.as_posix()
                    )
                else:
                    result[key] = transform(nested)
            return result
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, tuple):
            return [transform(item) for item in value]
        return value

    return redact_sensitive_payload(transform(payload))


def _resolve_project_progress_paths(
    payload: Any,
    *,
    project_dir: str | Path,
) -> Any:
    """Hydrate portable progress references for in-process resume consumers."""

    from ai_scientist.utils.privacy import resolve_portable_path

    root = Path(project_dir).expanduser().resolve()

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            result: dict[object, Any] = {}
            for key, nested in value.items():
                if (
                    _is_project_progress_path_field(key)
                    and isinstance(nested, str)
                    and nested.strip()
                    and "://" not in nested
                ):
                    resolved = resolve_portable_path(nested, base=root)
                    result[key] = str(resolved) if resolved is not None else nested
                else:
                    result[key] = transform(nested)
            return result
        if isinstance(value, list):
            return [transform(item) for item in value]
        return value

    return transform(payload)


def _save_project_progress(
    project_dir: str | Path,
    *,
    ideas_hash: str,
    results: list[dict[str, Any]],
    total: int,
    selected_indices: list[int],
) -> str:
    expected_ideas_hash = str(ideas_hash or "").strip()
    if not expected_ideas_hash:
        raise ValueError("cannot save project progress without the run ideas hash")
    current_ideas_hash = _managed_project_ideas_hash(project_dir)
    if current_ideas_hash is None:
        raise ValueError(
            "cannot save project progress before a canonical ideas file exists"
        )
    if current_ideas_hash != expected_ideas_hash:
        raise ValueError(
            "refused to save project progress because the managed ideas changed "
            "during this run"
        )
    latest: dict[int, dict[str, Any]] = {}
    for fallback_idx, result in enumerate(results):
        latest[int(result.get("idea_idx", fallback_idx))] = result
    ordered = [latest[index] for index in sorted(latest)]
    payload = _portable_project_progress_payload(
        {
            "schema_version": "xscientist.project-progress.v1",
            "ideas_hash": expected_ideas_hash,
            "completed": sum(
                1 for result in ordered if result.get("status") == "success"
            ),
            "attempted": len(ordered),
            "total": int(total),
            "selected_indices": list(selected_indices),
            "results": ordered,
        },
        project_dir=project_dir,
    )
    path = _managed_project_file(project_dir, "04_logs", _PROJECT_PROGRESS_FILENAME)
    atomic_write_json(path, payload, indent=2, ensure_ascii=False, default=str)
    return str(path)


_RESUME_INCOMPLETE_RESEARCH_STATUSES = {
    "blocked",
    "evidence_blocked",
    "failed",
    "incomplete",
    "quality_gate_failed",
    "rejected",
    "revision_needed",
}
_RESUME_COMPLETE_RESEARCH_STATUSES = {
    "exploratory_draft",
    "manuscript_draft",
    "submission_ready",
}
_OPEN_REVIEW_COUNT_FIELDS = (
    "active_issue_count",
    "blocker_count",
    "critic_active_issue_count",
    "critic_blocking_issue_count",
    "open_review_blocker_count",
    "review_active_issue_count",
    "review_blocker_count",
)
_OPEN_REVIEW_LIST_FIELDS = (
    "active_review_issues",
    "open_review_blockers",
    "review_blockers",
)


def _declared_review_count_blocks(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return True
    if not isinstance(value, int):
        return True
    return value != 0


def _review_contract_blocks_resume(
    project_dir: str | Path,
    result: dict[str, Any],
) -> bool:
    raw_path = str(result.get("review_state_file") or "").strip()
    if not raw_path:
        return False
    project_path = Path(project_dir).expanduser()
    lexical_root = Path(os.path.abspath(project_path))
    root = project_path.resolve()
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = lexical_root / candidate
    try:
        lexical = Path(os.path.abspath(candidate))
        try:
            relative = lexical.relative_to(lexical_root)
            cursor = lexical_root
        except ValueError:
            relative = lexical.relative_to(root)
            cursor = root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                return True
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        return True
    if not resolved.is_file() or resolved.is_symlink():
        return True
    review_state = _safe_load_json(resolved, default=None)
    if not isinstance(review_state, dict):
        return True
    for field in ("active_issue_records", "active_issues"):
        if field not in review_state:
            continue
        declared_issues = review_state[field]
        if not isinstance(declared_issues, list) or declared_issues:
            return True
    if "repair_metrics" in review_state:
        repair_metrics = review_state["repair_metrics"]
        if not isinstance(repair_metrics, dict):
            return True
    else:
        repair_metrics = {}
    for field in ("active_issue_count", "blocking_issue_count"):
        if _declared_review_count_blocks(repair_metrics.get(field)):
            return True
    if "lane_summaries" in review_state:
        lane_summaries = review_state["lane_summaries"]
        if not isinstance(lane_summaries, dict):
            return True
    else:
        lane_summaries = {}
    for lane in lane_summaries.values():
        if not isinstance(lane, dict):
            return True
        for field in ("active_issue_count", "blocking_issue_count"):
            if _declared_review_count_blocks(lane.get(field)):
                return True
    return False


def _result_is_resume_complete(
    result: dict[str, Any],
    *,
    project_dir: str | Path | None = None,
) -> bool:
    """Keep legacy successes resumable while honoring explicit paper blockers."""

    if result.get("status") != "success":
        return False
    for field in _OPEN_REVIEW_COUNT_FIELDS:
        if _declared_review_count_blocks(result.get(field)):
            return False
    for field in _OPEN_REVIEW_LIST_FIELDS:
        if field not in result:
            continue
        declared_issues = result[field]
        if not isinstance(declared_issues, list) or declared_issues:
            return False
    if project_dir is not None:
        if not _successful_paper_results(project_dir, [result]):
            return False
        if _review_contract_blocks_resume(project_dir, result):
            return False
    if "research_status" not in result:
        return True
    declared_research_status = result["research_status"]
    if not isinstance(declared_research_status, str):
        return False
    research_status = declared_research_status.strip().lower()
    if research_status in _RESUME_INCOMPLETE_RESEARCH_STATUSES:
        return False
    return research_status in _RESUME_COMPLETE_RESEARCH_STATUSES


def _latest_project_results(project_dir: str | Path) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for fallback_idx, result in enumerate(
        _load_project_progress(project_dir).get("results") or []
    ):
        if isinstance(result, dict):
            latest[int(result.get("idea_idx", fallback_idx))] = result
    return [latest[index] for index in sorted(latest)]


def _upsert_project_results(
    existing: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for fallback_idx, result in enumerate([*existing, *replacements]):
        latest[int(result.get("idea_idx", fallback_idx))] = result
    return [latest[index] for index in sorted(latest)]


def _resolve_resume_work(
    project_dir: str | Path,
    idea_indices: list[int],
    *,
    enabled: bool,
    authoritative_results: list[dict[str, Any]] | None = None,
) -> tuple[list[int], list[dict[str, Any]], dict[int, str]]:
    if not enabled:
        return list(idea_indices), [], {}
    progress = (
        {"results": authoritative_results}
        if authoritative_results is not None
        else _load_project_progress(project_dir)
    )
    latest: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, str] = {}
    for fallback_idx, result in enumerate(progress.get("results") or []):
        if not isinstance(result, dict):
            continue
        idea_idx = int(result.get("idea_idx", fallback_idx))
        latest[idea_idx] = result
        checkpoint = str(result.get("checkpoint_path") or "")
        if checkpoint and Path(checkpoint).expanduser().is_file():
            checkpoints[idea_idx] = checkpoint
    completed = {
        idea_idx
        for idea_idx, result in latest.items()
        if _result_is_resume_complete(result, project_dir=project_dir)
    }
    pending = [idea_idx for idea_idx in idea_indices if idea_idx not in completed]
    prior = [
        latest[idea_idx] for idea_idx in sorted(completed) if idea_idx in idea_indices
    ]
    return pending, prior, checkpoints


def _completed_resume_results(
    project_dir: str | Path,
) -> list[dict[str, Any]] | None:
    """Return the completed cached selection, or None when any work remains."""

    progress = _load_project_progress(project_dir)
    selected = progress.get("selected_indices")
    if not isinstance(selected, list) or not selected:
        return None
    latest: dict[int, dict[str, Any]] = {}
    for fallback_idx, result in enumerate(progress.get("results") or []):
        if not isinstance(result, dict):
            continue
        latest[int(result.get("idea_idx", fallback_idx))] = result
    try:
        selected_indices = [int(value) for value in selected]
    except (TypeError, ValueError):
        return None
    if not all(
        idea_idx in latest
        and _result_is_resume_complete(latest[idea_idx], project_dir=project_dir)
        for idea_idx in selected_indices
    ):
        return None
    return [latest[idea_idx] for idea_idx in selected_indices]


def _export_project_research_dag(
    args: argparse.Namespace,
    *,
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not getattr(args, "_research_git_active", False):
        return None
    from xscientist.research_git import ResearchGitError
    from xscientist.research_vcs import ResearchRepository

    ara_roots = [
        str(Path(str(result["ara_manifest"])).expanduser().parent)
        for result in results
        if result.get("ara_manifest")
    ]
    destination = (
        Path(args.output_root) / "views" / Path(args.project_dir).name / "research-dag"
    )
    try:
        exported = ResearchRepository(args.project_dir).export_dag(
            destination,
            ara_roots=ara_roots,
            disclose_summaries=True,
        )
        print(f"🕸️ 统一科研 DAG: {exported['html']}")
        return exported
    except (ResearchGitError, OSError, ValueError) as exc:
        _research_vcs_failure(args, "DAG export", exc)
        return None


def _local_research_git_question(topic: str | None) -> str | None:
    if not topic:
        return None
    path = Path(topic).expanduser()
    if not path.is_file():
        return f"# Research question\n\n{topic.strip()}\n"
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _initialize_local_research_git(args: argparse.Namespace) -> bool:
    if getattr(args, "research_git", "off") != "local":
        return False
    from xscientist.research_git import ResearchGitError, init_repository

    project_root = Path(args.project_dir).expanduser().resolve()
    try:
        if not (project_root / "research.yaml").is_file():
            result = init_repository(
                project_root,
                name=project_root.name,
                question=_local_research_git_question(args.topic),
                policy=args.git_checkpoint_policy,
                actor=args.git_user_name or "xscientist",
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
            )
            print(
                "🧭 XScientist 科研版本库已初始化: "
                f"commit={(result.commit or '')[:12]} policy={args.git_checkpoint_policy}"
            )
        else:
            from xscientist.research_git import load_repository_config

            load_repository_config(project_root)
            print("🧭 使用已有 XScientist 科研版本库；不会配置 remote 或自动 push。")
        return True
    except ResearchGitError as exc:
        if args.research_git_strict:
            raise RuntimeError(
                f"native Research VCS initialization failed: {exc}"
            ) from exc
        print(f"⚠️  科研版本库初始化失败，研究流程继续但不会自动 checkpoint: {exc}")
        return False


def _record_local_research_checkpoint(
    args: argparse.Namespace,
    *,
    stage: str,
    subject: str,
    summary: str,
    status: str = "completed",
    ara_paths: list[str] | None = None,
) -> None:
    if not getattr(args, "_research_git_active", False):
        return
    from xscientist.research_git import ResearchGitError, auto_checkpoint

    project_root = Path(args.project_dir).expanduser().resolve()
    relative_ara: list[str] = []
    for raw in ara_paths or []:
        try:
            relative_ara.append(
                Path(raw).expanduser().resolve().relative_to(project_root).as_posix()
            )
        except (ValueError, OSError):
            continue
    try:
        result = auto_checkpoint(
            project_root,
            stage=stage,
            subject=subject,
            summary=summary,
            status=status,
            ara_paths=relative_ara,
        )
        if result.committed:
            print(
                f"🧭 科研 checkpoint: {(result.commit or '')[:12]} "
                f"stage={stage} files={len(result.staged_paths)}"
            )
        elif result.reason:
            print(f"🧭 科研 checkpoint 跳过: {result.reason}")
    except ResearchGitError as exc:
        if args.research_git_strict:
            raise RuntimeError(f"native Research VCS checkpoint failed: {exc}") from exc
        print(f"⚠️  科研版本库 checkpoint 失败，研究产物已保留: {exc}")


def _successful_paper_results(
    project_dir: str | Path,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    project_path = Path(project_dir).expanduser()
    lexical_root = Path(os.path.abspath(project_path))
    root = project_path.resolve()
    successful: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") != "success":
            continue
        raw_pdf = str(result.get("pdf_path") or "").strip()
        if not raw_pdf:
            continue
        candidate = Path(raw_pdf).expanduser()
        if not candidate.is_absolute():
            candidate = lexical_root / candidate
        try:
            lexical = Path(os.path.abspath(candidate))
            try:
                relative = lexical.relative_to(lexical_root)
                cursor = lexical_root
            except ValueError:
                relative = lexical.relative_to(root)
                cursor = root
            for part in relative.parts:
                cursor /= part
                if cursor.is_symlink():
                    raise ValueError("paper path contains a symlink")
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if (
            resolved.is_file()
            and not resolved.is_symlink()
            and resolved.suffix.lower() == ".pdf"
        ):
            try:
                pdf_size = resolved.stat().st_size
                if pdf_size < 12:
                    continue
                with resolved.open("rb") as handle:
                    pdf_header = handle.read(5)
                    handle.seek(max(0, pdf_size - 1024))
                    pdf_trailer = handle.read()
            except OSError:
                continue
            if pdf_header == b"%PDF-" and pdf_trailer.rstrip().endswith(b"%%EOF"):
                successful.append(result)
    return successful


def _print_project_completion(
    project_dir: str | Path,
    results: list[dict[str, Any]],
) -> None:
    successful_papers = _successful_paper_results(project_dir, results)
    print("\n" + "=" * 80)
    if len(successful_papers) < len(results):
        print(
            "⚠️ 本次流程部分完成："
            f"{len(successful_papers)}/{len(results)} 个想法生成了可检查的 PDF 论文产物。"
        )
    else:
        print(
            "✅ 本次流程完成："
            f"{len(successful_papers)}/{len(results)} 个想法生成了可检查的 PDF 论文产物。"
        )
    print(f"📁 项目目录: {project_dir}")
    for result in successful_papers:
        print(f"📄 PDF: {result.get('pdf_path')}")
    print("=" * 80)


def _publication_gate_failure(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
) -> str | None:
    successful_papers = _successful_paper_results(args.project_dir, results)
    if not successful_papers:
        return "没有成功生成可检查的 PDF 论文产物，本次流程未完成"
    constraints_active = (
        args.require_quality_gate
        or args.min_submission_priority is not None
        or args.max_submission_blockers is not None
    )

    def passes_current_submission_gate(result: dict[str, Any]) -> bool:
        if result.get("submission_acceptance_passed") is not True:
            return False
        if args.require_quality_gate and result.get("quality_gate_passed") is not True:
            return False
        if args.min_submission_priority is not None:
            priority = result.get("submission_priority_score")
            if (
                isinstance(priority, bool)
                or not isinstance(priority, (int, float))
                or not math.isfinite(priority)
                or priority < args.min_submission_priority
            ):
                return False
        if args.max_submission_blockers is not None:
            blockers = result.get("blocker_count")
            if (
                isinstance(blockers, bool)
                or not isinstance(blockers, int)
                or blockers < 0
                or blockers > args.max_submission_blockers
            ):
                return False
        return True

    if constraints_active and not any(
        passes_current_submission_gate(result) for result in successful_papers
    ):
        return "没有任何项目论文通过当前本地投稿准备度门禁，任务视为失败"
    if (args.strict_writing_guardrails or args.high_quality_mode) and not any(
        result_passed_writeup_guardrails(result) for result in successful_papers
    ):
        return "严格写作守护开启，但没有任何论文通过写作守护检查，任务视为失败"
    return None


def _experiment_checkpoint_status(executed_results: list[dict[str, Any]]) -> str:
    """Describe this execution transition without borrowing prior successes."""

    return (
        "completed"
        if any(
            (result.get("registry_status") or result.get("status"))
            in {"success", "completed"}
            for result in executed_results
        )
        else "failed"
    )


def _finalize_local_research_handoff(
    args: argparse.Namespace,
    *,
    results: list[dict[str, Any]],
    ara_paths: list[str],
) -> str | None:
    """Checkpoint the terminal paper state before a strict gate can fail the run."""

    failure = _publication_gate_failure(args, results)
    _record_local_research_checkpoint(
        args,
        stage="paper",
        subject="freeze project summary and shortlist",
        summary=(
            "Record the project-level paper summary, shortlist, quality gates, "
            "and final handoff state."
        ),
        status="failed" if failure else "completed",
        ara_paths=ara_paths,
    )
    _export_project_research_dag(args, results=results)
    return failure


def _research_vcs_failure(
    args: argparse.Namespace, action: str, exc: Exception
) -> None:
    if getattr(args, "research_git_strict", False):
        raise RuntimeError(f"native Research VCS {action} failed: {exc}") from exc
    print(f"⚠️  Research VCS {action} 失败，研究产物已保留: {exc}")


def _record_local_research_planning_objects(
    args: argparse.Namespace,
    *,
    idea_cards: list[dict],
) -> None:
    if not getattr(args, "_research_git_active", False) or not idea_cards:
        return
    from xscientist.research_git import (
        ResearchGitError,
        load_repository_config_at_ref,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    project_root = Path(args.project_dir).expanduser().resolve()
    try:
        lifecycle = ResearchLifecycle(project_root)
        lead_plan = load_contract_artifact(project_root, "research_plan", default={})
        preregistration = load_contract_artifact(
            project_root, "preregistration", default={}
        )
        planning = lifecycle.planning(
            hypothesis=idea_cards[0],
            plan=lead_plan if isinstance(lead_plan, dict) else {},
            preregistration=(
                preregistration if isinstance(preregistration, dict) else None
            ),
            commit=False,
        )
        object_ids = {
            "hypotheses": {0: planning["hypothesis"].object_id},
            "plans": {0: planning["plan"].object_id},
            "attempts": {},
            "evidence": {},
            "claims": {},
            "gates": {},
            "manuscripts": {},
        }
        if planning["preregistration"] is not None:
            object_ids["preregistration"] = planning["preregistration"].object_id
        for index, idea_card in enumerate(idea_cards[1:], start=1):
            candidate = lifecycle.repository.record(
                "hypothesis",
                idea_card,
                state="draft",
            )
            object_ids["hypotheses"][index] = candidate.object_id
            candidate_plan = lifecycle.repository.record(
                "research_plan",
                {
                    "idea_idx": index,
                    "summary": str(
                        idea_card.get("title")
                        or idea_card.get("core_hypothesis")
                        or f"candidate idea {index}"
                    ),
                    "status": "candidate",
                    "source": "idea_card",
                },
                state="draft",
                relations=[{"type": "depends_on", "target": candidate.object_id}],
            )
            object_ids["plans"][index] = candidate_plan.object_id
        challenge = (
            lead_plan.get("socratic_challenge")
            if isinstance(lead_plan, dict)
            and isinstance(lead_plan.get("socratic_challenge"), dict)
            else {}
        )
        rivals = [
            item
            for item in challenge.get("rival_hypotheses") or []
            if isinstance(item, dict) and str(item.get("statement") or "").strip()
        ]
        tests = [
            item
            for item in challenge.get("discriminating_tests") or []
            if isinstance(item, dict) and str(item.get("design") or "").strip()
        ]
        if rivals and tests:
            from xscientist.research_strategy import (
                rank_experiment_candidates,
                save_discriminating_prediction,
                save_hypothesis_portfolio,
            )

            rival_objects = []
            rival_id_map: dict[str, str] = {}
            for rival in rivals:
                recorded_rival = lifecycle.repository.record(
                    "hypothesis",
                    {
                        "statement": str(rival["statement"]),
                        "falsifier": str(
                            rival.get("discriminating_prediction")
                            or "A predeclared discriminating test favors another hypothesis."
                        ),
                        "source": str(rival.get("source") or "socratic_challenge"),
                        "rival_class": rival.get("class"),
                    },
                    state="draft",
                )
                rival_objects.append(recorded_rival)
                rival_id_map[str(rival.get("rival_id") or recorded_rival.object_id)] = (
                    recorded_rival.object_id
                )
            portfolio = save_hypothesis_portfolio(
                project_root,
                question=str(
                    challenge.get("primary_hypothesis")
                    or idea_cards[0].get("title")
                    or "Which competing explanation survives the next test?"
                ),
                primary_id=planning["hypothesis"].object_id,
                alternative_ids=[item.object_id for item in rival_objects],
                commit=False,
            )["object"]
            hypothesis_ids = [
                planning["hypothesis"].object_id,
                *[item.object_id for item in rival_objects],
            ]
            prediction_ids: list[str] = []
            candidates: list[dict[str, object]] = []
            rival_by_object_id = {
                object_id: rival
                for rival, object_id in zip(rivals, rival_id_map.values())
            }
            for test in tests:
                condition = str(test["design"])
                targeted = {
                    rival_id_map[value]
                    for value in test.get("targets") or []
                    if value in rival_id_map
                }
                predictions: dict[str, str] = {}
                for hypothesis_id in hypothesis_ids:
                    if hypothesis_id == planning["hypothesis"].object_id:
                        outcome = "the primary hypothesis passes its preregistered success rule"
                    elif hypothesis_id in targeted:
                        outcome = str(
                            rival_by_object_id[hypothesis_id].get(
                                "discriminating_prediction"
                            )
                            or "the targeted rival is favored"
                        )
                    else:
                        outcome = "this test is not decisive for this rival"
                    predictions[hypothesis_id] = outcome
                    prediction = save_discriminating_prediction(
                        project_root,
                        portfolio_id=portfolio.object_id,
                        hypothesis_id=hypothesis_id,
                        when=condition,
                        expected_outcome=outcome,
                        distinguishes_from=[
                            value for value in hypothesis_ids if value != hypothesis_id
                        ],
                        falsifier="the observed outcome is closer to a competing prediction",
                        commit=False,
                    )["object"]
                    prediction_ids.append(prediction.object_id)
                test_id = str(test.get("test_id") or f"test-{len(candidates) + 1}")
                candidates.append(
                    {
                        "candidate_id": test_id,
                        "summary": condition,
                        "condition": condition,
                        "interventions": (
                            [str(challenge.get("proposed_mechanism"))]
                            if "mechanism" in test_id
                            and challenge.get("proposed_mechanism")
                            else []
                        ),
                        "predictions": predictions,
                        "novelty": 2,
                        "impact": 3,
                        "transfer_value": 3 if "boundary" in test_id else 2,
                        "cost": 2,
                        "risk": 1,
                        "redundancy": 0,
                    }
                )
            priority_result = rank_experiment_candidates(
                project_root,
                portfolio_id=portfolio.object_id,
                candidates=candidates,
                commit=False,
            )
            priority = priority_result["object"]
            object_ids["strategy"] = {
                "portfolio": portfolio.object_id,
                "predictions": prediction_ids,
                "priority": priority.object_id,
                "designs": [item.object_id for item in priority_result["related"]],
                "status": "proposed_not_executed",
            }
        args._research_vcs_ids = object_ids
        print(
            "🧬 Research VCS 已记录 "
            f"{len(object_ids['hypotheses'])} 个假设及其规划/预注册关系"
        )
    except (ResearchGitError, OSError, ValueError) as exc:
        _research_vcs_failure(args, "planning record", exc)


def _compact_research_result(result: dict) -> dict:
    scalar_keys = (
        "idea_idx",
        "status",
        "runtime_status",
        "registry_status",
        "terminal_receipt_hash",
        "stage",
        "failure_stage",
        "error",
        "resumable",
        "quality_score",
        "rigor_score",
        "claim_support_score",
        "quality_gate_passed",
        "submission_priority_score",
        "submission_priority_tier",
        "submission_acceptance_passed",
        "guardrail_blocking",
        "ara_node_count",
        "ara_claim_coverage_score",
        "ara_claim_coverage_severity",
        "workflow_mode",
        "template_profile",
        "template_capability",
        "execution_policy",
        "admission_post_execution_status",
        "research_status",
    )
    return {
        key: result.get(key)
        for key in scalar_keys
        if isinstance(result.get(key), (str, int, float, bool))
        or result.get(key) is None
    }


def _required_experiment_admission_enabled(project_dir: str | Path) -> bool:
    """Return whether an existing Research VCS requires experiment admission."""

    try:
        from xscientist.research_git import ResearchGitError
        from xscientist.research_vcs import ResearchRepository

        return ResearchRepository(project_dir).experiment_admission_mode() == "required"
    except (ResearchGitError, OSError, ValueError):
        return False


def _activate_existing_required_research_vcs(args: argparse.Namespace) -> bool:
    """Activate an existing required repository or fail closed on downgrade."""

    project_root = Path(args.project_dir).expanduser().resolve()
    config_path = project_root / "research.yaml"
    metadata_path = project_root / ".xscientist"
    git_control_path = project_root / ".git"
    if not any(
        path.exists() or path.is_symlink()
        for path in (config_path, metadata_path, git_control_path)
    ):
        return False
    try:
        from xscientist.research_git import (
            ResearchGitError,
            load_repository_config_at_ref,
            repository_experiment_admission_mode,
        )
        from xscientist.research_vcs import ResearchRepository

        if config_path.exists() or config_path.is_symlink():
            required = ResearchRepository(project_root).experiment_admission_mode() == (
                "required"
            )
        else:
            try:
                load_repository_config_at_ref(project_root, "HEAD")
            except (ResearchGitError, OSError, ValueError):
                if not (metadata_path.exists() or metadata_path.is_symlink()):
                    return False
                raise
            required = repository_experiment_admission_mode(project_root) == "required"
    except (ResearchGitError, OSError, ValueError) as exc:
        raise RuntimeError(
            "existing Research VCS configuration is unreadable; refusing to "
            "continue without admission enforcement"
        ) from exc
    if not required:
        return False
    if getattr(args, "research_git", "off") != "local":
        raise RuntimeError(
            "existing Research VCS requires experiment admission and cannot be "
            "resumed with --research-vcs off"
        )
    args._research_git_active = True
    return True


_PROJECT_BUDGET_CLI_OPTIONS = {
    "--max-project-tokens": "max_project_tokens",
    "--max-project-hours": "max_project_hours",
    "--max-cost-usd": "max_cost_usd",
}


def _project_cli_option_was_explicit(arguments: list[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in arguments)


def _autopilot_derivative_profile(config_text: str) -> str | None:
    prefix = "# profile:"
    for line in config_text.splitlines()[:8]:
        if line.startswith(prefix):
            profile = line[len(prefix) :].strip()
            return profile or None
    return None


def _validate_required_resume_autopilot_overrides(
    args: argparse.Namespace,
    *,
    config_text: str,
) -> None:
    """Reject explicit ceilings that would mutate an admitted derivative."""

    explicit = set(getattr(args, "_project_budget_cli_overrides", ()) or ())
    if not explicit:
        return
    import yaml

    try:
        payload = yaml.safe_load(config_text)
    except yaml.YAMLError as exc:
        raise RuntimeError(
            "required experiment admission resume configuration is unreadable"
        ) from exc
    budget = (payload or {}).get("llm_budget") if isinstance(payload, Mapping) else None
    if not isinstance(budget, Mapping):
        raise RuntimeError(
            "required experiment admission resume configuration has no budget"
        )

    comparisons: list[tuple[str, str, float]] = []
    if "max_project_tokens" in explicit:
        requested_tokens = int(args.max_project_tokens)
        if requested_tokens <= 0:
            raise ValueError("--max-project-tokens must be greater than zero")
        comparisons.append(
            ("max_project_tokens", "max_total_tokens", float(requested_tokens))
        )
    if "max_project_hours" in explicit:
        requested_hours = float(args.max_project_hours)
        if requested_hours <= 0:
            raise ValueError("--max-project-hours must be greater than zero")
        comparisons.append(
            (
                "max_project_hours",
                "max_wall_time_seconds",
                float(int(requested_hours * 3600)),
            )
        )
    if "max_cost_usd" in explicit:
        requested_cost = float(args.max_cost_usd)
        if requested_cost <= 0:
            raise ValueError("--max-cost-usd must be greater than zero")
        comparisons.append(("max_cost_usd", "max_cost_usd", requested_cost))

    for _attribute, config_field, requested_ceiling in comparisons:
        configured = budget.get(config_field)
        try:
            configured_ceiling = float(configured)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "required experiment admission resume budget controls would change "
                "the immutable configuration"
            ) from exc
        if requested_ceiling < configured_ceiling:
            raise RuntimeError(
                "required experiment admission resume budget controls would change "
                "the immutable configuration"
            )


def _required_admission_resume_configuration_path(
    args: argparse.Namespace,
    *,
    expected_configuration_hash: Any,
) -> Path:
    """Resolve exact admitted config bytes without rewriting a persisted derivative."""

    from ai_scientist.resources import resolve_bfts_config_path

    if (
        not isinstance(expected_configuration_hash, str)
        or not expected_configuration_hash
    ):
        raise RuntimeError(
            "unfinished required experiment admission batch has no configuration hash"
        )
    explicit_config = bool(getattr(args, "_project_bfts_config_cli_override", False))
    candidates: list[Path] = []
    if bool(getattr(args, "autopilot", None)) and not explicit_config:
        persisted = _managed_project_file(
            args.project_dir,
            "00_config",
            "autopilot_bfts.yaml",
        )
        if persisted.is_file():
            candidates.append(persisted)
    try:
        requested = resolve_bfts_config_path(args.bfts_config)
    except (OSError, ValueError) as exc:
        if explicit_config:
            raise RuntimeError(
                "required experiment admission resume configuration is unavailable"
            ) from exc
    else:
        if requested not in candidates:
            candidates.append(requested)

    for candidate in candidates:
        try:
            config_text = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if canonical_content_hash(config_text) != expected_configuration_hash:
            continue
        if bool(getattr(args, "autopilot", None)):
            if _autopilot_derivative_profile(config_text) != str(args.autopilot):
                continue
            _validate_required_resume_autopilot_overrides(
                args,
                config_text=config_text,
            )
        return candidate
    raise RuntimeError(
        "required experiment admission resume configuration does not match the "
        "immutable receipt"
    )


def _validate_required_admission_resume_execution_inputs(
    args: argparse.Namespace,
    scope: Mapping[str, Any],
) -> None:
    """Recheck the exact config and project budget before provider/model work."""

    from ai_scientist.resources import resolve_bfts_config_path

    try:
        configuration_text = resolve_bfts_config_path(args.bfts_config).read_text(
            encoding="utf-8"
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "required experiment admission resume configuration is unavailable"
        ) from exc
    if canonical_content_hash(configuration_text) != scope.get("configuration_hash"):
        raise RuntimeError(
            "required experiment admission resume configuration changed after "
            "reconciliation"
        )
    project_budget = getattr(args, "_project_budget", None)
    actual_budget_hash = (
        canonical_content_hash(project_budget)
        if isinstance(project_budget, Mapping)
        else None
    )
    if actual_budget_hash != scope.get("budget_hash"):
        raise RuntimeError(
            "required experiment admission resume budget does not match the immutable "
            "receipt"
        )


def _required_admission_resume_inputs(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Derive the immutable scope needed for recovery without mutable caches."""

    from xscientist.research_lifecycle import ResearchLifecycle

    requested_indices = (
        [int(value.strip()) for value in str(args.idea_indices).split(",")]
        if getattr(args, "idea_indices", None)
        else None
    )
    ideas_hash = _managed_project_ideas_hash(args.project_dir)
    selected_indices = requested_indices or [0]

    lifecycle = ResearchLifecycle(args.project_dir)
    batches = _experiment_admission_start_batches_at_head(lifecycle)
    if not batches:
        from ai_scientist.resources import resolve_bfts_config_path

        configuration_hash = canonical_content_hash(
            resolve_bfts_config_path(args.bfts_config).read_text(encoding="utf-8")
        )
        budget_hash = (
            canonical_content_hash({"enabled": False}) if not args.autopilot else None
        )
        return {
            "selected_indices": selected_indices,
            "ideas_hash": ideas_hash,
            "configuration_hash": configuration_hash,
            "budget_hash": budget_hash,
        }

    receipts = batches[-1]["receipts"]
    selected_indices = sorted(int(index) for index in receipts)
    if not all(
        isinstance(receipt.get("execution_scope"), Mapping)
        for receipt in receipts.values()
    ):
        raise RuntimeError("required experiment admission batch has an invalid scope")
    shared_scope: dict[str, Any] = {}
    for field in ("ideas_hash", "configuration_hash", "budget_hash"):
        values = {
            (receipt.get("execution_scope") or {}).get(field)
            for receipt in receipts.values()
            if isinstance(receipt.get("execution_scope"), Mapping)
        }
        if len(values) != 1:
            raise RuntimeError(
                f"required experiment admission batch has inconsistent {field}"
            )
        shared_scope[field] = next(iter(values))
    if (
        ideas_hash is not None
        and shared_scope["ideas_hash"] is not None
        and ideas_hash != shared_scope["ideas_hash"]
    ):
        raise RuntimeError(
            "persisted ideas do not match the required experiment admission batch"
        )
    return {"selected_indices": selected_indices, **shared_scope}


def _install_experiment_admission_batch_state(
    args: argparse.Namespace,
    receipts: Mapping[int, Mapping[str, Any]],
) -> None:
    """Install reconstructed immutable batch identities on the project runner."""

    object_ids = dict(getattr(args, "_research_vcs_ids", {}) or {})
    object_ids["plans"] = {
        int(index): str(receipt["plan_id"]) for index, receipt in receipts.items()
    }
    object_ids["admission_gates"] = {
        int(index): str(receipt["admission_gate_id"])
        for index, receipt in receipts.items()
    }
    object_ids["admission_starts"] = {
        int(index): str(receipt["start_object_id"])
        for index, receipt in receipts.items()
    }
    preregistrations = {
        str(receipt.get("preregistration_id") or "")
        for receipt in receipts.values()
        if receipt.get("preregistration_id")
    }
    if len(preregistrations) > 1:
        raise ValueError(
            "reconstructed experiment admission batch has ambiguous preregistrations"
        )
    if preregistrations:
        object_ids["preregistration"] = next(iter(preregistrations))
    else:
        object_ids.pop("preregistration", None)
    args._research_vcs_ids = object_ids
    args._experiment_admission_preflights = {
        int(index): dict(receipt) for index, receipt in receipts.items()
    }


def _experiment_admission_start_batches_at_head(lifecycle) -> list[dict[str, Any]]:
    """Return exact committed start batches visible on the first-parent history."""

    from xscientist.research_git import (
        ResearchGitError,
        _run_git,
        list_research_objects_at_ref,
    )

    repository_path = lifecycle.repository.path
    head = str(lifecycle.repository.status().get("head") or "").strip()
    if not head:
        return []
    visible = list_research_objects_at_ref(repository_path, head)
    starts = [
        item
        for item in visible
        if item.get("kind") == "context_snapshot"
        and isinstance(item.get("payload"), Mapping)
        and (item.get("payload") or {}).get("protocol_kind")
        == "xscientist.experiment-admission-start.v1"
    ]
    grouped: dict[str, set[str]] = {}
    for start in starts:
        start_object_id = str(start.get("object_id") or "")
        path = f".xscientist/objects/context_snapshot/{start_object_id}.json"
        history = _run_git(
            repository_path,
            ["rev-list", "--first-parent", "--reverse", head, "--", path],
        ).stdout.splitlines()
        if len(history) != 1 or re.fullmatch(r"[0-9a-f]{40,64}", history[0]) is None:
            raise ResearchGitError(
                "experiment admission start has no unique first-parent checkpoint"
            )
        grouped.setdefault(history[0], set()).add(start_object_id)
    history_order = {
        commit: index
        for index, commit in enumerate(
            _run_git(
                repository_path,
                ["rev-list", "--first-parent", "--reverse", head],
            ).stdout.splitlines()
        )
    }
    batches: list[dict[str, Any]] = []
    for start_checkpoint, visible_start_ids in grouped.items():
        receipts = lifecycle.reconstruct_experiment_admission_batch(start_checkpoint)
        reconstructed_ids = {
            str(receipt.get("start_object_id") or "") for receipt in receipts.values()
        }
        if reconstructed_ids != visible_start_ids:
            raise ResearchGitError(
                "experiment admission start checkpoint visibility is incomplete"
            )
        batches.append(
            {
                "start_checkpoint": start_checkpoint,
                "receipts": receipts,
                "history_order": history_order.get(start_checkpoint, -1),
            }
        )
    return sorted(batches, key=lambda item: int(item["history_order"]))


def _experiment_attempt_start_ids(attempt: Mapping[str, Any]) -> set[str]:
    return {
        str(relation.get("target") or "")
        for relation in attempt.get("relations") or []
        if isinstance(relation, Mapping)
        and relation.get("type") == "consumes"
        and relation.get("role") == "experiment_admission_start"
    }


def _validated_committed_experiment_admission_batch(
    lifecycle,
    receipts: Mapping[int, Mapping[str, Any]],
    *,
    terminal_commit: str,
) -> dict[int, dict[str, Any]]:
    """Prove and return every attempt in one terminal-only checkpoint."""

    from xscientist.research_git import (
        ResearchGitError,
        list_research_objects_at_ref,
        show_checkpoint,
    )

    committed = list_research_objects_at_ref(lifecycle.repository.path, terminal_commit)
    expected_starts = {
        str(receipt.get("start_object_id") or ""): int(index)
        for index, receipt in receipts.items()
    }
    attempts: dict[int, dict[str, Any]] = {}
    for attempt in committed:
        if attempt.get("kind") != "experiment_attempt":
            continue
        consumed = _experiment_attempt_start_ids(attempt)
        matched = consumed & set(expected_starts)
        if not matched:
            continue
        if len(consumed) != 1 or len(matched) != 1:
            raise ResearchGitError(
                "experiment admission terminal attempt has ambiguous start bindings"
            )
        start_object_id = next(iter(matched))
        idea_index = expected_starts[start_object_id]
        if idea_index in attempts:
            raise ResearchGitError(
                "experiment admission start has multiple committed terminal attempts"
            )
        embedded = (attempt.get("payload") or {}).get("admission_preflight_receipt")
        if not isinstance(embedded, Mapping) or dict(embedded) != dict(
            receipts[idea_index]
        ):
            raise ResearchGitError(
                "experiment admission terminal attempt receipt does not match its start"
            )
        attempts[idea_index] = dict(attempt)
    if set(attempts) != set(receipts):
        raise ResearchGitError(
            "experiment admission terminal checkpoint does not consume its full batch"
        )
    expected_paths = {
        f".xscientist/objects/experiment_attempt/{attempt['object_id']}.json"
        for attempt in attempts.values()
    }
    shown = show_checkpoint(lifecycle.repository.path, terminal_commit)
    checkpoint = shown.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    changed_paths = checkpoint.get("changed_paths")
    states = [str(attempt.get("state") or "") for attempt in attempts.values()]
    expected_status = (
        states[0]
        if len(states) == 1
        else ("completed" if "completed" in states else "failed")
    )
    if (
        shown.get("checkpoint_hash_valid") is not True
        or shown.get("commit") != terminal_commit
        or checkpoint.get("stage") != "experiment_admission_terminal"
        or checkpoint.get("status") != expected_status
        or not isinstance(changed_paths, list)
        or len(changed_paths) != len(expected_paths)
        or len(set(changed_paths)) != len(changed_paths)
        or set(changed_paths) != expected_paths
    ):
        raise ResearchGitError(
            "experiment admission terminal checkpoint does not exactly match its batch"
        )
    return {index: attempts[index] for index in sorted(attempts)}


def _discover_committed_experiment_admission_batch(
    lifecycle,
    receipts: Mapping[int, Mapping[str, Any]],
) -> tuple[str, dict[int, dict[str, Any]]] | None:
    """Find one exact terminal batch and repair its immutable retention refs."""

    from xscientist.research_git import (
        ResearchGitError,
        _load_experiment_admission_terminal_completion_ref,
        _parse_git_worktree_head_commits_porcelain_z,
        _publish_experiment_admission_terminal_completion_refs,
        _run_git,
        _run_git_bounded,
        show_checkpoint,
    )

    repository_path = lifecycle.repository.path
    ordered_receipts = [receipts[index] for index in sorted(receipts)]
    start_object_ids = [
        str(receipt.get("start_object_id") or "") for receipt in ordered_receipts
    ]
    if not start_object_ids or "" in start_object_ids:
        raise ResearchGitError(
            "experiment admission batch has an invalid start identity"
        )
    completion_refs = [
        _load_experiment_admission_terminal_completion_ref(
            repository_path,
            start_object_id=start_object_id,
        )
        for start_object_id in start_object_ids
    ]
    published = {commit for commit in completion_refs if commit is not None}
    if len(published) > 1:
        raise ResearchGitError("experiment admission batch completion refs disagree")

    terminal_commit = next(iter(published), None)
    if terminal_commit is None:
        worktrees = _run_git_bounded(
            repository_path,
            ["worktree", "list", "--porcelain", "-z"],
            max_output_bytes=_MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_BYTES,
        )
        worktree_heads = _parse_git_worktree_head_commits_porcelain_z(worktrees.stdout)
        histories: list[set[str]] = []
        for start_object_id in start_object_ids:
            history = _run_git_bounded(
                repository_path,
                [
                    "log",
                    "--full-history",
                    "--format=%H",
                    "--no-show-signature",
                    f"--max-count={_MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_COMMITS + 1}",
                    f"-S{start_object_id}",
                    "--all",
                    *worktree_heads,
                    "--",
                    ".xscientist/objects/experiment_attempt",
                ],
                max_output_bytes=_MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_BYTES,
            ).stdout.splitlines()
            if len(history) > _MAX_EXPERIMENT_ADMISSION_TERMINAL_SCAN_COMMITS:
                raise ResearchGitError(
                    "experiment admission terminal history exceeds the bounded "
                    "recovery limit"
                )
            if any(
                re.fullmatch(r"[0-9a-f]{40,64}", commit) is None for commit in history
            ):
                raise ResearchGitError(
                    "experiment admission terminal history is invalid"
                )
            histories.append(set(history))
        candidates = set.intersection(*histories) if histories else set()
        exact_candidates: list[str] = []
        for candidate in sorted(candidates):
            shown = show_checkpoint(repository_path, candidate)
            checkpoint = shown.get("checkpoint")
            checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
            if checkpoint.get("stage") != "experiment_admission_terminal":
                continue
            _validated_committed_experiment_admission_batch(
                lifecycle,
                receipts,
                terminal_commit=candidate,
            )
            exact_candidates.append(candidate)
        if len(exact_candidates) > 1:
            raise ResearchGitError(
                "experiment admission batch has multiple terminal checkpoints"
            )
        if not exact_candidates:
            return None
        terminal_commit = exact_candidates[0]

    start_checkpoints = {
        str(receipt.get("start_checkpoint") or "").strip()
        for receipt in ordered_receipts
    }
    if len(start_checkpoints) != 1 or "" in start_checkpoints:
        raise ResearchGitError(
            "experiment admission batch has an ambiguous start checkpoint"
        )
    ancestry = _run_git(
        repository_path,
        [
            "merge-base",
            "--is-ancestor",
            next(iter(start_checkpoints)),
            terminal_commit,
        ],
        check=False,
    )
    if ancestry.returncode != 0:
        raise ResearchGitError(
            "experiment admission terminal checkpoint is not a descendant of its start"
        )
    attempts = _validated_committed_experiment_admission_batch(
        lifecycle,
        receipts,
        terminal_commit=terminal_commit,
    )
    _publish_experiment_admission_terminal_completion_refs(
        repository_path,
        start_object_ids=start_object_ids,
        commit=terminal_commit,
    )
    return terminal_commit, attempts


def _project_result_from_committed_admission_attempt(
    repository_path: str | Path,
    attempt: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    terminal_commit: str,
    runtime_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rebuild progress from final Research VCS truth plus a verified outbox."""

    from xscientist.research_git import ResearchGitError

    payload = attempt.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    result: dict[str, Any] = {}
    if (
        runtime_record is not None
        and (runtime_record.get("claim") or {}).get("schema")
        == "xscientist.experiment-admission-execution-claim.v2"
    ):
        from xscientist.experiment_admission_runtime import (
            _RESULT_PATH_FIELDS,
            _validate_relative_path,
        )

        claim = runtime_record["claim"]
        idea_index = int((receipt.get("execution_scope") or {}).get("idea_index"))
        if (
            claim.get("start_object_id") != receipt.get("start_object_id")
            or claim.get("receipt_hash") != receipt.get("receipt_hash")
            or claim.get("idea_index") != idea_index
        ):
            raise ResearchGitError(
                "committed experiment attempt conflicts with its runtime claim"
            )
        bfts = runtime_record.get("bfts_terminal")
        worker = runtime_record.get("worker_terminal")
        bindings = {
            "admission_runtime_claim_hash": claim.get("claim_hash"),
        }
        if isinstance(bfts, Mapping):
            bindings["admission_bfts_terminal_hash"] = bfts.get("terminal_hash")
        if isinstance(worker, Mapping):
            bindings["admission_worker_terminal_hash"] = worker.get("terminal_hash")
        for field, expected in bindings.items():
            if payload.get(field) != expected:
                raise ResearchGitError(
                    "committed experiment attempt conflicts with its runtime outbox"
                )
        if isinstance(worker, Mapping):
            result.update(dict(worker.get("result") or {}))
            root = Path(repository_path).expanduser().resolve()
            for field in _RESULT_PATH_FIELDS & set(result):
                relative = _validate_relative_path(result[field], label=field)
                result[field] = str(root / relative)
    authoritative_pipeline_status = str(payload.get("pipeline_status") or "failed")
    authoritative_registry_status = str(
        payload.get("status") or attempt.get("state") or "failed"
    )
    idea_index = int((receipt.get("execution_scope") or {}).get("idea_index"))
    if result and (
        result.get("idea_idx") != idea_index
        or result.get("status") != authoritative_pipeline_status
        or result.get("registry_status") != authoritative_registry_status
    ):
        raise ResearchGitError(
            "committed experiment attempt outcome conflicts with its runtime outbox"
        )
    for field, value in _compact_research_result(dict(payload)).items():
        if field not in {"idea_idx", "status", "registry_status"}:
            result[field] = value
    result.update(
        {
            "idea_idx": idea_index,
            "status": authoritative_pipeline_status,
            "registry_status": authoritative_registry_status,
            "admission_terminal_commit": terminal_commit,
        }
    )
    return result


def _reconcile_required_experiment_admission_runtime(
    args: argparse.Namespace,
    *,
    selected_indices: list[int],
    ideas_hash: str | None,
    configuration_hash: str | None,
    budget_hash: str | None,
) -> dict[str, Any]:
    """Recover required-admission batches before issuing any new start."""

    empty = {
        "active": False,
        "committed_results": [],
        "recovered_results": [],
        "execute_indices": list(selected_indices),
        "batch_indices": list(selected_indices),
        "pending_batch": False,
    }
    if not getattr(args, "_research_git_active", False):
        return empty
    from xscientist.experiment_admission_runtime import (
        admission_runtime_records,
        result_from_runtime_record,
    )
    from xscientist.research_git import (
        ResearchGitError,
        _load_experiment_admission_terminal_completion_ref,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    lifecycle = ResearchLifecycle(args.project_dir)
    if lifecycle.repository.experiment_admission_mode() != "required":
        return empty
    batches = _experiment_admission_start_batches_at_head(lifecycle)
    runtime_records = admission_runtime_records(lifecycle.repository.path)
    records_by_start = {
        str(record.get("start_object_id") or ""): record for record in runtime_records
    }
    expected_scope = {
        "ideas_hash": ideas_hash,
        "configuration_hash": configuration_hash,
        "budget_hash": budget_hash,
    }
    committed_results: dict[int, dict[str, Any]] = {}
    pending_batches: list[dict[str, Any]] = []
    for batch in batches:
        receipts = batch["receipts"]
        discovered = _discover_committed_experiment_admission_batch(
            lifecycle,
            receipts,
        )
        if discovered is not None:
            terminal_commit, attempts = discovered
            for idea_index, receipt in receipts.items():
                scope = receipt.get("execution_scope")
                if not isinstance(scope, Mapping) or any(
                    scope.get(field) != expected
                    for field, expected in expected_scope.items()
                ):
                    continue
                committed_results[int(idea_index)] = (
                    _project_result_from_committed_admission_attempt(
                        lifecycle.repository.path,
                        attempts[int(idea_index)],
                        receipt,
                        terminal_commit=terminal_commit,
                        runtime_record=records_by_start.get(
                            str(receipt["start_object_id"])
                        ),
                    )
                )
            continue
        pending_batches.append(batch)
    if len(pending_batches) > 1:
        raise ResearchGitError(
            "multiple unresolved experiment admission start batches require manual recovery"
        )
    if not pending_batches:
        remaining = [
            int(index)
            for index in selected_indices
            if int(index) not in committed_results
        ]
        return {
            **empty,
            "active": True,
            "committed_results": [
                committed_results[index] for index in sorted(committed_results)
            ],
            "execute_indices": remaining,
            "batch_indices": sorted(
                {int(index) for index in selected_indices} | set(committed_results)
            ),
        }

    pending = pending_batches[0]
    receipts = pending["receipts"]
    for receipt in receipts.values():
        scope = receipt.get("execution_scope")
        if not isinstance(scope, Mapping) or any(
            scope.get(field) != expected for field, expected in expected_scope.items()
        ):
            raise ResearchGitError(
                "unresolved experiment admission batch does not match this run"
            )
    recovered_results: dict[int, dict[str, Any]] = {}
    execute_indices: list[int] = []
    for idea_index, receipt in receipts.items():
        start_object_id = str(receipt["start_object_id"])
        record = records_by_start.get(start_object_id)
        if record is None:
            execute_indices.append(int(idea_index))
            continue
        claim = record.get("claim")
        if not isinstance(claim, Mapping) or claim.get("schema") != (
            "xscientist.experiment-admission-execution-claim.v2"
        ):
            raise ResearchGitError(
                "legacy experiment admission claim is in doubt; manual recovery is required"
            )
        if (
            claim.get("receipt_hash") != receipt.get("receipt_hash")
            or claim.get("start_checkpoint_commit") != pending["start_checkpoint"]
            or claim.get("idea_index") != idea_index
        ):
            raise ResearchGitError(
                "experiment admission runtime claim does not match its committed batch"
            )
        if record.get("owner_state") in {"alive", "unknown"}:
            raise ResearchGitError(
                "experiment admission runtime owner is still alive or unknown; "
                "the whole batch remains blocked"
            )
        recovered = result_from_runtime_record(lifecycle.repository.path, record)
        if recovered is None:
            raise ResearchGitError(
                "experiment admission claimed start has no recoverable terminal state"
            )
        recovered_results[int(idea_index)] = recovered[0]

    _install_experiment_admission_batch_state(args, receipts)
    batch_indices = sorted(int(index) for index in receipts)
    if len(recovered_results) == len(receipts):
        _record_local_research_attempt_objects(
            args,
            results=[recovered_results[index] for index in sorted(recovered_results)],
        )
        completion_refs = {
            _load_experiment_admission_terminal_completion_ref(
                lifecycle.repository.path,
                start_object_id=str(receipt["start_object_id"]),
            )
            for receipt in receipts.values()
        }
        if len(completion_refs) != 1 or None in completion_refs:
            raise ResearchGitError(
                "recovered experiment admission batch has no atomic terminal refs"
            )
        terminal_commit = next(iter(completion_refs))
        attempts = _validated_committed_experiment_admission_batch(
            lifecycle,
            receipts,
            terminal_commit=terminal_commit,
        )
        for idea_index, receipt in receipts.items():
            committed_results[int(idea_index)] = (
                _project_result_from_committed_admission_attempt(
                    lifecycle.repository.path,
                    attempts[int(idea_index)],
                    receipt,
                    terminal_commit=terminal_commit,
                    runtime_record=records_by_start.get(
                        str(receipt["start_object_id"])
                    ),
                )
            )
        return {
            **empty,
            "active": True,
            "committed_results": [
                committed_results[index] for index in sorted(committed_results)
            ],
            "execute_indices": [],
            "batch_indices": batch_indices,
        }

    args._experiment_admission_reuse_receipts = {
        int(index): dict(receipt) for index, receipt in receipts.items()
    }
    args._experiment_admission_reuse_execute_indices = sorted(execute_indices)
    return {
        **empty,
        "active": True,
        "committed_results": [
            committed_results[index] for index in sorted(committed_results)
        ],
        "recovered_results": [
            recovered_results[index] for index in sorted(recovered_results)
        ],
        "execute_indices": sorted(execute_indices),
        "batch_indices": batch_indices,
        "pending_batch": True,
    }


def _sha256_values(value: object) -> list[str]:
    rows: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            rows.update(_sha256_values(child))
    elif isinstance(value, list):
        for child in value:
            rows.update(_sha256_values(child))
    elif isinstance(value, str) and len(value) == 71 and value.startswith("sha256:"):
        if all(character in "0123456789abcdef" for character in value[7:]):
            rows.add(value)
    return sorted(rows)


def _research_attempt_provenance(
    lifecycle,
    *,
    result: dict,
    preregistration_id: str | None,
) -> dict:
    from ai_scientist.protocol.hashing import content_hash, hash_manifest
    from xscientist.research_git import capture_environment_receipt

    environment = capture_environment_receipt(lifecycle.repository.path)
    provenance: dict[str, object] = {"environment_hash": environment["content_hash"]}
    dependency_lock_hashes = sorted(
        {
            str(item["hash"])
            for item in environment.get("dependency_locks") or []
            if item.get("hash")
        }
    )
    if dependency_lock_hashes:
        provenance["dependency_lock_hashes"] = dependency_lock_hashes
    if preregistration_id:
        registration = lifecycle.repository.get(preregistration_id)
        dataset_hashes = _sha256_values(registration.get("payload") or {})
        if dataset_hashes:
            provenance["dataset_hashes"] = dataset_hashes
    raw_seeds = result.get("seeds")
    if isinstance(raw_seeds, list):
        seeds = sorted({seed for seed in raw_seeds if isinstance(seed, int)})
        if seeds:
            provenance["seeds"] = seeds
    elif isinstance(result.get("seed"), int):
        provenance["seeds"] = [int(result["seed"])]
    manifest_path = result.get("ara_manifest")
    if manifest_path:
        manifest = _safe_load_json(manifest_path, default=None)
        if isinstance(manifest, dict):
            provenance["ara_manifest_hash"] = hash_manifest(manifest)
        graph = _safe_load_json(
            Path(str(manifest_path)).expanduser().parent / "exploration_graph.json",
            default=None,
        )
        if isinstance(graph, dict):
            provenance["ara_exploration_graph_hash"] = content_hash(graph)
            context_hashes = sorted(
                {
                    str(ref)
                    for node in graph.get("nodes") or []
                    if isinstance(node, dict)
                    for ref in node.get("context_pack_refs") or []
                    if isinstance(ref, str) and ref.startswith("sha256:")
                }
            )
            if context_hashes:
                provenance["context_hashes"] = context_hashes
    return provenance


def _preflight_local_research_experiment_admission(
    args: argparse.Namespace,
    *,
    idea_indices: list[int],
    ideas_hash: str | None = None,
    configuration_hash: str | None = None,
    budget_hash: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Authorize each selected experiment before creating a worker or BFTS run."""

    if not getattr(args, "_research_git_active", False):
        return {}
    from xscientist.research_git import (
        ResearchGitError,
        capture_environment_receipt,
        _fsync_directory,
        _repository_lock,
        create_checkpoint,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    lifecycle = ResearchLifecycle(args.project_dir)
    if lifecycle.repository.experiment_admission_mode() != "required":
        return {}
    reuse_receipts = getattr(args, "_experiment_admission_reuse_receipts", None)
    if isinstance(reuse_receipts, Mapping) and reuse_receipts:
        execute_indices = getattr(
            args, "_experiment_admission_reuse_execute_indices", []
        )
        if set(idea_indices) != set(execute_indices):
            raise ResearchGitError(
                "recovered experiment admission batch execution selection changed"
            )
        receipts = {
            int(index): lifecycle.validate_finalized_experiment_admission_start(
                receipt,
                allow_descendant_head=True,
                allow_terminal_state=True,
            )
            for index, receipt in reuse_receipts.items()
        }
        expected_scope = {
            "ideas_hash": ideas_hash,
            "configuration_hash": configuration_hash,
            "budget_hash": budget_hash,
        }
        if any(
            not isinstance(receipt.get("execution_scope"), Mapping)
            or any(
                expected is not None
                and receipt["execution_scope"].get(field) != expected
                for field, expected in expected_scope.items()
            )
            for receipt in receipts.values()
        ):
            raise ResearchGitError("recovered experiment admission batch scope changed")
        _install_experiment_admission_batch_state(args, receipts)
        print(
            "🛡️ Research VCS 复用未完成实验准入批次: "
            f"batch={len(receipts)} execute={len(execute_indices)}"
        )
        return receipts
    if not idea_indices:
        return {}
    object_ids = getattr(args, "_research_vcs_ids", {})
    plan_ids = object_ids.get("plans") or {}
    preregistration_id = object_ids.get("preregistration")
    receipts: dict[int, dict[str, Any]] = {}
    admission_gate_ids: dict[int, str] = {}
    admission_start_ids: dict[int, str] = {}
    with _repository_lock(lifecycle.repository.path):
        research_vcs_head = str(lifecycle.repository.status().get("head") or "").strip()
        if not research_vcs_head:
            raise ResearchGitError(
                "required experiment admission needs a committed Research VCS HEAD"
            )
        environment_hash = str(
            capture_environment_receipt(lifecycle.repository.path)["content_hash"]
        )
        requests: dict[int, dict[str, Any]] = {}
        for idea_index in idea_indices:
            if type(idea_index) is not int or idea_index < 0:
                raise ResearchGitError(
                    "required experiment admission has an invalid idea index"
                )
            if idea_index in requests:
                raise ResearchGitError(
                    f"required experiment admission repeats idea #{idea_index}"
                )
            plan_id = plan_ids.get(idea_index) or plan_ids.get(str(idea_index))
            if not plan_id:
                raise ResearchGitError(
                    f"required experiment admission has no plan for idea #{idea_index}"
                )
            execution_scope: dict[str, Any] = {
                "idea_index": idea_index,
                "study_phase": "exploratory",
                "environment_hash": environment_hash,
            }
            if ideas_hash:
                execution_scope["ideas_hash"] = ideas_hash
            if configuration_hash:
                execution_scope["configuration_hash"] = configuration_hash
            if budget_hash:
                execution_scope["budget_hash"] = budget_hash
            request = {
                "plan_id": str(plan_id),
                "preregistration_id": (
                    str(preregistration_id)
                    if idea_index == 0 and preregistration_id
                    else None
                ),
                "execution_scope": execution_scope,
                "expected_head": research_vcs_head,
            }
            preflight = lifecycle.preflight_experiment_admission(**request)
            if preflight.get("status") != "authorized" or not preflight.get(
                "admission_gate_id"
            ):
                raise ResearchGitError(
                    f"required experiment admission did not authorize idea #{idea_index}"
                )
            requests[idea_index] = request

        issued_receipts: dict[int, dict[str, Any]] = {}
        start_paths: list[str] = []
        created_start_paths: list[Path] = []
        starts_committed = False
        try:
            for idea_index, request in requests.items():
                issued = lifecycle.issue_experiment_admission_start(**request)
                if not issued["start"].created:
                    raise ResearchGitError(
                        "experiment admission start object unexpectedly pre-existed"
                    )
                created_start_paths.append(issued["start"].path)
                receipt = issued["receipt"]
                gate_id = str(receipt.get("admission_gate_id") or "")
                if receipt.get("status") != "authorized" or not gate_id:
                    raise ResearchGitError(
                        f"required experiment admission did not authorize idea #{idea_index}"
                    )
                issued_receipts[idea_index] = receipt
                admission_gate_ids[idea_index] = gate_id
                admission_start_ids[idea_index] = str(issued["start"].object_id)
                start_paths.append(
                    issued["start"]
                    .path.resolve()
                    .relative_to(lifecycle.repository.path)
                    .as_posix()
                )

            checkpoint = create_checkpoint(
                lifecycle.repository.path,
                stage="experiment_admission_start",
                subject=f"commit {len(start_paths)} experiment admission starts",
                summary=(
                    "Commit the complete selected runner batch before any experiment "
                    "workspace, worker, or BFTS execution is created."
                ),
                status="locked",
                only_paths=sorted(start_paths),
            )
            if not checkpoint.committed or not checkpoint.commit:
                raise ResearchGitError(
                    "required experiment admission start checkpoint was not committed"
                )
            starts_committed = True
            for idea_index, issued_receipt in issued_receipts.items():
                receipts[idea_index] = lifecycle.finalize_experiment_admission_start(
                    issued_receipt,
                    start_checkpoint=checkpoint.commit,
                )
        except BaseException:
            if not starts_committed:
                touched_parents: set[Path] = set()
                for path in created_start_paths:
                    path.unlink(missing_ok=True)
                    touched_parents.add(path.parent)
                for parent in touched_parents:
                    _fsync_directory(parent)
            raise

    object_ids["admission_gates"] = admission_gate_ids
    object_ids["admission_starts"] = admission_start_ids
    args._research_vcs_ids = object_ids
    args._experiment_admission_preflights = receipts
    print(
        "🛡️ Research VCS 实验准入预检通过: "
        f"{len(receipts)} 个实验，snapshot={research_vcs_head[:12]} "
        f"start={str(checkpoint.commit)[:12]}"
    )
    return receipts


def _revalidate_experiment_admission_preflight(
    project_dir: str | Path,
    receipt: Mapping[str, Any] | None,
    *,
    bfts_config_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Recheck a parent-issued receipt immediately before entering BFTS."""

    from xscientist.research_git import (
        ResearchGitError,
        load_repository_config_at_ref,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    project_root = Path(project_dir).expanduser().resolve()
    research_config = project_root / "research.yaml"
    try:
        lifecycle = ResearchLifecycle(project_root)
    except ResearchGitError:
        if receipt is None and not (
            research_config.is_file() or research_config.is_symlink()
        ):
            try:
                load_repository_config_at_ref(project_root, "HEAD")
            except ResearchGitError:
                return None
        raise
    if receipt is None:
        if lifecycle.repository.experiment_admission_mode() == "required":
            raise ResearchGitError(
                "required experiment admission preflight receipt is missing"
            )
        return None
    validated = lifecycle.validate_finalized_experiment_admission_start(receipt)
    scope = dict(validated["execution_scope"])
    approved_configuration_hash = scope.get("configuration_hash")
    if approved_configuration_hash is not None:
        from ai_scientist.resources import resolve_bfts_config_path

        try:
            resolved_config = resolve_bfts_config_path(bfts_config_path)
            current_configuration_hash = canonical_content_hash(
                resolved_config.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise ResearchGitError(
                "experiment admission could not verify the approved BFTS configuration"
            ) from exc
        if current_configuration_hash != approved_configuration_hash:
            raise ResearchGitError(
                "experiment admission BFTS configuration changed after parent preflight"
            )
    code_commit = scope.pop("code_commit", None)
    current = lifecycle.preflight_experiment_admission(
        plan_id=str(validated.get("plan_id") or ""),
        preregistration_id=(
            str(validated["preregistration_id"])
            if validated.get("preregistration_id")
            else None
        ),
        admission_gate_id=str(validated.get("admission_gate_id") or ""),
        execution_scope=scope,
        code_commit=str(code_commit) if code_commit else None,
        expected_head=str(validated["repository_head"]),
    )
    for field in (
        "repository_id",
        "plan_id",
        "plan_hash",
        "hypothesis_id",
        "hypothesis_hash",
        "preregistration_id",
        "preregistration_hash",
        "admission_gate_id",
        "admission_gate_hash",
        "execution_scope_hash",
    ):
        if current.get(field) != validated.get(field):
            raise ResearchGitError(
                f"experiment admission preflight changed at BFTS boundary: {field}"
            )
    return validated


@contextmanager
def _experiment_admission_execution_lease(
    project_dir: str | Path,
    receipt: Mapping[str, Any] | None,
    *,
    bfts_config_path: str | Path | None = None,
    experiment_dir: str | Path | None = None,
    idea_key: str | None = None,
):
    """Keep supported Research VCS writers off the admitted execution snapshot."""

    from xscientist.research_git import (
        ResearchGitError,
        _repository_read_lock,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    try:
        lifecycle = ResearchLifecycle(project_dir)
    except ResearchGitError:
        _revalidate_experiment_admission_preflight(
            project_dir,
            receipt,
            bfts_config_path=bfts_config_path,
        )
        yield None
        return
    if lifecycle.repository.experiment_admission_mode() != "required":
        _revalidate_experiment_admission_preflight(
            project_dir,
            receipt,
            bfts_config_path=bfts_config_path,
        )
        yield None
        return
    with _repository_read_lock(lifecycle.repository.path):
        validated = _revalidate_experiment_admission_preflight(
            project_dir,
            receipt,
            bfts_config_path=bfts_config_path,
        )
        if validated is None:
            raise ResearchGitError(
                "required experiment admission did not produce a finalized receipt"
            )
        if experiment_dir is None and idea_key is None:
            # Compatibility for direct lifecycle callers that predate the
            # project worker contract. New project execution always supplies
            # both fields and therefore always publishes claim v2.
            from xscientist.research_git import _claim_experiment_admission_start

            _claim_experiment_admission_start(
                lifecycle.repository.path,
                start_object_id=str(validated["start_object_id"]),
                receipt_hash=str(validated["receipt_hash"]),
            )
            yield None
            return
        from xscientist.experiment_admission_runtime import (
            claim_experiment_admission_start_v2,
        )

        claim = claim_experiment_admission_start_v2(
            lifecycle.repository.path,
            validated,
            experiment_dir=experiment_dir or lifecycle.repository.path,
            idea_key=(
                str(idea_key)
                if idea_key
                else f"idea-{int((validated.get('execution_scope') or {}).get('idea_index', 0))}"
            ),
        )
        yield claim


def _required_terminal_batch_checkpoint(
    repository_path: str | Path,
    *,
    attempts: list[Any],
    attempt_paths: list[str],
    start_object_ids: list[str],
    status: str,
) -> str | None:
    """Return the one exact committed terminal batch, or ``None`` if uncommitted."""

    from xscientist.research_git import (
        ResearchGitError,
        _load_experiment_admission_terminal_completion_ref,
        _run_git,
        list_research_objects_at_ref,
        show_checkpoint,
    )

    if (
        not attempts
        or len(attempts) != len(attempt_paths)
        or len(attempts) != len(start_object_ids)
    ):
        raise ResearchGitError("required experiment terminal batch is incomplete")
    if (
        len(set(attempt_paths)) != len(attempt_paths)
        or len({str(attempt.object_id) for attempt in attempts}) != len(attempts)
        or len(set(start_object_ids)) != len(start_object_ids)
    ):
        raise ResearchGitError(
            "required experiment terminal batch has duplicate attempt identities"
        )

    completion_commits = [
        _load_experiment_admission_terminal_completion_ref(
            repository_path,
            start_object_id=start_object_id,
        )
        for start_object_id in start_object_ids
    ]
    expected_paths = set(attempt_paths)
    for attempt, relative_path in zip(attempts, attempt_paths):
        if relative_path != (
            f".xscientist/objects/experiment_attempt/{attempt.object_id}.json"
        ):
            raise ResearchGitError(
                "required experiment terminal object path does not match its identity"
            )
    introductions: set[str] = set()
    committed_count = 0
    for attempt, relative_path in zip(attempts, attempt_paths):
        history = _run_git(
            Path(repository_path),
            [
                "log",
                "--all",
                "--full-history",
                "--topo-order",
                "--no-renames",
                "--max-count=2",
                "--format=%H",
                "--",
                relative_path,
            ],
        ).stdout.splitlines()
        if len(history) > 1:
            raise ResearchGitError(
                "required experiment terminal object has ambiguous Git history"
            )
        if not history:
            continue
        committed_count += 1
        introductions.add(history[0])

    if committed_count == 0:
        if any(commit is not None for commit in completion_commits):
            raise ResearchGitError(
                "required experiment terminal completion ref has no exact outcome"
            )
        return None
    if committed_count != len(attempts):
        raise ResearchGitError(
            "required experiment terminal batch has partially committed outcomes"
        )
    if len(introductions) != 1:
        raise ResearchGitError(
            "required experiment terminal batch does not share one checkpoint"
        )

    introduction = next(iter(introductions))
    shown = show_checkpoint(repository_path, introduction)
    checkpoint = shown.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
    changed_paths = checkpoint.get("changed_paths")
    if (
        shown.get("checkpoint_hash_valid") is not True
        or shown.get("commit") != introduction
        or checkpoint.get("stage") != "experiment_admission_terminal"
        or checkpoint.get("status") != status
        or not isinstance(changed_paths, list)
        or len(changed_paths) != len(expected_paths)
        or len(set(changed_paths)) != len(changed_paths)
        or set(changed_paths) != expected_paths
    ):
        raise ResearchGitError(
            "required experiment terminal checkpoint does not exactly match the batch"
        )
    committed_at_introduction = {
        str(item.get("object_id") or ""): item
        for item in list_research_objects_at_ref(repository_path, introduction)
    }
    for attempt in attempts:
        committed = committed_at_introduction.get(str(attempt.object_id))
        if (
            not isinstance(committed, Mapping)
            or committed.get("kind") != "experiment_attempt"
            or committed.get("content_hash") != attempt.object_hash
        ):
            raise ResearchGitError(
                "required experiment terminal checkpoint contains a foreign outcome"
            )
    if any(
        commit is not None and commit != introduction for commit in completion_commits
    ):
        raise ResearchGitError(
            "required experiment terminal batch completion ref does not match its "
            "unique checkpoint"
        )
    return introduction


def _record_local_research_attempt_objects(
    args: argparse.Namespace,
    *,
    results: list[dict],
) -> None:
    if not getattr(args, "_research_git_active", False):
        return
    from xscientist.research_git import (
        ResearchGitError,
        _fsync_directory,
        _load_experiment_admission_execution_claim,
        _load_experiment_admission_terminal_reservation,
        _publish_experiment_admission_terminal_completion_refs,
        _repository_lock,
        create_checkpoint,
        list_research_objects_at_ref,
        show_checkpoint,
    )
    from xscientist.research_lifecycle import ResearchLifecycle

    try:
        lifecycle = ResearchLifecycle(args.project_dir)
        object_ids = getattr(args, "_research_vcs_ids", {})
        preregistration_id = object_ids.get("preregistration")
        plan_ids = object_ids.get("plans") or {}
        gate_ids = object_ids.get("admission_gates") or {}
        preflights = getattr(args, "_experiment_admission_preflights", {}) or {}
        attempt_ids = dict(object_ids.get("attempts") or {})
        attempt_paths: list[str] = []
        terminal_start_ids: list[str] = []
        created_attempt_paths: list[Path] = []
        recorded_attempts: list[Any] = []
        created_terminal_reservation_paths: list[Path] = []
        prepared_attempts: list[tuple[int, dict[str, Any]]] = []
        with _repository_lock(lifecycle.repository.path):
            admission_required = (
                lifecycle.repository.experiment_admission_mode() == "required"
            )
            admission_starts = object_ids.get("admission_starts") or {}
            if admission_required and not all(
                isinstance(item, Mapping)
                for item in (plan_ids, gate_ids, preflights, admission_starts)
            ):
                raise ResearchGitError(
                    "required experiment admission batch mappings are invalid"
                )
            if (
                admission_required
                and not results
                and not preflights
                and not admission_starts
            ):
                return
            seen_idea_indices: set[int] = set()
            for fallback_index, result in enumerate(results):
                raw_idea_index = result.get("idea_idx", fallback_index)
                if admission_required:
                    if (
                        "idea_idx" not in result
                        or type(raw_idea_index) is not int
                        or raw_idea_index < 0
                    ):
                        raise ResearchGitError(
                            "required experiment result has an invalid idea_idx"
                        )
                    idea_index = raw_idea_index
                else:
                    idea_index = int(raw_idea_index)
                if admission_required and idea_index in seen_idea_indices:
                    raise ResearchGitError(
                        f"experiment result batch repeats idea #{idea_index}"
                    )
                seen_idea_indices.add(idea_index)
                preflight = preflights.get(idea_index) or preflights.get(
                    str(idea_index)
                )
                durable_result = dict(result)
                runtime_binding: dict[str, Any] = {}
                provenance: dict[str, Any] | None = None
                if admission_required and isinstance(preflight, Mapping):
                    start_object_id = str(preflight.get("start_object_id") or "")
                    execution_claim = _load_experiment_admission_execution_claim(
                        lifecycle.repository.path,
                        start_object_id=start_object_id,
                        receipt_hash=str(preflight.get("receipt_hash") or ""),
                    )
                    if (
                        isinstance(execution_claim, Mapping)
                        and execution_claim.get("schema")
                        == "xscientist.experiment-admission-execution-claim.v2"
                    ):
                        from xscientist.experiment_admission_runtime import (
                            execution_claim_owner_state,
                            load_bfts_terminal,
                            load_worker_terminal,
                            validate_result_against_runtime_record,
                        )

                        bfts_terminal = load_bfts_terminal(
                            lifecycle.repository.path,
                            start_object_id=start_object_id,
                            claim=execution_claim,
                        )
                        worker_terminal = load_worker_terminal(
                            lifecycle.repository.path,
                            start_object_id=start_object_id,
                            claim=execution_claim,
                        )
                        durable_result, provenance = (
                            validate_result_against_runtime_record(
                                lifecycle.repository.path,
                                {
                                    "claim": execution_claim,
                                    "bfts_terminal": bfts_terminal,
                                    "worker_terminal": worker_terminal,
                                    "owner_state": execution_claim_owner_state(
                                        execution_claim
                                    ),
                                },
                                result,
                            )
                        )
                        runtime_binding = {
                            "admission_runtime_claim_hash": execution_claim[
                                "claim_hash"
                            ],
                            "admission_bfts_terminal_hash": (
                                bfts_terminal.get("terminal_hash")
                                if bfts_terminal is not None
                                else None
                            ),
                            "admission_worker_terminal_hash": (
                                worker_terminal.get("terminal_hash")
                                if worker_terminal is not None
                                else None
                            ),
                        }
                payload = _compact_research_result(durable_result)
                payload.update(runtime_binding)
                payload["pipeline_status"] = str(
                    durable_result.get("status") or "failed"
                )
                payload["status"] = str(
                    durable_result.get("registry_status")
                    or durable_result.get("status")
                    or "failed"
                )
                if isinstance(preflight, Mapping):
                    for field, value in (
                        preflight.get("execution_scope") or {}
                    ).items():
                        if field != "code_commit":
                            payload.setdefault(str(field), value)
                if provenance is None:
                    provenance = _research_attempt_provenance(
                        lifecycle,
                        result=durable_result,
                        preregistration_id=(
                            preregistration_id if idea_index == 0 else None
                        ),
                    )
                if isinstance(preflight, Mapping):
                    preflight_scope = preflight.get("execution_scope")
                    if isinstance(preflight_scope, Mapping) and preflight_scope.get(
                        "code_commit"
                    ):
                        provenance["code_commit"] = preflight_scope["code_commit"]
                prepared_attempts.append(
                    (
                        idea_index,
                        lifecycle.validate_experiment_attempt(
                            payload,
                            preregistration_id=(
                                preregistration_id if idea_index == 0 else None
                            ),
                            plan_id=plan_ids.get(idea_index)
                            or plan_ids.get(str(idea_index)),
                            admission_gate_id=gate_ids.get(idea_index)
                            or gate_ids.get(str(idea_index)),
                            admission_preflight=preflight,
                            provenance=provenance,
                        ),
                    )
                )

            if admission_required:

                def normalized_indices(
                    values: Mapping[Any, Any], *, label: str
                ) -> set[int]:
                    normalized: set[int] = set()
                    for raw_index in values:
                        if isinstance(raw_index, bool):
                            raise ResearchGitError(
                                f"required experiment {label} has an invalid idea index"
                            )
                        try:
                            index = int(raw_index)
                        except (TypeError, ValueError) as exc:
                            raise ResearchGitError(
                                f"required experiment {label} has an invalid idea index"
                            ) from exc
                        if index < 0 or str(index) != str(raw_index):
                            raise ResearchGitError(
                                f"required experiment {label} has an invalid idea index"
                            )
                        if index in normalized:
                            raise ResearchGitError(
                                f"required experiment {label} has an ambiguous idea index"
                            )
                        normalized.add(index)
                    return normalized

                start_indices = normalized_indices(
                    admission_starts, label="admission starts"
                )
                preflight_indices = normalized_indices(
                    preflights, label="admission preflights"
                )
                start_checkpoint_heads = {
                    str(receipt.get("start_checkpoint") or "").strip()
                    for receipt in preflights.values()
                    if isinstance(receipt, Mapping)
                }
                if len(start_checkpoint_heads) != 1 or "" in start_checkpoint_heads:
                    raise ResearchGitError(
                        "required experiment terminal batch is not based on one "
                        "dedicated admission-start checkpoint"
                    )
                start_checkpoint_head = next(iter(start_checkpoint_heads))
                shown = show_checkpoint(
                    lifecycle.repository.path,
                    start_checkpoint_head,
                )
                start_checkpoint = shown.get("checkpoint")
                start_checkpoint = (
                    start_checkpoint if isinstance(start_checkpoint, Mapping) else {}
                )
                changed_paths = start_checkpoint.get("changed_paths")
                parent_commit = str(start_checkpoint.get("parent_commit") or "").strip()
                if (
                    shown.get("checkpoint_hash_valid") is not True
                    or start_checkpoint.get("stage") != "experiment_admission_start"
                    or not parent_commit
                    or not isinstance(changed_paths, list)
                    or not changed_paths
                    or len(set(changed_paths)) != len(changed_paths)
                ):
                    raise ResearchGitError(
                        "required experiment terminal batch is not based on one "
                        "dedicated admission-start checkpoint"
                    )
                committed_now = {
                    str(item["object_id"]): item
                    for item in list_research_objects_at_ref(
                        lifecycle.repository.path, start_checkpoint_head
                    )
                }
                committed_before = {
                    str(item["object_id"]): item
                    for item in list_research_objects_at_ref(
                        lifecycle.repository.path, parent_commit
                    )
                }
                committed_starts: dict[int, Mapping[str, Any]] = {}
                for logical_path in changed_paths:
                    parts = str(logical_path).split("/")
                    if (
                        len(parts) != 4
                        or parts[:3] != [".xscientist", "objects", "context_snapshot"]
                        or not parts[3].endswith(".json")
                    ):
                        raise ResearchGitError(
                            "required experiment admission-start checkpoint contains "
                            "non-start material"
                        )
                    start_object_id = parts[3][:-5]
                    start_object = committed_now.get(start_object_id)
                    start_payload = (
                        start_object.get("payload")
                        if isinstance(start_object, Mapping)
                        else None
                    )
                    start_scope = (
                        start_payload.get("execution_scope")
                        if isinstance(start_payload, Mapping)
                        else None
                    )
                    committed_idea_index = (
                        start_scope.get("idea_index")
                        if isinstance(start_scope, Mapping)
                        else None
                    )
                    if (
                        start_object_id in committed_before
                        or not isinstance(start_object, Mapping)
                        or start_object.get("kind") != "context_snapshot"
                        or start_object.get("state") != "locked"
                        or not isinstance(start_payload, Mapping)
                        or start_payload.get("protocol_kind")
                        != "xscientist.experiment-admission-start.v1"
                        or start_payload.get("status") != "issued"
                        or type(committed_idea_index) is not int
                        or committed_idea_index < 0
                        or committed_idea_index in committed_starts
                    ):
                        raise ResearchGitError(
                            "required experiment admission-start checkpoint is "
                            "ambiguous or invalid"
                        )
                    committed_starts[committed_idea_index] = start_object
                committed_start_indices = set(committed_starts)
                if (
                    not committed_start_indices
                    or start_indices != committed_start_indices
                    or preflight_indices != committed_start_indices
                    or seen_idea_indices != committed_start_indices
                ):
                    raise ResearchGitError(
                        "required experiment result batch must exactly match every "
                        "committed admission start"
                    )
                for idea_index in sorted(start_indices):
                    receipt = preflights.get(idea_index) or preflights.get(
                        str(idea_index)
                    )
                    start_id = admission_starts.get(
                        idea_index, admission_starts.get(str(idea_index))
                    )
                    if (
                        not isinstance(receipt, Mapping)
                        or str(receipt.get("start_object_id") or "")
                        != str(start_id or "")
                        or str(start_id or "")
                        != str(committed_starts[idea_index].get("object_id") or "")
                        or receipt.get("start_checkpoint") != start_checkpoint_head
                    ):
                        raise ResearchGitError(
                            "required experiment preflight does not match its admitted "
                            f"start for idea #{idea_index}"
                        )

            terminal_committed = False
            try:
                for idea_index, prepared in prepared_attempts:
                    reservation_preexisted = False
                    if admission_required:
                        start_id = admission_starts.get(
                            idea_index, admission_starts.get(str(idea_index))
                        )
                        reservation_preexisted = (
                            _load_experiment_admission_terminal_reservation(
                                lifecycle.repository.path,
                                start_object_id=str(start_id or ""),
                            )
                            is not None
                        )
                        terminal_start_ids.append(str(start_id or ""))
                    recorded = lifecycle.record_validated_experiment_attempt(
                        prepared,
                        commit=False,
                    )
                    reservation_path = recorded.get("terminal_reservation")
                    if (
                        isinstance(reservation_path, Path)
                        and not reservation_preexisted
                    ):
                        created_terminal_reservation_paths.append(reservation_path)
                    if not admission_required and not recorded["attempt"].created:
                        raise ResearchGitError(
                            "experiment attempt object unexpectedly pre-existed"
                        )
                    if recorded["attempt"].created:
                        created_attempt_paths.append(recorded["attempt"].path)
                    recorded_attempts.append(recorded["attempt"])
                    attempt_ids[idea_index] = recorded["attempt"].object_id
                    attempt_paths.append(
                        recorded["attempt"]
                        .path.resolve()
                        .relative_to(lifecycle.repository.path)
                        .as_posix()
                    )
                if attempt_paths and admission_required:
                    checkpoint_status = (
                        recorded_attempts[0].state
                        if len(recorded_attempts) == 1
                        else (
                            "completed"
                            if any(
                                attempt.state == "completed"
                                for attempt in recorded_attempts
                            )
                            else "failed"
                        )
                    )
                    committed_batch = _required_terminal_batch_checkpoint(
                        lifecycle.repository.path,
                        attempts=recorded_attempts,
                        attempt_paths=attempt_paths,
                        start_object_ids=terminal_start_ids,
                        status=checkpoint_status,
                    )
                    created_terminal_checkpoint = committed_batch is None
                    if committed_batch is None:
                        checkpoint = create_checkpoint(
                            lifecycle.repository.path,
                            stage="experiment_admission_terminal",
                            subject=(
                                f"commit {len(attempt_paths)} admitted experiment outcomes"
                            ),
                            summary=(
                                "Durably consume every committed admission start in a "
                                "later, terminal-only checkpoint."
                            ),
                            status=checkpoint_status,
                            only_paths=sorted(attempt_paths),
                        )
                        if not checkpoint.committed or not checkpoint.commit:
                            raise ResearchGitError(
                                "required experiment admission terminal checkpoint was "
                                "not committed"
                            )
                        committed_batch = _required_terminal_batch_checkpoint(
                            lifecycle.repository.path,
                            attempts=recorded_attempts,
                            attempt_paths=attempt_paths,
                            start_object_ids=terminal_start_ids,
                            status=checkpoint_status,
                        )
                        if committed_batch != checkpoint.commit:
                            raise ResearchGitError(
                                "required experiment terminal checkpoint identity changed"
                            )
                    _publish_experiment_admission_terminal_completion_refs(
                        lifecycle.repository.path,
                        start_object_ids=terminal_start_ids,
                        commit=committed_batch,
                    )
                    if not created_terminal_checkpoint and created_attempt_paths:
                        committed_paths = {
                            (
                                f".xscientist/objects/{item.get('kind')}/"
                                f"{item.get('object_id')}.json"
                            )
                            for item in list_research_objects_at_ref(
                                lifecycle.repository.path, "HEAD"
                            )
                        }
                        for path in created_attempt_paths:
                            relative = path.relative_to(
                                lifecycle.repository.path
                            ).as_posix()
                            if relative not in committed_paths:
                                path.unlink(missing_ok=True)
                                _fsync_directory(path.parent)
                    terminal_committed = True
            except BaseException:
                if not terminal_committed:
                    try:
                        committed_paths = {
                            (
                                f".xscientist/objects/{item.get('kind')}/"
                                f"{item.get('object_id')}.json"
                            )
                            for item in list_research_objects_at_ref(
                                lifecycle.repository.path, "HEAD"
                            )
                        }
                        terminal_committed = any(
                            path in committed_paths for path in attempt_paths
                        )
                    except BaseException:
                        # Preserve recoverable state when commit state is uncertain.
                        terminal_committed = True
                if not terminal_committed:
                    touched_parents: set[Path] = set()
                    for path in (
                        *created_attempt_paths,
                        *created_terminal_reservation_paths,
                    ):
                        path.unlink(missing_ok=True)
                        touched_parents.add(path.parent)
                    for parent in touched_parents:
                        _fsync_directory(parent)
                raise
        updated_object_ids = dict(object_ids)
        updated_object_ids["attempts"] = attempt_ids
        args._research_vcs_ids = updated_object_ids
        print(f"🧬 Research VCS 已记录 {len(results)} 个实验尝试（含失败结果）")
    except (ResearchGitError, OSError, ValueError) as exc:
        try:
            admission_required = (
                lifecycle.repository.experiment_admission_mode() == "required"
            )
        except (ResearchGitError, OSError, UnboundLocalError):
            admission_required = False
        if admission_required:
            raise RuntimeError(
                f"required Research VCS experiment recording failed: {exc}"
            ) from exc
        _research_vcs_failure(args, "experiment record", exc)


def _manuscript_artifact_bindings(
    project_root: Path,
    *,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Snapshot exact paper/review artifacts into CAS for a manuscript revision."""

    from xscientist.research_git import add_research_object

    project_root = project_root.expanduser().resolve()
    candidates: list[tuple[str, Path]] = []

    def add_candidate(role: str, raw_path: Any, *, base: Path = project_root) -> None:
        rendered = str(raw_path or "").strip()
        if not rendered or "://" in rendered:
            return
        candidate = Path(rendered).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        try:
            lexical = Path(os.path.abspath(candidate))
            if sys.platform == "darwin" and lexical.parts[:2] in {
                ("/", "etc"),
                ("/", "tmp"),
                ("/", "var"),
            }:
                lexical = Path("/private") / lexical.relative_to("/")
            relative = lexical.relative_to(project_root)
            cursor = project_root
            for part in relative.parts:
                cursor /= part
                if cursor.is_symlink():
                    return
            resolved = lexical.resolve()
            resolved.relative_to(project_root)
        except (OSError, ValueError):
            return
        if resolved.is_file():
            candidates.append((role, resolved))

    for role, field in (
        ("manuscript_pdf", "pdf_path"),
        ("pipeline_manifest", "pipeline_manifest"),
        ("claim_evidence_graph", "claim_evidence_graph_file"),
        ("experiment_registry", "experiment_registry_file"),
        ("figure_spec", "figure_spec_file"),
        ("plot_execution_receipt", "plot_execution_receipt_file"),
        ("manuscript_state", "manuscript_state_file"),
        ("review_state", "review_state_file"),
        ("repair_plan", "repair_plan_file"),
        ("critic_findings", "critic_findings_file"),
        ("stage_standards", "stage_standards_file"),
    ):
        add_candidate(role, result.get(field))

    manuscript_state_path = next(
        (path for role, path in candidates if role == "manuscript_state"),
        None,
    )
    if manuscript_state_path is not None:
        manuscript_state = _safe_load_json(manuscript_state_path, default={}) or {}
        add_candidate(
            "manuscript_source",
            manuscript_state.get("latex_path"),
            base=manuscript_state_path.parent,
        )

    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for role, path in candidates:
        relative = path.relative_to(project_root).as_posix()
        identity = (role, relative)
        if identity in seen:
            continue
        seen.add(identity)
        pointer = add_research_object(
            project_root,
            path,
            logical_path=relative,
        )
        bindings.append(
            {
                "role": role,
                "logical_path": relative,
                "content_hash": pointer.object_hash,
                "pointer_path": pointer.pointer_path.relative_to(
                    project_root
                ).as_posix(),
                "size": pointer.size,
            }
        )
    return sorted(bindings, key=lambda item: (item["role"], item["logical_path"]))


def _paper_result_artifact_fields(
    exp_dir: str | Path,
    *,
    pdf_path: str | Path | None = None,
    critic_findings_file: str | Path | None = None,
) -> dict[str, Any]:
    """Return the common durable paper paths for success and terminal failure."""

    root = Path(exp_dir).expanduser()
    resolved_pdf = str(pdf_path or "").strip() or str(find_latest_pdf(str(root)) or "")
    if resolved_pdf:
        candidate = Path(resolved_pdf).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved_pdf = str(candidate)
    fields: dict[str, Any] = {
        "exp_dir": str(root),
        "pdf_path": (
            resolved_pdf if resolved_pdf and Path(resolved_pdf).is_file() else None
        ),
        "pipeline_manifest": str(root / "pipeline_manifest.json"),
        "claim_evidence_graph_file": str(root / "claim_evidence_graph.json"),
        "experiment_registry_file": str(root / "experiment_registry.jsonl"),
        "figure_spec_file": str(root / "figure_spec.json"),
        "plot_execution_receipt_file": str(root / "plot_execution_receipt.json"),
        "manuscript_state_file": str(root / "manuscript_state.json"),
        "review_state_file": str(root / "review_state.json"),
        "repair_plan_file": str(root / "repair_plan.json"),
        "stage_standards_file": str(root / "stage_standards.json"),
    }
    critic_path = str(critic_findings_file or "").strip()
    if not critic_path:
        candidate = root / "hostile_critic" / "findings.json"
        critic_path = str(candidate) if candidate.is_file() else ""
    fields["critic_findings_file"] = critic_path or None
    return fields


def _record_local_research_handoff_objects(
    args: argparse.Namespace,
    *,
    results: list[dict],
    insight_report: dict[str, Any] | None = None,
) -> None:
    if not getattr(args, "_research_git_active", False):
        return
    from xscientist.research_git import ResearchGitError
    from xscientist.research_lifecycle import ResearchLifecycle

    try:
        from ai_scientist.protocol.hashing import content_hash, hash_manifest
        from xscientist.research_context import record_research_context_snapshot

        lifecycle = ResearchLifecycle(args.project_dir)
        object_ids = getattr(args, "_research_vcs_ids", {})
        evidence_count = 0
        claim_count = 0
        insights_by_idea: dict[int, list[dict[str, Any]]] = {}
        for insight in (insight_report or {}).get("insights") or []:
            if not isinstance(insight, dict):
                continue
            try:
                insight_index = int(insight.get("idea_idx"))
            except (TypeError, ValueError):
                continue
            insights_by_idea.setdefault(insight_index, []).append(insight)
        for fallback_index, result in enumerate(results):
            idea_index = int(result.get("idea_idx", fallback_index))
            attempt_id = (object_ids.get("attempts") or {}).get(idea_index)
            if not attempt_id:
                continue
            compact = _compact_research_result(result)
            manifest_path = result.get("ara_manifest")
            manifest = (
                _safe_load_json(manifest_path, default=None) if manifest_path else None
            )
            manifest_hash = (
                hash_manifest(manifest) if isinstance(manifest, dict) else None
            )
            exploration_graph = (
                _safe_load_json(
                    Path(str(manifest_path)).expanduser().parent
                    / "exploration_graph.json",
                    default=None,
                )
                if manifest_path
                else None
            )
            exploration_graph_hash = (
                content_hash(exploration_graph)
                if isinstance(exploration_graph, dict)
                else None
            )
            context_pack_refs = sorted(
                {
                    str(ref)
                    for node in (exploration_graph or {}).get("nodes") or []
                    if isinstance(node, dict)
                    for ref in node.get("context_pack_refs") or []
                    if isinstance(ref, str) and ref.startswith("sha256:")
                }
            )
            evidence_payload: dict[str, object] = {
                "result": str(result.get("status") or "unknown"),
                "measurement_hash": content_hash(compact),
                "metrics": {
                    key: value
                    for key, value in compact.items()
                    if key.endswith("_score") or key.endswith("_passed")
                },
                "verification_state": "awaiting_independent_verification",
            }
            if manifest_hash:
                evidence_payload["ara_manifest_hash"] = manifest_hash
                if exploration_graph_hash:
                    evidence_payload["ara_exploration_graph_hash"] = (
                        exploration_graph_hash
                    )
                if context_pack_refs:
                    evidence_payload["context_pack_refs"] = context_pack_refs
                try:
                    evidence_payload["ara_manifest_path"] = (
                        Path(str(manifest_path))
                        .expanduser()
                        .resolve()
                        .relative_to(lifecycle.repository.path)
                        .as_posix()
                    )
                except (OSError, ValueError):
                    evidence_payload["ara_manifest_path"] = "manifest.json"
            hypothesis_id = (object_ids.get("hypotheses") or {}).get(idea_index)
            evidence_record = lifecycle.evidence(
                evidence_payload,
                attempt_ids=[attempt_id],
                supports=[hypothesis_id] if hypothesis_id else [],
                verified=False,
                commit=False,
            )["evidence"]
            object_ids.setdefault("evidence", {})[
                idea_index
            ] = evidence_record.object_id
            evidence_count += 1

            claim_ids: list[str] = []
            if manifest_hash and manifest_path:
                claims_dir = Path(str(manifest_path)).expanduser().parent / "claims"
                for claim_path in sorted(claims_dir.glob("*.json")):
                    if claim_path.name.startswith("_"):
                        continue
                    claim_payload = _safe_load_json(claim_path, default=None)
                    if not isinstance(claim_payload, dict):
                        continue
                    ara_claim_id = str(claim_payload.get("claim_id") or claim_path.stem)
                    claim_hash = str(claim_payload.get("claim_hash") or "")
                    if not claim_hash.startswith("sha256:"):
                        claim_hash = content_hash(
                            {
                                "claim_id": ara_claim_id,
                                "node_id": claim_payload.get("node_id"),
                                "evidence_refs": claim_payload.get("evidence_refs")
                                or [],
                                "source": claim_payload.get("source") or {},
                            }
                        )
                    research_claim_payload: dict[str, object] = {
                        "claim_hash": claim_hash,
                        "ara_claim_id": ara_claim_id,
                        "node_id": claim_payload.get("node_id"),
                        "ara_manifest_hash": manifest_hash,
                        "resolved": bool(claim_payload.get("resolved")),
                        "evidence_refs": sorted(
                            set(claim_payload.get("evidence_refs") or [])
                        ),
                        "context_pack_refs": sorted(
                            set(claim_payload.get("context_pack_refs") or [])
                        ),
                    }
                    if exploration_graph_hash:
                        research_claim_payload["ara_exploration_graph_hash"] = (
                            exploration_graph_hash
                        )
                    claim_record = lifecycle.claim(
                        research_claim_payload,
                        evidence_ids=[evidence_record.object_id],
                        verified=False,
                        commit=False,
                    )["claim"]
                    claim_ids.append(claim_record.object_id)
                    claim_count += 1
            for insight in insights_by_idea.get(idea_index, []):
                insight_claim = lifecycle.claim(
                    {
                        "statement": str(insight.get("claim") or ""),
                        "claim_hash": content_hash(insight),
                        "insight_kind": insight.get("kind"),
                        "confidence": insight.get("confidence"),
                        "epistemic_status": "machine_synthesized_unverified",
                        "evidence_refs": list(insight.get("evidence_refs") or []),
                        "uncertainty": insight.get("uncertainty"),
                        "next_experiment": insight.get("next_experiment") or {},
                    },
                    evidence_ids=[evidence_record.object_id],
                    verified=False,
                    commit=False,
                )["claim"]
                claim_ids.append(insight_claim.object_id)
                claim_count += 1
            object_ids.setdefault("claims", {})[idea_index] = claim_ids
            context = record_research_context_snapshot(
                lifecycle.repository,
                target_ids=claim_ids or [evidence_record.object_id],
                decision_kind="automated_research_handoff",
                selected="hold",
                options_considered=[
                    {
                        "option": "promote",
                        "rejected_because": "independent verification is missing",
                    },
                    {"option": "hold", "rejected_because": ""},
                ],
                rationale=[
                    "Keep generated claims and manuscript in draft until independent verification."
                ],
                constraints=["independent_verification_missing"],
                actor_id="research-handoff-context-recorder",
            )
            context_payload = lifecycle.repository.get(context.object_id)["payload"]
            gate = lifecycle.repository.record(
                "gate_decision",
                {
                    "decision": "hold",
                    "claim_promotion_allowed": False,
                    "required_failures": ["independent_verification_missing"],
                    "quality_gate_passed": result.get("quality_gate_passed"),
                    "submission_acceptance_passed": result.get(
                        "submission_acceptance_passed"
                    ),
                    "local_submission_ready": result.get(
                        "local_submission_ready",
                        result.get("submission_acceptance_passed"),
                    ),
                    "acceptance_guaranteed": False,
                    "context_required": True,
                    "context_hash": context_payload["context_hash"],
                },
                state="rejected",
                relations=[
                    {"type": "evaluates", "target": target}
                    for target in (claim_ids or [evidence_record.object_id])
                ]
                + [
                    {
                        "type": "depends_on",
                        "target": context.object_id,
                        "role": "decision_context",
                    }
                ],
                actor={
                    "actor_id": "research-handoff-gate",
                    "authority": "deterministic_gate",
                },
            )
            artifact_bindings = _manuscript_artifact_bindings(
                lifecycle.repository.path,
                result=result,
            )
            revision_hash = (
                content_hash(
                    {
                        "idea_idx": idea_index,
                        "artifact_bindings": artifact_bindings,
                    }
                )
                if artifact_bindings
                else None
            )
            manuscript = lifecycle.manuscript(
                {
                    "idea_idx": idea_index,
                    "status": result.get("status"),
                    "quality_score": result.get("quality_score"),
                    "rigor_score": result.get("rigor_score"),
                    "final": False,
                    "blocker": "independent_verification_missing",
                    "revision_hash": revision_hash,
                    "artifact_bindings": artifact_bindings,
                    "artifact_hashes": sorted(
                        {str(item["content_hash"]) for item in artifact_bindings}
                    ),
                },
                claim_ids=claim_ids,
                gate_id=gate.object_id,
                final=False,
                commit=False,
            )["manuscript"]
            object_ids.setdefault("gates", {})[idea_index] = gate.object_id
            object_ids.setdefault("contexts", {})[idea_index] = context.object_id
            object_ids.setdefault("manuscripts", {})[idea_index] = manuscript.object_id
        from xscientist.research_strategy import (
            record_research_followup_queue,
            review_research_program,
        )

        strategy_review = review_research_program(
            lifecycle.repository.path,
            record=True,
            commit=False,
        )
        if strategy_review.get("object") is not None:
            object_ids["program_review"] = strategy_review["object"].object_id
        followup_limit = (
            max(1, int(getattr(args, "autonomous_quality_followup_rounds", 0)))
            if str(getattr(args, "autopilot", "") or "") == "discovery"
            else max(0, int(getattr(args, "autonomous_quality_followup_rounds", 0)))
        )
        followup_queue = record_research_followup_queue(
            lifecycle.repository.path,
            review=strategy_review["report"],
            review_id=(
                strategy_review["object"].object_id
                if strategy_review.get("object") is not None
                else None
            ),
            max_actions=followup_limit,
            commit=False,
        )
        if followup_queue["active"]:
            object_ids["strategy_followups"] = [
                item["object_id"] for item in followup_queue["active"]
            ]
        atomic_write_json(
            Path(args.project_dir) / "04_logs" / "research_strategy_followups.json",
            followup_queue,
            ensure_ascii=False,
        )
        args._research_vcs_ids = object_ids
        print(
            "🧬 Research VCS 已投影 "
            f"{evidence_count} 个证据对象、{claim_count} 个 claim 锚点；"
            "未通过独立验证的稿件保持 draft"
        )
    except (ResearchGitError, OSError, ValueError) as exc:
        _research_vcs_failure(args, "paper handoff record", exc)


def _safe_load_json(path: str | Path, *, default=None):
    path_obj = Path(path)
    if not path_obj.exists():
        return default
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _resolve_workflow_strategy(
    *,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
):
    workflow_spec = resolve_workflow_mode(
        workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    template_profile, template_capability = resolve_template_mode_for_workflow(
        workflow_spec,
        submission_mode=submission_mode,
    )
    workflow_metadata = build_workflow_manifest_metadata(workflow_spec)
    return workflow_spec, workflow_metadata, template_profile, template_capability


def _collect_requested_models(args: argparse.Namespace) -> list[str]:
    candidates = [
        args.model_ideation,
        args.model_agg_plots,
        args.model_writeup,
        args.model_writeup_small,
        args.model_citation,
        args.model_review,
        args.idea_rank_model,
        args.quality_model,
    ]
    models: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        for model in [item.strip() for item in str(value or "").split(",")]:
            if not model or model in seen:
                continue
            models.append(model)
            seen.add(model)
    return models


def _write_project_pipeline_seed_artifacts(
    *,
    project_dir: str | Path,
    ideas: list[dict],
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
) -> list[dict]:
    project_root = Path(project_dir).expanduser().resolve()
    workflow_spec, workflow_metadata, template_profile, template_capability = (
        _resolve_workflow_strategy(
            workflow_mode=workflow_mode,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
    )
    initialize_pipeline_contracts(
        project_root,
        project_name=project_root.name,
        template_profile=template_profile,
        template_capability=template_capability,
        pipeline_goal=workflow_spec.pipeline_goal,
        workflow_mode=workflow_metadata["workflow_mode"],
        workflow_label=workflow_metadata["workflow_label"],
        workflow_summary=workflow_metadata["workflow_summary"],
        workflow_inspirations=workflow_metadata["workflow_inspirations"],
        workflow_sequence=workflow_metadata["workflow_sequence"],
    )
    idea_cards = build_idea_cards(
        ideas,
        target_venue=target_venue,
        template_profile=template_profile,
        workflow_mode=workflow_spec.name,
    )
    save_contract_artifact(
        project_root,
        "idea_cards",
        idea_cards,
        producer="run_project.project_seed",
    )
    save_hypothesis_archive(
        project_root,
        build_hypothesis_archive(idea_cards),
        producer="run_project.project_seed",
    )
    seed_scientific_foundation(
        project_root,
        idea_cards,
        producer="run_project.project_seed",
    )
    lead_idea = idea_cards[0] if idea_cards else {}
    lead_plan = (
        build_research_plan(
            lead_idea,
            target_venue=target_venue,
            submission_mode=submission_mode,
            breakthrough_mode=breakthrough_mode,
            high_quality_mode=high_quality_mode,
        )
        if lead_idea
        else {}
    )
    if lead_plan:
        save_contract_artifact(
            project_root,
            "research_plan",
            lead_plan,
            producer="run_project.project_planning",
            depends_on=["idea_cards"],
        )
        save_preregistration(
            project_root,
            build_preregistration(lead_idea, lead_plan),
            producer="run_project.project_planning",
        )
    research_program = render_research_program_markdown(
        project_name=project_root.name,
        target_venue=target_venue,
        template_profile=template_profile,
        idea_name=str(lead_idea.get("name") or project_root.name),
        hypothesis=str(lead_idea.get("core_hypothesis") or ""),
        workflow_mode=workflow_spec.name,
        workflow_summary=str(workflow_metadata["workflow_summary"]),
        workflow_inspirations=list(workflow_metadata["workflow_inspirations"]),
        workflow_sequence=list(workflow_metadata["workflow_sequence"]),
        budget=lead_plan.get("budget"),
        execution_policy=lead_plan.get("execution_policy"),
        success_criteria=[
            str(task.get("success_criterion") or "").strip()
            for task in (lead_plan.get("tasks") or [])[:4]
            if str(task.get("success_criterion") or "").strip()
        ],
        failure_handling_rules=[
            "Preserve partial outputs and mark failed stages in pipeline_manifest.json.",
            "Do not let downstream stages consume artifacts unless their contract status is ready.",
        ],
    )
    save_contract_artifact(
        project_root,
        "research_program",
        research_program,
        producer="run_project.project_seed",
        depends_on=["idea_cards"],
    )
    save_stage_standards(project_root)
    return idea_cards


def _write_experiment_pipeline_seed_artifacts(
    *,
    exp_dir: str | Path,
    idea: dict,
    idea_idx: int,
    target_venue: str | None,
    workflow_mode: str | None,
    submission_mode: bool,
    breakthrough_mode: bool,
    high_quality_mode: bool,
    template_profile: str,
    template_capability: str,
) -> tuple[dict, dict, dict]:
    exp_root = Path(exp_dir).expanduser().resolve()
    workflow_spec, workflow_metadata, _, _ = _resolve_workflow_strategy(
        workflow_mode=workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue,
    )
    initialize_pipeline_contracts(
        exp_root,
        project_name=exp_root.name,
        template_profile=template_profile,
        template_capability=template_capability,
        pipeline_goal=workflow_spec.pipeline_goal,
        workflow_mode=workflow_metadata["workflow_mode"],
        workflow_label=workflow_metadata["workflow_label"],
        workflow_summary=workflow_metadata["workflow_summary"],
        workflow_inspirations=workflow_metadata["workflow_inspirations"],
        workflow_sequence=workflow_metadata["workflow_sequence"],
    )
    idea_cards = build_idea_cards(
        [idea],
        target_venue=target_venue,
        template_profile=template_profile,
        workflow_mode=workflow_spec.name,
    )
    idea_card = dict(idea_cards[0] if idea_cards else {})
    idea_card["idea_id"] = f"idea_{idea_idx}"
    save_contract_artifact(
        exp_root,
        "idea_cards",
        [idea_card],
        producer="run_project.idea_seed",
    )
    save_hypothesis_archive(
        exp_root,
        build_hypothesis_archive([idea_card]),
        producer="run_project.idea_seed",
    )
    seed_scientific_foundation(
        exp_root,
        [idea_card],
        producer="run_project.idea_seed",
    )
    research_plan = build_research_plan(
        idea_card,
        target_venue=target_venue,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
    )
    save_contract_artifact(
        exp_root,
        "research_plan",
        research_plan,
        producer="run_project.planning",
        depends_on=["idea_cards"],
    )
    save_preregistration(
        exp_root,
        build_preregistration(idea_card, research_plan),
        producer="run_project.planning",
    )
    save_truth_contract_bundle(
        exp_root,
        idea_card=idea_card,
        research_plan=research_plan,
        producer="run_project.planning",
    )
    save_sample_gate_plan(
        exp_root,
        research_plan=research_plan,
        producer="run_project.planning",
    )
    claim_evidence_graph = build_claim_evidence_graph(idea_card, research_plan)
    save_contract_artifact(
        exp_root,
        "claim_evidence_graph",
        claim_evidence_graph,
        producer="run_project.planning",
        depends_on=["research_plan"],
    )
    research_program = render_research_program_markdown(
        project_name=exp_root.name,
        target_venue=target_venue,
        template_profile=template_profile,
        idea_name=str(idea_card.get("name") or f"idea_{idea_idx}"),
        hypothesis=str(idea_card.get("core_hypothesis") or ""),
        workflow_mode=workflow_spec.name,
        workflow_summary=str(workflow_metadata["workflow_summary"]),
        workflow_inspirations=list(workflow_metadata["workflow_inspirations"]),
        workflow_sequence=list(workflow_metadata["workflow_sequence"]),
        budget=research_plan.get("budget"),
        execution_policy=research_plan.get("execution_policy"),
        success_criteria=[
            str(task.get("success_criterion") or "").strip()
            for task in (research_plan.get("tasks") or [])[:6]
            if str(task.get("success_criterion") or "").strip()
        ],
        failure_handling_rules=[
            "Failed experiments must still be recorded in experiment_registry.jsonl.",
            "Claims and figures remain blocked until their supporting evidence is ready.",
        ],
    )
    save_contract_artifact(
        exp_root,
        "research_program",
        research_program,
        producer="run_project.planning",
        depends_on=["research_plan"],
    )
    save_stage_standards(exp_root)
    return idea_card, research_plan, claim_evidence_graph


def _build_experiment_registry_rows(
    *,
    exp_dir: str | Path,
    research_plan: dict,
) -> list[dict]:
    exp_root = Path(exp_dir).expanduser().resolve()
    experiment_report = (
        _safe_load_json(exp_root / "experiment_report.json", default={}) or {}
    )
    stages = list(experiment_report.get("stages") or [])
    warnings = list(experiment_report.get("warnings") or [])
    latest_run_dir = experiment_report.get("latest_run_dir")
    rows: list[dict] = []
    for idx, task in enumerate(research_plan.get("tasks") or []):
        stage = stages[min(idx, len(stages) - 1)] if stages else {}
        best = stage.get("best") or {}
        attempts = [
            item for item in stage.get("attempts") or [] if isinstance(item, dict)
        ]
        if not attempts and best:
            attempts = [
                {**best, "node_id": best.get("best_node_id"), "is_buggy": False}
            ]
        if not attempts:
            attempts = [{"node_id": f"missing-{idx}", "is_buggy": True}]
        for attempt_idx, attempt in enumerate(attempts):
            node_id = str(attempt.get("node_id") or f"attempt-{attempt_idx}")
            has_metric = attempt.get("metric_mean") is not None
            buggy = attempt.get("is_buggy") is True
            status = "completed" if has_metric and not buggy else "failed"
            if not stages and not best:
                status = "planned"
            is_best = bool(best) and node_id == str(
                best.get("node_id") or best.get("best_node_id") or ""
            )
            result_summary = {
                "metric_name": attempt.get("metric_name"),
                "metric_mean": attempt.get("metric_mean"),
                "metric_objective": attempt.get("metric_objective"),
                "dataset_names": attempt.get("dataset_names") or [task.get("dataset")],
                "seed": attempt.get("seed"),
                "evaluation_report": attempt.get("evaluation_report"),
                "delta_objective_vs_prev_stage": stage.get(
                    "delta_objective_vs_prev_stage"
                ),
                "warnings": warnings,
                "attempt_node_id": node_id,
                "node_counts": stage.get("node_counts"),
            }
            artifacts = {
                "exp_dir": str(exp_root),
                "experiment_report_json": str(exp_root / "experiment_report.json"),
                "experiment_report_md": str(exp_root / "experiment_report.md"),
                "latest_run_dir": latest_run_dir,
                "stage_dir": stage.get("stage_dir"),
                "journal_path": stage.get("journal_path"),
            }
            if isinstance(attempt.get("artifacts"), dict):
                artifacts.update(attempt["artifacts"])
            rows.append(
                build_experiment_record(
                    record_id=f"{str(task.get('task_id') or f'task_{idx}')}:stage_{idx}:{node_id}",
                    task_id=str(task.get("task_id") or f"task_{idx}"),
                    dataset=str(task.get("dataset") or "dataset_to_be_selected"),
                    metric=str(task.get("metric") or "primary_task_metric"),
                    baseline_ref=str(
                        task.get("baseline") or "strong_existing_baseline"
                    ),
                    config={
                        "goal": task.get("goal"),
                        "priority": task.get("priority"),
                    },
                    seed=attempt.get("seed"),
                    status=status,
                    result_summary=result_summary,
                    artifacts=artifacts,
                    error_type=(
                        None if status == "completed" else "buggy_or_missing_attempt"
                    ),
                    error_message=(
                        None
                        if status == "completed"
                        else "; ".join(warnings[:3])
                        or "Attempt did not produce a valid metric."
                    ),
                    entered_storyline=is_best,
                    budget=task.get("budget"),
                    workflow_mode=research_plan.get("workflow_mode"),
                    policy_name=(research_plan.get("execution_policy") or {}).get(
                        "policy_name"
                    ),
                    acceptance_checks=task.get("acceptance_checks"),
                    acceptance_results=[
                        {
                            "check": str(check),
                            "passed": status == "completed",
                            "source": "experiment_journal_attempt",
                        }
                        for check in (task.get("acceptance_checks") or [])
                    ],
                    budget_audit={
                        "audited": status == "completed",
                        "within_budget": status == "completed",
                        "source": "experiment_registry_builder",
                    },
                    budget_status="within_budget" if status == "completed" else None,
                    evidence_role=task.get("evidence_role"),
                    paired_control_task_id=task.get("paired_control_task_id"),
                    intervention_variant=task.get("intervention_variant"),
                    stress_condition=task.get("stress_condition"),
                )
            )
    return rows


def _save_manuscript_contract_state(
    *,
    exp_dir: str | Path,
    writeup_type: str,
    target_venue: str | None,
    writing_profile: str,
) -> dict:
    exp_root = Path(exp_dir).expanduser().resolve()
    claim_evidence_graph = load_contract_artifact(
        exp_root,
        "claim_evidence_graph",
        default={},
    )
    figure_spec = load_contract_artifact(
        exp_root,
        "figure_spec",
        default={},
    )
    manuscript_state = build_manuscript_state(
        writeup_type=writeup_type,
        target_venue=target_venue,
        writing_profile=writing_profile,
        skill_pack=list_writing_skills(),
        claim_evidence_graph=(
            claim_evidence_graph if isinstance(claim_evidence_graph, dict) else {}
        ),
        figure_spec=figure_spec if isinstance(figure_spec, dict) else {},
        latex_path=str(exp_root / "template.tex"),
    )
    save_manuscript_state(str(exp_root), manuscript_state)
    return manuscript_state


def generate_ideas(
    project_dir: str, topic_file: str, model: str, num_ideas: int, reflections: int
):
    """步骤1: 生成研究想法"""
    print("\n" + "=" * 80)
    print("🔬 步骤 1: 生成研究想法")
    print("=" * 80)

    # 读取主题描述
    with open(topic_file, "r") as f:
        workshop_description = f.read()

    # 输出文件路径
    idea_json = _managed_project_file(project_dir, "01_ideas", "generated_ideas.json")
    stable_idea_json = _managed_project_file(project_dir, "01_ideas", "ideas.json")

    # 创建客户端
    client, client_model = create_client(model)

    # 生成想法
    ideas = generate_temp_free_idea(
        idea_fname=str(idea_json),
        client=client,
        model=client_model,
        workshop_description=workshop_description,
        max_num_generations=num_ideas,
        num_reflections=reflections,
    )

    print(f"✅ 生成了 {len(ideas)} 个想法，保存到 {idea_json}")
    # Benchmarks (and downstream workflows) expect a stable `ideas.json` fixture.
    # Keep the historical `generated_ideas.json` name but also mirror it to `ideas.json`.
    try:
        atomic_write_json(stable_idea_json, ideas, indent=4, ensure_ascii=False)
        print(f"✅ 已同步 ideas fixture: {stable_idea_json}")
    except OSError as exc:
        print(f"⚠️  未能同步 ideas.json fixture: {exc}")
    return str(idea_json)


def find_latest_pdf(exp_dir: str):
    """查找最新的PDF文件"""
    return find_latest_pdf_path(exp_dir)


def _maybe_pareto_seed_for_rewrite(exp_dir: str) -> str | None:
    try:
        from ai_scientist.utils.pareto_pool import maybe_select_seed_path
    except Exception:
        return None
    return maybe_select_seed_path(exp_dir)


def improve_paper_with_review(
    exp_dir: str,
    review_text: dict,
    review_img: dict,
    model: str,
    *,
    round_index: int,
    review_dir: str,
    previous_issue_ledger=None,
    target_venue: str | None = None,
):
    """根据审查意见进行 issue-driven 论文改进。"""
    print("\n" + "=" * 80)
    print(f"🔧 根据审查意见改进论文 (issue-driven, round {round_index})")
    print("=" * 80)

    issue_ledger = build_issue_ledger(
        text_review=review_text,
        img_review=review_img,
        max_issues=14,
        previous_ledger=previous_issue_ledger,
        target_venue=target_venue,
    )
    print(
        "🧭 本轮问题台账: "
        f"{issue_ledger.get('issue_count', 0)} "
        f"(critical={issue_ledger.get('critical_count', 0)}, "
        f"major={issue_ledger.get('major_count', 0)}, minor={issue_ledger.get('minor_count', 0)})"
    )

    progress = None
    if isinstance(previous_issue_ledger, dict):
        progress = evaluate_issue_progress(
            previous_issues=list(previous_issue_ledger.get("issues") or []),
            current_issues=list(issue_ledger.get("issues") or []),
        )
        print(
            "📈 跨轮问题进展: "
            f"resolved={progress.get('resolved_issue_count', 0)}, "
            f"persistent={progress.get('persistent_issue_count', 0)}, "
            f"new={progress.get('new_issue_count', 0)}, "
            f"unresolved_critical={progress.get('unresolved_critical_count', 0)}"
        )

    rewrite_result = apply_issue_driven_rewrite(
        paper_dir=exp_dir,
        model=model,
        ledger=issue_ledger,
        round_index=round_index,
        artifact_dir=review_dir,
        target_venue=target_venue,
        temperature=0.35,
        seed_latex_path=_maybe_pareto_seed_for_rewrite(exp_dir),
        project_root=str(Path(exp_dir).expanduser().resolve()),
        job_id=f"rewrite_round_{round_index}",
    )
    round_gate = assess_self_review_gate(
        ledger=issue_ledger,
        progress=progress,
        rewrite_result=rewrite_result,
        round_index=round_index,
        target_venue=target_venue,
    )
    artifact_files = save_self_review_artifacts(
        review_dir=review_dir,
        ledger=issue_ledger,
        progress=progress,
        gate=round_gate,
    )
    if rewrite_result.get("status") == "success":
        print(
            "✅ Issue-driven 改稿完成: "
            f"coverage={rewrite_result.get('coverage_ratio', 0):.2f}, "
            f"high_value_coverage={rewrite_result.get('high_value_coverage_ratio', 0):.2f}, "
            f"compile_ok={rewrite_result.get('compile_ok')}"
        )
    else:
        print(
            "⚠️  Issue-driven 改稿未完全成功: "
            f"status={rewrite_result.get('status')}, "
            f"reason={rewrite_result.get('reason')}"
        )
    print(
        "🚦 Round gate: "
        f"ready={round_gate.get('ready')}, "
        f"score={round_gate.get('score')}, "
        f"reasons={round_gate.get('reasons', [])}"
    )

    return {
        "status": "success" if rewrite_result.get("status") == "success" else "failed",
        "issue_ledger": issue_ledger,
        "issue_progress": progress,
        "round_gate": round_gate,
        "artifact_files": artifact_files,
        "rewrite_result": rewrite_result,
    }


def _process_single_idea_impl(args):
    """处理单个想法的完整流程（可在子进程中运行）"""
    args = list(args)
    admission_preflight = None
    if len(args) == 42:
        admission_preflight = args.pop()
    resume_from = None
    if len(args) == 41:
        resume_from = args.pop()
    requested_workflow_mode = None
    if len(args) == 40:
        requested_workflow_mode = args.pop()
    elif len(args) == 39 and isinstance(args[-1], str) and isinstance(args[-2], bool):
        requested_workflow_mode = args.pop()
        args.append(False)
    elif len(args) == 38:
        args.append(False)
    (
        project_dir,
        research_root,
        idea_idx,
        idea,
        bfts_config_path,
        model_writeup,
        model_citation,
        model_review,
        model_agg_plots,
        model_writeup_small,
        num_cite_rounds,
        writeup_retries,
        writeup_type,
        improvement_rounds,
        review_reflections,
        review_ensemble,
        review_fewshot,
        review_temperature,
        review_strategy,
        high_quality_mode,
        quality_preset,
        quality_model,
        target_venue,
        quality_threshold,
        rigor_threshold,
        quality_rewrite_rounds,
        autonomous_quality_followup_rounds,
        require_quality_gate,
        min_submission_priority,
        max_submission_blockers,
        writing_profile,
        writing_audit_rounds,
        strict_writing_guardrails,
        guardrail_repair_rounds,
        workflow_mode,
        template_profile,
        template_capability,
        strict_fallbacks,
        integrity_forensics_enabled,
    ) = args
    requested_workflow_mode = requested_workflow_mode or workflow_mode
    current_stage = "experiment_admission"
    registry_status = "failed"

    try:
        _revalidate_experiment_admission_preflight(
            project_dir,
            admission_preflight,
            bfts_config_path=bfts_config_path,
        )
        current_stage = "initialization"
        strict_writing_guardrails = bool(strict_writing_guardrails or high_quality_mode)
        guardrail_repair_rounds = max(0, int(guardrail_repair_rounds))
        research_root = Path(research_root).expanduser()
        idea_name = str(idea.get("Name", f"idea_{idea_idx}"))
        idea_key = idea_storage_key(idea, idea_index=int(idea_idx))
        integrity_forensics_result = {"enabled": False, "status": "disabled"}
        workflow_spec, workflow_metadata, _, _ = _resolve_workflow_strategy(
            workflow_mode=workflow_mode,
            submission_mode=(template_profile == "template_first"),
            breakthrough_mode=False,
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )

        print(f"\n{'='*80}")
        print(f"🚀 开始处理想法 #{idea_idx}: {idea_name}")
        print(f"{'='*80}")

        # 创建实验目录，或严格限定在当前项目内恢复已有实验。
        if resume_from:
            checkpoint = Path(str(resume_from)).expanduser().resolve()
            if not checkpoint.is_file() or len(checkpoint.parents) < 4:
                raise ValueError(f"Invalid resume checkpoint: {resume_from}")
            resumed_exp_dir = checkpoint.parents[3]
            experiments_root = (
                Path(project_dir).expanduser().resolve() / "02_experiments"
            )
            try:
                resumed_exp_dir.relative_to(experiments_root)
            except ValueError as exc:
                raise ValueError(
                    "Resume checkpoint must belong to this project's 02_experiments directory"
                ) from exc
            exp_dir = str(resumed_exp_dir)
            timestamp = resumed_exp_dir.name.split("_", 1)[0]
            print(f"♻️ 从 BFTS checkpoint 恢复: {checkpoint}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            experiments_root = confined_path(project_dir, "02_experiments")
            exp_dir = str(confined_path(experiments_root, f"{timestamp}_{idea_key}"))
            os.makedirs(exp_dir, exist_ok=True)

        # Turn on LLM tracing NOW: every prompt from ideation → writeup →
        # review will be journaled into <ara>/llm/calls.jsonl with its
        # payloads content-addressed into <ara>/objects/. We activate here
        # (not after finalize_ara_for_idea) because most LLM calls happen
        # BEFORE the ARA is exported — activating late would miss them.
        # deactivate_llm_tracing() runs in the outer finally so a crash
        # partway through leaves this process's env clean.
        activate_llm_tracing(
            project_dir=project_dir,
            idea=idea,
            timestamp=timestamp,
            stage="pipeline",
            strict=bool(strict_fallbacks or require_quality_gate),
        )
        initialize_pipeline_contracts(
            exp_dir,
            project_name=Path(exp_dir).name,
            template_profile=template_profile,
            template_capability=template_capability,
            pipeline_goal=workflow_spec.pipeline_goal,
            workflow_mode=workflow_metadata["workflow_mode"],
            workflow_label=workflow_metadata["workflow_label"],
            workflow_summary=workflow_metadata["workflow_summary"],
            workflow_inspirations=workflow_metadata["workflow_inspirations"],
            workflow_sequence=workflow_metadata["workflow_sequence"],
        )
        record_decision(
            exp_dir,
            category="workflow_strategy",
            selected=workflow_metadata["workflow_mode"],
            options_considered=build_workflow_strategy_decision_options(
                selected=workflow_metadata["workflow_mode"],
                requested_workflow_mode=requested_workflow_mode,
                submission_mode=(template_profile == "template_first"),
                high_quality_mode=bool(high_quality_mode),
                target_venue=target_venue,
            ),
            producer="run_project.workflow_strategy",
            metadata={
                "requested_workflow_mode": requested_workflow_mode,
                "resolved_workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "target_venue": target_venue,
                "high_quality_mode": bool(high_quality_mode),
            },
        )

        # 保存想法
        idea_path_json = osp.join(exp_dir, "idea.json")
        with open(idea_path_json, "w") as f:
            json.dump(idea, f, indent=4)
        idea_card, research_plan, claim_evidence_graph = (
            _write_experiment_pipeline_seed_artifacts(
                exp_dir=exp_dir,
                idea=idea,
                idea_idx=idea_idx,
                target_venue=target_venue,
                workflow_mode=workflow_mode,
                submission_mode=(template_profile == "template_first"),
                breakthrough_mode=False,
                high_quality_mode=high_quality_mode,
                template_profile=template_profile,
                template_capability=template_capability,
            )
        )
        assert_gate_preconditions_satisfiable(
            GatePreconditionContext(
                research_plan=research_plan,
                improvement_rounds=improvement_rounds,
                require_quality_gate=require_quality_gate,
                high_quality_mode=high_quality_mode,
                target_venue=target_venue,
                require_sample_gate=False,
            )
        )

        # ========== 步骤1: 运行实验 ==========
        print(f"\n📊 [想法 #{idea_idx}] 步骤 1/4: 运行实验")
        idea_path_md = osp.join(exp_dir, "idea.md")
        idea_to_markdown(
            idea,
            idea_path_md,
            None,
            research_plan=research_plan,
        )
        bfts_task_path = osp.join(exp_dir, "bfts_task.json")
        write_bfts_task_descriptor(
            idea,
            bfts_task_path,
            research_plan=research_plan,
        )

        config_path = str(bfts_config_path or "bfts_config.yaml")
        idea_config_path = edit_bfts_config_file(
            config_path,
            exp_dir,
            bfts_task_path,
            resume_from=resume_from,
        )

        current_stage = "experiment_admission"
        with _experiment_admission_execution_lease(
            project_dir,
            admission_preflight,
            bfts_config_path=bfts_config_path,
            experiment_dir=exp_dir,
            idea_key=idea_key,
        ) as admission_claim:
            current_stage = "experiment"
            if admission_claim is None:
                experiment_result = perform_experiments_bfts(idea_config_path)
            else:
                experiment_result = perform_experiments_bfts(
                    idea_config_path,
                    admission_runtime={
                        "repository_path": str(
                            Path(project_dir).expanduser().resolve()
                        ),
                        "claim": admission_claim,
                    },
                )
            current_stage = "experiment_admission"
            from xscientist.research_git import ResearchGitError

            try:
                _revalidate_experiment_admission_preflight(
                    project_dir,
                    admission_preflight,
                    bfts_config_path=bfts_config_path,
                )
            except ResearchGitError:
                # The durable claim is the execution linearization point.  A later
                # policy/configuration change quarantines the outcome but must not
                # erase evidence that the run already happened.
                if isinstance(experiment_result, dict):
                    experiment_result["admission_post_execution_status"] = (
                        "changed_after_claim"
                    )
            else:
                if isinstance(experiment_result, dict):
                    experiment_result["admission_post_execution_status"] = "unchanged"
        current_stage = "experiment"
        if (
            isinstance(experiment_result, dict)
            and experiment_result.get("status") != "completed"
        ):
            terminal = record_terminal_experiment_failure(
                exp_dir,
                research_plan=research_plan,
                experiment_result=experiment_result,
                producer="run_project.experiment_terminal_outcome",
            )
            return {
                "idea_idx": idea_idx,
                "exp_dir": exp_dir,
                "status": terminal["runtime_status"],
                "runtime_status": terminal["runtime_status"],
                "registry_status": terminal["status"],
                "terminal_receipt_hash": terminal["terminal_receipt_hash"],
                "stage": "experiment",
                "error": terminal["error"],
                "experiment_registry_rows": terminal["registry_rows"],
                "resumable": bool(experiment_result.get("resumable")),
                "checkpoint_path": experiment_result.get("checkpoint_path"),
                "run_status_path": experiment_result.get("run_status_path"),
                "manager_state_path": experiment_result.get("manager_state_path"),
                "initialization_status_path": experiment_result.get(
                    "initialization_status_path"
                ),
                "initialization_phase": experiment_result.get("initialization_phase"),
                "lock_path": experiment_result.get("lock_path"),
                "lock_owner": experiment_result.get("lock_owner"),
            }

        registry_status = "completed"

        # 复制实验结果
        latest_run_dir = find_latest_bfts_run_dir(exp_dir, logs_subdir="logs")
        experiment_results_dir = (
            osp.join(str(latest_run_dir), "experiment_results")
            if latest_run_dir is not None
            else osp.join(exp_dir, "logs/0-run/experiment_results")
        )
        if osp.exists(experiment_results_dir):
            shutil.copytree(
                experiment_results_dir,
                osp.join(exp_dir, "experiment_results"),
                dirs_exist_ok=True,
            )
        # Persist a deterministic experiment evaluation report for iterative debugging.
        try:
            write_experiment_report(exp_dir)
        except Exception as exc:
            print(f"[想法 #{idea_idx}] ⚠️  实验报告生成失败: {exc}")
        registry_rows = _build_experiment_registry_rows(
            exp_dir=exp_dir,
            research_plan=research_plan,
        )
        save_experiment_registry(exp_dir, registry_rows)
        sample_gate = evaluate_and_save_sample_gate(
            exp_dir,
            producer="run_project.experiment_registry",
        )
        sample_gate_decision = (
            "continue_full_generation"
            if sample_gate.get("full_generation_allowed")
            else "block_full_generation"
        )
        record_decision(
            exp_dir,
            category="sample_gate_full_generation",
            selected=sample_gate_decision,
            options_considered=build_sample_gate_decision_options(
                bool(sample_gate.get("full_generation_allowed"))
            ),
            producer="run_project.sample_gate",
            metadata={
                "sample_gate_reasons": sample_gate.get("result", {}).get("reasons", []),
                "sample_task_count": sample_gate.get("sample_task_count"),
            },
        )
        if not sample_gate.get("full_generation_allowed"):
            print(
                f"[想法 #{idea_idx}] ⚠️  Sample gate 未通过: "
                f"{sample_gate.get('result', {}).get('reasons', [])}"
            )
            return {
                "idea_idx": idea_idx,
                "exp_dir": exp_dir,
                "status": "failed",
                "registry_status": registry_status,
                "research_status": "blocked",
                "stage": "sample_gate",
                "sample_gate": sample_gate,
                "pause_reason": "unsatisfiable_gate",
                "gate_reasons": ["sample_gate_blocked"],
                "resumable": False,
            }
        assert_gate_preconditions_satisfiable(
            GatePreconditionContext(
                research_plan=research_plan,
                sample_gate=sample_gate,
                improvement_rounds=improvement_rounds,
                require_quality_gate=require_quality_gate,
                high_quality_mode=high_quality_mode,
                target_venue=target_venue,
            )
        )

        # ========== 步骤2: 生成论文 ==========
        current_stage = "plot_aggregation"
        print(f"\n📝 [想法 #{idea_idx}] 步骤 2/4: 生成论文")
        aggregate_plots(
            base_folder=exp_dir,
            model=model_agg_plots,
            execution_config_path=idea_config_path,
        )
        current_stage = "writeup"
        _save_manuscript_contract_state(
            exp_dir=exp_dir,
            writeup_type=writeup_type,
            target_venue=target_venue,
            writing_profile=writing_profile,
        )

        # Keep the original experiment outputs available for audit and
        # reproduction.  Manuscript generation reads the frozen registry and
        # snapshot rather than deleting the source results directory.

        writeup_plan = build_writeup_execution_plan(
            writeup_type,
            num_cite_rounds=num_cite_rounds,
            writeup_retries=writeup_retries,
            target_venue=target_venue,
            high_quality_mode=high_quality_mode,
            research_root=research_root,
        )
        selected_venue = writeup_plan["target_venue"]
        strategy_feedback = writeup_plan["strategy_feedback"]
        effective_num_cite_rounds = writeup_plan["num_cite_rounds"]
        effective_writeup_retries = writeup_plan["writeup_retries"]
        _save_manuscript_contract_state(
            exp_dir=exp_dir,
            writeup_type=writeup_type,
            target_venue=selected_venue,
            writing_profile=writing_profile,
        )
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 历史预算反馈: {strategy_feedback.get('rationale', [])}"
            )

        citations_text = gather_citations(
            exp_dir,
            num_cite_rounds=effective_num_cite_rounds,
            small_model=model_citation,
        )

        writeup_success = False
        writeup_func = (
            perform_normal_writeup
            if writeup_plan["writeup_engine"] == "normal"
            else perform_icbinb_writeup
        )
        page_limit = writeup_plan["page_limit"]
        for attempt in range(effective_writeup_retries):
            print(
                f"[想法 #{idea_idx}] 论文写作尝试 {attempt + 1}/{effective_writeup_retries}"
            )
            writeup_success = writeup_func(
                base_folder=exp_dir,
                small_model=model_writeup_small,
                big_model=model_writeup,
                page_limit=page_limit,
                citations_text=citations_text,
                writing_profile=writing_profile,
                writing_audit_rounds=writing_audit_rounds,
                target_venue=selected_venue,
                strict_guardrails=strict_writing_guardrails,
                guardrail_repair_rounds=guardrail_repair_rounds,
            )

            if writeup_success:
                print(f"[想法 #{idea_idx}] ✅ 论文写作成功!")
                break
            else:
                print(f"[想法 #{idea_idx}] ⚠️  尝试 {attempt + 1} 失败")

        if not writeup_success:
            print(f"[想法 #{idea_idx}] ❌ 论文写作失败")
            guardrail_findings, guardrail_reasons = load_guardrail_artifacts(exp_dir)
            guardrail_blocking = bool(guardrail_reasons)
            return {
                **_paper_result_artifact_fields(exp_dir),
                "idea_idx": idea_idx,
                "status": "failed",
                "registry_status": registry_status,
                "stage": "writeup",
                "writing_profile": writing_profile,
                "writing_audit_rounds": writing_audit_rounds,
                "strict_writing_guardrails": strict_writing_guardrails,
                "guardrail_repair_rounds": guardrail_repair_rounds,
                "guardrail_blocking": guardrail_blocking,
                "guardrail_blocking_reasons": guardrail_reasons,
                "guardrail_findings": guardrail_findings,
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
            }

        quality_policy = research_plan.get("execution_policy") or {}
        quality_pass = {}
        quality_result = {}
        acceptance = {}
        final_submission_gate = {}
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 质量兜底策略: "
                f"{quality_policy.get('quality_fallback_policy', 'allowed')}"
            )
            quality_pass = execute_quality_workflow_with_followups(
                run_high_quality_pass_fn=run_high_quality_pass,
                run_dir=exp_dir,
                paper_type=writeup_type,
                rewrite_model=model_writeup,
                quality_model=quality_model or model_review,
                target_venue=selected_venue,
                quality_preset=quality_preset,
                quality_threshold=quality_threshold,
                rigor_threshold=rigor_threshold,
                max_quality_rewrites=quality_rewrite_rounds,
                require_quality_gate=require_quality_gate,
                min_submission_priority=min_submission_priority,
                max_submission_blockers=max_submission_blockers,
                autonomous_followup_rounds=autonomous_quality_followup_rounds,
                allow_auto_improvement_fallback=quality_policy.get(
                    "allow_auto_improvement_fallback"
                ),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                resume=False,
                logger=lambda msg: print(f"[想法 #{idea_idx}] {msg}"),
            )
            quality_result = quality_pass["quality_result"]
            quality_fallback_event = record_quality_fallback_if_needed(
                exp_dir,
                quality_result,
                producer="run_project.high_quality",
                strict=strict_fallbacks,
            )
            if strict_fallbacks and quality_fallback_event:
                return {
                    **_paper_result_artifact_fields(exp_dir),
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "registry_status": registry_status,
                    "stage": "quality_fallback_blocked",
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "reason": format_strict_fallback_error(
                        quality_fallback_event,
                        workflow_mode=workflow_mode,
                        stage_hint="quality_review",
                    ),
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                    "scientific_evidence_status": quality_result.get(
                        "scientific_evidence_status"
                    ),
                    "scientific_evidence_failures": quality_result.get(
                        "scientific_evidence_failures", []
                    ),
                    "scientific_evidence_next_actions": quality_result.get(
                        "scientific_evidence_next_actions", []
                    ),
                }
            acceptance = quality_pass["acceptance"]
            print(
                f"[想法 #{idea_idx}] {quality_pass['summary'].replace('High-quality pass: ', '高质量模式: ')}"
            )
            if not acceptance.get("accepted"):
                print(
                    f"[想法 #{idea_idx}] ⚠️  预审质量门槛暂未满足，将继续进入严格 review/self-improvement："
                    f"{acceptance.get('reasons', [])}"
                )

        # ========== 步骤3: 反思和改进 ==========
        review_plan = build_review_execution_plan(
            writeup_type,
            target_venue=selected_venue,
            review_reflections=review_reflections,
            review_ensemble=review_ensemble,
            review_fewshot=review_fewshot,
            review_temperature=review_temperature,
            review_strategy=review_strategy,
            high_quality_mode=high_quality_mode,
            research_root=research_root,
            default_quality_requirement=(
                "high" if improvement_rounds > 0 else "standard"
            ),
        )
        workflow_runtime_plan = build_workflow_runtime_plan(
            workflow_spec,
            submission_mode=(template_profile == "template_first"),
            high_quality_mode=high_quality_mode,
            target_venue=selected_venue,
        )
        print(
            f"[想法 #{idea_idx}] 审稿编排: 改进轮={list(workflow_runtime_plan.improvement_review_roles)} "
            f"终审={list(workflow_runtime_plan.final_review_roles)}"
        )
        if high_quality_mode:
            print(
                f"[想法 #{idea_idx}] 历史审稿反馈: "
                f"{review_plan['strategy_feedback'].get('rationale', [])}"
            )

        previous_issue_ledger = None
        self_review_round_records = []
        self_review_summary_file = None
        self_review_summary_payload = None
        experiment_todo_payload = load_experiment_todo_payload(exp_dir)
        experiment_todo_round_snapshots = []
        experiment_todo_progress_payload = None
        experiment_todo_progress_files = {}
        final_experiment_todo_snapshot = None

        for round_num in range(improvement_rounds):
            print(
                f"\n🔍 [想法 #{idea_idx}] 步骤 3/4: 反思改进轮次 {round_num + 1}/{improvement_rounds}"
            )

            review_dir = osp.join(exp_dir, f"reviews_round_{round_num + 1}")
            review_pass = execute_review_suite(
                review_roles=workflow_runtime_plan.improvement_review_roles,
                paper_dir=exp_dir,
                model_review=model_review,
                review_plan=review_plan,
                create_client_fn=create_client,
                load_paper_fn=load_paper,
                perform_review_fn=perform_review,
                perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
                pdf_path_resolver=find_latest_pdf,
                save_dir=review_dir,
                project_root=exp_dir,
                persist_job=True,
                evidence_refs=[
                    "claim_evidence_graph.json",
                    "experiment_registry.jsonl",
                    "figure_spec.json",
                    "manuscript_state.json",
                ],
                suite_name=f"improvement_round_{round_num + 1}",
                lane_name="review_board",
                strictness_profile="standard",
            )
            if not review_pass["found"]:
                print(f"[想法 #{idea_idx}] ⚠️  未找到PDF，跳过改进")
                break

            review_text = review_pass["review_text"]
            review_img = review_pass["review_img"]

            # 基于 issue ledger 的改进论文
            improvement_result = improve_paper_with_review(
                exp_dir,
                review_text,
                review_img,
                model_writeup,
                round_index=round_num + 1,
                review_dir=review_dir,
                previous_issue_ledger=previous_issue_ledger,
                target_venue=selected_venue,
            )
            round_gate = (
                improvement_result.get("round_gate")
                if isinstance(improvement_result.get("round_gate"), dict)
                else {}
            )
            issue_progress = (
                improvement_result.get("issue_progress")
                if isinstance(improvement_result.get("issue_progress"), dict)
                else None
            )

            if not (experiment_todo_payload.get("tasks") or []):
                bootstrap_tasks = bootstrap_todo_tasks_from_round_gate(
                    round_gate,
                    prefix=f"idea{idea_idx}",
                    max_tasks=8,
                )
                if bootstrap_tasks:
                    experiment_todo_payload["tasks"] = bootstrap_tasks
                    experiment_todo_payload["generated_at"] = datetime.now().isoformat()
                    experiment_todo_payload["bootstrap"] = True
                    todo_file = Path(exp_dir) / "experiment_todo.json"
                    todo_file.write_text(
                        json.dumps(
                            experiment_todo_payload, indent=2, ensure_ascii=False
                        ),
                        encoding="utf-8",
                    )
                    experiment_todo_payload["file"] = str(todo_file)

            round_todo_snapshot = evaluate_todo_progress_snapshot(
                experiment_todo_payload,
                round_gate=round_gate,
                issue_progress=issue_progress,
                round_index=round_num + 1,
            )
            if (
                int((round_todo_snapshot.get("counts") or {}).get("total_tasks") or 0)
                > 0
            ):
                experiment_todo_round_snapshots.append(round_todo_snapshot)
                print(
                    f"[想法 #{idea_idx}] TODO进展: closure={round_todo_snapshot.get('closure_rate')} "
                    f"p0_closure={round_todo_snapshot.get('p0_closure_rate')} "
                    f"unresolved={round_todo_snapshot.get('counts', {}).get('unresolved_tasks')}"
                )

            self_review_round_records.append(
                {
                    "round": round_num + 1,
                    "issue_count": (
                        improvement_result.get("issue_ledger", {}).get("issue_count")
                        if isinstance(improvement_result.get("issue_ledger"), dict)
                        else None
                    ),
                    "critical_count": (
                        improvement_result.get("issue_ledger", {}).get("critical_count")
                        if isinstance(improvement_result.get("issue_ledger"), dict)
                        else None
                    ),
                    "progress": issue_progress,
                    "round_gate": round_gate,
                    "rewrite": improvement_result.get("rewrite_result"),
                    "artifacts": improvement_result.get("artifact_files"),
                    "experiment_todo_progress": (
                        round_todo_snapshot
                        if int(
                            (round_todo_snapshot.get("counts") or {}).get("total_tasks")
                            or 0
                        )
                        > 0
                        else None
                    ),
                }
            )

            if isinstance(improvement_result.get("issue_ledger"), dict):
                previous_issue_ledger = improvement_result["issue_ledger"]

            rewrite_payload = improvement_result.get("rewrite_result", {})
            if rewrite_payload.get("status") != "success":
                print(
                    f"[想法 #{idea_idx}] ⚠️  本轮改稿未成功，提前结束改进循环: "
                    f"{rewrite_payload.get('status')}"
                )
                break

            if isinstance(round_gate, dict) and round_gate.get("ready"):
                print(
                    f"[想法 #{idea_idx}] ✅ Round gate 已达标，提前结束改进循环 "
                    f"(score={round_gate.get('score')})"
                )
                break

            if isinstance(issue_progress, dict):
                if (
                    issue_progress.get("unresolved_critical_count", 0) == 0
                    and issue_progress.get("persistent_issue_count", 0) == 0
                    and (round_num + 1) >= 2
                ):
                    print(
                        f"[想法 #{idea_idx}] ✅ 问题台账已收敛（无未解决关键问题且无持续问题），提前结束改进循环"
                    )
                    break

            if (
                isinstance(improvement_result.get("issue_ledger"), dict)
                and improvement_result["issue_ledger"].get("issue_count", 0) == 0
            ):
                print(f"[想法 #{idea_idx}] ✅ 本轮未发现可执行问题，提前结束改进循环")
                break

        if self_review_round_records:
            self_review_summary_file = osp.join(
                exp_dir, "self_review_iteration_summary.json"
            )
            latest_round_gate = self_review_round_records[-1].get("round_gate")
            self_review_summary_payload = {
                "generated_at": datetime.now().isoformat(),
                "rounds": self_review_round_records,
                "rounds_completed": len(self_review_round_records),
                "latest_issue_ledger": previous_issue_ledger,
                "latest_round_gate": latest_round_gate,
                "round_gate_ready": bool(
                    isinstance(latest_round_gate, dict)
                    and latest_round_gate.get("ready")
                ),
                "experiment_todo_round_snapshots": experiment_todo_round_snapshots,
            }
            with open(self_review_summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    self_review_summary_payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if high_quality_mode:
            print(f"\n🧪 [想法 #{idea_idx}] 最终投稿级质量复核")
            final_quality_pass = execute_quality_workflow_with_followups(
                run_high_quality_pass_fn=run_high_quality_pass,
                run_dir=exp_dir,
                paper_type=writeup_type,
                rewrite_model=model_writeup,
                quality_model=quality_model or model_review,
                target_venue=selected_venue,
                quality_preset=quality_preset,
                quality_threshold=quality_threshold,
                rigor_threshold=rigor_threshold,
                max_quality_rewrites=quality_rewrite_rounds,
                require_quality_gate=require_quality_gate,
                min_submission_priority=min_submission_priority,
                max_submission_blockers=max_submission_blockers,
                autonomous_followup_rounds=autonomous_quality_followup_rounds,
                allow_auto_improvement_fallback=quality_policy.get(
                    "allow_auto_improvement_fallback"
                ),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                resume=False,
                logger=lambda msg: print(f"[想法 #{idea_idx}] {msg}"),
            )
            quality_pass = final_quality_pass
            quality_result = final_quality_pass["quality_result"]
            acceptance = final_quality_pass["acceptance"]
            quality_fallback_event = record_quality_fallback_if_needed(
                exp_dir,
                quality_result,
                producer="run_project.final_quality",
                strict=strict_fallbacks,
            )
            if strict_fallbacks and quality_fallback_event:
                return {
                    **_paper_result_artifact_fields(exp_dir),
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "registry_status": registry_status,
                    "stage": "quality_fallback_blocked",
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "reason": format_strict_fallback_error(
                        quality_fallback_event,
                        workflow_mode=workflow_mode,
                        stage_hint="quality_review",
                    ),
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                }
            print(
                f"[想法 #{idea_idx}] {final_quality_pass['summary'].replace('High-quality pass: ', '最终质量复核: ')}"
            )

        # ========== 步骤4: 最终审查 ==========
        print(f"\n🎯 [想法 #{idea_idx}] 步骤 4/4: 最终审查")

        final_review_pass = execute_review_suite(
            review_roles=workflow_runtime_plan.final_review_roles,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf,
            save_dir=exp_dir,
            text_filename="final_review.json",
            image_filename="final_review_img.json",
            project_root=exp_dir,
            persist_job=True,
            evidence_refs=[
                "claim_evidence_graph.json",
                "experiment_registry.jsonl",
                "figure_spec.json",
                "manuscript_state.json",
            ],
            suite_name="final_review",
            lane_name="review_board",
            strictness_profile="standard",
        )
        pdf_path = final_review_pass["pdf_path"]
        critic_pass = run_independent_critic_pass(
            workflow_runtime_plan=workflow_runtime_plan,
            paper_dir=exp_dir,
            model_review=model_review,
            review_plan=review_plan,
            create_client_fn=create_client,
            load_paper_fn=load_paper,
            perform_review_fn=perform_review,
            perform_imgs_cap_ref_review_fn=perform_imgs_cap_ref_review,
            pdf_path_resolver=find_latest_pdf,
            save_dir=Path(exp_dir) / "hostile_critic",
            project_root=exp_dir,
            evidence_refs=[
                "claim_evidence_graph.json",
                "experiment_registry.jsonl",
                "figure_spec.json",
                "manuscript_state.json",
            ],
        )
        if critic_pass.get("ran"):
            print(
                f"[想法 #{idea_idx}] 尖锐 reviewer: roles={critic_pass.get('review_roles_used', [])} "
                f"active={critic_pass.get('active_issue_count', 0)} "
                f"blocking={critic_pass.get('blocking_issue_count', 0)}"
            )
        if critic_pass.get("blocking_issue_count", 0) > 0:
            return {
                **_paper_result_artifact_fields(
                    exp_dir,
                    pdf_path=pdf_path,
                    critic_findings_file=critic_pass.get("critic_findings_file"),
                ),
                "idea_idx": idea_idx,
                "status": "failed",
                "registry_status": registry_status,
                "stage": "hostile_critic",
                "workflow_mode": workflow_mode,
                "template_profile": template_profile,
                "template_capability": template_capability,
                "review_roles_used": sorted(
                    set(workflow_runtime_plan.improvement_review_roles)
                    | set(workflow_runtime_plan.final_review_roles)
                    | set(workflow_runtime_plan.critic_review_roles)
                ),
                "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
                "critic_active_issue_count": critic_pass.get("active_issue_count"),
                "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
                "reason": "Independent hostile critic reported blocking issues.",
            }
        final_self_review_progress = None
        final_self_review_progress_file = None
        if final_review_pass["found"]:
            if isinstance(previous_issue_ledger, dict):
                final_issue_ledger = build_issue_ledger(
                    text_review=final_review_pass.get("review_text"),
                    img_review=final_review_pass.get("review_img"),
                    max_issues=14,
                    previous_ledger=previous_issue_ledger,
                    target_venue=selected_venue,
                )
                final_self_review_progress = evaluate_issue_progress(
                    previous_issues=list(previous_issue_ledger.get("issues") or []),
                    current_issues=list(final_issue_ledger.get("issues") or []),
                )
                final_self_review_progress_file = osp.join(
                    exp_dir, "self_review_final_progress.json"
                )
                with open(final_self_review_progress_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "generated_at": datetime.now().isoformat(),
                            "final_progress": final_self_review_progress,
                            "final_issue_ledger": final_issue_ledger,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

            # 复制最终PDF到papers目录
            final_pdf_name = f"{idea_key}_final.pdf"
            final_pdf_dst = str(confined_path(project_dir, "03_papers", final_pdf_name))
            if osp.exists(pdf_path):
                shutil.copy(pdf_path, final_pdf_dst)

        integrity_forensics_result = run_integrity_forensics_for_manuscript(
            root=exp_dir,
            paper_id=idea_key,
            enabled=bool(integrity_forensics_enabled),
        )
        if integrity_forensics_result.get("status") == "completed":
            print(
                f"[想法 #{idea_idx}] Integrity forensics: "
                f"verdict={integrity_forensics_result.get('overall_verdict')} "
                f"findings={integrity_forensics_result.get('finding_count')} "
                f"dir={integrity_forensics_result.get('output_dir')}"
            )
        elif integrity_forensics_result.get("enabled"):
            print(
                f"[想法 #{idea_idx}] ⚠️  Integrity forensics "
                f"{integrity_forensics_result.get('status')}: "
                f"{integrity_forensics_result.get('reason') or integrity_forensics_result.get('error')}"
            )

        # ========== ARA finalisation ==========
        # Delegated to ai_scientist/utils/ara_pipeline.py; each stage catches
        # its own exceptions so a claim-scan failure doesn't lose the export.
        ara_result = finalize_ara_for_idea(
            project_dir=project_dir,
            exp_dir=exp_dir,
            idea=idea,
            timestamp=timestamp,
            bfts_config_path=config_path,
            model_spec={
                "writeup": model_writeup,
                "writeup_small": model_writeup_small,
                "citation": model_citation,
                "review": model_review,
                "agg_plots": model_agg_plots,
                "quality": quality_model,
            },
            writing_profile=writing_profile,
        )
        for line in summarise_finalize(idea_idx, ara_result):
            print(line)
        ara_export = ara_result.export

        final_round_gate_for_todo = (
            self_review_round_records[-1].get("round_gate")
            if self_review_round_records
            and isinstance(self_review_round_records[-1].get("round_gate"), dict)
            else {}
        )
        final_experiment_todo_snapshot = evaluate_todo_progress_snapshot(
            experiment_todo_payload,
            round_gate=final_round_gate_for_todo,
            issue_progress=(
                final_self_review_progress
                if isinstance(final_self_review_progress, dict)
                else None
            ),
            round_index=(len(self_review_round_records) or None),
        )
        if (
            int(
                (final_experiment_todo_snapshot.get("counts") or {}).get("total_tasks")
                or 0
            )
            > 0
        ):
            experiment_todo_progress_payload = build_todo_progress_payload(
                experiment_todo_payload,
                round_snapshots=experiment_todo_round_snapshots,
                final_snapshot=final_experiment_todo_snapshot,
            )
            experiment_todo_progress_files = save_todo_progress_artifacts(
                exp_dir,
                experiment_todo_progress_payload,
            )
            print(
                f"[想法 #{idea_idx}] TODO闭环: closure={final_experiment_todo_snapshot.get('closure_rate')} "
                f"p0_closure={final_experiment_todo_snapshot.get('p0_closure_rate')} "
                f"unresolved={final_experiment_todo_snapshot.get('counts', {}).get('unresolved_tasks')}"
            )
        if self_review_summary_file and self_review_summary_payload is not None:
            self_review_summary_payload["experiment_todo_progress_file"] = (
                experiment_todo_progress_files.get("json")
            )
            self_review_summary_payload["experiment_todo_progress_markdown_file"] = (
                experiment_todo_progress_files.get("markdown")
            )
            self_review_summary_payload["experiment_todo_final_snapshot"] = (
                final_experiment_todo_snapshot
                if int(
                    (final_experiment_todo_snapshot.get("counts") or {}).get(
                        "total_tasks"
                    )
                    or 0
                )
                > 0
                else None
            )
            with open(self_review_summary_file, "w", encoding="utf-8") as f:
                json.dump(
                    self_review_summary_payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if high_quality_mode:
            final_submission_gate = evaluate_final_submission_readiness(
                run_dir=exp_dir,
                quality_result=quality_result,
                require_quality_gate=require_quality_gate,
                min_submission_priority=quality_pass.get("effective_priority_bar"),
                max_submission_blockers=quality_pass.get("effective_blocker_bar"),
                reject_on_auto_improvement_fallback=bool(
                    quality_policy.get("reject_on_auto_improvement_fallback")
                ),
                final_issue_progress=(
                    final_self_review_progress
                    if isinstance(final_self_review_progress, dict)
                    else None
                ),
                final_todo_snapshot=(
                    final_experiment_todo_snapshot
                    if int(
                        (final_experiment_todo_snapshot.get("counts") or {}).get(
                            "total_tasks"
                        )
                        or 0
                    )
                    > 0
                    else None
                ),
                integrity_report=integrity_report_payload(integrity_forensics_result),
            )
            if not final_submission_gate.get("accepted"):
                return {
                    **_paper_result_artifact_fields(exp_dir, pdf_path=pdf_path),
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "registry_status": registry_status,
                    "research_status": "manuscript_draft",
                    "stage": "final_submission_bar",
                    "quality_score": quality_result.get("quality_score_after"),
                    "rigor_score": quality_result.get("rigor_score_after"),
                    "claim_support_score": quality_result.get("claim_support_after"),
                    "submission_priority_score": quality_result.get(
                        "submission_priority_score"
                    ),
                    "submission_priority_tier": quality_result.get(
                        "submission_priority_tier"
                    ),
                    "blocker_count": quality_result.get("blocker_count"),
                    "autonomous_followup_rounds_run": quality_result.get(
                        "autonomous_followup_rounds_run"
                    ),
                    "quality_fallback_policy": quality_policy.get(
                        "quality_fallback_policy"
                    ),
                    "acceptance_reasons": final_submission_gate.get("reasons", []),
                    "acceptance_signals": final_submission_gate.get("signals", {}),
                    "integrity_forensics": integrity_forensics_result,
                    "workflow_mode": workflow_mode,
                    "template_profile": template_profile,
                    "template_capability": template_capability,
                }

        print(f"\n✅ [想法 #{idea_idx}] 完成! 结果保存在: {exp_dir}")

        return {
            "idea_idx": idea_idx,
            "status": "success",
            "registry_status": registry_status,
            "research_status": (
                "submission_ready"
                if high_quality_mode and final_submission_gate.get("accepted") is True
                else ("manuscript_draft" if high_quality_mode else "exploratory_draft")
            ),
            "exp_dir": exp_dir,
            "pdf_path": pdf_path if pdf_path and osp.exists(pdf_path) else None,
            "ara_dir": str(ara_export.root) if ara_export is not None else None,
            "ara_manifest": (
                str(ara_export.manifest_path) if ara_export is not None else None
            ),
            "ara_node_count": ara_export.node_count if ara_export is not None else None,
            "ara_claim_coverage_score": (
                ara_result.claim_coverage.coverage_score
                if ara_result.claim_coverage is not None
                else None
            ),
            "ara_claim_coverage_severity": (
                ara_result.claim_coverage.severity
                if ara_result.claim_coverage is not None
                else None
            ),
            "workflow_mode": workflow_mode,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "execution_policy": (research_plan.get("execution_policy") or {}).get(
                "policy_name"
            ),
            "evidence_pressure": (research_plan.get("execution_policy") or {}).get(
                "evidence_pressure"
            ),
            "quality_fallback_policy": (
                research_plan.get("execution_policy") or {}
            ).get("quality_fallback_policy"),
            "review_roles_improvement": list(
                workflow_runtime_plan.improvement_review_roles
            ),
            "review_roles_final": list(workflow_runtime_plan.final_review_roles),
            "review_roles_used": sorted(
                set(workflow_runtime_plan.improvement_review_roles)
                | set(workflow_runtime_plan.final_review_roles)
                | set(workflow_runtime_plan.critic_review_roles)
            ),
            "critic_roles_used": list(workflow_runtime_plan.critic_review_roles),
            "critic_active_issue_count": critic_pass.get("active_issue_count"),
            "critic_blocking_issue_count": critic_pass.get("blocking_issue_count"),
            "critic_findings_file": critic_pass.get("critic_findings_file"),
            "pipeline_manifest": str(Path(exp_dir) / "pipeline_manifest.json"),
            "idea_card_id": idea_card.get("idea_id"),
            "research_plan_file": str(Path(exp_dir) / "research_plan.json"),
            "claim_evidence_graph_file": str(
                Path(exp_dir) / "claim_evidence_graph.json"
            ),
            "experiment_registry_file": str(
                Path(exp_dir) / "experiment_registry.jsonl"
            ),
            "figure_spec_file": str(Path(exp_dir) / "figure_spec.json"),
            "plot_execution_receipt_file": str(
                Path(exp_dir) / "plot_execution_receipt.json"
            ),
            "manuscript_state_file": str(Path(exp_dir) / "manuscript_state.json"),
            "review_state_file": str(Path(exp_dir) / "review_state.json"),
            "repair_plan_file": str(Path(exp_dir) / "repair_plan.json"),
            "self_evolution_file": str(Path(exp_dir) / "self_evolution.json"),
            "stage_standards_file": str(Path(exp_dir) / "stage_standards.json"),
            "quality_score": (
                quality_result.get("quality_score_after") if high_quality_mode else None
            ),
            "rigor_score": (
                quality_result.get("rigor_score_after") if high_quality_mode else None
            ),
            "claim_support_score": (
                quality_result.get("claim_support_after") if high_quality_mode else None
            ),
            "quality_gate_passed": (
                quality_result.get("quality_gate_passed") if high_quality_mode else None
            ),
            "scientific_evidence_status": (
                quality_result.get("scientific_evidence_status")
                if high_quality_mode
                else None
            ),
            "scientific_evidence_failures": (
                quality_result.get("scientific_evidence_failures", [])
                if high_quality_mode
                else []
            ),
            "scientific_evidence_next_actions": (
                quality_result.get("scientific_evidence_next_actions", [])
                if high_quality_mode
                else []
            ),
            "submission_priority_score": (
                quality_result.get("submission_priority_score")
                if high_quality_mode
                else None
            ),
            "submission_priority_tier": (
                quality_result.get("submission_priority_tier")
                if high_quality_mode
                else None
            ),
            "blocker_count": (
                quality_result.get("blocker_count") if high_quality_mode else None
            ),
            "submission_acceptance_passed": (
                final_submission_gate.get("accepted") if high_quality_mode else None
            ),
            "local_submission_ready": (
                final_submission_gate.get("accepted") if high_quality_mode else None
            ),
            "acceptance_guaranteed": False,
            "submission_acceptance_reasons": (
                final_submission_gate.get("reasons", []) if high_quality_mode else []
            ),
            **integrity_forensics_result_fields(integrity_forensics_result),
            "quality_acceptance_passed": (
                acceptance.get("accepted") if high_quality_mode else None
            ),
            "auto_improvement_fallback_used": (
                quality_result.get("auto_improvement_fallback_used")
                if high_quality_mode
                else None
            ),
            "submission_package_file": (
                quality_result.get("submission_package_file")
                if high_quality_mode
                else None
            ),
            "submission_dashboard_file": (
                quality_result.get("submission_dashboard_file")
                if high_quality_mode
                else None
            ),
            "autonomous_followup_rounds_run": (
                quality_result.get("autonomous_followup_rounds_run")
                if high_quality_mode
                else None
            ),
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_writing_guardrails": strict_writing_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "self_review_rounds_completed": len(self_review_round_records),
            "self_review_summary_file": self_review_summary_file,
            "self_review_final_progress_file": final_self_review_progress_file,
            "self_review_unresolved_critical": (
                final_self_review_progress.get("unresolved_critical_count")
                if isinstance(final_self_review_progress, dict)
                else None
            ),
            "self_review_round_gate_ready": (
                bool(
                    isinstance(self_review_round_records[-1].get("round_gate"), dict)
                    and self_review_round_records[-1]["round_gate"].get("ready")
                )
                if self_review_round_records
                else None
            ),
            "experiment_todo_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("total_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_p0_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("p0_total")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_closed_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("closed_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_unresolved_count": (
                final_experiment_todo_snapshot.get("counts", {}).get("unresolved_tasks")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_closure_rate": (
                final_experiment_todo_snapshot.get("closure_rate")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_p0_closure_rate": (
                final_experiment_todo_snapshot.get("p0_closure_rate")
                if isinstance(final_experiment_todo_snapshot, dict)
                else None
            ),
            "experiment_todo_progress_file": experiment_todo_progress_files.get("json"),
            "experiment_todo_progress_markdown_file": experiment_todo_progress_files.get(
                "markdown"
            ),
        }

    except Exception as e:
        print(f"\n❌ [想法 #{idea_idx}] 处理失败: {e}")
        traceback.print_exc()
        failure_runtime_plan = build_workflow_runtime_plan(
            workflow_mode,
            submission_mode=(template_profile == "template_first"),
            high_quality_mode=high_quality_mode,
            target_venue=target_venue,
        )
        return {
            **(_paper_result_artifact_fields(exp_dir) if "exp_dir" in locals() else {}),
            "idea_idx": idea_idx,
            "status": "failed",
            "registry_status": registry_status,
            "stage": current_stage,
            "error": str(e),
            "pause_reason": (
                "unsatisfiable_gate" if isinstance(e, UnsatisfiableGateError) else None
            ),
            "gate_reasons": (
                str(e).replace("Unsatisfiable gate preconditions: ", "").split(", ")
                if isinstance(e, UnsatisfiableGateError)
                else []
            ),
            "traceback": traceback.format_exc(),
            "writing_profile": writing_profile,
            "writing_audit_rounds": writing_audit_rounds,
            "strict_writing_guardrails": strict_writing_guardrails,
            "guardrail_repair_rounds": guardrail_repair_rounds,
            "workflow_mode": workflow_mode,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "execution_policy": (
                (research_plan.get("execution_policy") or {}).get("policy_name")
                if "research_plan" in locals()
                else None
            ),
            "review_roles_improvement": list(
                failure_runtime_plan.improvement_review_roles
            ),
            "review_roles_final": list(failure_runtime_plan.final_review_roles),
            "review_roles_used": sorted(
                set(failure_runtime_plan.improvement_review_roles)
                | set(failure_runtime_plan.final_review_roles)
                | set(failure_runtime_plan.critic_review_roles)
            ),
            "critic_roles_used": list(failure_runtime_plan.critic_review_roles),
        }
    finally:
        # Clear tracer env vars so a subsequent process_single_idea call
        # (multi-idea batches) starts clean and cannot leak this idea's ARA
        # root into the next one.
        deactivate_llm_tracing()


def _process_admission_receipt(process_args: object) -> Mapping[str, Any] | None:
    if not isinstance(process_args, (list, tuple)) or len(process_args) != 42:
        return None
    candidate = process_args[-1]
    return candidate if isinstance(candidate, Mapping) else None


def _worker_terminal_provenance(
    project_dir: str | Path,
    receipt: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze provenance now so recovery never re-hashes a changed checkout."""

    scope = receipt.get("execution_scope")
    scope = scope if isinstance(scope, Mapping) else {}
    fallback = {
        field: scope[field]
        for field in ("environment_hash", "code_commit")
        if isinstance(scope.get(field), str) and scope.get(field)
    }
    try:
        from xscientist.research_lifecycle import ResearchLifecycle

        lifecycle = ResearchLifecycle(project_dir)
        provenance = _research_attempt_provenance(
            lifecycle,
            result=dict(result),
            preregistration_id=(
                str(receipt["preregistration_id"])
                if receipt.get("preregistration_id")
                else None
            ),
        )
        if isinstance(scope.get("code_commit"), str) and scope.get("code_commit"):
            provenance["code_commit"] = scope["code_commit"]
        return provenance
    except Exception:
        return fallback


def _publish_process_worker_terminal(
    process_args: object,
    result: Mapping[str, Any],
    *,
    failure: object = None,
) -> None:
    receipt = _process_admission_receipt(process_args)
    if receipt is None:
        return
    project_dir = process_args[0]  # type: ignore[index]
    from xscientist.experiment_admission_runtime import (
        load_execution_claim_v2,
        publish_worker_terminal,
    )

    start_object_id = str(receipt.get("start_object_id") or "")
    claim = load_execution_claim_v2(
        project_dir,
        start_object_id=start_object_id,
        receipt_hash=str(receipt.get("receipt_hash") or ""),
    )
    if claim is None:
        return
    publish_worker_terminal(
        project_dir,
        claim,
        result,
        provenance=_worker_terminal_provenance(project_dir, receipt, result),
        failure=failure,
    )


def process_single_idea(args):
    """Run one idea and publish a durable worker terminal for every return."""

    try:
        result = _process_single_idea_impl(args)
    except BaseException as exc:
        receipt = _process_admission_receipt(args)
        if receipt is not None:
            scope = receipt.get("execution_scope")
            scope = scope if isinstance(scope, Mapping) else {}
            terminal_result = {
                "idea_idx": scope.get("idea_index", 0),
                "status": "failed",
                "registry_status": "failed",
                "research_status": "failed",
                "stage": "worker_aborted",
                "resumable": False,
            }
            _publish_process_worker_terminal(
                args,
                terminal_result,
                failure=exc,
            )
        raise
    if not isinstance(result, Mapping):
        raise RuntimeError("experiment worker returned an invalid result")
    _publish_process_worker_terminal(
        args,
        result,
        failure=result.get("error"),
    )
    return dict(result)


def run_parallel_experiments(
    project_dir, idea_json, num_workers, idea_indices, **kwargs
):
    """并行运行多个实验"""
    print("\n" + "=" * 80)
    print(f"🚀 并行运行 {len(idea_indices)} 个实验 (使用 {num_workers} 个worker)")
    print("=" * 80)

    # 加载想法
    with open(idea_json, "r") as f:
        ideas = json.load(f)

    # 准备参数
    args_list = []
    for idx in idea_indices:
        if idx >= len(ideas):
            print(f"⚠️  想法索引 {idx} 超出范围，跳过")
            continue
        args_list.append(
            (
                project_dir,
                kwargs["output_root"],
                idx,
                ideas[idx],
                kwargs["bfts_config"],
                kwargs["model_writeup"],
                kwargs["model_citation"],
                kwargs["model_review"],
                kwargs["model_agg_plots"],
                kwargs["model_writeup_small"],
                kwargs["num_cite_rounds"],
                kwargs["writeup_retries"],
                kwargs["writeup_type"],
                kwargs["improvement_rounds"],
                kwargs["review_reflections"],
                kwargs["review_ensemble"],
                kwargs["review_fewshot"],
                kwargs["review_temperature"],
                kwargs["review_strategy"],
                kwargs["high_quality_mode"],
                kwargs["quality_preset"],
                kwargs["quality_model"],
                kwargs["target_venue"],
                kwargs["quality_threshold"],
                kwargs["rigor_threshold"],
                kwargs["quality_rewrite_rounds"],
                kwargs["autonomous_quality_followup_rounds"],
                kwargs["require_quality_gate"],
                kwargs["min_submission_priority"],
                kwargs["max_submission_blockers"],
                kwargs["writing_profile"],
                kwargs["writing_audit_rounds"],
                kwargs["strict_writing_guardrails"],
                kwargs["guardrail_repair_rounds"],
                kwargs["workflow_mode"],
                kwargs["template_profile"],
                kwargs["template_capability"],
                kwargs["strict_fallbacks"],
                kwargs["integrity_forensics_enabled"],
                kwargs.get("requested_workflow_mode", kwargs["workflow_mode"]),
                (kwargs.get("resume_checkpoints") or {}).get(idx),
                (kwargs.get("admission_preflights") or {}).get(idx),
            )
        )

    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_idea, args): args[2] for args in args_list
        }

        for future in as_completed(futures):
            idea_idx = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"❌ 想法 #{idea_idx} 执行出错: {e}")
                traceback.print_exc()
                result = {
                    "idea_idx": idea_idx,
                    "status": "failed",
                    "error": str(e),
                }
            results.append(result)
            _save_project_progress(
                project_dir,
                ideas_hash=str(kwargs["ideas_hash"]),
                results=list(kwargs.get("prior_results") or []) + results,
                total=int(kwargs.get("progress_total") or len(args_list)),
                selected_indices=list(kwargs.get("selected_indices") or idea_indices),
            )

    return results


def save_project_summary(
    project_dir: str,
    results: list[dict],
    *,
    insight_report: dict[str, Any] | None = None,
) -> tuple[str, str, dict]:
    project_root = _assert_managed_project_layout(project_dir)
    logs_dir = project_root / "04_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _assert_managed_project_layout(project_root)

    quality_scores = [
        result.get("quality_score")
        for result in results
        if isinstance(result.get("quality_score"), (int, float))
    ]
    rigor_scores = [
        result.get("rigor_score")
        for result in results
        if isinstance(result.get("rigor_score"), (int, float))
    ]
    priority_scores = [
        result.get("submission_priority_score")
        for result in results
        if isinstance(result.get("submission_priority_score"), (int, float))
    ]
    followup_rounds = [
        result.get("autonomous_followup_rounds_run")
        for result in results
        if isinstance(result.get("autonomous_followup_rounds_run"), int)
    ]
    gate_passed = sum(
        1 for result in results if result.get("quality_gate_passed") is True
    )
    submission_accepted = sum(
        1 for result in results if result.get("submission_acceptance_passed") is True
    )
    guardrail_passed = sum(
        1 for result in results if result_passed_writeup_guardrails(result)
    )
    failed = [result for result in results if result.get("status") != "success"]
    guardrail_blocked = sum(
        1 for result in failed if result.get("guardrail_blocking") is True
    )
    guardrail_reason_counts: dict[str, int] = {}
    integrity_status_counts: dict[str, int] = {}
    integrity_verdict_counts: dict[str, int] = {}
    workflow_mode_counts: dict[str, int] = {}
    template_profile_counts: dict[str, int] = {}
    template_capability_counts: dict[str, int] = {}
    execution_policy_counts: dict[str, int] = {}
    review_role_counts: dict[str, int] = {}
    stage_counts: dict[str, int] = {}
    research_status_counts: dict[str, int] = {}
    scientific_evidence_status_counts: dict[str, int] = {}
    for result in failed:
        reasons = result.get("guardrail_blocking_reasons", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            key = str(reason).strip()
            if not key:
                continue
            guardrail_reason_counts[key] = guardrail_reason_counts.get(key, 0) + 1
    for result in results:
        research_status = str(result.get("research_status") or "unknown")
        research_status_counts[research_status] = (
            research_status_counts.get(research_status, 0) + 1
        )
        evidence_status = str(result.get("scientific_evidence_status") or "not_run")
        scientific_evidence_status_counts[evidence_status] = (
            scientific_evidence_status_counts.get(evidence_status, 0) + 1
        )
        workflow_mode = str(result.get("workflow_mode") or "unknown")
        template_profile = str(result.get("template_profile") or "unknown")
        template_capability = str(result.get("template_capability") or "unknown")
        workflow_mode_counts[workflow_mode] = (
            workflow_mode_counts.get(workflow_mode, 0) + 1
        )
        template_profile_counts[template_profile] = (
            template_profile_counts.get(template_profile, 0) + 1
        )
        template_capability_counts[template_capability] = (
            template_capability_counts.get(template_capability, 0) + 1
        )
        integrity = summarize_integrity_forensics_result(result)
        if integrity.get("enabled") is True:
            status = str(integrity.get("status") or "unknown")
            integrity_status_counts[status] = integrity_status_counts.get(status, 0) + 1
            verdict = str(integrity.get("verdict") or "unknown")
            integrity_verdict_counts[verdict] = (
                integrity_verdict_counts.get(verdict, 0) + 1
            )
        execution_policy = str(result.get("execution_policy") or "unknown")
        execution_policy_counts[execution_policy] = (
            execution_policy_counts.get(execution_policy, 0) + 1
        )
        review_roles = result.get("review_roles_used") or []
        if isinstance(review_roles, list):
            for role in review_roles:
                key = str(role).strip()
                if not key:
                    continue
                review_role_counts[key] = review_role_counts.get(key, 0) + 1
        if result.get("status") != "success":
            stage = str(result.get("stage") or "unknown")
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    summary = {
        "project_dir": project_dir,
        "generated_at": datetime.now().isoformat(),
        "total": len(results),
        "success": sum(1 for result in results if result.get("status") == "success"),
        "failed": sum(1 for result in results if result.get("status") != "success"),
        "quality_summary": {
            "avg_quality_score": (
                (sum(quality_scores) / len(quality_scores)) if quality_scores else None
            ),
            "avg_rigor_score": (
                (sum(rigor_scores) / len(rigor_scores)) if rigor_scores else None
            ),
            "avg_submission_priority_score": (
                (sum(priority_scores) / len(priority_scores))
                if priority_scores
                else None
            ),
            "avg_autonomous_followup_rounds": (
                (sum(followup_rounds) / len(followup_rounds))
                if followup_rounds
                else None
            ),
            "gate_passed": gate_passed,
            "submission_accepted": submission_accepted,
            "guardrail_passed": guardrail_passed,
            "guardrail_blocked": guardrail_blocked,
            "guardrail_blocking_reasons": guardrail_reason_counts,
            "integrity_forensics_enabled": sum(
                1
                for result in results
                if summarize_integrity_forensics_result(result).get("enabled") is True
            ),
            "integrity_forensics_completed": integrity_status_counts.get(
                "completed", 0
            ),
            "integrity_forensics_hard_flags": integrity_verdict_counts.get(
                "HARD_FLAGS", 0
            ),
            "integrity_forensics_soft_flags": integrity_verdict_counts.get(
                "SOFT_FLAGS", 0
            ),
            "integrity_forensics_status_counts": integrity_status_counts,
            "integrity_forensics_verdict_counts": integrity_verdict_counts,
            "research_status_counts": research_status_counts,
            "scientific_evidence_status_counts": scientific_evidence_status_counts,
        },
        "pipeline_summary": {
            "workflow_mode_counts": workflow_mode_counts,
            "template_profile_counts": template_profile_counts,
            "template_capability_counts": template_capability_counts,
            "execution_policy_counts": execution_policy_counts,
            "review_role_counts": review_role_counts,
            "failed_stage_counts": stage_counts,
        },
        "insight_summary": {
            "count": len((insight_report or {}).get("insights") or []),
            "synthesis_mode": (insight_report or {}).get("synthesis_mode"),
            "epistemic_status": (insight_report or {}).get("epistemic_status"),
            "independent_verification": (insight_report or {}).get(
                "independent_verification"
            ),
            "report_hash": (insight_report or {}).get("report_hash"),
        },
        "results": results,
    }

    persisted_summary = _portable_project_progress_payload(
        summary,
        project_dir=project_dir,
    )
    summary_file = _managed_project_file(
        project_root, "04_logs", "project_summary.json"
    )
    atomic_write_json(summary_file, persisted_summary, indent=4, default=str)

    persisted_results = list(persisted_summary.get("results") or [])
    shortlist_pool = [
        result for result in persisted_results if result.get("status") == "success"
    ] or persisted_results
    ranked = sorted(
        shortlist_pool,
        key=lambda result: (
            result.get("submission_acceptance_passed") is True,
            (
                result.get("submission_priority_score")
                if isinstance(result.get("submission_priority_score"), (int, float))
                else -1
            ),
            -(
                result.get("blocker_count")
                if isinstance(result.get("blocker_count"), int)
                else 999
            ),
            result.get("quality_gate_passed") is True,
            (
                result.get("quality_score")
                if isinstance(result.get("quality_score"), (int, float))
                else -1
            ),
            (
                result.get("rigor_score")
                if isinstance(result.get("rigor_score"), (int, float))
                else -1
            ),
        ),
        reverse=True,
    )
    shortlist_file = _managed_project_file(
        project_root, "04_logs", "submission_shortlist.md"
    )
    lines = ["# Project Submission Shortlist", ""]
    for result in ranked[:5]:
        integrity = summarize_integrity_forensics_result(result)
        lines.extend(
            [
                f"## idea #{result.get('idea_idx')}",
                f"- Status: {result.get('status')}",
                f"- Research status: {result.get('research_status', 'unknown')}",
                f"- Scientific evidence: {result.get('scientific_evidence_status', 'not_run')}",
                f"- Evidence next actions: {'; '.join(result.get('scientific_evidence_next_actions', [])[:3]) or 'none'}",
                f"- PDF: {result.get('pdf_path')}",
                f"- Submission Priority: {result.get('submission_priority_score')} ({result.get('submission_priority_tier')})",
                f"- Blockers: {result.get('blocker_count')}",
                f"- Quality: {result.get('quality_score')}",
                f"- Rigor: {result.get('rigor_score')}",
                f"- Gate Passed: {result.get('quality_gate_passed')}",
                "- Local submission-ready: "
                f"{result.get('local_submission_ready', result.get('submission_acceptance_passed'))}",
                f"- Integrity Forensics: {integrity.get('verdict')} "
                f"({integrity.get('status')}, findings={integrity.get('findings')})",
                f"- Integrity Report: {integrity.get('report_file')}",
                f"- Autonomous Followups: {result.get('autonomous_followup_rounds_run')}",
                f"- Submission Package: {result.get('submission_package_file')}",
                "",
            ]
        )
    atomic_write_text(shortlist_file, "\n".join(lines) + "\n")
    return str(summary_file), str(shortlist_file), summary


def main(argv=None):
    from ai_scientist.apps.project_cli import build_parser

    invocation_arguments = [
        str(item) for item in (sys.argv[1:] if argv is None else argv)
    ]
    default_writing_profile = resolve_writing_profile_env(
        invalid_profile_logger=lambda exc, raw: print(
            "⚠️  忽略无效的 AI_SCIENTIST_WRITING_PROFILE="
            f"{raw!r}，回退为 {DEFAULT_WRITING_PROFILE}"
        )
    )
    parser = build_parser(
        default_output_root=str(resolve_output_path()),
        default_writing_profile=default_writing_profile,
        writing_profiles=list_writing_profiles(),
        workflow_modes=list_workflow_modes(),
    )
    args = parser.parse_args(argv)
    args._project_bfts_config_cli_override = _project_cli_option_was_explicit(
        invocation_arguments,
        "--bfts-config",
    )
    args._project_budget_cli_overrides = frozenset(
        attribute
        for option, attribute in _PROJECT_BUDGET_CLI_OPTIONS.items()
        if _project_cli_option_was_explicit(invocation_arguments, option)
    )
    from ai_scientist.utils.research_model_roles import (
        enforce_glm53_project_role_boundary,
    )

    try:
        enforce_glm53_project_role_boundary(args)
    except ValueError as exc:
        parser.error(str(exc))
    if sum(bool(value) for value in (args.question, args.topic, args.ideas)) > 1:
        parser.error("--question, --topic, and --ideas are mutually exclusive")
    apply_project_autopilot_profile(args)
    require_login("项目管理操作(run_project)")
    requested_workflow_mode = args.workflow_mode
    normalize_project_workflow_args(
        args,
        invalid_profile_logger=lambda exc: print(
            f"⚠️  无效 writing profile: {exc}，回退为 {DEFAULT_WRITING_PROFILE}"
        ),
    )
    ablation_config = configure_benchmark_ablation_env(
        disable_hostile_critic=bool(args.disable_hostile_critic),
        disable_owner_aware_repair=bool(args.disable_owner_aware_repair),
    )
    if ablation_config["disable_hostile_critic"]:
        print("🧪 Benchmark ablation active: hostile critic disabled.")
    if ablation_config["disable_owner_aware_repair"]:
        print("🧪 Benchmark ablation active: owner-aware repair routing disabled.")
    runtime = initialize_runtime(
        source_file=__file__,
        project_root=PROJECT_ROOT,
        output_root=args.output_root,
        ensure_dirs=True,
        apply_cache=True,
    )
    args.output_root = str(runtime.research_root)
    args.project_dir = str(
        _assert_managed_project_layout(
            resolve_project_path(args.project_dir, output_root=args.output_root)
        )
    )

    strict_fallbacks = should_enforce_strict_fallbacks(
        args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        high_quality_mode=bool(args.high_quality_mode),
        target_venue=args.target_venue,
    )
    if args.override_strict_fallbacks and strict_fallbacks:
        print(
            "⚠️  override-strict-fallbacks: 严格兜底拦截已禁用，本次运行将继续记录但不阻断 fallback。"
        )
        strict_fallbacks = False
    elif strict_fallbacks:
        print(
            "🛡️  启用严格兜底策略：出现 fallback 将直接阻断流程（可通过 --override-strict-fallbacks 临时放宽）。"
        )

    integrity_forensics_enabled = (
        bool(args.integrity_forensics)
        if args.integrity_forensics is not None
        else bool(args.submission_mode or args.high_quality_mode)
    )
    if integrity_forensics_enabled:
        print("🔎 启用 integrity forensics：最终稿将生成可追溯一致性检查报告。")

    existing_required_repository = _activate_existing_required_research_vcs(args)
    early_admission_scope: dict[str, Any] | None = None
    early_admission_recovery: dict[str, Any] | None = None
    if existing_required_repository and args.resume:
        early_admission_scope = _required_admission_resume_inputs(args)
        early_admission_recovery = _reconcile_required_experiment_admission_runtime(
            args,
            selected_indices=early_admission_scope["selected_indices"],
            ideas_hash=early_admission_scope["ideas_hash"],
            configuration_hash=early_admission_scope["configuration_hash"],
            budget_hash=early_admission_scope["budget_hash"],
        )
        if (
            not early_admission_recovery.get("committed_results")
            and not early_admission_recovery.get("recovered_results")
            and not early_admission_recovery.get("pending_batch")
        ):
            early_admission_scope = None
            early_admission_recovery = None
    fully_recovered_resume = bool(
        early_admission_recovery
        and early_admission_recovery.get("active")
        and not early_admission_recovery.get("execute_indices")
        and early_admission_recovery.get("committed_results")
    )
    unfinished_required_resume = bool(
        early_admission_scope is not None
        and early_admission_recovery is not None
        and early_admission_recovery.get("active")
        and not fully_recovered_resume
    )

    # 创建项目结构
    dirs = create_project_structure(args.project_dir, output_root=args.output_root)
    args.project_dir = str(dirs["root"])
    _prepare_project_input(args, dirs)

    completed_resume = (
        _completed_resume_results(args.project_dir)
        if args.resume
        and not args.skip_experiment
        and not args.idea_indices
        and not args.seed_from_ara
        and not args.seed_node_id
        and not _required_experiment_admission_enabled(args.project_dir)
        else None
    )
    if completed_resume is not None:
        args._research_git_active = _initialize_local_research_git(args)
        successful_papers = _successful_paper_results(
            args.project_dir, completed_resume
        )
        if not successful_papers:
            print("❌ 已有进度未包含可用 PDF，项目尚未完成。")
            sys.exit(1)
        gate_failure = _publication_gate_failure(args, completed_resume)
        if gate_failure:
            print(f"❌ {gate_failure}")
            sys.exit(1)
        print(
            "♻️ 复用已有成功结果与可检查的论文产物，不再调用模型；"
            "审查与投稿状态以保存的门禁字段为准。"
        )
        _export_project_research_dag(args, results=completed_resume)
        _print_project_completion(args.project_dir, completed_resume)
        return

    reuse_admitted_autopilot_config = False
    if unfinished_required_resume:
        admitted_config = _required_admission_resume_configuration_path(
            args,
            expected_configuration_hash=early_admission_scope["configuration_hash"],
        )
        args.bfts_config = str(admitted_config)
        reuse_admitted_autopilot_config = bool(args.autopilot)
    autopilot_config = (
        None
        if fully_recovered_resume
        else _prepare_autopilot_bfts_config(
            args,
            reuse_existing=reuse_admitted_autopilot_config,
        )
    )
    if autopilot_config:
        print(f"⏱️ Autopilot 有限预算配置: {autopilot_config}")

    if fully_recovered_resume:
        args._data_contract = None
        args._project_budget = None
    else:
        from ai_scientist.utils.data_readiness import prepare_data_contract

        args._data_contract = prepare_data_contract(
            args.project_dir,
            data_dir=args.data_dir,
            allow_synthetic=bool(args.allow_synthetic_data),
            required=bool(args.autopilot and not args.skip_experiment),
        )
        args._project_budget = _configure_autopilot_project_budget(args)
        if args.autopilot:
            print(
                "🧾 Autopilot 数据/预算门禁: "
                f"data={args._data_contract['mode']} "
                "budget=project-wide"
            )
    if unfinished_required_resume:
        _validate_required_admission_resume_execution_inputs(
            args,
            early_admission_scope,
        )

    # 按实际模型检查 provider 凭证；完整恢复路径不会再次要求模型凭证。
    requested_models = _collect_requested_models(args)
    if not fully_recovered_resume:
        require_model_credentials(requested_models)
        autopilot_checks = _run_autopilot_preflight(args)
        autopilot_receipt = _write_autopilot_receipt(
            args,
            checks=autopilot_checks,
        )
        if autopilot_receipt:
            print(f"🧭 Autopilot 运行契约: {autopilot_receipt}")
    if not getattr(args, "_research_git_active", False):
        args._research_git_active = _initialize_local_research_git(args)
    if not fully_recovered_resume:
        record_model_provider_decisions(
            args.project_dir,
            requested_models,
            producer="run_project.model_provider",
        )

    # ARA fork-continue: stage a seed manifest so the first BFTS draft skips
    # the LLM. Uses env vars so it works transparently across subprocess
    # boundaries (parallel workers pick it up too).
    seed_stage = stage_seed_from_cli(
        seed_from_ara=getattr(args, "seed_from_ara", None),
        seed_node_id=getattr(args, "seed_node_id", None),
        project_dir=Path(args.project_dir),
    )
    banner = summarise_seed_stage(seed_stage)
    if banner:
        print(banner)
    if seed_stage.error:
        sys.exit(1)

    args.writeup_type = resolve_paper_type_for_venue(
        args.writeup_type,
        args.target_venue,
        auto_adjust=args.auto_adjust_paper_type,
        warning_template="⚠️  警告: writeup_type '{paper_type}' 与目标 venue '{target_venue}' 可能不够匹配",
        adjusted_template="✅ 自动调整 writeup_type -> {adjusted}",
    )

    # 确定想法文件
    idea_json = None

    if not args.skip_ideation:
        if args.topic:
            idea_json = generate_ideas(
                args.project_dir,
                args.topic,
                args.model_ideation,
                args.num_ideas,
                args.num_reflections,
            )
        elif args.ideas:
            idea_json = _materialize_project_ideas_input(
                args.project_dir,
                args.ideas,
            )
        else:
            print("❌ 错误: 需要指定 --topic 或 --ideas")
            sys.exit(1)
    else:
        idea_json = osp.join(args.project_dir, "01_ideas", "generated_ideas.json")
        if not osp.exists(idea_json):
            idea_json = osp.join(args.project_dir, "01_ideas", "ideas.json")
        if not osp.exists(idea_json):
            print(f"❌ 错误: 未找到想法文件")
            sys.exit(1)

    # 加载想法
    with open(idea_json, "r") as f:
        ideas = json.load(f)
    run_ideas_hash = canonical_content_hash(ideas)
    project_idea_cards = _write_project_pipeline_seed_artifacts(
        project_dir=args.project_dir,
        ideas=ideas,
        target_venue=args.target_venue,
        workflow_mode=args.workflow_mode,
        submission_mode=bool(args.submission_mode),
        breakthrough_mode=bool(args.breakthrough_mode),
        high_quality_mode=bool(args.high_quality_mode),
    )
    if project_idea_cards:
        print(
            "🧭 已生成结构化 idea cards / research_program，用于后续 planning、实验追踪和看板消费"
        )
        _record_local_research_planning_objects(
            args,
            idea_cards=project_idea_cards,
        )
    _record_local_research_checkpoint(
        args,
        stage="ideation",
        subject=f"record {len(ideas)} candidate ideas",
        summary=(
            "Record the ranked research candidates, research program, and "
            "claim/evidence planning state before experiment execution."
        ),
    )

    requested_indices = (
        [int(x.strip()) for x in args.idea_indices.split(",")]
        if args.idea_indices
        else None
    )
    default_indices = list(range(len(ideas))) if args.parallel else [0]
    required_admission = bool(
        getattr(args, "_research_git_active", False)
        and _required_experiment_admission_enabled(args.project_dir)
    )
    cached_selection = None
    idea_indices: list[int] = []
    if early_admission_recovery is not None:
        idea_indices = [
            int(index)
            for index in early_admission_recovery.get("batch_indices")
            or early_admission_scope["selected_indices"]
        ]
        rankings = []
    else:
        cached_selection = (
            _load_project_progress(args.project_dir).get("selected_indices")
            if args.resume and requested_indices is None and not required_admission
            else None
        )
        if isinstance(cached_selection, list) and cached_selection:
            for value in cached_selection:
                try:
                    index = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(ideas):
                    idea_indices.append(index)
        if idea_indices:
            rankings = []
        else:
            idea_indices, rankings = select_ranked_idea_candidates(
                ideas,
                ranking_enabled=args.rank_ideas,
                ranking_model=args.idea_rank_model or args.model_writeup,
                target_venue=args.target_venue,
                prioritize_breakthrough=args.breakthrough_mode,
                research_root=Path(args.output_root).expanduser(),
                ranking_output_path=osp.join(
                    args.project_dir, "04_logs", "idea_rankings.json"
                ),
                requested_indices=requested_indices,
                default_indices=default_indices,
                fallback_to_ranked=args.fallback_ranked_ideas,
                use_ranked_all=args.parallel,
                limit=args.top_k_ideas if args.rank_ideas else None,
            )
    if idea_indices and not rankings:
        print(f"♻️ 复用上次已记录的 idea 选择: {idea_indices}")
    if rankings:
        ranking_event = record_ranking_fallbacks(
            args.project_dir,
            rankings,
            producer="run_project.idea_ranking",
            strict=strict_fallbacks,
        )
        if strict_fallbacks and ranking_event:
            raise RuntimeError(
                format_strict_fallback_error(
                    ranking_event,
                    workflow_mode=args.workflow_mode,
                    stage_hint="idea_ranking",
                )
            )
        print("\n🏅 Idea ranking complete")
        for item in rankings[: min(5, len(rankings))]:
            print(
                f"  - idea #{item['idea_idx']} ranking_score={item.get('ranking_score')} total={item.get('total_score')} "
                f"name={item.get('idea_name')}"
            )

    selected_idea_indices = list(idea_indices)
    from ai_scientist.resources import resolve_bfts_config_path

    if early_admission_recovery is not None:
        if (
            early_admission_scope.get("ideas_hash") is not None
            and run_ideas_hash != early_admission_scope["ideas_hash"]
        ):
            raise RuntimeError(
                "persisted ideas changed after required admission reconciliation"
            )
        admission_configuration_hash = early_admission_scope["configuration_hash"]
        admission_budget_hash = early_admission_scope["budget_hash"]
        admission_recovery = early_admission_recovery
    else:
        admission_config_path = resolve_bfts_config_path(args.bfts_config)
        admission_configuration_hash = canonical_content_hash(
            admission_config_path.read_text(encoding="utf-8")
        )
        admission_budget = getattr(args, "_project_budget", None)
        admission_budget_hash = (
            canonical_content_hash(admission_budget)
            if isinstance(admission_budget, Mapping)
            else None
        )
        admission_recovery = _reconcile_required_experiment_admission_runtime(
            args,
            selected_indices=selected_idea_indices,
            ideas_hash=run_ideas_hash,
            configuration_hash=admission_configuration_hash,
            budget_hash=admission_budget_hash,
        )
    if admission_recovery["active"]:
        retained_progress = {}
        authoritative_results = _upsert_project_results(
            admission_recovery["committed_results"],
            admission_recovery["recovered_results"],
        )
        retained_progress_results = list(authoritative_results)
        if admission_recovery["batch_indices"]:
            selected_idea_indices = list(admission_recovery["batch_indices"])
    else:
        retained_progress = (
            _load_project_progress(args.project_dir) if args.resume else {}
        )
        authoritative_results = None
        retained_progress_results = (
            _latest_project_results(args.project_dir) if args.resume else []
        )
    progress_selected_indices: list[int] = []
    for value in [
        *(retained_progress.get("selected_indices") or []),
        *selected_idea_indices,
    ]:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in progress_selected_indices:
            progress_selected_indices.append(index)
    try:
        retained_total = int(retained_progress.get("total") or 0)
    except (TypeError, ValueError):
        retained_total = 0
    progress_total = max(
        retained_total,
        len(progress_selected_indices),
        len(retained_progress_results),
    )
    if (
        admission_recovery["active"]
        and not admission_recovery["execute_indices"]
        and admission_recovery["committed_results"]
    ):
        idea_indices = []
        prior_results = list(admission_recovery["committed_results"])
        resume_checkpoints = {}
    elif admission_recovery["pending_batch"]:
        idea_indices = list(admission_recovery["execute_indices"])
        prior_results = list(admission_recovery["committed_results"])
        resume_checkpoints = {
            int(result["idea_idx"]): str(result["checkpoint_path"])
            for result in admission_recovery["recovered_results"]
            if result.get("checkpoint_path")
            and Path(str(result["checkpoint_path"])).expanduser().is_file()
        }
    else:
        idea_indices, prior_results, resume_checkpoints = _resolve_resume_work(
            args.project_dir,
            selected_idea_indices,
            enabled=bool(args.resume),
            authoritative_results=authoritative_results,
        )
    if admission_recovery["active"]:
        _save_project_progress(
            args.project_dir,
            ideas_hash=run_ideas_hash,
            results=retained_progress_results,
            total=progress_total,
            selected_indices=progress_selected_indices,
        )
    if args.resume:
        print(
            "♻️ 恢复计划: "
            f"selected={len(selected_idea_indices)} completed={len(prior_results)} "
            f"pending={len(idea_indices)} checkpoints={len(resume_checkpoints)}"
        )
    if args.skip_experiment:
        print(
            "\n⏭️ 已按 --skip-experiment 跳过每个想法的 "
            "experiment → plot → writeup → review 流程。"
        )
        print(f"📁 项目目录: {args.project_dir}")
        print("📄 本次运行未生成论文。")
        return

    _preflight_local_research_experiment_admission(
        args,
        idea_indices=idea_indices,
        ideas_hash=run_ideas_hash,
        configuration_hash=admission_configuration_hash,
        budget_hash=admission_budget_hash,
    )

    # 运行实验
    if not args.skip_experiment:
        workflow_spec, _, template_profile, template_capability = (
            _resolve_workflow_strategy(
                workflow_mode=args.workflow_mode,
                submission_mode=bool(args.submission_mode),
                breakthrough_mode=bool(args.breakthrough_mode),
                high_quality_mode=bool(args.high_quality_mode),
                target_venue=args.target_venue,
            )
        )
        kwargs = {
            "output_root": args.output_root,
            "bfts_config": args.bfts_config,
            "model_writeup": args.model_writeup,
            "model_citation": args.model_citation,
            "model_review": args.model_review,
            "model_agg_plots": args.model_agg_plots,
            "model_writeup_small": args.model_writeup_small,
            "num_cite_rounds": args.num_cite_rounds,
            "writeup_retries": args.writeup_retries,
            "writeup_type": args.writeup_type,
            "improvement_rounds": args.improvement_rounds,
            "review_reflections": args.review_reflections,
            "review_ensemble": args.review_ensemble,
            "review_fewshot": args.review_fewshot,
            "review_temperature": args.review_temperature,
            "review_strategy": args.review_strategy,
            "high_quality_mode": args.high_quality_mode,
            "quality_preset": args.quality_preset,
            "quality_model": args.quality_model,
            "target_venue": args.target_venue,
            "quality_threshold": args.quality_threshold,
            "rigor_threshold": args.rigor_threshold,
            "quality_rewrite_rounds": args.quality_rewrite_rounds,
            "autonomous_quality_followup_rounds": args.autonomous_quality_followup_rounds,
            "require_quality_gate": args.require_quality_gate,
            "min_submission_priority": args.min_submission_priority,
            "max_submission_blockers": args.max_submission_blockers,
            "writing_profile": args.writing_profile,
            "writing_audit_rounds": args.writing_audit_rounds,
            "strict_writing_guardrails": args.strict_writing_guardrails,
            "guardrail_repair_rounds": args.guardrail_repair_rounds,
            "workflow_mode": workflow_spec.name,
            "requested_workflow_mode": requested_workflow_mode,
            "template_profile": template_profile,
            "template_capability": template_capability,
            "strict_fallbacks": strict_fallbacks,
            "integrity_forensics_enabled": integrity_forensics_enabled,
            "admission_preflights": getattr(
                args, "_experiment_admission_preflights", {}
            ),
        }
        recovered_execution_results = list(
            admission_recovery.get("recovered_results") or []
        )
        results = _upsert_project_results(
            list(prior_results), recovered_execution_results
        )
        executed_results: list[dict[str, Any]] = list(recovered_execution_results)

        if args.parallel and len(idea_indices) > 1:
            # 并行处理
            new_results = run_parallel_experiments(
                args.project_dir,
                idea_json,
                args.num_workers,
                idea_indices,
                prior_results=retained_progress_results,
                resume_checkpoints=resume_checkpoints,
                progress_total=progress_total,
                selected_indices=progress_selected_indices,
                ideas_hash=run_ideas_hash,
                **kwargs,
            )
            results = _upsert_project_results(results, new_results)
            executed_results.extend(new_results)

            # 打印结果摘要
            print("\n" + "=" * 80)
            print("📊 执行结果摘要")
            print("=" * 80)
            for r in results:
                status_icon = "✅" if r["status"] == "success" else "❌"
                print(f"{status_icon} 想法 #{r['idea_idx']}: {r['status']}")
                if r["status"] == "success" and "pdf_path" in r:
                    print(f"   📄 {r['pdf_path']}")

        else:
            # 串行处理
            for idx in idea_indices:
                if idx >= len(ideas):
                    continue

                # 构建参数元组
                process_args = (
                    args.project_dir,
                    args.output_root,
                    idx,
                    ideas[idx],
                    kwargs["bfts_config"],
                    kwargs["model_writeup"],
                    kwargs["model_citation"],
                    kwargs["model_review"],
                    kwargs["model_agg_plots"],
                    kwargs["model_writeup_small"],
                    kwargs["num_cite_rounds"],
                    kwargs["writeup_retries"],
                    kwargs["writeup_type"],
                    kwargs["improvement_rounds"],
                    kwargs["review_reflections"],
                    kwargs["review_ensemble"],
                    kwargs["review_fewshot"],
                    kwargs["review_temperature"],
                    kwargs["review_strategy"],
                    kwargs["high_quality_mode"],
                    kwargs["quality_preset"],
                    kwargs["quality_model"],
                    kwargs["target_venue"],
                    kwargs["quality_threshold"],
                    kwargs["rigor_threshold"],
                    kwargs["quality_rewrite_rounds"],
                    kwargs["autonomous_quality_followup_rounds"],
                    kwargs["require_quality_gate"],
                    kwargs["min_submission_priority"],
                    kwargs["max_submission_blockers"],
                    kwargs["writing_profile"],
                    kwargs["writing_audit_rounds"],
                    kwargs["strict_writing_guardrails"],
                    kwargs["guardrail_repair_rounds"],
                    kwargs["workflow_mode"],
                    kwargs["template_profile"],
                    kwargs["template_capability"],
                    kwargs["strict_fallbacks"],
                    kwargs["integrity_forensics_enabled"],
                    kwargs.get("requested_workflow_mode", kwargs["workflow_mode"]),
                    resume_checkpoints.get(idx),
                    (kwargs.get("admission_preflights") or {}).get(idx),
                )

                result = process_single_idea(process_args)
                results = _upsert_project_results(results, [result])
                retained_progress_results = _upsert_project_results(
                    retained_progress_results,
                    [result],
                )
                executed_results.append(result)
                _save_project_progress(
                    args.project_dir,
                    ideas_hash=run_ideas_hash,
                    results=retained_progress_results,
                    total=progress_total,
                    selected_indices=progress_selected_indices,
                )

                status_icon = "✅" if result["status"] == "success" else "❌"
                print(f"\n{status_icon} 想法 #{idx}: {result['status']}")

        results.sort(key=lambda item: int(item.get("idea_idx", 0)))
        experiment_ara_paths = [
            str(result.get("ara_manifest"))
            for result in results
            if result.get("ara_manifest")
        ]
        if executed_results:
            _record_local_research_attempt_objects(args, results=executed_results)
        if admission_recovery["active"] and (
            early_admission_recovery is None or executed_results
        ):
            finalized_recovery = _reconcile_required_experiment_admission_runtime(
                args,
                selected_indices=progress_selected_indices,
                ideas_hash=run_ideas_hash,
                configuration_hash=admission_configuration_hash,
                budget_hash=admission_budget_hash,
            )
            if finalized_recovery["recovered_results"]:
                raise RuntimeError(
                    "required experiment admission remained uncommitted after recording"
                )
            results = _upsert_project_results(
                results, finalized_recovery["committed_results"]
            )
            retained_progress_results = list(results)
            _save_project_progress(
                args.project_dir,
                ideas_hash=run_ideas_hash,
                results=retained_progress_results,
                total=progress_total,
                selected_indices=progress_selected_indices,
            )
        if executed_results:
            _record_local_research_checkpoint(
                args,
                stage="experiment",
                subject=f"record {len(executed_results)} new experiment outcomes",
                summary=(
                    f"Record {len(executed_results)} newly executed research outcomes while "
                    f"retaining {len(prior_results)} completed resume results, including "
                    "their ARA manifests and compact evidence state."
                ),
                status=_experiment_checkpoint_status(executed_results),
                ara_paths=experiment_ara_paths,
            )

        from ai_scientist.utils.insight_synthesis import synthesize_project_insights

        insight_report = synthesize_project_insights(
            args.project_dir,
            results,
            model=args.model_review,
            use_llm=bool(args.autopilot and not fully_recovered_resume),
        )
        print(
            "💡 洞见报告: "
            f"mode={insight_report.get('synthesis_mode')} "
            f"insights={len(insight_report.get('insights') or [])} "
            "status=machine_synthesized_unverified"
        )
        summary_file, shortlist_file, summary_payload = save_project_summary(
            args.project_dir,
            results,
            insight_report=insight_report,
        )
        _record_local_research_handoff_objects(
            args,
            results=executed_results,
            insight_report=insight_report,
        )
        save_contract_artifact(
            args.project_dir,
            "pipeline_manifest",
            load_contract_artifact(args.project_dir, "pipeline_manifest", default={}),
            producer="run_project.summary",
            warnings=[],
        )
        print(f"\n📋 项目总结: {summary_file}")
        print(f"📝 投稿 shortlist: {shortlist_file}")
        summary_quality = summary_payload.get("quality_summary", {})
        print(f"🛡️ 写作守护通过: {summary_quality.get('guardrail_passed', 0)}")
        print(f"🛡️ 写作守护拦截: {summary_quality.get('guardrail_blocked', 0)}")
        print(
            "🔎 Integrity forensics: "
            f"completed={summary_quality.get('integrity_forensics_completed', 0)} "
            f"hard={summary_quality.get('integrity_forensics_hard_flags', 0)} "
            f"soft={summary_quality.get('integrity_forensics_soft_flags', 0)}"
        )
        reason_counts = summary_quality.get("guardrail_blocking_reasons", {}) or {}
        if reason_counts:
            print("🧾 写作守护拦截原因:")
            for reason, count in sorted(
                reason_counts.items(), key=lambda item: (-item[1], item[0])
            ):
                print(f"   - {reason}: {count}")

        publication_failure = _finalize_local_research_handoff(
            args,
            results=results,
            ara_paths=experiment_ara_paths,
        )
        if publication_failure:
            print(f"❌ {publication_failure}")
            sys.exit(1)

    _print_project_completion(args.project_dir, results)


if __name__ == "__main__":
    main()
