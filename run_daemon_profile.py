#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ai_scientist.utils.auth_session import require_login

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError:
        _toml_loader = None

PROJECT_ROOT = Path(__file__).resolve().parent
PATH_KEYS = {"research_dir", "source_config", "topic", "ideas"}
PATH_LIST_KEYS = {"topic_files", "ideas_files"}

SCALAR_FLAGS = {
    "research_dir": "--research-dir",
    "daemon_name": "--daemon-name",
    "source_config": "--source-config",
    "topic": "--topic",
    "ideas": "--ideas",
    "source_rotation": "--source-rotation",
    "duration_hours": "--duration-hours",
    "max_cycles": "--max-cycles",
    "sleep_minutes": "--sleep-minutes",
    "failure_backoff_minutes": "--failure-backoff-minutes",
    "cycle_timeout_minutes": "--cycle-timeout-minutes",
    "auto_failure_guard_threshold": "--auto-failure-guard-threshold",
    "auto_failure_guard_cooldown_cycles": "--auto-failure-guard-cooldown-cycles",
    "day_start_hour": "--day-start-hour",
    "night_start_hour": "--night-start-hour",
    "rewrite_followup_top_k": "--rewrite-followup-top-k",
    "rewrite_followup_model": "--rewrite-followup-model",
    "rewrite_followup_quality_model": "--rewrite-followup-quality-model",
    "rewrite_followup_preset": "--rewrite-followup-preset",
    "rewrite_followup_max_rounds": "--rewrite-followup-max-rounds",
    "rewrite_followup_quality_threshold": "--rewrite-followup-quality-threshold",
    "rewrite_followup_rigor_threshold": "--rewrite-followup-rigor-threshold",
    "rewrite_followup_skip_blocker_threshold": "--rewrite-followup-skip-blocker-threshold",
    "rewrite_followup_blocker_reduction_threshold": "--rewrite-followup-blocker-reduction-threshold",
    "rewrite_followup_ready_max_rounds": "--rewrite-followup-ready-max-rounds",
    "rewrite_followup_publishable_priority_threshold": "--rewrite-followup-publishable-priority-threshold",
    "rewrite_followup_publishable_gain_threshold": "--rewrite-followup-publishable-gain-threshold",
    "guardrail_submission_target": "--guardrail-submission-target",
    "guardrail_min_followup_gain": "--guardrail-min-followup-gain",
    "guardrail_stagnation_cycles": "--guardrail-stagnation-cycles",
    "guardrail_empty_rewrite_cycles": "--guardrail-empty-rewrite-cycles",
    "guardrail_strong_cycles": "--guardrail-strong-cycles",
    "guardrail_default_num_ideas": "--guardrail-default-num-ideas",
    "guardrail_num_ideas_step": "--guardrail-num-ideas-step",
    "guardrail_min_num_ideas": "--guardrail-min-num-ideas",
    "guardrail_max_num_ideas": "--guardrail-max-num-ideas",
    "guardrail_generate_sleep_minutes": "--guardrail-generate-sleep-minutes",
    "guardrail_focus_sleep_minutes": "--guardrail-focus-sleep-minutes",
    "phase_warmup_cycles": "--phase-warmup-cycles",
    "phase_cold_start_submission_target": "--phase-cold-start-submission-target",
    "phase_hot_polish_submission_target": "--phase-hot-polish-submission-target",
    "summary_every_cycles": "--summary-every-cycles",
    "daily_summary_every_cycles": "--daily-summary-every-cycles",
    "dashboard_refresh_seconds": "--dashboard-refresh-seconds",
    "auto_source_plan_max_actions": "--auto-source-plan-max-actions",
    "auto_source_plan_min_health": "--auto-source-plan-min-health",
    "auto_source_plan_expires_after_cycles": "--auto-source-plan-expires-after-cycles",
    "source_quality_feedback_min_papers": "--source-quality-feedback-min-papers",
    "source_quality_feedback_max_boost": "--source-quality-feedback-max-boost",
    "source_quality_feedback_max_penalty": "--source-quality-feedback-max-penalty",
    "quality_strategy_submission_priority_threshold": "--quality-strategy-submission-priority-threshold",
    "quality_strategy_ready_rate_threshold": "--quality-strategy-ready-rate-threshold",
    "quality_strategy_exploration_priority_ceiling": "--quality-strategy-exploration-priority-ceiling",
    "quality_strategy_gate_pass_floor": "--quality-strategy-gate-pass-floor",
    "quality_strategy_max_num_ideas_for_strong_sources": "--quality-strategy-max-num-ideas-for-strong-sources",
    "quality_strategy_max_num_ideas_for_weak_sources": "--quality-strategy-max-num-ideas-for-weak-sources",
    "quality_strategy_dominant_venue_rate_threshold": "--quality-strategy-dominant-venue-rate-threshold",
    "quality_strategy_dominant_paper_type_rate_threshold": "--quality-strategy-dominant-paper-type-rate-threshold",
    "quality_governor_recent_cycles": "--quality-governor-recent-cycles",
    "quality_governor_stabilize_health_threshold": "--quality-governor-stabilize-health-threshold",
    "quality_governor_exploit_followup_gain": "--quality-governor-exploit-followup-gain",
    "quality_governor_max_rewrite_top_k": "--quality-governor-max-rewrite-top-k",
    "quality_governor_max_dossier_top_k": "--quality-governor-max-dossier-top-k",
    "quality_governor_max_source_plan_actions": "--quality-governor-max-source-plan-actions",
    "evidence_strategy_claim_support_floor": "--evidence-strategy-claim-support-floor",
    "evidence_strategy_numeric_coverage_floor": "--evidence-strategy-numeric-coverage-floor",
    "evidence_strategy_evidence_density_floor": "--evidence-strategy-evidence-density-floor",
    "evidence_strategy_min_quality_rewrite_rounds": "--evidence-strategy-min-quality-rewrite-rounds",
    "evidence_strategy_review_strategy": "--evidence-strategy-review-strategy",
    "evidence_strategy_claim_alignment_floor": "--evidence-strategy-claim-alignment-floor",
    "evidence_strategy_unsupported_claims_ceiling": "--evidence-strategy-unsupported-claims-ceiling",
    "dashboard_host": "--dashboard-host",
    "dashboard_port": "--dashboard-port",
    "python": "--python",
    "submission_board_top": "--submission-board-top",
    "submission_board_min_priority": "--submission-board-min-priority",
    "submission_board_max_blockers": "--submission-board-max-blockers",
    "submission_board_min_rewrite_gain": "--submission-board-min-rewrite-gain",
    "rewrite_board_top": "--rewrite-board-top",
    "rewrite_board_venue": "--rewrite-board-venue",
    "rewrite_board_min_priority": "--rewrite-board-min-priority",
    "rewrite_board_min_gain": "--rewrite-board-min-gain",
    "rewrite_board_max_blockers": "--rewrite-board-max-blockers",
    "shortlist_top": "--shortlist-top",
    "shortlist_venue": "--shortlist-venue",
    "shortlist_min_priority": "--shortlist-min-priority",
    "shortlist_max_blockers": "--shortlist-max-blockers",
    "shortlist_min_rewrite_gain": "--shortlist-min-rewrite-gain",
    "auto_submission_dossier_top_k": "--auto-submission-dossier-top-k",
    "auto_submission_dossier_min_priority": "--auto-submission-dossier-min-priority",
    "auto_submission_dossier_max_blockers": "--auto-submission-dossier-max-blockers",
    "auto_submission_dossier_min_rewrite_gain": "--auto-submission-dossier-min-rewrite-gain",
}

BOOL_FLAGS = {
    "run_forever": "--run-forever",
    "serve_dashboard": "--serve-dashboard",
    "auto_failure_guard": "--auto-failure-guard",
    "auto_apply_source_plan": "--auto-apply-source-plan",
    "auto_source_quality_feedback": "--auto-source-quality-feedback",
    "auto_quality_strategy_feedback": "--auto-quality-strategy-feedback",
    "auto_quality_governor": "--auto-quality-governor",
    "auto_evidence_strategy_feedback": "--auto-evidence-strategy-feedback",
    "auto_export_submission_dossier": "--auto-export-submission-dossier",
    "enable_rewrite_followup": "--enable-rewrite-followup",
    "rewrite_followup_include_ready": "--rewrite-followup-include-ready",
    "rewrite_board_include_ready": "--rewrite-board-include-ready",
    "shortlist_require_ready": "--shortlist-require-ready",
}

LIST_FLAGS = {
    "topic_files": "--topic-files",
    "ideas_files": "--ideas-files",
}

BOOLEAN_SWITCHES = {
    "default_submission_mode": (
        "--default-submission-mode",
        "--no-default-submission-mode",
    ),
    "submission_board_require_gate": (
        "--submission-board-require-gate",
        "--no-submission-board-require-gate",
    ),
    "rewrite_board_require_gate": (
        "--rewrite-board-require-gate",
        "--no-rewrite-board-require-gate",
    ),
    "shortlist_require_gate": (
        "--shortlist-require-gate",
        "--no-shortlist-require-gate",
    ),
    "adaptive_rewrite_followup": (
        "--adaptive-rewrite-followup",
        "--no-adaptive-rewrite-followup",
    ),
    "auto_submission_dossier_require_gate": (
        "--auto-submission-dossier-require-gate",
        "--no-auto-submission-dossier-require-gate",
    ),
    "auto_submission_dossier_require_ready": (
        "--auto-submission-dossier-require-ready",
        "--no-auto-submission-dossier-require-ready",
    ),
}

KNOWN_KEYS = (
    set(SCALAR_FLAGS)
    | set(BOOL_FLAGS)
    | set(LIST_FLAGS)
    | set(BOOLEAN_SWITCHES)
    | {"generator_args"}
)


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"profile not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".toml", ".tml"}:
        if _toml_loader is None:
            raise SystemExit(
                "TOML profile requires Python 3.11+ or the 'tomli' package"
            )
        payload = _toml_loader.loads(path.read_text(encoding="utf-8"))
    else:
        raise SystemExit("profile must be .json or .toml")
    if not isinstance(payload, dict):
        raise SystemExit("profile must be an object/table")
    return payload


def _resolve_profile_path(raw_value: str, *, profile_dir: Path) -> str:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    profile_relative = (profile_dir / candidate).resolve()
    project_relative = (PROJECT_ROOT / candidate).resolve()
    if profile_relative.exists():
        return str(profile_relative)
    if project_relative.exists():
        return str(project_relative)
    return str(profile_relative)


def _normalize_profile_paths(
    payload: dict[str, Any], *, profile_path: Path
) -> dict[str, Any]:
    normalized = dict(payload)
    profile_dir = profile_path.resolve().parent
    for key in PATH_KEYS:
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _resolve_profile_path(value, profile_dir=profile_dir)
    for key in PATH_LIST_KEYS:
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = [
                (
                    _resolve_profile_path(str(item), profile_dir=profile_dir)
                    if isinstance(item, str)
                    else item
                )
                for item in value
            ]
    return normalized


def _merge_profiles(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_profiles(merged[key], value)
        else:
            merged[key] = value
    return merged


def _discover_local_overlays(profile_path: Path) -> list[Path]:
    stem = profile_path.stem
    suffix = profile_path.suffix
    candidates = []
    if stem.endswith(".example"):
        candidates.append(profile_path.with_name(stem[:-8] + ".local" + suffix))
    candidates.append(profile_path.with_name(stem + ".local" + suffix))
    found = []
    seen = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            found.append(candidate)
            seen.add(candidate)
    return found


def _load_profile_with_overlays(
    profile_path: Path, explicit_overlays: list[str] | None = None
) -> tuple[dict[str, Any], list[str]]:
    applied: list[str] = []
    profile = _normalize_profile_paths(
        _load_mapping(profile_path), profile_path=profile_path
    )
    for overlay_path in _discover_local_overlays(profile_path):
        overlay = _normalize_profile_paths(
            _load_mapping(overlay_path), profile_path=overlay_path
        )
        profile = _merge_profiles(profile, overlay)
        applied.append(str(overlay_path.resolve()))
    for raw_overlay in explicit_overlays or []:
        overlay_path = Path(
            _resolve_profile_path(raw_overlay, profile_dir=profile_path.parent)
        )
        overlay = _normalize_profile_paths(
            _load_mapping(overlay_path), profile_path=overlay_path
        )
        profile = _merge_profiles(profile, overlay)
        applied.append(str(overlay_path.resolve()))
    return profile, applied


def _validate_profile(payload: dict[str, Any]) -> list[str]:
    errors = []
    if not any(
        payload.get(key)
        for key in ["source_config", "topic", "ideas", "topic_files", "ideas_files"]
    ):
        errors.append(
            "profile must define at least one of source_config, topic, ideas, topic_files, or ideas_files"
        )
    if "source_rotation" in payload and payload.get("source_rotation") not in {
        "fixed",
        "round_robin",
    }:
        errors.append("source_rotation must be fixed or round_robin")
    if "rewrite_followup_preset" in payload and payload.get(
        "rewrite_followup_preset"
    ) not in {"balanced", "high", "publishable"}:
        errors.append("rewrite_followup_preset must be balanced/high/publishable")
    generator_args = payload.get("generator_args") or []
    if generator_args and not isinstance(generator_args, list):
        errors.append("generator_args must be a list of strings")
    if (
        isinstance(generator_args, list)
        and generator_args
        and not all(isinstance(item, str) for item in generator_args)
    ):
        errors.append("generator_args must contain only strings")
    for list_key in LIST_FLAGS:
        value = payload.get(list_key)
        if value is not None and not isinstance(value, list):
            errors.append(f"{list_key} must be a list")
        if (
            isinstance(value, list)
            and value
            and not all(isinstance(item, str) for item in value)
        ):
            errors.append(f"{list_key} must contain only strings")
    for key in ["source_config", "topic", "ideas"]:
        value = payload.get(key)
        if isinstance(value, str) and value and not Path(value).exists():
            errors.append(f"{key} path does not exist: {value}")
    for key in PATH_LIST_KEYS:
        value = payload.get(key) or []
        if isinstance(value, list):
            missing = [
                item
                for item in value
                if isinstance(item, str) and not Path(item).exists()
            ]
            if missing:
                errors.append(f"{key} contains missing paths: {', '.join(missing)}")
    unknown_keys = sorted(set(payload) - KNOWN_KEYS)
    if unknown_keys:
        errors.append("unknown profile keys: " + ", ".join(unknown_keys))
    return errors


def _build_command(profile: dict[str, Any], *, dry_run: bool = False) -> list[str]:
    cmd = [sys.executable, str(PROJECT_ROOT / "continuous_research_daemon.py")]
    for key, flag in SCALAR_FLAGS.items():
        value = profile.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    for key, flag in BOOL_FLAGS.items():
        if profile.get(key):
            cmd.append(flag)
    for key, flag in LIST_FLAGS.items():
        values = profile.get(key) or []
        if values:
            cmd.append(flag)
            cmd.extend(str(item) for item in values)
    for key, (true_flag, false_flag) in BOOLEAN_SWITCHES.items():
        if key not in profile:
            continue
        cmd.append(true_flag if profile.get(key) else false_flag)
    if dry_run:
        cmd.append("--dry-run")
    generator_args = profile.get("generator_args") or []
    if generator_args:
        cmd.append("--")
        cmd.extend(str(item) for item in generator_args)
    return cmd


def main() -> int:
    require_login("守护进程配置运行(run_daemon_profile)")

    parser = argparse.ArgumentParser(
        description="Launch the continuous daemon from a reusable profile config"
    )
    parser.add_argument("profile", help="Path to JSON or TOML daemon profile")
    parser.add_argument(
        "--overlay",
        action="append",
        default=[],
        help="Optional overlay JSON/TOML applied after the base profile",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Pass --dry-run to the daemon"
    )
    parser.add_argument(
        "--print-command", action="store_true", help="Only print the built command"
    )
    args = parser.parse_args()

    profile_path = Path(args.profile).expanduser().resolve()
    profile, applied_overlays = _load_profile_with_overlays(
        profile_path, explicit_overlays=args.overlay
    )
    errors = _validate_profile(profile)
    if errors:
        raise SystemExit("invalid daemon profile:\n- " + "\n- ".join(errors))

    cmd = _build_command(profile, dry_run=args.dry_run)
    if args.print_command:
        print(
            json.dumps(
                {
                    "command": cmd,
                    "profile_path": str(profile_path),
                    "applied_overlays": applied_overlays,
                    "daemon_name": profile.get("daemon_name"),
                    "research_dir": profile.get("research_dir"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
