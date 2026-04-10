#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError:
        _toml_loader = None
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ai_scientist.config.paths import PRIMARY_OUTPUT_ENV_VAR, resolve_output_path
from ai_scientist.utils.high_quality_pipeline import (
    QUALITY_PRESETS,
    run_high_quality_pass,
)
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.daemon_feedback import (
    build_active_source_feedback_snapshot,
    get_active_source_feedback,
)
from ai_scientist.utils.source_planning import (
    build_source_planning_profile,
    normalize_batch_profile,
    normalize_source_archetype,
    normalize_workflow_mode_list,
    normalize_workflow_mode_name,
)
from ai_scientist.utils.workflow_execution_policy import (
    build_workflow_execution_policy,
)
from research_manager import ResearchManager

SHUTDOWN_REQUESTED = False
RESERVED_GENERATOR_FLAGS = {
    "--research-dir",
    "--batch-name",
    "--topic",
    "--ideas",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _current_daypart(parsed: argparse.Namespace) -> str:
    hour = datetime.now().hour
    day_start = int(parsed.day_start_hour)
    night_start = int(parsed.night_start_hour)
    if day_start <= hour < night_start:
        return "day"
    return "night"


def _maybe_bool(value: Any) -> Any:
    if value is None:
        return None
    return bool(value)


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _coerce_one_or_many_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if not isinstance(value, list):
        return []
    return _coerce_str_list(value)


def _start_dashboard_server(
    daemon_dir: Path, host: str, port: int
) -> tuple[ThreadingHTTPServer | None, threading.Thread | None, str]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(daemon_dir))
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        local_dashboard = daemon_dir / "latest_live_dashboard.html"
        print(
            "⚠️ Dashboard server unavailable; falling back to static dashboard file: "
            f"{local_dashboard} ({exc})",
            file=sys.stderr,
        )
        return None, None, str(local_dashboard)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    actual_host, actual_port = server.server_address[:2]
    url = f"http://{actual_host}:{actual_port}/latest_live_dashboard.html"
    return server, thread, url


def _stop_dashboard_server(
    server: ThreadingHTTPServer | None, thread: threading.Thread | None
) -> None:
    if server is None:
        return
    try:
        server.shutdown()
        server.server_close()
    finally:
        if thread is not None:
            thread.join(timeout=2)


def _safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{_now_iso()}] {message}\n")


def _default_control_payload() -> dict[str, Any]:
    return {
        "paused": False,
        "stop_after_cycle": False,
        "force_phase": None,
        "force_mode": None,
        "source_priority_overrides": {},
        "disabled_sources": [],
        "source_commands": {},
        "sleep_override_minutes": None,
        "dashboard_refresh_seconds": None,
        "expires_after_cycles": None,
    }


def _validate_control_payload(payload: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["control payload must be a JSON object"]

    allowed = {
        "paused",
        "stop_after_cycle",
        "force_phase",
        "force_mode",
        "source_priority_overrides",
        "disabled_sources",
        "source_commands",
        "sleep_override_minutes",
        "dashboard_refresh_seconds",
        "expires_after_cycles",
        "_expires_after_cycle",
    }
    for key in payload.keys():
        if key not in allowed:
            errors.append(f"unknown top-level control key: {key}")

    if payload.get("force_phase") not in {
        None,
        "cold_start",
        "steady_state",
        "hot_polish",
    }:
        errors.append(
            "force_phase must be one of cold_start/steady_state/hot_polish/null"
        )
    if payload.get("force_mode") not in {
        None,
        "balanced",
        "generate_more",
        "focus_rewrite",
    }:
        errors.append(
            "force_mode must be one of balanced/generate_more/focus_rewrite/null"
        )
    if not isinstance(payload.get("disabled_sources", []), list):
        errors.append("disabled_sources must be a list")
    if not isinstance(payload.get("source_priority_overrides", {}), dict):
        errors.append("source_priority_overrides must be an object")
    if not isinstance(payload.get("source_commands", {}), dict):
        errors.append("source_commands must be an object")

    for key, command in (payload.get("source_commands") or {}).items():
        if not isinstance(command, dict):
            errors.append(f"source_commands[{key}] must be an object")
            continue
        allowed_command_keys = {
            "force_next_cycle",
            "priority_boost_next",
            "disable_once",
            "cooldown_cycles_once",
            "expires_after_cycles",
            "_expires_after_cycle",
        }
        for command_key in command.keys():
            if command_key not in allowed_command_keys:
                errors.append(f"unknown source command key for {key}: {command_key}")

    return errors[:20]


def _ensure_control_file(daemon_dir: Path) -> Path:
    control_path = daemon_dir / "daemon_control.json"
    if not control_path.exists():
        _safe_write_json(control_path, _default_control_payload())
    return control_path


def _load_control_payload(daemon_dir: Path, current_cycle: int = 0) -> dict[str, Any]:
    control_path = _ensure_control_file(daemon_dir)
    errors = []
    try:
        payload = json.loads(control_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = _default_control_payload()
        errors.append("control file is not valid JSON; falling back to defaults")
    default = _default_control_payload()
    default.update(payload if isinstance(payload, dict) else {})
    errors.extend(_validate_control_payload(default))
    default["validation_errors"] = errors
    default, events, changed = _apply_control_expiry(daemon_dir, default, current_cycle)
    if changed:
        _save_control_payload(daemon_dir, default)
    for event in events:
        _append_control_event(daemon_dir, event)
    return default


def _save_control_payload(daemon_dir: Path, payload: dict[str, Any]) -> None:
    control_path = _ensure_control_file(daemon_dir)
    _safe_write_json(control_path, payload)


def _expiry_cycle_from_relative(
    current_cycle: int, expires_after_cycles: Any, existing_expiry: Any = None
) -> int | None:
    if existing_expiry is not None:
        try:
            return int(existing_expiry)
        except (TypeError, ValueError):
            return None
    if expires_after_cycles is None:
        return None
    try:
        return current_cycle + int(expires_after_cycles)
    except (TypeError, ValueError):
        return None


def _apply_control_expiry(
    daemon_dir: Path, payload: dict[str, Any], current_cycle: int
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    changed = False
    events: list[dict[str, Any]] = []

    global_expiry = _expiry_cycle_from_relative(
        current_cycle,
        payload.get("expires_after_cycles"),
        payload.get("_expires_after_cycle"),
    )
    if (
        global_expiry is not None
        and payload.get("_expires_after_cycle") != global_expiry
    ):
        payload["_expires_after_cycle"] = global_expiry
        changed = True
    if global_expiry is not None and current_cycle >= global_expiry:
        for key in [
            "paused",
            "stop_after_cycle",
            "force_phase",
            "force_mode",
            "sleep_override_minutes",
        ]:
            if payload.get(key) not in (None, False):
                events.append(
                    {
                        "type": "control_override_expired",
                        "field": key,
                        "expires_at_cycle": global_expiry,
                    }
                )
            if isinstance(payload.get(key), bool):
                payload[key] = False
            else:
                payload[key] = None
        payload["expires_after_cycles"] = None
        payload["_expires_after_cycle"] = None
        changed = True

    commands = dict(payload.get("source_commands") or {})
    new_commands = {}
    for key, command in commands.items():
        if not isinstance(command, dict):
            new_commands[key] = command
            continue
        expires_cycle = _expiry_cycle_from_relative(
            current_cycle,
            command.get("expires_after_cycles"),
            command.get("_expires_after_cycle"),
        )
        if (
            expires_cycle is not None
            and command.get("_expires_after_cycle") != expires_cycle
        ):
            command["_expires_after_cycle"] = expires_cycle
            changed = True
        if expires_cycle is not None and current_cycle >= expires_cycle:
            events.append(
                {
                    "type": "source_command_expired",
                    "matched_key": key,
                    "expires_at_cycle": expires_cycle,
                    "command": command,
                }
            )
            changed = True
            continue
        new_commands[key] = command
    if new_commands != commands:
        payload["source_commands"] = new_commands
        changed = True

    return payload, events, changed


def _append_control_event(daemon_dir: Path, event: dict[str, Any]) -> None:
    _append_jsonl(
        daemon_dir / "daemon_control_history.jsonl", {"timestamp": _now_iso(), **event}
    )


def _load_recent_control_events(
    daemon_dir: Path, max_entries: int = 20
) -> list[dict[str, Any]]:
    path = daemon_dir / "daemon_control_history.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-max_entries:]


def _apply_control_overrides(
    status: dict[str, Any], daemon_dir: Path
) -> dict[str, Any]:
    control = _load_control_payload(
        daemon_dir, current_cycle=int(status.get("cycle", 0) or 0)
    )
    status["control"] = control
    if control.get("dashboard_refresh_seconds") is not None:
        status["dashboard_refresh_seconds"] = control.get("dashboard_refresh_seconds")
    return status


def _signal_handler(_signum, _frame) -> None:
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True


def _clean_generator_args(args: list[str]) -> list[str]:
    cleaned = list(args)
    if cleaned and cleaned[0] == "--":
        cleaned = cleaned[1:]
    duplicates = [flag for flag in RESERVED_GENERATOR_FLAGS if flag in cleaned]
    if duplicates:
        raise SystemExit(
            "Do not pass reserved generator flags through daemon passthrough args: "
            + ", ".join(sorted(duplicates))
        )
    return cleaned


def _source_command_for(
    control: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    commands = control.get("source_commands") or {}
    for key in [_source_key(source), source.get("name"), source.get("value")]:
        if key in commands and isinstance(commands[key], dict):
            command = dict(commands[key])
            command["_matched_key"] = key
            return command
    return {}


def _consume_source_command(
    daemon_dir: Path, status: dict[str, Any], matched_key: str
) -> None:
    control = _load_control_payload(
        daemon_dir, current_cycle=int(status.get("cycle", 0) or 0)
    )
    commands = dict(control.get("source_commands") or {})
    if matched_key in commands:
        consumed_command = commands.pop(matched_key, None)
        control["source_commands"] = commands
        _save_control_payload(daemon_dir, control)
        status["control"] = control
        _append_control_event(
            daemon_dir,
            {
                "type": "source_command_consumed",
                "matched_key": matched_key,
                "active_source": status.get("active_source"),
                "command": consumed_command,
            },
        )


def _load_source_queue(
    parsed: argparse.Namespace,
    control: dict[str, Any] | None = None,
    source_feedback: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    control = control or {}
    source_feedback = source_feedback or {}
    if parsed.source_config:
        for entry in _load_source_config(parsed.source_config):
            queue.append(_normalize_source_entry(entry))
    if parsed.topic:
        queue.append(
            _normalize_source_entry(
                {
                    "type": "topic",
                    "value": parsed.topic,
                    "name": Path(parsed.topic).stem,
                }
            )
        )
    if parsed.ideas:
        queue.append(
            _normalize_source_entry(
                {
                    "type": "ideas",
                    "value": parsed.ideas,
                    "name": Path(parsed.ideas).stem,
                }
            )
        )
    for item in parsed.topic_files:
        queue.append(
            _normalize_source_entry(
                {"type": "topic", "value": item, "name": Path(item).stem}
            )
        )
    for item in parsed.ideas_files:
        queue.append(
            _normalize_source_entry(
                {"type": "ideas", "value": item, "name": Path(item).stem}
            )
        )
    if not queue:
        raise SystemExit("At least one topic or ideas source is required")

    disabled = set(control.get("disabled_sources") or [])
    priority_overrides = control.get("source_priority_overrides") or {}
    patched = []
    for item in queue:
        item = dict(item)
        key = item.get("name") or item.get("value")
        command = _source_command_for(control, item)
        item["source_command"] = command
        if key in disabled or _source_key(item) in disabled:
            item["disabled_by_control"] = True
        if command.get("disable_once"):
            item["disabled_by_control"] = True
            item["disabled_reason"] = "disabled once by source command"
        override = priority_overrides.get(key)
        if override is None:
            override = priority_overrides.get(_source_key(item))
        if override is not None:
            try:
                item["priority"] = float(override)
                item["priority_override_applied"] = True
            except (TypeError, ValueError):
                pass
        if command.get("priority_boost_next") is not None:
            try:
                item["priority"] = float(item.get("priority", 0)) + float(
                    command.get("priority_boost_next")
                )
                item["priority_boost_applied"] = True
            except (TypeError, ValueError):
                pass
        if command.get("force_next_cycle"):
            item["priority"] = float(item.get("priority", 0)) + 10_000
            item["force_next_cycle_applied"] = True
        feedback = source_feedback.get(str(key)) or source_feedback.get(
            str(_source_key(item))
        )
        if feedback is None and item.get("name") is not None:
            feedback = source_feedback.get(str(item.get("name")))
        if feedback is not None and parsed.auto_source_quality_feedback:
            bonus = float(feedback.get("priority_bonus", 0.0) or 0.0)
            item["priority"] = float(item.get("priority", 0)) + bonus
            item["quality_feedback_priority_bonus"] = bonus
            item["quality_feedback_summary"] = feedback
        patched.append(item)
    patched.sort(key=lambda item: item.get("priority", 0), reverse=True)
    return patched


def _source_key(source: dict[str, Any]) -> str:
    return f"{source.get('type')}::{source.get('name')}::{source.get('value')}"


def _refresh_source_runtime_state(
    status: dict[str, Any], queue: list[dict[str, Any]]
) -> dict[str, Any]:
    runtime = status.setdefault("source_runtime", {})
    today = datetime.now().date().isoformat()
    for source in queue:
        key = _source_key(source)
        state = runtime.setdefault(
            key,
            {
                "name": source.get("name"),
                "today": today,
                "cycles_today": 0,
                "successes_today": 0,
                "total_cycles": 0,
                "total_successes": 0,
                "consecutive_failures": 0,
                "cooldown_until_cycle": 0,
                "last_selected_at": None,
                "last_finished_at": None,
            },
        )
        if state.get("today") != today:
            state["today"] = today
            state["cycles_today"] = 0
            state["successes_today"] = 0
    return runtime


def _source_is_eligible(
    source: dict[str, Any], state: dict[str, Any], cycle_number: int, daypart: str
) -> tuple[bool, str | None]:
    if source.get("disabled_by_control"):
        return False, source.get("disabled_reason") or "disabled by control"
    preference = (source.get("time_of_day_preference") or "any").lower()
    if preference in {"day", "night"} and preference != daypart:
        return False, f"preferred at {preference}, current daypart is {daypart}"
    if int(state.get("cooldown_until_cycle", 0) or 0) > cycle_number:
        return False, f"cooldown until cycle {state.get('cooldown_until_cycle')}"
    max_cycles_per_day = int(source.get("max_cycles_per_day", 0) or 0)
    if (
        max_cycles_per_day > 0
        and int(state.get("cycles_today", 0) or 0) >= max_cycles_per_day
    ):
        return False, "daily cycle quota reached"
    success_budget = int(source.get("success_budget", 0) or 0)
    if (
        success_budget > 0
        and int(state.get("successes_today", 0) or 0) >= success_budget
    ):
        return False, "daily success budget reached"
    return True, None


def _eligible_sources(
    parsed: argparse.Namespace, status: dict[str, Any], queue: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], str]]:
    runtime = _refresh_source_runtime_state(status, queue)
    cycle_number = int(status.get("cycle", 0) or 0) + 1
    daypart = _current_daypart(parsed)
    status["current_daypart"] = daypart
    eligible: list[tuple[dict[str, Any], str]] = []
    for source in queue:
        key = _source_key(source)
        state = runtime.get(key, {})
        ok, reason = _source_is_eligible(source, state, cycle_number, daypart)
        if ok:
            eligible.append((source, key))
    return eligible


def _select_source(
    parsed: argparse.Namespace, status: dict[str, Any]
) -> dict[str, Any]:
    queue = _load_source_queue(
        parsed, status.get("control"), status.get("source_quality_feedback")
    )
    eligible = _eligible_sources(parsed, status, queue)
    if not eligible:
        raise RuntimeError(
            "no eligible sources available under current cooldown/quota settings"
        )

    if parsed.source_rotation == "fixed":
        source, key = eligible[0]
        source_index = queue.index(source)
    else:
        start_index = int(status.get("source_index", 0)) % len(queue)
        source = None
        key = None
        source_index = start_index
        for offset in range(len(queue)):
            idx = (start_index + offset) % len(queue)
            candidate = queue[idx]
            candidate_key = _source_key(candidate)
            if any(candidate_key == eligible_key for _, eligible_key in eligible):
                source = candidate
                key = candidate_key
                source_index = idx
                break
        if source is None:
            source, key = eligible[0]
            source_index = queue.index(source)

    runtime = _refresh_source_runtime_state(status, queue)
    runtime[key]["last_selected_at"] = _now_iso()
    status["source_index"] = source_index
    status["active_source"] = dict(source)
    status["active_source_key"] = key
    status["active_source_command_key"] = (source.get("source_command") or {}).get(
        "_matched_key"
    )
    status["source_queue"] = queue
    return dict(source)


def _passthrough_arg_value(args: list[str], flag: str, default: Any = None) -> Any:
    cleaned = _clean_generator_args(args)
    for idx, item in enumerate(cleaned):
        if item == flag and idx + 1 < len(cleaned):
            return cleaned[idx + 1]
    return default


def _set_passthrough_arg(args: list[str], flag: str, value: Any) -> list[str]:
    updated = list(_clean_generator_args(args))
    for idx, item in enumerate(updated):
        if item == flag:
            if idx + 1 < len(updated):
                updated[idx + 1] = str(value)
            else:
                updated.append(str(value))
            return updated
    updated.extend([flag, str(value)])
    return updated


def _remove_passthrough_flag(args: list[str], flag: str) -> list[str]:
    cleaned = []
    original = list(_clean_generator_args(args))
    idx = 0
    while idx < len(original):
        item = original[idx]
        if item == flag:
            idx += 1
            if idx < len(original) and not original[idx].startswith("--"):
                idx += 1
            continue
        cleaned.append(item)
        idx += 1
    return cleaned


def _ensure_passthrough_flag(args: list[str], flag: str) -> list[str]:
    updated = list(_clean_generator_args(args))
    if flag not in updated:
        updated.append(flag)
    return updated


def _remove_passthrough_switch(args: list[str], flag: str) -> list[str]:
    updated = list(_clean_generator_args(args))
    return [item for item in updated if item != flag]


def _set_passthrough_multi_arg(
    args: list[str], flag: str, values: list[str]
) -> list[str]:
    cleaned = []
    original = list(_clean_generator_args(args))
    idx = 0
    while idx < len(original):
        item = original[idx]
        if item == flag:
            idx += 1
            while idx < len(original) and not original[idx].startswith("--"):
                idx += 1
            continue
        cleaned.append(item)
        idx += 1
    cleaned.append(flag)
    cleaned.extend(str(value) for value in values)
    return cleaned


def _validate_source_entry(entry: dict[str, Any], index: int) -> list[str]:
    errors = []
    if not isinstance(entry, dict):
        return [f"sources[{index}] must be an object"]
    if not any(entry.get(key) for key in ["value", "topic", "ideas"]):
        errors.append(f"sources[{index}] must define one of value/topic/ideas")
    source_type = entry.get("type")
    if source_type is not None and source_type not in {"topic", "ideas"}:
        errors.append(f"sources[{index}].type must be 'topic' or 'ideas'")
    venue_fields = ["target_venue", "day_target_venue", "night_target_venue"]
    valid_venues = {None, "neurips", "iclr", "cvpr", "journal", "nature"}
    for field in venue_fields:
        if entry.get(field) not in valid_venues:
            errors.append(f"sources[{index}].{field} is not a supported venue")
    valid_paper_types = {"icbinb", "normal", "journal", "extended"}
    for field in ["paper_types", "day_paper_types", "night_paper_types"]:
        value = entry.get(field)
        if value is None:
            continue
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list) or any(
            item not in valid_paper_types for item in values
        ):
            errors.append(f"sources[{index}].{field} contains unsupported paper types")
    if entry.get("time_of_day_preference") not in {None, "any", "day", "night"}:
        errors.append(f"sources[{index}].time_of_day_preference must be any/day/night")
    workflow_fields = ["workflow_mode", "day_workflow_mode", "night_workflow_mode"]
    for field in workflow_fields:
        try:
            normalize_workflow_mode_name(entry.get(field))
        except ValueError:
            errors.append(
                f"sources[{index}].{field} is not a supported workflow mode"
            )
    workflow_list_fields = [
        "workflow_modes",
        "day_workflow_modes",
        "night_workflow_modes",
    ]
    for field in workflow_list_fields:
        try:
            normalize_workflow_mode_list(entry.get(field))
        except ValueError:
            errors.append(
                f"sources[{index}].{field} contains unsupported workflow modes"
            )
    archetype_fields = ["source_archetype", "day_source_archetype", "night_source_archetype"]
    for field in archetype_fields:
        try:
            normalize_source_archetype(entry.get(field))
        except ValueError:
            errors.append(
                f"sources[{index}].{field} is not a supported source archetype"
            )
    batch_fields = ["batch_profile", "day_batch_profile", "night_batch_profile"]
    for field in batch_fields:
        try:
            normalize_batch_profile(entry.get(field))
        except ValueError:
            errors.append(
                f"sources[{index}].{field} is not a supported batch profile"
            )
    return errors


def _validate_source_config(payload: dict[str, Any]) -> list[str]:
    errors = []
    if not isinstance(payload, dict):
        return ["source config must be a JSON/TOML object"]
    sources = payload.get("sources")
    if not isinstance(sources, list):
        return ["source config must contain a top-level 'sources' list"]
    for idx, entry in enumerate(sources):
        errors.extend(_validate_source_entry(entry, idx))
    return errors[:50]


def _load_source_config(path: str) -> list[dict[str, Any]]:
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"source config not found: {path}")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    elif config_path.suffix.lower() in {".toml", ".tml"}:
        if _toml_loader is None:
            raise SystemExit(
                "TOML source config requires Python 3.11+ or the 'tomli' package"
            )
        payload = _toml_loader.loads(config_path.read_text(encoding="utf-8"))
    else:
        raise SystemExit("source config must be .json or .toml")
    errors = _validate_source_config(payload)
    if errors:
        raise SystemExit("invalid source config:\n- " + "\n- ".join(errors))
    return payload["sources"]


def _normalize_source_entry(entry: dict[str, Any]) -> dict[str, Any]:
    value = entry.get("value") or entry.get("topic") or entry.get("ideas")
    if not value:
        raise SystemExit(f"invalid source entry without value: {entry}")
    source_type = entry.get("type")
    if source_type is None:
        source_type = "ideas" if str(value).endswith(".json") else "topic"
    if source_type not in {"topic", "ideas"}:
        raise SystemExit(f"unsupported source type: {source_type}")
    paper_types = entry.get("paper_types") or []
    if isinstance(paper_types, str):
        paper_types = [paper_types]
    generator_args = entry.get("generator_args") or []
    if isinstance(generator_args, str):
        generator_args = shlex.split(generator_args)

    day_paper_types = entry.get("day_paper_types") or []
    if isinstance(day_paper_types, str):
        day_paper_types = [day_paper_types]
    night_paper_types = entry.get("night_paper_types") or []
    if isinstance(night_paper_types, str):
        night_paper_types = [night_paper_types]
    day_generator_args = entry.get("day_generator_args") or []
    if isinstance(day_generator_args, str):
        day_generator_args = shlex.split(day_generator_args)
    night_generator_args = entry.get("night_generator_args") or []
    if isinstance(night_generator_args, str):
        night_generator_args = shlex.split(night_generator_args)
    workflow_modes = normalize_workflow_mode_list(entry.get("workflow_modes"))
    day_workflow_modes = normalize_workflow_mode_list(entry.get("day_workflow_modes"))
    night_workflow_modes = normalize_workflow_mode_list(
        entry.get("night_workflow_modes")
    )

    return {
        "name": entry.get("name") or Path(str(value)).stem,
        "type": source_type,
        "value": str(value),
        "priority": float(entry.get("priority", 0)),
        "target_venue": entry.get("target_venue"),
        "paper_types": [str(item) for item in paper_types],
        "num_ideas": entry.get("num_ideas"),
        "submission_mode": _maybe_bool(entry.get("submission_mode", False)),
        "breakthrough_mode": _maybe_bool(entry.get("breakthrough_mode", False)),
        "cooldown_cycles": int(entry.get("cooldown_cycles", 0) or 0),
        "max_cycles_per_day": int(entry.get("max_cycles_per_day", 0) or 0),
        "success_budget": int(entry.get("success_budget", 0) or 0),
        "time_of_day_preference": entry.get("time_of_day_preference", "any"),
        "workflow_mode": normalize_workflow_mode_name(entry.get("workflow_mode")),
        "day_workflow_mode": normalize_workflow_mode_name(
            entry.get("day_workflow_mode")
        ),
        "night_workflow_mode": normalize_workflow_mode_name(
            entry.get("night_workflow_mode")
        ),
        "workflow_modes": workflow_modes,
        "day_workflow_modes": day_workflow_modes,
        "night_workflow_modes": night_workflow_modes,
        "source_archetype": normalize_source_archetype(entry.get("source_archetype")),
        "day_source_archetype": normalize_source_archetype(
            entry.get("day_source_archetype")
        ),
        "night_source_archetype": normalize_source_archetype(
            entry.get("night_source_archetype")
        ),
        "batch_profile": normalize_batch_profile(entry.get("batch_profile")),
        "day_batch_profile": normalize_batch_profile(entry.get("day_batch_profile")),
        "night_batch_profile": normalize_batch_profile(
            entry.get("night_batch_profile")
        ),
        "alignment_tags": _coerce_one_or_many_str(entry.get("alignment_tags")),
        "day_alignment_tags": _coerce_one_or_many_str(entry.get("day_alignment_tags")),
        "night_alignment_tags": _coerce_one_or_many_str(
            entry.get("night_alignment_tags")
        ),
        "planning_notes": str(entry.get("planning_notes") or "").strip(),
        "day_target_venue": entry.get("day_target_venue"),
        "night_target_venue": entry.get("night_target_venue"),
        "day_paper_types": [str(item) for item in day_paper_types],
        "night_paper_types": [str(item) for item in night_paper_types],
        "day_num_ideas": entry.get("day_num_ideas"),
        "night_num_ideas": entry.get("night_num_ideas"),
        "day_submission_mode": _maybe_bool(entry.get("day_submission_mode")),
        "night_submission_mode": _maybe_bool(entry.get("night_submission_mode")),
        "day_breakthrough_mode": _maybe_bool(entry.get("day_breakthrough_mode")),
        "night_breakthrough_mode": _maybe_bool(entry.get("night_breakthrough_mode")),
        "generator_args": [str(item) for item in generator_args],
        "day_generator_args": [str(item) for item in day_generator_args],
        "night_generator_args": [str(item) for item in night_generator_args],
    }


def _apply_source_overrides_to_args(
    args: list[str],
    source: dict[str, Any],
    daypart: str,
    *,
    desired_execution_policy: str | None = None,
) -> list[str]:
    updated = list(_clean_generator_args(args))
    updated.extend(source.get("generator_args") or [])
    updated.extend(source.get(f"{daypart}_generator_args") or [])

    target_venue = source.get(f"{daypart}_target_venue") or source.get("target_venue")
    if target_venue:
        updated = _set_passthrough_arg(updated, "--target-venue", target_venue)

    num_ideas = source.get(f"{daypart}_num_ideas")
    if num_ideas is None:
        num_ideas = source.get("num_ideas")
    if num_ideas is not None:
        updated = _set_passthrough_arg(updated, "--num-ideas", num_ideas)

    paper_types = source.get(f"{daypart}_paper_types") or source.get("paper_types")
    if paper_types:
        updated = _set_passthrough_multi_arg(updated, "--paper-types", paper_types)

    submission_mode = source.get(f"{daypart}_submission_mode")
    if submission_mode is None:
        submission_mode = source.get("submission_mode")
    if submission_mode and "--submission-mode" not in updated:
        updated.append("--submission-mode")

    breakthrough_mode = source.get(f"{daypart}_breakthrough_mode")
    if breakthrough_mode is None:
        breakthrough_mode = source.get("breakthrough_mode")
    if breakthrough_mode and "--breakthrough-mode" not in updated:
        updated.append("--breakthrough-mode")

    source_plan = build_source_planning_profile(
        source,
        daypart=daypart,
        desired_execution_policy=desired_execution_policy,
    )
    for item in source_plan.get("recommended_generator_defaults") or []:
        flag = item.get("flag")
        if not flag:
            continue
        if item.get("switch"):
            updated = _ensure_passthrough_flag(updated, str(flag))
            continue
        if item.get("value") is not None:
            updated = _set_passthrough_arg(updated, str(flag), item.get("value"))
    return updated


def _summarize_followup_uplift(summary: dict[str, Any]) -> dict[str, Any]:
    items = summary.get("items") or []
    deltas = [
        item.get("priority_delta")
        for item in items
        if isinstance(item.get("priority_delta"), (int, float))
    ]
    improved = [delta for delta in deltas if delta > 0]
    return {
        "count": len(items),
        "avg_priority_delta": round(sum(deltas) / len(deltas), 3) if deltas else 0.0,
        "improved_count": len(improved),
        "max_priority_delta": round(max(deltas), 3) if deltas else 0.0,
    }


def _derive_evidence_gap_actions(
    paper: dict[str, Any], policy_mode: str | None = None
) -> list[str]:
    reasons = set(_coerce_str_list(paper.get("self_review_round_gate_reasons")))
    next_focus = _coerce_str_list(paper.get("self_review_next_focus"))
    actions: list[str] = []

    if "critical_issues_unresolved" in reasons or (
        isinstance(paper.get("self_review_unresolved_critical"), int)
        and paper.get("self_review_unresolved_critical") > 0
    ):
        actions.append(
            "Resolve unresolved critical issues first; prioritize soundness and evidence validity before cosmetic rewrites."
        )
    if "high_value_coverage_low" in reasons:
        actions.append(
            "Increase high-value issue coverage by explicitly targeting P0/P1 issue_ids in the next rewrite pass."
        )
    if "rewrite_coverage_low" in reasons:
        actions.append(
            "Improve issue-linked rewrite coverage: ensure addressed_issue_ids align with recommended_targets."
        )
    if "persistent_issues_high" in reasons:
        actions.append(
            "Escalate persistent issues with section-level focused rewrites instead of global edits."
        )
    if "latex_compile_failed" in reasons:
        actions.append(
            "Stabilize LaTeX build before further rewrites to avoid rollback loops."
        )
    if isinstance(paper.get("unsupported_claims_count"), int) and paper.get(
        "unsupported_claims_count"
    ) > 0:
        actions.append(
            "Reduce unsupported claims by linking every core claim to a specific figure/table/result."
        )
    if isinstance(paper.get("evidence_density_score"), (int, float)) and float(
        paper.get("evidence_density_score")
    ) < 2.0:
        actions.append(
            "Increase evidence density with concrete quantitative statements and explicit result references."
        )
    if (
        isinstance(paper.get("experiment_todo_p0_count"), int)
        and paper.get("experiment_todo_p0_count") > 0
    ):
        top_todo = str(paper.get("experiment_todo_top_action") or "").strip()
        if top_todo:
            actions.append(f"Execute P0 experiment TODO first: {top_todo}")
        else:
            actions.append(
                "Execute unresolved P0 experiment TODO items before broad stylistic rewrites."
            )
    elif (
        isinstance(paper.get("experiment_todo_count"), int)
        and paper.get("experiment_todo_count") >= 3
    ):
        actions.append(
            "Batch similar experiment TODO items and close at least one high-value task this round."
        )
    if (
        isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
        and float(paper.get("experiment_todo_closure_rate")) < 0.5
        and isinstance(paper.get("experiment_todo_count"), int)
        and paper.get("experiment_todo_count") > 0
    ):
        actions.append(
            "TODO closure rate is low; run one strict close-the-loop pass and avoid adding new speculative tasks this round."
        )

    if next_focus:
        for focus in next_focus[:3]:
            actions.append(f"Focus item: {focus}")

    if policy_mode == "evidence_gap_repair":
        actions.append(
            "Use a strict evidence-repair rewrite style: tighten claims and foreground reproducible quantitative evidence."
        )

    deduped: list[str] = []
    for action in actions:
        text = str(action).strip()
        if not text or text in deduped:
            continue
        deduped.append(text)
    return deduped[:8]


def _build_autonomous_followup_focus(
    paper: dict[str, Any],
    policy: dict[str, Any],
    evidence_gap_actions: list[str],
) -> dict[str, Any]:
    if not evidence_gap_actions:
        return {}
    preferred_sections: list[str] = []
    if paper.get("rewrite_top_section"):
        preferred_sections.append(str(paper.get("rewrite_top_section")).strip())
    required_actions = [
        {
            "priority": "P0" if idx == 0 else "P1",
            "focus": "self_review_gate",
            "action": action,
            "reason": str(policy.get("reason") or "self-review gate indicates evidence gaps"),
        }
        for idx, action in enumerate(evidence_gap_actions[:4])
    ]
    notes = _coerce_str_list(paper.get("self_review_round_gate_reasons")) + evidence_gap_actions[:3]
    focus = {
        "preferred_sections": [item for item in preferred_sections if item],
        "notes": notes[:6],
        "required_actions": required_actions,
        "rebuttal_focus": evidence_gap_actions[:3],
        "anticipated_objections": _coerce_str_list(paper.get("self_review_next_focus"))[:3],
        "claim_softening_advice": (
            [
                "Soften over-strong claims when evidence is incomplete; scope claims to supported settings."
            ]
            if isinstance(paper.get("unsupported_claims_count"), int)
            and paper.get("unsupported_claims_count") > 0
            else []
        ),
        "candidate_boost": 2 if policy.get("mode") == "evidence_gap_repair" else 1,
        "target_section_limit": 5,
    }
    return focus


def _update_guardrail_state(
    parsed: argparse.Namespace, status: dict[str, Any]
) -> dict[str, Any]:
    views = status.get("last_views") or {}
    followup = status.get("last_rewrite_followup") or {}
    submission_items = int(views.get("submission_board_items") or 0)
    rewrite_items = int(views.get("rewrite_board_items") or 0)
    cycle_count = int(status.get("cycle", 0))
    followup_metrics = _summarize_followup_uplift(followup)

    low_uplift = (
        parsed.enable_rewrite_followup
        and parsed.rewrite_followup_top_k > 0
        and followup_metrics.get("count", 0) > 0
        and followup_metrics.get("avg_priority_delta", 0.0)
        < parsed.guardrail_min_followup_gain
    )
    strong_submission = submission_items >= parsed.guardrail_submission_target
    empty_rewrite = rewrite_items == 0

    status["consecutive_low_uplift_cycles"] = (
        (status.get("consecutive_low_uplift_cycles", 0) + 1) if low_uplift else 0
    )
    status["consecutive_strong_submission_cycles"] = (
        (status.get("consecutive_strong_submission_cycles", 0) + 1)
        if strong_submission
        else 0
    )
    status["consecutive_empty_rewrite_cycles"] = (
        (status.get("consecutive_empty_rewrite_cycles", 0) + 1) if empty_rewrite else 0
    )

    phase = "steady_state"
    phase_reason = "generation and rewrite are both active"
    if not parsed.ideas and (
        cycle_count <= parsed.phase_warmup_cycles
        or submission_items < parsed.phase_cold_start_submission_target
    ):
        phase = "cold_start"
        phase_reason = (
            "early cycles or too few strong drafts; bias toward discovering more ideas"
        )
    elif (
        submission_items >= parsed.phase_hot_polish_submission_target
        and rewrite_items > 0
    ):
        phase = "hot_polish"
        phase_reason = "enough strong drafts exist; bias toward polishing and uplift"

    mode = "balanced"
    reason = "default balanced mode"
    if phase == "cold_start":
        mode = "generate_more"
        reason = phase_reason
    elif phase == "hot_polish":
        mode = "focus_rewrite"
        reason = phase_reason
    elif (
        status.get("consecutive_strong_submission_cycles", 0)
        >= parsed.guardrail_strong_cycles
    ):
        mode = "focus_rewrite"
        reason = (
            "submission board has enough strong drafts; focus on deeper improvement"
        )
    elif (
        status.get("consecutive_low_uplift_cycles", 0)
        >= parsed.guardrail_stagnation_cycles
        or status.get("consecutive_empty_rewrite_cycles", 0)
        >= parsed.guardrail_empty_rewrite_cycles
    ):
        mode = "generate_more"
        reason = "rewrite uplift is weak or rewrite board is empty; bias back toward fresh generation"

    control = status.get("control") or {}
    if control.get("force_phase") in {"cold_start", "steady_state", "hot_polish"}:
        phase = control.get("force_phase")
        phase_reason = "forced by daemon_control.json"
    if control.get("force_mode") in {"balanced", "generate_more", "focus_rewrite"}:
        mode = control.get("force_mode")
        reason = "forced by daemon_control.json"
    status["guardrail_phase"] = phase
    status["guardrail_phase_reason"] = phase_reason
    status["guardrail_mode"] = mode
    status["guardrail_reason"] = reason
    status["guardrail_followup_metrics"] = followup_metrics
    return status


def _active_source_feedback(status: dict[str, Any]) -> dict[str, Any]:
    return get_active_source_feedback(status)


def _apply_quality_strategy_feedback(
    parsed: argparse.Namespace,
    status: dict[str, Any],
    passthrough: list[str],
) -> list[str]:
    if not parsed.auto_quality_strategy_feedback:
        status["active_quality_strategy"] = {
            "enabled": False,
            "mode": "disabled",
            "reason": "quality strategy feedback disabled",
        }
        return passthrough

    feedback = _active_source_feedback(status)
    active_source = status.get("active_source") or {}
    current_daypart = status.get("current_daypart") or _current_daypart(parsed)
    if not feedback:
        status["active_quality_strategy"] = {
            "enabled": True,
            "mode": "neutral",
            "reason": "no source quality feedback available yet",
        }
        return passthrough

    updated = list(_clean_generator_args(passthrough))
    avg_priority = feedback.get("avg_priority")
    ready_rate = feedback.get("ready_rate")
    gate_pass_rate = feedback.get("gate_pass_rate")
    best_priority = feedback.get("best_priority")

    if (
        isinstance(avg_priority, (int, float))
        and avg_priority >= parsed.quality_strategy_submission_priority_threshold
    ) or (
        isinstance(ready_rate, (int, float))
        and ready_rate >= parsed.quality_strategy_ready_rate_threshold
    ):
        updated = _ensure_passthrough_flag(updated, "--submission-mode")
        updated = _ensure_passthrough_flag(updated, "--rank-ideas")
        updated = _set_passthrough_arg(updated, "--top-k-ideas", 1)
        dominant_paper_type = feedback.get("dominant_paper_type")
        dominant_paper_type_rate = feedback.get("dominant_paper_type_rate")
        target_paper_types = ["journal"]
        if (
            dominant_paper_type
            and isinstance(dominant_paper_type_rate, (int, float))
            and dominant_paper_type_rate
            >= parsed.quality_strategy_dominant_paper_type_rate_threshold
        ):
            target_paper_types = [str(dominant_paper_type)]
        updated = _set_passthrough_multi_arg(
            updated, "--paper-types", target_paper_types
        )
        if not active_source.get("target_venue") and not active_source.get(
            f"{current_daypart}_target_venue"
        ):
            target_venue = "nature"
            dominant_venue = feedback.get("dominant_venue")
            dominant_venue_rate = feedback.get("dominant_venue_rate")
            if (
                dominant_venue
                and isinstance(dominant_venue_rate, (int, float))
                and dominant_venue_rate
                >= parsed.quality_strategy_dominant_venue_rate_threshold
            ):
                target_venue = str(dominant_venue)
            updated = _set_passthrough_arg(updated, "--target-venue", target_venue)
        num_ideas = _passthrough_arg_value(
            updated, "--num-ideas", parsed.guardrail_default_num_ideas
        )
        try:
            num_ideas = max(
                1,
                min(
                    int(num_ideas),
                    parsed.quality_strategy_max_num_ideas_for_strong_sources,
                ),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_quality_strategy"] = {
            "enabled": True,
            "mode": "submission_push",
            "reason": "source has strong historical quality outcomes; bias toward fewer, higher-quality Nature-style drafts",
            "feedback": feedback,
        }
        return updated

    if (
        isinstance(best_priority, (int, float))
        and best_priority < parsed.quality_strategy_exploration_priority_ceiling
        and isinstance(gate_pass_rate, (int, float))
        and gate_pass_rate <= parsed.quality_strategy_gate_pass_floor
    ):
        updated = _remove_passthrough_flag(updated, "--target-venue")
        updated = _remove_passthrough_switch(updated, "--breakthrough-mode")
        updated = _ensure_passthrough_flag(updated, "--rank-ideas")
        top_k = _passthrough_arg_value(updated, "--top-k-ideas", 2)
        try:
            updated = _set_passthrough_arg(
                updated,
                "--top-k-ideas",
                max(2, int(top_k)),
            )
        except (TypeError, ValueError):
            updated = _set_passthrough_arg(updated, "--top-k-ideas", 2)
        num_ideas = _passthrough_arg_value(
            updated, "--num-ideas", parsed.guardrail_default_num_ideas
        )
        try:
            num_ideas = min(
                parsed.quality_strategy_max_num_ideas_for_weak_sources,
                max(int(num_ideas) + 1, parsed.guardrail_default_num_ideas),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_quality_strategy"] = {
            "enabled": True,
            "mode": "explore_more",
            "reason": "source has weak historical quality outcomes; broaden exploration before polishing harder",
            "feedback": feedback,
        }
        return updated

    status["active_quality_strategy"] = {
        "enabled": True,
        "mode": "balanced",
        "reason": "source quality feedback suggests keeping a balanced generation strategy",
        "feedback": feedback,
    }
    return updated


def _apply_evidence_strategy_feedback(
    parsed: argparse.Namespace,
    status: dict[str, Any],
    passthrough: list[str],
) -> list[str]:
    if not parsed.auto_evidence_strategy_feedback:
        status["active_evidence_strategy"] = {
            "enabled": False,
            "mode": "disabled",
            "reason": "evidence strategy feedback disabled",
        }
        return passthrough

    feedback = _active_source_feedback(status)
    if not feedback:
        status["active_evidence_strategy"] = {
            "enabled": True,
            "mode": "neutral",
            "reason": "no evidence feedback available yet",
        }
        return passthrough

    avg_claim_support = feedback.get("avg_claim_support")
    avg_claim_alignment = feedback.get("avg_claim_alignment")
    avg_numeric_coverage = feedback.get("avg_numeric_coverage")
    avg_evidence_density = feedback.get("avg_evidence_density")
    avg_unsupported_claims = feedback.get("avg_unsupported_claims")
    self_review_gate_ready_rate = feedback.get("self_review_gate_ready_rate")
    avg_self_review_high_value_coverage = feedback.get(
        "avg_self_review_high_value_coverage"
    )
    avg_self_review_unresolved_critical = feedback.get(
        "avg_self_review_unresolved_critical"
    )
    avg_experiment_todo = feedback.get("avg_experiment_todo")
    avg_experiment_todo_p0 = feedback.get("avg_experiment_todo_p0")
    avg_experiment_todo_closure_rate = feedback.get("avg_experiment_todo_closure_rate")
    claim_support_floor = float(
        getattr(parsed, "evidence_strategy_claim_support_floor", 3.6)
    )
    claim_alignment_floor = float(
        getattr(parsed, "evidence_strategy_claim_alignment_floor", 3.2)
    )
    numeric_coverage_floor = float(
        getattr(parsed, "evidence_strategy_numeric_coverage_floor", 3.8)
    )
    evidence_density_floor = float(
        getattr(parsed, "evidence_strategy_evidence_density_floor", 2.0)
    )
    unsupported_claims_ceiling = float(
        getattr(parsed, "evidence_strategy_unsupported_claims_ceiling", 1.0)
    )
    round_gate_ready_floor = float(
        getattr(parsed, "evidence_strategy_round_gate_ready_floor", 0.55)
    )
    high_value_coverage_floor = float(
        getattr(parsed, "evidence_strategy_high_value_coverage_floor", 0.65)
    )
    self_review_critical_ceiling = float(
        getattr(parsed, "evidence_strategy_self_review_critical_ceiling", 0.5)
    )
    experiment_todo_ceiling = float(
        getattr(parsed, "evidence_strategy_experiment_todo_ceiling", 2.5)
    )
    experiment_todo_p0_ceiling = float(
        getattr(parsed, "evidence_strategy_experiment_todo_p0_ceiling", 0.5)
    )
    experiment_todo_closure_floor = float(
        getattr(parsed, "evidence_strategy_experiment_todo_closure_floor", 0.35)
    )
    min_quality_rewrite_rounds = int(
        getattr(parsed, "evidence_strategy_min_quality_rewrite_rounds", 2)
    )
    todo_min_quality_rewrite_rounds = int(
        getattr(parsed, "evidence_strategy_todo_min_quality_rewrite_rounds", 3)
    )
    max_num_ideas_under_todo_pressure = int(
        getattr(parsed, "evidence_strategy_max_num_ideas_under_todo_pressure", 2)
    )
    review_strategy = str(getattr(parsed, "evidence_strategy_review_strategy", "depth"))
    guardrail_default_num_ideas = int(getattr(parsed, "guardrail_default_num_ideas", 3))
    updated = list(_clean_generator_args(passthrough))
    evidence_low = any(
        [
            isinstance(avg_claim_support, (int, float))
            and avg_claim_support < claim_support_floor,
            isinstance(avg_numeric_coverage, (int, float))
            and avg_numeric_coverage < numeric_coverage_floor,
            isinstance(avg_evidence_density, (int, float))
            and avg_evidence_density < evidence_density_floor,
            isinstance(avg_claim_alignment, (int, float))
            and avg_claim_alignment < claim_alignment_floor,
            isinstance(avg_unsupported_claims, (int, float))
            and avg_unsupported_claims > unsupported_claims_ceiling,
            isinstance(self_review_gate_ready_rate, (int, float))
            and self_review_gate_ready_rate < round_gate_ready_floor,
            isinstance(avg_self_review_high_value_coverage, (int, float))
            and avg_self_review_high_value_coverage < high_value_coverage_floor,
            isinstance(avg_self_review_unresolved_critical, (int, float))
            and avg_self_review_unresolved_critical > self_review_critical_ceiling,
            isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo > experiment_todo_ceiling,
            isinstance(avg_experiment_todo_p0, (int, float))
            and avg_experiment_todo_p0 > experiment_todo_p0_ceiling,
            isinstance(avg_experiment_todo_closure_rate, (int, float))
            and avg_experiment_todo_closure_rate < experiment_todo_closure_floor
            and isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo > 0,
        ]
    )
    if not evidence_low:
        status["active_evidence_strategy"] = {
            "enabled": True,
            "mode": "evidence_healthy",
            "reason": "recent evidence metrics are strong enough to keep the current review depth",
            "feedback": feedback,
        }
        return updated

    updated = _ensure_passthrough_flag(updated, "--high-quality-mode")
    updated = _remove_passthrough_switch(updated, "--breakthrough-mode")
    current_preset = _passthrough_arg_value(updated, "--quality-preset", "balanced")
    target_preset = (
        "publishable"
        if (status.get("active_quality_strategy") or {}).get("mode")
        == "submission_push"
        else "high"
    )
    preset_order = {"balanced": 0, "high": 1, "publishable": 2}
    if preset_order.get(str(current_preset), 0) < preset_order.get(target_preset, 1):
        updated = _set_passthrough_arg(updated, "--quality-preset", target_preset)
    updated = _set_passthrough_arg(
        updated, "--review-strategy", review_strategy
    )
    current_rounds = _passthrough_arg_value(updated, "--quality-rewrite-rounds", 0)
    try:
        effective_rounds = max(
            int(current_rounds),
            min_quality_rewrite_rounds,
        )
    except (TypeError, ValueError):
        effective_rounds = min_quality_rewrite_rounds
    todo_pressure = any(
        [
            isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo > experiment_todo_ceiling,
            isinstance(avg_experiment_todo_p0, (int, float))
            and avg_experiment_todo_p0 > experiment_todo_p0_ceiling,
            isinstance(avg_experiment_todo_closure_rate, (int, float))
            and avg_experiment_todo_closure_rate < experiment_todo_closure_floor
            and isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo > 0,
        ]
    )
    if todo_pressure:
        effective_rounds = max(
            effective_rounds,
            todo_min_quality_rewrite_rounds,
        )
        current_num_ideas = _passthrough_arg_value(
            updated, "--num-ideas", guardrail_default_num_ideas
        )
        try:
            capped_num_ideas = max(
                1,
                min(
                    int(current_num_ideas),
                    max_num_ideas_under_todo_pressure,
                ),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", capped_num_ideas)
        except (TypeError, ValueError):
            pass
    updated = _set_passthrough_arg(
        updated, "--quality-rewrite-rounds", effective_rounds
    )
    status["active_evidence_strategy"] = {
        "enabled": True,
        "mode": "evidence_todo_repair" if todo_pressure else "evidence_rebuild",
        "reason": (
            "evidence metrics are weak and experiment TODO backlog is high; deepen review, raise rewrite rounds, and cap idea exploration"
            if todo_pressure
            else "claim/evidence metrics or self-review round-gate health are weak; deepen review and rewrite pressure"
        ),
        "feedback": feedback,
    }
    return updated


def _apply_pipeline_contract_feedback(
    parsed: argparse.Namespace,
    status: dict[str, Any],
    passthrough: list[str],
) -> list[str]:
    summary = _build_pipeline_contract_summary(ResearchManager(parsed.research_dir))
    status["pipeline_contract_summary"] = summary
    if not summary.get("enabled"):
        status["active_pipeline_contract_strategy"] = {
            "enabled": False,
            "mode": "disabled",
            "reason": "no contract-enabled pipeline roots detected yet",
        }
        return list(_clean_generator_args(passthrough))

    blocked_project_floor = int(
        getattr(parsed, "pipeline_contract_blocked_project_floor", 1)
    )
    blocked_stage_floor = int(
        getattr(parsed, "pipeline_contract_blocked_stage_floor", 1)
    )
    missing_stage_floor = int(
        getattr(parsed, "pipeline_contract_missing_stage_floor", 2)
    )
    attention_stage_floor = int(
        getattr(parsed, "pipeline_contract_attention_stage_floor", 3)
    )
    review_low_resolution_floor = int(
        getattr(parsed, "pipeline_contract_review_low_resolution_floor", 2)
    )
    review_low_binding_floor = int(
        getattr(parsed, "pipeline_contract_review_low_binding_floor", 2)
    )
    review_low_repair_ready_floor = int(
        getattr(parsed, "pipeline_contract_review_low_repair_ready_floor", 2)
    )
    review_persistent_issue_floor = int(
        getattr(parsed, "pipeline_contract_review_persistent_issue_floor", 3)
    )
    failed_experiment_floor = int(
        getattr(parsed, "pipeline_contract_failed_experiment_floor", 2)
    )
    blocked_figure_floor = int(
        getattr(parsed, "pipeline_contract_blocked_figure_floor", 2)
    )
    budget_exhausted_floor = int(
        getattr(parsed, "pipeline_contract_budget_exhausted_floor", 2)
    )
    strict_fallback_floor = int(
        getattr(parsed, "pipeline_contract_strict_fallback_floor", 2)
    )
    fallback_heavy_project_floor = int(
        getattr(parsed, "pipeline_contract_fallback_heavy_project_floor", 2)
    )
    self_evolution_blocked_floor = int(
        getattr(parsed, "pipeline_contract_self_evolution_blocked_floor", 1)
    )
    self_evolution_attention_floor = int(
        getattr(parsed, "pipeline_contract_self_evolution_attention_floor", 2)
    )
    self_evolution_required_failure_floor = int(
        getattr(parsed, "pipeline_contract_self_evolution_required_failure_floor", 2)
    )
    process_alignment_blocked_floor = int(
        getattr(parsed, "pipeline_contract_process_alignment_blocked_floor", 1)
    )
    process_alignment_missing_floor = int(
        getattr(parsed, "pipeline_contract_process_alignment_missing_floor", 2)
    )
    min_quality_rewrite_rounds = int(
        getattr(parsed, "pipeline_contract_min_quality_rewrite_rounds", 3)
    )
    max_num_ideas_under_pressure = int(
        getattr(parsed, "pipeline_contract_max_num_ideas_under_pressure", 2)
    )
    program_max_num_ideas = int(
        getattr(parsed, "pipeline_contract_program_max_num_ideas", 2)
    )
    agentic_min_num_ideas = int(
        getattr(parsed, "pipeline_contract_agentic_min_num_ideas", 4)
    )
    min_guardrail_repairs = int(
        getattr(parsed, "pipeline_contract_min_guardrail_repairs", 2)
    )
    review_strategy = str(
        getattr(parsed, "pipeline_contract_review_strategy", "depth")
    )
    updated = list(_clean_generator_args(passthrough))
    dominant_execution_policy = str(
        summary.get("dominant_execution_policy") or "unknown"
    )
    budget_exhausted_experiments = int(
        summary.get("budget_exhausted_experiment_count") or 0
    )

    pressure = any(
        [
            int(summary.get("blocked_project_count") or 0) >= blocked_project_floor,
            int(summary.get("stage_blocked_project_count") or 0) >= blocked_stage_floor,
            int(summary.get("stage_missing_project_count") or 0) >= missing_stage_floor,
            int(summary.get("stage_attention_project_count") or 0)
            >= attention_stage_floor,
            int(summary.get("review_low_resolution_project_count") or 0)
            >= review_low_resolution_floor,
            int(summary.get("review_low_binding_project_count") or 0)
            >= review_low_binding_floor,
            int(summary.get("review_low_repair_ready_project_count") or 0)
            >= review_low_repair_ready_floor,
            int(summary.get("review_persistent_issue_count") or 0)
            >= review_persistent_issue_floor,
            int(summary.get("failed_experiment_count") or 0)
            >= failed_experiment_floor,
            int(summary.get("blocked_figure_count") or 0) >= blocked_figure_floor,
            budget_exhausted_experiments >= budget_exhausted_floor,
            int(summary.get("strict_fallback_count") or 0) >= strict_fallback_floor,
            int(summary.get("fallback_heavy_project_count") or 0)
            >= fallback_heavy_project_floor,
            int(summary.get("process_alignment_blocked_project_count") or 0)
            >= process_alignment_blocked_floor,
            int(summary.get("process_alignment_missing_project_count") or 0)
            >= process_alignment_missing_floor,
            int(summary.get("blocked_self_evolution_project_count") or 0)
            >= self_evolution_blocked_floor,
            int(summary.get("self_evolution_attention_project_count") or 0)
            >= self_evolution_attention_floor,
            int(summary.get("self_evolution_required_failure_count") or 0)
            >= self_evolution_required_failure_floor,
        ]
    )
    if not pressure:
        status["active_pipeline_contract_strategy"] = {
            "enabled": True,
            "mode": "contracts_healthy",
            "reason": "contract artifacts, stage standards, reviewer repair closure, and self-evolution self-checks do not currently show severe experiment, evidence, fallback, or repair-planning debt pressure",
            "summary": summary,
        }
        return updated

    current_num_ideas = _passthrough_arg_value(
        updated,
        "--num-ideas",
        getattr(parsed, "guardrail_default_num_ideas", 3),
    )

    if (
        dominant_execution_policy == "program_driven"
        and budget_exhausted_experiments >= budget_exhausted_floor
    ):
        updated = _set_passthrough_arg(updated, "--workflow-mode", "program_driven")
        updated = _ensure_passthrough_flag(updated, "--submission-mode")
        updated = _remove_passthrough_switch(updated, "--breakthrough-mode")
        updated = _ensure_passthrough_flag(updated, "--high-quality-mode")
        updated = _ensure_passthrough_flag(updated, "--strict-writing-guardrails")
        updated = _set_passthrough_arg(updated, "--review-strategy", review_strategy)
        updated = _set_passthrough_arg(updated, "--writing-audit-rounds", 1)
        current_repairs = _passthrough_arg_value(updated, "--guardrail-repair-rounds", 1)
        try:
            effective_repairs = max(int(current_repairs), min_guardrail_repairs)
        except (TypeError, ValueError):
            effective_repairs = min_guardrail_repairs
        updated = _set_passthrough_arg(
            updated, "--guardrail-repair-rounds", effective_repairs
        )
        current_rounds = _passthrough_arg_value(updated, "--quality-rewrite-rounds", 0)
        try:
            effective_rounds = max(int(current_rounds), min_quality_rewrite_rounds)
        except (TypeError, ValueError):
            effective_rounds = min_quality_rewrite_rounds
        updated = _set_passthrough_arg(
            updated, "--quality-rewrite-rounds", effective_rounds
        )
        try:
            capped_num_ideas = max(
                1,
                min(int(current_num_ideas), program_max_num_ideas),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", capped_num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_pipeline_contract_strategy"] = {
            "enabled": True,
            "mode": "program_budget_repair",
            "reason": "program-driven runs are exhausting budget; tighten exploration and reinforce research-program discipline",
            "selected_execution_policy": dominant_execution_policy,
            "summary": summary,
        }
        return updated

    if (
        dominant_execution_policy == "agentic_tree"
        and int(summary.get("failed_experiment_count") or 0) >= failed_experiment_floor
        and int(summary.get("blocked_figure_count") or 0) < blocked_figure_floor
        and budget_exhausted_experiments < budget_exhausted_floor
    ):
        updated = _set_passthrough_arg(updated, "--workflow-mode", "agentic_tree")
        updated = _ensure_passthrough_flag(updated, "--breakthrough-mode")
        updated = _remove_passthrough_switch(updated, "--submission-mode")
        updated = _set_passthrough_arg(updated, "--review-strategy", review_strategy)
        current_followups = _passthrough_arg_value(
            updated, "--autonomous-quality-followup-rounds", 0
        )
        try:
            effective_followups = max(int(current_followups), 1)
        except (TypeError, ValueError):
            effective_followups = 1
        updated = _set_passthrough_arg(
            updated, "--autonomous-quality-followup-rounds", effective_followups
        )
        try:
            expanded_num_ideas = max(int(current_num_ideas), agentic_min_num_ideas)
            updated = _set_passthrough_arg(updated, "--num-ideas", expanded_num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_pipeline_contract_strategy"] = {
            "enabled": True,
            "mode": "agentic_exploration_rebuild",
            "reason": "agentic exploration is failing before evidence packaging becomes the bottleneck; widen search pressure and preserve exploratory branching",
            "selected_execution_policy": dominant_execution_policy,
            "summary": summary,
        }
        return updated

    if (
        int(summary.get("process_alignment_blocked_project_count") or 0)
        >= process_alignment_blocked_floor
        or int(summary.get("process_alignment_missing_project_count") or 0)
        >= process_alignment_missing_floor
    ):
        updated = _set_passthrough_arg(updated, "--workflow-mode", "program_driven")
        updated = _ensure_passthrough_flag(updated, "--high-quality-mode")
        updated = _ensure_passthrough_flag(updated, "--strict-writing-guardrails")
        updated = _set_passthrough_arg(updated, "--review-strategy", review_strategy)
        current_rounds = _passthrough_arg_value(updated, "--quality-rewrite-rounds", 0)
        try:
            effective_rounds = max(int(current_rounds), min_quality_rewrite_rounds)
        except (TypeError, ValueError):
            effective_rounds = min_quality_rewrite_rounds
        updated = _set_passthrough_arg(
            updated, "--quality-rewrite-rounds", effective_rounds
        )
        try:
            capped_num_ideas = max(
                1,
                min(int(current_num_ideas), max_num_ideas_under_pressure),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", capped_num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_pipeline_contract_strategy"] = {
            "enabled": True,
            "mode": "process_alignment_repair",
            "reason": "process-level alignment against ideation/program/exploration/experiment/figure/writing/review/evolution/packaging standards is blocked or incomplete; switch to program-driven repair until the process map is coherent again",
            "selected_execution_policy": "program_driven",
            "summary": summary,
        }
        return updated

    if (
        int(summary.get("blocked_self_evolution_project_count") or 0)
        >= self_evolution_blocked_floor
        or int(summary.get("self_evolution_required_failure_count") or 0)
        >= self_evolution_required_failure_floor
        or int(summary.get("self_evolution_attention_project_count") or 0)
        >= self_evolution_attention_floor
    ):
        updated = _set_passthrough_arg(updated, "--workflow-mode", "review_board")
        updated = _ensure_passthrough_flag(updated, "--high-quality-mode")
        updated = _ensure_passthrough_flag(updated, "--strict-writing-guardrails")
        updated = _set_passthrough_arg(updated, "--review-strategy", review_strategy)
        updated = _set_passthrough_arg(updated, "--writing-audit-rounds", 1)
        current_rounds = _passthrough_arg_value(updated, "--quality-rewrite-rounds", 0)
        try:
            effective_rounds = max(int(current_rounds), min_quality_rewrite_rounds)
        except (TypeError, ValueError):
            effective_rounds = min_quality_rewrite_rounds
        updated = _set_passthrough_arg(
            updated, "--quality-rewrite-rounds", effective_rounds
        )
        current_repairs = _passthrough_arg_value(
            updated, "--guardrail-repair-rounds", 1
        )
        try:
            effective_repairs = max(int(current_repairs), min_guardrail_repairs)
        except (TypeError, ValueError):
            effective_repairs = min_guardrail_repairs
        updated = _set_passthrough_arg(
            updated, "--guardrail-repair-rounds", effective_repairs
        )
        try:
            capped_num_ideas = max(
                1,
                min(int(current_num_ideas), max_num_ideas_under_pressure),
            )
            updated = _set_passthrough_arg(updated, "--num-ideas", capped_num_ideas)
        except (TypeError, ValueError):
            pass
        status["active_pipeline_contract_strategy"] = {
            "enabled": True,
            "mode": "self_evolution_rebuild",
            "reason": "self-evolution self-checks are blocked or under-specified; switch to review-board hardening, deepen rewrite pressure, and force explicit repair verification before scaling the next batch",
            "selected_execution_policy": "review_board",
            "summary": summary,
        }
        return updated

    if dominant_execution_policy in {"review_board", "multi_agent_board"}:
        updated = _set_passthrough_arg(
            updated, "--workflow-mode", dominant_execution_policy
        )

    updated = _ensure_passthrough_flag(updated, "--high-quality-mode")
    updated = _ensure_passthrough_flag(updated, "--strict-writing-guardrails")
    updated = _set_passthrough_arg(updated, "--review-strategy", review_strategy)

    current_rounds = _passthrough_arg_value(updated, "--quality-rewrite-rounds", 0)
    try:
        effective_rounds = max(int(current_rounds), min_quality_rewrite_rounds)
    except (TypeError, ValueError):
        effective_rounds = min_quality_rewrite_rounds
    updated = _set_passthrough_arg(
        updated, "--quality-rewrite-rounds", effective_rounds
    )

    current_repairs = _passthrough_arg_value(updated, "--guardrail-repair-rounds", 1)
    try:
        effective_repairs = max(int(current_repairs), min_guardrail_repairs)
    except (TypeError, ValueError):
        effective_repairs = min_guardrail_repairs
    updated = _set_passthrough_arg(
        updated, "--guardrail-repair-rounds", effective_repairs
    )

    try:
        capped_num_ideas = max(
            1,
            min(int(current_num_ideas), max_num_ideas_under_pressure),
        )
        updated = _set_passthrough_arg(updated, "--num-ideas", capped_num_ideas)
    except (TypeError, ValueError):
        pass

    status["active_pipeline_contract_strategy"] = {
        "enabled": True,
        "mode": (
            "review_board_hardening"
            if dominant_execution_policy == "review_board"
            else "contract_blocker_repair"
        ),
        "reason": (
            "review-board style runs are bottlenecked on blocked artifacts, blocked stage standards, reviewer repair debt, weak issue-target bindings, under-specified repair queues, self-evolution debt, or strict fallback debt; intensify guardrails and review-driven repair"
            if dominant_execution_policy == "review_board"
            else "pipeline contracts show blocked experiments, blocked stage standards, unresolved reviewer debt, weak issue-target bindings, under-specified repair queues, self-evolution debt, figures, or fallback debt; reduce exploration and intensify evidence repair"
        ),
        "selected_execution_policy": dominant_execution_policy,
        "summary": summary,
    }
    return updated


def _effective_generator_args(
    parsed: argparse.Namespace, status: dict[str, Any]
) -> list[str]:
    active_source = status.get("active_source") or {}
    daypart = status.get("current_daypart") or _current_daypart(parsed)
    desired_execution_policy = str(
        (
            status.get("active_pipeline_contract_strategy") or {}
        ).get("selected_execution_policy")
        or (status.get("pipeline_contract_summary") or {}).get(
            "dominant_execution_policy"
        )
        or ""
    ).strip() or None
    passthrough = _apply_source_overrides_to_args(
        parsed.generator_args,
        active_source,
        daypart,
        desired_execution_policy=desired_execution_policy,
    )
    if active_source.get("type") == "ideas":
        return passthrough

    base_num_ideas = _passthrough_arg_value(passthrough, "--num-ideas", None)
    try:
        base_num_ideas = int(base_num_ideas) if base_num_ideas is not None else None
    except ValueError:
        base_num_ideas = None
    if base_num_ideas is None:
        base_num_ideas = parsed.guardrail_default_num_ideas

    phase = status.get("guardrail_phase", "steady_state")
    mode = status.get("guardrail_mode", "balanced")
    effective_num_ideas = base_num_ideas
    if phase == "cold_start":
        effective_num_ideas = min(
            parsed.guardrail_max_num_ideas,
            base_num_ideas + parsed.guardrail_num_ideas_step * 2,
        )
    elif phase == "hot_polish" or mode == "focus_rewrite":
        effective_num_ideas = max(
            parsed.guardrail_min_num_ideas,
            base_num_ideas - parsed.guardrail_num_ideas_step,
        )
    elif mode == "generate_more":
        effective_num_ideas = min(
            parsed.guardrail_max_num_ideas,
            base_num_ideas + parsed.guardrail_num_ideas_step,
        )

    passthrough = _set_passthrough_arg(passthrough, "--num-ideas", effective_num_ideas)
    passthrough = _apply_quality_strategy_feedback(parsed, status, passthrough)
    passthrough = _apply_evidence_strategy_feedback(parsed, status, passthrough)
    passthrough = _apply_pipeline_contract_feedback(parsed, status, passthrough)
    return passthrough


def _build_generator_command(
    parsed: argparse.Namespace, batch_name: str, status: dict[str, Any]
) -> list[str]:
    source = _select_source(parsed, status)
    cmd = [
        parsed.python,
        "continuous_paper_generator.py",
        "--research-dir",
        parsed.research_dir,
        "--batch-name",
        batch_name,
    ]
    if source.get("type") == "topic":
        cmd.extend(["--topic", source.get("value")])
    elif source.get("type") == "ideas":
        cmd.extend(["--ideas", source.get("value")])

    passthrough = _effective_generator_args(parsed, status)
    if (
        parsed.default_submission_mode
        and "--submission-mode" not in passthrough
        and "--breakthrough-mode" not in passthrough
    ):
        cmd.append("--submission-mode")
    cmd.extend(passthrough)
    return cmd


def _export_cycle_views(
    manager: ResearchManager, parsed: argparse.Namespace, daemon_dir: Path
) -> dict[str, str | None]:
    submission_board = manager.submission_board(
        top_n_per_venue=parsed.submission_board_top,
        min_priority=parsed.submission_board_min_priority,
        max_blockers=parsed.submission_board_max_blockers,
        min_rewrite_gain=parsed.submission_board_min_rewrite_gain,
        require_gate=parsed.submission_board_require_gate,
    )
    submission_board_path = daemon_dir / "latest_submission_board.md"
    manager.export_submission_board_markdown(
        submission_board, str(submission_board_path)
    )

    rewrite_board = manager.rewrite_board(
        top_n=parsed.rewrite_board_top,
        target_venue=parsed.rewrite_board_venue,
        min_priority=parsed.rewrite_board_min_priority,
        min_rewrite_gain=parsed.rewrite_board_min_gain,
        max_blockers=parsed.rewrite_board_max_blockers,
        require_gate=parsed.rewrite_board_require_gate,
        include_ready=parsed.rewrite_board_include_ready,
    )
    rewrite_board_path = daemon_dir / "latest_rewrite_board.md"
    manager.export_rewrite_board_markdown(rewrite_board, str(rewrite_board_path))

    shortlist = manager.shortlist_papers(
        target_venue=parsed.shortlist_venue,
        require_gate=parsed.shortlist_require_gate,
        require_ready=parsed.shortlist_require_ready,
        min_priority=parsed.shortlist_min_priority,
        max_blockers=parsed.shortlist_max_blockers,
        min_rewrite_gain=parsed.shortlist_min_rewrite_gain,
        top_n=parsed.shortlist_top,
    )
    shortlist_path = daemon_dir / "latest_shortlist.md"
    manager.export_shortlist_markdown(shortlist, str(shortlist_path))

    return {
        "submission_board": str(submission_board_path),
        "rewrite_board": str(rewrite_board_path),
        "shortlist": str(shortlist_path),
        "submission_board_items": sum(
            len(items) for items in submission_board.values()
        ),
        "rewrite_board_items": len(rewrite_board),
        "shortlist_items": len(shortlist),
    }


def _derive_rewrite_followup_policy(
    parsed: argparse.Namespace,
    paper: dict[str, Any],
) -> dict[str, Any]:
    base_preset_name = parsed.rewrite_followup_preset
    base_preset = QUALITY_PRESETS.get(base_preset_name, QUALITY_PRESETS["publishable"])
    execution_policy = build_workflow_execution_policy(
        paper.get("workflow_mode"),
        submission_mode=(paper.get("template_profile") == "template_first"),
        high_quality_mode=paper.get("quality_gate_passed") is not None,
        target_venue=paper.get("target_venue"),
    )
    policy = {
        "skip": False,
        "reason": "base follow-up policy",
        "mode": "standard",
        "preset_name": base_preset_name,
        "quality_threshold": parsed.rewrite_followup_quality_threshold
        or base_preset["quality_threshold"],
        "rigor_threshold": parsed.rewrite_followup_rigor_threshold
        or base_preset["rigor_threshold"],
        "max_rewrite_rounds": parsed.rewrite_followup_max_rounds
        or base_preset["max_rewrite_rounds"],
        "auto_improvement_fallback": (
            execution_policy.allow_auto_improvement_fallback
        ),
        "quality_fallback_policy": execution_policy.quality_fallback_policy,
    }
    if not parsed.adaptive_rewrite_followup:
        policy["reason"] = "adaptive rewrite follow-up disabled"
        return policy

    blockers = paper.get("blocker_count")
    if not isinstance(blockers, int):
        blockers = None
    rewrite_gain = paper.get("rewrite_priority_gain_total")
    if not isinstance(rewrite_gain, (int, float)):
        rewrite_gain = None
    submission_priority = paper.get("submission_priority_score")
    if not isinstance(submission_priority, (int, float)):
        submission_priority = None
    submission_status = str(paper.get("submission_status") or "").lower()
    gate_passed = paper.get("quality_gate_passed")
    self_review_gate_ready = paper.get("self_review_round_gate_ready")
    self_review_gate_score = paper.get("self_review_round_gate_score")
    if not isinstance(self_review_gate_score, (int, float)):
        self_review_gate_score = None
    self_review_high_value_coverage = paper.get("self_review_high_value_coverage")
    if not isinstance(self_review_high_value_coverage, (int, float)):
        self_review_high_value_coverage = None
    self_review_unresolved_critical = paper.get("self_review_unresolved_critical")
    if not isinstance(self_review_unresolved_critical, int):
        self_review_unresolved_critical = 0
    experiment_todo_count = (
        int(paper.get("experiment_todo_count"))
        if isinstance(paper.get("experiment_todo_count"), int)
        else 0
    )
    experiment_todo_p0_count = (
        int(paper.get("experiment_todo_p0_count"))
        if isinstance(paper.get("experiment_todo_p0_count"), int)
        else 0
    )
    self_review_gate_score_floor = float(
        getattr(parsed, "rewrite_followup_self_review_gate_score_floor", 75.0)
    )
    self_review_high_value_floor = float(
        getattr(
            parsed,
            "rewrite_followup_self_review_high_value_coverage_floor",
            0.7,
        )
    )
    self_review_min_rounds = int(
        getattr(parsed, "rewrite_followup_self_review_min_rounds", 2)
    )
    experiment_todo_p0_floor = int(
        getattr(parsed, "rewrite_followup_experiment_todo_p0_floor", 1)
    )
    experiment_todo_count_floor = int(
        getattr(parsed, "rewrite_followup_experiment_todo_count_floor", 4)
    )
    experiment_todo_min_rounds = int(
        getattr(parsed, "rewrite_followup_experiment_todo_min_rounds", 2)
    )
    experiment_todo_closure_rate = (
        float(paper.get("experiment_todo_closure_rate"))
        if isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
        else None
    )
    experiment_todo_closure_floor = float(
        getattr(parsed, "rewrite_followup_experiment_todo_closure_floor", 0.35)
    )

    if (
        blockers is not None
        and blockers >= parsed.rewrite_followup_skip_blocker_threshold
    ):
        policy.update(
            {
                "skip": True,
                "mode": "skip_high_blockers",
                "reason": f"skip rewrite follow-up because blocker_count={blockers} >= {parsed.rewrite_followup_skip_blocker_threshold}",
            }
        )
        return policy

    if self_review_gate_ready is False and (
        (
            self_review_gate_score is not None
            and self_review_gate_score < self_review_gate_score_floor
        )
        or (
            self_review_high_value_coverage is not None
            and self_review_high_value_coverage
            < self_review_high_value_floor
        )
        or self_review_unresolved_critical > 0
    ):
        evidence_gap_preset = QUALITY_PRESETS["high"]
        policy.update(
            {
                "mode": "evidence_gap_repair",
                "preset_name": "high",
                "quality_threshold": max(
                    policy["quality_threshold"],
                    evidence_gap_preset["quality_threshold"],
                ),
                "rigor_threshold": max(
                    policy["rigor_threshold"],
                    evidence_gap_preset["rigor_threshold"],
                ),
                "max_rewrite_rounds": max(
                    int(policy["max_rewrite_rounds"]),
                    self_review_min_rounds,
                    int(evidence_gap_preset["max_rewrite_rounds"]),
                ),
                "reason": "self-review round gate signals unresolved high-value evidence gaps",
            }
        )
        return policy

    if (
        experiment_todo_p0_count >= max(1, experiment_todo_p0_floor)
        or experiment_todo_count >= max(1, experiment_todo_count_floor)
        or (
            experiment_todo_count > 0
            and experiment_todo_closure_rate is not None
            and experiment_todo_closure_rate < experiment_todo_closure_floor
        )
    ):
        evidence_gap_preset = QUALITY_PRESETS["high"]
        todo_reason = (
            f"closure_rate={experiment_todo_closure_rate:.2f} < {experiment_todo_closure_floor:.2f}"
            if experiment_todo_count > 0
            and experiment_todo_closure_rate is not None
            and experiment_todo_closure_rate < experiment_todo_closure_floor
            else f"p0={experiment_todo_p0_count}, total={experiment_todo_count}"
        )
        policy.update(
            {
                "mode": "evidence_gap_repair",
                "preset_name": "high",
                "quality_threshold": max(
                    policy["quality_threshold"],
                    evidence_gap_preset["quality_threshold"],
                ),
                "rigor_threshold": max(
                    policy["rigor_threshold"],
                    evidence_gap_preset["rigor_threshold"],
                ),
                "max_rewrite_rounds": max(
                    int(policy["max_rewrite_rounds"]),
                    experiment_todo_min_rounds,
                    int(evidence_gap_preset["max_rewrite_rounds"]),
                ),
                "reason": (
                    "experiment TODO backlog indicates unresolved high-priority "
                    f"evidence tasks ({todo_reason})"
                ),
            }
        )
        return policy

    if submission_status == "ready":
        ready_preset = QUALITY_PRESETS["balanced"]
        policy.update(
            {
                "mode": "final_polish",
                "preset_name": "balanced",
                "quality_threshold": max(
                    policy["quality_threshold"], ready_preset["quality_threshold"]
                ),
                "rigor_threshold": max(
                    policy["rigor_threshold"], ready_preset["rigor_threshold"]
                ),
                "max_rewrite_rounds": max(
                    1,
                    min(
                        int(policy["max_rewrite_rounds"]),
                        int(parsed.rewrite_followup_ready_max_rounds),
                    ),
                ),
                "reason": "paper is already ready; use a short low-risk polish pass",
            }
        )
        return policy

    if gate_passed is False or (
        blockers is not None
        and blockers >= parsed.rewrite_followup_blocker_reduction_threshold
    ):
        blocker_preset = QUALITY_PRESETS["high"]
        policy.update(
            {
                "mode": "blocker_reduction",
                "preset_name": "high",
                "quality_threshold": max(
                    policy["quality_threshold"], blocker_preset["quality_threshold"]
                ),
                "rigor_threshold": max(
                    policy["rigor_threshold"], blocker_preset["rigor_threshold"]
                ),
                "max_rewrite_rounds": max(
                    int(policy["max_rewrite_rounds"]),
                    int(blocker_preset["max_rewrite_rounds"]),
                ),
                "reason": "quality gate or blocker signal suggests a blocker-reduction rewrite pass",
            }
        )
        return policy

    if (
        submission_priority is not None
        and submission_priority
        >= parsed.rewrite_followup_publishable_priority_threshold
    ) or (
        rewrite_gain is not None
        and rewrite_gain >= parsed.rewrite_followup_publishable_gain_threshold
    ):
        publishable_preset = QUALITY_PRESETS["publishable"]
        policy.update(
            {
                "mode": "submission_push",
                "preset_name": "publishable",
                "quality_threshold": max(
                    policy["quality_threshold"], publishable_preset["quality_threshold"]
                ),
                "rigor_threshold": max(
                    policy["rigor_threshold"], publishable_preset["rigor_threshold"]
                ),
                "max_rewrite_rounds": max(
                    int(policy["max_rewrite_rounds"]),
                    int(publishable_preset["max_rewrite_rounds"]),
                ),
                "reason": "high submission priority or strong rewrite gain justifies a submission-push rewrite pass",
            }
        )
        return policy

    policy["reason"] = "base rewrite policy remains appropriate"
    return policy


def _run_rewrite_followup(
    parsed: argparse.Namespace,
    status: dict[str, Any],
    daemon_dir: Path,
    manager: ResearchManager,
) -> dict[str, Any]:
    heartbeat_log = daemon_dir / "heartbeat.log"
    history_path = daemon_dir / "cycle_history.jsonl"
    followup_log_dir = daemon_dir / "rewrite_followups"
    followup_log_dir.mkdir(parents=True, exist_ok=True)

    effective_top_k = int(
        (status.get("quality_governor") or {}).get(
            "rewrite_followup_top_k_effective", parsed.rewrite_followup_top_k
        )
    )
    if effective_top_k <= 0:
        _append_log(
            heartbeat_log,
            "rewrite follow-up skipped: quality governor reduced effective top-k to 0",
        )
        return {
            "count": 0,
            "items": [],
            "skipped_reason": "quality_governor_disabled_followup",
        }

    papers = manager.rewrite_board(
        top_n=effective_top_k,
        target_venue=parsed.rewrite_board_venue,
        min_priority=parsed.rewrite_board_min_priority,
        min_rewrite_gain=parsed.rewrite_board_min_gain,
        max_blockers=parsed.rewrite_board_max_blockers,
        require_gate=parsed.rewrite_board_require_gate,
        include_ready=parsed.rewrite_followup_include_ready
        or parsed.rewrite_board_include_ready,
    )
    if not papers:
        _append_log(heartbeat_log, "rewrite follow-up skipped: no qualifying papers")
        return {"count": 0, "items": []}

    rewrite_model = parsed.rewrite_followup_model or _passthrough_arg_value(
        parsed.generator_args, "--model-writeup", "glm-4-plus"
    )
    quality_model = parsed.rewrite_followup_quality_model or _passthrough_arg_value(
        parsed.generator_args, "--quality-model", rewrite_model
    )

    summary = {"count": 0, "items": [], "skipped_count": 0, "policy_counts": {}}
    for paper in papers:
        base_folder = Path(paper["path"]).parent
        policy = _derive_rewrite_followup_policy(parsed, paper)
        evidence_gap_actions = _derive_evidence_gap_actions(
            paper, policy_mode=str(policy.get("mode") or "")
        )
        autonomous_focus = _build_autonomous_followup_focus(
            paper, policy, evidence_gap_actions
        )
        entry = {
            "paper": paper.get("name"),
            "folder": paper.get("folder"),
            "path": str(base_folder),
            "started_at": _now_iso(),
            "status": "planned" if parsed.dry_run else "running",
            "target_venue": paper.get("target_venue"),
            "paper_type": paper.get("type"),
            "priority_before": paper.get("submission_priority_score"),
            "rewrite_policy": {
                "mode": policy.get("mode"),
                "reason": policy.get("reason"),
                "preset_name": policy.get("preset_name"),
                "quality_threshold": policy.get("quality_threshold"),
                "rigor_threshold": policy.get("rigor_threshold"),
                "max_rewrite_rounds": policy.get("max_rewrite_rounds"),
            },
            "self_review_gate_ready": paper.get("self_review_round_gate_ready"),
            "self_review_gate_score": paper.get("self_review_round_gate_score"),
            "self_review_unresolved_critical": paper.get(
                "self_review_unresolved_critical"
            ),
            "experiment_todo_count": paper.get("experiment_todo_count"),
            "experiment_todo_p0_count": paper.get("experiment_todo_p0_count"),
            "experiment_todo_closure_rate": paper.get("experiment_todo_closure_rate"),
            "experiment_todo_p0_closure_rate": paper.get(
                "experiment_todo_p0_closure_rate"
            ),
            "experiment_todo_top_action": paper.get("experiment_todo_top_action"),
            "evidence_gap_actions": evidence_gap_actions,
            "autonomous_followup_focus": autonomous_focus,
        }
        mode = str(policy.get("mode") or "standard")
        summary["policy_counts"][mode] = (
            int(summary["policy_counts"].get(mode) or 0) + 1
        )
        if policy.get("skip"):
            entry.update(
                {
                    "status": "skipped",
                    "finished_at": _now_iso(),
                    "skip_reason": policy.get("reason"),
                }
            )
            summary["skipped_count"] = int(summary.get("skipped_count") or 0) + 1
            summary["items"].append(entry)
            _append_log(
                heartbeat_log,
                f"rewrite follow-up skipped for {paper.get('folder')}: {policy.get('reason')}",
            )
            continue
        if parsed.dry_run:
            summary["items"].append(entry)
            continue

        _append_log(
            heartbeat_log,
            f"rewrite follow-up for {paper.get('folder')} using model={rewrite_model}; mode={policy.get('mode')}; preset={policy.get('preset_name')}; max_rounds={policy.get('max_rewrite_rounds')}",
        )
        try:
            result = run_high_quality_pass(
                base_folder,
                paper_type=paper.get("type") or "normal",
                rewrite_model=rewrite_model,
                quality_model=quality_model,
                target_venue=paper.get("target_venue"),
                quality_threshold=policy["quality_threshold"],
                rigor_threshold=policy["rigor_threshold"],
                max_rewrite_rounds=policy["max_rewrite_rounds"],
                auto_improvement_fallback=policy["auto_improvement_fallback"],
                autonomous_followup_focus=autonomous_focus,
                resume=False,
                logger=lambda msg: _append_log(
                    heartbeat_log, f"rewrite follow-up [{paper.get('folder')}]: {msg}"
                ),
            )
            after_priority = result.get("submission_priority_score")
            before_priority = entry.get("priority_before")
            priority_delta = (
                (after_priority - before_priority)
                if isinstance(after_priority, (int, float))
                and isinstance(before_priority, (int, float))
                else None
            )
            entry.update(
                {
                    "status": result.get("status", "unknown"),
                    "finished_at": _now_iso(),
                    "quality_score_after": result.get("quality_score_after"),
                    "submission_priority_score": after_priority,
                    "priority_delta": (
                        round(priority_delta, 3)
                        if isinstance(priority_delta, (int, float))
                        else None
                    ),
                    "quality_gate_passed": result.get("quality_gate_passed"),
                }
            )
            log_path = followup_log_dir / f"{paper.get('folder')}.json"
            _safe_write_json(log_path, entry | {"result": result})
            entry["log"] = str(log_path)
        except Exception as exc:
            entry.update(
                {"status": "failed", "finished_at": _now_iso(), "error": str(exc)}
            )
            _append_log(
                heartbeat_log,
                f"rewrite follow-up failed for {paper.get('folder')}: {exc}",
            )
        summary["items"].append(entry)

    summary["count"] = len(summary["items"])
    _safe_write_json(daemon_dir / "latest_rewrite_followup.json", summary)
    _append_jsonl(
        history_path,
        {"type": "rewrite_followup", "finished_at": _now_iso(), "summary": summary},
    )
    return summary


def _compute_health_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    views = status.get("last_views") or {}
    followup = status.get("last_rewrite_followup") or {}
    followup_metrics = _summarize_followup_uplift(followup)

    submission_items = int(views.get("submission_board_items") or 0)
    rewrite_items = int(views.get("rewrite_board_items") or 0)
    shortlist_items = int(views.get("shortlist_items") or 0)
    last_returncode = status.get("last_returncode")
    success_count = int(status.get("success_count") or 0)
    failure_count = int(status.get("failure_count") or 0)
    low_uplift_cycles = int(status.get("consecutive_low_uplift_cycles") or 0)
    empty_rewrite_cycles = int(status.get("consecutive_empty_rewrite_cycles") or 0)
    strong_submission_cycles = int(
        status.get("consecutive_strong_submission_cycles") or 0
    )
    phase = status.get("guardrail_phase")

    score = 65.0
    reasons = []
    if last_returncode == 0:
        score += 10
        reasons.append("last cycle finished successfully")
    elif last_returncode is not None:
        score -= 18
        reasons.append("last cycle failed")

    score += min(12.0, submission_items * 2.0)
    if submission_items:
        reasons.append(f"submission board has {submission_items} item(s)")
    if rewrite_items:
        score += min(8.0, rewrite_items * 1.5)
        reasons.append(f"rewrite board has {rewrite_items} item(s)")
    if shortlist_items:
        score += min(5.0, shortlist_items * 1.0)

    avg_delta = followup_metrics.get("avg_priority_delta", 0.0)
    if avg_delta >= 1.0:
        score += 10
        reasons.append("rewrite follow-up is producing strong priority uplift")
    elif avg_delta >= 0.25:
        score += 4
        reasons.append("rewrite follow-up is still improving drafts")
    elif followup_metrics.get("count", 0) > 0:
        score -= 8
        reasons.append("rewrite follow-up uplift is currently weak")

    if low_uplift_cycles:
        score -= min(12.0, low_uplift_cycles * 4.0)
    if empty_rewrite_cycles:
        score -= min(10.0, empty_rewrite_cycles * 3.0)
    if failure_count > success_count and failure_count > 0:
        score -= 8
        reasons.append("failures currently outnumber successes")
    if strong_submission_cycles:
        score += min(6.0, strong_submission_cycles * 2.0)

    if phase == "cold_start":
        reasons.append("system is still in cold-start exploration")
    elif phase == "hot_polish":
        reasons.append("system is in hot-polish mode with existing strong drafts")

    score = round(max(0.0, min(100.0, score)), 2)
    if score >= 80:
        state = "healthy"
        recommendation = "Keep the daemon running as configured; current throughput and uplift look healthy."
    elif score >= 60:
        state = "attention"
        recommendation = "Monitor the next few cycles; the system is productive but some signals are softening."
    elif score >= 40:
        state = "stalled"
        recommendation = "Bias back toward idea generation or loosen filters; rewrite uplift is no longer strong enough."
    else:
        state = "risk"
        recommendation = "Investigate environment, model/API reliability, or overly strict bars before continuing long unattended runs."

    return {
        "score": score,
        "state": state,
        "reasons": reasons[:6],
        "recommendation": recommendation,
        "followup_metrics": followup_metrics,
    }


def _suggest_source_action(
    source: dict[str, Any],
    state: dict[str, Any],
    availability_state: str,
    reason: str | None,
) -> str:
    if availability_state == "ready":
        if int(state.get("cycles_today", 0) or 0) == 0:
            return "High priority and unused today; schedule soon."
        if int(state.get("successes_today", 0) or 0) > 0:
            return "Already yielding results today; keep it active if capacity allows."
        return "Eligible now; keep rotating through it."
    if availability_state == "cooldown":
        return f"Pause temporarily until {reason or 'cooldown clears'}."
    if availability_state == "quota_exhausted":
        return "Daily cycle quota reached; resume next day or raise the quota."
    if availability_state == "success_budget_reached":
        return "Success budget reached; deprioritize today and spend budget elsewhere."
    if availability_state == "daypart_mismatch":
        return "Keep queued, but only activate it in its preferred daypart."
    return "Investigate this source or lower its priority until conditions improve."


def _build_source_quality_feedback(
    manager: ResearchManager, parsed: argparse.Namespace
) -> dict[str, dict[str, Any]]:
    if not parsed.auto_source_quality_feedback:
        return {}
    papers = manager.list_papers(sort_by="quality")
    aggregates: dict[str, dict[str, Any]] = {}
    for paper in papers:
        source_name = paper.get("source_name") or paper.get("source_key")
        if not source_name:
            continue
        entry = aggregates.setdefault(
            str(source_name),
            {
                "source_name": source_name,
                "count": 0,
                "ready_count": 0,
                "gate_pass_count": 0,
                "priority_scores": [],
                "rewrite_gains": [],
                "best_priority": None,
                "venue_counts": {},
                "paper_type_counts": {},
                "claim_support_scores": [],
                "claim_alignment_scores": [],
                "numeric_coverage_scores": [],
                "evidence_density_scores": [],
                "unsupported_claims_counts": [],
                "self_review_gate_seen_count": 0,
                "self_review_gate_ready_count": 0,
                "self_review_gate_scores": [],
                "self_review_unresolved_critical_counts": [],
                "self_review_high_value_coverage_scores": [],
                "experiment_todo_counts": [],
                "experiment_todo_p0_counts": [],
                "experiment_todo_closure_rates": [],
            },
        )
        entry["count"] += 1
        if paper.get("submission_status") == "ready":
            entry["ready_count"] += 1
        if paper.get("quality_gate_passed") is True:
            entry["gate_pass_count"] += 1
        venue = paper.get("target_venue")
        if venue:
            entry["venue_counts"][venue] = (
                int(entry["venue_counts"].get(venue) or 0) + 1
            )
        paper_type = paper.get("type")
        if paper_type:
            entry["paper_type_counts"][paper_type] = (
                int(entry["paper_type_counts"].get(paper_type) or 0) + 1
            )
        if isinstance(paper.get("submission_priority_score"), (int, float)):
            priority = float(paper.get("submission_priority_score"))
            entry["priority_scores"].append(priority)
            entry["best_priority"] = max(
                priority,
                (
                    entry.get("best_priority")
                    if isinstance(entry.get("best_priority"), (int, float))
                    else priority
                ),
            )
        if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
            entry["rewrite_gains"].append(
                float(paper.get("rewrite_priority_gain_total"))
            )
        if isinstance(paper.get("claim_support_score"), (int, float)):
            entry["claim_support_scores"].append(
                float(paper.get("claim_support_score"))
            )
        if isinstance(paper.get("claim_alignment_score"), (int, float)):
            entry["claim_alignment_scores"].append(
                float(paper.get("claim_alignment_score"))
            )
        if isinstance(paper.get("numeric_coverage_score"), (int, float)):
            entry["numeric_coverage_scores"].append(
                float(paper.get("numeric_coverage_score"))
            )
        if isinstance(paper.get("evidence_density_score"), (int, float)):
            entry["evidence_density_scores"].append(
                float(paper.get("evidence_density_score"))
            )
        if isinstance(paper.get("unsupported_claims_count"), int):
            entry["unsupported_claims_counts"].append(
                int(paper.get("unsupported_claims_count"))
            )
        if isinstance(paper.get("self_review_round_gate_ready"), bool):
            entry["self_review_gate_seen_count"] = (
                int(entry.get("self_review_gate_seen_count") or 0) + 1
            )
            if paper.get("self_review_round_gate_ready") is True:
                entry["self_review_gate_ready_count"] = (
                    int(entry.get("self_review_gate_ready_count") or 0) + 1
                )
        if isinstance(paper.get("self_review_round_gate_score"), (int, float)):
            entry["self_review_gate_scores"].append(
                float(paper.get("self_review_round_gate_score"))
            )
        if isinstance(paper.get("self_review_unresolved_critical"), int):
            entry["self_review_unresolved_critical_counts"].append(
                int(paper.get("self_review_unresolved_critical"))
            )
        if isinstance(paper.get("self_review_high_value_coverage"), (int, float)):
            entry["self_review_high_value_coverage_scores"].append(
                float(paper.get("self_review_high_value_coverage"))
            )
        if isinstance(paper.get("experiment_todo_count"), int):
            entry["experiment_todo_counts"].append(
                int(paper.get("experiment_todo_count"))
            )
        if isinstance(paper.get("experiment_todo_p0_count"), int):
            entry["experiment_todo_p0_counts"].append(
                int(paper.get("experiment_todo_p0_count"))
            )
        if isinstance(paper.get("experiment_todo_closure_rate"), (int, float)):
            entry["experiment_todo_closure_rates"].append(
                float(paper.get("experiment_todo_closure_rate"))
            )

    feedback: dict[str, dict[str, Any]] = {}
    for source_name, entry in aggregates.items():
        count = int(entry.get("count") or 0)
        if count < parsed.source_quality_feedback_min_papers:
            continue
        priority_scores = entry.get("priority_scores") or []
        rewrite_gains = entry.get("rewrite_gains") or []
        claim_support_scores = entry.get("claim_support_scores") or []
        claim_alignment_scores = entry.get("claim_alignment_scores") or []
        numeric_coverage_scores = entry.get("numeric_coverage_scores") or []
        evidence_density_scores = entry.get("evidence_density_scores") or []
        unsupported_claims_counts = entry.get("unsupported_claims_counts") or []
        self_review_gate_scores = entry.get("self_review_gate_scores") or []
        self_review_unresolved_critical_counts = (
            entry.get("self_review_unresolved_critical_counts") or []
        )
        self_review_high_value_coverage_scores = (
            entry.get("self_review_high_value_coverage_scores") or []
        )
        experiment_todo_counts = entry.get("experiment_todo_counts") or []
        experiment_todo_p0_counts = entry.get("experiment_todo_p0_counts") or []
        experiment_todo_closure_rates = entry.get("experiment_todo_closure_rates") or []
        self_review_gate_seen_count = int(entry.get("self_review_gate_seen_count") or 0)
        self_review_gate_ready_count = int(
            entry.get("self_review_gate_ready_count") or 0
        )
        avg_priority = (
            round(sum(priority_scores) / len(priority_scores), 3)
            if priority_scores
            else None
        )
        avg_rewrite_gain = (
            round(sum(rewrite_gains) / len(rewrite_gains), 3) if rewrite_gains else None
        )
        gate_pass_rate = round(entry.get("gate_pass_count", 0) / max(1, count), 3)
        ready_rate = round(entry.get("ready_count", 0) / max(1, count), 3)
        avg_claim_support = (
            round(sum(claim_support_scores) / len(claim_support_scores), 3)
            if claim_support_scores
            else None
        )
        avg_claim_alignment = (
            round(sum(claim_alignment_scores) / len(claim_alignment_scores), 3)
            if claim_alignment_scores
            else None
        )
        avg_numeric_coverage = (
            round(sum(numeric_coverage_scores) / len(numeric_coverage_scores), 3)
            if numeric_coverage_scores
            else None
        )
        avg_unsupported_claims = (
            round(sum(unsupported_claims_counts) / len(unsupported_claims_counts), 3)
            if unsupported_claims_counts
            else None
        )
        avg_evidence_density = (
            round(sum(evidence_density_scores) / len(evidence_density_scores), 3)
            if evidence_density_scores
            else None
        )
        avg_self_review_gate_score = (
            round(sum(self_review_gate_scores) / len(self_review_gate_scores), 3)
            if self_review_gate_scores
            else None
        )
        avg_self_review_unresolved_critical = (
            round(
                sum(self_review_unresolved_critical_counts)
                / len(self_review_unresolved_critical_counts),
                3,
            )
            if self_review_unresolved_critical_counts
            else None
        )
        avg_self_review_high_value_coverage = (
            round(
                sum(self_review_high_value_coverage_scores)
                / len(self_review_high_value_coverage_scores),
                3,
            )
            if self_review_high_value_coverage_scores
            else None
        )
        avg_experiment_todo = (
            round(sum(experiment_todo_counts) / len(experiment_todo_counts), 3)
            if experiment_todo_counts
            else None
        )
        avg_experiment_todo_p0 = (
            round(sum(experiment_todo_p0_counts) / len(experiment_todo_p0_counts), 3)
            if experiment_todo_p0_counts
            else None
        )
        experiment_todo_pressure_rate = (
            round(
                sum(count > 0 for count in experiment_todo_counts)
                / len(experiment_todo_counts),
                3,
            )
            if experiment_todo_counts
            else None
        )
        avg_experiment_todo_closure_rate = (
            round(
                sum(experiment_todo_closure_rates) / len(experiment_todo_closure_rates),
                3,
            )
            if experiment_todo_closure_rates
            else None
        )
        self_review_gate_ready_rate = (
            round(self_review_gate_ready_count / max(1, self_review_gate_seen_count), 3)
            if self_review_gate_seen_count > 0
            else None
        )
        bonus = 0.0
        if avg_priority is not None:
            if avg_priority >= 85:
                bonus += 1.5
            elif avg_priority >= 75:
                bonus += 0.75
            elif avg_priority < 60 and count >= 2:
                bonus -= 1.0
        if gate_pass_rate >= 0.5:
            bonus += 1.0
        elif gate_pass_rate == 0 and count >= 2:
            bonus -= 1.0
        if self_review_gate_ready_rate is not None:
            if self_review_gate_ready_rate >= 0.6:
                bonus += 0.75
            elif self_review_gate_ready_rate <= 0.3 and count >= 2:
                bonus -= 0.75
        if (
            avg_self_review_unresolved_critical is not None
            and avg_self_review_unresolved_critical > 1.0
            and count >= 2
        ):
            bonus -= 0.5
        if (
            avg_experiment_todo_p0 is not None
            and avg_experiment_todo_p0 > 0.5
            and count >= 2
        ):
            bonus -= 0.5
        if (
            avg_experiment_todo_closure_rate is not None
            and avg_experiment_todo_closure_rate < 0.35
            and count >= 2
        ):
            bonus -= 0.5
        if ready_rate > 0:
            bonus += 1.5
        if avg_rewrite_gain is not None and avg_rewrite_gain >= 1.0:
            bonus += 0.5
        bonus = round(
            max(
                -float(parsed.source_quality_feedback_max_penalty),
                min(float(parsed.source_quality_feedback_max_boost), bonus),
            ),
            3,
        )
        venue_counts = dict(entry.get("venue_counts") or {})
        paper_type_counts = dict(entry.get("paper_type_counts") or {})
        dominant_venue = None
        dominant_venue_rate = None
        if venue_counts:
            dominant_venue, dominant_venue_count = max(
                venue_counts.items(), key=lambda item: (item[1], item[0])
            )
            dominant_venue_rate = round(dominant_venue_count / max(1, count), 3)
        dominant_paper_type = None
        dominant_paper_type_rate = None
        if paper_type_counts:
            dominant_paper_type, dominant_paper_type_count = max(
                paper_type_counts.items(), key=lambda item: (item[1], item[0])
            )
            dominant_paper_type_rate = round(
                dominant_paper_type_count / max(1, count), 3
            )
        feedback[str(source_name)] = {
            "source_name": source_name,
            "count": count,
            "ready_count": entry.get("ready_count"),
            "gate_pass_count": entry.get("gate_pass_count"),
            "gate_pass_rate": gate_pass_rate,
            "ready_rate": ready_rate,
            "avg_priority": avg_priority,
            "avg_rewrite_gain": avg_rewrite_gain,
            "avg_claim_support": avg_claim_support,
            "avg_claim_alignment": avg_claim_alignment,
            "avg_numeric_coverage": avg_numeric_coverage,
            "avg_evidence_density": avg_evidence_density,
            "avg_unsupported_claims": avg_unsupported_claims,
            "avg_self_review_gate_score": avg_self_review_gate_score,
            "avg_self_review_unresolved_critical": avg_self_review_unresolved_critical,
            "avg_self_review_high_value_coverage": avg_self_review_high_value_coverage,
            "self_review_gate_ready_rate": self_review_gate_ready_rate,
            "avg_experiment_todo": avg_experiment_todo,
            "avg_experiment_todo_p0": avg_experiment_todo_p0,
            "avg_experiment_todo_closure_rate": avg_experiment_todo_closure_rate,
            "experiment_todo_pressure_rate": experiment_todo_pressure_rate,
            "best_priority": entry.get("best_priority"),
            "priority_bonus": bonus,
            "dominant_venue": dominant_venue,
            "dominant_venue_rate": dominant_venue_rate,
            "dominant_paper_type": dominant_paper_type,
            "dominant_paper_type_rate": dominant_paper_type_rate,
        }
    return feedback


def _classify_source_runtime(
    source: dict[str, Any],
    state: dict[str, Any],
    parsed: argparse.Namespace,
    status: dict[str, Any],
) -> dict[str, Any]:
    cycle_number = int(status.get("cycle", 0) or 0) + 1
    daypart = status.get("current_daypart") or _current_daypart(parsed)
    eligible, reason = _source_is_eligible(source, state, cycle_number, daypart)
    feedback = (
        (status.get("source_quality_feedback") or {}).get(str(source.get("name")))
        or (status.get("source_quality_feedback") or {}).get(_source_key(source))
        or {}
    )

    availability_state = "ready"
    if not eligible:
        lowered = (reason or "").lower()
        if "cooldown" in lowered:
            availability_state = "cooldown"
        elif "quota" in lowered:
            availability_state = "quota_exhausted"
        elif "budget" in lowered:
            availability_state = "success_budget_reached"
        elif "daypart" in lowered:
            availability_state = "daypart_mismatch"
        else:
            availability_state = "blocked"

    score = 75.0 + min(15.0, float(source.get("priority", 0)))
    if availability_state == "ready":
        score += 10
    elif availability_state == "daypart_mismatch":
        score -= 5
    elif availability_state == "cooldown":
        score -= 20
    else:
        score -= 30
    if int(state.get("successes_today", 0) or 0) > 0:
        score += 5
    if int(state.get("cycles_today", 0) or 0) == 0:
        score += 3

    active_pipeline_strategy = status.get("active_pipeline_contract_strategy") or {}
    desired_policy = str(
        active_pipeline_strategy.get("selected_execution_policy")
        or (status.get("pipeline_contract_summary") or {}).get("dominant_execution_policy")
        or ""
    ).strip()
    source_plan = build_source_planning_profile(
        source,
        daypart=daypart,
        desired_execution_policy=desired_policy or None,
    )
    target_venue = source.get(f"{daypart}_target_venue") or source.get("target_venue")
    paper_types = source.get(f"{daypart}_paper_types") or source.get("paper_types")
    submission_mode = source.get(f"{daypart}_submission_mode")
    if submission_mode is None:
        submission_mode = source.get("submission_mode")
    breakthrough_mode = source.get(f"{daypart}_breakthrough_mode")
    if breakthrough_mode is None:
        breakthrough_mode = source.get("breakthrough_mode")
    workflow_alignment_score = 0
    workflow_alignment_reason = "No active execution-policy preference."
    if desired_policy and desired_policy == source_plan.get("resolved_workflow_mode"):
        workflow_alignment_score += 2
    if desired_policy and desired_policy in (
        source_plan.get("compatible_workflow_modes") or []
    ):
        workflow_alignment_score += 1
    if desired_policy == "program_driven":
        if submission_mode:
            workflow_alignment_score += 2
        if target_venue in {"nature", "journal", "neurips", "iclr", "cvpr"}:
            workflow_alignment_score += 1
        workflow_alignment_reason = (
            "Program-driven repair prefers submission-oriented, budget-disciplined sources."
        )
    elif desired_policy == "agentic_tree":
        if breakthrough_mode:
            workflow_alignment_score += 2
        if not submission_mode:
            workflow_alignment_score += 1
        workflow_alignment_reason = (
            "Agentic rebuild prefers exploratory sources with breakthrough pressure."
        )
    elif desired_policy == "review_board":
        if submission_mode:
            workflow_alignment_score += 1
        if target_venue in {"nature", "journal"}:
            workflow_alignment_score += 1
        if isinstance(paper_types, list) and any(
            str(item) in {"journal", "extended"} for item in paper_types
        ):
            workflow_alignment_score += 1
        workflow_alignment_reason = (
            "Review-board hardening prefers reviewer-facing sources with stricter submission posture."
        )
    elif desired_policy == "multi_agent_board":
        if submission_mode:
            workflow_alignment_score += 2
        if target_venue in {"nature", "journal", "neurips", "iclr", "cvpr"}:
            workflow_alignment_score += 1
        if isinstance(paper_types, list) and any(
            str(item) in {"journal", "extended"} for item in paper_types
        ):
            workflow_alignment_score += 1
        workflow_alignment_reason = (
            "Multi-agent board prefers submission-facing sources that can support quality gate, hostile critic, and repair ownership."
        )
    elif desired_policy == "writing_studio":
        if isinstance(paper_types, list) and any(
            str(item) in {"journal", "extended"} for item in paper_types
        ):
            workflow_alignment_score += 2
        workflow_alignment_reason = (
            "Writing-studio preference favors sources that can turn evidence into polished long-form writeups."
        )
    elif desired_policy == "classic_pipeline":
        if source_plan.get("source_archetype") in {"template_first", "adaptive"}:
            workflow_alignment_score += 1
        workflow_alignment_reason = (
            "Classic pipeline prefers stable, repeatable sources with lower orchestration friction."
        )
    elif source_plan.get("workflow_reason"):
        workflow_alignment_reason = str(source_plan.get("workflow_reason"))
    score += workflow_alignment_score * 2
    score = round(max(0.0, min(100.0, score)), 2)

    return {
        "key": _source_key(source),
        "name": source.get("name"),
        "type": source.get("type"),
        "value": source.get("value"),
        "priority": source.get("priority"),
        "time_of_day_preference": source.get("time_of_day_preference"),
        "current_daypart": daypart,
        "availability_state": availability_state,
        "availability_reason": reason,
        "suggested_action": _suggest_source_action(
            source, state, availability_state, reason
        ),
        "health_score": score,
        "target_venue": source.get(f"{daypart}_target_venue")
        or source.get("target_venue"),
        "paper_types": source.get(f"{daypart}_paper_types")
        or source.get("paper_types"),
        "quality_feedback_priority_bonus": feedback.get("priority_bonus"),
        "quality_feedback_count": feedback.get("count"),
        "quality_feedback_avg_priority": feedback.get("avg_priority"),
        "active_quality_strategy_mode": (
            status.get("active_quality_strategy") or {}
        ).get("mode"),
        "active_evidence_strategy_mode": (
            status.get("active_evidence_strategy") or {}
        ).get("mode"),
        "preferred_execution_policy": desired_policy or None,
        "workflow_alignment_score": workflow_alignment_score,
        "workflow_alignment_reason": workflow_alignment_reason,
        "resolved_workflow_mode": source_plan.get("resolved_workflow_mode"),
        "compatible_workflow_modes": source_plan.get("compatible_workflow_modes"),
        "source_archetype": source_plan.get("source_archetype"),
        "source_archetype_label": source_plan.get("source_archetype_label"),
        "source_archetype_summary": source_plan.get("source_archetype_summary"),
        "batch_profile": source_plan.get("batch_profile"),
        "batch_profile_label": source_plan.get("batch_profile_label"),
        "batch_profile_summary": source_plan.get("batch_profile_summary"),
        "batch_goal": source_plan.get("batch_goal"),
        "planning_notes": source_plan.get("planning_notes"),
        "alignment_tags": source_plan.get("alignment_tags"),
        "archetype_inspirations": source_plan.get("archetype_inspirations"),
        "recommended_generator_preview": source_plan.get(
            "recommended_generator_preview"
        ),
        "num_ideas": (
            source.get(f"{daypart}_num_ideas")
            if source.get(f"{daypart}_num_ideas") is not None
            else source.get("num_ideas")
        ),
        "cooldown_until_cycle": state.get("cooldown_until_cycle"),
        "cycles_today": state.get("cycles_today"),
        "successes_today": state.get("successes_today"),
        "total_cycles": state.get("total_cycles"),
        "consecutive_failures": state.get("consecutive_failures"),
        "total_successes": state.get("total_successes"),
        "last_selected_at": state.get("last_selected_at"),
        "last_finished_at": state.get("last_finished_at"),
    }


def _build_source_runtime_rows(
    parsed: argparse.Namespace, status: dict[str, Any]
) -> list[dict[str, Any]]:
    queue = _load_source_queue(
        parsed, status.get("control"), status.get("source_quality_feedback")
    )
    runtime = _refresh_source_runtime_state(status, queue)
    rows = [
        _classify_source_runtime(
            source, runtime.get(_source_key(source), {}), parsed, status
        )
        for source in queue
    ]
    rows.sort(
        key=lambda item: (
            item.get("availability_state") != "ready",
            -(item.get("workflow_alignment_score") or 0),
            -(item.get("health_score") or 0),
            -(item.get("priority") or 0),
            item.get("name") or "",
        )
    )
    return rows


def _build_source_runtime_board_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Runtime Board", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('name')}",
                f"- Availability: {row.get('availability_state')} ({row.get('availability_reason') or 'ok'})",
                f"- Priority: {row.get('priority')}",
                f"- Current Daypart: {row.get('current_daypart')}",
                f"- Time Preference: {row.get('time_of_day_preference')}",
                f"- Target Venue: {row.get('target_venue')}",
                f"- Paper Types: {', '.join(row.get('paper_types') or [])}",
                f"- Source Archetype: {row.get('source_archetype_label')} ({row.get('source_archetype')})",
                f"- Batch Profile: {row.get('batch_profile_label')} ({row.get('batch_profile')})",
                f"- Resolved Workflow Mode: {row.get('resolved_workflow_mode')}",
                f"- Compatible Workflow Modes: {', '.join(row.get('compatible_workflow_modes') or [])}",
                f"- Preferred Policy: {row.get('preferred_execution_policy') or 'n/a'}",
                f"- Workflow Alignment: {row.get('workflow_alignment_score')} ({row.get('workflow_alignment_reason')})",
                f"- Batch Goal: {row.get('batch_goal')}",
                f"- Recommended Defaults: {', '.join(row.get('recommended_generator_preview') or []) or 'n/a'}",
                f"- Alignment Tags: {', '.join(row.get('alignment_tags') or []) or 'n/a'}",
                f"- Inspirations: {', '.join(row.get('archetype_inspirations') or []) or 'n/a'}",
                f"- Planning Notes: {row.get('planning_notes') or 'n/a'}",
                f"- Num Ideas: {row.get('num_ideas')}",
                f"- Cycles Today: {row.get('cycles_today')}",
                f"- Successes Today: {row.get('successes_today')}",
                f"- Cooldown Until Cycle: {row.get('cooldown_until_cycle')}",
                f"- Last Selected: {row.get('last_selected_at')}",
                f"- Last Finished: {row.get('last_finished_at')}",
                f"- Suggested Action: {row.get('suggested_action')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_source_health_board_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Health Board", ""]
    for row in sorted(rows, key=lambda item: item.get("health_score", 0), reverse=True):
        lines.append(
            f"- {row.get('name')} | health={row.get('health_score')} | align={row.get('workflow_alignment_score')} | state={row.get('availability_state')} | "
            f"workflow={row.get('resolved_workflow_mode')} | profile={row.get('batch_profile')} | "
            f"priority={row.get('priority')} | successes_today={row.get('successes_today')} | cycles_today={row.get('cycles_today')} | action={row.get('suggested_action')}"
        )
    return "\n".join(lines) + "\n"


def _build_source_batch_plan_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = [row for row in rows if row.get("availability_state") == "ready"]
    deferred = [row for row in rows if row.get("availability_state") != "ready"]
    plan_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(ready[:5]):
        if idx == 0:
            tier = "run-now"
            recommendation = (
                f"Use {row.get('name')} for the next batch because it best matches "
                f"{row.get('resolved_workflow_mode')} with health={row.get('health_score')}."
            )
        elif idx < 3:
            tier = "queue-next"
            recommendation = (
                f"Keep {row.get('name')} warm as the next candidate for "
                f"{row.get('batch_profile_label')} work."
            )
        else:
            tier = "hold-ready"
            recommendation = (
                f"Leave {row.get('name')} ready, but let higher-alignment sources run first."
            )
        plan_rows.append(
            {
                "tier": tier,
                "source": row.get("name"),
                "availability_state": row.get("availability_state"),
                "resolved_workflow_mode": row.get("resolved_workflow_mode"),
                "source_archetype": row.get("source_archetype"),
                "source_archetype_label": row.get("source_archetype_label"),
                "batch_profile": row.get("batch_profile"),
                "batch_profile_label": row.get("batch_profile_label"),
                "batch_goal": row.get("batch_goal"),
                "workflow_alignment_score": row.get("workflow_alignment_score"),
                "health_score": row.get("health_score"),
                "recommended_generator_preview": row.get(
                    "recommended_generator_preview"
                ),
                "planning_notes": row.get("planning_notes"),
                "alignment_tags": row.get("alignment_tags"),
                "archetype_inspirations": row.get("archetype_inspirations"),
                "recommendation": recommendation,
            }
        )

    for row in deferred[:3]:
        plan_rows.append(
            {
                "tier": "defer",
                "source": row.get("name"),
                "availability_state": row.get("availability_state"),
                "resolved_workflow_mode": row.get("resolved_workflow_mode"),
                "source_archetype": row.get("source_archetype"),
                "source_archetype_label": row.get("source_archetype_label"),
                "batch_profile": row.get("batch_profile"),
                "batch_profile_label": row.get("batch_profile_label"),
                "batch_goal": row.get("batch_goal"),
                "workflow_alignment_score": row.get("workflow_alignment_score"),
                "health_score": row.get("health_score"),
                "recommended_generator_preview": row.get(
                    "recommended_generator_preview"
                ),
                "planning_notes": row.get("planning_notes"),
                "alignment_tags": row.get("alignment_tags"),
                "archetype_inspirations": row.get("archetype_inspirations"),
                "recommendation": (
                    f"Defer {row.get('name')} until {row.get('availability_state')} clears."
                ),
            }
        )
    return plan_rows


def _build_source_batch_plan_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# Source Batch Plan", ""]
    if not rows:
        lines.extend(["- No batch plan items available yet.", ""])
        return "\n".join(lines)
    for row in rows:
        lines.extend(
            [
                f"## {row.get('source')} [{row.get('tier')}]",
                f"- Availability: {row.get('availability_state')}",
                f"- Workflow Mode: {row.get('resolved_workflow_mode')}",
                f"- Source Archetype: {row.get('source_archetype_label')} ({row.get('source_archetype')})",
                f"- Batch Profile: {row.get('batch_profile_label')} ({row.get('batch_profile')})",
                f"- Batch Goal: {row.get('batch_goal')}",
                f"- Workflow Alignment: {row.get('workflow_alignment_score')}",
                f"- Health Score: {row.get('health_score')}",
                f"- Recommended Defaults: {', '.join(row.get('recommended_generator_preview') or []) or 'n/a'}",
                f"- Alignment Tags: {', '.join(row.get('alignment_tags') or []) or 'n/a'}",
                f"- Inspirations: {', '.join(row.get('archetype_inspirations') or []) or 'n/a'}",
                f"- Planning Notes: {row.get('planning_notes') or 'n/a'}",
                f"- Recommendation: {row.get('recommendation')}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _hydrate_source_next_batch_advisory(
    advisory: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    hydrated = dict(advisory or {})
    runtime_by_name = {
        str(row.get("name")): row for row in rows if row.get("name") is not None
    }
    runtime_by_key = {
        str(row.get("key")): row for row in rows if row.get("key") is not None
    }
    slots = []
    for slot in hydrated.get("slots") or []:
        item = dict(slot)
        match = runtime_by_key.get(str(item.get("source_key") or "")) or runtime_by_name.get(
            str(item.get("source") or "")
        )
        if match:
            item["availability_state"] = match.get("availability_state")
            item["health_score"] = match.get("health_score")
            item["workflow_alignment_score"] = match.get("workflow_alignment_score")
            item["recommended_generator_preview"] = match.get(
                "recommended_generator_preview"
            )
            item["planning_notes"] = match.get("planning_notes")
            item["alignment_tags"] = match.get("alignment_tags")
        slots.append(item)
    hydrated["slots"] = slots
    return hydrated


def _build_source_next_batch_markdown(advisory: dict[str, Any]) -> str:
    lines = ["# Next Batch Source Mix", ""]
    summary = advisory.get("summary") or {}
    cadence = advisory.get("cadence") or {}
    lines.append(f"- Desired Policy: {advisory.get('desired_policy') or 'n/a'}")
    lines.append(f"- Source Count: {summary.get('source_count')}")
    lines.append(f"- Dominant Archetype: {summary.get('dominant_archetype')}")
    lines.append(f"- Dominant Workflow: {summary.get('dominant_workflow_mode')}")
    lines.append(
        f"- Cadence: {cadence.get('label') or 'n/a'} | {cadence.get('reason') or 'n/a'}"
    )
    lines.append("")
    if advisory.get("slots"):
        for slot in advisory.get("slots", []):
            lines.extend(
                [
                    f"## {slot.get('source')} [{slot.get('lane')}]",
                    f"- Share: {slot.get('share')}",
                    f"- Availability: {slot.get('availability_state') or 'n/a'}",
                    f"- Health Score: {slot.get('health_score')}",
                    f"- Workflow Alignment: {slot.get('workflow_alignment_score')}",
                    f"- Workflow Mode: {slot.get('source_workflow_mode')}",
                    f"- Source Archetype: {slot.get('source_archetype')}",
                    f"- Batch Profile: {slot.get('source_batch_profile')}",
                    f"- Target Venue: {slot.get('target_venue')}",
                    f"- Ready / Gate: {slot.get('ready_count')} / {slot.get('gate_pass_count')}",
                    f"- Avg Submission Priority: {slot.get('avg_submission_priority')}",
                    f"- Recommended Defaults: {', '.join(slot.get('recommended_generator_preview') or []) or 'n/a'}",
                    f"- Alignment Tags: {', '.join(slot.get('alignment_tags') or []) or 'n/a'}",
                    f"- Focus: {slot.get('focus')}",
                    f"- Rationale: {slot.get('rationale')}",
                    f"- Planning Notes: {slot.get('planning_notes') or 'n/a'}",
                    "",
                ]
            )
    else:
        lines.extend(["- No next-batch slots available yet.", ""])
    if advisory.get("recommendations"):
        lines.append("## Mix Recommendations")
        for item in advisory.get("recommendations", [])[:5]:
            lines.append(f"- [{item.get('tier')}] {item.get('recommendation')}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _write_source_boards(
    status: dict[str, Any], daemon_dir: Path, parsed: argparse.Namespace
) -> list[dict[str, Any]]:
    rows = _build_source_runtime_rows(parsed, status)
    _safe_write_json(
        daemon_dir / "latest_source_runtime_board.json",
        {"generated_at": _now_iso(), "rows": rows},
    )
    (daemon_dir / "latest_source_runtime_board.md").write_text(
        _build_source_runtime_board_markdown(rows), encoding="utf-8"
    )
    _safe_write_json(
        daemon_dir / "latest_source_health_board.json",
        {"generated_at": _now_iso(), "rows": rows},
    )
    (daemon_dir / "latest_source_health_board.md").write_text(
        _build_source_health_board_markdown(rows), encoding="utf-8"
    )
    batch_plan_rows = _build_source_batch_plan_rows(rows)
    _safe_write_json(
        daemon_dir / "latest_source_batch_plan.json",
        {"generated_at": _now_iso(), "rows": batch_plan_rows},
    )
    (daemon_dir / "latest_source_batch_plan.md").write_text(
        _build_source_batch_plan_markdown(batch_plan_rows), encoding="utf-8"
    )
    return rows


def _build_cycle_summary(status: dict[str, Any]) -> dict[str, Any]:
    views = status.get("last_views") or {}
    followup = status.get("last_rewrite_followup") or {}
    followup_metrics = _summarize_followup_uplift(followup)
    health = _compute_health_snapshot(status)
    active_feedback = status.get("active_source_feedback_snapshot") or {}
    return {
        "generated_at": _now_iso(),
        "active_source": status.get("active_source"),
        "current_daypart": status.get("current_daypart"),
        "health": health,
        "guardrail_phase": status.get("guardrail_phase"),
        "guardrail_phase_reason": status.get("guardrail_phase_reason"),
        "guardrail_mode": status.get("guardrail_mode"),
        "guardrail_reason": status.get("guardrail_reason"),
        "success_count": status.get("success_count"),
        "failure_count": status.get("failure_count"),
        "last_returncode": status.get("last_returncode"),
        "last_cycle_finished_at": status.get("last_cycle_finished_at"),
        "submission_board_items": views.get("submission_board_items"),
        "rewrite_board_items": views.get("rewrite_board_items"),
        "shortlist_items": views.get("shortlist_items"),
        "followup_count": followup_metrics.get("count"),
        "followup_avg_priority_delta": followup_metrics.get("avg_priority_delta"),
        "followup_improved_count": followup_metrics.get("improved_count"),
        "active_source_feedback": active_feedback,
        "active_source_todo_closure_rate": active_feedback.get(
            "avg_experiment_todo_closure_rate"
        ),
        "active_source_todo_backlog": active_feedback.get("avg_experiment_todo"),
        "active_source_todo_p0_backlog": active_feedback.get("avg_experiment_todo_p0"),
        "last_views": views,
        "last_rewrite_followup": followup,
        "source_runtime_board": (
            str(Path(status.get("daemon_dir", ".")) / "latest_source_runtime_board.md")
            if status.get("daemon_dir")
            else None
        ),
        "source_health_board": (
            str(Path(status.get("daemon_dir", ".")) / "latest_source_health_board.md")
            if status.get("daemon_dir")
            else None
        ),
        "source_batch_plan": (
            str(Path(status.get("daemon_dir", ".")) / "latest_source_batch_plan.md")
            if status.get("daemon_dir")
            else None
        ),
        "source_next_batch": (
            str(Path(status.get("daemon_dir", ".")) / "latest_source_next_batch.md")
            if status.get("daemon_dir")
            else None
        ),
    }


def _build_cycle_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Daemon Cycle Summary",
        "",
        f"- Generated at: {summary.get('generated_at')}",
        f"- Health: {summary.get('health', {}).get('score')} ({summary.get('health', {}).get('state')})",
        f"- Guardrail phase: {summary.get('guardrail_phase')} ({summary.get('guardrail_phase_reason')})",
        f"- Guardrail mode: {summary.get('guardrail_mode')} ({summary.get('guardrail_reason')})",
        f"- Success count: {summary.get('success_count')}",
        f"- Failure count: {summary.get('failure_count')}",
        f"- Submission board items: {summary.get('submission_board_items')}",
        f"- Rewrite board items: {summary.get('rewrite_board_items')}",
        f"- Shortlist items: {summary.get('shortlist_items')}",
        f"- Follow-up count: {summary.get('followup_count')}",
        f"- Follow-up avg priority delta: {summary.get('followup_avg_priority_delta')}",
        f"- Follow-up improved count: {summary.get('followup_improved_count')}",
        f"- Active source TODO closure: {summary.get('active_source_todo_closure_rate')}",
        f"- Active source TODO backlog: {summary.get('active_source_todo_backlog')}",
        f"- Active source TODO P0 backlog: {summary.get('active_source_todo_p0_backlog')}",
        f"- Health recommendation: {summary.get('health', {}).get('recommendation')}",
        "",
        "## Health Signals",
    ]
    for item in summary.get("health", {}).get("reasons", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Latest Artifacts",
            f"- Submission board: {summary.get('last_views', {}).get('submission_board')}",
            f"- Rewrite board: {summary.get('last_views', {}).get('rewrite_board')}",
            f"- Shortlist: {summary.get('last_views', {}).get('shortlist')}",
            f"- Source runtime board: {summary.get('source_runtime_board')}",
            f"- Source health board: {summary.get('source_health_board')}",
            f"- Source batch plan: {summary.get('source_batch_plan')}",
            f"- Source next batch: {summary.get('source_next_batch')}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_autonomy_program(
    status: dict[str, Any], parsed: argparse.Namespace
) -> dict[str, Any]:
    active_source = status.get("active_source") or {}
    feedback_snapshot = _active_source_feedback(status)
    return {
        "generated_at": _now_iso(),
        "goal": "maximize submission-ready, evidence-backed high-quality papers with minimal human intervention",
        "primary_target_venue": parsed.shortlist_venue
        or parsed.rewrite_board_venue
        or "nature",
        "fixed_evaluation_harness": [
            "high_quality_result.json quality / rigor / claim-support scores",
            "submission board and shortlist growth",
            "rewrite follow-up uplift and dossier readiness",
            "claim risk and evidence density signals",
            "experiment TODO backlog volume and P0 closure pressure",
        ],
        "current_active_source": {
            "name": active_source.get("name"),
            "type": active_source.get("type"),
            "value": active_source.get("value"),
            "target_venue": active_source.get(
                f"{status.get('current_daypart')}_target_venue"
            )
            or active_source.get("target_venue"),
            "paper_types": active_source.get(
                f"{status.get('current_daypart')}_paper_types"
            )
            or active_source.get("paper_types"),
        },
        "automation_stack": {
            "source_autopilot": bool(parsed.auto_apply_source_plan),
            "rewrite_autopilot": bool(parsed.enable_rewrite_followup),
            "submission_autopilot": bool(parsed.auto_export_submission_dossier),
            "failure_guard": bool(parsed.auto_failure_guard),
            "source_quality_feedback": bool(parsed.auto_source_quality_feedback),
            "quality_strategy_feedback": bool(parsed.auto_quality_strategy_feedback),
            "quality_governor": bool(parsed.auto_quality_governor),
            "evidence_strategy_feedback": bool(parsed.auto_evidence_strategy_feedback),
        },
        "current_strategies": {
            "guardrail_phase": status.get("guardrail_phase"),
            "guardrail_mode": status.get("guardrail_mode"),
            "quality_strategy": status.get("active_quality_strategy") or {},
            "evidence_strategy": status.get("active_evidence_strategy") or {},
            "quality_governor": status.get("quality_governor") or {},
        },
        "source_feedback_snapshot": feedback_snapshot,
        "keep_criteria": [
            "submission board or shortlist grows",
            "rewrite follow-up improves priority",
            "health score improves while blockers remain controlled",
            "new dossier-ready artifacts appear for strong papers",
        ],
        "discard_criteria": [
            "cycle crashes or times out",
            "health weakens without stronger shortlist candidates",
            "rewrite uplift remains weak and evidence quality does not improve",
        ],
        "adjustable_levers": [
            "source selection and one-shot priority boosts",
            "target venue / paper type strategy",
            "num-ideas exploration width",
            "rewrite follow-up and dossier export intensity",
            "review depth and quality rewrite rounds",
        ],
    }


def _build_autonomy_program_markdown(program: dict[str, Any]) -> str:
    lines = [
        "# Autonomous Research Program",
        "",
        f"- Generated at: {program.get('generated_at')}",
        f"- Goal: {program.get('goal')}",
        f"- Primary target venue: {program.get('primary_target_venue')}",
        "",
        "## Fixed Evaluation Harness",
    ]
    for item in program.get("fixed_evaluation_harness", []):
        lines.append(f"- {item}")

    active_source = program.get("current_active_source") or {}
    lines.extend(["", "## Active Source"])
    lines.append(f"- Name: {active_source.get('name')}")
    lines.append(f"- Type: {active_source.get('type')}")
    lines.append(f"- Value: {active_source.get('value')}")
    lines.append(f"- Target venue: {active_source.get('target_venue')}")
    lines.append(
        f"- Paper types: {', '.join(active_source.get('paper_types') or []) or 'n/a'}"
    )

    lines.extend(["", "## Automation Stack"])
    for key, value in (program.get("automation_stack") or {}).items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Current Strategies"])
    strategies = program.get("current_strategies") or {}
    lines.append(f"- Guardrail phase: {strategies.get('guardrail_phase')}")
    lines.append(f"- Guardrail mode: {strategies.get('guardrail_mode')}")
    lines.append(
        f"- Quality strategy: {(strategies.get('quality_strategy') or {}).get('mode')} ({(strategies.get('quality_strategy') or {}).get('reason')})"
    )
    lines.append(
        f"- Evidence strategy: {(strategies.get('evidence_strategy') or {}).get('mode')} ({(strategies.get('evidence_strategy') or {}).get('reason')})"
    )
    lines.append(
        f"- Quality governor: {(strategies.get('quality_governor') or {}).get('mode')} ({(strategies.get('quality_governor') or {}).get('reason')})"
    )

    feedback = program.get("source_feedback_snapshot") or {}
    lines.extend(["", "## Source Feedback Snapshot"])
    if feedback:
        for key in [
            "count",
            "avg_priority",
            "gate_pass_rate",
            "ready_rate",
            "avg_claim_support",
            "avg_claim_alignment",
            "avg_numeric_coverage",
            "avg_evidence_density",
            "avg_unsupported_claims",
            "avg_experiment_todo",
            "avg_experiment_todo_p0",
            "avg_experiment_todo_closure_rate",
            "priority_bonus",
            "dominant_venue",
            "dominant_paper_type",
        ]:
            lines.append(f"- {key}: {feedback.get(key)}")
    else:
        lines.append("- No source feedback available yet.")

    lines.extend(["", "## Keep Criteria"])
    for item in program.get("keep_criteria", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Discard Criteria"])
    for item in program.get("discard_criteria", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Adjustable Levers"])
    for item in program.get("adjustable_levers", []):
        lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def _classify_experiment_outcome(
    status: dict[str, Any], previous_summary: dict[str, Any] | None = None
) -> dict[str, Any]:
    previous_summary = previous_summary or {}
    current_views = status.get("last_views") or {}
    previous_submission = int(previous_summary.get("submission_board_items") or 0)
    previous_shortlist = int(previous_summary.get("shortlist_items") or 0)
    current_submission = int(current_views.get("submission_board_items") or 0)
    current_shortlist = int(current_views.get("shortlist_items") or 0)
    current_health = float((_compute_health_snapshot(status) or {}).get("score") or 0.0)
    previous_health = float((previous_summary.get("health") or {}).get("score") or 0.0)
    followup_gain = float(
        (status.get("guardrail_followup_metrics") or {}).get("avg_priority_delta")
        or 0.0
    )
    if status.get("last_returncode") not in (0, None):
        return {
            "status": "crash",
            "reasons": ["cycle returned non-zero"],
            "submission_delta": current_submission - previous_submission,
            "shortlist_delta": current_shortlist - previous_shortlist,
            "health_delta": round(current_health - previous_health, 3),
            "followup_gain": followup_gain,
        }

    reasons = []
    if current_submission > previous_submission:
        reasons.append("submission board expanded")
    if current_shortlist > previous_shortlist:
        reasons.append("shortlist expanded")
    if followup_gain > 0:
        reasons.append("rewrite follow-up improved priority")
    if current_health > previous_health:
        reasons.append("health score improved")
    if reasons:
        status_label = "keep"
    else:
        status_label = "discard"
        reasons.append("no strong quality improvement detected")
    return {
        "status": status_label,
        "reasons": reasons,
        "submission_delta": current_submission - previous_submission,
        "shortlist_delta": current_shortlist - previous_shortlist,
        "health_delta": round(current_health - previous_health, 3),
        "followup_gain": followup_gain,
    }


def _append_autonomous_experiment_ledger(
    daemon_dir: Path,
    status: dict[str, Any],
    previous_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = _classify_experiment_outcome(status, previous_summary)
    active_source = status.get("active_source") or {}
    quality_strategy = status.get("active_quality_strategy") or {}
    evidence_strategy = status.get("active_evidence_strategy") or {}
    description = (
        f"phase={status.get('guardrail_phase')} mode={status.get('guardrail_mode')} "
        f"quality_strategy={quality_strategy.get('mode')} "
        f"evidence_strategy={evidence_strategy.get('mode')}"
    )

    ledger_path = daemon_dir / "autonomous_experiment_ledger.tsv"
    if not ledger_path.exists():
        ledger_path.write_text(
            "cycle\tsource\tstatus\tsubmission_delta\tshortlist_delta\thealth_delta\tfollowup_gain\tdescription\n",
            encoding="utf-8",
        )
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(
            "\t".join(
                [
                    str(status.get("cycle")),
                    str(
                        active_source.get("name")
                        or status.get("active_source_key")
                        or "unknown"
                    ),
                    str(outcome.get("status")),
                    str(outcome.get("submission_delta")),
                    str(outcome.get("shortlist_delta")),
                    str(outcome.get("health_delta")),
                    str(outcome.get("followup_gain")),
                    description,
                ]
            )
            + "\n"
        )

    payload = {
        "generated_at": _now_iso(),
        "cycle": status.get("cycle"),
        "source": active_source.get("name") or status.get("active_source_key"),
        "phase": status.get("guardrail_phase"),
        "mode": status.get("guardrail_mode"),
        "quality_strategy": quality_strategy,
        "evidence_strategy": evidence_strategy,
        "outcome": outcome,
        "description": description,
    }
    _append_jsonl(daemon_dir / "autonomous_experiment_ledger.jsonl", payload)
    return payload


def _write_autonomy_program(
    daemon_dir: Path, status: dict[str, Any], parsed: argparse.Namespace
) -> dict[str, Any]:
    payload = _build_autonomy_program(status, parsed)
    _safe_write_json(daemon_dir / "latest_autonomy_program.json", payload)
    (daemon_dir / "latest_autonomy_program.md").write_text(
        _build_autonomy_program_markdown(payload), encoding="utf-8"
    )
    return payload


def _flatten_submission_board(board: dict[str, list[dict]]) -> list[dict]:
    papers = []
    for venue, items in (board or {}).items():
        for item in items:
            item = dict(item)
            item.setdefault("target_venue", venue)
            papers.append(item)
    papers.sort(
        key=lambda item: (
            (
                item.get("submission_priority_score")
                if isinstance(item.get("submission_priority_score"), (int, float))
                else -1
            ),
            (
                item.get("rewrite_priority_gain_total")
                if isinstance(item.get("rewrite_priority_gain_total"), (int, float))
                else -999
            ),
        ),
        reverse=True,
    )
    return papers


def _command_preview_for_next_cycle(
    parsed: argparse.Namespace, status: dict[str, Any]
) -> str:
    batch_name = f"{parsed.daemon_name}_preview_<timestamp>"
    cmd = _build_generator_command(parsed, batch_name, status)
    return shlex.join(cmd)


def _manager_command_preview(parsed: argparse.Namespace, command: str) -> str:
    base = [
        parsed.python,
        "research_manager.py",
        "--research-dir",
        parsed.research_dir,
        command,
    ]
    return shlex.join(base)


def _stable_wrapper_command_preview(daemon_dir: Path, command: str, *args: Any) -> str:
    parts = [
        "bash",
        str(Path(__file__).resolve().parent / "run_stable_daemon.sh"),
        command,
    ]
    parts.extend(str(item) for item in args if item is not None)
    parts.extend(["--daemon-dir", str(daemon_dir)])
    return shlex.join(parts)


def _build_primary_action_queue(
    *,
    do_now_actions: list[dict[str, Any]] | None = None,
    recovery_command: str | None = None,
    recovery_reason: str | None = None,
    trend_action_command: str | None = None,
    trend_action_reason: str | None = None,
    recommended_commands: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(
        priority: int,
        category: str,
        label: str,
        command: str | None,
        reason: str | None,
        source: str | None = None,
    ) -> None:
        if not command:
            return
        command_text = str(command).strip()
        if not command_text or command_text in seen:
            return
        seen.add(command_text)
        queue.append(
            {
                "priority": priority,
                "category": category,
                "label": label,
                "source": source,
                "reason": reason or "",
                "command": command_text,
            }
        )

    for item in (do_now_actions or [])[:3]:
        add(
            10,
            "do_now",
            f"do-now:{item.get('source') or 'unknown'}",
            item.get("command"),
            item.get("recommendation"),
            item.get("source"),
        )

    add(20, "recovery", "handoff-recovery", recovery_command, recovery_reason)
    add(30, "trend", "trend-action", trend_action_command, trend_action_reason)

    ordered_keys = [
        ("source_plan_command", 40),
        ("source_mix_command", 42),
        ("source_next_batch_command", 43),
        ("source_summary_command", 45),
        ("submission_board_command", 50),
        ("rewrite_board_command", 55),
        ("shortlist_command", 60),
        ("next_cycle_command", 70),
    ]
    commands = recommended_commands or {}
    for key, priority in ordered_keys:
        value = commands.get(key)
        if not value:
            continue
        add(priority, "recommended", key, str(value), key.replace("_", " "))

    queue.sort(key=lambda item: (item.get("priority", 999), item.get("label") or ""))
    return queue


def _build_primary_action_queue_markdown(queue: list[dict[str, Any]]) -> str:
    lines = ["# Primary Action Queue", ""]
    if queue:
        for item in queue[:10]:
            source = f" | source={item.get('source')}" if item.get("source") else ""
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | category={item.get('category')}{source} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")
    return "\n".join(lines) + "\n"


def _build_operator_commands(
    manager: ResearchManager, parsed: argparse.Namespace, status: dict[str, Any]
) -> dict[str, str]:
    daemon_dir = Path(status.get("daemon_dir") or parsed.research_dir)
    commands = {
        "next_cycle_command": _command_preview_for_next_cycle(parsed, status),
        "pipeline_status_command": _manager_command_preview(parsed, "pipeline-status"),
        "fallback_board_command": _manager_command_preview(parsed, "fallback-board"),
        "source_board_command": _manager_command_preview(parsed, "source-board"),
        "source_mix_command": _manager_command_preview(parsed, "source-mix"),
        "source_next_batch_command": _manager_command_preview(
            parsed, "source-next-batch"
        ),
        "idea_board_command": _manager_command_preview(parsed, "idea-board"),
        "experiment_board_command": _manager_command_preview(
            parsed, "experiment-board"
        ),
        "figure_board_command": _manager_command_preview(parsed, "figure-board"),
        "evolution_board_command": _manager_command_preview(parsed, "evolution-board"),
        "submission_board_command": shlex.join(
            [
                parsed.python,
                "research_manager.py",
                "--research-dir",
                parsed.research_dir,
                "submission-board",
                "--top",
                str(parsed.submission_board_top),
                "--min-priority",
                str(parsed.submission_board_min_priority),
                "--max-blockers",
                str(parsed.submission_board_max_blockers),
                "--min-rewrite-gain",
                str(parsed.submission_board_min_rewrite_gain),
            ]
            + (["--require-gate"] if parsed.submission_board_require_gate else [])
        ),
        "rewrite_board_command": shlex.join(
            [
                parsed.python,
                "research_manager.py",
                "--research-dir",
                parsed.research_dir,
                "rewrite-board",
                "--top",
                str(parsed.rewrite_board_top),
                "--min-priority",
                str(parsed.rewrite_board_min_priority),
                "--min-rewrite-gain",
                str(parsed.rewrite_board_min_gain),
                "--max-blockers",
                str(parsed.rewrite_board_max_blockers),
            ]
            + (["--require-gate"] if parsed.rewrite_board_require_gate else [])
            + (["--include-ready"] if parsed.rewrite_board_include_ready else [])
            + (
                ["--venue", parsed.rewrite_board_venue]
                if parsed.rewrite_board_venue
                else []
            )
        ),
        "shortlist_command": shlex.join(
            [
                parsed.python,
                "research_manager.py",
                "--research-dir",
                parsed.research_dir,
                "shortlist",
                "--top",
                str(parsed.shortlist_top),
                "--min-priority",
                str(parsed.shortlist_min_priority),
                "--max-blockers",
                str(parsed.shortlist_max_blockers),
                "--min-rewrite-gain",
                str(parsed.shortlist_min_rewrite_gain),
            ]
            + (["--require-gate"] if parsed.shortlist_require_gate else [])
            + (["--require-ready"] if parsed.shortlist_require_ready else [])
            + (["--venue", parsed.shortlist_venue] if parsed.shortlist_venue else [])
        ),
        "source_summary_command": _stable_wrapper_command_preview(
            daemon_dir, "source-summary", "--lines", "10"
        ),
        "source_plan_command": _stable_wrapper_command_preview(
            daemon_dir, "source-plan"
        ),
    }
    if parsed.enable_rewrite_followup and parsed.rewrite_followup_top_k > 0:
        commands["rewrite_followup_mode"] = (
            f"enabled (top_k={parsed.rewrite_followup_top_k}, preset={parsed.rewrite_followup_preset}, "
            f"max_rounds={parsed.rewrite_followup_max_rounds or QUALITY_PRESETS.get(parsed.rewrite_followup_preset, QUALITY_PRESETS['publishable']).get('max_rewrite_rounds')})"
        )
    else:
        commands["rewrite_followup_mode"] = "disabled"
    return commands


def _build_pipeline_contract_summary(manager: ResearchManager) -> dict[str, Any]:
    pipeline_rows = manager.pipeline_status(top_n=30)
    if not pipeline_rows:
        return {
            "enabled": False,
            "project_count": 0,
            "blocked_project_count": 0,
            "failed_project_count": 0,
            "fallback_count": 0,
            "strict_fallback_count": 0,
            "fallback_heavy_project_count": 0,
            "top_blocked_projects": [],
            "artifact_blockers": {},
            "blocked_figure_count": 0,
            "failed_experiment_count": 0,
            "budget_exhausted_experiment_count": 0,
            "stage_blocked_project_count": 0,
            "stage_missing_project_count": 0,
            "stage_attention_project_count": 0,
            "avg_stage_overall_score": None,
            "top_stage_standard_risks": {},
            "review_issue_project_count": 0,
            "review_low_resolution_project_count": 0,
            "review_persistent_issue_count": 0,
            "avg_review_resolution_rate": None,
            "review_low_binding_project_count": 0,
            "avg_review_binding_coverage": None,
            "review_low_repair_ready_project_count": 0,
            "avg_review_repair_ready_coverage": None,
            "process_alignment_blocked_project_count": 0,
            "process_alignment_missing_project_count": 0,
            "avg_process_alignment_score": None,
            "top_process_alignment_risks": {},
            "blocked_self_evolution_project_count": 0,
            "self_evolution_attention_project_count": 0,
            "self_evolution_required_failure_count": 0,
            "avg_self_evolution_score": None,
            "top_self_evolution_risks": {},
            "execution_policy_counts": {},
            "budget_status_counts": {},
            "dominant_execution_policy": None,
        }

    blocked_project_count = 0
    failed_project_count = 0
    fallback_count = 0
    strict_fallback_count = 0
    fallback_heavy_project_count = 0
    artifact_blockers: dict[str, int] = {}
    execution_policy_counts: dict[str, int] = {}
    top_blocked_projects: list[dict[str, Any]] = []
    stage_blocked_project_count = 0
    stage_missing_project_count = 0
    stage_attention_project_count = 0
    stage_overall_scores: list[float] = []
    top_stage_standard_risks: dict[str, int] = {}
    review_issue_project_count = 0
    review_low_resolution_project_count = 0
    review_persistent_issue_count = 0
    review_resolution_rates: list[float] = []
    review_low_binding_project_count = 0
    review_binding_coverages: list[float] = []
    review_low_repair_ready_project_count = 0
    review_repair_ready_coverages: list[float] = []
    process_alignment_blocked_project_count = 0
    process_alignment_missing_project_count = 0
    process_alignment_scores: list[float] = []
    top_process_alignment_risks: dict[str, int] = {}
    blocked_self_evolution_project_count = 0
    self_evolution_attention_project_count = 0
    self_evolution_required_failure_count = 0
    self_evolution_scores: list[float] = []
    top_self_evolution_risks: dict[str, int] = {}

    for row in pipeline_rows:
        blocked = list(row.get("blocked_artifacts") or [])
        failed = list(row.get("failed_artifacts") or [])
        missing = list(row.get("missing_artifacts") or [])
        row_fallback_count = int(row.get("fallback_count") or 0)
        row_strict_fallback_count = int(row.get("strict_fallback_count") or 0)
        fallback_count += row_fallback_count
        strict_fallback_count += row_strict_fallback_count
        if row_fallback_count >= 2 or row_strict_fallback_count >= 1:
            fallback_heavy_project_count += 1
        execution_policy = str(row.get("execution_policy") or "unknown")
        execution_policy_counts[execution_policy] = (
            execution_policy_counts.get(execution_policy, 0) + 1
        )
        blocked_stage_count = int(row.get("blocked_stage_count") or 0)
        missing_stage_count = int(row.get("missing_stage_count") or 0)
        attention_stage_count = int(row.get("needs_attention_stage_count") or 0)
        if blocked_stage_count > 0:
            stage_blocked_project_count += 1
        if missing_stage_count > 0:
            stage_missing_project_count += 1
        if attention_stage_count > 0:
            stage_attention_project_count += 1
        if isinstance(row.get("stage_overall_score"), (int, float)):
            stage_overall_scores.append(float(row.get("stage_overall_score")))
        review_active_issue_count = int(row.get("review_active_issue_count") or 0)
        review_persistent_count = int(row.get("review_persistent_issue_count") or 0)
        review_resolution_rate = row.get("review_resolution_rate")
        review_binding_coverage = row.get("review_target_binding_coverage")
        review_repair_ready_coverage = row.get("review_repair_ready_coverage")
        if review_active_issue_count > 0 or review_persistent_count > 0:
            review_issue_project_count += 1
        review_persistent_issue_count += review_persistent_count
        if isinstance(review_resolution_rate, (int, float)):
            review_resolution_rates.append(float(review_resolution_rate))
            if review_active_issue_count > 0 and float(review_resolution_rate) < 0.35:
                review_low_resolution_project_count += 1
        if isinstance(review_binding_coverage, (int, float)):
            review_binding_coverages.append(float(review_binding_coverage))
            if review_active_issue_count > 0 and float(review_binding_coverage) < 0.5:
                review_low_binding_project_count += 1
        if isinstance(review_repair_ready_coverage, (int, float)):
            review_repair_ready_coverages.append(float(review_repair_ready_coverage))
            if review_active_issue_count > 0 and float(review_repair_ready_coverage) < 0.7:
                review_low_repair_ready_project_count += 1
        process_alignment_blocked = int(
            row.get("process_alignment_blocked_process_count") or 0
        )
        process_alignment_missing = int(
            row.get("process_alignment_missing_process_count") or 0
        )
        process_alignment_score = row.get("process_alignment_overall_score")
        if process_alignment_blocked > 0:
            process_alignment_blocked_project_count += 1
        if process_alignment_missing > 0:
            process_alignment_missing_project_count += 1
        if isinstance(process_alignment_score, (int, float)):
            process_alignment_scores.append(float(process_alignment_score))
        self_evolution_status = str(row.get("self_evolution_status") or "").strip()
        self_evolution_score = row.get("self_evolution_score")
        self_evolution_required_failures = int(
            row.get("self_evolution_required_failure_count") or 0
        )
        self_evolution_required_failure_count += self_evolution_required_failures
        if self_evolution_status == "blocked":
            blocked_self_evolution_project_count += 1
        if self_evolution_status == "needs_attention":
            self_evolution_attention_project_count += 1
        if isinstance(self_evolution_score, (int, float)):
            self_evolution_scores.append(float(self_evolution_score))
        for risk in row.get("top_standard_risks") or []:
            label = str(risk).strip()
            if label:
                top_stage_standard_risks[label] = (
                    top_stage_standard_risks.get(label, 0) + 1
                )
        for risk in row.get("process_alignment_top_risks") or []:
            label = str(risk).strip()
            if label:
                top_process_alignment_risks[label] = (
                    top_process_alignment_risks.get(label, 0) + 1
                )
        for risk in row.get("self_evolution_top_risks") or []:
            label = str(risk).strip()
            if label:
                top_self_evolution_risks[label] = (
                    top_self_evolution_risks.get(label, 0) + 1
                )
        if blocked:
            blocked_project_count += 1
        if failed:
            failed_project_count += 1
        for name in blocked + failed:
            artifact_blockers[name] = artifact_blockers.get(name, 0) + 1
        if blocked or failed:
            top_blocked_projects.append(
                {
                    "project": row.get("project"),
                    "blocked_artifacts": blocked,
                    "failed_artifacts": failed,
                    "missing_artifacts": missing[:4],
                    "fallback_count": row_fallback_count,
                    "strict_fallback_count": row_strict_fallback_count,
                    "stage_overall_score": row.get("stage_overall_score"),
                    "blocked_stage_count": blocked_stage_count,
                    "needs_attention_stage_count": attention_stage_count,
                    "missing_stage_count": missing_stage_count,
                    "review_active_issue_count": review_active_issue_count,
                    "review_persistent_issue_count": review_persistent_count,
                    "review_resolution_rate": review_resolution_rate,
                    "review_target_binding_coverage": review_binding_coverage,
                    "review_repair_ready_coverage": review_repair_ready_coverage,
                    "process_alignment_overall_score": process_alignment_score,
                    "process_alignment_blocked_process_count": process_alignment_blocked,
                    "process_alignment_missing_process_count": process_alignment_missing,
                    "top_process_alignment_risks": (
                        row.get("process_alignment_top_risks") or []
                    )[:3],
                    "self_evolution_status": self_evolution_status,
                    "self_evolution_score": self_evolution_score,
                    "self_evolution_required_failure_count": (
                        self_evolution_required_failures
                    ),
                    "top_self_evolution_risks": (
                        row.get("self_evolution_top_risks") or []
                    )[:3],
                    "top_standard_risks": (row.get("top_standard_risks") or [])[:3],
                }
            )

    figure_rows = manager.figure_board(top_n=50, include_blocked=True)
    blocked_figure_count = sum(
        1 for row in figure_rows if str(row.get("status") or "") != "ready"
    )
    experiment_rows = manager.experiment_board(top_n=50)
    failed_experiment_count = sum(
        1 for row in experiment_rows if str(row.get("status") or "") == "failed"
    )
    budget_status_counts: dict[str, int] = {}
    experiment_policy_counts: dict[str, int] = {}
    budget_exhausted_experiment_count = 0
    for row in experiment_rows:
        budget_status = str(row.get("budget_status") or "unknown")
        budget_status_counts[budget_status] = (
            budget_status_counts.get(budget_status, 0) + 1
        )
        if budget_status == "budget_exhausted":
            budget_exhausted_experiment_count += 1
        policy_name = str(
            row.get("policy_name")
            or row.get("workflow_mode")
            or "unknown"
        )
        experiment_policy_counts[policy_name] = (
            experiment_policy_counts.get(policy_name, 0) + 1
        )

    dominant_execution_policy = None
    dominant_counts = experiment_policy_counts or execution_policy_counts
    if dominant_counts:
        dominant_execution_policy = sorted(
            dominant_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    return {
        "enabled": True,
        "project_count": len(pipeline_rows),
        "blocked_project_count": blocked_project_count,
        "failed_project_count": failed_project_count,
        "fallback_count": fallback_count,
        "strict_fallback_count": strict_fallback_count,
        "fallback_heavy_project_count": fallback_heavy_project_count,
        "top_blocked_projects": top_blocked_projects[:5],
        "artifact_blockers": dict(
            sorted(artifact_blockers.items(), key=lambda item: (-item[1], item[0]))[:8]
        ),
        "blocked_figure_count": blocked_figure_count,
        "failed_experiment_count": failed_experiment_count,
        "budget_exhausted_experiment_count": budget_exhausted_experiment_count,
        "stage_blocked_project_count": stage_blocked_project_count,
        "stage_missing_project_count": stage_missing_project_count,
        "stage_attention_project_count": stage_attention_project_count,
        "avg_stage_overall_score": (
            round(sum(stage_overall_scores) / len(stage_overall_scores), 2)
            if stage_overall_scores
            else None
        ),
        "top_stage_standard_risks": dict(
            sorted(
                top_stage_standard_risks.items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ),
        "review_issue_project_count": review_issue_project_count,
        "review_low_resolution_project_count": review_low_resolution_project_count,
        "review_persistent_issue_count": review_persistent_issue_count,
        "avg_review_resolution_rate": (
            round(sum(review_resolution_rates) / len(review_resolution_rates), 3)
            if review_resolution_rates
            else None
        ),
        "review_low_binding_project_count": review_low_binding_project_count,
        "avg_review_binding_coverage": (
            round(sum(review_binding_coverages) / len(review_binding_coverages), 3)
            if review_binding_coverages
            else None
        ),
        "review_low_repair_ready_project_count": review_low_repair_ready_project_count,
        "avg_review_repair_ready_coverage": (
            round(
                sum(review_repair_ready_coverages) / len(review_repair_ready_coverages),
                3,
            )
            if review_repair_ready_coverages
            else None
        ),
        "process_alignment_blocked_project_count": process_alignment_blocked_project_count,
        "process_alignment_missing_project_count": process_alignment_missing_project_count,
        "avg_process_alignment_score": (
            round(sum(process_alignment_scores) / len(process_alignment_scores), 2)
            if process_alignment_scores
            else None
        ),
        "top_process_alignment_risks": dict(
            sorted(
                top_process_alignment_risks.items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ),
        "blocked_self_evolution_project_count": blocked_self_evolution_project_count,
        "self_evolution_attention_project_count": self_evolution_attention_project_count,
        "self_evolution_required_failure_count": self_evolution_required_failure_count,
        "avg_self_evolution_score": (
            round(sum(self_evolution_scores) / len(self_evolution_scores), 2)
            if self_evolution_scores
            else None
        ),
        "top_self_evolution_risks": dict(
            sorted(
                top_self_evolution_risks.items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        ),
        "execution_policy_counts": dict(
            sorted(execution_policy_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "budget_status_counts": dict(
            sorted(budget_status_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "dominant_execution_policy": dominant_execution_policy,
    }


def _summarize_recent_failure_reasons(
    daemon_dir: Path, max_entries: int = 30
) -> list[dict[str, Any]]:
    history = _load_recent_cycle_history(daemon_dir, max_entries=max_entries)
    counts: dict[str, int] = {}
    for item in history:
        returncode = item.get("returncode")
        if returncode in (0, None):
            continue
        if returncode == 124:
            reason = "cycle_timeout"
        else:
            reason = f"generator_returncode_{returncode}"
        counts[reason] = counts.get(reason, 0) + 1

    latest_followup = _safe_read_json(daemon_dir / "latest_rewrite_followup.json")
    for item in latest_followup.get("items", []):
        if item.get("status") == "failed":
            reason = item.get("error") or "rewrite_followup_failed"
            counts[reason] = counts.get(reason, 0) + 1

    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            counts.items(), key=lambda pair: pair[1], reverse=True
        )[:5]
    ]


def _summarize_rewrite_style_hotspots(
    rewrite_board: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    style_scores: dict[str, float] = {}
    for paper in rewrite_board[:10]:
        gain = paper.get("rewrite_priority_gain_total")
        weight = float(gain) if isinstance(gain, (int, float)) else 0.0
        for style in [
            paper.get("rewrite_top_frontmatter_style"),
            paper.get("rewrite_top_section_style"),
        ]:
            if not style:
                continue
            style_scores[style] = style_scores.get(style, 0.0) + max(weight, 0.1)
    return [
        {"style": style, "score": round(score, 3)}
        for style, score in sorted(
            style_scores.items(), key=lambda pair: pair[1], reverse=True
        )[:5]
    ]


def _summarize_control_state(control: dict[str, Any]) -> list[str]:
    items = []
    if not control:
        return ["no control overrides loaded"]
    if control.get("paused"):
        items.append("daemon is paused")
    if control.get("stop_after_cycle"):
        items.append("daemon will stop after the current cycle")
    if control.get("force_phase"):
        items.append(f"phase forced to {control.get('force_phase')}")
    if control.get("force_mode"):
        items.append(f"mode forced to {control.get('force_mode')}")
    if control.get("sleep_override_minutes") is not None:
        items.append(
            f"sleep override = {control.get('sleep_override_minutes')} minutes"
        )
    if control.get("disabled_sources"):
        items.append(
            f"disabled sources: {', '.join(control.get('disabled_sources')[:5])}"
        )
    if control.get("source_priority_overrides"):
        items.append(
            f"source priority overrides: {len(control.get('source_priority_overrides'))}"
        )
    if control.get("source_commands"):
        items.append(f"one-shot source commands: {len(control.get('source_commands'))}")
    if control.get("dashboard_refresh_seconds") is not None:
        items.append(
            f"dashboard refresh = {control.get('dashboard_refresh_seconds')} seconds"
        )
    if control.get("expires_after_cycles") is not None:
        items.append(
            f"global control expires after {control.get('expires_after_cycles')} cycle(s)"
        )
    if control.get("validation_errors"):
        items.append(
            f"control validation errors: {len(control.get('validation_errors'))}"
        )
    if not items:
        items.append("control file loaded with no active overrides")
    return items[:6]


def _select_do_now_actions(
    source_advisory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tier_rank = {"do-now": 0, "queue-next": 1, "watch": 2, "leave-alone": 3}
    ranked = sorted(
        source_advisory or [],
        key=lambda item: (
            tier_rank.get(item.get("tier"), 99),
            -(item.get("health_score") or -1),
            item.get("source") or "",
        ),
    )
    return ranked[:3]


def _apply_control_operation_to_payload(
    payload: dict[str, Any],
    *,
    operation: str,
    source: str,
    value: Any = None,
    expires_after_cycles: int | None = None,
) -> tuple[dict[str, Any], bool]:
    updated = dict(payload)
    updated["disabled_sources"] = list(updated.get("disabled_sources") or [])
    updated["source_priority_overrides"] = dict(
        updated.get("source_priority_overrides") or {}
    )
    updated["source_commands"] = {
        key: dict(command) if isinstance(command, dict) else command
        for key, command in (updated.get("source_commands") or {}).items()
    }
    changed = False

    if operation == "enable-source":
        filtered = [item for item in updated["disabled_sources"] if item != source]
        if filtered != updated["disabled_sources"]:
            updated["disabled_sources"] = filtered
            changed = True
    elif operation == "disable-source":
        if source not in updated["disabled_sources"]:
            updated["disabled_sources"].append(source)
            changed = True
    elif operation in {
        "source-force-next",
        "source-boost-next",
        "source-cooldown-once",
        "clear-source-command",
    }:
        commands = dict(updated.get("source_commands") or {})
        command = dict(commands.get(source) or {})
        if operation == "clear-source-command":
            if source in commands:
                commands.pop(source, None)
                changed = True
        else:
            if (
                operation == "source-force-next"
                and command.get("force_next_cycle") is not True
            ):
                command["force_next_cycle"] = True
                changed = True
            if operation == "source-boost-next":
                try:
                    boost_value = float(value)
                except (TypeError, ValueError):
                    boost_value = None
                if (
                    boost_value is not None
                    and command.get("priority_boost_next") != boost_value
                ):
                    command["priority_boost_next"] = boost_value
                    changed = True
            if operation == "source-cooldown-once":
                try:
                    cooldown_value = int(value)
                except (TypeError, ValueError):
                    cooldown_value = None
                if (
                    cooldown_value is not None
                    and command.get("cooldown_cycles_once") != cooldown_value
                ):
                    command["cooldown_cycles_once"] = cooldown_value
                    changed = True
            if (
                expires_after_cycles is not None
                and command.get("expires_after_cycles") != expires_after_cycles
            ):
                command["expires_after_cycles"] = expires_after_cycles
                changed = True
            if command:
                commands[source] = command
        updated["source_commands"] = commands
    else:
        return updated, False

    return updated, changed


def _build_source_advisory_rows(
    source_rows: list[dict[str, Any]], control: dict[str, Any], daemon_dir: Path
) -> list[dict[str, Any]]:
    disabled = set(control.get("disabled_sources") or [])
    priority_overrides = control.get("source_priority_overrides") or {}
    source_commands = control.get("source_commands") or {}
    ready = [row for row in source_rows if row.get("availability_state") == "ready"]
    daypart = [
        row
        for row in source_rows
        if row.get("availability_state") == "daypart_mismatch"
    ]
    blocked = [
        row
        for row in source_rows
        if row.get("availability_state")
        in {"blocked", "quota_exhausted", "success_budget_reached", "cooldown"}
    ]
    ready.sort(
        key=lambda row: (
            -(row.get("workflow_alignment_score") or 0),
            -(row.get("health_score") or 0),
            -(row.get("priority") or 0),
            row.get("name") or "",
        )
    )
    daypart.sort(
        key=lambda row: (
            -(row.get("workflow_alignment_score") or 0),
            -(row.get("priority") or 0),
            -(row.get("health_score") or 0),
            row.get("name") or "",
        )
    )
    blocked.sort(
        key=lambda row: (
            row.get("availability_state") != "cooldown",
            -(row.get("priority") or 0),
            row.get("name") or "",
        )
    )
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add_item(
        tier: str,
        row: dict[str, Any],
        recommendation: str,
        command: str | None = None,
        *,
        control_operation: str | None = None,
        control_value: Any = None,
    ) -> None:
        name = row.get("name") or row.get("key") or "unknown"
        marker = (tier, str(name))
        if marker in seen:
            return
        seen.add(marker)
        items.append(
            {
                "tier": tier,
                "source": name,
                "state": row.get("availability_state"),
                "health_score": row.get("health_score"),
                "recommendation": recommendation,
                "suggested_action": row.get("suggested_action"),
                "command": command or "",
                "control_operation": control_operation,
                "control_value": control_value,
            }
        )

    if ready:
        top = ready[0]
        top_name = top.get("name") or top.get("key")
        command = (
            ""
            if top_name in source_commands
            else _stable_wrapper_command_preview(
                daemon_dir, "source-force-next", top_name
            )
        )
        add_item(
            "do-now",
            top,
            (
                f"Prioritize {top_name} next; it is ready with health={top.get('health_score')} "
                f"and workflow alignment={top.get('workflow_alignment_score')}."
            ),
            command,
            control_operation="source-force-next" if command else None,
        )
        for row in ready[1:3]:
            name = row.get("name") or row.get("key")
            if name in disabled:
                add_item(
                    "do-now",
                    row,
                    f"Re-enable {name}; it is disabled but otherwise ready.",
                    _stable_wrapper_command_preview(daemon_dir, "enable-source", name),
                    control_operation="enable-source",
                )
            elif (
                name not in priority_overrides and (row.get("health_score") or 0) >= 90
            ):
                add_item(
                    "queue-next",
                    row,
                    f"Give {name} a one-off priority boost; it is strong but not explicitly boosted.",
                    _stable_wrapper_command_preview(
                        daemon_dir, "source-boost-next", name, 3
                    ),
                    control_operation="source-boost-next",
                    control_value=3,
                )
    if daypart:
        row = daypart[0]
        add_item(
            "leave-alone",
            row,
            f"Leave {row.get('name')} queued for its preferred {row.get('time_of_day_preference')} window.",
        )
    if blocked:
        row = blocked[0]
        add_item(
            "watch",
            row,
            f"Monitor {row.get('name')} because it is currently {row.get('availability_state')} ({row.get('availability_reason') or 'n/a'}).",
        )
    for name in sorted(disabled)[:2]:
        if any(item.get("source") == name for item in items):
            continue
        matching = next(
            (row for row in source_rows if (row.get("name") or row.get("key")) == name),
            {
                "name": name,
                "availability_state": "disabled",
                "health_score": None,
                "suggested_action": "Review whether this source should stay disabled.",
            },
        )
        add_item(
            "watch",
            matching,
            f"Review disabled source {name} before leaving it idle for too long.",
            _stable_wrapper_command_preview(daemon_dir, "enable-source", name),
            control_operation="enable-source",
        )
    for name, command in list(source_commands.items())[:2]:
        matching = next(
            (row for row in source_rows if (row.get("name") or row.get("key")) == name),
            {
                "name": name,
                "availability_state": "pending_command",
                "health_score": None,
                "suggested_action": "A one-shot source command is already queued.",
            },
        )
        add_item(
            "watch",
            matching,
            f"One-shot source command already queued for {name}: {command}",
            _stable_wrapper_command_preview(daemon_dir, "clear-source-command", name),
            control_operation="clear-source-command",
        )
    return items[:8]


def _build_operator_brief(
    manager: ResearchManager, parsed: argparse.Namespace, status: dict[str, Any]
) -> dict[str, Any]:
    submission_board = manager.submission_board(
        top_n_per_venue=parsed.submission_board_top,
        min_priority=parsed.submission_board_min_priority,
        max_blockers=parsed.submission_board_max_blockers,
        min_rewrite_gain=parsed.submission_board_min_rewrite_gain,
        require_gate=parsed.submission_board_require_gate,
    )
    rewrite_board = manager.rewrite_board(
        top_n=parsed.rewrite_board_top,
        target_venue=parsed.rewrite_board_venue,
        min_priority=parsed.rewrite_board_min_priority,
        min_rewrite_gain=parsed.rewrite_board_min_gain,
        max_blockers=parsed.rewrite_board_max_blockers,
        require_gate=parsed.rewrite_board_require_gate,
        include_ready=parsed.rewrite_board_include_ready,
    )
    shortlist = manager.shortlist_papers(
        target_venue=parsed.shortlist_venue,
        require_gate=parsed.shortlist_require_gate,
        require_ready=parsed.shortlist_require_ready,
        min_priority=parsed.shortlist_min_priority,
        max_blockers=parsed.shortlist_max_blockers,
        min_rewrite_gain=parsed.shortlist_min_rewrite_gain,
        top_n=parsed.shortlist_top,
    )
    top_submission = _flatten_submission_board(submission_board)[:3]
    top_rewrite = rewrite_board[:3]
    followup = status.get("last_rewrite_followup") or {}
    followup_items = [
        item
        for item in (followup.get("items") or [])
        if item.get("status") == "success"
    ][:3]

    source_rows = _build_source_runtime_rows(parsed, status)
    control = status.get("control") or {}
    source_advisory = _build_source_advisory_rows(
        source_rows, control, Path(status.get("daemon_dir") or parsed.research_dir)
    )
    control_summary = _summarize_control_state(control)
    recent_control_events = _load_recent_control_events(
        Path(status.get("daemon_dir") or "."), max_entries=5
    )
    failure_hotspots = _summarize_recent_failure_reasons(
        Path(status.get("daemon_dir") or ".")
    )
    rewrite_style_hotspots = _summarize_rewrite_style_hotspots(rewrite_board)
    pipeline_contracts = _build_pipeline_contract_summary(manager)
    desired_policy = str(
        (
            status.get("active_pipeline_contract_strategy") or {}
        ).get("selected_execution_policy")
        or pipeline_contracts.get("dominant_execution_policy")
        or ""
    ).strip() or None
    source_mix_advisory = manager.source_mix_advisory(
        desired_policy=desired_policy,
        top_n=max(20, int(parsed.shortlist_top or 10) * 2),
    )
    source_next_batch_advisory = _hydrate_source_next_batch_advisory(
        manager.source_next_batch_advisory(
            desired_policy=desired_policy,
            top_n=max(20, int(parsed.shortlist_top or 10) * 2),
            max_slots=3,
        ),
        source_rows,
    )
    health = _compute_health_snapshot(status)
    priorities = []
    actions = []
    if status.get("guardrail_phase") == "cold_start":
        priorities.append(
            "Keep exploring; the system is still in cold-start and needs more strong draft candidates."
        )
    elif status.get("guardrail_phase") == "hot_polish":
        priorities.append(
            "Strong drafts exist; prioritize polishing and risk reduction over raw generation volume."
        )
    else:
        priorities.append(
            "Maintain balanced generation and polishing while monitoring rewrite uplift."
        )

    if top_submission:
        actions.append(
            f"Top submission target: {top_submission[0].get('name')} (priority={top_submission[0].get('submission_priority_score')})."
        )
    if top_rewrite:
        actions.append(
            f"Top rewrite target: {top_rewrite[0].get('name')} → {top_rewrite[0].get('suggested_next_step')}"
        )
    if followup_items:
        best_followup = max(
            followup_items,
            key=lambda item: (
                item.get("priority_delta")
                if isinstance(item.get("priority_delta"), (int, float))
                else -999
            ),
        )
        actions.append(
            f"Best recent follow-up: {best_followup.get('paper')} (priority_delta={best_followup.get('priority_delta')})."
        )
    if not actions:
        actions.append(
            "No high-priority drafts surfaced yet; keep generating and monitoring the boards."
        )

    commands = _build_operator_commands(manager, parsed, status)

    blockers = []
    if (status.get("guardrail_followup_metrics") or {}).get(
        "avg_priority_delta", 0.0
    ) < parsed.guardrail_min_followup_gain:
        blockers.append(
            "Rewrite uplift is currently weak; more idea exploration may be needed."
        )
    if not top_submission:
        blockers.append(
            "No submission-board candidates currently meet the configured bar."
        )
    if not top_rewrite:
        blockers.append(
            "No rewrite-board candidates currently justify another polishing pass."
        )
    if pipeline_contracts.get("blocked_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('blocked_project_count')} pipeline roots still have blocked artifacts; inspect pipeline-status and figure-board."
        )
    if pipeline_contracts.get("stage_blocked_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('stage_blocked_project_count')} pipeline roots fail stage standards outright; run stage-standards and repair the blocked criteria before scaling throughput."
        )
    if pipeline_contracts.get("stage_missing_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('stage_missing_project_count')} pipeline roots are still missing stage-standard evidence; missing stages should be treated as incomplete research loops."
        )
    if pipeline_contracts.get("review_low_resolution_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('review_low_resolution_project_count')} pipeline roots show low reviewer-resolution progress; reviewer feedback needs issue-by-issue repair instead of broad polish."
        )
    if pipeline_contracts.get("review_low_binding_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('review_low_binding_project_count')} pipeline roots still have reviewer issues that are weakly bound to claims, figures, or sections; repair routing is still too fuzzy."
        )
    if pipeline_contracts.get("review_low_repair_ready_project_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('review_low_repair_ready_project_count')} pipeline roots still lack executable reviewer repair tasks or verification plans; self-optimization is underspecified."
        )
    if pipeline_contracts.get("review_persistent_issue_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('review_persistent_issue_count')} persistent reviewer issues are still open across active pipeline roots; focus on verified closure before new expansion."
        )
    if pipeline_contracts.get("failed_experiment_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('failed_experiment_count')} recent experiment registry entries are failed; experiment-board should be triaged."
        )
    if pipeline_contracts.get("budget_exhausted_experiment_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('budget_exhausted_experiment_count')} recent experiment registry entries exhausted budget; the next cycle should adapt workflow discipline."
        )
    if pipeline_contracts.get("strict_fallback_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('strict_fallback_count')} strict fallback events were recorded recently; inspect fallback-board and reduce hidden debt before scaling throughput."
        )
    if pipeline_contracts.get("blocked_figure_count", 0) > 0:
        blockers.append(
            f"{pipeline_contracts.get('blocked_figure_count')} recent figure specs are blocked; plotting and evidence packaging need attention."
        )
    blocked_projects = pipeline_contracts.get("top_blocked_projects") or []
    if blocked_projects:
        top_blocked = blocked_projects[0]
        actions.append(
            f"Top pipeline blocker: {top_blocked.get('project')} blocked={top_blocked.get('blocked_artifacts')} failed={top_blocked.get('failed_artifacts')}."
        )
    if pipeline_contracts.get("dominant_execution_policy"):
        actions.append(
            f"Dominant execution policy: {pipeline_contracts.get('dominant_execution_policy')}."
        )
    if pipeline_contracts.get("strict_fallback_count", 0) > 0:
        actions.append(
            f"Fallback debt pressure: strict={pipeline_contracts.get('strict_fallback_count')} total={pipeline_contracts.get('fallback_count')}."
        )
    if pipeline_contracts.get("avg_stage_overall_score") is not None:
        actions.append(
            f"Average stage-standard score across active pipeline roots: {pipeline_contracts.get('avg_stage_overall_score')}."
        )
    if pipeline_contracts.get("avg_review_resolution_rate") is not None:
        actions.append(
            f"Average reviewer-resolution rate across active pipeline roots: {pipeline_contracts.get('avg_review_resolution_rate')}."
        )
    if pipeline_contracts.get("avg_review_binding_coverage") is not None:
        actions.append(
            f"Average reviewer target-binding coverage across active pipeline roots: {pipeline_contracts.get('avg_review_binding_coverage')}."
        )
    if pipeline_contracts.get("avg_review_repair_ready_coverage") is not None:
        actions.append(
            f"Average reviewer repair-ready coverage across active pipeline roots: {pipeline_contracts.get('avg_review_repair_ready_coverage')}."
        )
    top_stage_risks = pipeline_contracts.get("top_stage_standard_risks") or {}
    if top_stage_risks:
        risk_label = next(iter(top_stage_risks.keys()))
        actions.append(
            f"Top recurring stage-standard risk: {risk_label} ({top_stage_risks.get(risk_label)} runs)."
        )
    source_mix_recommendations = source_mix_advisory.get("recommendations") or []
    if source_mix_recommendations:
        actions.append(
            f"Source mix recommendation: {source_mix_recommendations[0].get('recommendation')}"
        )
    next_batch_slots = source_next_batch_advisory.get("slots") or []
    if next_batch_slots:
        primary_slot = next_batch_slots[0]
        actions.append(
            f"Next batch primary lane: {primary_slot.get('source')} under {primary_slot.get('source_workflow_mode')} with share={primary_slot.get('share')}."
        )
    next_batch_cadence = source_next_batch_advisory.get("cadence") or {}
    if next_batch_cadence.get("label"):
        priorities.append(
            f"Next-batch cadence is {next_batch_cadence.get('label')}: {next_batch_cadence.get('reason')}"
        )
    mix_summary = source_mix_advisory.get("summary") or {}
    if mix_summary.get("dominant_archetype"):
        priorities.append(
            f"Current source mix is led by {mix_summary.get('dominant_archetype')} under {mix_summary.get('dominant_workflow_mode')}."
        )

    brief = {
        "generated_at": _now_iso(),
        "active_source": status.get("active_source"),
        "current_daypart": status.get("current_daypart"),
        "health": health,
        "guardrail_phase": status.get("guardrail_phase"),
        "guardrail_mode": status.get("guardrail_mode"),
        "guardrail_reason": status.get("guardrail_reason"),
        "success_count": status.get("success_count"),
        "failure_count": status.get("failure_count"),
        "top_submission_targets": [
            {
                "name": item.get("name"),
                "venue": item.get("target_venue"),
                "priority": item.get("submission_priority_score"),
                "rewrite_gain": item.get("rewrite_priority_gain_total"),
            }
            for item in top_submission
        ],
        "top_rewrite_targets": [
            {
                "name": item.get("name"),
                "venue": item.get("target_venue"),
                "priority": item.get("submission_priority_score"),
                "rewrite_gain": item.get("rewrite_priority_gain_total"),
                "next_step": item.get("suggested_next_step"),
            }
            for item in top_rewrite
        ],
        "recent_followup_wins": [
            {
                "paper": item.get("paper"),
                "priority_delta": item.get("priority_delta"),
                "priority_after": item.get("submission_priority_score"),
            }
            for item in followup_items
        ],
        "priorities": priorities,
        "actions": actions,
        "blockers": blockers,
        "counts": {
            "submission_board_items": len(_flatten_submission_board(submission_board)),
            "rewrite_board_items": len(rewrite_board),
            "shortlist_items": len(shortlist),
        },
        "source_runtime": status.get("source_runtime", {}),
        "source_runtime_rows": source_rows,
        "source_actions": [
            {
                "name": row.get("name"),
                "availability_state": row.get("availability_state"),
                "health_score": row.get("health_score"),
                "suggested_action": row.get("suggested_action"),
            }
            for row in source_rows[:5]
        ],
        "source_advisory": source_advisory,
        "source_mix_advisory": source_mix_advisory,
        "source_next_batch_advisory": source_next_batch_advisory,
        "do_now_actions": _select_do_now_actions(source_advisory),
        "control_summary": control_summary,
        "recent_control_events": recent_control_events,
        "failure_hotspots": failure_hotspots,
        "rewrite_style_hotspots": rewrite_style_hotspots,
        "pipeline_contracts": pipeline_contracts,
        "pipeline_contract_strategy": status.get("active_pipeline_contract_strategy")
        or {},
        "recommended_commands": commands,
    }
    brief.update(
        _build_handoff_recovery(
            status, brief, Path(status.get("daemon_dir") or parsed.research_dir)
        )
    )
    brief["primary_action_queue"] = _build_primary_action_queue(
        do_now_actions=brief.get("do_now_actions") or [],
        recovery_command=brief.get("recovery_command"),
        recovery_reason=brief.get("recovery_reason"),
        recommended_commands=brief.get("recommended_commands") or {},
    )
    return brief


def _build_operator_brief_markdown(brief: dict[str, Any]) -> str:
    lines = [
        "# Operator Brief",
        "",
        f"- Generated at: {brief.get('generated_at')}",
        f"- Health: {brief.get('health', {}).get('score')} ({brief.get('health', {}).get('state')})",
        f"- Active source: {brief.get('active_source')}",
        f"- Daypart: {brief.get('current_daypart')}",
        f"- Phase: {brief.get('guardrail_phase')}",
        f"- Mode: {brief.get('guardrail_mode')} ({brief.get('guardrail_reason')})",
        f"- Success count: {brief.get('success_count')}",
        f"- Failure count: {brief.get('failure_count')}",
        "",
        "## Priorities",
    ]
    for item in brief.get("priorities", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Do Now"])
    if brief.get("do_now_actions"):
        for item in brief.get("do_now_actions", [])[:3]:
            command = item.get("command")
            if command:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{command}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate source actions right now.")

    lines.extend(["", "## Primary Commands"])
    if brief.get("primary_action_queue"):
        for item in brief.get("primary_action_queue", [])[:5]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")

    lines.extend(["", "## Health Signals"])
    for item in brief.get("health", {}).get("reasons", []):
        lines.append(f"- {item}")
    lines.append(f"- Recommendation: {brief.get('health', {}).get('recommendation')}")
    lines.extend(["", "## Immediate Actions"])
    for item in brief.get("actions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Source Health Snapshot"])
    for row in (brief.get("source_runtime_rows") or [])[:5]:
        lines.append(
            f"- {row.get('name')} | health={row.get('health_score')} | state={row.get('availability_state')} | venue={row.get('target_venue')} | paper_types={','.join(row.get('paper_types') or [])}"
        )

    lines.extend(["", "## Source Actions"])
    for row in (brief.get("source_actions") or [])[:5]:
        lines.append(
            f"- {row.get('name')} | state={row.get('availability_state')} | health={row.get('health_score')} | action={row.get('suggested_action')}"
        )

    lines.extend(["", "## Source Plan"])
    if brief.get("source_advisory"):
        for item in brief.get("source_advisory", [])[:6]:
            command = item.get("command")
            if command:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | state={item.get('state')} | health={item.get('health_score')} | {item.get('recommendation')} | command=`{command}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | state={item.get('state')} | health={item.get('health_score')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No source advisory recommendations yet.")

    lines.extend(["", "## Source Mix"])
    source_mix = brief.get("source_mix_advisory") or {}
    mix_summary = source_mix.get("summary") or {}
    lines.append(
        f"- desired_policy={source_mix.get('desired_policy') or 'n/a'} | dominant_archetype={mix_summary.get('dominant_archetype')} | dominant_workflow={mix_summary.get('dominant_workflow_mode')}"
    )
    lines.append(
        f"- archetype_counts={mix_summary.get('archetype_counts')} | workflow_counts={mix_summary.get('workflow_mode_counts')}"
    )
    if source_mix.get("recommendations"):
        for item in source_mix.get("recommendations", [])[:5]:
            lines.append(
                f"- [{item.get('tier')}] {item.get('recommendation')}"
            )
    else:
        lines.append("- No source mix recommendations yet.")

    lines.extend(["", "## Next Batch Recipe"])
    next_batch = brief.get("source_next_batch_advisory") or {}
    next_batch_cadence = next_batch.get("cadence") or {}
    lines.append(
        f"- cadence={next_batch_cadence.get('label') or 'n/a'} | {next_batch_cadence.get('reason') or 'n/a'}"
    )
    if next_batch.get("slots"):
        for slot in next_batch.get("slots", [])[:5]:
            lines.append(
                f"- [{slot.get('lane')}] {slot.get('source')} | share={slot.get('share')} | state={slot.get('availability_state') or 'n/a'} | health={slot.get('health_score')} | workflow={slot.get('source_workflow_mode')} | {slot.get('rationale')}"
            )
    else:
        lines.append("- No next-batch source slots yet.")

    lines.extend(["", "## Evidence Strategy"])
    evidence_strategy = brief.get("evidence_strategy") or {}
    if evidence_strategy.get("enabled"):
        lines.append(
            f"- mode={evidence_strategy.get('mode')} | reason={evidence_strategy.get('reason')}"
        )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Pipeline Contract Strategy"])
    pipeline_strategy = brief.get("pipeline_contract_strategy") or {}
    if pipeline_strategy.get("enabled"):
        lines.append(
            f"- mode={pipeline_strategy.get('mode')} | reason={pipeline_strategy.get('reason')}"
        )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Quality Governor"])
    quality_governor = brief.get("quality_governor") or {}
    if quality_governor.get("enabled"):
        lines.append(
            f"- mode={quality_governor.get('mode')} | rewrite_top_k={quality_governor.get('rewrite_followup_top_k_effective')} | dossier_top_k={quality_governor.get('auto_submission_dossier_top_k_effective')} | source_plan_max_actions={quality_governor.get('auto_source_plan_max_actions_effective')}"
        )
        lines.append(f"- reason={quality_governor.get('reason')}")
    else:
        lines.append("- disabled")

    lines.extend(["", "## Auto Source Plan"])
    auto_source_plan = brief.get("auto_source_plan") or {}
    if auto_source_plan.get("enabled"):
        lines.append(
            f"- enabled | min_health={auto_source_plan.get('min_health')} | max_actions={auto_source_plan.get('max_actions')} | expires_after_cycles={auto_source_plan.get('expires_after_cycles')}"
        )
        if auto_source_plan.get("applied"):
            for item in auto_source_plan.get("applied", [])[:5]:
                lines.append(
                    f"- applied {item.get('operation')} to {item.get('source')} | health={item.get('health_score')} | tier={item.get('tier')}"
                )
        else:
            lines.append(
                f"- no action applied | reason={auto_source_plan.get('skipped_reason')}"
            )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Submission Autopilot"])
    submission_autopilot = brief.get("submission_autopilot") or {}
    if submission_autopilot.get("enabled"):
        lines.append(
            f"- enabled | top_k={submission_autopilot.get('top_k')} | require_ready={submission_autopilot.get('require_ready')} | require_gate={submission_autopilot.get('require_gate')}"
        )
        if submission_autopilot.get("exported"):
            for item in submission_autopilot.get("exported", [])[:5]:
                lines.append(
                    f"- exported {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
                )
        elif submission_autopilot.get("reused"):
            for item in submission_autopilot.get("reused", [])[:5]:
                lines.append(
                    f"- reused {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
                )
        else:
            lines.append(
                f"- no dossier exported | reason={submission_autopilot.get('skipped_reason')}"
            )
    else:
        lines.append("- disabled")

    lines.extend(["", "## Failure Guard"])
    failure_guard = brief.get("failure_guard") or {}
    if failure_guard.get("enabled"):
        lines.append(
            f"- source={failure_guard.get('source')} | consecutive_failures={failure_guard.get('consecutive_failures')} | threshold={failure_guard.get('threshold')} | cooldown={failure_guard.get('cooldown_cycles')} | applied={failure_guard.get('applied')}"
        )
        if failure_guard.get("reason"):
            lines.append(f"- reason={failure_guard.get('reason')}")
    else:
        lines.append("- disabled")

    lines.extend(["", "## Control Summary"])
    for item in brief.get("control_summary") or []:
        lines.append(f"- {item}")

    lines.extend(["", "## Recent Control Events"])
    if brief.get("recent_control_events"):
        for item in brief.get("recent_control_events", []):
            lines.append(
                f"- {item.get('type')} | source={item.get('matched_key') or item.get('active_source')} | at={item.get('timestamp')}"
            )
    else:
        lines.append("- No recent control events recorded.")

    lines.extend(["", "## Recent Failure Hotspots"])
    if brief.get("failure_hotspots"):
        for item in brief.get("failure_hotspots", []):
            lines.append(f"- {item.get('reason')}: {item.get('count')}")
    else:
        lines.append("- No recent failure hotspots detected.")

    lines.extend(["", "## Rewrite Style Hotspots"])
    if brief.get("rewrite_style_hotspots"):
        for item in brief.get("rewrite_style_hotspots", []):
            lines.append(f"- {item.get('style')}: {item.get('score')}")
    else:
        lines.append("- No rewrite style hotspot signals yet.")

    lines.extend(["", "## Pipeline Contracts"])
    pipeline_contracts = brief.get("pipeline_contracts") or {}
    if pipeline_contracts.get("enabled"):
        lines.append(
            f"- project_count={pipeline_contracts.get('project_count')} | blocked_projects={pipeline_contracts.get('blocked_project_count')} | stage_blocked_projects={pipeline_contracts.get('stage_blocked_project_count')} | stage_missing_projects={pipeline_contracts.get('stage_missing_project_count')} | review_low_resolution_projects={pipeline_contracts.get('review_low_resolution_project_count')} | review_low_binding_projects={pipeline_contracts.get('review_low_binding_project_count')} | review_low_repair_ready_projects={pipeline_contracts.get('review_low_repair_ready_project_count')} | failed_projects={pipeline_contracts.get('failed_project_count')} | blocked_figures={pipeline_contracts.get('blocked_figure_count')} | failed_experiments={pipeline_contracts.get('failed_experiment_count')} | budget_exhausted={pipeline_contracts.get('budget_exhausted_experiment_count')}"
        )
        if pipeline_contracts.get("dominant_execution_policy"):
            lines.append(
                f"- dominant execution policy: {pipeline_contracts.get('dominant_execution_policy')}"
            )
        if pipeline_contracts.get("avg_stage_overall_score") is not None:
            lines.append(
                f"- average stage-standard score: {pipeline_contracts.get('avg_stage_overall_score')}"
            )
        if pipeline_contracts.get("avg_review_resolution_rate") is not None:
            lines.append(
                f"- average reviewer-resolution rate: {pipeline_contracts.get('avg_review_resolution_rate')}"
            )
        if pipeline_contracts.get("avg_review_binding_coverage") is not None:
            lines.append(
                f"- average reviewer target-binding coverage: {pipeline_contracts.get('avg_review_binding_coverage')}"
            )
        if pipeline_contracts.get("avg_review_repair_ready_coverage") is not None:
            lines.append(
                f"- average reviewer repair-ready coverage: {pipeline_contracts.get('avg_review_repair_ready_coverage')}"
            )
        artifact_blockers = pipeline_contracts.get("artifact_blockers") or {}
        if artifact_blockers:
            lines.append(
                "- artifact blockers: "
                + ", ".join(f"{name}={count}" for name, count in artifact_blockers.items())
            )
        stage_risks = pipeline_contracts.get("top_stage_standard_risks") or {}
        if stage_risks:
            lines.append(
                "- stage-standard risks: "
                + ", ".join(f"{name}={count}" for name, count in stage_risks.items())
            )
        execution_policy_counts = pipeline_contracts.get("execution_policy_counts") or {}
        if execution_policy_counts:
            lines.append(
                "- execution policies: "
                + ", ".join(
                    f"{name}={count}" for name, count in execution_policy_counts.items()
                )
            )
        budget_status_counts = pipeline_contracts.get("budget_status_counts") or {}
        if budget_status_counts:
            lines.append(
                "- budget status: "
                + ", ".join(
                    f"{name}={count}" for name, count in budget_status_counts.items()
                )
            )
        for item in pipeline_contracts.get("top_blocked_projects") or []:
            lines.append(
                f"- {item.get('project')} | blocked={item.get('blocked_artifacts')} | failed={item.get('failed_artifacts')} | missing={item.get('missing_artifacts')} | "
                f"stage_score={item.get('stage_overall_score')} blocked_stages={item.get('blocked_stage_count')} attention_stages={item.get('needs_attention_stage_count')} missing_stages={item.get('missing_stage_count')} | "
                f"review_resolution={item.get('review_resolution_rate')} review_binding={item.get('review_target_binding_coverage')} repair_ready={item.get('review_repair_ready_coverage')} persistent_review_issues={item.get('review_persistent_issue_count')}"
            )
    else:
        lines.append("- No contract-enabled pipeline roots detected yet.")

    lines.extend(["", "## Active Blockers"])
    if brief.get("blockers"):
        for item in brief.get("blockers", []):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")

    lines.extend(["", "## Top Submission Targets"])
    for item in brief.get("top_submission_targets", []):
        lines.append(
            f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')}"
        )
    if not brief.get("top_submission_targets"):
        lines.append("- No strong submission targets yet.")

    lines.extend(["", "## Top Rewrite Targets"])
    for item in brief.get("top_rewrite_targets", []):
        lines.append(
            f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')} | next={item.get('next_step')}"
        )
    if not brief.get("top_rewrite_targets"):
        lines.append("- No rewrite targets currently meet the bar.")

    lines.extend(["", "## Primary Commands"])
    if brief.get("primary_action_queue"):
        for item in brief.get("primary_action_queue", [])[:5]:
            lines.append(
                f"- [{item.get('priority')}] {item.get('label')} | {item.get('reason')} | command=`{item.get('command')}`"
            )
    else:
        lines.append("- No prioritized commands available.")

    lines.extend(["", "## Recommended Commands"])
    for key, value in (brief.get("recommended_commands") or {}).items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Recent Follow-up Wins"])
    for item in brief.get("recent_followup_wins", []):
        lines.append(
            f"- {item.get('paper')} | priority_delta={item.get('priority_delta')} | priority_after={item.get('priority_after')}"
        )
    if not brief.get("recent_followup_wins"):
        lines.append("- No recent follow-up wins recorded.")

    return "\n".join(lines) + "\n"


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_recent_cycle_history(
    daemon_dir: Path, max_entries: int = 20
) -> list[dict[str, Any]]:
    history_path = daemon_dir / "cycle_history.jsonl"
    if not history_path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "cycle" not in item:
            continue
        items.append(item)
    return items[-max_entries:]


def _render_section(title: str, content: str, *, open_by_default: bool = True) -> str:
    open_attr = " open" if open_by_default else ""
    return f"<details class='panel'{open_attr}><summary>{_html_escape(title)}</summary><div class='panel-body'>{content}</div></details>"


def _sparkline_svg(
    values: list[float], *, width: int = 280, height: int = 48, color: str = "#7fb3ff"
) -> str:
    if not values:
        return "<p class='small'>No recent data.</p>"
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)
    points = []
    for idx, value in enumerate(values):
        x = (idx / max(1, len(values) - 1)) * (width - 8) + 4
        y = height - (((value - min_v) / span) * (height - 12) + 6)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='trend'>"
        f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}' />"
        "</svg>"
    )


def _render_trend_card(title: str, values: list[float], *, latest: Any = None) -> str:
    latest_text = f"<div class='trend-latest'>{_html_escape(latest if latest is not None else (values[-1] if values else 'n/a'))}</div>"
    return (
        "<div class='trend-card'>"
        f"<div class='trend-title'>{_html_escape(title)}</div>"
        f"{latest_text}"
        f"{_sparkline_svg(values)}"
        "</div>"
    )


def _build_recent_trend_metrics(
    cycle_history: list[dict[str, Any]],
) -> dict[str, list[float]]:
    closure_series: list[float] = []
    backlog_series: list[float] = []
    p0_backlog_series: list[float] = []
    gate_ready_series: list[float] = []
    for item in cycle_history:
        feedback = item.get("active_source_feedback") or {}
        closure = feedback.get("avg_experiment_todo_closure_rate")
        backlog = feedback.get("avg_experiment_todo")
        p0_backlog = feedback.get("avg_experiment_todo_p0")
        gate_ready = feedback.get("self_review_gate_ready_rate")
        if isinstance(closure, (int, float)):
            closure_series.append(float(closure))
        if isinstance(backlog, (int, float)):
            backlog_series.append(float(backlog))
        if isinstance(p0_backlog, (int, float)):
            p0_backlog_series.append(float(p0_backlog))
        if isinstance(gate_ready, (int, float)):
            gate_ready_series.append(float(gate_ready))
    return {
        "submission_board_items": [
            float((item.get("views") or {}).get("submission_board_items") or 0)
            for item in cycle_history
        ],
        "rewrite_board_items": [
            float((item.get("views") or {}).get("rewrite_board_items") or 0)
            for item in cycle_history
        ],
        "shortlist_items": [
            float((item.get("views") or {}).get("shortlist_items") or 0)
            for item in cycle_history
        ],
        "duration_seconds": [
            float(item.get("duration_seconds") or 0) for item in cycle_history
        ],
        "returncode": [float(item.get("returncode") or 0) for item in cycle_history],
        "experiment_todo_closure_rate": closure_series,
        "experiment_todo_backlog": backlog_series,
        "experiment_todo_p0_backlog": p0_backlog_series,
        "self_review_gate_ready_rate": gate_ready_series,
    }


def _build_live_dashboard_payload(
    status: dict[str, Any], daemon_dir: Path
) -> dict[str, Any]:
    cycle_summary = _safe_read_json(daemon_dir / "latest_cycle_summary.json")
    daily_summary = _safe_read_json(daemon_dir / "latest_daily_summary.json")
    daily_report = _safe_read_json(daemon_dir / "latest_daily_report.json")
    report_index = _safe_read_json(daemon_dir / "reports" / "index.json")
    report_trends = _safe_read_json(daemon_dir / "reports" / "trends.json")
    operator_brief = _safe_read_json(daemon_dir / "latest_operator_brief.json")
    handoff_report = _safe_read_json(daemon_dir / "latest_handoff_report.json")
    primary_action_queue = _safe_read_json(
        daemon_dir / "latest_primary_action_queue.json"
    )
    source_runtime_board = _safe_read_json(
        daemon_dir / "latest_source_runtime_board.json"
    )
    source_health_board = _safe_read_json(
        daemon_dir / "latest_source_health_board.json"
    )
    source_batch_plan = _safe_read_json(daemon_dir / "latest_source_batch_plan.json")
    source_next_batch = _safe_read_json(daemon_dir / "latest_source_next_batch.json")
    rewrite_followup = _safe_read_json(daemon_dir / "latest_rewrite_followup.json")
    control_history = _load_recent_control_events(daemon_dir)
    cycle_history = _load_recent_cycle_history(daemon_dir)
    trend_metrics = _build_recent_trend_metrics(cycle_history)
    return {
        "generated_at": _now_iso(),
        "refresh_seconds": status.get("dashboard_refresh_seconds", 30),
        "daemon_status": status,
        "daemon_control": status.get("control") or {},
        "cycle_summary": cycle_summary,
        "daily_summary": daily_summary,
        "daily_report": daily_report,
        "report_index": report_index,
        "report_trends": report_trends,
        "operator_brief": operator_brief,
        "handoff_report": handoff_report,
        "primary_action_queue": primary_action_queue,
        "source_runtime_board": source_runtime_board,
        "source_health_board": source_health_board,
        "source_batch_plan": source_batch_plan,
        "source_next_batch": source_next_batch,
        "rewrite_followup": rewrite_followup,
        "control_history": control_history,
        "cycle_history": cycle_history,
        "trend_metrics": trend_metrics,
        "dashboard_url": status.get("dashboard_url"),
    }


def _html_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _path_to_href(value: Any) -> str:
    if not value:
        return ""
    try:
        return Path(str(value)).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _render_stat_cards(cards: list[tuple[str, Any]]) -> str:
    items = []
    for label, value in cards:
        items.append(
            f"<div class='stat-card'><div class='stat-label'>{_html_escape(label)}</div><div class='stat-value'>{_html_escape(value)}</div></div>"
        )
    return (
        f"<section><div class='stats-grid'>{''.join(items)}</div></section>"
        if items
        else ""
    )


def _render_link_list(title: str, mapping: dict[str, Any]) -> str:
    if not mapping:
        return _render_section(title, "<p>No links.</p>", open_by_default=False)
    items = []
    for key, value in mapping.items():
        href = _path_to_href(value)
        if href:
            items.append(
                f"<li><strong>{_html_escape(key)}</strong>: <a href='{_html_escape(href)}'>{_html_escape(value)}</a></li>"
            )
        else:
            items.append(
                f"<li><strong>{_html_escape(key)}</strong>: {_html_escape(value)}</li>"
            )
    return _render_section(title, f"<ul>{''.join(items)}</ul>", open_by_default=False)


def _render_key_value_table(title: str, mapping: dict[str, Any]) -> str:
    if not mapping:
        return _render_section(title, "<p>No data.</p>")
    rows = []
    for key, value in mapping.items():
        rows.append(
            f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(value)}</td></tr>"
        )
    return _render_section(title, f"<table>{''.join(rows)}</table>")


def _render_rows_table(
    title: str, rows: list[dict[str, Any]], columns: list[str]
) -> str:
    if not rows:
        return _render_section(title, "<p>No rows.</p>", open_by_default=False)
    thead = "".join(f"<th>{_html_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        body = "".join(
            f"<td>{_html_escape(row.get(column, ''))}</td>" for column in columns
        )
        body_rows.append(f"<tr>{body}</tr>")
    return _render_section(
        title,
        f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>",
        open_by_default=False,
    )


def _render_cycle_history_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _render_section(
            title, "<p>No recent cycle history.</p>", open_by_default=False
        )
    columns = [
        "cycle",
        "returncode",
        "duration_seconds",
        "guardrail_phase",
        "guardrail_mode",
        "submission_board_items",
        "rewrite_board_items",
        "shortlist_items",
        "todo_closure_rate",
        "todo_backlog",
        "todo_p0_backlog",
    ]
    thead = "".join(f"<th>{_html_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        views = row.get("views") or {}
        feedback = row.get("active_source_feedback") or {}
        flattened = {
            "cycle": row.get("cycle"),
            "returncode": row.get("returncode"),
            "duration_seconds": row.get("duration_seconds"),
            "guardrail_phase": row.get("guardrail_phase"),
            "guardrail_mode": row.get("guardrail_mode"),
            "submission_board_items": views.get("submission_board_items"),
            "rewrite_board_items": views.get("rewrite_board_items"),
            "shortlist_items": views.get("shortlist_items"),
            "todo_closure_rate": feedback.get("avg_experiment_todo_closure_rate"),
            "todo_backlog": feedback.get("avg_experiment_todo"),
            "todo_p0_backlog": feedback.get("avg_experiment_todo_p0"),
        }
        row_class = (
            "danger-row"
            if row.get("returncode") not in (0, None)
            else ("warn-row" if (views.get("submission_board_items") or 0) == 0 else "")
        )
        body = "".join(
            f"<td>{_html_escape(flattened.get(column, ''))}</td>" for column in columns
        )
        body_rows.append(f"<tr class='{row_class}'>{body}</tr>")
    content = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    return _render_section(title, content, open_by_default=False)


def _build_live_dashboard_html(payload: dict[str, Any]) -> str:
    status = payload.get("daemon_status") or {}
    cycle = payload.get("cycle_summary") or {}
    daily = payload.get("daily_summary") or {}
    brief = payload.get("operator_brief") or {}
    runtime_rows = (payload.get("source_runtime_board") or {}).get("rows") or []
    health_rows = (payload.get("source_health_board") or {}).get("rows") or []
    batch_plan_rows = (payload.get("source_batch_plan") or {}).get("rows") or []
    next_batch_rows = (payload.get("source_next_batch") or {}).get("slots") or []
    followup_items = (payload.get("rewrite_followup") or {}).get("items") or []
    trend_metrics = payload.get("trend_metrics") or {}
    active_feedback = status.get("active_source_feedback_snapshot") or {}
    refresh_seconds = payload.get("refresh_seconds") or 30

    style = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #0b1020; color: #e8ecf1; }
a { color: #9fc2ff; }
h1, h2 { color: #ffffff; margin-top: 0; }
details.panel { background: #121936; border: 1px solid #273056; border-radius: 12px; padding: 0; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); overflow: hidden; }
details.panel > summary { cursor: pointer; list-style: none; padding: 14px 16px; font-weight: 700; color: #fff; background: #111833; }
details.panel > summary::-webkit-details-marker { display: none; }
.panel-body { padding: 16px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #2a355f; vertical-align: top; }
th { color: #9fb3ff; width: 220px; }
ul { margin: 0; padding-left: 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }
.stat-card, .trend-card { background: #121936; border: 1px solid #273056; border-radius: 12px; padding: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); }
.stat-label, .trend-title { color: #9aa6c0; font-size: 12px; margin-bottom: 6px; }
.stat-value, .trend-latest { color: #ffffff; font-size: 24px; font-weight: 700; }
.small { color: #9aa6c0; font-size: 12px; }
.badges { margin: 10px 0 18px 0; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #24305c; color: #dbe3ff; margin-right: 8px; margin-bottom: 8px; }
.hero { margin-bottom: 18px; }
.hero p { margin: 6px 0; }
.trend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 18px; }
.danger-row td { background: rgba(180, 60, 60, 0.18); }
.warn-row td { background: rgba(180, 140, 60, 0.14); }
"""

    hero = _render_stat_cards(
        [
            (
                "Health",
                f"{(cycle.get('health') or brief.get('health') or {}).get('score', 'n/a')}",
            ),
            ("Phase", status.get("guardrail_phase") or "n/a"),
            ("Mode", status.get("guardrail_mode") or "n/a"),
            ("Cycle", status.get("cycle") or 0),
            ("Submissions", (cycle.get("submission_board_items") or 0)),
            ("Rewrite Targets", (cycle.get("rewrite_board_items") or 0)),
            ("Follow-up Δ", (cycle.get("followup_avg_priority_delta") or 0)),
        ]
    )

    trend_cards = (
        "<section><div class='trend-grid'>"
        + "".join(
            [
                _render_trend_card(
                    "Submission Board",
                    trend_metrics.get("submission_board_items") or [],
                    latest=cycle.get("submission_board_items"),
                ),
                _render_trend_card(
                    "Rewrite Board",
                    trend_metrics.get("rewrite_board_items") or [],
                    latest=cycle.get("rewrite_board_items"),
                ),
                _render_trend_card(
                    "Shortlist",
                    trend_metrics.get("shortlist_items") or [],
                    latest=cycle.get("shortlist_items"),
                ),
                _render_trend_card(
                    "Cycle Duration",
                    trend_metrics.get("duration_seconds") or [],
                    latest=(
                        (payload.get("cycle_history") or [{}])[-1].get(
                            "duration_seconds"
                        )
                        if payload.get("cycle_history")
                        else None
                    ),
                ),
                _render_trend_card(
                    "Return Code",
                    trend_metrics.get("returncode") or [],
                    latest=status.get("last_returncode"),
                ),
                _render_trend_card(
                    "TODO Closure",
                    trend_metrics.get("experiment_todo_closure_rate") or [],
                    latest=active_feedback.get("avg_experiment_todo_closure_rate"),
                ),
                _render_trend_card(
                    "TODO Backlog",
                    trend_metrics.get("experiment_todo_backlog") or [],
                    latest=active_feedback.get("avg_experiment_todo"),
                ),
            ]
        )
        + "</div></section>"
    )

    artifact_links = _render_link_list(
        "Artifact Links",
        {
            "submission_board": cycle.get("last_views", {}).get("submission_board"),
            "rewrite_board": cycle.get("last_views", {}).get("rewrite_board"),
            "shortlist": cycle.get("last_views", {}).get("shortlist"),
            "source_runtime_board": cycle.get("source_runtime_board"),
            "source_health_board": cycle.get("source_health_board"),
            "source_batch_plan": cycle.get("source_batch_plan"),
            "operator_brief": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_operator_brief.md")
            )
            or None,
            "handoff_report": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_handoff_report.md")
            )
            or None,
            "daily_report": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_daily_report.md")
            )
            or None,
            "report_index": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "reports" / "index.md")
            )
            or None,
            "report_trends": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "reports" / "trends.md")
            )
            or None,
            "primary_action_queue": (
                status.get("daemon_dir")
                and str(
                    Path(status.get("daemon_dir")) / "latest_primary_action_queue.md"
                )
            )
            or None,
            "cycle_summary": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_cycle_summary.md")
            )
            or None,
            "daily_summary": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_daily_summary.md")
            )
            or None,
            "live_dashboard": payload.get("dashboard_url"),
            "daemon_control": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "daemon_control.json")
            )
            or None,
        },
    )

    sections = [
        _render_key_value_table(
            "Daemon Status",
            {
                "generated_at": payload.get("generated_at"),
                "state": status.get("state"),
                "cycle": status.get("cycle"),
                "guardrail_phase": status.get("guardrail_phase"),
                "guardrail_mode": status.get("guardrail_mode"),
                "active_source": status.get("active_source"),
                "current_daypart": status.get("current_daypart"),
                "last_error": status.get("last_error"),
                "next_cycle_at": status.get("next_cycle_at"),
            },
        ),
        _render_key_value_table("Daemon Control", payload.get("daemon_control") or {}),
        _render_key_value_table(
            "Control Summary", {"summary": (brief.get("control_summary") or [])}
        ),
        _render_key_value_table(
            "Cycle Summary",
            {
                "success_count": cycle.get("success_count"),
                "failure_count": cycle.get("failure_count"),
                "submission_board_items": cycle.get("submission_board_items"),
                "rewrite_board_items": cycle.get("rewrite_board_items"),
                "shortlist_items": cycle.get("shortlist_items"),
                "followup_avg_priority_delta": cycle.get("followup_avg_priority_delta"),
                "health_recommendation": (cycle.get("health") or {}).get(
                    "recommendation"
                ),
            },
        ),
        _render_key_value_table(
            "Daily Summary",
            {
                "success_count": daily.get("success_count"),
                "failure_count": daily.get("failure_count"),
                "submission_board_items": daily.get("submission_board_items"),
                "rewrite_board_items": daily.get("rewrite_board_items"),
                "shortlist_items": daily.get("shortlist_items"),
            },
        ),
        _render_key_value_table(
            "Daily Report",
            {
                "report_date": (payload.get("daily_report") or {}).get("report_date"),
                "health_state": (payload.get("daily_report") or {}).get("health_state"),
                "top_submission": (
                    (
                        (payload.get("daily_report") or {}).get(
                            "top_submission_targets"
                        )
                        or [{}]
                    )[0].get("name")
                    if (payload.get("daily_report") or {}).get("top_submission_targets")
                    else None
                ),
                "top_rewrite": (
                    (
                        (payload.get("daily_report") or {}).get("top_rewrite_targets")
                        or [{}]
                    )[0].get("name")
                    if (payload.get("daily_report") or {}).get("top_rewrite_targets")
                    else None
                ),
            },
        ),
        _render_key_value_table(
            "Report Archive Index",
            {
                "daily": ((payload.get("report_index") or {}).get("counts") or {}).get(
                    "daily"
                ),
                "handoff": (
                    (payload.get("report_index") or {}).get("counts") or {}
                ).get("handoff"),
            },
        ),
        _render_key_value_table(
            "Report Trends",
            {
                "average_daily_health_score": (payload.get("report_trends") or {}).get(
                    "average_daily_health_score"
                ),
                "daily_health_delta": (payload.get("report_trends") or {}).get(
                    "daily_health_delta"
                ),
                "latest_daily_health_score": (payload.get("report_trends") or {}).get(
                    "latest_daily_health_score"
                ),
                "trend_action_label": (payload.get("report_trends") or {}).get(
                    "trend_action_label"
                ),
                "trend_action_reason": (payload.get("report_trends") or {}).get(
                    "trend_action_reason"
                ),
                "trend_action_command": (payload.get("report_trends") or {}).get(
                    "trend_action_command"
                ),
                "average_todo_closure_rate": (payload.get("report_trends") or {}).get(
                    "average_todo_closure_rate"
                ),
                "latest_todo_closure_rate": (payload.get("report_trends") or {}).get(
                    "latest_todo_closure_rate"
                ),
                "todo_closure_delta": (payload.get("report_trends") or {}).get(
                    "todo_closure_delta"
                ),
                "average_todo_backlog": (payload.get("report_trends") or {}).get(
                    "average_todo_backlog"
                ),
            },
        ),
        _render_key_value_table(
            "Do Now",
            {
                "actions": [
                    f"[{item.get('tier')}] {item.get('source')} :: {item.get('recommendation')}"
                    + (f" | {item.get('command')}" if item.get("command") else "")
                    for item in (brief.get("do_now_actions") or [])
                ]
            },
        ),
        _render_key_value_table(
            "Operator Priorities",
            {
                "priorities": brief.get("priorities"),
                "actions": brief.get("actions"),
                "blockers": brief.get("blockers"),
            },
        ),
        _render_rows_table(
            "Primary Action Queue",
            (payload.get("primary_action_queue") or {}).get("items") or [],
            ["priority", "label", "category", "source", "reason", "command"],
        ),
        _render_key_value_table(
            "Recommended Commands", brief.get("recommended_commands") or {}
        ),
        _render_key_value_table(
            "Handoff Snapshot",
            {
                "attention_label": (payload.get("handoff_report") or {}).get(
                    "attention_label"
                ),
                "recovery_reason": (payload.get("handoff_report") or {}).get(
                    "recovery_reason"
                ),
                "recovery_command": (payload.get("handoff_report") or {}).get(
                    "recovery_command"
                ),
                "health_state": (payload.get("handoff_report") or {}).get(
                    "health_state"
                ),
                "phase": (payload.get("handoff_report") or {}).get("phase"),
                "mode": (payload.get("handoff_report") or {}).get("mode"),
            },
        ),
        _render_rows_table(
            "Source Advisory",
            brief.get("source_advisory") or [],
            ["tier", "source", "state", "health_score", "recommendation", "command"],
        ),
        _render_key_value_table(
            "Source Mix",
            {
                "desired_policy": (brief.get("source_mix_advisory") or {}).get(
                    "desired_policy"
                ),
                "dominant_archetype": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("dominant_archetype"),
                "dominant_workflow_mode": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("dominant_workflow_mode"),
                "archetype_counts": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("archetype_counts"),
                "workflow_mode_counts": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("workflow_mode_counts"),
                "recommendations": [
                    item.get("recommendation")
                    for item in (
                        (brief.get("source_mix_advisory") or {}).get(
                            "recommendations"
                        )
                        or []
                    )
                ],
            },
        ),
        artifact_links,
        _render_rows_table(
            "Source Runtime Board",
            runtime_rows,
            [
                "name",
                "availability_state",
                "availability_reason",
                "priority",
                "current_daypart",
                "target_venue",
                "paper_types",
                "num_ideas",
                "cycles_today",
                "successes_today",
                "cooldown_until_cycle",
                "suggested_action",
            ],
        ),
        _render_rows_table(
            "Source Health Board",
            health_rows,
            [
                "name",
                "health_score",
                "availability_state",
                "priority",
                "current_daypart",
                "target_venue",
                "cycles_today",
                "successes_today",
                "suggested_action",
            ],
        ),
        _render_rows_table(
            "Source Batch Plan",
            batch_plan_rows,
            [
                "tier",
                "source",
                "availability_state",
                "resolved_workflow_mode",
                "source_archetype",
                "batch_profile",
                "workflow_alignment_score",
                "health_score",
                "recommendation",
            ],
        ),
        _render_rows_table(
            "Next Batch Recipe",
            next_batch_rows,
            [
                "lane",
                "source",
                "availability_state",
                "source_workflow_mode",
                "source_archetype",
                "source_batch_profile",
                "share",
                "health_score",
                "rationale",
            ],
        ),
        _render_rows_table(
            "Latest Rewrite Follow-up",
            followup_items,
            [
                "paper",
                "status",
                "priority_before",
                "submission_priority_score",
                "priority_delta",
                "quality_gate_passed",
            ],
        ),
        _render_rows_table(
            "Recent Control Events",
            payload.get("control_history") or [],
            ["timestamp", "type", "matched_key"],
        ),
        _render_rows_table(
            "Recent Failure Hotspots",
            brief.get("failure_hotspots") or [],
            ["reason", "count"],
        ),
        _render_rows_table(
            "Rewrite Style Hotspots",
            brief.get("rewrite_style_hotspots") or [],
            ["style", "score"],
        ),
        _render_cycle_history_table(
            "Recent Cycle History", payload.get("cycle_history") or []
        ),
    ]

    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <meta http-equiv='refresh' content='{_html_escape(refresh_seconds)}'>
  <title>AI Scientist Daemon Dashboard</title>
  <style>{style}</style>
</head>
<body>
  <div class='hero'>
    <h1>AI Scientist Daemon Dashboard</h1>
    <p class='small'>Generated at {_html_escape(payload.get('generated_at'))} • auto-refresh every {_html_escape(refresh_seconds)}s</p>
    <div class='badges'>
      <span class='badge'>health={_html_escape((cycle.get('health') or brief.get('health') or {}).get('state', 'n/a'))}</span>
      <span class='badge'>phase={_html_escape(status.get('guardrail_phase'))}</span>
      <span class='badge'>mode={_html_escape(status.get('guardrail_mode'))}</span>
      <span class='badge'>daypart={_html_escape(status.get('current_daypart'))}</span>
      <span class='badge'>paused={_html_escape((status.get('control') or {}).get('paused'))}</span>
    </div>
  </div>
  {hero}
  {trend_cards}
  <div class='grid'>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def _build_handoff_recovery(
    status: dict[str, Any], brief: dict[str, Any], daemon_dir: Path
) -> dict[str, Any]:
    health = brief.get("health") or {}
    control = status.get("control") or {}
    recommended_commands = brief.get("recommended_commands") or {}
    do_now_actions = brief.get("do_now_actions") or []

    label = "stable"
    reason = (
        health.get("recommendation") or "System is operating within expected limits."
    )
    command = (
        recommended_commands.get("source_plan_command")
        or recommended_commands.get("next_cycle_command")
        or ""
    )

    if control.get("paused"):
        label = "recover"
        reason = "Daemon is paused and needs to be resumed before progress continues."
        command = _stable_wrapper_command_preview(daemon_dir, "resume")
    elif control.get("stop_after_cycle"):
        label = "recover"
        reason = "Daemon is configured to stop after the current cycle; clear this override if long-running execution should continue."
        command = _stable_wrapper_command_preview(daemon_dir, "clear-stop-after-cycle")
    elif health.get("state") == "risk":
        label = "risk"
        reason = "Health is in risk state; investigate blockers or environment reliability before leaving it unattended."
        command = (
            (
                do_now_actions[0].get("command")
                if do_now_actions and do_now_actions[0].get("command")
                else None
            )
            or recommended_commands.get("source_plan_command")
            or _stable_wrapper_command_preview(daemon_dir, "doctor", "--full")
        )
    elif health.get("state") == "stalled":
        label = "stalled"
        reason = "The system is no longer improving strongly enough; bias toward generation or take the top do-now action."
        command = (
            do_now_actions[0].get("command")
            if do_now_actions and do_now_actions[0].get("command")
            else None
        ) or _stable_wrapper_command_preview(daemon_dir, "set-mode", "generate_more")
    elif health.get("state") == "attention":
        label = "attention"
        reason = "The system is still productive, but needs monitoring over the next few cycles."
        command = (
            (
                do_now_actions[0].get("command")
                if do_now_actions and do_now_actions[0].get("command")
                else None
            )
            or recommended_commands.get("source_summary_command")
            or command
        )

    return {
        "attention_label": label,
        "recovery_reason": reason,
        "recovery_command": command,
    }


def _build_handoff_report(
    status: dict[str, Any],
    brief: dict[str, Any],
    cycle_summary: dict[str, Any],
    daily_summary: dict[str, Any],
    daemon_dir: Path,
) -> dict[str, Any]:
    recovery = _build_handoff_recovery(status, brief, daemon_dir)
    return {
        "generated_at": _now_iso(),
        "daemon_name": status.get("daemon_name"),
        "daemon_dir": str(daemon_dir),
        "health_score": (brief.get("health") or {}).get("score"),
        "health_state": (brief.get("health") or {}).get("state"),
        "phase": brief.get("guardrail_phase") or status.get("guardrail_phase"),
        "mode": brief.get("guardrail_mode") or status.get("guardrail_mode"),
        "active_source": brief.get("active_source") or status.get("active_source"),
        "daypart": brief.get("current_daypart") or status.get("current_daypart"),
        "success_count": brief.get("success_count") or status.get("success_count"),
        "failure_count": brief.get("failure_count") or status.get("failure_count"),
        "attention_label": recovery.get("attention_label"),
        "recovery_reason": recovery.get("recovery_reason"),
        "recovery_command": recovery.get("recovery_command"),
        "do_now_actions": brief.get("do_now_actions") or [],
        "blockers": brief.get("blockers") or [],
        "top_submission_targets": brief.get("top_submission_targets") or [],
        "top_rewrite_targets": brief.get("top_rewrite_targets") or [],
        "source_advisory": brief.get("source_advisory") or [],
        "recommended_commands": brief.get("recommended_commands") or {},
        "primary_action_queue": _build_primary_action_queue(
            do_now_actions=brief.get("do_now_actions") or [],
            recovery_command=recovery.get("recovery_command"),
            recovery_reason=recovery.get("recovery_reason"),
            recommended_commands=brief.get("recommended_commands") or {},
        ),
        "cycle_counts": {
            "submission_board_items": cycle_summary.get("submission_board_items"),
            "rewrite_board_items": cycle_summary.get("rewrite_board_items"),
            "shortlist_items": cycle_summary.get("shortlist_items"),
        },
        "daily_counts": {
            "submission_board_items": daily_summary.get("submission_board_items"),
            "rewrite_board_items": daily_summary.get("rewrite_board_items"),
            "shortlist_items": daily_summary.get("shortlist_items"),
        },
    }


def _build_handoff_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Handoff Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Health: {report.get('health_score')} ({report.get('health_state')})",
        f"- Phase: {report.get('phase')}",
        f"- Mode: {report.get('mode')}",
        f"- Active source: {report.get('active_source')}",
        f"- Daypart: {report.get('daypart')}",
        f"- Success count: {report.get('success_count')}",
        f"- Failure count: {report.get('failure_count')}",
        f"- Attention Label: {report.get('attention_label')}",
        "",
        "## Recovery",
        f"- Reason: {report.get('recovery_reason')}",
        (
            f"- Command: `{report.get('recovery_command')}`"
            if report.get("recovery_command")
            else "- Command: n/a"
        ),
        "",
        "## Do Now",
    ]
    if report.get("do_now_actions"):
        for item in report.get("do_now_actions", [])[:3]:
            if item.get("command"):
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{item.get('command')}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate actions available.")
    lines.extend(["", "## Top Submission Targets"])
    if report.get("top_submission_targets"):
        for item in report.get("top_submission_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | rewrite_gain={item.get('rewrite_gain')}"
            )
    else:
        lines.append("- No strong submission targets yet.")
    lines.extend(["", "## Top Rewrite Targets"])
    if report.get("top_rewrite_targets"):
        for item in report.get("top_rewrite_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | next={item.get('next_step')}"
            )
    else:
        lines.append("- No active rewrite targets.")
    lines.extend(["", "## Blockers"])
    if report.get("blockers"):
        for item in report.get("blockers"):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")
    lines.extend(["", "## Recommended Commands"])
    for key, value in (report.get("recommended_commands") or {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def _build_daily_report(
    status: dict[str, Any],
    brief: dict[str, Any],
    daily_summary: dict[str, Any],
    handoff_report: dict[str, Any],
    daemon_dir: Path,
) -> dict[str, Any]:
    report_date = datetime.now().date().isoformat()
    return {
        "generated_at": _now_iso(),
        "report_date": report_date,
        "daemon_name": status.get("daemon_name"),
        "daemon_dir": str(daemon_dir),
        "health_score": (brief.get("health") or {}).get("score"),
        "health_state": (brief.get("health") or {}).get("state"),
        "phase": brief.get("guardrail_phase") or status.get("guardrail_phase"),
        "mode": brief.get("guardrail_mode") or status.get("guardrail_mode"),
        "success_count": daily_summary.get("success_count")
        or brief.get("success_count")
        or status.get("success_count"),
        "failure_count": daily_summary.get("failure_count")
        or brief.get("failure_count")
        or status.get("failure_count"),
        "do_now_actions": brief.get("do_now_actions") or [],
        "top_submission_targets": brief.get("top_submission_targets") or [],
        "top_rewrite_targets": brief.get("top_rewrite_targets") or [],
        "blockers": brief.get("blockers") or [],
        "recovery_reason": handoff_report.get("recovery_reason"),
        "recovery_command": handoff_report.get("recovery_command"),
        "recommended_commands": brief.get("recommended_commands") or {},
        "daily_counts": {
            "submission_board_items": daily_summary.get("submission_board_items"),
            "rewrite_board_items": daily_summary.get("rewrite_board_items"),
            "shortlist_items": daily_summary.get("shortlist_items"),
        },
    }


def _build_daily_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Daily Report",
        "",
        f"- Generated at: {report.get('generated_at')}",
        f"- Report date: {report.get('report_date')}",
        f"- Health: {report.get('health_score')} ({report.get('health_state')})",
        f"- Phase: {report.get('phase')}",
        f"- Mode: {report.get('mode')}",
        f"- Success count: {report.get('success_count')}",
        f"- Failure count: {report.get('failure_count')}",
        "",
        "## Do Now",
    ]
    if report.get("do_now_actions"):
        for item in report.get("do_now_actions", [])[:3]:
            if item.get("command"):
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')} | command=`{item.get('command')}`"
                )
            else:
                lines.append(
                    f"- [{item.get('tier')}] {item.get('source')} | {item.get('recommendation')}"
                )
    else:
        lines.append("- No immediate actions recorded.")
    lines.extend(["", "## Top Submission Targets"])
    if report.get("top_submission_targets"):
        for item in report.get("top_submission_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')}"
            )
    else:
        lines.append("- No strong submission targets yet.")
    lines.extend(["", "## Top Rewrite Targets"])
    if report.get("top_rewrite_targets"):
        for item in report.get("top_rewrite_targets", [])[:3]:
            lines.append(
                f"- {item.get('name')} | venue={item.get('venue')} | priority={item.get('priority')} | next={item.get('next_step')}"
            )
    else:
        lines.append("- No active rewrite targets.")
    lines.extend(["", "## Blockers"])
    if report.get("blockers"):
        for item in report.get("blockers"):
            lines.append(f"- {item}")
    else:
        lines.append("- No major blockers detected.")
    lines.extend(["", "## Recovery"])
    lines.append(f"- Reason: {report.get('recovery_reason')}")
    lines.append(
        f"- Command: `{report.get('recovery_command')}`"
        if report.get("recovery_command")
        else "- Command: n/a"
    )
    return "\n".join(lines) + "\n"


def _build_report_archive_index(daemon_dir: Path) -> dict[str, Any]:
    reports_root = daemon_dir / "reports"
    entries: list[dict[str, Any]] = []
    counts = {"daily": 0, "handoff": 0}
    for kind in ["daily", "handoff"]:
        report_dir = reports_root / kind
        if not report_dir.exists():
            continue
        md_files = sorted(
            report_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        counts[kind] = len(md_files)
        for md_path in md_files[:200]:
            json_path = md_path.with_suffix(".json")
            payload = _safe_read_json(json_path) if json_path.exists() else {}
            entries.append(
                {
                    "kind": kind,
                    "name": md_path.name,
                    "path": str(md_path),
                    "json_path": str(json_path) if json_path.exists() else None,
                    "generated_at": payload.get("generated_at"),
                    "report_date": payload.get("report_date"),
                    "health_state": payload.get("health_state"),
                    "mtime": md_path.stat().st_mtime,
                }
            )
    entries.sort(key=lambda item: item.get("mtime") or 0, reverse=True)
    return {
        "generated_at": _now_iso(),
        "daemon_dir": str(daemon_dir),
        "counts": counts,
        "entries": entries,
    }


def _build_report_archive_index_markdown(index: dict[str, Any]) -> str:
    lines = [
        "# Report Archive Index",
        "",
        f"- Generated at: {index.get('generated_at')}",
        f"- Daily reports: {(index.get('counts') or {}).get('daily')}",
        f"- Handoff reports: {(index.get('counts') or {}).get('handoff')}",
        "",
        "## Recent Entries",
    ]
    if index.get("entries"):
        for item in index.get("entries", [])[:20]:
            lines.append(
                f"- [{item.get('kind')}] {item.get('name')} | report_date={item.get('report_date')} | health={item.get('health_state')} | path={item.get('path')}"
            )
    else:
        lines.append("- No archived reports found.")
    return "\n".join(lines) + "\n"


def _build_report_archive_trends(daemon_dir: Path) -> dict[str, Any]:
    index = _safe_read_json(daemon_dir / "reports" / "index.json")
    entries = index.get("entries") or []
    daily_entries = [item for item in entries if item.get("kind") == "daily"][:7]
    handoff_entries = [item for item in entries if item.get("kind") == "handoff"][:10]

    daily_reports = []
    for item in daily_entries:
        json_path = item.get("json_path")
        if json_path:
            payload = _safe_read_json(Path(json_path))
            if payload:
                daily_reports.append(payload)

    handoff_reports = []
    for item in handoff_entries:
        json_path = item.get("json_path")
        if json_path:
            payload = _safe_read_json(Path(json_path))
            if payload:
                handoff_reports.append(payload)
    cycle_history = _load_recent_cycle_history(daemon_dir, max_entries=20)

    daily_scores = [
        float(item.get("health_score"))
        for item in daily_reports
        if isinstance(item.get("health_score"), (int, float))
    ]
    avg_daily_health = (
        round(sum(daily_scores) / len(daily_scores), 3) if daily_scores else None
    )
    daily_health_delta = (
        round(daily_scores[0] - daily_scores[-1], 3) if len(daily_scores) >= 2 else None
    )
    todo_closure_series = [
        float((item.get("active_source_feedback") or {}).get("avg_experiment_todo_closure_rate"))
        for item in cycle_history
        if isinstance(
            (item.get("active_source_feedback") or {}).get(
                "avg_experiment_todo_closure_rate"
            ),
            (int, float),
        )
    ]
    todo_backlog_series = [
        float((item.get("active_source_feedback") or {}).get("avg_experiment_todo"))
        for item in cycle_history
        if isinstance(
            (item.get("active_source_feedback") or {}).get("avg_experiment_todo"),
            (int, float),
        )
    ]
    avg_todo_closure_rate = (
        round(sum(todo_closure_series) / len(todo_closure_series), 3)
        if todo_closure_series
        else None
    )
    latest_todo_closure_rate = (
        round(todo_closure_series[-1], 3) if todo_closure_series else None
    )
    todo_closure_delta = (
        round(todo_closure_series[-1] - todo_closure_series[0], 3)
        if len(todo_closure_series) >= 2
        else None
    )
    avg_todo_backlog = (
        round(sum(todo_backlog_series) / len(todo_backlog_series), 3)
        if todo_backlog_series
        else None
    )
    latest_todo_backlog = (
        round(todo_backlog_series[-1], 3) if todo_backlog_series else None
    )

    attention_counts: dict[str, int] = {}
    recovery_reasons: dict[str, int] = {}
    source_hotspots: dict[str, int] = {}
    for report in handoff_reports:
        label = report.get("attention_label") or "unknown"
        attention_counts[label] = attention_counts.get(label, 0) + 1
        reason = report.get("recovery_reason")
        if reason:
            recovery_reasons[reason] = recovery_reasons.get(reason, 0) + 1
        for item in report.get("do_now_actions") or []:
            source = item.get("source")
            if source:
                source_hotspots[source] = source_hotspots.get(source, 0) + 1

    sorted_attention = [
        {"label": label, "count": count}
        for label, count in sorted(
            attention_counts.items(), key=lambda pair: pair[1], reverse=True
        )
    ]
    sorted_recovery = [
        {"reason": reason, "count": count}
        for reason, count in sorted(
            recovery_reasons.items(), key=lambda pair: pair[1], reverse=True
        )[:5]
    ]
    sorted_hotspots = [
        {"source": source, "count": count}
        for source, count in sorted(
            source_hotspots.items(), key=lambda pair: pair[1], reverse=True
        )[:5]
    ]

    trend_action_label = "hold-course"
    trend_action_reason = "Trend signals are stable; continue the current operating pattern while monitoring the next reports."
    trend_action_command = _stable_wrapper_command_preview(
        daemon_dir, "report-trends", "--lines", "30"
    )

    top_attention = sorted_attention[0] if sorted_attention else None
    if avg_daily_health is None:
        trend_action_label = "insufficient-data"
        trend_action_reason = "Not enough archived daily reports yet; keep collecting runs before changing strategy."
        trend_action_command = _stable_wrapper_command_preview(
            daemon_dir, "daily-report", "--lines", "30"
        )
    elif (daily_health_delta is not None and daily_health_delta <= -8) or (
        top_attention and top_attention.get("label") in {"risk", "stalled"}
    ):
        trend_action_label = "stabilize"
        trend_action_reason = "Health is falling or risk/stalled labels are dominating recent handoffs; prefer recovery-first actions."
        trend_action_command = _stable_wrapper_command_preview(daemon_dir, "recover")
    elif (
        latest_todo_closure_rate is not None
        and latest_todo_closure_rate < 0.45
        and latest_todo_backlog is not None
        and latest_todo_backlog > 0
    ):
        trend_action_label = "close-todo-loop"
        trend_action_reason = "Experiment TODO closure remains low while backlog persists; bias toward rewrite-focused closure before expanding exploration."
        trend_action_command = _stable_wrapper_command_preview(
            daemon_dir, "set-mode", "focus_rewrite"
        )
    elif sorted_hotspots and (sorted_hotspots[0].get("count") or 0) >= 3:
        hotspot = sorted_hotspots[0].get("source")
        trend_action_label = "focus-hotspot"
        trend_action_reason = (
            f"{hotspot} keeps recurring in do-now actions; inspect source-level recommendations and consider focused scheduling."
            if hotspot
            else "One source is repeatedly surfacing in do-now actions; inspect source-level recommendations."
        )
        trend_action_command = _stable_wrapper_command_preview(
            daemon_dir, "source-plan"
        )
    elif avg_daily_health is not None and avg_daily_health < 60:
        trend_action_label = "bias-generate"
        trend_action_reason = "Average daily health is soft; bias toward generation to surface stronger candidates."
        trend_action_command = _stable_wrapper_command_preview(
            daemon_dir, "set-mode", "generate_more"
        )

    return {
        "generated_at": _now_iso(),
        "daemon_dir": str(daemon_dir),
        "daily_reports_considered": len(daily_reports),
        "handoff_reports_considered": len(handoff_reports),
        "average_daily_health_score": avg_daily_health,
        "daily_health_delta": daily_health_delta,
        "latest_daily_health_score": daily_scores[0] if daily_scores else None,
        "average_todo_closure_rate": avg_todo_closure_rate,
        "latest_todo_closure_rate": latest_todo_closure_rate,
        "todo_closure_delta": todo_closure_delta,
        "average_todo_backlog": avg_todo_backlog,
        "latest_todo_backlog": latest_todo_backlog,
        "attention_label_counts": sorted_attention,
        "recovery_reason_hotspots": sorted_recovery,
        "do_now_source_hotspots": sorted_hotspots,
        "trend_action_label": trend_action_label,
        "trend_action_reason": trend_action_reason,
        "trend_action_command": trend_action_command,
    }


def _build_report_archive_trends_markdown(trends: dict[str, Any]) -> str:
    lines = [
        "# Report Trends",
        "",
        f"- Generated at: {trends.get('generated_at')}",
        f"- Daily reports considered: {trends.get('daily_reports_considered')}",
        f"- Handoff reports considered: {trends.get('handoff_reports_considered')}",
        f"- Average daily health: {trends.get('average_daily_health_score')}",
        f"- Daily health delta: {trends.get('daily_health_delta')}",
        f"- Latest daily health: {trends.get('latest_daily_health_score')}",
        f"- Average TODO closure rate: {trends.get('average_todo_closure_rate')}",
        f"- Latest TODO closure rate: {trends.get('latest_todo_closure_rate')}",
        f"- TODO closure delta: {trends.get('todo_closure_delta')}",
        f"- Average TODO backlog: {trends.get('average_todo_backlog')}",
        f"- Latest TODO backlog: {trends.get('latest_todo_backlog')}",
        f"- Trend action: {trends.get('trend_action_label')}",
        "",
        "## Trend Action",
        f"- Reason: {trends.get('trend_action_reason')}",
        (
            f"- Command: `{trends.get('trend_action_command')}`"
            if trends.get("trend_action_command")
            else "- Command: n/a"
        ),
        "",
        "## Attention Labels",
    ]
    if trends.get("attention_label_counts"):
        for item in trends.get("attention_label_counts") or []:
            lines.append(f"- {item.get('label')}: {item.get('count')}")
    else:
        lines.append("- No handoff labels yet.")
    lines.extend(["", "## Recovery Reason Hotspots"])
    if trends.get("recovery_reason_hotspots"):
        for item in trends.get("recovery_reason_hotspots") or []:
            lines.append(f"- {item.get('reason')}: {item.get('count')}")
    else:
        lines.append("- No recovery hotspots yet.")
    lines.extend(["", "## Do-Now Source Hotspots"])
    if trends.get("do_now_source_hotspots"):
        for item in trends.get("do_now_source_hotspots") or []:
            lines.append(f"- {item.get('source')}: {item.get('count')}")
    else:
        lines.append("- No do-now source hotspots yet.")
    return "\n".join(lines) + "\n"


def _write_report_archive_trends(daemon_dir: Path) -> dict[str, Any]:
    reports_root = daemon_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    trends = _build_report_archive_trends(daemon_dir)
    _safe_write_json(reports_root / "trends.json", trends)
    (reports_root / "trends.md").write_text(
        _build_report_archive_trends_markdown(trends), encoding="utf-8"
    )
    return trends


def _write_report_archive_index(daemon_dir: Path) -> dict[str, Any]:
    reports_root = daemon_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    index = _build_report_archive_index(daemon_dir)
    _safe_write_json(reports_root / "index.json", index)
    (reports_root / "index.md").write_text(
        _build_report_archive_index_markdown(index), encoding="utf-8"
    )
    _write_report_archive_trends(daemon_dir)
    return index


def _write_daily_report(
    status: dict[str, Any],
    daemon_dir: Path,
    brief: dict[str, Any],
    daily_summary: dict[str, Any],
    handoff_report: dict[str, Any],
) -> dict[str, Any]:
    report = _build_daily_report(
        status, brief, daily_summary, handoff_report, daemon_dir
    )
    _safe_write_json(daemon_dir / "latest_daily_report.json", report)
    (daemon_dir / "latest_daily_report.md").write_text(
        _build_daily_report_markdown(report), encoding="utf-8"
    )
    reports_dir = daemon_dir / "reports" / "daily"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_date = report.get("report_date") or datetime.now().date().isoformat()
    _safe_write_json(reports_dir / f"{report_date}.json", report)
    (reports_dir / f"{report_date}.md").write_text(
        _build_daily_report_markdown(report), encoding="utf-8"
    )
    _write_report_archive_index(daemon_dir)
    return report


def _write_handoff_report(
    status: dict[str, Any],
    daemon_dir: Path,
    brief: dict[str, Any],
    cycle_summary: dict[str, Any],
    daily_summary: dict[str, Any],
) -> dict[str, Any]:
    report = _build_handoff_report(
        status, brief, cycle_summary, daily_summary, daemon_dir
    )
    _safe_write_json(daemon_dir / "latest_handoff_report.json", report)
    markdown = _build_handoff_report_markdown(report)
    (daemon_dir / "latest_handoff_report.md").write_text(markdown, encoding="utf-8")
    reports_dir = daemon_dir / "reports" / "handoff"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    _safe_write_json(reports_dir / f"{stamp}.json", report)
    (reports_dir / f"{stamp}.md").write_text(markdown, encoding="utf-8")
    _write_report_archive_index(daemon_dir)
    return report


def _build_primary_action_queue_payload(
    daemon_dir: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    brief = payload.get("operator_brief") or {}
    handoff_report = payload.get("handoff_report") or {}
    report_trends = payload.get("report_trends") or {}
    queue = _build_primary_action_queue(
        do_now_actions=brief.get("do_now_actions") or [],
        recovery_command=handoff_report.get("recovery_command"),
        recovery_reason=handoff_report.get("recovery_reason"),
        trend_action_command=report_trends.get("trend_action_command"),
        trend_action_reason=report_trends.get("trend_action_reason"),
        recommended_commands=brief.get("recommended_commands") or {},
    )
    return {
        "generated_at": _now_iso(),
        "daemon_dir": str(daemon_dir),
        "items": queue,
    }


def _write_primary_action_queue(
    daemon_dir: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    queue_payload = _build_primary_action_queue_payload(daemon_dir, payload)
    _safe_write_json(daemon_dir / "latest_primary_action_queue.json", queue_payload)
    (daemon_dir / "latest_primary_action_queue.md").write_text(
        _build_primary_action_queue_markdown(queue_payload.get("items") or []),
        encoding="utf-8",
    )
    return queue_payload


def _write_live_dashboard(status: dict[str, Any], daemon_dir: Path) -> dict[str, Any]:
    payload = _build_live_dashboard_payload(status, daemon_dir)
    payload["primary_action_queue"] = _write_primary_action_queue(daemon_dir, payload)
    _safe_write_json(daemon_dir / "latest_live_dashboard.json", payload)
    (daemon_dir / "latest_live_dashboard.html").write_text(
        _build_live_dashboard_html(payload), encoding="utf-8"
    )
    return payload


def _derive_quality_governor(
    status: dict[str, Any], daemon_dir: Path, parsed: argparse.Namespace
) -> dict[str, Any]:
    active_feedback = _active_source_feedback(status)
    avg_experiment_todo = (
        float(active_feedback.get("avg_experiment_todo"))
        if isinstance(active_feedback.get("avg_experiment_todo"), (int, float))
        else None
    )
    avg_experiment_todo_p0 = (
        float(active_feedback.get("avg_experiment_todo_p0"))
        if isinstance(active_feedback.get("avg_experiment_todo_p0"), (int, float))
        else None
    )
    avg_experiment_todo_closure_rate = (
        float(active_feedback.get("avg_experiment_todo_closure_rate"))
        if isinstance(
            active_feedback.get("avg_experiment_todo_closure_rate"), (int, float)
        )
        else None
    )
    summary = {
        "enabled": bool(parsed.auto_quality_governor),
        "mode": "disabled",
        "reason": "quality governor disabled",
        "recent_cycles": parsed.quality_governor_recent_cycles,
        "rewrite_followup_top_k_effective": parsed.rewrite_followup_top_k,
        "auto_submission_dossier_top_k_effective": parsed.auto_submission_dossier_top_k,
        "auto_submission_dossier_enabled": bool(parsed.auto_export_submission_dossier),
        "auto_source_plan_max_actions_effective": parsed.auto_source_plan_max_actions,
        "active_source_avg_experiment_todo": avg_experiment_todo,
        "active_source_avg_experiment_todo_p0": avg_experiment_todo_p0,
        "active_source_avg_experiment_todo_closure_rate": (
            avg_experiment_todo_closure_rate
        ),
    }
    if not parsed.auto_quality_governor:
        return summary

    history = _load_recent_cycle_history(
        daemon_dir, max_entries=max(1, int(parsed.quality_governor_recent_cycles))
    )
    health = _compute_health_snapshot(status)
    trend_metrics = _build_recent_trend_metrics(history)
    latest_submission = int(
        (status.get("last_views") or {}).get("submission_board_items") or 0
    )
    latest_shortlist = int((status.get("last_views") or {}).get("shortlist_items") or 0)
    avg_followup_gain = float(
        (status.get("guardrail_followup_metrics") or {}).get("avg_priority_delta")
        or 0.0
    )
    failure_count = sum(
        1 for item in history if (item.get("returncode") not in (0, None))
    )
    recent_window = max(1, len(history))
    experiment_todo_closure_floor = float(
        getattr(parsed, "quality_governor_experiment_todo_closure_floor", 0.45)
    )
    experiment_todo_p0_floor = float(
        getattr(parsed, "quality_governor_experiment_todo_p0_floor", 0.5)
    )
    experiment_todo_count_floor = float(
        getattr(parsed, "quality_governor_experiment_todo_count_floor", 2.5)
    )

    if health.get(
        "score", 0
    ) < parsed.quality_governor_stabilize_health_threshold or failure_count >= max(
        2, recent_window // 2
    ):
        summary.update(
            {
                "mode": "stabilize",
                "reason": "recent health or failure rate indicates the system should reduce expensive downstream actions",
                "rewrite_followup_top_k_effective": min(
                    int(parsed.rewrite_followup_top_k), 1
                ),
                "auto_submission_dossier_top_k_effective": 0,
                "auto_submission_dossier_enabled": False,
                "auto_source_plan_max_actions_effective": min(
                    int(parsed.auto_source_plan_max_actions), 1
                ),
            }
        )
        return summary

    todo_closure_pressure = any(
        [
            isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo >= experiment_todo_count_floor,
            isinstance(avg_experiment_todo_p0, (int, float))
            and avg_experiment_todo_p0 >= experiment_todo_p0_floor,
            isinstance(avg_experiment_todo_closure_rate, (int, float))
            and isinstance(avg_experiment_todo, (int, float))
            and avg_experiment_todo > 0
            and avg_experiment_todo_closure_rate < experiment_todo_closure_floor,
        ]
    )
    if todo_closure_pressure:
        summary.update(
            {
                "mode": "closure_repair",
                "reason": (
                    "active source still has unresolved experiment TODO pressure; prioritize closure and deeper rewrite passes before dossier expansion"
                ),
                "rewrite_followup_top_k_effective": max(
                    int(parsed.rewrite_followup_top_k),
                    min(
                        int(parsed.quality_governor_max_rewrite_top_k),
                        max(1, latest_shortlist or 1),
                    ),
                ),
                "auto_submission_dossier_top_k_effective": 0,
                "auto_submission_dossier_enabled": False,
                "auto_source_plan_max_actions_effective": min(
                    int(parsed.auto_source_plan_max_actions), 1
                ),
            }
        )
        return summary

    if (
        latest_submission >= parsed.guardrail_submission_target
        and avg_followup_gain >= parsed.quality_governor_exploit_followup_gain
    ):
        summary.update(
            {
                "mode": "exploit_quality",
                "reason": "recent cycles show strong submission inventory and healthy rewrite uplift",
                "rewrite_followup_top_k_effective": max(
                    int(parsed.rewrite_followup_top_k),
                    min(
                        int(parsed.quality_governor_max_rewrite_top_k),
                        max(1, latest_shortlist or 1),
                    ),
                ),
                "auto_submission_dossier_top_k_effective": max(
                    int(parsed.auto_submission_dossier_top_k),
                    min(
                        int(parsed.quality_governor_max_dossier_top_k),
                        max(1, latest_shortlist or 1),
                    ),
                ),
                "auto_submission_dossier_enabled": bool(
                    parsed.auto_export_submission_dossier
                ),
                "auto_source_plan_max_actions_effective": max(
                    int(parsed.auto_source_plan_max_actions),
                    min(int(parsed.quality_governor_max_source_plan_actions), 2),
                ),
            }
        )
        return summary

    if (
        latest_submission == 0
        and latest_shortlist == 0
        and avg_followup_gain < parsed.guardrail_min_followup_gain
    ):
        summary.update(
            {
                "mode": "expand_search",
                "reason": "recent cycles are not surfacing shortlist candidates, so shift effort away from expensive polishing",
                "rewrite_followup_top_k_effective": 0,
                "auto_submission_dossier_top_k_effective": 0,
                "auto_submission_dossier_enabled": False,
                "auto_source_plan_max_actions_effective": min(
                    int(parsed.auto_source_plan_max_actions), 1
                ),
            }
        )
        return summary

    summary.update(
        {
            "mode": "balanced",
            "reason": "quality signals do not justify stronger intervention; keep base automation settings",
        }
    )
    return summary


def _apply_auto_source_plan(
    status: dict[str, Any],
    daemon_dir: Path,
    brief: dict[str, Any],
    parsed: argparse.Namespace,
) -> dict[str, Any]:
    summary = {
        "enabled": bool(parsed.auto_apply_source_plan),
        "applied": [],
        "skipped_reason": None,
        "min_health": parsed.auto_source_plan_min_health,
        "max_actions": int(
            (status.get("quality_governor") or {}).get(
                "auto_source_plan_max_actions_effective",
                parsed.auto_source_plan_max_actions,
            )
        ),
        "expires_after_cycles": parsed.auto_source_plan_expires_after_cycles,
    }
    if not parsed.auto_apply_source_plan:
        brief["auto_source_plan"] = summary
        return brief

    control = _load_control_payload(
        daemon_dir, current_cycle=int(status.get("cycle", 0) or 0)
    )
    if control.get("paused"):
        summary["skipped_reason"] = "daemon paused by control"
        brief["auto_source_plan"] = summary
        return brief
    if control.get("source_commands"):
        summary["skipped_reason"] = "existing one-shot source commands already queued"
        brief["auto_source_plan"] = summary
        return brief

    def _row_matches_desired_policy(
        row: dict[str, Any], desired_policy: str | None
    ) -> bool:
        if not desired_policy:
            return False
        desired = str(desired_policy).strip()
        if not desired:
            return False
        policy_to_archetype = {
            "classic_pipeline": "template_first",
            "agentic_tree": "frontier_exploration",
            "program_driven": "program_guarded",
            "writing_studio": "writing_polish",
            "review_board": "review_hardening",
            "multi_agent_board": "paper_hardening_board",
        }
        if row.get("resolved_workflow_mode") == desired:
            return True
        compatible = row.get("compatible_workflow_modes") or []
        if desired in compatible:
            return True
        return row.get("source_archetype") == policy_to_archetype.get(desired)

    def _build_mix_candidates() -> list[dict[str, Any]]:
        rows = [
            row
            for row in (brief.get("source_runtime_rows") or [])
            if row.get("availability_state") == "ready"
        ]
        mix = brief.get("source_mix_advisory") or {}
        summary_block = mix.get("summary") or {}
        desired_policy = mix.get("desired_policy")
        existing_action_sources = {
            str(item.get("source"))
            for item in (brief.get("source_advisory") or [])
            if item.get("control_operation") in {"source-force-next", "source-boost-next"}
            and item.get("source")
        }
        candidates: list[dict[str, Any]] = []

        if desired_policy:
            aligned_rows = sorted(
                [
                    row
                    for row in rows
                    if _row_matches_desired_policy(row, str(desired_policy))
                ],
                key=lambda row: (
                    -(row.get("health_score") or -1),
                    -(row.get("workflow_alignment_score") or -1),
                    row.get("name") or "",
                ),
            )
            if aligned_rows:
                top = aligned_rows[0]
                source_name = top.get("name") or top.get("key")
                if source_name and str(source_name) not in existing_action_sources:
                    force_next = (top.get("health_score") or 0) >= 90
                    candidates.append(
                        {
                            "tier": "policy-align",
                            "source": source_name,
                            "state": top.get("availability_state"),
                            "health_score": top.get("health_score"),
                            "recommendation": (
                                f"Prioritize {source_name} to align the next cycle with desired policy {desired_policy}."
                            ),
                            "control_operation": (
                                "source-force-next"
                                if force_next
                                else "source-boost-next"
                            ),
                            "control_value": None if force_next else 3,
                            "source_plan_origin": "mix_advisory",
                            "mix_reason": "desired_policy_alignment",
                        }
                    )

        recommendation_labels = {
            str(item.get("label")): item
            for item in (mix.get("recommendations") or [])
            if isinstance(item, dict) and item.get("label")
        }
        dominant_archetype = str(summary_block.get("dominant_archetype") or "").strip()
        if "mix_too_narrow" in recommendation_labels and dominant_archetype:
            alternatives = sorted(
                [
                    row
                    for row in rows
                    if str(row.get("source_archetype") or "") != dominant_archetype
                ],
                key=lambda row: (
                    -(row.get("health_score") or -1),
                    -(row.get("workflow_alignment_score") or -1),
                    row.get("name") or "",
                ),
            )
            if alternatives:
                top = alternatives[0]
                source_name = top.get("name") or top.get("key")
                existing_mix_sources = {
                    str(item.get("source"))
                    for item in candidates
                    if item.get("source")
                }
                if (
                    source_name
                    and str(source_name) not in existing_action_sources
                    and str(source_name) not in existing_mix_sources
                ):
                    candidates.append(
                        {
                            "tier": "mix-rebalance",
                            "source": source_name,
                            "state": top.get("availability_state"),
                            "health_score": top.get("health_score"),
                            "recommendation": (
                                f"Rebalance the source mix by giving {source_name} a near-term boost."
                            ),
                            "control_operation": "source-boost-next",
                            "control_value": 2,
                            "source_plan_origin": "mix_advisory",
                            "mix_reason": "mix_too_narrow",
                        }
                    )
        return candidates

    tier_rank = {
        "policy-align": 0,
        "mix-rebalance": 1,
        "do-now": 2,
        "queue-next": 3,
        "watch": 4,
        "leave-alone": 5,
    }
    applied: list[dict[str, Any]] = []
    updated_control = control
    mix_candidates = _build_mix_candidates()
    summary["mix_candidates_considered"] = mix_candidates
    candidates = sorted(
        (brief.get("source_advisory") or []) + mix_candidates,
        key=lambda item: (
            tier_rank.get(item.get("tier"), 99),
            -(item.get("health_score") or -1),
            item.get("source") or "",
        ),
    )
    for item in candidates:
        operation = item.get("control_operation")
        source = item.get("source")
        if operation not in {"source-force-next", "source-boost-next"}:
            continue
        if not source:
            continue
        health_score = item.get("health_score")
        if (
            isinstance(health_score, (int, float))
            and health_score < parsed.auto_source_plan_min_health
        ):
            continue
        updated_control, changed = _apply_control_operation_to_payload(
            updated_control,
            operation=operation,
            source=str(source),
            value=item.get("control_value"),
            expires_after_cycles=parsed.auto_source_plan_expires_after_cycles,
        )
        if not changed:
            continue
        action = {
            "source": source,
            "operation": operation,
            "value": item.get("control_value"),
            "tier": item.get("tier"),
            "health_score": health_score,
            "recommendation": item.get("recommendation"),
            "source_plan_origin": item.get("source_plan_origin") or "source_advisory",
            "mix_reason": item.get("mix_reason"),
        }
        applied.append(action)
        _append_control_event(
            daemon_dir,
            {
                "type": "auto_source_plan_applied",
                "matched_key": source,
                "operation": operation,
                "value": item.get("control_value"),
                "health_score": health_score,
                "tier": item.get("tier"),
                "expires_after_cycles": parsed.auto_source_plan_expires_after_cycles,
            },
        )
        if len(applied) >= max(1, int(summary.get("max_actions") or 1)):
            break

    if applied:
        _save_control_payload(daemon_dir, updated_control)
        status["control"] = updated_control
        status["last_auto_source_plan"] = {
            "generated_at": _now_iso(),
            "items": applied,
        }
        summary["applied"] = applied
    else:
        summary["skipped_reason"] = (
            "no eligible high-health source action met auto-apply criteria"
        )
    brief["auto_source_plan"] = summary
    return brief


def _build_submission_autopilot_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Submission Autopilot", ""]
    lines.append(f"- Enabled: {summary.get('enabled')}")
    lines.append(f"- Top K: {summary.get('top_k')}")
    lines.append(f"- Require Gate: {summary.get('require_gate')}")
    lines.append(f"- Require Ready: {summary.get('require_ready')}")
    lines.append(f"- Min Priority: {summary.get('min_priority')}")
    lines.append(f"- Max Blockers: {summary.get('max_blockers')}")
    lines.append(f"- Min Rewrite Gain: {summary.get('min_rewrite_gain')}")
    if summary.get("skipped_reason"):
        lines.append(f"- Skipped: {summary.get('skipped_reason')}")
    lines.extend(["", "## Exported"])
    if summary.get("exported"):
        for item in summary.get("exported", [])[:10]:
            lines.append(
                f"- {item.get('folder')} | priority={item.get('priority')} | blockers={item.get('blockers')} | dossier={item.get('output_dir')}"
            )
    else:
        lines.append("- No dossiers exported in this cycle.")
    lines.extend(["", "## Reused"])
    if summary.get("reused"):
        for item in summary.get("reused", [])[:10]:
            lines.append(
                f"- {item.get('folder')} | priority={item.get('priority')} | dossier={item.get('output_dir')}"
            )
    else:
        lines.append("- No existing dossiers reused.")
    return "\n".join(lines) + "\n"


def _apply_submission_autopilot(
    status: dict[str, Any],
    daemon_dir: Path,
    manager: ResearchManager | None,
    brief: dict[str, Any],
    parsed: argparse.Namespace,
) -> dict[str, Any]:
    min_priority = (
        parsed.auto_submission_dossier_min_priority
        if parsed.auto_submission_dossier_min_priority is not None
        else parsed.shortlist_min_priority
    )
    max_blockers = (
        parsed.auto_submission_dossier_max_blockers
        if parsed.auto_submission_dossier_max_blockers is not None
        else parsed.shortlist_max_blockers
    )
    min_rewrite_gain = (
        parsed.auto_submission_dossier_min_rewrite_gain
        if parsed.auto_submission_dossier_min_rewrite_gain is not None
        else parsed.shortlist_min_rewrite_gain
    )
    governor = status.get("quality_governor") or {}
    effective_enabled = bool(
        governor.get(
            "auto_submission_dossier_enabled", parsed.auto_export_submission_dossier
        )
    )
    effective_top_k = int(
        governor.get(
            "auto_submission_dossier_top_k_effective",
            parsed.auto_submission_dossier_top_k,
        )
    )
    summary = {
        "enabled": effective_enabled,
        "top_k": effective_top_k,
        "require_gate": parsed.auto_submission_dossier_require_gate,
        "require_ready": parsed.auto_submission_dossier_require_ready,
        "min_priority": min_priority,
        "max_blockers": max_blockers,
        "min_rewrite_gain": min_rewrite_gain,
        "exported": [],
        "reused": [],
        "skipped_reason": None,
        "output_root": str(daemon_dir / "submission_autopilot"),
    }
    if not effective_enabled:
        brief["submission_autopilot"] = summary
        return brief
    if manager is None:
        summary["skipped_reason"] = "manager unavailable"
        brief["submission_autopilot"] = summary
        return brief

    candidates = manager.shortlist_papers(
        target_venue=parsed.shortlist_venue,
        require_gate=parsed.auto_submission_dossier_require_gate,
        require_ready=parsed.auto_submission_dossier_require_ready,
        min_priority=min_priority,
        max_blockers=max_blockers,
        min_rewrite_gain=min_rewrite_gain,
        top_n=effective_top_k,
    )
    if not candidates:
        summary["skipped_reason"] = (
            "no shortlist papers met submission autopilot criteria"
        )
        brief["submission_autopilot"] = summary
        _safe_write_json(daemon_dir / "latest_submission_autopilot.json", summary)
        (daemon_dir / "latest_submission_autopilot.md").write_text(
            _build_submission_autopilot_markdown(summary), encoding="utf-8"
        )
        status["last_submission_autopilot"] = summary
        return brief

    previous = _safe_read_json(daemon_dir / "latest_submission_autopilot.json")
    previous_items = {
        item.get("folder"): item
        for key in ["exported", "reused"]
        for item in (previous.get(key) or [])
        if item.get("folder")
    }
    output_root = daemon_dir / "submission_autopilot"
    output_root.mkdir(parents=True, exist_ok=True)

    for paper in candidates:
        folder = paper.get("folder")
        if not folder:
            continue
        dossier_dir = output_root / folder
        manifest_path = dossier_dir / "dossier_manifest.json"
        record = {
            "folder": folder,
            "name": paper.get("name"),
            "priority": paper.get("submission_priority_score"),
            "blockers": paper.get("blocker_count"),
            "rewrite_gain": paper.get("rewrite_priority_gain_total"),
            "submission_status": paper.get("submission_status"),
            "modified_at": paper.get("modified_at"),
            "output_dir": str(dossier_dir),
        }
        previous_record = previous_items.get(folder) or {}
        if manifest_path.exists() and previous_record.get("modified_at") == paper.get(
            "modified_at"
        ):
            summary["reused"].append(record | {"manifest": str(manifest_path)})
            continue
        result = manager.export_submission_dossier(folder, str(dossier_dir))
        if result.get("status") == "success":
            summary["exported"].append(record | {"manifest": result.get("manifest")})
        else:
            summary.setdefault("errors", []).append(
                {"folder": folder, "reason": result.get("reason")}
            )

    if (
        not summary["exported"]
        and not summary["reused"]
        and not summary.get("skipped_reason")
    ):
        summary["skipped_reason"] = (
            "submission dossier export produced no usable artifacts"
        )
    _safe_write_json(daemon_dir / "latest_submission_autopilot.json", summary)
    (daemon_dir / "latest_submission_autopilot.md").write_text(
        _build_submission_autopilot_markdown(summary), encoding="utf-8"
    )
    status["last_submission_autopilot"] = summary
    brief["submission_autopilot"] = summary
    return brief


def _build_failure_guard_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Failure Guard", ""]
    lines.append(f"- Enabled: {summary.get('enabled')}")
    lines.append(f"- Source: {summary.get('source')}")
    lines.append(f"- Consecutive Failures: {summary.get('consecutive_failures')}")
    lines.append(f"- Threshold: {summary.get('threshold')}")
    lines.append(f"- Cooldown Cycles: {summary.get('cooldown_cycles')}")
    lines.append(f"- Applied: {summary.get('applied')}")
    if summary.get("reason"):
        lines.append(f"- Reason: {summary.get('reason')}")
    return "\n".join(lines) + "\n"


def _apply_failure_guard(
    status: dict[str, Any], daemon_dir: Path, parsed: argparse.Namespace
) -> dict[str, Any]:
    active_source = status.get("active_source") or {}
    active_key = status.get("active_source_key")
    source_name = active_source.get("name") or active_source.get("value") or active_key
    runtime = (
        ((status.get("source_runtime") or {}).get(active_key) or {})
        if active_key
        else {}
    )
    consecutive_failures = int(runtime.get("consecutive_failures", 0) or 0)
    summary = {
        "enabled": bool(parsed.auto_failure_guard),
        "source": source_name,
        "consecutive_failures": consecutive_failures,
        "threshold": parsed.auto_failure_guard_threshold,
        "cooldown_cycles": parsed.auto_failure_guard_cooldown_cycles,
        "applied": False,
        "reason": None,
    }
    if not parsed.auto_failure_guard:
        status["last_failure_guard"] = summary
        return status
    if status.get("last_returncode") in (0, None):
        summary["reason"] = "last cycle did not fail"
        status["last_failure_guard"] = summary
        return status
    if not source_name:
        summary["reason"] = "no active source recorded for failure guard"
        status["last_failure_guard"] = summary
        return status
    if consecutive_failures < parsed.auto_failure_guard_threshold:
        summary["reason"] = (
            f"consecutive failures below threshold ({consecutive_failures} < {parsed.auto_failure_guard_threshold})"
        )
        status["last_failure_guard"] = summary
        return status

    control = _load_control_payload(
        daemon_dir, current_cycle=int(status.get("cycle", 0) or 0)
    )
    updated_control, changed = _apply_control_operation_to_payload(
        control,
        operation="source-cooldown-once",
        source=str(source_name),
        value=parsed.auto_failure_guard_cooldown_cycles,
        expires_after_cycles=1,
    )
    if changed:
        _save_control_payload(daemon_dir, updated_control)
        status["control"] = updated_control
        summary["applied"] = True
        summary["reason"] = "queued one-shot cooldown for repeatedly failing source"
        _append_control_event(
            daemon_dir,
            {
                "type": "auto_failure_guard_applied",
                "matched_key": source_name,
                "cooldown_cycles_once": parsed.auto_failure_guard_cooldown_cycles,
                "consecutive_failures": consecutive_failures,
            },
        )
    else:
        summary["reason"] = "failure guard command already queued or produced no change"
    status["last_failure_guard"] = summary
    _safe_write_json(daemon_dir / "latest_failure_guard.json", summary)
    (daemon_dir / "latest_failure_guard.md").write_text(
        _build_failure_guard_markdown(summary), encoding="utf-8"
    )
    return status


def _write_operator_brief(
    status: dict[str, Any],
    daemon_dir: Path,
    manager: ResearchManager | None,
    parsed: argparse.Namespace,
) -> dict[str, Any]:
    if manager is None:
        brief = {
            "generated_at": _now_iso(),
            "health": _compute_health_snapshot(status),
            "guardrail_phase": status.get("guardrail_phase"),
            "guardrail_mode": status.get("guardrail_mode"),
            "guardrail_reason": status.get("guardrail_reason"),
            "success_count": status.get("success_count"),
            "failure_count": status.get("failure_count"),
            "top_submission_targets": [],
            "top_rewrite_targets": [],
            "recent_followup_wins": [],
            "priorities": ["Daemon initialized; no live manager data yet."],
            "actions": ["Run at least one cycle to populate the operator brief."],
            "blockers": [],
            "counts": {},
            "failure_hotspots": [],
            "rewrite_style_hotspots": [],
            "source_actions": [
                {
                    "name": row.get("name"),
                    "availability_state": row.get("availability_state"),
                    "health_score": row.get("health_score"),
                    "suggested_action": row.get("suggested_action"),
                }
                for row in _build_source_runtime_rows(parsed, status)[:5]
            ],
            "source_advisory": _build_source_advisory_rows(
                _build_source_runtime_rows(parsed, status),
                status.get("control") or {},
                daemon_dir,
            ),
            "source_mix_advisory": {
                "desired_policy": None,
                "summary": {
                    "source_count": 0,
                    "archetype_counts": {},
                    "workflow_mode_counts": {},
                    "batch_profile_counts": {},
                    "dominant_archetype": None,
                    "dominant_workflow_mode": None,
                },
                "top_sources": [],
                "recommendations": [],
            },
            "source_next_batch_advisory": {
                "desired_policy": None,
                "summary": {
                    "source_count": 0,
                    "slot_count": 0,
                    "dominant_archetype": None,
                    "dominant_workflow_mode": None,
                },
                "cadence": {
                    "label": "no_sources",
                    "reason": "Run at least one cycle to build source lineage before orchestrating the next batch.",
                },
                "slots": [],
                "recommendations": [],
            },
            "do_now_actions": _select_do_now_actions(
                _build_source_advisory_rows(
                    _build_source_runtime_rows(parsed, status),
                    status.get("control") or {},
                    daemon_dir,
                )
            ),
            "control_summary": _summarize_control_state(status.get("control") or {}),
            "recent_control_events": _load_recent_control_events(
                daemon_dir, max_entries=5
            ),
            "failure_guard": status.get("last_failure_guard") or {},
            "quality_governor": status.get("quality_governor") or {},
            "evidence_strategy": status.get("active_evidence_strategy") or {},
            "recommended_commands": {
                "next_cycle_command": _command_preview_for_next_cycle(parsed, status),
                "pipeline_status_command": _manager_command_preview(
                    parsed, "pipeline-status"
                ),
                "fallback_board_command": _manager_command_preview(
                    parsed, "fallback-board"
                ),
                "source_board_command": _manager_command_preview(
                    parsed, "source-board"
                ),
                "source_mix_command": _manager_command_preview(parsed, "source-mix"),
                "source_next_batch_command": _manager_command_preview(
                    parsed, "source-next-batch"
                ),
                "idea_board_command": _manager_command_preview(parsed, "idea-board"),
                "experiment_board_command": _manager_command_preview(
                    parsed, "experiment-board"
                ),
                "figure_board_command": _manager_command_preview(
                    parsed, "figure-board"
                ),
                "evolution_board_command": _manager_command_preview(
                    parsed, "evolution-board"
                ),
                "submission_board_command": _manager_command_preview(
                    parsed, "submission-board"
                ),
                "rewrite_board_command": _manager_command_preview(
                    parsed, "rewrite-board"
                ),
                "shortlist_command": _manager_command_preview(parsed, "shortlist"),
                "source_summary_command": _stable_wrapper_command_preview(
                    daemon_dir, "source-summary", "--lines", "10"
                ),
                "source_plan_command": _stable_wrapper_command_preview(
                    daemon_dir, "source-plan"
                ),
                "rewrite_followup_mode": (
                    "enabled" if parsed.enable_rewrite_followup else "disabled"
                ),
            },
        }
    else:
        brief = _build_operator_brief(manager, parsed, status)
    if "recovery_command" not in brief:
        brief.update(_build_handoff_recovery(status, brief, daemon_dir))
    if "primary_action_queue" not in brief:
        brief["primary_action_queue"] = _build_primary_action_queue(
            do_now_actions=brief.get("do_now_actions") or [],
            recovery_command=brief.get("recovery_command"),
            recovery_reason=brief.get("recovery_reason"),
            recommended_commands=brief.get("recommended_commands") or {},
        )
    brief["quality_governor"] = status.get("quality_governor") or {}
    brief["evidence_strategy"] = status.get("active_evidence_strategy") or {}
    brief = _apply_auto_source_plan(status, daemon_dir, brief, parsed)
    brief = _apply_submission_autopilot(status, daemon_dir, manager, brief, parsed)
    brief["failure_guard"] = status.get("last_failure_guard") or {}
    _safe_write_json(daemon_dir / "latest_operator_brief.json", brief)
    (daemon_dir / "latest_operator_brief.md").write_text(
        _build_operator_brief_markdown(brief), encoding="utf-8"
    )
    _safe_write_json(
        daemon_dir / "latest_source_next_batch.json",
        brief.get("source_next_batch_advisory") or {},
    )
    (daemon_dir / "latest_source_next_batch.md").write_text(
        _build_source_next_batch_markdown(
            brief.get("source_next_batch_advisory") or {}
        ),
        encoding="utf-8",
    )
    return brief


def _write_cycle_summary(status: dict[str, Any], daemon_dir: Path) -> dict[str, Any]:
    summary = _build_cycle_summary(status)
    _safe_write_json(daemon_dir / "latest_cycle_summary.json", summary)
    (daemon_dir / "latest_cycle_summary.md").write_text(
        _build_cycle_summary_markdown(summary), encoding="utf-8"
    )
    return summary


def _write_daily_summary(status: dict[str, Any], daemon_dir: Path) -> dict[str, Any]:
    summary = _build_cycle_summary(status)
    summary["type"] = "daily_summary"
    _safe_write_json(daemon_dir / "latest_daily_summary.json", summary)
    (daemon_dir / "latest_daily_summary.md").write_text(
        _build_cycle_summary_markdown(summary), encoding="utf-8"
    )
    return summary


def _default_daemon_status(
    parsed: argparse.Namespace, daemon_dir: Path
) -> dict[str, Any]:
    return {
        "daemon_name": parsed.daemon_name,
        "daemon_dir": str(daemon_dir),
        "started_at": _now_iso(),
        "updated_at": _now_iso(),
        "state": "initialized",
        "cycle": 0,
        "success_count": 0,
        "failure_count": 0,
        "last_cycle_started_at": None,
        "last_cycle_finished_at": None,
        "last_returncode": None,
        "last_error": None,
        "next_cycle_at": None,
        "last_views": {},
        "last_rewrite_followup": {},
        "source_index": 0,
        "active_source": None,
        "active_source_key": None,
        "source_queue": [],
        "source_runtime": {},
        "last_cycle_summary_at": None,
        "last_daily_summary_at": None,
        "last_operator_brief_at": None,
        "dashboard_refresh_seconds": parsed.dashboard_refresh_seconds,
        "dashboard_url": None,
        "dashboard_server": parsed.serve_dashboard,
        "control": _default_control_payload(),
        "guardrail_phase": "cold_start",
        "guardrail_phase_reason": "initial startup",
        "guardrail_mode": "balanced",
        "guardrail_reason": "initial state",
        "guardrail_followup_metrics": {},
        "consecutive_low_uplift_cycles": 0,
        "consecutive_strong_submission_cycles": 0,
        "consecutive_empty_rewrite_cycles": 0,
        "generator_args": _clean_generator_args(parsed.generator_args),
        "last_auto_source_plan": {},
        "last_submission_autopilot": {},
        "last_failure_guard": {},
        "source_quality_feedback": {},
        "active_source_feedback_snapshot": {},
        "quality_governor": {},
        "active_evidence_strategy": {},
    }


def _run_cycle(
    parsed: argparse.Namespace, status: dict[str, Any], daemon_dir: Path
) -> int:
    cycle_index = int(status.get("cycle", 0)) + 1
    status["cycle"] = cycle_index
    status["state"] = "running_cycle"
    status["last_cycle_started_at"] = _now_iso()
    status["updated_at"] = _now_iso()

    cycle_label = f"cycle_{cycle_index:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_name = f"{parsed.daemon_name}_{cycle_label}"
    cycle_log = daemon_dir / "cycles" / f"{cycle_label}.log"
    status = _update_guardrail_state(parsed, status)
    cmd = _build_generator_command(parsed, batch_name, status)
    status["current_cycle_log"] = str(cycle_log)
    status["last_cycle_command"] = cmd

    status_path = daemon_dir / "daemon_status.json"
    history_path = daemon_dir / "cycle_history.jsonl"
    heartbeat_log = daemon_dir / "heartbeat.log"
    _safe_write_json(status_path, status)
    _append_log(heartbeat_log, f"starting cycle {cycle_index}: {' '.join(cmd)}")

    if parsed.dry_run:
        _append_jsonl(
            history_path,
            {
                "cycle": cycle_index,
                "started_at": status["last_cycle_started_at"],
                "finished_at": _now_iso(),
                "returncode": 0,
                "mode": "dry_run",
                "command": cmd,
                "guardrail_mode": status.get("guardrail_mode"),
                "guardrail_reason": status.get("guardrail_reason"),
            },
        )
        return 0

    cycle_log.parent.mkdir(parents=True, exist_ok=True)
    child_env = os.environ.copy()
    active_source = status.get("active_source") or {}
    child_env["AI_SCIENTIST_DAEMON_NAME"] = str(parsed.daemon_name)
    child_env["AI_SCIENTIST_BATCH_NAME"] = str(batch_name)
    if active_source.get("name") is not None:
        child_env["AI_SCIENTIST_SOURCE_NAME"] = str(active_source.get("name"))
    if status.get("active_source_key") is not None:
        child_env["AI_SCIENTIST_SOURCE_KEY"] = str(status.get("active_source_key"))
    if active_source.get("type") is not None:
        child_env["AI_SCIENTIST_SOURCE_TYPE"] = str(active_source.get("type"))
    if active_source.get("value") is not None:
        child_env["AI_SCIENTIST_SOURCE_VALUE"] = str(active_source.get("value"))
    source_plan = build_source_planning_profile(
        active_source,
        daypart=status.get("current_daypart") or _current_daypart(parsed),
        desired_execution_policy=(
            (status.get("active_pipeline_contract_strategy") or {}).get(
                "selected_execution_policy"
            )
            or (status.get("pipeline_contract_summary") or {}).get(
                "dominant_execution_policy"
            )
        ),
    )
    if source_plan.get("resolved_workflow_mode") is not None:
        child_env["AI_SCIENTIST_SOURCE_WORKFLOW_MODE"] = str(
            source_plan.get("resolved_workflow_mode")
        )
    if source_plan.get("source_archetype") is not None:
        child_env["AI_SCIENTIST_SOURCE_ARCHETYPE"] = str(
            source_plan.get("source_archetype")
        )
    if source_plan.get("batch_profile") is not None:
        child_env["AI_SCIENTIST_SOURCE_BATCH_PROFILE"] = str(
            source_plan.get("batch_profile")
        )
    target_venue = active_source.get(
        f"{status.get('current_daypart')}_target_venue"
    ) or active_source.get("target_venue")
    if target_venue is not None:
        child_env["AI_SCIENTIST_SOURCE_TARGET_VENUE"] = str(target_venue)
    paper_types = (
        active_source.get(f"{status.get('current_daypart')}_paper_types")
        or active_source.get("paper_types")
        or []
    )
    if isinstance(paper_types, str):
        paper_types = [paper_types]
    child_env["AI_SCIENTIST_SOURCE_PAPER_TYPES"] = ",".join(
        str(item) for item in paper_types
    )
    started = time.time()
    with open(cycle_log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"# Command\n{' '.join(cmd)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=child_env,
        )
        try:
            timeout_seconds = (
                parsed.cycle_timeout_minutes * 60
                if parsed.cycle_timeout_minutes
                else None
            )
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            returncode = 124
            log_handle.write(
                f"\n[daemon] cycle timed out after {parsed.cycle_timeout_minutes} minutes\n"
            )
        duration_seconds = round(time.time() - started, 2)

    manager = ResearchManager(parsed.research_dir)
    manager.rebuild_index()
    status["source_quality_feedback"] = _build_source_quality_feedback(manager, parsed)
    history_feedback = build_active_source_feedback_snapshot(status)
    status["active_source_feedback_snapshot"] = history_feedback
    views = _export_cycle_views(manager, parsed, daemon_dir)
    status["last_views"] = views
    _append_jsonl(
        history_path,
        {
            "cycle": cycle_index,
            "started_at": status["last_cycle_started_at"],
            "finished_at": _now_iso(),
            "duration_seconds": duration_seconds,
            "returncode": returncode,
            "command": cmd,
            "batch_name": batch_name,
            "cycle_log": str(cycle_log),
            "views": views,
            "guardrail_mode": status.get("guardrail_mode"),
            "guardrail_reason": status.get("guardrail_reason"),
            "active_source_feedback": history_feedback,
        },
    )
    runtime = _refresh_source_runtime_state(status, status.get("source_queue") or [])
    active_key = status.get("active_source_key")
    if active_key and active_key in runtime:
        runtime[active_key]["total_cycles"] = (
            int(runtime[active_key].get("total_cycles", 0) or 0) + 1
        )
        runtime[active_key]["cycles_today"] = (
            int(runtime[active_key].get("cycles_today", 0) or 0) + 1
        )
        runtime[active_key]["last_finished_at"] = _now_iso()
        if returncode == 0:
            runtime[active_key]["total_successes"] = (
                int(runtime[active_key].get("total_successes", 0) or 0) + 1
            )
            runtime[active_key]["successes_today"] = (
                int(runtime[active_key].get("successes_today", 0) or 0) + 1
            )
            runtime[active_key]["consecutive_failures"] = 0
        else:
            runtime[active_key]["consecutive_failures"] = (
                int(runtime[active_key].get("consecutive_failures", 0) or 0) + 1
            )
        active_source = status.get("active_source") or {}
        cooldown_cycles = int(active_source.get("cooldown_cycles", 0) or 0)
        command = active_source.get("source_command") or {}
        if command.get("cooldown_cycles_once") is not None:
            try:
                cooldown_cycles = max(
                    cooldown_cycles, int(command.get("cooldown_cycles_once") or 0)
                )
            except (TypeError, ValueError):
                pass
        if cooldown_cycles > 0:
            runtime[active_key]["cooldown_until_cycle"] = cycle_index + cooldown_cycles
    matched_key = status.get("active_source_command_key")
    if matched_key:
        _consume_source_command(daemon_dir, status, matched_key)
        status["active_source_command_key"] = None
    _append_log(
        heartbeat_log,
        f"finished cycle {cycle_index} with returncode={returncode}; views={views}",
    )
    if parsed.source_rotation == "round_robin" and status.get("source_queue"):
        status["source_index"] = (int(status.get("source_index", 0)) + 1) % len(
            status.get("source_queue", [])
        )
    return returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Long-running daemon for continuous AI Scientist paper generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python continuous_research_daemon.py \
    --topic example_topic.md \
    --duration-hours 24 \
    --sleep-minutes 10 \
    -- --paper-types normal journal --num-ideas 4 --submission-mode
        """,
    )
    parser.add_argument(
        "--research-dir",
        type=str,
        default=str(resolve_output_path()),
        help="Research output directory",
    )
    parser.add_argument(
        "--daemon-name",
        type=str,
        default=datetime.now().strftime("daemon_%Y%m%d_%H%M%S"),
        help="Stable daemon run name",
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Topic markdown file passed to continuous_paper_generator.py",
    )
    parser.add_argument(
        "--source-config",
        type=str,
        default=None,
        help="JSON or TOML source queue config with priorities and per-source overrides",
    )
    parser.add_argument(
        "--topic-files",
        type=str,
        nargs="*",
        default=[],
        help="Additional topic markdown files to rotate through",
    )
    parser.add_argument(
        "--ideas-files",
        type=str,
        nargs="*",
        default=[],
        help="Additional ideas JSON files to rotate through",
    )
    parser.add_argument(
        "--source-rotation",
        choices=["fixed", "round_robin"],
        default="fixed",
        help="How to select among multiple topic/ideas sources",
    )
    parser.add_argument(
        "--ideas",
        type=str,
        help="Existing ideas JSON passed to continuous_paper_generator.py",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=24.0,
        help="How long the daemon should run before stopping",
    )
    parser.add_argument(
        "--run-forever",
        action="store_true",
        help="Ignore duration and keep cycling until interrupted",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None, help="Optional hard limit on cycles"
    )
    parser.add_argument(
        "--sleep-minutes",
        type=float,
        default=10.0,
        help="Sleep time after a successful cycle",
    )
    parser.add_argument(
        "--failure-backoff-minutes",
        type=float,
        default=20.0,
        help="Sleep time after a failed cycle",
    )
    parser.add_argument(
        "--cycle-timeout-minutes",
        type=float,
        default=None,
        help="Optional timeout for one generator cycle",
    )
    parser.add_argument(
        "--auto-failure-guard",
        action="store_true",
        help="Automatically queue a one-shot source cooldown after repeated failures",
    )
    parser.add_argument(
        "--auto-failure-guard-threshold",
        type=int,
        default=2,
        help="Consecutive failures required before auto-queueing a source cooldown",
    )
    parser.add_argument(
        "--auto-failure-guard-cooldown-cycles",
        type=int,
        default=2,
        help="One-shot cooldown cycles applied by the automatic failure guard",
    )
    parser.add_argument(
        "--day-start-hour", type=int, default=8, help="Local hour when day mode starts"
    )
    parser.add_argument(
        "--night-start-hour",
        type=int,
        default=20,
        help="Local hour when night mode starts",
    )
    parser.add_argument(
        "--enable-rewrite-followup",
        action="store_true",
        help="After each cycle, rerun high-quality improvement on top rewrite-board papers",
    )
    parser.add_argument(
        "--rewrite-followup-top-k",
        type=int,
        default=0,
        help="How many rewrite-board papers to revisit after each cycle",
    )
    parser.add_argument(
        "--rewrite-followup-include-ready",
        action="store_true",
        help="Allow rewrite follow-up for ready papers as well",
    )
    parser.add_argument(
        "--rewrite-followup-model",
        type=str,
        default=None,
        help="Rewrite model used for follow-up passes",
    )
    parser.add_argument(
        "--rewrite-followup-quality-model",
        type=str,
        default=None,
        help="Quality model used for follow-up passes",
    )
    parser.add_argument(
        "--rewrite-followup-preset",
        choices=["balanced", "high", "publishable"],
        default="publishable",
    )
    parser.add_argument("--rewrite-followup-max-rounds", type=int, default=None)
    parser.add_argument(
        "--rewrite-followup-quality-threshold", type=float, default=None
    )
    parser.add_argument("--rewrite-followup-rigor-threshold", type=float, default=None)
    parser.add_argument(
        "--adaptive-rewrite-followup",
        action="store_true",
        default=True,
        help="Adapt rewrite preset and rounds per paper during follow-up",
    )
    parser.add_argument(
        "--no-adaptive-rewrite-followup",
        dest="adaptive_rewrite_followup",
        action="store_false",
    )
    parser.add_argument(
        "--rewrite-followup-skip-blocker-threshold",
        type=int,
        default=6,
        help="Skip rewrite follow-up when blocker count is at or above this threshold",
    )
    parser.add_argument(
        "--rewrite-followup-blocker-reduction-threshold",
        type=int,
        default=4,
        help="Use blocker-reduction rewrite policy when blocker count is at or above this threshold",
    )
    parser.add_argument(
        "--rewrite-followup-ready-max-rounds",
        type=int,
        default=1,
        help="Maximum rewrite rounds for near-ready papers during adaptive follow-up",
    )
    parser.add_argument(
        "--rewrite-followup-publishable-priority-threshold",
        type=float,
        default=85.0,
        help="Escalate to publishable rewrite policy once submission priority reaches this threshold",
    )
    parser.add_argument(
        "--rewrite-followup-publishable-gain-threshold",
        type=float,
        default=1.25,
        help="Escalate to publishable rewrite policy once rewrite gain reaches this threshold",
    )
    parser.add_argument(
        "--rewrite-followup-self-review-gate-score-floor",
        type=float,
        default=75.0,
        help="Escalate to evidence-gap repair mode when self-review round gate score falls below this floor",
    )
    parser.add_argument(
        "--rewrite-followup-self-review-high-value-coverage-floor",
        type=float,
        default=0.7,
        help="Escalate to evidence-gap repair mode when self-review high-value coverage falls below this floor",
    )
    parser.add_argument(
        "--rewrite-followup-self-review-min-rounds",
        type=int,
        default=2,
        help="Minimum rewrite rounds when self-review gate indicates unresolved evidence gaps",
    )
    parser.add_argument(
        "--rewrite-followup-experiment-todo-p0-floor",
        type=int,
        default=1,
        help="Escalate rewrite follow-up to evidence-gap repair when P0 experiment TODO count reaches this floor",
    )
    parser.add_argument(
        "--rewrite-followup-experiment-todo-count-floor",
        type=int,
        default=4,
        help="Escalate rewrite follow-up when total open experiment TODO count reaches this floor",
    )
    parser.add_argument(
        "--rewrite-followup-experiment-todo-min-rounds",
        type=int,
        default=2,
        help="Minimum rewrite rounds under experiment TODO pressure",
    )
    parser.add_argument(
        "--rewrite-followup-experiment-todo-closure-floor",
        type=float,
        default=0.35,
        help="Escalate rewrite follow-up when experiment TODO closure rate falls below this floor",
    )
    parser.add_argument(
        "--guardrail-submission-target",
        type=int,
        default=5,
        help="Switch toward rewrite focus once submission-board reaches this many items",
    )
    parser.add_argument(
        "--guardrail-min-followup-gain",
        type=float,
        default=0.25,
        help="Treat rewrite follow-up as stagnant if avg priority delta falls below this value",
    )
    parser.add_argument(
        "--guardrail-stagnation-cycles",
        type=int,
        default=2,
        help="Cycles of weak rewrite uplift before biasing back toward generation",
    )
    parser.add_argument(
        "--guardrail-empty-rewrite-cycles",
        type=int,
        default=2,
        help="Cycles with an empty rewrite-board before biasing back toward generation",
    )
    parser.add_argument(
        "--guardrail-strong-cycles",
        type=int,
        default=2,
        help="Cycles with enough strong drafts before biasing toward rewrite focus",
    )
    parser.add_argument(
        "--guardrail-default-num-ideas",
        type=int,
        default=3,
        help="Fallback num-ideas if passthrough args do not specify one",
    )
    parser.add_argument(
        "--guardrail-num-ideas-step",
        type=int,
        default=2,
        help="How much to raise/lower num-ideas when switching modes",
    )
    parser.add_argument("--guardrail-min-num-ideas", type=int, default=1)
    parser.add_argument("--guardrail-max-num-ideas", type=int, default=12)
    parser.add_argument("--guardrail-generate-sleep-minutes", type=float, default=5.0)
    parser.add_argument("--guardrail-focus-sleep-minutes", type=float, default=20.0)
    parser.add_argument(
        "--phase-warmup-cycles",
        type=int,
        default=2,
        help="Number of initial cycles treated as cold-start exploration",
    )
    parser.add_argument(
        "--phase-cold-start-submission-target",
        type=int,
        default=2,
        help="Remain in cold-start until at least this many submission-board items exist",
    )
    parser.add_argument(
        "--phase-hot-polish-submission-target",
        type=int,
        default=6,
        help="Enter hot-polish when submission-board has at least this many items",
    )
    parser.add_argument(
        "--summary-every-cycles",
        type=int,
        default=1,
        help="Write a cycle summary every N cycles",
    )
    parser.add_argument(
        "--daily-summary-every-cycles",
        type=int,
        default=6,
        help="Refresh the daily summary every N cycles",
    )
    parser.add_argument(
        "--dashboard-refresh-seconds",
        type=int,
        default=30,
        help="Auto-refresh interval for the generated live dashboard HTML",
    )
    parser.add_argument(
        "--auto-apply-source-plan",
        action="store_true",
        help="Automatically queue the top healthy source-plan action into daemon_control.json",
    )
    parser.add_argument(
        "--auto-source-plan-max-actions",
        type=int,
        default=1,
        help="Maximum source-plan actions to auto-queue per cycle",
    )
    parser.add_argument(
        "--auto-source-plan-min-health",
        type=float,
        default=88.0,
        help="Minimum source health score required before auto-queuing a source-plan action",
    )
    parser.add_argument(
        "--auto-source-plan-expires-after-cycles",
        type=int,
        default=1,
        help="Expire auto-queued source-plan commands after this many cycles",
    )
    parser.add_argument(
        "--auto-source-quality-feedback",
        action="store_true",
        help="Adjust source priority using historical paper quality outcomes linked to each source",
    )
    parser.add_argument(
        "--source-quality-feedback-min-papers",
        type=int,
        default=1,
        help="Minimum papers attributed to a source before applying quality feedback",
    )
    parser.add_argument(
        "--source-quality-feedback-max-boost",
        type=float,
        default=4.0,
        help="Maximum positive priority bonus contributed by source quality feedback",
    )
    parser.add_argument(
        "--source-quality-feedback-max-penalty",
        type=float,
        default=2.0,
        help="Maximum negative priority penalty contributed by source quality feedback",
    )
    parser.add_argument(
        "--auto-quality-strategy-feedback",
        action="store_true",
        help="Adapt target venue, paper types, and idea count from source quality feedback",
    )
    parser.add_argument(
        "--quality-strategy-submission-priority-threshold",
        type=float,
        default=85.0,
        help="Treat a source as strong once avg submission priority reaches this threshold",
    )
    parser.add_argument(
        "--quality-strategy-ready-rate-threshold",
        type=float,
        default=0.25,
        help="Treat a source as strong once its ready-rate reaches this threshold",
    )
    parser.add_argument(
        "--quality-strategy-exploration-priority-ceiling",
        type=float,
        default=65.0,
        help="Treat a source as weak if its best priority stays below this ceiling",
    )
    parser.add_argument(
        "--quality-strategy-gate-pass-floor",
        type=float,
        default=0.25,
        help="Treat a source as weak if its gate-pass rate stays at or below this floor",
    )
    parser.add_argument(
        "--quality-strategy-max-num-ideas-for-strong-sources",
        type=int,
        default=2,
        help="Upper bound for num-ideas when a source is already yielding strong drafts",
    )
    parser.add_argument(
        "--quality-strategy-max-num-ideas-for-weak-sources",
        type=int,
        default=6,
        help="Upper bound for num-ideas when a source needs broader exploration",
    )
    parser.add_argument(
        "--auto-quality-governor",
        action="store_true",
        help="Use recent cycle health and trend signals to tune autopilot intensity",
    )
    parser.add_argument(
        "--quality-governor-recent-cycles",
        type=int,
        default=6,
        help="How many recent cycles the quality governor should inspect",
    )
    parser.add_argument(
        "--quality-governor-stabilize-health-threshold",
        type=float,
        default=55.0,
        help="Health score below which the quality governor enters stabilize mode",
    )
    parser.add_argument(
        "--quality-governor-exploit-followup-gain",
        type=float,
        default=0.5,
        help="Minimum follow-up gain required before the quality governor amplifies rewrite and dossier automation",
    )
    parser.add_argument(
        "--quality-governor-max-rewrite-top-k",
        type=int,
        default=2,
        help="Maximum rewrite follow-up top-k when the quality governor is exploiting strong quality signals",
    )
    parser.add_argument(
        "--quality-governor-max-dossier-top-k",
        type=int,
        default=2,
        help="Maximum submission dossier top-k when the quality governor is exploiting strong quality signals",
    )
    parser.add_argument(
        "--quality-governor-max-source-plan-actions",
        type=int,
        default=2,
        help="Maximum source-plan actions when the quality governor is exploiting strong quality signals",
    )
    parser.add_argument(
        "--quality-governor-experiment-todo-closure-floor",
        type=float,
        default=0.45,
        help="Trigger closure-repair governor mode when active-source TODO closure rate falls below this floor",
    )
    parser.add_argument(
        "--quality-governor-experiment-todo-p0-floor",
        type=float,
        default=0.5,
        help="Trigger closure-repair governor mode when active-source average P0 TODO backlog reaches this floor",
    )
    parser.add_argument(
        "--quality-governor-experiment-todo-count-floor",
        type=float,
        default=2.5,
        help="Trigger closure-repair governor mode when active-source average TODO backlog reaches this floor",
    )
    parser.add_argument(
        "--quality-strategy-dominant-venue-rate-threshold",
        type=float,
        default=0.5,
        help="Adopt a source's dominant historical venue once its share reaches this threshold",
    )
    parser.add_argument(
        "--quality-strategy-dominant-paper-type-rate-threshold",
        type=float,
        default=0.5,
        help="Adopt a source's dominant historical paper type once its share reaches this threshold",
    )
    parser.add_argument(
        "--auto-evidence-strategy-feedback",
        action="store_true",
        help="Deepen review and rewrite settings when evidence metrics are too weak",
    )
    parser.add_argument(
        "--evidence-strategy-claim-support-floor",
        type=float,
        default=3.6,
        help="Escalate evidence strategy when average claim support falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-numeric-coverage-floor",
        type=float,
        default=3.8,
        help="Escalate evidence strategy when average numeric coverage falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-evidence-density-floor",
        type=float,
        default=2.0,
        help="Escalate evidence strategy when average evidence density falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-claim-alignment-floor",
        type=float,
        default=3.2,
        help="Escalate evidence strategy when average claim alignment falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-unsupported-claims-ceiling",
        type=float,
        default=1.0,
        help="Escalate evidence strategy when average unsupported claims rises above this ceiling",
    )
    parser.add_argument(
        "--evidence-strategy-round-gate-ready-floor",
        type=float,
        default=0.55,
        help="Escalate evidence strategy when average self-review round-gate ready rate falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-high-value-coverage-floor",
        type=float,
        default=0.65,
        help="Escalate evidence strategy when average self-review high-value coverage falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-self-review-critical-ceiling",
        type=float,
        default=0.5,
        help="Escalate evidence strategy when average unresolved self-review critical count rises above this ceiling",
    )
    parser.add_argument(
        "--evidence-strategy-experiment-todo-ceiling",
        type=float,
        default=2.5,
        help="Escalate evidence strategy when average open experiment TODO count rises above this ceiling",
    )
    parser.add_argument(
        "--evidence-strategy-experiment-todo-p0-ceiling",
        type=float,
        default=0.5,
        help="Escalate evidence strategy when average P0 experiment TODO count rises above this ceiling",
    )
    parser.add_argument(
        "--evidence-strategy-experiment-todo-closure-floor",
        type=float,
        default=0.35,
        help="Escalate evidence strategy when average experiment TODO closure rate falls below this floor",
    )
    parser.add_argument(
        "--evidence-strategy-min-quality-rewrite-rounds",
        type=int,
        default=2,
        help="Minimum quality rewrite rounds enforced when the evidence strategy escalates",
    )
    parser.add_argument(
        "--evidence-strategy-todo-min-quality-rewrite-rounds",
        type=int,
        default=3,
        help="Minimum quality rewrite rounds when evidence strategy escalates under experiment TODO pressure",
    )
    parser.add_argument(
        "--evidence-strategy-max-num-ideas-under-todo-pressure",
        type=int,
        default=2,
        help="Upper bound for num-ideas when unresolved experiment TODO pressure is high",
    )
    parser.add_argument(
        "--evidence-strategy-review-strategy",
        type=str,
        default="depth",
        choices=[
            "standard",
            "fast",
            "depth",
            "neurips",
            "iclr",
            "cvpr",
            "journal",
            "nature",
        ],
        help="Review strategy forced when the evidence strategy escalates",
    )
    parser.add_argument(
        "--serve-dashboard",
        action="store_true",
        help="Serve the daemon directory over a local HTTP server",
    )
    parser.add_argument(
        "--dashboard-host",
        type=str,
        default="127.0.0.1",
        help="Host for the local dashboard server",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8000,
        help="Port for the local dashboard server (use 0 for auto)",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used for child runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not execute generator; only write the first planned command",
    )
    parser.add_argument(
        "--default-submission-mode",
        action="store_true",
        default=True,
        help="Automatically add --submission-mode if passthrough args do not set submission/breakthrough mode",
    )
    parser.add_argument(
        "--no-default-submission-mode",
        dest="default_submission_mode",
        action="store_false",
    )

    parser.add_argument("--submission-board-top", type=int, default=5)
    parser.add_argument("--submission-board-min-priority", type=float, default=75.0)
    parser.add_argument("--submission-board-max-blockers", type=int, default=2)
    parser.add_argument("--submission-board-min-rewrite-gain", type=float, default=0.0)
    parser.add_argument(
        "--submission-board-require-gate", action="store_true", default=True
    )
    parser.add_argument(
        "--no-submission-board-require-gate",
        dest="submission_board_require_gate",
        action="store_false",
    )

    parser.add_argument("--rewrite-board-top", type=int, default=10)
    parser.add_argument(
        "--rewrite-board-venue",
        type=str,
        default=None,
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
    )
    parser.add_argument("--rewrite-board-min-priority", type=float, default=70.0)
    parser.add_argument("--rewrite-board-min-gain", type=float, default=0.5)
    parser.add_argument("--rewrite-board-max-blockers", type=int, default=3)
    parser.add_argument(
        "--rewrite-board-require-gate", action="store_true", default=True
    )
    parser.add_argument(
        "--no-rewrite-board-require-gate",
        dest="rewrite_board_require_gate",
        action="store_false",
    )
    parser.add_argument(
        "--rewrite-board-include-ready",
        action="store_true",
        help="Include already ready papers in rewrite board",
    )

    parser.add_argument("--shortlist-top", type=int, default=10)
    parser.add_argument(
        "--shortlist-venue",
        type=str,
        default=None,
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
    )
    parser.add_argument("--shortlist-min-priority", type=float, default=75.0)
    parser.add_argument("--shortlist-max-blockers", type=int, default=2)
    parser.add_argument("--shortlist-min-rewrite-gain", type=float, default=0.0)
    parser.add_argument("--shortlist-require-gate", action="store_true", default=True)
    parser.add_argument(
        "--no-shortlist-require-gate",
        dest="shortlist_require_gate",
        action="store_false",
    )
    parser.add_argument("--shortlist-require-ready", action="store_true", default=False)
    parser.add_argument(
        "--auto-export-submission-dossier",
        action="store_true",
        help="Automatically export dossier bundles for the top shortlist papers",
    )
    parser.add_argument(
        "--auto-submission-dossier-top-k",
        type=int,
        default=1,
        help="Maximum shortlist papers to export into submission dossiers per cycle",
    )
    parser.add_argument(
        "--auto-submission-dossier-min-priority",
        type=float,
        default=None,
        help="Override the minimum priority required for automatic submission dossier export",
    )
    parser.add_argument(
        "--auto-submission-dossier-max-blockers",
        type=int,
        default=None,
        help="Override the maximum blocker count allowed for automatic submission dossier export",
    )
    parser.add_argument(
        "--auto-submission-dossier-min-rewrite-gain",
        type=float,
        default=None,
        help="Override the minimum rewrite gain required for automatic submission dossier export",
    )
    parser.add_argument(
        "--auto-submission-dossier-require-gate",
        action="store_true",
        default=True,
        help="Require quality-gate pass before auto-exporting a submission dossier",
    )
    parser.add_argument(
        "--no-auto-submission-dossier-require-gate",
        dest="auto_submission_dossier_require_gate",
        action="store_false",
    )
    parser.add_argument(
        "--auto-submission-dossier-require-ready",
        action="store_true",
        default=True,
        help="Require ready submission status before auto-exporting a submission dossier",
    )
    parser.add_argument(
        "--no-auto-submission-dossier-require-ready",
        dest="auto_submission_dossier_require_ready",
        action="store_false",
    )

    parser.add_argument(
        "generator_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed to continuous_paper_generator.py; use '--' before these args.",
    )
    return parser


def main() -> int:
    require_login("连续研究守护进程(continuous_research_daemon)")

    parser = build_parser()
    args = parser.parse_args()
    args.research_dir = str(Path(args.research_dir).expanduser())
    os.environ[PRIMARY_OUTPUT_ENV_VAR] = args.research_dir
    if (
        not args.topic
        and not args.ideas
        and not args.topic_files
        and not args.ideas_files
        and not args.source_config
    ):
        parser.error(
            "at least one of --topic, --ideas, --topic-files, --ideas-files, or --source-config is required"
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    daemon_dir = Path(args.research_dir) / "daemon_runs" / args.daemon_name
    daemon_dir.mkdir(parents=True, exist_ok=True)
    _ensure_control_file(daemon_dir)
    status_path = daemon_dir / "daemon_status.json"
    heartbeat_log = daemon_dir / "heartbeat.log"

    dashboard_server = None
    dashboard_thread = None
    dashboard_url = None
    if args.serve_dashboard:
        dashboard_server, dashboard_thread, dashboard_url = _start_dashboard_server(
            daemon_dir, args.dashboard_host, args.dashboard_port
        )

    status = _default_daemon_status(args, daemon_dir)
    status["dashboard_url"] = dashboard_url
    status = _apply_control_overrides(status, daemon_dir)
    _safe_write_json(status_path, status)
    _append_log(heartbeat_log, f"daemon initialized: {args.daemon_name}")

    if args.dry_run:
        status["state"] = "dry_run"
        _run_cycle(args, status, daemon_dir)
        status["source_runtime_rows"] = _write_source_boards(status, daemon_dir, args)
        status["updated_at"] = _now_iso()
        status["state"] = "finished"
        cycle_summary = _write_cycle_summary(status, daemon_dir)
        daily_summary = _write_daily_summary(status, daemon_dir)
        _append_autonomous_experiment_ledger(daemon_dir, status, {})
        _write_autonomy_program(daemon_dir, status, args)
        brief = _write_operator_brief(status, daemon_dir, None, args)
        handoff_report = _write_handoff_report(
            status, daemon_dir, brief, cycle_summary, daily_summary
        )
        _write_daily_report(status, daemon_dir, brief, daily_summary, handoff_report)
        _write_live_dashboard(status, daemon_dir)
        status["last_operator_brief_at"] = _now_iso()
        _safe_write_json(status_path, status)
        print(
            json.dumps(
                {
                    "daemon_dir": str(daemon_dir),
                    "command": status.get("last_cycle_command"),
                    "dashboard_url": status.get("dashboard_url"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        _stop_dashboard_server(dashboard_server, dashboard_thread)
        return 0

    started_at = time.time()
    success_count = 0
    failure_count = 0

    while not SHUTDOWN_REQUESTED:
        status = _apply_control_overrides(status, daemon_dir)
        if status.get("control", {}).get("paused"):
            if status.get("state") != "paused_by_control":
                _append_control_event(
                    daemon_dir,
                    {"type": "daemon_paused", "control": status.get("control")},
                )
            status["state"] = "paused_by_control"
            status["updated_at"] = _now_iso()
            _write_live_dashboard(status, daemon_dir)
            _safe_write_json(status_path, status)
            time.sleep(10)
            continue
        if (
            not args.run_forever
            and (time.time() - started_at) >= args.duration_hours * 3600
        ):
            break
        if args.max_cycles is not None and status.get("cycle", 0) >= args.max_cycles:
            break

        returncode = _run_cycle(args, status, daemon_dir)
        status["last_cycle_finished_at"] = _now_iso()
        status["last_returncode"] = returncode
        status["updated_at"] = _now_iso()
        status = _apply_failure_guard(status, daemon_dir, args)
        status["quality_governor"] = _derive_quality_governor(status, daemon_dir, args)

        if (
            args.enable_rewrite_followup
            and args.rewrite_followup_top_k > 0
            and returncode == 0
            and status.get("guardrail_phase") != "cold_start"
        ):
            manager = ResearchManager(args.research_dir)
            manager.rebuild_index()
            followup_summary = _run_rewrite_followup(args, status, daemon_dir, manager)
            status["last_rewrite_followup"] = followup_summary
            manager.rebuild_index()
            status["source_quality_feedback"] = _build_source_quality_feedback(
                manager, args
            )
            status["last_views"] = _export_cycle_views(manager, args, daemon_dir)
            status["quality_governor"] = _derive_quality_governor(
                status, daemon_dir, args
            )
        status = _update_guardrail_state(args, status)
        _append_log(
            heartbeat_log,
            f"guardrail phase now '{status.get('guardrail_phase')}' because {status.get('guardrail_phase_reason')} | mode '{status.get('guardrail_mode')}' because {status.get('guardrail_reason')}",
        )
        if returncode == 0:
            success_count += 1
            status["success_count"] = success_count
            status["state"] = "sleeping_after_success"
            status["last_error"] = None
            if status.get("guardrail_mode") == "generate_more":
                sleep_minutes = min(
                    args.sleep_minutes, args.guardrail_generate_sleep_minutes
                )
            elif status.get("guardrail_mode") == "focus_rewrite":
                sleep_minutes = max(
                    args.sleep_minutes, args.guardrail_focus_sleep_minutes
                )
            else:
                sleep_minutes = args.sleep_minutes
        else:
            failure_count += 1
            status["failure_count"] = failure_count
            status["state"] = "sleeping_after_failure"
            status["last_error"] = f"generator returned {returncode}"
            sleep_minutes = args.failure_backoff_minutes

        control = status.get("control") or {}
        if control.get("sleep_override_minutes") is not None:
            try:
                sleep_minutes = float(control.get("sleep_override_minutes"))
            except (TypeError, ValueError):
                pass

        next_run = datetime.now() + timedelta(minutes=sleep_minutes)
        status["next_cycle_at"] = next_run.isoformat()
        status["source_runtime_rows"] = _write_source_boards(status, daemon_dir, args)
        manager_for_brief = ResearchManager(args.research_dir)
        previous_cycle_summary = _safe_read_json(
            daemon_dir / "latest_cycle_summary.json"
        )
        if status.get("cycle", 0) % max(1, args.summary_every_cycles) == 0:
            _write_cycle_summary(status, daemon_dir)
            status["last_cycle_summary_at"] = _now_iso()
            _append_autonomous_experiment_ledger(
                daemon_dir, status, previous_cycle_summary
            )
            _write_autonomy_program(daemon_dir, status, args)
            _write_operator_brief(status, daemon_dir, manager_for_brief, args)
            _write_live_dashboard(status, daemon_dir)
            status["last_operator_brief_at"] = _now_iso()
        if status.get("cycle", 0) % max(1, args.daily_summary_every_cycles) == 0:
            _write_daily_summary(status, daemon_dir)
            status["last_daily_summary_at"] = _now_iso()
        _safe_write_json(status_path, status)
        _append_log(
            heartbeat_log,
            f"sleeping for {sleep_minutes} minutes; next cycle at {status['next_cycle_at']}",
        )

        if control.get("stop_after_cycle"):
            status["state"] = "stopping_after_cycle"
            status["updated_at"] = _now_iso()
            status["next_cycle_at"] = None
            _safe_write_json(status_path, status)
            _append_log(
                heartbeat_log, "stop_after_cycle requested via daemon_control.json"
            )
            _append_control_event(
                daemon_dir, {"type": "stop_after_cycle_triggered", "control": control}
            )
            break

        slept = 0.0
        while slept < sleep_minutes * 60 and not SHUTDOWN_REQUESTED:
            time.sleep(min(30.0, sleep_minutes * 60 - slept))
            slept += min(30.0, sleep_minutes * 60 - slept)
            status["updated_at"] = _now_iso()
            _safe_write_json(status_path, status)

    status["state"] = "stopped" if SHUTDOWN_REQUESTED else "completed"
    status["updated_at"] = _now_iso()
    status["next_cycle_at"] = None
    status["source_runtime_rows"] = _write_source_boards(status, daemon_dir, args)
    cycle_summary = _write_cycle_summary(status, daemon_dir)
    daily_summary = _write_daily_summary(status, daemon_dir)
    _append_autonomous_experiment_ledger(
        daemon_dir,
        status,
        _safe_read_json(daemon_dir / "latest_cycle_summary.json"),
    )
    _write_autonomy_program(daemon_dir, status, args)
    brief = _write_operator_brief(
        status, daemon_dir, ResearchManager(args.research_dir), args
    )
    handoff_report = _write_handoff_report(
        status, daemon_dir, brief, cycle_summary, daily_summary
    )
    _write_daily_report(status, daemon_dir, brief, daily_summary, handoff_report)
    _write_live_dashboard(status, daemon_dir)
    status["last_operator_brief_at"] = _now_iso()
    _safe_write_json(status_path, status)
    _append_log(
        heartbeat_log,
        f"daemon stopped with success_count={success_count}, failure_count={failure_count}",
    )
    _stop_dashboard_server(dashboard_server, dashboard_thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
