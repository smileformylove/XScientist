"""Persistent operator controls for the long-running research daemon."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json, durable_append_text


def _now_iso() -> str:
    return datetime.now().isoformat()


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
    for key in payload:
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
        for command_key in command:
            if command_key not in allowed_command_keys:
                errors.append(f"unknown source command key for {key}: {command_key}")

    return errors[:20]


def _persistable_control_payload(payload: dict[str, Any]) -> dict[str, Any]:
    persisted = dict(payload)
    persisted.pop("validation_errors", None)
    return persisted


def _write_control_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        _persistable_control_payload(payload),
        indent=2,
        ensure_ascii=False,
    )


def _ensure_control_file(daemon_dir: Path) -> Path:
    control_path = daemon_dir / "daemon_control.json"
    if not control_path.exists():
        _write_control_json(control_path, _default_control_payload())
    return control_path


def _load_control_payload(daemon_dir: Path, current_cycle: int = 0) -> dict[str, Any]:
    control_path = _ensure_control_file(daemon_dir)
    errors = []
    try:
        payload = json.loads(control_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = _default_control_payload()
        errors.append("control file is not valid JSON; falling back to defaults")
    if not isinstance(payload, dict):
        errors.append("control payload must be a JSON object")
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
    _write_control_json(control_path, payload)


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
    del daemon_dir  # Retained in the signature for legacy callers.
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
    durable_append_text(
        daemon_dir / "daemon_control_history.jsonl",
        json.dumps({"timestamp": _now_iso(), **event}, ensure_ascii=False) + "\n",
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
