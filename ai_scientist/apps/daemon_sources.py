"""Source queue configuration and normalization for the research daemon."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    try:
        import tomli as _toml_loader  # type: ignore
    except ModuleNotFoundError:
        _toml_loader = None

from ai_scientist.utils.source_planning import (
    normalize_batch_profile,
    normalize_source_archetype,
    normalize_workflow_mode_list,
    normalize_workflow_mode_name,
)


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
            errors.append(f"sources[{index}].{field} is not a supported workflow mode")
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
    archetype_fields = [
        "source_archetype",
        "day_source_archetype",
        "night_source_archetype",
    ]
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
            errors.append(f"sources[{index}].{field} is not a supported batch profile")
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
