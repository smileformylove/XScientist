from __future__ import annotations

"""Decision-log helpers for auditable runtime choices."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    load_jsonl_artifact,
    update_pipeline_artifact,
)
from ai_scientist.utils.provider_registry import (
    describe_model_requirements,
    resolve_model_provider,
)


class DecisionLogError(ValueError):
    """Raised when a decision log entry is too weak to audit."""


def _now_iso() -> str:
    return datetime.now().isoformat()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _coerce_options(options_considered: list[Any] | tuple[Any, ...] | None) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    for idx, item in enumerate(options_considered or []):
        if isinstance(item, dict):
            name = _clean_text(item.get("option") or item.get("name") or item.get("value"))
            status = _clean_text(item.get("status"))
            rejected_because = _clean_text(item.get("rejected_because"))
            selected = bool(item.get("selected"))
        else:
            name = _clean_text(item)
            status = ""
            rejected_because = ""
            selected = False
        if not name:
            raise DecisionLogError(f"options_considered[{idx}] is missing an option name")
        options.append(
            {
                "option": name,
                "selected": selected,
                "status": status or ("selected" if selected else "available"),
                "rejected_because": rejected_because,
            }
        )
    return options


def _validate_decision_entry(entry: dict[str, Any]) -> None:
    category = _clean_text(entry.get("category"))
    selected = _clean_text(entry.get("selected"))
    options = entry.get("options_considered")
    rejected_because = entry.get("rejected_because")
    if not category:
        raise DecisionLogError("decision category is required")
    if not selected:
        raise DecisionLogError("selected decision value is required")
    if not isinstance(options, list) or not options:
        raise DecisionLogError("options_considered must contain at least one option")
    if selected not in {str(option.get("option")) for option in options}:
        raise DecisionLogError("selected decision value must be present in options_considered")
    rejected_map = rejected_because if isinstance(rejected_because, dict) else {}
    if len(options) <= 1:
        raise DecisionLogError(
            "options_considered must include selected and rejected runtime options"
        )
    for option in options:
        name = str(option.get("option") or "")
        if name == selected:
            continue
        reason = _clean_text(option.get("rejected_because")) or _clean_text(
            rejected_map.get(name)
        )
        if not reason:
            raise DecisionLogError(f"rejected option {name!r} must include rejected_because")


def build_decision_entry(
    *,
    category: str,
    selected: str,
    options_considered: list[Any] | tuple[Any, ...],
    rejected_because: dict[str, str] | None = None,
    producer: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_text = _clean_text(selected)
    rejected_map = {
        _clean_text(key): _clean_text(value)
        for key, value in (rejected_because or {}).items()
        if _clean_text(key) and _clean_text(value)
    }
    options = _coerce_options(options_considered)
    for option in options:
        name = str(option["option"])
        option["selected"] = name == selected_text
        option["status"] = "selected" if option["selected"] else "rejected"
        if not option["selected"] and not option["rejected_because"]:
            option["rejected_because"] = rejected_map.get(name, "")
    entry = {
        "schema_version": 1,
        "recorded_at": _now_iso(),
        "category": _clean_text(category),
        "selected": selected_text,
        "options_considered": options,
        "rejected_because": rejected_map,
        "producer": _clean_text(producer),
        "metadata": dict(metadata or {}),
    }
    _validate_decision_entry(entry)
    return entry


def record_decision(
    project_root: str | Path,
    *,
    category: str,
    selected: str,
    options_considered: list[Any] | tuple[Any, ...],
    rejected_because: dict[str, str] | None = None,
    producer: str = "decision_log",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    entry = build_decision_entry(
        category=category,
        selected=selected,
        options_considered=options_considered,
        rejected_because=rejected_because,
        producer=producer,
        metadata=metadata,
    )
    append_jsonl_artifact(artifact_path(root, "decision_log"), entry)
    update_pipeline_artifact(
        root,
        "decision_log",
        status="ready",
        producer=producer,
        depends_on=[],
        notes="Runtime choices must record selected option, options considered, and rejection reasons.",
    )
    return entry


def load_decision_log(project_root: str | Path) -> list[dict[str, Any]]:
    return load_jsonl_artifact(artifact_path(project_root, "decision_log"))


def build_workflow_strategy_decision_options(
    *,
    selected: str,
    requested_workflow_mode: str | None,
    submission_mode: bool = False,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
    target_venue: str | None = None,
) -> list[dict[str, Any]]:
    selected_text = _clean_text(selected)
    requested = _clean_text(requested_workflow_mode).lower() or "adaptive"
    venue = _clean_text(target_venue).lower()
    candidates: list[tuple[str, str]] = []

    if requested != "adaptive":
        candidates.append(("adaptive", "adaptive resolver bypassed by explicit workflow_mode"))
        candidates.append((requested, "explicit workflow_mode request"))
    else:
        candidates.append(("adaptive", "adaptive resolver entrypoint"))
        if breakthrough_mode:
            candidates.append(("agentic_tree", "breakthrough_mode=True"))
        if submission_mode:
            candidates.append(("program_driven", "submission_mode=True"))
        if high_quality_mode and venue in {"nature", "journal"}:
            candidates.append(
                ("review_board", f"high_quality_mode=True and target_venue={venue}")
            )
        elif high_quality_mode:
            candidates.append(("writing_studio", "high_quality_mode=True"))
        candidates.append(("classic_pipeline", "default candidate when no stronger mode applies"))

    if selected_text and all(name != selected_text for name, _ in candidates):
        candidates.append((selected_text, "selected by workflow resolver"))

    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, reason in candidates:
        option = _clean_text(name)
        if not option or option in seen:
            continue
        seen.add(option)
        selected_option = option == selected_text
        if selected_option:
            rejected_reason = ""
        elif option == "adaptive":
            rejected_reason = (
                "explicit workflow_mode request bypassed adaptive resolver"
                if requested != "adaptive"
                else f"resolver produced concrete workflow_mode={selected_text}"
            )
        elif requested != "adaptive" and option != selected_text:
            rejected_reason = f"explicit workflow_mode={requested} excluded {option}"
        elif option == "agentic_tree":
            rejected_reason = "breakthrough_mode=False"
        elif option == "program_driven":
            rejected_reason = "submission_mode=False"
        elif option == "review_board":
            rejected_reason = (
                "requires high_quality_mode=True and target_venue in {nature,journal}"
            )
        elif option == "writing_studio":
            rejected_reason = "requires high_quality_mode=True without journal/nature routing"
        elif option == "classic_pipeline":
            rejected_reason = "a more specific workflow condition matched first"
        else:
            rejected_reason = f"{reason} did not match selected workflow_mode={selected_text}"
        options.append(
            {
                "option": option,
                "selected": selected_option,
                "status": "selected" if selected_option else "rejected",
                "rejected_because": rejected_reason,
            }
        )
    return options


def build_sample_gate_decision_options(sample_gate_passed: bool) -> list[dict[str, Any]]:
    return [
        {
            "option": "continue_full_generation",
            "selected": bool(sample_gate_passed),
            "status": "selected" if sample_gate_passed else "rejected",
            "rejected_because": ""
            if sample_gate_passed
            else "sample gate did not pass, so full generation remains blocked",
        },
        {
            "option": "block_full_generation",
            "selected": not bool(sample_gate_passed),
            "status": "selected" if not sample_gate_passed else "rejected",
            "rejected_because": ""
            if not sample_gate_passed
            else "sample gate passed, no blocking reason remains",
        },
    ]


def record_model_provider_decisions(
    project_root: str | Path,
    models: list[str] | tuple[str, ...],
    *,
    producer: str,
    env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    env_source = env or os.environ
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in describe_model_requirements(models, env=env_source):
        model = _clean_text(row.get("model"))
        if not model or model in seen:
            continue
        seen.add(model)
        spec = resolve_model_provider(model)
        implicit_openai = (
            spec.provider == "openai"
            and "/" not in model
            and model.startswith(("gpt-", "chatgpt-", "o1", "o3"))
        )
        prefixed = "/" in model and model.split("/", 1)[0] == spec.provider
        if implicit_openai:
            rejected_option = "openai_compat"
            rejected_reason = "model has no openai_compat/ prefix, so OpenAI-compatible override was not considered"
        elif prefixed:
            rejected_option = "unprefixed_model_resolution"
            rejected_reason = f"explicit {spec.provider}/ prefix fixed provider selection"
        else:
            rejected_option = "provider_prefix_override"
            rejected_reason = "model resolver matched the model family before any provider-prefix override"
        options = [
            {
                "option": spec.provider,
                "selected": True,
                "status": "selected",
                "rejected_because": "",
            },
            {
                "option": rejected_option,
                "selected": False,
                "status": "rejected",
                "rejected_because": rejected_reason,
            },
        ]
        entries.append(
            record_decision(
                project_root,
                category="model_provider",
                selected=spec.provider,
                options_considered=options,
                producer=producer,
                metadata={
                    "model": model,
                    "display_name": spec.display_name,
                    "client_family": spec.client_family,
                    "client_model": spec.client_model,
                    "request_style": spec.request_style,
                    "missing_credentials": row.get("missing"),
                },
            )
        )
    return entries
