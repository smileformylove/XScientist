from __future__ import annotations

"""Shared pipeline artifact contracts for end-to-end autonomous research runs."""

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PIPELINE_SCHEMA_VERSION = 1

ARTIFACT_FILENAMES = {
    "pipeline_manifest": "pipeline_manifest.json",
    "idea_cards": "idea_cards.json",
    "research_plan": "research_plan.json",
    "claim_evidence_graph": "claim_evidence_graph.json",
    "experiment_registry": "experiment_registry.jsonl",
    "figure_spec": "figure_spec.json",
    "manuscript_state": "manuscript_state.json",
    "review_state": "review_state.json",
    "critic_findings": "critic_findings.json",
    "repair_plan": "repair_plan.json",
    "self_evolution": "self_evolution.json",
    "stage_standards": "stage_standards.json",
    "process_alignment": "process_alignment.json",
    "research_program": "research_program.md",
}

ARTIFACT_DEFAULT_STATUS = {
    "pipeline_manifest": "ready",
    "idea_cards": "missing",
    "research_plan": "missing",
    "claim_evidence_graph": "missing",
    "experiment_registry": "missing",
    "figure_spec": "missing",
    "manuscript_state": "missing",
    "review_state": "missing",
    "critic_findings": "missing",
    "repair_plan": "missing",
    "self_evolution": "missing",
    "stage_standards": "missing",
    "process_alignment": "missing",
    "research_program": "missing",
}

ARTIFACT_ALLOWED_STATUS = {
    "missing",
    "ready",
    "blocked",
    "failed",
    "stale",
}


def _now_iso() -> str:
    return datetime.now().isoformat()


def artifact_path(project_root: str | Path, artifact_name: str) -> Path:
    if artifact_name not in ARTIFACT_FILENAMES:
        raise KeyError(f"Unknown artifact name: {artifact_name}")
    return Path(project_root).expanduser().resolve() / ARTIFACT_FILENAMES[artifact_name]


def iter_project_roots(research_root: str | Path) -> list[Path]:
    root = Path(research_root).expanduser().resolve()
    candidate_roots = [
        root / "projects",
        root / "papers",
        root / "batches",
    ]
    project_roots: list[Path] = []
    seen: set[Path] = set()
    for parent in candidate_roots:
        if not parent.exists():
            continue
        for candidate in sorted(parent.iterdir()):
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if artifact_path(resolved, "pipeline_manifest").exists():
                project_roots.append(resolved)
                seen.add(resolved)
    return sorted(project_roots, key=lambda path: str(path))


def _default_artifact_entry(project_root: str | Path, artifact_name: str) -> dict[str, Any]:
    path = artifact_path(project_root, artifact_name)
    return {
        "name": artifact_name,
        "filename": path.name,
        "path": str(path),
        "status": ARTIFACT_DEFAULT_STATUS[artifact_name],
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": None,
        "producer": None,
        "depends_on": [],
        "warnings": [],
        "recovery_hint": None,
        "notes": None,
    }


def build_pipeline_manifest(
    project_root: str | Path,
    *,
    project_name: str | None = None,
    template_profile: str = "open_ended",
    template_capability: str = "adaptive",
    pipeline_goal: str = "conference_submission",
    workflow_mode: str = "classic_pipeline",
    workflow_label: str | None = None,
    workflow_summary: str | None = None,
    workflow_inspirations: list[str] | None = None,
    workflow_sequence: list[str] | None = None,
) -> dict[str, Any]:
    resolved_root = Path(project_root).expanduser().resolve()
    manifest = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "project_root": str(resolved_root),
        "project_name": project_name or resolved_root.name,
        "pipeline_goal": pipeline_goal,
        "template_profile": template_profile,
        "template_capability": template_capability,
        "workflow_mode": workflow_mode,
        "workflow_label": workflow_label or workflow_mode.replace("_", " ").title(),
        "workflow_summary": workflow_summary or "",
        "workflow_inspirations": list(workflow_inspirations or []),
        "workflow_sequence": list(workflow_sequence or []),
        "fallback_events": [],
        "fallback_summary": {
            "count": 0,
            "strict_count": 0,
            "stage_counts": {},
            "kind_counts": {},
            "latest_event": None,
        },
        "artifacts": {
            artifact_name: _default_artifact_entry(resolved_root, artifact_name)
            for artifact_name in ARTIFACT_FILENAMES
        },
    }
    manifest["artifacts"]["pipeline_manifest"]["generated_at"] = manifest["generated_at"]
    manifest["artifacts"]["pipeline_manifest"]["producer"] = "pipeline_contracts"
    return manifest


def load_json_artifact(path: str | Path, *, default: Any = None) -> Any:
    artifact_path_obj = Path(path)
    if not artifact_path_obj.exists():
        return deepcopy(default)
    try:
        with open(artifact_path_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return deepcopy(default)


def load_jsonl_artifact(path: str | Path) -> list[dict[str, Any]]:
    artifact_path_obj = Path(path)
    if not artifact_path_obj.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(artifact_path_obj, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except OSError:
        return []
    return rows


def save_json_artifact(path: str | Path, payload: Any) -> str:
    artifact_path_obj = Path(path).expanduser().resolve()
    artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path_obj, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(artifact_path_obj)


def save_jsonl_artifact(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    artifact_path_obj = Path(path).expanduser().resolve()
    artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path_obj, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(artifact_path_obj)


def append_jsonl_artifact(path: str | Path, row: dict[str, Any]) -> str:
    artifact_path_obj = Path(path).expanduser().resolve()
    artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(artifact_path_obj, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(artifact_path_obj)


def save_text_artifact(path: str | Path, text: str) -> str:
    artifact_path_obj = Path(path).expanduser().resolve()
    artifact_path_obj.parent.mkdir(parents=True, exist_ok=True)
    artifact_path_obj.write_text(text, encoding="utf-8")
    return str(artifact_path_obj)


def _summarize_fallback_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    latest_event: dict[str, Any] | None = None
    count = 0
    strict_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        count += 1
        stage = str(event.get("stage") or "unknown")
        kind = str(event.get("fallback_kind") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        if event.get("strict"):
            strict_count += 1
        latest_event = dict(event)
    return {
        "count": count,
        "strict_count": strict_count,
        "stage_counts": stage_counts,
        "kind_counts": kind_counts,
        "latest_event": latest_event,
    }


def load_pipeline_manifest(project_root: str | Path) -> dict[str, Any]:
    path = artifact_path(project_root, "pipeline_manifest")
    payload = load_json_artifact(path, default=None)
    if isinstance(payload, dict) and isinstance(payload.get("artifacts"), dict):
        payload.setdefault("fallback_events", [])
        payload["fallback_summary"] = _summarize_fallback_events(
            payload.get("fallback_events") or []
        )
        return payload
    manifest = build_pipeline_manifest(project_root)
    save_json_artifact(path, manifest)
    return manifest


def save_pipeline_manifest(project_root: str | Path, manifest: dict[str, Any]) -> str:
    manifest.setdefault("fallback_events", [])
    manifest["fallback_summary"] = _summarize_fallback_events(
        manifest.get("fallback_events") or []
    )
    manifest["generated_at"] = _now_iso()
    return save_json_artifact(artifact_path(project_root, "pipeline_manifest"), manifest)


def initialize_pipeline_contracts(
    project_root: str | Path,
    *,
    project_name: str | None = None,
    template_profile: str | None = None,
    template_capability: str | None = None,
    pipeline_goal: str | None = None,
    workflow_mode: str | None = None,
    workflow_label: str | None = None,
    workflow_summary: str | None = None,
    workflow_inspirations: list[str] | None = None,
    workflow_sequence: list[str] | None = None,
) -> dict[str, Any]:
    manifest = load_pipeline_manifest(project_root)
    manifest["schema_version"] = PIPELINE_SCHEMA_VERSION
    manifest["project_root"] = str(Path(project_root).expanduser().resolve())
    manifest["project_name"] = project_name or manifest.get("project_name") or Path(project_root).name
    manifest["pipeline_goal"] = pipeline_goal or manifest.get("pipeline_goal")
    manifest["template_profile"] = template_profile or manifest.get("template_profile")
    manifest["template_capability"] = template_capability or manifest.get("template_capability")
    manifest["workflow_mode"] = workflow_mode or manifest.get("workflow_mode") or "classic_pipeline"
    manifest["workflow_label"] = workflow_label or manifest.get("workflow_label") or str(manifest["workflow_mode"]).replace("_", " ").title()
    manifest["workflow_summary"] = workflow_summary or manifest.get("workflow_summary") or ""
    manifest["workflow_inspirations"] = list(
        workflow_inspirations
        or manifest.get("workflow_inspirations")
        or []
    )
    manifest["workflow_sequence"] = list(
        workflow_sequence
        or manifest.get("workflow_sequence")
        or []
    )
    artifacts = manifest.setdefault("artifacts", {})
    for artifact_name in ARTIFACT_FILENAMES:
        artifacts.setdefault(
            artifact_name,
            _default_artifact_entry(project_root, artifact_name),
        )
        artifacts[artifact_name].setdefault("path", str(artifact_path(project_root, artifact_name)))
        artifacts[artifact_name]["filename"] = artifact_path(project_root, artifact_name).name
        artifacts[artifact_name]["schema_version"] = PIPELINE_SCHEMA_VERSION
    manifest.setdefault("fallback_events", [])
    manifest["fallback_summary"] = _summarize_fallback_events(
        manifest.get("fallback_events") or []
    )
    save_pipeline_manifest(project_root, manifest)
    return manifest


def update_pipeline_artifact(
    project_root: str | Path,
    artifact_name: str,
    *,
    status: str,
    producer: str,
    warnings: list[str] | None = None,
    depends_on: list[str] | None = None,
    recovery_hint: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    if status not in ARTIFACT_ALLOWED_STATUS:
        raise ValueError(
            f"Invalid artifact status '{status}'. Allowed: {sorted(ARTIFACT_ALLOWED_STATUS)}"
        )
    manifest = initialize_pipeline_contracts(project_root)
    artifact_entry = manifest["artifacts"][artifact_name]
    artifact_entry["status"] = status
    artifact_entry["producer"] = producer
    artifact_entry["generated_at"] = _now_iso()
    artifact_entry["depends_on"] = list(depends_on or artifact_entry.get("depends_on") or [])
    artifact_entry["warnings"] = list(warnings or [])
    artifact_entry["recovery_hint"] = recovery_hint
    artifact_entry["notes"] = notes
    save_pipeline_manifest(project_root, manifest)
    return artifact_entry


def mark_compat_fallback(
    project_root: str | Path,
    artifact_name: str,
    *,
    producer: str,
    warning: str,
    recovery_hint: str | None = None,
) -> dict[str, Any]:
    manifest = initialize_pipeline_contracts(project_root)
    artifact_entry = manifest["artifacts"][artifact_name]
    warnings = list(artifact_entry.get("warnings") or [])
    if warning not in warnings:
        warnings.append(warning)
    return update_pipeline_artifact(
        project_root,
        artifact_name,
        status=artifact_entry.get("status") or "missing",
        producer=producer,
        warnings=warnings,
        depends_on=list(artifact_entry.get("depends_on") or []),
        recovery_hint=recovery_hint or artifact_entry.get("recovery_hint"),
        notes=artifact_entry.get("notes"),
    )


def record_fallback_event(
    project_root: str | Path,
    *,
    stage: str,
    producer: str,
    fallback_kind: str,
    reason: str,
    strict: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = initialize_pipeline_contracts(project_root)
    event = {
        "recorded_at": _now_iso(),
        "stage": str(stage or "unknown"),
        "producer": str(producer or "unknown"),
        "fallback_kind": str(fallback_kind or "unknown"),
        "reason": str(reason or "").strip(),
        "strict": bool(strict),
        "metadata": deepcopy(metadata or {}),
    }
    events = list(manifest.get("fallback_events") or [])
    events.append(event)
    manifest["fallback_events"] = events
    manifest["fallback_summary"] = _summarize_fallback_events(events)
    save_pipeline_manifest(project_root, manifest)
    return event


def save_contract_artifact(
    project_root: str | Path,
    artifact_name: str,
    payload: Any,
    *,
    producer: str,
    depends_on: list[str] | None = None,
    warnings: list[str] | None = None,
    recovery_hint: str | None = None,
    notes: str | None = None,
) -> str:
    path = artifact_path(project_root, artifact_name)
    if path.suffix == ".json":
        output_path = save_json_artifact(path, payload)
    elif path.suffix == ".jsonl":
        if not isinstance(payload, list):
            raise TypeError(f"{artifact_name} expects a list payload for jsonl serialization")
        output_path = save_jsonl_artifact(path, payload)
    else:
        output_path = save_text_artifact(path, str(payload))
    update_pipeline_artifact(
        project_root,
        artifact_name,
        status="ready",
        producer=producer,
        depends_on=depends_on,
        warnings=warnings,
        recovery_hint=recovery_hint,
        notes=notes,
    )
    return output_path


def load_contract_artifact(project_root: str | Path, artifact_name: str, *, default: Any = None) -> Any:
    path = artifact_path(project_root, artifact_name)
    if path.suffix == ".json":
        return load_json_artifact(path, default=default)
    if path.suffix == ".jsonl":
        return load_jsonl_artifact(path)
    if not path.exists():
        return deepcopy(default)
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return deepcopy(default)


def render_research_program_markdown(
    *,
    project_name: str,
    target_venue: str | None,
    template_profile: str,
    idea_name: str,
    hypothesis: str,
    workflow_mode: str | None = None,
    workflow_summary: str | None = None,
    workflow_inspirations: list[str] | None = None,
    workflow_sequence: list[str] | None = None,
    budget: dict[str, Any] | None = None,
    execution_policy: dict[str, Any] | None = None,
    success_criteria: list[str] | None = None,
    failure_handling_rules: list[str] | None = None,
) -> str:
    budget = budget or {}
    execution_policy = execution_policy or {}
    success_criteria = list(success_criteria or [])
    failure_handling_rules = list(failure_handling_rules or [])
    if execution_policy.get("reject_on_auto_improvement_fallback"):
        failure_handling_rules.append(
            "Do not treat auto-improvement fallback rewrites as submission-ready evidence under the current workflow discipline."
        )
    lines = [
        "# Research Program",
        "",
        f"- Project: {project_name}",
        f"- Target venue: {target_venue or 'unspecified'}",
        f"- Template profile: {template_profile}",
        f"- Workflow mode: {workflow_mode or 'unspecified'}",
        f"- Lead idea: {idea_name}",
        "",
        "## Workflow Strategy",
        workflow_summary or "Use the current workflow mode to adapt execution order and evidence pressure.",
        "",
    ]
    if workflow_inspirations:
        lines.append(f"- Inspirations: {', '.join(workflow_inspirations)}")
    if workflow_sequence:
        lines.append(f"- Preferred sequence: {' -> '.join(workflow_sequence)}")
    lines.extend(
        [
            "",
            "## Core Hypothesis",
            hypothesis or "TBD",
            "",
            "## Execution Budget",
            f"- Max steps: {budget.get('max_steps')}",
            f"- Max wallclock minutes: {budget.get('max_wallclock_minutes')}",
            f"- Max retries per task: {budget.get('max_retry_per_task')}",
            "",
            "## Operating Policy",
            f"- Execution style: {execution_policy.get('execution_style') or 'adaptive'}",
            f"- Evidence pressure: {execution_policy.get('evidence_pressure') or 'standard'}",
            f"- Quality fallback policy: {execution_policy.get('quality_fallback_policy') or 'allowed'}",
            f"- Auto-improvement fallback allowed: {'yes' if execution_policy.get('allow_auto_improvement_fallback', True) else 'no'}",
            f"- Reject submission after quality fallback: {'yes' if execution_policy.get('reject_on_auto_improvement_fallback') else 'no'}",
            "",
            "### Acceptance Rules",
        ]
    )
    for item in execution_policy.get("acceptance_rules") or [
        "Every promoted claim should remain traceable to an experiment and a figure-ready evidence path."
    ]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Registry Discipline",
        ]
    )
    for item in execution_policy.get("registry_expectations") or [
        "Preserve failures and blocked runs in experiment_registry.jsonl with explicit reasons."
    ]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Success Criteria",
        ]
    )
    for item in success_criteria or ["Demonstrate a measurable gain or a credible negative result with clear evidence."]:
        lines.append(f"- {item}")
    lines.extend(["", "## Failure Handling"])
    for item in failure_handling_rules or [
        "Record failed tasks in experiment_registry.jsonl instead of silently dropping them.",
        "Do not advance claims or figures unless the supporting artifact is ready.",
    ]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Review Discipline",
            "- Prefer evidence-backed claims over broad framing.",
            "- Preserve failed or inconclusive findings when they clarify boundary conditions.",
        ]
    )
    return "\n".join(lines) + "\n"
