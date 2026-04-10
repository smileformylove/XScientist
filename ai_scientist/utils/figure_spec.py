from __future__ import annotations

"""Figure specification helpers for a two-stage plotting pipeline."""

import hashlib
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import save_contract_artifact


def _dedupe_signature(parts: list[str]) -> str:
    payload = "||".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _flatten_npy_files(summary: Any) -> list[str]:
    if isinstance(summary, list):
        values: list[str] = []
        for item in summary:
            values.extend(_flatten_npy_files(item))
        return values
    if isinstance(summary, dict):
        files = summary.get("exp_results_npy_files")
        if isinstance(files, list):
            return [str(item) for item in files if str(item).strip()]
        values: list[str] = []
        for item in summary.values():
            values.extend(_flatten_npy_files(item))
        return values
    return []


def _infer_source_kind(stage_name: str, payload: Any) -> str:
    lowered = str(stage_name or "").lower()
    payload_dict = payload if isinstance(payload, dict) else {}
    text_hints = " ".join(
        str(item)
        for item in [
            lowered,
            payload_dict.get("plot_type", ""),
            payload_dict.get("analysis_type", ""),
        ]
    ).lower()
    if any(token in text_hints for token in ("ablation", "sensitivity")):
        return "ablation"
    if any(token in text_hints for token in ("baseline", "compare", "comparison")):
        return "comparison"
    if any(token in text_hints for token in ("qualitative", "visual", "image", "attention")):
        return "qualitative"
    if any(token in text_hints for token in ("architecture", "pipeline", "workflow")):
        return "architecture"
    return "results"


def _resolve_file_availability(
    data_files: list[str],
    *,
    base_folder: str | Path | None,
) -> tuple[list[str], list[str], bool]:
    if base_folder is None:
        return [], [], False
    base_path = Path(base_folder).expanduser().resolve()
    available: list[str] = []
    missing: list[str] = []
    for item in data_files:
        text = str(item).strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        resolved = candidate if candidate.is_absolute() else (base_path / candidate)
        if resolved.exists():
            available.append(text)
        else:
            missing.append(text)
    return available, missing, True


def summarize_figure_spec(
    spec: dict[str, Any] | None,
    *,
    claim_evidence_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = spec if isinstance(spec, dict) else {}
    graph = claim_evidence_graph if isinstance(claim_evidence_graph, dict) else {}
    figures = [item for item in (payload.get("figures") or []) if isinstance(item, dict)]
    ready_figures = [item for item in figures if str(item.get("status") or "").strip() == "ready"]
    blocked_figures = [
        item for item in figures if str(item.get("status") or "").strip() == "blocked"
    ]
    main_figures = [
        item for item in figures if str(item.get("paper_slot") or "").strip() == "main"
    ]
    if not main_figures and figures:
        main_figures = list(figures[:4])
    main_ready_figures = [
        item
        for item in main_figures
        if str(item.get("status") or "").strip() == "ready"
    ]
    main_blocked_figures = [
        item
        for item in main_figures
        if str(item.get("status") or "").strip() == "blocked"
    ]
    claim_ids = {
        str(node.get("id")).strip()
        for node in (graph.get("nodes") or [])
        if isinstance(node, dict)
        and node.get("type") == "claim"
        and str(node.get("id") or "").strip()
    }
    covered_claim_ids = {
        str(item.get("claim_id")).strip()
        for item in ready_figures
        if str(item.get("claim_id") or "").strip()
    }
    checked_data_file_availability = any(
        "available_data_files" in item or "missing_data_files" in item for item in figures
    )
    missing_data_file_count = sum(len(item.get("missing_data_files") or []) for item in figures)
    available_data_file_count = sum(
        len(item.get("available_data_files") or []) for item in figures
    )
    ready_missing_data_file_count = sum(
        len(item.get("missing_data_files") or []) for item in ready_figures
    )
    dedupe_signatures = [
        str(item.get("dedupe_signature") or "").strip() for item in figures if str(item.get("dedupe_signature") or "").strip()
    ]
    uncovered_claim_ids = sorted(claim_ids - covered_claim_ids)
    claim_count = len(claim_ids)
    coverage_ratio = (
        round(len(covered_claim_ids) / claim_count, 3) if claim_count else 1.0
    )
    return {
        "figure_count": len(figures),
        "ready_figure_count": len(ready_figures),
        "blocked_figure_count": len(blocked_figures),
        "main_figure_count": len(main_figures),
        "main_ready_count": len(main_ready_figures),
        "main_blocked_count": len(main_blocked_figures),
        "claim_count": claim_count,
        "covered_claim_count": len(covered_claim_ids),
        "claim_coverage_ratio": coverage_ratio,
        "covered_claim_ids": sorted(covered_claim_ids),
        "uncovered_claim_ids": uncovered_claim_ids,
        "checked_data_file_availability": checked_data_file_availability,
        "available_data_file_count": available_data_file_count,
        "missing_data_file_count": missing_data_file_count,
        "ready_missing_data_file_count": ready_missing_data_file_count,
        "duplicate_dedupe_signature_count": max(
            len(dedupe_signatures) - len(set(dedupe_signatures)),
            0,
        ),
    }


def build_figure_spec_from_summaries(
    summaries: dict[str, Any],
    *,
    claim_evidence_graph: dict[str, Any] | None = None,
    base_folder: str | Path | None = None,
    max_figures: int = 12,
) -> dict[str, Any]:
    graph = claim_evidence_graph or {}
    claim_nodes = [
        node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "claim"
    ]
    figures: list[dict[str, Any]] = []
    unique_signatures: set[str] = set()

    for idx, (stage_name, payload) in enumerate((summaries or {}).items()):
        if len(figures) >= max_figures:
            break
        data_files = _flatten_npy_files(payload)
        claim_node = claim_nodes[min(idx, len(claim_nodes) - 1)] if claim_nodes else {}
        claim_id = str(claim_node.get("id") or f"claim_{idx}")
        claim_label = str(claim_node.get("label") or f"Claim {idx}")
        dedupe_signature = _dedupe_signature([stage_name, claim_id] + data_files[:4])
        if dedupe_signature in unique_signatures:
            continue
        unique_signatures.add(dedupe_signature)
        blocking_reasons = []
        if not data_files:
            blocking_reasons.append("no_exp_results_npy_files")
        available_data_files, missing_data_files, checked_availability = (
            _resolve_file_availability(data_files, base_folder=base_folder)
        )
        if checked_availability and missing_data_files:
            blocking_reasons.append("missing_data_files")
        source_kind = _infer_source_kind(stage_name, payload)
        figure_type = "line_plot"
        if source_kind == "ablation":
            figure_type = "ablation_grid"
        elif source_kind == "comparison":
            figure_type = "comparison_plot"
        elif source_kind == "qualitative":
            figure_type = "qualitative_panel"
        elif source_kind == "architecture":
            figure_type = "diagram"
        spec = {
            "figure_id": f"figure_{idx}",
            "claim_id": claim_id,
            "source_records": [stage_name],
            "source_kind": source_kind,
            "data_files": data_files,
            "available_data_files": available_data_files,
            "missing_data_files": missing_data_files,
            "figure_type": figure_type,
            "panel_layout": "1x2" if len(data_files) > 1 else "1x1",
            "caption_intent": f"Show evidence for: {claim_label}",
            "paper_slot": "main" if idx < 4 else "appendix",
            "dedupe_signature": dedupe_signature,
            "blocking_reasons": blocking_reasons,
            "status": "blocked" if blocking_reasons else "ready",
            "suggested_title": f"{stage_name.replace('_', ' ').title()} evidence",
        }
        figures.append(spec)

    summary = summarize_figure_spec({"figures": figures}, claim_evidence_graph=graph)
    return {
        "schema_version": 1,
        "figure_count": len(figures),
        "summary": summary,
        "figures": figures,
    }


def render_figure_spec_markdown(spec: dict[str, Any]) -> str:
    lines = ["# Figure Spec", ""]
    for figure in spec.get("figures", []):
        lines.extend(
            [
                f"## {figure.get('figure_id')}",
                f"- Claim: {figure.get('claim_id')}",
                f"- Figure type: {figure.get('figure_type')}",
                f"- Paper slot: {figure.get('paper_slot')}",
                f"- Status: {figure.get('status')}",
                f"- Suggested title: {figure.get('suggested_title')}",
                f"- Caption intent: {figure.get('caption_intent')}",
                f"- Data files: {', '.join(figure.get('data_files') or []) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def save_figure_spec(
    project_root: str | Path,
    spec: dict[str, Any],
    *,
    producer: str = "figure_spec",
) -> str:
    output_path = save_contract_artifact(
        project_root,
        "figure_spec",
        spec,
        producer=producer,
        depends_on=["claim_evidence_graph", "experiment_registry"],
    )
    from ai_scientist.utils.stage_standards import save_stage_standards

    save_stage_standards(project_root)
    return output_path
