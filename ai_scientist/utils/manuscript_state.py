from __future__ import annotations

"""Structured manuscript state derived from claims, figures, and writeup policy."""

from typing import Any

from ai_scientist.utils.pipeline_contracts import save_contract_artifact


DEFAULT_OUTLINE = {
    "normal": [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "discussion",
        "conclusion",
    ],
    "icbinb": [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "lessons_learned",
        "conclusion",
    ],
    "journal": [
        "abstract",
        "introduction",
        "related_work",
        "method",
        "experiments",
        "results",
        "discussion",
        "limitations",
        "conclusion",
    ],
    "extended": ["abstract", "introduction", "key_results", "conclusion"],
}

RESULT_SECTIONS = {"experiments", "results", "discussion", "conclusion"}
METHOD_SECTIONS = {"method", "experiments"}


def _build_section_bindings(
    outline: list[str],
    claim_bindings: list[str],
    claim_figure_bindings: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    section_claim_bindings: dict[str, list[str]] = {}
    section_figure_bindings: dict[str, list[str]] = {}
    figure_ids: list[str] = []
    for figure_group in claim_figure_bindings.values():
        for item in figure_group or []:
            figure_id = str(item).strip()
            if figure_id and figure_id not in figure_ids:
                figure_ids.append(figure_id)
    for section in outline:
        section_name = str(section or "").strip()
        if not section_name:
            continue
        lowered = section_name.lower()
        if lowered in RESULT_SECTIONS or lowered in {"abstract", "introduction"}:
            section_claim_bindings[section_name] = list(claim_bindings)
        elif lowered in METHOD_SECTIONS:
            section_claim_bindings[section_name] = list(claim_bindings[:1] or claim_bindings)
        else:
            section_claim_bindings[section_name] = []
        if lowered in RESULT_SECTIONS:
            section_figure_bindings[section_name] = list(figure_ids)
        elif lowered == "abstract":
            section_figure_bindings[section_name] = list(figure_ids[:1])
        else:
            section_figure_bindings[section_name] = []
    return section_claim_bindings, section_figure_bindings


def build_manuscript_state(
    *,
    writeup_type: str,
    target_venue: str | None,
    writing_profile: str,
    skill_pack: list[str],
    claim_evidence_graph: dict[str, Any] | None = None,
    figure_spec: dict[str, Any] | None = None,
    latex_path: str | None = None,
) -> dict[str, Any]:
    graph = claim_evidence_graph or {}
    spec = figure_spec or {}
    outline = list(DEFAULT_OUTLINE.get(writeup_type, DEFAULT_OUTLINE["normal"]))
    claim_nodes = [
        node for node in graph.get("nodes", []) if isinstance(node, dict) and node.get("type") == "claim"
    ]
    ready_figures = [
        figure
        for figure in spec.get("figures", [])
        if isinstance(figure, dict) and figure.get("status") == "ready"
    ]
    explicit_main_exists = any(
        str(figure.get("paper_slot") or "").strip() == "main" for figure in ready_figures
    )
    claim_figure_bindings: dict[str, list[str]] = {}
    main_claim_figure_bindings: dict[str, list[str]] = {}
    for idx, figure in enumerate(ready_figures):
        claim_id = str(figure.get("claim_id") or "").strip()
        figure_id = str(figure.get("figure_id") or "").strip()
        if not claim_id or not figure_id:
            continue
        claim_figure_bindings.setdefault(claim_id, [])
        if figure_id not in claim_figure_bindings[claim_id]:
            claim_figure_bindings[claim_id].append(figure_id)
        is_main_figure = str(figure.get("paper_slot") or "").strip() == "main"
        if not explicit_main_exists and idx < 4:
            is_main_figure = True
        if is_main_figure:
            main_claim_figure_bindings.setdefault(claim_id, [])
            if figure_id not in main_claim_figure_bindings[claim_id]:
                main_claim_figure_bindings[claim_id].append(figure_id)
    figure_bindings = {
        claim_id: figure_ids[0]
        for claim_id, figure_ids in claim_figure_bindings.items()
        if figure_ids
    }
    claim_bindings = [node.get("id") for node in claim_nodes]
    section_claim_bindings, section_figure_bindings = _build_section_bindings(
        outline,
        claim_bindings,
        claim_figure_bindings,
    )
    missing_evidence = [
        f"claim {node.get('id')} has no ready figure support"
        for node in claim_nodes
        if node.get("id") not in figure_bindings
    ]
    return {
        "schema_version": 1,
        "target_venue": target_venue,
        "writing_profile": writing_profile,
        "skill_pack": list(skill_pack),
        "outline": outline,
        "section_briefs": {
            section: f"{section.replace('_', ' ').title()} should stay aligned with the bound claims and figures."
            for section in outline
        },
        "claim_bindings": claim_bindings,
        "figure_bindings": figure_bindings,
        "claim_figure_bindings": claim_figure_bindings,
        "main_claim_figure_bindings": main_claim_figure_bindings,
        "section_claim_bindings": section_claim_bindings,
        "section_figure_bindings": section_figure_bindings,
        "table_bindings": {},
        "citation_uncertainties": [],
        "evidence_summary": {
            "claim_count": len(claim_bindings),
            "supported_claim_count": len(claim_figure_bindings),
            "unsupported_claim_count": max(len(claim_bindings) - len(claim_figure_bindings), 0),
            "ready_figure_count": sum(
                len(figure_ids) for figure_ids in claim_figure_bindings.values()
            ),
            "main_supported_claim_count": len(main_claim_figure_bindings),
        },
        "missing_evidence": missing_evidence,
        "guardrail_status": "blocked" if missing_evidence else "ready",
        "latex_path": latex_path,
    }


def save_manuscript_state(project_root: str, state: dict[str, Any]) -> str:
    output_path = save_contract_artifact(
        project_root,
        "manuscript_state",
        state,
        producer="manuscript_state",
        depends_on=["claim_evidence_graph", "figure_spec"],
    )
    from ai_scientist.utils.stage_standards import save_stage_standards

    save_stage_standards(project_root)
    return output_path


def render_manuscript_prompt_context(state: dict[str, Any]) -> str:
    outline = ", ".join(state.get("outline") or [])
    missing = state.get("missing_evidence") or []
    lines = [
        "Structured manuscript state:",
        f"- Target venue: {state.get('target_venue')}",
        f"- Writing profile: {state.get('writing_profile')}",
        f"- Skill pack: {', '.join(state.get('skill_pack') or [])}",
        f"- Outline: {outline}",
        f"- Claim bindings: {', '.join(state.get('claim_bindings') or []) or 'none'}",
        f"- Figure bindings: {state.get('figure_bindings') or {}}",
        f"- Claim figure bindings: {state.get('claim_figure_bindings') or {}}",
        f"- Section claim bindings: {state.get('section_claim_bindings') or {}}",
        f"- Section figure bindings: {state.get('section_figure_bindings') or {}}",
        f"- Missing evidence: {missing or 'none'}",
        f"- Guardrail status: {state.get('guardrail_status')}",
    ]
    return "\n".join(lines)
