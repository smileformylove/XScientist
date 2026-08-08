from __future__ import annotations

"""Evidence-bound project insight synthesis.

This module intentionally produces *candidate* insights. Internal model review is
not independent verification, so every emitted insight remains low/medium
confidence and carries an explicit unverified epistemic status.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.atomic_io import atomic_write_text
from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
)

INSIGHT_SCHEMA_VERSION = "xscientist.insight-report.v1"
ALLOWED_KINDS = {
    "mechanism",
    "boundary",
    "negative",
    "comparative",
    "methodological",
}
ALLOWED_CONFIDENCE = {"low", "medium"}
FORBIDDEN_CERTAINTY = re.compile(
    r"\b(proven|proved|confirmed|conclusive|definitive|independently verified)\b|"
    r"已证明|已证实|结论性证明|独立验证通过",
    flags=re.IGNORECASE,
)


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact_stage(
    stage: dict[str, Any], *, idea_idx: int, index: int
) -> dict[str, Any]:
    best = stage.get("best") if isinstance(stage.get("best"), dict) else {}
    return {
        "evidence_ref": f"metric:{idea_idx}:{index}",
        "stage": Path(str(stage.get("stage_dir") or f"stage_{index}")).name,
        "metric_name": best.get("metric_name"),
        "metric_mean": best.get("metric_mean"),
        "metric_objective": best.get("metric_objective"),
        "dataset_names": list(best.get("dataset_names") or [])[:8],
        "seed_eval": best.get("seed_eval"),
        "delta_objective_vs_previous_stage": stage.get("delta_objective_vs_prev_stage"),
        "node_counts": stage.get("node_counts") or {},
    }


def build_project_evidence_pack(
    project_root: str | Path,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a path-free, prompt-safe summary with stable evidence selectors."""

    root = Path(project_root).expanduser().resolve()
    idea_cards = load_contract_artifact(root, "idea_cards", default=[])
    if not isinstance(idea_cards, list):
        idea_cards = []
    plan = load_contract_artifact(root, "research_plan", default={})
    if not isinstance(plan, dict):
        plan = {}
    challenge = plan.get("socratic_challenge") or {}
    rivals = [
        {
            "rival_id": str(item.get("rival_id") or ""),
            "class": str(item.get("class") or ""),
            "statement": str(item.get("statement") or "")[:1200],
            "discriminating_prediction": str(
                item.get("discriminating_prediction") or ""
            )[:1200],
        }
        for item in (challenge.get("rival_hypotheses") or [])
        if isinstance(item, dict)
    ][:8]

    result_rows: list[dict[str, Any]] = []
    allowed_refs: list[str] = []
    for fallback_idx, result in enumerate(results):
        idea_idx = int(result.get("idea_idx", fallback_idx))
        result_ref = f"result:{idea_idx}"
        allowed_refs.append(result_ref)
        card = idea_cards[idea_idx] if idea_idx < len(idea_cards) else {}
        raw_exp_dir = str(result.get("exp_dir") or "").strip()
        report = (
            _safe_json(Path(raw_exp_dir).expanduser() / "experiment_report.json")
            if raw_exp_dir
            else {}
        )
        stages = [
            _compact_stage(item, idea_idx=idea_idx, index=index)
            for index, item in enumerate(report.get("stages") or [])
            if isinstance(item, dict)
        ][:12]
        allowed_refs.extend(str(stage["evidence_ref"]) for stage in stages)
        result_rows.append(
            {
                "idea_idx": idea_idx,
                "evidence_ref": result_ref,
                "idea": {
                    "title": str(card.get("title") or card.get("name") or "")[:800],
                    "hypothesis": str(card.get("core_hypothesis") or "")[:1600],
                    "mechanism": str(card.get("mechanism") or "")[:1600],
                    "falsifiers": list(card.get("falsifiers") or [])[:8],
                },
                "outcome": {
                    "status": str(result.get("status") or "unknown"),
                    "quality_score": result.get("quality_score"),
                    "rigor_score": result.get("rigor_score"),
                    "claim_support_score": result.get("claim_support_score"),
                    "quality_gate_passed": result.get("quality_gate_passed"),
                    "submission_acceptance_passed": result.get(
                        "submission_acceptance_passed"
                    ),
                    "ara_claim_coverage_score": result.get("ara_claim_coverage_score"),
                },
                "stages": stages,
                "warnings": [str(item)[:800] for item in report.get("warnings") or []][
                    :8
                ],
            }
        )

    return {
        "schema_version": "xscientist.insight-evidence-pack.v1",
        "results": result_rows,
        "rival_hypotheses": rivals,
        "allowed_evidence_refs": sorted(set(allowed_refs)),
        "interpretation_boundary": (
            "Run-internal evidence only; no independent replication or external "
            "verification is implied."
        ),
    }


def _default_rivals(pack: dict[str, Any]) -> list[str]:
    rows = [
        str(item.get("statement") or "").strip()
        for item in pack.get("rival_hypotheses") or []
        if isinstance(item, dict)
    ]
    rows = [item for item in rows if item]
    return rows[:3] or [
        "The observed result is compatible with run-to-run variation.",
        "A measurement artifact, not the proposed mechanism, explains the result.",
        "The effect does not transfer outside the evaluated scope.",
    ]


def _deterministic_insights(pack: dict[str, Any]) -> list[dict[str, Any]]:
    insights: list[dict[str, Any]] = []
    default_rivals = _default_rivals(pack)
    for row in pack.get("results") or []:
        idea_idx = int(row.get("idea_idx", len(insights)))
        stages = list(row.get("stages") or [])
        first = next(
            (item for item in stages if item.get("metric_objective") is not None), None
        )
        last = next(
            (
                item
                for item in reversed(stages)
                if item.get("metric_objective") is not None
            ),
            None,
        )
        evidence_refs = [str(row.get("evidence_ref"))]
        delta = None
        if first and last:
            evidence_refs.extend(
                [str(first.get("evidence_ref")), str(last.get("evidence_ref"))]
            )
            try:
                delta = float(last["metric_objective"]) - float(
                    first["metric_objective"]
                )
            except (TypeError, ValueError):
                delta = None
        title = str((row.get("idea") or {}).get("title") or f"Idea {idea_idx}")
        if delta is None:
            kind = "methodological"
            claim = (
                f"Run #{idea_idx} for {title} produced an auditable outcome, but the "
                "available stage metrics do not support a directional scientific claim."
            )
        elif delta > 0:
            kind = "comparative"
            claim = (
                f"Within run #{idea_idx}, the recorded search objective increased by "
                f"{delta:.6g} from the first to the last measurable stage."
            )
        elif delta < 0:
            kind = "negative"
            claim = (
                f"Within run #{idea_idx}, the recorded search objective decreased by "
                f"{abs(delta):.6g}; this narrows rather than supports the proposed direction."
            )
        else:
            kind = "boundary"
            claim = (
                f"Within run #{idea_idx}, the recorded search objective did not change "
                "between the first and last measurable stage."
            )
        insights.append(
            {
                "idea_idx": idea_idx,
                "title": title,
                "claim": claim,
                "kind": kind,
                "confidence": "low",
                "epistemic_status": "machine_synthesized_unverified",
                "evidence_refs": sorted(set(evidence_refs)),
                "rival_hypotheses": default_rivals,
                "why_it_matters": (
                    "It updates which branch deserves replication without promoting the "
                    "result beyond the evidence available in this run."
                ),
                "uncertainty": (
                    "Independent replication, multi-seed uncertainty, external validity, "
                    "and alternative mechanisms remain unresolved."
                ),
                "next_experiment": {
                    "question": "Does the result survive a preregistered paired replication?",
                    "design": (
                        "Run the primary and null/rival mechanism under matched seeds, add a "
                        "negative control, and retain all failed or null outcomes."
                    ),
                    "expected_information_gain": (
                        "Separates a stable mechanism signal from variance, measurement "
                        "artifact, and scope-bound effects."
                    ),
                },
            }
        )
    return insights


def validate_insights(
    candidate: Any,
    *,
    allowed_refs: set[str],
    valid_idea_indices: set[int],
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = candidate.get("insights") if isinstance(candidate, dict) else None
    if not isinstance(payload, list):
        return [], ["model_output_missing_insights_array"]
    clean: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, item in enumerate(payload[:12]):
        if not isinstance(item, dict):
            warnings.append(f"insight_{index}_not_object")
            continue
        try:
            idea_idx = int(item.get("idea_idx"))
        except (TypeError, ValueError):
            warnings.append(f"insight_{index}_invalid_idea_idx")
            continue
        refs = sorted(set(str(ref) for ref in item.get("evidence_refs") or []))
        next_experiment = item.get("next_experiment")
        claim = str(item.get("claim") or "").strip()
        rivals = [str(value).strip() for value in item.get("rival_hypotheses") or []]
        required_next = isinstance(next_experiment, dict) and all(
            str(next_experiment.get(key) or "").strip()
            for key in ("question", "design", "expected_information_gain")
        )
        if (
            idea_idx not in valid_idea_indices
            or not claim
            or FORBIDDEN_CERTAINTY.search(claim)
            or not refs
            or any(ref not in allowed_refs for ref in refs)
            or str(item.get("kind")) not in ALLOWED_KINDS
            or str(item.get("confidence")) not in ALLOWED_CONFIDENCE
            or not rivals
            or not required_next
            or not str(item.get("uncertainty") or "").strip()
            or not str(item.get("why_it_matters") or "").strip()
        ):
            warnings.append(f"insight_{index}_failed_evidence_contract")
            continue
        clean.append(
            {
                "idea_idx": idea_idx,
                "title": str(item.get("title") or f"Idea {idea_idx}").strip(),
                "claim": claim,
                "kind": str(item["kind"]),
                "confidence": str(item["confidence"]),
                "epistemic_status": "machine_synthesized_unverified",
                "evidence_refs": refs,
                "rival_hypotheses": rivals[:6],
                "why_it_matters": str(item["why_it_matters"]).strip(),
                "uncertainty": str(item["uncertainty"]).strip(),
                "next_experiment": {
                    key: str(next_experiment[key]).strip()
                    for key in ("question", "design", "expected_information_gain")
                },
            }
        )
    if not clean:
        warnings.append("no_model_insight_passed_contract")
    return clean, warnings


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Project Insight Report",
        "",
        "> These are machine-synthesized, run-internal candidate insights. They are not independently verified.",
        "",
        f"- Synthesis mode: `{report.get('synthesis_mode')}`",
        f"- Evidence pack: `{report.get('evidence_pack_hash')}`",
        f"- Report: `{report.get('report_hash')}`",
        "",
    ]
    for insight in report.get("insights") or []:
        lines.extend(
            [
                f"## {insight.get('title')}",
                "",
                f"- Candidate claim: {insight.get('claim')}",
                f"- Kind / confidence: {insight.get('kind')} / {insight.get('confidence')}",
                f"- Evidence: {', '.join(insight.get('evidence_refs') or [])}",
                f"- Why it matters: {insight.get('why_it_matters')}",
                f"- Uncertainty: {insight.get('uncertainty')}",
                "- Rival hypotheses:",
            ]
        )
        lines.extend(f"  - {item}" for item in insight.get("rival_hypotheses") or [])
        next_exp = insight.get("next_experiment") or {}
        lines.extend(
            [
                "- Next high-information experiment:",
                f"  - Question: {next_exp.get('question')}",
                f"  - Design: {next_exp.get('design')}",
                f"  - Expected information gain: {next_exp.get('expected_information_gain')}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def synthesize_project_insights(
    project_root: str | Path,
    results: list[dict[str, Any]],
    *,
    model: str | None = None,
    use_llm: bool = False,
    client_factory: Callable[[str], tuple[Any, str]] | None = None,
    response_fn: Callable[..., tuple[str, list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    pack = build_project_evidence_pack(project_root, results)
    fallback = _deterministic_insights(pack)
    insights = fallback
    mode = "deterministic"
    warnings: list[str] = []

    if use_llm and model:
        try:
            if client_factory is None or response_fn is None:
                from ai_scientist.llm import create_client, get_response_from_llm

                client_factory = client_factory or create_client
                response_fn = response_fn or get_response_from_llm
            client, client_model = client_factory(model)
            prompt = (
                "Synthesize high-information scientific candidate insights from the JSON "
                "evidence pack below. Return one fenced JSON object with an `insights` array. "
                "Every item must contain idea_idx, title, claim, kind "
                "(mechanism|boundary|negative|comparative|methodological), confidence "
                "(low|medium only), evidence_refs (only exact allowed selectors), at least one "
                "rival_hypothesis, why_it_matters, uncertainty, and next_experiment with "
                "question/design/expected_information_gain. Scope claims to these runs. Never "
                "say proven, confirmed, conclusive, or independently verified. Prefer results "
                "that distinguish mechanisms or boundaries over generic summaries.\n\n"
                + json.dumps(pack, ensure_ascii=False, sort_keys=True)
            )
            raw, _history = response_fn(
                prompt=prompt,
                client=client,
                model=client_model,
                system_message=(
                    "You are a skeptical scientific synthesis board. Evidence selectors and "
                    "epistemic calibration are hard constraints."
                ),
                temperature=0.2,
            )
            from ai_scientist.llm import extract_json_between_markers

            parsed = extract_json_between_markers(raw)
            validated, validation_warnings = validate_insights(
                parsed,
                allowed_refs=set(pack["allowed_evidence_refs"]),
                valid_idea_indices={
                    int(item["idea_idx"]) for item in pack.get("results") or []
                },
            )
            warnings.extend(validation_warnings)
            if validated:
                insights = validated
                mode = "llm_evidence_bound"
            else:
                mode = "deterministic_fallback"
        except Exception as exc:  # keep the completed research artifacts usable
            warnings.append(f"llm_synthesis_failed:{type(exc).__name__}")
            mode = "deterministic_fallback"

    report: dict[str, Any] = {
        "schema_version": INSIGHT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "epistemic_status": "machine_synthesized_unverified",
        "independent_verification": False,
        "synthesis_mode": mode,
        "evidence_pack_hash": canonical_content_hash(pack),
        "evidence_selectors": pack["allowed_evidence_refs"],
        "insights": insights,
        "global_limitations": [
            "No internal reviewer counts as an independent verifier.",
            "Claims remain scoped to recorded runs until external replication.",
            "Model-generated interpretation may omit plausible rival mechanisms.",
        ],
        "warnings": sorted(set(warnings)),
    }
    report["report_hash"] = canonical_content_hash(report)
    save_contract_artifact(
        project_root,
        "insight_report",
        report,
        producer="run_project.insight_synthesis",
        depends_on=["idea_cards", "research_plan", "experiment_registry"],
        warnings=report["warnings"],
        notes="Machine-synthesized candidate insights; independent verification remains false.",
    )
    markdown_path = Path(project_root) / "04_logs" / "insight_report.md"
    atomic_write_text(markdown_path, _render_markdown(report))
    return report


__all__ = [
    "INSIGHT_SCHEMA_VERSION",
    "build_project_evidence_pack",
    "synthesize_project_insights",
    "validate_insights",
]
