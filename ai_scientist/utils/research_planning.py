from __future__ import annotations

"""Helpers for turning raw ideas into explicit planning artifacts."""

import re
from typing import Any

from ai_scientist.utils.truth_contracts import (
    HIGH_RISK_WORKFLOW_MODES,
    build_truth_contract,
    derive_hallucination_checks,
)
from ai_scientist.utils.workflow_execution_policy import (
    build_workflow_execution_policy,
    policy_snapshot,
)

DEFAULT_EXPERIMENT_BUDGET = {
    "max_steps": 12,
    "max_wallclock_minutes": 90,
    "max_retry_per_task": 2,
}

STRUCTURED_EVIDENCE_PORTFOLIO_VENUES = frozenset({"neurips", "icml"})
EVIDENCE_PORTFOLIO_ROLES = ("primary", "ablation", "robustness")

_ABLATION_PLAN_MARKERS = (
    "ablation",
    "without ",
    "w/o",
    "remove ",
    "disable ",
    "component contribution",
)
_ROBUSTNESS_PLAN_MARKERS = (
    "robustness",
    "sensitivity",
    "out-of-distribution",
    "out of distribution",
    "ood",
    "stress test",
    "boundary condition",
    "distribution shift",
)

SOCRATIC_CHALLENGE_POLICY = {
    "minimum_rival_hypotheses": 3,
    "minimum_discriminating_tests": 3,
    "require_null_explanation": True,
    "require_measurement_artifact_explanation": True,
    "require_scope_boundary_explanation": True,
    "posterior_update_requires_evidence": True,
}

DEFAULT_AGENT_LANES = {
    "classic_pipeline": ("planner", "experiment", "writer", "reviewer"),
    "agentic_tree": (
        "planner",
        "experiment_manager",
        "experiment_worker",
        "writer",
        "reviewer",
    ),
    "program_driven": ("planner", "experiment", "writer", "reviewer", "repair"),
    "writing_studio": (
        "planner",
        "results_analyst",
        "storyline_editor",
        "latex_hygiene_editor",
        "humanizer",
        "reviewer",
    ),
    "review_board": (
        "planner",
        "experiment",
        "writer",
        "reviewer_board",
        "repair",
    ),
    "multi_agent_board": (
        "planner",
        "experiment_manager",
        "experiment_worker",
        "results_analyst",
        "storyline_editor",
        "latex_hygiene_editor",
        "humanizer",
        "quality_gate",
        "reviewer_board",
        "hostile_critic",
        "repair",
    ),
}

AGENT_LANE_SPECS = {
    "planner": {
        "responsibility": "Owns the research program, dependencies, acceptance rules, and kill criteria.",
        "inputs": ["idea_card", "research_program", "submission_policy"],
        "outputs": ["research_plan", "claim_priorities", "kill_criteria"],
        "handoff_gate": "Research tasks are scoped, claim-linked, and budgeted before execution starts.",
    },
    "experiment_manager": {
        "responsibility": "Owns experiment ordering, pruning, budget discipline, and keep/discard/crash decisions.",
        "inputs": ["research_plan", "experiment_registry", "claim_evidence_graph"],
        "outputs": ["execution_queue", "branch_decisions", "evidence_board"],
        "handoff_gate": "Only baseline-comparable branches with claim value continue into the manuscript candidate set.",
    },
    "experiment_worker": {
        "responsibility": "Executes bounded experiment branches and reports evidence without rewriting the narrative.",
        "inputs": ["task_contract", "dataset_spec", "baseline_spec"],
        "outputs": ["run_logs", "metrics_summary", "artifact_candidates"],
        "handoff_gate": "Branch ends with keep/discard/crash and explicit evidence quality notes.",
    },
    "results_analyst": {
        "responsibility": "Converts experiment artifacts into evidence-faithful result paragraphs, captions, and claim cards.",
        "inputs": ["experiment_registry", "figure_spec", "claim_evidence_graph"],
        "outputs": [
            "claim_cards",
            "caption_briefs",
            "result_takeaways",
            "rival_hypothesis_outcomes",
        ],
        "handoff_gate": "Every surviving claim cites a metric delta and at least one figure/table candidate.",
    },
    "storyline_editor": {
        "responsibility": "Tightens contribution framing, novelty positioning, and claim scope without inventing evidence.",
        "inputs": ["claim_cards", "related_work_notes", "reviewer_findings"],
        "outputs": ["storyline_outline", "claim_scope_rewrites", "novelty_deltas"],
        "handoff_gate": "Frontmatter tells one coherent story and avoids unsupported breadth.",
    },
    "latex_hygiene_editor": {
        "responsibility": "Applies venue style, formatting, escaping, and structural cleanup.",
        "inputs": ["manuscript_draft", "venue_template", "figure_assets"],
        "outputs": ["clean_tex", "formatting_fixups"],
        "handoff_gate": "The manuscript compiles cleanly and matches venue hygiene requirements.",
    },
    "humanizer": {
        "responsibility": "Reduces generic LLM phrasing after technical content is frozen.",
        "inputs": ["near_final_manuscript"],
        "outputs": ["tone_polish", "redundancy_cuts"],
        "handoff_gate": "Polish improves readability without mutating evidence or claims.",
    },
    "quality_gate": {
        "responsibility": "Applies submission-grade quality and evidence checks before reviewer escalation.",
        "inputs": ["manuscript_state", "figure_spec", "claim_evidence_graph"],
        "outputs": ["quality_gate_report", "followup_focus"],
        "handoff_gate": "Claim, figure, and writing debt are explicit before the review board runs.",
    },
    "reviewer": {
        "responsibility": "Produces normal reviewer-style feedback and open questions.",
        "inputs": ["paper_pdf", "review_plan"],
        "outputs": ["review_feedback"],
        "handoff_gate": "Feedback is actionable, role-specific, and anchored to manuscript sections.",
    },
    "reviewer_board": {
        "responsibility": "Runs multi-role review and converts findings into repair-ready debt.",
        "inputs": ["paper_pdf", "review_plan", "claim_evidence_graph"],
        "outputs": ["review_state", "repair_queue", "blocker_map"],
        "handoff_gate": "Blockers have owners, targets, and verification paths.",
    },
    "hostile_critic": {
        "responsibility": "Read-only red-team reviewer trying to reject the paper with anchored blockers.",
        "inputs": ["paper_pdf", "review_state", "claim_evidence_graph"],
        "outputs": ["critic_findings", "reject_case_summary"],
        "handoff_gate": "Every surviving lead claim has withstood an adversarial reject case.",
    },
    "repair": {
        "responsibility": "Executes targeted fixes against explicit reviewer or critic blockers.",
        "inputs": ["repair_plan", "review_state", "critic_findings"],
        "outputs": ["repair_execution_log", "closure_evidence", "recheck_requests"],
        "handoff_gate": "Repairs are verified, not just edited.",
    },
    "experiment": {
        "responsibility": "Runs the planned experiments and preserves comparability metadata.",
        "inputs": ["research_plan", "dataset_spec", "baseline_spec"],
        "outputs": ["experiment_registry", "metric_traces", "candidate_figures"],
        "handoff_gate": "Each run records dataset, metric, baseline, and acceptance status.",
    },
    "writer": {
        "responsibility": "Drafts the manuscript around evidence-backed claims only.",
        "inputs": [
            "claim_cards",
            "figure_spec",
            "citation_pack",
            "socratic_challenge",
        ],
        "outputs": ["manuscript_draft", "section_claim_bindings"],
        "handoff_gate": "Every major claim is traceable to evidence and section placement.",
    },
}

WORKFLOW_PHASE_GATES = {
    "classic_pipeline": [
        "At least one baseline-comparable result survives into the writeup.",
        "Final review must not leave unresolved critical evidence debt.",
    ],
    "agentic_tree": [
        "Keep redirected or negative branches when they sharpen the hypothesis boundary.",
        "Promote only branches with a clear novelty direction into the main storyline.",
    ],
    "program_driven": [
        "Every task must keep its success criterion, stop condition, and retry budget visible.",
        "Budget overruns force a program rewrite instead of silent continuation.",
    ],
    "writing_studio": [
        "Each core claim needs a figure/table path before the final polish pass.",
        "Frontmatter must surface the strongest numeric takeaways and scope caveats.",
    ],
    "review_board": [
        "Reviewer blockers must bind to a claim, section, or figure owner before repair starts.",
        "Repairs need verification checks, not just rewrite suggestions.",
    ],
    "multi_agent_board": [
        "Lead claims must bind to baseline delta, experiment record, figure/table path, and citation support.",
        "Hostile critic blockers stay read-only and must either trigger repair or block readiness.",
        "The final board recheck requires both reviewer-board and critic-board clearance.",
    ],
}

WORKFLOW_REVIEW_BUNDLES = {
    "classic_pipeline": {
        "improvement_roles": ["rigor"],
        "final_roles": ["clarity"],
        "critic_roles": [],
    },
    "agentic_tree": {
        "improvement_roles": ["novelty", "rigor"],
        "final_roles": ["clarity", "reproducibility"],
        "critic_roles": [],
    },
    "program_driven": {
        "improvement_roles": ["rigor", "reproducibility"],
        "final_roles": ["clarity", "reproducibility"],
        "critic_roles": [],
    },
    "writing_studio": {
        "improvement_roles": ["clarity", "rigor"],
        "final_roles": ["clarity"],
        "critic_roles": [],
    },
    "review_board": {
        "improvement_roles": [
            "novelty",
            "rigor",
            "clarity",
            "reproducibility",
            "claim_cross_examiner",
        ],
        "final_roles": [
            "novelty",
            "rigor",
            "clarity",
            "reproducibility",
            "skeptical_pc_member",
        ],
        "critic_roles": [],
    },
    "multi_agent_board": {
        "improvement_roles": [
            "novelty",
            "rigor",
            "clarity",
            "reproducibility",
            "claim_cross_examiner",
        ],
        "final_roles": [
            "novelty",
            "rigor",
            "clarity",
            "reproducibility",
            "skeptical_pc_member",
            "meta_reviewer",
        ],
        "critic_roles": [
            "skeptical_pc_member",
            "claim_cross_examiner",
            "reproducibility_assassin",
            "novelty_executioner",
            "stats_sniper",
            "related_work_skeptic",
            "meta_reviewer",
            "desk_reject_editor",
        ],
    },
}


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    split_items = re.split(r"[\n;]+", text)
    return [
        item.strip("-* ").strip() for item in split_items if item.strip("-* ").strip()
    ]


def _structured_evidence_portfolio_required(
    *,
    target_venue: str | None,
    submission_mode: bool,
    high_quality_mode: bool,
) -> bool:
    venue = str(target_venue or "").strip().lower()
    return venue in STRUCTURED_EVIDENCE_PORTFOLIO_VENUES and bool(
        submission_mode or high_quality_mode
    )


def _build_evidence_task_specs(
    task_descriptions: list[str],
    *,
    portfolio_required: bool,
) -> list[dict[str, Any]]:
    """Attach explicit scientific roles and fill missing top-venue controls."""

    specs = [
        {
            "description": str(description).strip(),
            "evidence_role": None,
            "paired_control_task_id": None,
            "intervention_variant": None,
            "stress_condition": None,
        }
        for description in task_descriptions
        if str(description).strip()
    ]
    if not portfolio_required:
        return specs

    specs[0].update(
        {
            "evidence_role": "primary",
            "intervention_variant": "full_method",
        }
    )
    seen_roles = {"primary"}
    for spec in specs[1:]:
        description = str(spec["description"])
        lowered = description.lower()
        if any(marker in lowered for marker in _ABLATION_PLAN_MARKERS):
            spec.update(
                {
                    "evidence_role": "ablation",
                    "paired_control_task_id": "task_0",
                    "intervention_variant": description,
                }
            )
            seen_roles.add("ablation")
        elif any(marker in lowered for marker in _ROBUSTNESS_PLAN_MARKERS):
            spec.update(
                {
                    "evidence_role": "robustness",
                    "paired_control_task_id": "task_0",
                    "intervention_variant": "full_method",
                    "stress_condition": description,
                }
            )
            seen_roles.add("robustness")
        else:
            spec["evidence_role"] = "supporting"

    if "ablation" not in seen_roles:
        specs.append(
            {
                "description": (
                    "Run a controlled ablation of the key proposed component while "
                    "holding the primary dataset, metric, and evaluation protocol fixed."
                ),
                "evidence_role": "ablation",
                "paired_control_task_id": "task_0",
                "intervention_variant": "remove_or_disable_key_component",
                "stress_condition": None,
            }
        )
    if "robustness" not in seen_roles:
        specs.append(
            {
                "description": (
                    "Run a robustness stress test under a declared distribution shift, "
                    "perturbation, or boundary condition while preserving the primary metric."
                ),
                "evidence_role": "robustness",
                "paired_control_task_id": "task_0",
                "intervention_variant": "full_method",
                "stress_condition": "declared_distribution_shift_or_boundary_condition",
            }
        )
    return specs


def _extract_keywords(text: str, *, prefix: str, limit: int = 4) -> list[str]:
    lowered = str(text or "")
    matches = re.findall(
        rf"{prefix}\s*[:=]?\s*([A-Za-z0-9_\-./ ]+)", lowered, flags=re.IGNORECASE
    )
    cleaned: list[str] = []
    for match in matches:
        item = str(match).strip().strip(".,")
        if not item or item in cleaned:
            continue
        cleaned.append(item)
        if len(cleaned) >= limit:
            break
    return cleaned


def _infer_candidate_datasets(idea: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(idea.get("Experiments") or ""),
            str(idea.get("Abstract") or ""),
            str(idea.get("Related Work") or ""),
        ]
    )
    datasets = _extract_keywords(text, prefix="dataset", limit=6)
    if datasets:
        return datasets
    return ["dataset_to_be_selected"]


def _infer_candidate_metrics(idea: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [str(idea.get("Experiments") or ""), str(idea.get("Abstract") or "")]
    )
    metrics = _extract_keywords(text, prefix="metric", limit=6)
    if metrics:
        return metrics
    fallback = []
    lowered = text.lower()
    if "accuracy" in lowered:
        fallback.append("accuracy")
    if "f1" in lowered:
        fallback.append("f1")
    if "auc" in lowered:
        fallback.append("auc")
    return fallback or ["primary_task_metric"]


def _infer_candidate_baselines(idea: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [str(idea.get("Experiments") or ""), str(idea.get("Related Work") or "")]
    )
    baselines = _extract_keywords(text, prefix="baseline", limit=6)
    return baselines or ["strong_existing_baseline"]


def _infer_failure_criteria(idea: dict[str, Any], metrics: list[str]) -> list[str]:
    falsifiers = _coerce_list(idea.get("Falsifiers"))
    if falsifiers:
        return falsifiers[:6]
    risks = _coerce_list(idea.get("Risk Factors and Limitations"))
    if risks:
        return [f"Risk-triggered failure: {risk}" for risk in risks[:4]]
    metric_name = metrics[0] if metrics else "primary metric"
    return [
        f"No credible gain or insight on {metric_name}.",
        "Evidence remains too weak to support the main claim.",
    ]


def _build_socratic_challenge(
    idea_card: dict[str, Any],
    *,
    datasets: list[str],
    metrics: list[str],
    baselines: list[str],
    failure_criteria: list[str],
) -> dict[str, Any]:
    """Turn one favored hypothesis into a pre-experiment adversarial contract."""

    primary = str(
        idea_card.get("core_hypothesis") or idea_card.get("title") or ""
    ).strip()
    mechanism = str(idea_card.get("mechanism") or "mechanism_not_yet_resolved").strip()
    dataset = str((datasets or ["dataset_to_be_selected"])[0])
    metric = str((metrics or ["primary_task_metric"])[0])
    baseline = str((baselines or ["strong_existing_baseline"])[0])
    source_idea = (
        idea_card.get("source_idea")
        if isinstance(idea_card.get("source_idea"), dict)
        else {}
    )
    supplied_rivals = _coerce_list(
        source_idea.get("Alternative Hypotheses") or source_idea.get("Rival Hypotheses")
    )
    default_rivals = [
        {
            "rival_id": "rival_null",
            "class": "null_effect",
            "statement": (
                f"The proposed method has no reliable advantage over {baseline}; "
                "the observed delta is compatible with run-to-run variation."
            ),
            "discriminating_prediction": (
                f"A paired, repeated comparison on {dataset} does not show a stable "
                f"improvement in {metric}."
            ),
        },
        {
            "rival_id": "rival_substitute_mechanism",
            "class": "mechanism_substitution",
            "statement": (
                "Any gain comes from matched compute, tuning, preprocessing, or data "
                f"exposure rather than the proposed mechanism ({mechanism})."
            ),
            "discriminating_prediction": (
                "Removing only the proposed mechanism while holding budget and data "
                "fixed preserves the gain."
            ),
        },
        {
            "rival_id": "rival_measurement_artifact",
            "class": "measurement_artifact",
            "statement": (
                "The apparent result is produced by leakage, selection, metric choice, "
                "or evaluator coupling rather than a scientific effect."
            ),
            "discriminating_prediction": (
                "Leakage checks, a negative control, or an independent metric/evaluator "
                "removes the apparent advantage."
            ),
        },
        {
            "rival_id": "rival_scope_boundary",
            "class": "scope_boundary",
            "statement": (
                f"The result is specific to {dataset} or the tested regime and does not "
                "support the broader claim."
            ),
            "discriminating_prediction": (
                "The effect collapses under a declared distribution shift, perturbation, "
                "or second dataset/regime."
            ),
        },
    ]
    rivals = [dict(item, source="protocol_default") for item in default_rivals]
    for index, statement in enumerate(supplied_rivals):
        rivals.append(
            {
                "rival_id": f"rival_user_{index}",
                "class": "domain_rival",
                "statement": statement,
                "discriminating_prediction": (
                    "Specify an observation that separates this rival from the primary "
                    "hypothesis before confirmatory execution."
                ),
                "source": "researcher_supplied",
            }
        )
    tests = [
        {
            "test_id": "socratic_paired_baseline",
            "targets": ["rival_null"],
            "design": (
                f"Run paired repeated comparisons against {baseline} on {dataset} using "
                f"the preregistered {metric} analysis."
            ),
            "required_controls": ["matched budget", "matched data", "seed sensitivity"],
        },
        {
            "test_id": "socratic_mechanism_ablation",
            "targets": ["rival_substitute_mechanism"],
            "design": (
                "Remove only the proposed causal mechanism and preserve all other "
                "training, inference, and evaluation conditions."
            ),
            "required_controls": ["single-change attribution", "compute parity"],
        },
        {
            "test_id": "socratic_negative_control",
            "targets": ["rival_measurement_artifact"],
            "design": (
                "Run leakage and selection audits plus a negative control and an "
                "independent metric or evaluator where feasible."
            ),
            "required_controls": ["negative control", "independent evaluation"],
        },
        {
            "test_id": "socratic_boundary_probe",
            "targets": ["rival_scope_boundary"],
            "design": (
                "Probe one declared distribution shift, perturbation, subgroup, or "
                "second regime and record the claim boundary even when it fails."
            ),
            "required_controls": ["predeclared boundary", "negative-result retention"],
        },
    ]
    for index, statement in enumerate(supplied_rivals):
        tests.append(
            {
                "test_id": f"socratic_user_rival_{index}",
                "targets": [f"rival_user_{index}"],
                "design": (
                    "Preregister an observation that would distinguish the primary "
                    f"hypothesis from this researcher-supplied rival: {statement}"
                ),
                "required_controls": [
                    "researcher-approved discriminator",
                    "evidence-linked decision",
                ],
            }
        )
    return {
        "policy": dict(SOCRATIC_CHALLENGE_POLICY),
        "primary_hypothesis": primary,
        "proposed_mechanism": mechanism,
        "rival_hypotheses": rivals,
        "discriminating_tests": tests,
        "falsifiers": list(failure_criteria),
        "uncertainty_contract": {
            "prior_confidence": None,
            "posterior_confidence": None,
            "update_status": "awaiting_evidence",
            "required_update": (
                "Report which primary or rival hypothesis gained or lost support, with "
                "evidence references and calibrated uncertainty."
            ),
            "self_score_is_evidence": False,
        },
        "status": "planned",
    }


def validate_socratic_challenge(payload: dict[str, Any] | None) -> dict[str, Any]:
    challenge = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    if challenge.get("policy") != SOCRATIC_CHALLENGE_POLICY:
        errors.append("policy_modified_or_missing")
    if not str(challenge.get("primary_hypothesis") or "").strip():
        errors.append("primary_hypothesis_missing")
    rivals = [
        item
        for item in challenge.get("rival_hypotheses") or []
        if isinstance(item, dict)
    ]
    tests = [
        item
        for item in challenge.get("discriminating_tests") or []
        if isinstance(item, dict)
    ]
    if len(rivals) < int(SOCRATIC_CHALLENGE_POLICY["minimum_rival_hypotheses"]):
        errors.append("rival_hypothesis_floor_not_met")
    if len(tests) < int(SOCRATIC_CHALLENGE_POLICY["minimum_discriminating_tests"]):
        errors.append("discriminating_test_floor_not_met")
    rival_ids = [str(item.get("rival_id") or "").strip() for item in rivals]
    if not all(rival_ids) or len(set(rival_ids)) != len(rival_ids):
        errors.append("rival_ids_invalid")
    required_classes = {"null_effect", "measurement_artifact", "scope_boundary"}
    present_classes = {str(item.get("class") or "") for item in rivals}
    if not required_classes.issubset(present_classes):
        errors.append("required_rival_classes_missing")
    covered_ids = {
        str(target).strip()
        for test in tests
        for target in (test.get("targets") or [])
        if str(target).strip()
    }
    if set(rival_ids) - covered_ids:
        errors.append("rival_without_discriminating_test")
    if any(
        not str(test.get("test_id") or "").strip()
        or not str(test.get("design") or "").strip()
        or not test.get("required_controls")
        for test in tests
    ):
        errors.append("discriminating_test_incomplete")
    uncertainty = (
        challenge.get("uncertainty_contract")
        if isinstance(challenge.get("uncertainty_contract"), dict)
        else {}
    )
    if uncertainty.get("self_score_is_evidence") is not False:
        errors.append("self_score_may_not_be_evidence")
    if not str(uncertainty.get("required_update") or "").strip():
        errors.append("uncertainty_update_missing")
    return {"passed": not errors, "errors": errors}


def _task_owner_for_workflow(workflow_mode: str, *, task_kind: str) -> str:
    if workflow_mode == "multi_agent_board":
        if task_kind == "review_hardening":
            return "experiment_manager"
        if task_kind == "evidence_pack":
            return "results_analyst"
        if task_kind in {"branch_probe", "exploration_seed"}:
            return "experiment_worker"
        return "experiment_manager"
    if workflow_mode == "writing_studio":
        return "results_analyst" if task_kind == "evidence_pack" else "experiment"
    if workflow_mode == "review_board":
        return "experiment"
    return "experiment"


def _lane_payload(lane: str) -> dict[str, Any]:
    spec = AGENT_LANE_SPECS.get(lane, {})
    return {
        "lane": lane,
        "responsibility": spec.get("responsibility", lane.replace("_", " ").title()),
        "inputs": list(spec.get("inputs") or []),
        "outputs": list(spec.get("outputs") or []),
        "handoff_gate": str(spec.get("handoff_gate") or "").strip() or None,
    }


def _build_agent_plan(
    *,
    workflow_mode: str,
    tasks: list[dict[str, Any]],
    execution_policy: dict[str, Any],
    failure_criteria: list[str],
) -> dict[str, Any]:
    lanes = list(
        DEFAULT_AGENT_LANES.get(workflow_mode, DEFAULT_AGENT_LANES["classic_pipeline"])
    )
    review_bundle = WORKFLOW_REVIEW_BUNDLES.get(
        workflow_mode,
        WORKFLOW_REVIEW_BUNDLES["classic_pipeline"],
    )
    return {
        "lanes": [_lane_payload(lane) for lane in lanes],
        "task_ownership": [
            {
                "task_id": task.get("task_id"),
                "owner": task.get("owner"),
                "dependencies": task.get("dependencies") or [],
                "evidence_role": task.get("evidence_role"),
                "paired_control_task_id": task.get("paired_control_task_id"),
                "intervention_variant": task.get("intervention_variant"),
                "stress_condition": task.get("stress_condition"),
                "claim_targets": task.get("claim_targets") or [],
                "kill_criteria": task.get("kill_criteria") or [],
                "required_inputs": task.get("required_inputs") or [],
                "produced_artifacts": task.get("produced_artifacts") or [],
                "verifier": task.get("verifier"),
                "escalation_lane": task.get("escalation_lane"),
            }
            for task in tasks
        ],
        "phase_gates": list(WORKFLOW_PHASE_GATES.get(workflow_mode, [])),
        "review_bundles": {
            "improvement_roles": list(review_bundle.get("improvement_roles") or []),
            "final_roles": list(review_bundle.get("final_roles") or []),
            "critic_roles": list(review_bundle.get("critic_roles") or []),
        },
        "keep_discard_policy": {
            "keep": "Evidence is baseline-comparable and materially strengthens or clarifies a target claim.",
            "discard": "Run fails acceptance checks, weakens the claim, or is dominated by a stronger comparable branch.",
            "crash": "Execution failed before producing trustworthy evidence; keep the trace but do not use it in the storyline.",
        },
        "failure_criteria": list(failure_criteria),
        "acceptance_rules": list(execution_policy.get("acceptance_rules") or []),
        "requires_hostile_critic": workflow_mode == "multi_agent_board",
    }


def build_idea_cards(
    ideas: list[dict[str, Any]],
    *,
    target_venue: str | None = None,
    template_profile: str = "open_ended",
    workflow_mode: str = "classic_pipeline",
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for idx, idea in enumerate(ideas):
        experiments = _coerce_list(idea.get("Experiments"))
        metrics = _infer_candidate_metrics(idea)
        datasets = _infer_candidate_datasets(idea)
        baselines = _infer_candidate_baselines(idea)
        literature_search = (
            idea.get("Literature Search")
            if isinstance(idea.get("Literature Search"), dict)
            else {}
        )
        literature_queries = [
            str(item).strip()
            for item in literature_search.get("queries") or []
            if str(item).strip()
        ]
        if not literature_queries:
            literature_queries = [
                str(item).strip()
                for item in [
                    idea.get("Title"),
                    idea.get("Name"),
                    idea.get("Short Hypothesis"),
                ]
                if str(item or "").strip()
            ][:3]
        card = {
            "idea_id": f"idea_{idx}",
            "name": idea.get("Name") or f"idea_{idx}",
            "title": idea.get("Title") or idea.get("Name") or f"Idea {idx}",
            "core_hypothesis": str(idea.get("Short Hypothesis") or "").strip(),
            "mechanism": str(idea.get("Mechanism") or "").strip(),
            "generation_operator": str(
                idea.get("Generation Operator") or "initial"
            ).strip(),
            "novelty_claim": str(idea.get("Related Work") or "").strip(),
            "related_work_notes": str(idea.get("Related Work") or "").strip(),
            "minimum_viable_experiment": (
                experiments[0]
                if experiments
                else "Run a first-pass experiment for the main claim."
            ),
            "candidate_datasets": datasets,
            "candidate_metrics": metrics,
            "candidate_baselines": baselines,
            "compute_risk": "moderate" if len(experiments) > 3 else "low",
            "failure_criteria": _infer_failure_criteria(idea, metrics),
            "negative_result_value": "Clarifies when the hypothesis fails and which evidence is still missing.",
            "literature_queries": literature_queries,
            "literature_search_verified": int(
                literature_search.get("successful_search_count") or 0
            )
            > 0,
            "supporting_papers": list(literature_search.get("evidence") or []),
            "target_venue": target_venue,
            "template_profile": template_profile,
            "workflow_mode": workflow_mode,
            "status": "proposed",
            "source_idea": idea,
        }
        cards.append(card)
    return cards


def build_research_plan(
    idea_card: dict[str, Any],
    *,
    target_venue: str | None = None,
    budget: dict[str, Any] | None = None,
    submission_mode: bool = False,
    breakthrough_mode: bool = False,
    high_quality_mode: bool = False,
) -> dict[str, Any]:
    workflow_mode = str(idea_card.get("workflow_mode") or "classic_pipeline")
    resolved_target_venue = (
        str(target_venue or idea_card.get("target_venue") or "").strip().lower()
    )
    execution_policy = build_workflow_execution_policy(
        workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=resolved_target_venue or None,
    )
    budget_payload = dict(DEFAULT_EXPERIMENT_BUDGET)
    budget_payload.update(execution_policy.budget)
    budget_payload.update(budget or {})
    task_descriptions = _coerce_list(
        idea_card.get("source_idea", {}).get("Experiments")
        or idea_card.get("minimum_viable_experiment")
    )
    if not task_descriptions:
        task_descriptions = ["Run the minimum viable experiment for the main claim."]
    portfolio_required = _structured_evidence_portfolio_required(
        target_venue=resolved_target_venue,
        submission_mode=submission_mode,
        high_quality_mode=high_quality_mode,
    )
    task_specs = _build_evidence_task_specs(
        task_descriptions,
        portfolio_required=portfolio_required,
    )

    datasets = list(idea_card.get("candidate_datasets") or ["dataset_to_be_selected"])
    metrics = list(idea_card.get("candidate_metrics") or ["primary_task_metric"])
    baselines = list(
        idea_card.get("candidate_baselines") or ["strong_existing_baseline"]
    )

    tasks: list[dict[str, Any]] = []
    failure_criteria = list(idea_card.get("failure_criteria") or [])
    socratic_challenge = _build_socratic_challenge(
        idea_card,
        datasets=datasets,
        metrics=metrics,
        baselines=baselines,
        failure_criteria=failure_criteria,
    )
    discriminating_tests = socratic_challenge["discriminating_tests"]
    for idx, task_spec in enumerate(task_specs):
        description = str(task_spec["description"])
        evidence_role = task_spec.get("evidence_role")
        paired_control_task_id = task_spec.get("paired_control_task_id")
        claim_id = f"claim_{idx}"
        if workflow_mode == "agentic_tree":
            task_kind = "branch_probe" if idx > 0 else "exploration_seed"
        elif workflow_mode == "program_driven":
            task_kind = "program_milestone"
        elif workflow_mode == "writing_studio":
            task_kind = "evidence_pack"
        elif workflow_mode == "review_board":
            task_kind = "review_hardening"
        else:
            task_kind = "core_experiment"
        owner = _task_owner_for_workflow(workflow_mode, task_kind=task_kind)
        dependencies = (
            []
            if idx == 0
            else (
                [str(paired_control_task_id)]
                if paired_control_task_id
                else [f"task_{idx - 1}"]
            )
        )
        escalation_lane = (
            "hostile_critic"
            if workflow_mode == "multi_agent_board"
            else ("reviewer_board" if workflow_mode == "review_board" else "reviewer")
        )
        discriminating_test = discriminating_tests[idx % len(discriminating_tests)]
        evidence_requirements = [
            "baseline-comparable metric delta",
            "claim-linked experiment record",
            "figure-or-table-ready artifact",
            "rival-hypothesis outcome with calibrated uncertainty",
        ]
        if evidence_role == "primary":
            evidence_requirements.extend(
                [
                    "numeric primary outcome",
                    "structured uncertainty or statistical estimate",
                ]
            )
        elif evidence_role == "ablation":
            evidence_requirements.extend(
                [
                    "paired completed primary control",
                    "declared intervention variant",
                    "numeric paired comparison",
                ]
            )
        elif evidence_role == "robustness":
            evidence_requirements.extend(
                [
                    "paired completed primary control",
                    "declared stress condition",
                    "numeric paired comparison",
                ]
            )
        dataset_index = (
            0 if evidence_role == "ablation" else min(idx, len(datasets) - 1)
        )
        metric_index = 0 if paired_control_task_id else min(idx, len(metrics) - 1)
        baseline_index = 0 if paired_control_task_id else min(idx, len(baselines) - 1)
        task = {
            "task_id": f"task_{idx}",
            "goal": description,
            "priority": "P0" if idx == 0 else "P1",
            "task_kind": task_kind,
            "evidence_role": evidence_role,
            "paired_control_task_id": paired_control_task_id,
            "intervention_variant": task_spec.get("intervention_variant"),
            "stress_condition": task_spec.get("stress_condition"),
            "owner": owner,
            "dependencies": dependencies,
            "dataset": datasets[dataset_index],
            "metric": metrics[metric_index],
            "baseline": baselines[baseline_index],
            "success_criterion": f"Produce evidence relevant to {claim_id} with a clear {metrics[metric_index]} outcome.",
            "stop_condition": "Stop when the claim is supported, weakened, or the budget is exhausted.",
            "branch_outcome_labels": ["keep", "discard", "crash"],
            "branch_keep_rule": (
                f"Keep only if the run is baseline-comparable and materially strengthens {claim_id}."
            ),
            "kill_criteria": failure_criteria[:2]
            or [
                "Stop the branch if the evidence does not materially support the target claim."
            ],
            "evidence_requirements": evidence_requirements,
            "expected_outputs": [
                "experiment logs",
                "summary json",
                "candidate figure inputs",
            ],
            "required_inputs": [
                "research_plan",
                f"dataset:{datasets[dataset_index]}",
                f"metric:{metrics[metric_index]}",
                f"baseline:{baselines[baseline_index]}",
                f"claim:{claim_id}",
            ],
            "produced_artifacts": [
                "experiment_registry_record",
                "run_summary_json",
                f"figure_candidate:{claim_id}",
                f"claim_note:{claim_id}",
                f"rival_hypothesis_decision:{claim_id}",
            ],
            "artifact_intent": (
                [
                    "claim_survival",
                    "figure_packaging",
                    "reviewer_rebuttal",
                ]
                if workflow_mode == "multi_agent_board"
                else ["claim_survival", "figure_packaging"]
            ),
            "verifier": (
                "quality_gate" if workflow_mode == "multi_agent_board" else "reviewer"
            ),
            "close_condition": (
                "The task is only done when the evidence either survives storyline selection "
                "or is explicitly discarded with a recorded reason."
            ),
            "closure_evidence_refs": [
                claim_id,
                f"dataset:{datasets[dataset_index]}",
                f"metric:{metrics[metric_index]}",
                f"baseline:{baselines[baseline_index]}",
            ],
            "escalation_lane": escalation_lane,
            "claim_targets": [claim_id],
            "socratic_challenge_refs": sorted(
                {
                    str(target)
                    for test in discriminating_tests
                    for target in test["targets"]
                }
            ),
            "discriminating_test": dict(discriminating_test),
            "required_discriminating_tests": [
                dict(test) for test in discriminating_tests
            ],
            "uncertainty_update_required": True,
            "budget": budget_payload,
            "acceptance_checks": list(execution_policy.acceptance_rules[:2]),
            "execution_style": execution_policy.execution_style,
            "status": "planned",
            "study_phase": "exploratory",
            "claim_promotion_gate": "independent_verification_required",
        }
        tasks.append(task)

    agent_plan = _build_agent_plan(
        workflow_mode=workflow_mode,
        tasks=tasks,
        execution_policy=policy_snapshot(execution_policy),
        failure_criteria=failure_criteria,
    )
    task_ids_by_evidence_role = {
        role: [
            str(task.get("task_id"))
            for task in tasks
            if task.get("evidence_role") == role
        ]
        for role in EVIDENCE_PORTFOLIO_ROLES
    }
    plan_payload = {
        "plan_id": f"{idea_card.get('idea_id')}_plan",
        "idea_id": idea_card.get("idea_id"),
        "idea_name": idea_card.get("name"),
        "workflow_mode": workflow_mode,
        "target_venue": resolved_target_venue or None,
        "budget": budget_payload,
        "execution_policy": policy_snapshot(execution_policy),
        "agent_plan": agent_plan,
        "integrity_policy": {
            "exploratory_runs_allowed": True,
            "preregistration_required_for_confirmatory": True,
            "blind_holdout_required": True,
            "deterministic_metric_verification_required": True,
            "independent_reproduction_required": True,
            "claim_promotion_requires_verified_report": True,
        },
        "socratic_challenge": socratic_challenge,
        "evidence_portfolio": {
            "required": portfolio_required,
            "target_venue": resolved_target_venue or None,
            "required_roles": (
                list(EVIDENCE_PORTFOLIO_ROLES) if portfolio_required else []
            ),
            "task_ids_by_role": task_ids_by_evidence_role,
            "pairing_policy": (
                "ablation_and_robustness_pair_to_completed_primary"
                if portfolio_required
                else "not_required"
            ),
            "numeric_evidence_required": portfolio_required,
            "statistical_evidence_required": portfolio_required,
        },
        "tasks": tasks,
    }
    truth_contract = build_truth_contract(idea_card, plan_payload)
    hallucination_checks = derive_hallucination_checks(truth_contract, plan_payload)
    checks_by_task: dict[str, list[dict[str, Any]]] = {}
    for check in hallucination_checks.get("checks") or []:
        if not isinstance(check, dict):
            continue
        task_id = str(check.get("task_id") or "").strip()
        if task_id:
            checks_by_task.setdefault(task_id, []).append(check)
    for task in tasks:
        task_id = str(task.get("task_id") or "").strip()
        task_checks = checks_by_task.get(task_id, [])
        high_risk_task = (
            workflow_mode in HIGH_RISK_WORKFLOW_MODES
            or str(task.get("priority") or "").strip().upper() == "P0"
            or str(task.get("escalation_lane") or "").strip() == "hostile_critic"
        )
        task["truth_contract_status"] = (
            "pending" if high_risk_task and task_checks else "not_applicable"
        )
        task["truth_contract_refs"] = (
            [
                f"{check.get('category')}:{check.get('evidence_source')}"
                for check in task_checks
            ]
            if high_risk_task
            else []
        )
        task["hallucination_checks"] = task_checks if high_risk_task else []
    plan_payload["truth_contract"] = truth_contract
    plan_payload["hallucination_checks"] = hallucination_checks
    return plan_payload


def build_claim_evidence_graph(
    idea_card: dict[str, Any],
    research_plan: dict[str, Any],
) -> dict[str, Any]:
    hypothesis_id = "hypothesis_0"
    nodes: list[dict[str, Any]] = [
        {
            "id": hypothesis_id,
            "type": "hypothesis",
            "label": idea_card.get("core_hypothesis") or idea_card.get("title"),
            "status": "proposed",
        }
    ]
    edges: list[dict[str, Any]] = []

    socratic_challenge = (
        research_plan.get("socratic_challenge")
        if isinstance(research_plan.get("socratic_challenge"), dict)
        else {}
    )
    rival_ids: list[str] = []
    for rival in socratic_challenge.get("rival_hypotheses") or []:
        if not isinstance(rival, dict):
            continue
        rival_id = str(rival.get("rival_id") or "").strip()
        if not rival_id:
            continue
        rival_ids.append(rival_id)
        nodes.append(
            {
                "id": rival_id,
                "type": "hypothesis",
                "label": rival.get("statement"),
                "status": "proposed",
                "hypothesis_role": "rival",
                "discriminating_prediction": rival.get("discriminating_prediction"),
            }
        )
        edges.append(
            {"source": rival_id, "target": hypothesis_id, "type": "contradicts"}
        )

    for task in research_plan.get("tasks", []):
        task_id = task["task_id"]
        claim_id = (task.get("claim_targets") or [f"{task_id}_claim"])[0]
        metric_id = f"{task_id}_metric"
        figure_id = f"{task_id}_figure"
        limitation_id = f"{task_id}_limitation"

        nodes.extend(
            [
                {
                    "id": task_id,
                    "type": "experiment",
                    "label": task.get("goal"),
                    "status": task.get("status", "planned"),
                    "evidence_role": task.get("evidence_role"),
                    "paired_control_task_id": task.get("paired_control_task_id"),
                    "intervention_variant": task.get("intervention_variant"),
                    "stress_condition": task.get("stress_condition"),
                },
                {
                    "id": metric_id,
                    "type": "metric",
                    "label": task.get("metric"),
                    "status": "planned",
                },
                {
                    "id": claim_id,
                    "type": "claim",
                    "label": f"Claim supported by {task.get('goal')}",
                    "status": "proposed",
                },
                {
                    "id": figure_id,
                    "type": "figure",
                    "label": f"Figure for {task.get('goal')}",
                    "status": "planned",
                },
                {
                    "id": limitation_id,
                    "type": "limitation",
                    "label": f"Boundary condition for {task.get('goal')}",
                    "status": "planned",
                },
            ]
        )
        edges.extend(
            [
                {"source": hypothesis_id, "target": task_id, "type": "tests"},
                {"source": task_id, "target": metric_id, "type": "supports"},
                {"source": metric_id, "target": claim_id, "type": "supports"},
                {"source": task_id, "target": figure_id, "type": "visualizes"},
                {"source": claim_id, "target": limitation_id, "type": "qualifies"},
            ]
        )
        for rival_id in task.get("socratic_challenge_refs") or []:
            if rival_id in rival_ids:
                edges.append(
                    {"source": rival_id, "target": task_id, "type": "tested_by"}
                )
        paired_control_task_id = str(task.get("paired_control_task_id") or "").strip()
        if paired_control_task_id:
            relation_type = (
                "ablates" if task.get("evidence_role") == "ablation" else "stress_tests"
            )
            edges.append(
                {
                    "source": task_id,
                    "target": paired_control_task_id,
                    "type": relation_type,
                }
            )

    return {
        "graph_id": f"{idea_card.get('idea_id')}_claim_graph",
        "idea_id": idea_card.get("idea_id"),
        "nodes": nodes,
        "edges": edges,
    }
