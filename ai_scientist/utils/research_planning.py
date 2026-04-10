from __future__ import annotations

"""Helpers for turning raw ideas into explicit planning artifacts."""

import re
from typing import Any

from ai_scientist.utils.workflow_execution_policy import (
    build_workflow_execution_policy,
    policy_snapshot,
)


DEFAULT_EXPERIMENT_BUDGET = {
    "max_steps": 12,
    "max_wallclock_minutes": 90,
    "max_retry_per_task": 2,
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
        "outputs": ["claim_cards", "caption_briefs", "result_takeaways"],
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
        "inputs": ["claim_cards", "figure_spec", "citation_pack"],
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
    return [item.strip("-* ").strip() for item in split_items if item.strip("-* ").strip()]


def _extract_keywords(text: str, *, prefix: str, limit: int = 4) -> list[str]:
    lowered = str(text or "")
    matches = re.findall(rf"{prefix}\s*[:=]?\s*([A-Za-z0-9_\-./ ]+)", lowered, flags=re.IGNORECASE)
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
    text = "\n".join([str(idea.get("Experiments") or ""), str(idea.get("Abstract") or "")])
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
    text = "\n".join([str(idea.get("Experiments") or ""), str(idea.get("Related Work") or "")])
    baselines = _extract_keywords(text, prefix="baseline", limit=6)
    return baselines or ["strong_existing_baseline"]


def _infer_failure_criteria(idea: dict[str, Any], metrics: list[str]) -> list[str]:
    risks = _coerce_list(idea.get("Risk Factors and Limitations"))
    if risks:
        return [f"Risk-triggered failure: {risk}" for risk in risks[:4]]
    metric_name = metrics[0] if metrics else "primary metric"
    return [
        f"No credible gain or insight on {metric_name}.",
        "Evidence remains too weak to support the main claim.",
    ]


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
    lanes = list(DEFAULT_AGENT_LANES.get(workflow_mode, DEFAULT_AGENT_LANES["classic_pipeline"]))
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
        card = {
            "idea_id": f"idea_{idx}",
            "name": idea.get("Name") or f"idea_{idx}",
            "title": idea.get("Title") or idea.get("Name") or f"Idea {idx}",
            "core_hypothesis": str(idea.get("Short Hypothesis") or "").strip(),
            "novelty_claim": str(idea.get("Related Work") or "").strip(),
            "related_work_notes": str(idea.get("Related Work") or "").strip(),
            "minimum_viable_experiment": experiments[0] if experiments else "Run a first-pass experiment for the main claim.",
            "candidate_datasets": datasets,
            "candidate_metrics": metrics,
            "candidate_baselines": baselines,
            "compute_risk": "moderate" if len(experiments) > 3 else "low",
            "failure_criteria": _infer_failure_criteria(idea, metrics),
            "negative_result_value": "Clarifies when the hypothesis fails and which evidence is still missing.",
            "literature_queries": [
                item for item in [idea.get("Title"), idea.get("Name"), idea.get("Short Hypothesis")] if str(item or "").strip()
            ][:3],
            "supporting_papers": [],
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
    execution_policy = build_workflow_execution_policy(
        workflow_mode,
        submission_mode=submission_mode,
        breakthrough_mode=breakthrough_mode,
        high_quality_mode=high_quality_mode,
        target_venue=target_venue or idea_card.get("target_venue"),
    )
    budget_payload = dict(DEFAULT_EXPERIMENT_BUDGET)
    budget_payload.update(execution_policy.budget)
    budget_payload.update(budget or {})
    task_descriptions = _coerce_list(
        idea_card.get("source_idea", {}).get("Experiments") or idea_card.get("minimum_viable_experiment")
    )
    if not task_descriptions:
        task_descriptions = ["Run the minimum viable experiment for the main claim."]

    datasets = list(idea_card.get("candidate_datasets") or ["dataset_to_be_selected"])
    metrics = list(idea_card.get("candidate_metrics") or ["primary_task_metric"])
    baselines = list(idea_card.get("candidate_baselines") or ["strong_existing_baseline"])

    tasks: list[dict[str, Any]] = []
    failure_criteria = list(idea_card.get("failure_criteria") or [])
    for idx, description in enumerate(task_descriptions):
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
        dependencies = [] if idx == 0 else [f"task_{idx - 1}"]
        escalation_lane = (
            "hostile_critic"
            if workflow_mode == "multi_agent_board"
            else ("reviewer_board" if workflow_mode == "review_board" else "reviewer")
        )
        task = {
            "task_id": f"task_{idx}",
            "goal": description,
            "priority": "P0" if idx == 0 else "P1",
            "task_kind": task_kind,
            "owner": owner,
            "dependencies": dependencies,
            "dataset": datasets[min(idx, len(datasets) - 1)],
            "metric": metrics[min(idx, len(metrics) - 1)],
            "baseline": baselines[min(idx, len(baselines) - 1)],
            "success_criterion": f"Produce evidence relevant to {claim_id} with a clear {metrics[min(idx, len(metrics) - 1)]} outcome.",
            "stop_condition": "Stop when the claim is supported, weakened, or the budget is exhausted.",
            "branch_outcome_labels": ["keep", "discard", "crash"],
            "branch_keep_rule": (
                f"Keep only if the run is baseline-comparable and materially strengthens {claim_id}."
            ),
            "kill_criteria": failure_criteria[:2]
            or ["Stop the branch if the evidence does not materially support the target claim."],
            "evidence_requirements": [
                "baseline-comparable metric delta",
                "claim-linked experiment record",
                "figure-or-table-ready artifact",
            ],
            "expected_outputs": [
                "experiment logs",
                "summary json",
                "candidate figure inputs",
            ],
            "required_inputs": [
                "research_plan",
                f"dataset:{datasets[min(idx, len(datasets) - 1)]}",
                f"metric:{metrics[min(idx, len(metrics) - 1)]}",
                f"baseline:{baselines[min(idx, len(baselines) - 1)]}",
                f"claim:{claim_id}",
            ],
            "produced_artifacts": [
                "experiment_registry_record",
                "run_summary_json",
                f"figure_candidate:{claim_id}",
                f"claim_note:{claim_id}",
            ],
            "artifact_intent": [
                "claim_survival",
                "figure_packaging",
                "reviewer_rebuttal",
            ]
            if workflow_mode == "multi_agent_board"
            else ["claim_survival", "figure_packaging"],
            "verifier": "quality_gate" if workflow_mode == "multi_agent_board" else "reviewer",
            "close_condition": (
                "The task is only done when the evidence either survives storyline selection "
                "or is explicitly discarded with a recorded reason."
            ),
            "closure_evidence_refs": [
                claim_id,
                f"dataset:{datasets[min(idx, len(datasets) - 1)]}",
                f"metric:{metrics[min(idx, len(metrics) - 1)]}",
                f"baseline:{baselines[min(idx, len(baselines) - 1)]}",
            ],
            "escalation_lane": escalation_lane,
            "claim_targets": [claim_id],
            "budget": budget_payload,
            "acceptance_checks": list(execution_policy.acceptance_rules[:2]),
            "execution_style": execution_policy.execution_style,
            "status": "planned",
        }
        tasks.append(task)

    agent_plan = _build_agent_plan(
        workflow_mode=workflow_mode,
        tasks=tasks,
        execution_policy=policy_snapshot(execution_policy),
        failure_criteria=failure_criteria,
    )

    return {
        "plan_id": f"{idea_card.get('idea_id')}_plan",
        "idea_id": idea_card.get("idea_id"),
        "idea_name": idea_card.get("name"),
        "workflow_mode": workflow_mode,
        "target_venue": target_venue or idea_card.get("target_venue"),
        "budget": budget_payload,
        "execution_policy": policy_snapshot(execution_policy),
        "agent_plan": agent_plan,
        "tasks": tasks,
    }


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

    return {
        "graph_id": f"{idea_card.get('idea_id')}_claim_graph",
        "idea_id": idea_card.get("idea_id"),
        "nodes": nodes,
        "edges": edges,
    }
