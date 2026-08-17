"""Provider-free product demo for the Research VCS and evidence DAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text

from .research_dag import export_research_dag
from .research_git import ResearchGitError
from .research_journey import build_research_guide, start_guided_research
from .research_lifecycle import ResearchLifecycle
from .research_vcs import ResearchRepository

DEMO_SCHEMA = "xscientist.demo.v1"
AUTOPILOT_FIXTURE_SCHEMA = "xscientist.autopilot-fixture.v1"


def _ensure_empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ResearchGitError(
            "demo destination must be absent or empty; choose a new directory"
        )


def create_demo(
    destination: str | Path,
    *,
    language: str = "auto",
    git_user_name: str | None = None,
    git_user_email: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic, contested research history and offline browser."""

    root = Path(destination).expanduser().resolve()
    _ensure_empty_destination(root)
    started = start_guided_research(
        root,
        question="Does retrieval reduce unsupported factual claims?",
        hypothesis=(
            "Retrieval lowers the unsupported-claim rate on both familiar and "
            "held-out question sets."
        ),
        falsifier=("Retrieval does not improve the held-out unsupported-claim rate."),
        name="retrieval-evidence-demo",
        actor="human:demo-researcher",
        language=language,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    repository = ResearchRepository(root)
    lifecycle = ResearchLifecycle(repository)
    hypothesis_id = str(started["hypothesis_id"])

    plan = repository.record(
        "research_plan",
        {
            "summary": "Compare retrieval with a no-retrieval baseline.",
            "tests": [
                "Measure unsupported claims on a familiar development set.",
                "Repeat the locked comparison on a held-out question set.",
            ],
            "study_phase": "exploratory",
        },
        relations=[{"type": "depends_on", "target": hypothesis_id}],
        actor={"actor_id": "human:demo-researcher", "authority": "human"},
    )

    failed_attempt = lifecycle.experiment_attempt(
        {
            "name": "initial parser-backed evaluation",
            "status": "failed",
            "study_phase": "exploratory",
            "failure_reason": "The first fixture contained malformed records.",
            "seed": 17,
        },
        plan_id=plan.object_id,
        commit=False,
    )["attempt"]
    development_attempt = lifecycle.experiment_attempt(
        {
            "name": "development-set retrieval comparison",
            "status": "completed",
            "study_phase": "exploratory",
            "dataset": "bundled-demo-development",
            "seed": 42,
            "metrics": {
                "retrieval_unsupported_rate": 0.18,
                "baseline_unsupported_rate": 0.31,
            },
        },
        plan_id=plan.object_id,
        commit=False,
    )["attempt"]
    supporting = lifecycle.evidence(
        {
            "summary": "Retrieval reduced unsupported claims on development data.",
            "effect": -0.13,
            "metric": "unsupported_claim_rate",
            "scope": "development set",
        },
        attempt_ids=[development_attempt.object_id],
        supports=[hypothesis_id],
        commit=False,
    )["evidence"]
    claim = lifecycle.claim(
        {
            "statement": "Retrieval reduces unsupported factual claims.",
            "depth_level": "descriptive",
            "scope": "development and held-out question sets",
            "epistemic_status": "contested",
        },
        evidence_ids=[supporting.object_id],
        verified=False,
        commit=False,
    )["claim"]

    heldout_attempt = lifecycle.experiment_attempt(
        {
            "name": "held-out retrieval comparison",
            "status": "completed",
            "study_phase": "exploratory",
            "dataset": "bundled-demo-heldout",
            "seed": 43,
            "metrics": {
                "retrieval_unsupported_rate": 0.29,
                "baseline_unsupported_rate": 0.27,
            },
        },
        plan_id=plan.object_id,
        commit=False,
    )["attempt"]
    refuting = lifecycle.evidence(
        {
            "summary": "The development benefit did not transfer to held-out data.",
            "effect": 0.02,
            "metric": "unsupported_claim_rate",
            "scope": "held-out set",
        },
        attempt_ids=[heldout_attempt.object_id],
        refutes=[hypothesis_id, claim.object_id],
        commit=False,
    )["evidence"]
    from .research_commands import save_inference

    inference_result = save_inference(
        str(root),
        statement=(
            "Retrieval improved the bundled development fixture, but the "
            "held-out result does not support a transferable reduction in "
            "unsupported factual claims."
        ),
        premises=[supporting.object_id, refuting.object_id],
        warrant=(
            "A claim about transfer requires improvement on the held-out "
            "condition; a development-only gain cannot outweigh a reversed "
            "held-out effect."
        ),
        message="record the bounded contested-evidence inference",
        commit=False,
    )
    inference = inference_result["object"]
    review = lifecycle.evaluation(
        {
            "status": "rejected",
            "claim_promotion_allowed": False,
            "required_failures": [
                "heldout_result_refutes_transfer",
                "independent_reproduction_missing",
            ],
            "summary": (
                "Hold the broad claim: development support conflicts with the "
                "held-out result."
            ),
        },
        evaluates=[
            claim.object_id,
            inference.object_id,
            supporting.object_id,
            refuting.object_id,
        ],
        verifier_id="human:demo-independent-reviewer",
        commit=False,
    )
    manuscript = lifecycle.manuscript(
        {
            "title": "A contested retrieval result",
            "status": "draft",
            "conclusion": (
                "The observed development-set gain is not yet transferable."
            ),
        },
        claim_ids=[claim.object_id],
        gate_id=review["gate"].object_id,
        final=False,
        commit=False,
    )["manuscript"]
    checkpoint = repository.commit(
        stage="paper",
        subject="complete provider-free contested-evidence demo",
        status="draft",
    )

    exported = export_research_dag(root, root / "research-dag")
    graph = exported["graph"]
    guide = build_research_guide(root, language=language)
    return {
        "schema": DEMO_SCHEMA,
        "ok": True,
        "repository": root.as_posix(),
        "checkpoint_id": checkpoint.checkpoint_id,
        "objects": {
            "hypothesis": hypothesis_id,
            "plan": plan.object_id,
            "failed_attempt": failed_attempt.object_id,
            "supporting_evidence": supporting.object_id,
            "refuting_evidence": refuting.object_id,
            "bounded_inference": inference.object_id,
            "contested_claim": claim.object_id,
            "review": review["review"].object_id,
            "context": review["context"].object_id,
            "hold_gate": review["gate"].object_id,
            "manuscript": manuscript.object_id,
        },
        "dag": {
            "json": exported["json"],
            "html": exported["html"],
            "nodes": len(graph.get("nodes") or []),
            "relations": len(graph.get("edges") or []),
            "integrity_ok": bool((graph.get("integrity") or {}).get("is_dag")),
            "closure": (graph.get("scientific_closure") or {}).get("status"),
        },
        "guide": guide,
        "network_used": False,
        "provider_used": False,
        "cost_usd": 0.0,
    }


def create_autopilot_demo(
    destination: str | Path,
    *,
    profile: str = "balanced",
    language: str = "auto",
    git_user_name: str | None = None,
    git_user_email: str | None = None,
) -> dict[str, Any]:
    """Exercise the installed Autopilot output contract with trusted fixtures.

    The fixture deliberately does not pretend to validate a model or generated
    program.  A bundled deterministic provider and trusted tiny executor emit
    the same progress, budget, negative-result, insight, and DAG surfaces used
    by a paid run, for zero cost and without network access.
    """

    normalized_profile = str(profile or "balanced").strip().lower()
    if normalized_profile not in {"balanced", "discovery", "publication"}:
        raise ResearchGitError("autopilot fixture profile is invalid")
    result = create_demo(
        destination,
        language=language,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )
    root = Path(destination).expanduser().resolve()
    logs = root / "04_logs"
    experiment = root / "02_experiments" / "offline-autopilot-fixture"
    experiment.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        experiment / "code.py",
        (
            "# Bundled trusted fixture; no generated code is executed.\n"
            "development_effect = 0.18 - 0.31\n"
            "heldout_effect = 0.29 - 0.27\n"
            "print({'development_effect': development_effect, "
            "'heldout_effect': heldout_effect})\n"
        ),
    )
    atomic_write_text(
        experiment / "term_out.log",
        "{'development_effect': -0.13, 'heldout_effect': 0.02}\n",
    )
    atomic_write_json(
        experiment / "metrics.json",
        {
            "development_effect": -0.13,
            "heldout_effect": 0.02,
            "primary_metric": "unsupported_claim_rate",
            "is_buggy": False,
        },
    )
    progress = {
        "schema": "xscientist.project-progress.v1",
        "current_stage": "complete",
        "selected_indices": [0],
        "results": [
            {
                "idea_idx": 0,
                "status": "failed",
                "reason": "malformed fixture records were preserved",
            },
            {
                "idea_idx": 0,
                "status": "success",
                "quality_gate_passed": False,
                "claim_support_score": 0.5,
            },
        ],
        "profile": normalized_profile,
        "resumable": True,
    }
    budget = {
        "schema": "xscientist.llm-budget.v1",
        "limits": {"tokens": 0, "cost_usd": 0.0, "wall_time_seconds": 30},
        "used": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        "reserved": {"tokens": 0, "cost_usd": 0.0},
        "provider": "deterministic-fixture",
    }
    insight = {
        "schema": "xscientist.insight-report.v1",
        "epistemic_status": "machine_synthesized_unverified",
        "insights": [
            "The development-set benefit did not transfer to held-out questions."
        ],
        "contradictions": [
            "Development support conflicts with the reversed held-out effect."
        ],
        "claim_promotion_allowed": False,
    }
    receipt = {
        "schema": AUTOPILOT_FIXTURE_SCHEMA,
        "profile": normalized_profile,
        "provider": "deterministic-fixture",
        "executor": "bundled-trusted-python-fixture",
        "network_used": False,
        "provider_used": False,
        "generated_code_executed": False,
        "cost_usd": 0.0,
        "phases": [
            "question",
            "ideation",
            "experiment_failure",
            "experiment_success",
            "evidence",
            "inference",
            "independent_hold_review",
            "insight",
            "research_vcs",
            "dag",
        ],
        "expected_closure": "blocked",
        "resumable": True,
    }
    atomic_write_json(logs / "progress.json", progress)
    atomic_write_json(logs / "llm_budget.json", budget)
    atomic_write_json(logs / "insight_report.json", insight)
    atomic_write_json(logs / "autopilot_fixture_receipt.json", receipt)
    result["autopilot_fixture"] = receipt
    result["runtime"] = {
        "progress": "04_logs/progress.json",
        "budget": "04_logs/llm_budget.json",
        "insight": "04_logs/insight_report.json",
        "experiment": "02_experiments/offline-autopilot-fixture",
        "receipt": "04_logs/autopilot_fixture_receipt.json",
    }
    return result


__all__ = [
    "AUTOPILOT_FIXTURE_SCHEMA",
    "DEMO_SCHEMA",
    "create_autopilot_demo",
    "create_demo",
]
