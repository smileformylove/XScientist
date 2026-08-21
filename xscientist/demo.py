"""Provider-free product demo for the Research VCS and evidence DAG."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text
from ai_scientist.utils.privacy import redact_sensitive_payload

from .research_dag import export_research_dag
from .research_git import ResearchGitError, create_checkpoint
from .research_journey import build_research_guide, start_guided_research
from .research_lifecycle import ResearchLifecycle
from .research_vcs import ResearchRepository

DEMO_SCHEMA = "xscientist.demo.v1"
AUTOPILOT_FIXTURE_SCHEMA = "xscientist.autopilot-fixture.v1"

_DEMO_CODE = """# Bundled trusted fixture; no generated code is executed.
development_effect = 0.18 - 0.31
heldout_effect = 0.29 - 0.27
print({'development_effect': development_effect, 'heldout_effect': heldout_effect})
"""


def _fixture_hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


_DEMO_CODE_HASH = _fixture_hash(_DEMO_CODE)
_DEMO_ENVIRONMENT_HASH = _fixture_hash(
    "xscientist trusted offline fixture environment v1"
)
_DEMO_DEPENDENCY_HASH = _fixture_hash("python-standard-library-only")


def public_demo_payload(
    payload: dict[str, Any], *, workspace: str | Path
) -> dict[str, Any]:
    """Make demo JSON portable without hiding the files from local callers."""

    safe = deepcopy(payload)
    root = Path(workspace).expanduser().resolve()

    def relative_path(value: Any) -> str:
        try:
            return Path(str(value)).expanduser().resolve().relative_to(root).as_posix()
        except (OSError, TypeError, ValueError):
            return "[REDACTED_PATH]"

    safe["repository"] = "."
    dag = safe.get("dag")
    if isinstance(dag, dict):
        for field in ("json", "html"):
            if dag.get(field):
                dag[field] = relative_path(dag[field])
    guide = safe.get("guide")
    if isinstance(guide, dict):
        guide["repository"] = "."
        try:
            refreshed = build_research_guide(
                root,
                language=str(guide.get("language") or "auto"),
                command_repo=".",
            )
            for field in ("primary_action", "next_steps", "warnings", "program_review"):
                if field in refreshed:
                    guide[field] = refreshed[field]
        except (OSError, ResearchGitError, ValueError):
            pass
    safe["privacy"] = {
        "host_paths_disclosed": False,
        "matched_values_disclosed": False,
        "workspace_reference": ".",
    }
    return redact_sensitive_payload(safe)


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
    experiment = root / "02_experiments" / "offline-autopilot-fixture"
    experiment.mkdir(parents=True, exist_ok=True)
    atomic_write_text(experiment / "code.py", _DEMO_CODE)

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
        provenance={
            "code_hash": _DEMO_CODE_HASH,
            "environment_hash": _DEMO_ENVIRONMENT_HASH,
            "dependency_lock_hashes": [_DEMO_DEPENDENCY_HASH],
            "dataset_hashes": [_fixture_hash("malformed-demo-fixture")],
            "seeds": [17],
        },
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
        provenance={
            "code_hash": _DEMO_CODE_HASH,
            "environment_hash": _DEMO_ENVIRONMENT_HASH,
            "dependency_lock_hashes": [_DEMO_DEPENDENCY_HASH],
            "dataset_hashes": [_fixture_hash("bundled-demo-development")],
            "seeds": [42],
        },
        commit=False,
    )["attempt"]
    supporting = lifecycle.evidence(
        {
            "summary": "Retrieval reduced unsupported claims on development data.",
            "effect": -0.13,
            "metric": "unsupported_claim_rate",
            "scope": "development set",
            "measurement_hash": _fixture_hash(
                "development:retrieval=0.18;baseline=0.31"
            ),
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
        provenance={
            "code_hash": _DEMO_CODE_HASH,
            "environment_hash": _DEMO_ENVIRONMENT_HASH,
            "dependency_lock_hashes": [_DEMO_DEPENDENCY_HASH],
            "dataset_hashes": [_fixture_hash("bundled-demo-heldout")],
            "seeds": [43],
        },
        commit=False,
    )["attempt"]
    refuting = lifecycle.evidence(
        {
            "summary": "The development benefit did not transfer to held-out data.",
            "effect": 0.02,
            "metric": "unsupported_claim_rate",
            "scope": "held-out set",
            "measurement_hash": _fixture_hash("heldout:retrieval=0.29;baseline=0.27"),
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
    checkpoint = create_checkpoint(
        root,
        stage="paper",
        subject="complete provider-free contested-evidence demo",
        status="draft",
        reproduce_command=("python 02_experiments/offline-autopilot-fixture/code.py"),
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


def _apply_autopilot_profile_fixture(
    root: Path,
    *,
    profile: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Materially exercise the contract promised by each Autopilot profile."""

    repository = ResearchRepository(root)
    lifecycle = ResearchLifecycle(repository)
    profile_objects: dict[str, Any] = {}
    behavior: list[str] = ["contested_evidence", "failure_preservation"]

    if profile == "discovery":
        from .research_strategy import (
            rank_experiment_candidates,
            save_discriminating_prediction,
            save_hypothesis_portfolio,
        )

        primary_id = str(result["objects"]["hypothesis"])
        rival = repository.record(
            "hypothesis",
            {
                "statement": (
                    "Retrieval helps only when the retrieved source quality exceeds "
                    "the model's unaided knowledge quality."
                ),
                "falsifier": (
                    "Source-quality ablation leaves the retrieval effect unchanged."
                ),
                "role": "rival_mechanism",
            },
            relations=[{"type": "contradicts", "target": primary_id}],
            actor={"actor_id": "discovery-fixture", "authority": "research_agent"},
        )
        null = repository.record(
            "hypothesis",
            {
                "statement": (
                    "Retrieval does not reliably change unsupported-claim rates "
                    "outside sampling variation."
                ),
                "falsifier": "A preregistered held-out effect is stable and non-zero.",
                "role": "null",
            },
            relations=[{"type": "contradicts", "target": primary_id}],
            actor={"actor_id": "discovery-fixture", "authority": "research_agent"},
        )
        portfolio = save_hypothesis_portfolio(
            root,
            question="Which explanation best predicts when retrieval transfers?",
            primary_id=primary_id,
            alternative_ids=[rival.object_id],
            null_id=null.object_id,
            prior_weights={primary_id: 1, rival.object_id: 1, null.object_id: 1},
            commit=False,
        )["object"]
        hypotheses = [primary_id, rival.object_id, null.object_id]
        conditions = [
            (
                "source-quality-ablation",
                "Retrieved passages are replaced with low-quality but topic-matched passages.",
                {
                    primary_id: "retrieval still lowers unsupported claims",
                    rival.object_id: "the retrieval benefit disappears",
                    null.object_id: "rates remain statistically unchanged",
                },
            ),
            (
                "held-out-domain",
                "The locked comparison is repeated on a disjoint held-out domain.",
                {
                    primary_id: "retrieval lowers unsupported claims",
                    rival.object_id: "benefit depends on held-out source quality",
                    null.object_id: "rates remain statistically unchanged",
                },
            ),
        ]
        predictions_by_condition: dict[str, dict[str, str]] = {}
        for candidate_id, condition, outcomes in conditions:
            predictions_by_condition[candidate_id] = {}
            for hypothesis_id in hypotheses:
                prediction = save_discriminating_prediction(
                    root,
                    portfolio_id=portfolio.object_id,
                    hypothesis_id=hypothesis_id,
                    when=condition,
                    expected_outcome=outcomes[hypothesis_id],
                    distinguishes_from=[
                        other for other in hypotheses if other != hypothesis_id
                    ],
                    falsifier=(
                        "The observed outcome matches a competing locked prediction."
                    ),
                    commit=False,
                )["object"]
                predictions_by_condition[candidate_id][
                    hypothesis_id
                ] = prediction.object_id
        candidates = [
            {
                "candidate_id": candidate_id,
                "summary": f"Run the {candidate_id} discriminating test.",
                "condition": condition,
                "predictions": outcomes,
                "prediction_ids": predictions_by_condition[candidate_id],
                "interventions": [candidate_id],
                "novelty": 3,
                "impact": 3,
                "transfer_value": 4 if candidate_id == "held-out-domain" else 3,
                "cost": 1 if candidate_id == "source-quality-ablation" else 2,
                "risk": 1,
                "redundancy": 0,
            }
            for candidate_id, condition, outcomes in conditions
        ]
        priority = rank_experiment_candidates(
            root,
            portfolio_id=portfolio.object_id,
            candidates=candidates,
            commit=False,
        )
        repository.commit(
            stage="plan",
            subject="exercise discovery portfolio and information-value ranking",
            status="locked",
        )
        profile_objects = {
            "rival_hypothesis": rival.object_id,
            "null_hypothesis": null.object_id,
            "portfolio": portfolio.object_id,
            "predictions": [
                prediction_id
                for rows in predictions_by_condition.values()
                for prediction_id in rows.values()
            ],
            "priority": priority["object"].object_id,
            "candidate_designs": [
                item.object_id for item in priority.get("related") or []
            ],
        }
        behavior.extend(
            [
                "competitive_hypothesis_portfolio",
                "discriminating_predictions",
                "information_value_ranking",
            ]
        )
    elif profile == "publication":
        evaluates = [
            str(result["objects"]["contested_claim"]),
            str(result["objects"]["bounded_inference"]),
            str(result["objects"]["supporting_evidence"]),
            str(result["objects"]["refuting_evidence"]),
        ]
        reviews = []
        gates = []
        contexts = []
        for reviewer, focus in (
            ("human:publication-methods-reviewer", "methods and transfer validity"),
            ("human:publication-claims-reviewer", "claim scope and evidence binding"),
        ):
            evaluation = lifecycle.evaluation(
                {
                    "status": "rejected",
                    "claim_promotion_allowed": False,
                    "required_failures": ["independent_reproduction_missing"],
                    "summary": f"Publication board hold: {focus} requires repair.",
                    "review_focus": focus,
                },
                evaluates=evaluates,
                verifier_id=reviewer,
                commit=False,
            )
            reviews.append(evaluation["review"].object_id)
            gates.append(evaluation["gate"].object_id)
            contexts.append(evaluation["context"].object_id)
        repository.commit(
            stage="review",
            subject="exercise publication multi-role review board",
            status="rejected",
        )
        profile_objects = {
            "publication_reviews": reviews,
            "publication_gates": gates,
            "decision_contexts": contexts,
        }
        behavior.extend(["multi_role_review", "strict_publication_hold_gates"])
    else:
        behavior.append("bounded_first_complete_run")

    result["objects"]["autopilot_profile"] = profile_objects
    exported = export_research_dag(root, root / "research-dag")
    graph = exported["graph"]
    result["dag"].update(
        {
            "json": exported["json"],
            "html": exported["html"],
            "nodes": len(graph.get("nodes") or []),
            "relations": len(graph.get("edges") or []),
            "integrity_ok": bool((graph.get("integrity") or {}).get("is_dag")),
            "closure": (graph.get("scientific_closure") or {}).get("status"),
        }
    )
    result["guide"] = build_research_guide(root)
    return {"behavior": behavior, "objects": profile_objects}


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
    profile_fixture = _apply_autopilot_profile_fixture(
        root,
        profile=normalized_profile,
        result=result,
    )
    logs = root / "04_logs"
    experiment = root / "02_experiments" / "offline-autopilot-fixture"
    atomic_write_text(experiment / "code.py", _DEMO_CODE)
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
        "profile_behavior": profile_fixture["behavior"],
        "profile_objects": profile_fixture["objects"],
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
        ]
        + (
            ["competitive_portfolio", "information_value_ranking"]
            if normalized_profile == "discovery"
            else (
                ["multi_role_review_board"]
                if normalized_profile == "publication"
                else []
            )
        ),
        "expected_closure": "blocked",
        "resumable": True,
    }
    atomic_write_json(logs / "progress.json", progress)
    atomic_write_json(logs / "llm_budget.json", budget)
    atomic_write_json(logs / "insight_report.json", insight)
    atomic_write_json(logs / "autopilot_fixture_receipt.json", receipt)
    runtime_checkpoint = create_checkpoint(
        root,
        stage="experiment",
        subject="record deterministic offline runtime surfaces",
        status="completed",
        reproduce_command=("python 02_experiments/offline-autopilot-fixture/code.py"),
        include=[
            "02_experiments/offline-autopilot-fixture/code.py",
            "02_experiments/offline-autopilot-fixture/metrics.json",
            "04_logs/progress.json",
            "04_logs/llm_budget.json",
            "04_logs/insight_report.json",
            "04_logs/autopilot_fixture_receipt.json",
        ],
    )
    scientific_checkpoint_id = result["checkpoint_id"]
    result["scientific_checkpoint_id"] = scientific_checkpoint_id
    result["checkpoint_id"] = runtime_checkpoint.checkpoint_id
    result["runtime_checkpoint"] = {
        "checkpoint_id": runtime_checkpoint.checkpoint_id,
        "commit": runtime_checkpoint.commit,
        "content_hash": runtime_checkpoint.content_hash,
        "staged_paths": list(runtime_checkpoint.staged_paths),
    }
    result["autopilot_fixture"] = receipt
    result["runtime"] = {
        "progress": "04_logs/progress.json",
        "budget": "04_logs/llm_budget.json",
        "insight": "04_logs/insight_report.json",
        "experiment": "02_experiments/offline-autopilot-fixture",
        "receipt": "04_logs/autopilot_fixture_receipt.json",
    }
    exported = export_research_dag(root, root / "research-dag")
    graph = exported["graph"]
    result["dag"].update(
        {
            "json": exported["json"],
            "html": exported["html"],
            "nodes": len(graph.get("nodes") or []),
            "relations": len(graph.get("edges") or []),
            "integrity_ok": bool((graph.get("integrity") or {}).get("is_dag")),
            "closure": (graph.get("scientific_closure") or {}).get("status"),
        }
    )
    result["guide"] = build_research_guide(root, language=language)
    return result


__all__ = [
    "AUTOPILOT_FIXTURE_SCHEMA",
    "DEMO_SCHEMA",
    "create_autopilot_demo",
    "create_demo",
    "public_demo_payload",
]
